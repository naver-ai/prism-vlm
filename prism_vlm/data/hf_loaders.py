"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

HuggingFace dataset adapters for each benchmark.

Each benchmark has its own quirks — different config names, field layouts,
image vs. image-list, answer-as-index vs answer-as-letter — so we declare a
loader spec per benchmark rather than trying to generalise. Adding a new
benchmark = add one entry to ``_BUILDERS``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets, get_dataset_config_names, load_dataset

from ..config.loader import BenchmarkSpec
from ..utils.hashing import item_id_for
from ..utils.io import load_image_bytes
from .grounding_adapter import BoundedMC4Adapter
from .registry import SourceNotProvidedError
from .schemas import Item

# ---------------------------------------------------------------------------
# Dataset spec: path, config, split, field mapping, and any pre-processor.
# ---------------------------------------------------------------------------


@dataclass
class _DSSpec:
    path: str
    split: str
    config: str | None = None
    configs: list[str] | None = None     # concat multiple configs (MMMU, BLINK)
    q: str = "question"
    img: str = "image"
    gold: str = "answer"
    multi_image: bool = False
    preprocessor: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    filter_fn: Callable[[dict[str, Any]], bool] | None = None
    grounding: bool = False


_MC_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _scienceqa_preprocess(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild ScienceQA row as MC: options inline, gold as letter."""
    choices = row["choices"]
    answer_idx = int(row["answer"])
    q = row["question"]
    options = "\n".join(f"{_MC_LETTERS[i]}. {c}" for i, c in enumerate(choices))
    row["question_mc"] = f"{q}\n\nOptions:\n{options}\n\nAnswer with the letter."
    row["gold_letter"] = _MC_LETTERS[answer_idx]
    return row


def _mmmu_preprocess(row: dict[str, Any]) -> dict[str, Any]:
    """MMMU stores options as a stringified Python list; answer is the letter."""
    opts_raw = row.get("options", "[]")
    if isinstance(opts_raw, str):
        import ast

        try:
            opts = ast.literal_eval(opts_raw)
        except Exception:
            opts = []
    else:
        opts = list(opts_raw)
    q = row["question"]
    if opts:
        options = "\n".join(f"{_MC_LETTERS[i]}. {o}" for i, o in enumerate(opts))
        row["question_mc"] = f"{q}\n\nOptions:\n{options}\n\nAnswer with the letter."
    else:
        row["question_mc"] = q
    return row


def _blink_preprocess(row: dict[str, Any]) -> dict[str, Any]:
    """BLINK has image_1..N separately + a ``prompt`` field with options inline."""
    imgs = [row[k] for k in ("image_1", "image_2", "image_3", "image_4") if k in row and row[k] is not None]
    row["images_list"] = imgs
    # BLINK's ``prompt`` field already contains ``question\nSelect from ...\n(A) ...``.
    row["question_with_options"] = row.get("prompt", row.get("question", ""))
    ans = str(row.get("answer", "")).strip()
    # "(A)" → "A"
    if ans.startswith("(") and ans.endswith(")"):
        ans = ans[1:-1]
    row["gold_letter"] = ans
    return row


def _ai2d_preprocess(row: dict[str, Any]) -> dict[str, Any]:
    """AI2D: answer is an index into ``options``; rebuild as A/B/C/D MC."""
    q = row["question"]
    opts = list(row.get("options", []))
    answer_idx = int(row.get("answer", 0))
    if opts:
        options_str = "\n".join(f"{_MC_LETTERS[i]}. {o}" for i, o in enumerate(opts))
        row["question_mc"] = f"{q}\n\nOptions:\n{options_str}\n\nAnswer with the letter."
        row["gold_letter"] = _MC_LETTERS[answer_idx] if 0 <= answer_idx < len(opts) else ""
    else:
        row["question_mc"] = q
        row["gold_letter"] = str(answer_idx)
    return row



def _refcoco_preprocess(row: dict[str, Any]) -> dict[str, Any]:
    """RefCOCOg/+: pick first caption as question, keep bbox for grounding."""
    ans = row.get("answer", [])
    if isinstance(ans, list) and ans:
        row["first_caption"] = ans[0]
    else:
        row["first_caption"] = str(ans)
    return row


def _nlvr2_preprocess(row: dict[str, Any]) -> dict[str, Any]:
    """NLVR2: left_image + right_image → images list; answer bool → True/False."""
    imgs = [row[k] for k in ("left_image", "right_image") if row.get(k) is not None]
    row["images_pair"] = imgs
    ans = row.get("answer")
    if isinstance(ans, bool):
        row["gold_tf"] = "True" if ans else "False"
    else:
        row["gold_tf"] = str(ans)
    # Prefix the claim so the model knows to judge true/false.
    row["claim_question"] = (
        f"Claim: {row.get('question', '')}\n\n"
        "Is this claim TRUE or FALSE given the two images? Answer with True or False."
    )
    return row


# GQA requires joining two configs: images (id→image) and instructions (id→q+a).
# We provide a dedicated loader class below.
_GQA_CONFIG_IMAGES = "testdev_balanced_images"
_GQA_CONFIG_INSTR = "testdev_balanced_instructions"


_BUILDERS: dict[str, _DSSpec] = {
    # --- General VQA --------------------------------------------------------
    "mmstar": _DSSpec(
        path="Lin-Chen/MMStar", split="val",
        q="question", img="image", gold="answer",
    ),
    "gqa": _DSSpec(
        path="lmms-lab/gqa", split="testdev",
        config=_GQA_CONFIG_IMAGES,  # sentinel; custom handling below
        q="question", img="image", gold="answer",
    ),
    "okvqa": _DSSpec(
        path="Multimodal-Fatima/OK-VQA_test", split="test",
        q="question", img="image", gold="answers",
    ),

    # --- Math & Reasoning ---------------------------------------------------
    "mmmu": _DSSpec(
        path="MMMU/MMMU", split="validation",
        configs=None,  # populated below to all subjects
        q="question_mc", img="image_1", gold="answer",
        preprocessor=_mmmu_preprocess,
    ),
    "mathvista": _DSSpec(
        path="AI4Math/MathVista", split="testmini",
        q="query", img="decoded_image", gold="answer",
    ),
    "scienceqa": _DSSpec(
        path="derek-thomas/ScienceQA", split="test",
        q="question_mc", img="image", gold="gold_letter",
        preprocessor=_scienceqa_preprocess,
        filter_fn=lambda r: r.get("image") is not None,
    ),

    # --- Chart & Document ---------------------------------------------------
    "ai2d": _DSSpec(
        path="lmms-lab/ai2d", split="test",
        q="question_mc", img="image", gold="gold_letter",
        preprocessor=_ai2d_preprocess,
    ),
    "chartqa": _DSSpec(
        path="lmms-lab/ChartQA", split="test",
        q="question", img="image", gold="answer",
    ),
    "docvqa": _DSSpec(
        path="lmms-lab/DocVQA", config="DocVQA", split="validation",
        q="question", img="image", gold="answers",
    ),

    # --- Grounding ----------------------------------------------------------
    "refcocog": _DSSpec(
        path="lmms-lab/RefCOCOg", split="val",
        q="first_caption", img="image", gold="bbox",
        preprocessor=_refcoco_preprocess,
        grounding=True,
    ),
    "refcocoplus": _DSSpec(
        path="lmms-lab/RefCOCOplus", split="testA",
        q="first_caption", img="image", gold="bbox",
        preprocessor=_refcoco_preprocess,
        grounding=True,
    ),
    # TallyQA not available on HF as of 2026-04 — substitute an open counting
    # benchmark. Using ``BLINK`` Counting subtask keeps us within the grounding
    # domain theme and loads cleanly. BLINK stores gold as "(A)"/"(B)"/... so
    # we strip the parentheses for single-letter parsing.
    "tallyqa": _DSSpec(
        path="BLINK-Benchmark/BLINK", config="Counting", split="val",
        q="question_with_options", img="image_1", gold="gold_letter",
        preprocessor=_blink_preprocess,
    ),

    # --- Multi-Image --------------------------------------------------------
    # Mantis-Instruct images are externally-stored (dict with path only); use
    # Mantis-Eval which has real bytes.
    "mantis": _DSSpec(
        path="TIGER-Lab/Mantis-Eval", config="mantis_eval", split="test",
        q="question", img="images", gold="answer",
        multi_image=True,
    ),
    # MMT-Bench "default" config has a CSV schema mismatch; NLVR2 is a robust
    # multi-image substitute (binary reasoning over paired images). Its authors
    # do not distribute the images publicly, so it is read from the user's own
    # copy (``_load_nlvr2``), never from a Hub mirror.
    "mmt_bench": _DSSpec(
        path="lil-lab/nlvr", split="balanced_test1",
        q="claim_question", img="images_pair", gold="gold_tf",
        multi_image=True,
        preprocessor=_nlvr2_preprocess,
    ),
    "blink": _DSSpec(
        path="BLINK-Benchmark/BLINK", split="val",
        configs=None,  # populated below to all subtasks
        q="question_with_options", img="images_list", gold="gold_letter",
        multi_image=True,
        preprocessor=_blink_preprocess,
    ),
}


@dataclass
class _GenericLoader:
    spec: BenchmarkSpec
    ds: Dataset
    dspec: _DSSpec
    grounding: BoundedMC4Adapter | None = None

    def pool_size(self) -> int:
        return int(len(self.ds))

    def item_at(self, domain: str, orig_index: int) -> Item:
        row = dict(self.ds[orig_index])
        if self.dspec.preprocessor is not None:
            row = self.dspec.preprocessor(row)

        question = str(row[self.dspec.q])
        gold_raw = row[self.dspec.gold]
        gold: str | tuple[str, ...]
        if isinstance(gold_raw, list):
            gold = tuple(str(x) for x in gold_raw if x is not None)
        else:
            gold = str(gold_raw)

        if self.dspec.multi_image:
            raw_imgs = row[self.dspec.img]
            if raw_imgs is None:
                raw_imgs = []
            images = tuple(load_image_bytes(img) for img in raw_imgs if img is not None)
        else:
            img = row[self.dspec.img]
            images = (load_image_bytes(img),) if img is not None else ()

        item_id = item_id_for(self.spec.id, orig_index, question)
        item = Item(
            item_id=item_id,
            domain=domain,
            benchmark=self.spec.id,
            question=question,
            images=images,
            gold=gold,
            meta={"orig_index": orig_index},
        )
        if self.grounding is not None:
            item = self.grounding.wrap(item, row)
        return item


@dataclass
class _GQALoader:
    """Custom loader that joins lmms-lab/gqa images + instructions configs."""

    spec: BenchmarkSpec
    images_ds: Dataset
    instructions_ds: Dataset
    image_by_id: dict[str, Any] = field(default_factory=dict)

    def pool_size(self) -> int:
        return int(len(self.instructions_ds))

    def item_at(self, domain: str, orig_index: int) -> Item:
        row = self.instructions_ds[orig_index]
        image_id = str(row.get("imageId", row.get("image_id", "")))
        img = self.image_by_id.get(image_id)
        images = (load_image_bytes(img),) if img is not None else ()
        question = str(row["question"])
        gold = str(row["answer"])
        item_id = item_id_for(self.spec.id, orig_index, question)
        return Item(
            item_id=item_id, domain=domain, benchmark=self.spec.id,
            question=question, images=images, gold=gold,
            meta={"orig_index": orig_index, "image_id": image_id},
        )


def _resolve_mmmu_subjects() -> list[str]:
    return get_dataset_config_names("MMMU/MMMU")


def _resolve_blink_subjects() -> list[str]:
    return get_dataset_config_names("BLINK-Benchmark/BLINK")


def _load_concat(path: str, configs: list[str], split: str) -> Dataset:
    parts: list[Dataset] = []
    for cfg in configs:
        try:
            ds = load_dataset(path, name=cfg, split=split)
            parts.append(ds)
        except Exception:
            continue
    if not parts:
        raise RuntimeError(f"No configs loaded for {path}")
    try:
        return concatenate_datasets(parts)
    except Exception:
        # Schema mismatch across subjects — take the largest part.
        parts.sort(key=len, reverse=True)
        return parts[0]


def _load_hf(dspec: _DSSpec) -> Dataset:
    if dspec.configs is not None:
        return _load_concat(dspec.path, dspec.configs, dspec.split)
    kwargs = {"path": dspec.path, "split": dspec.split}
    if dspec.config:
        kwargs["name"] = dspec.config
    ds = load_dataset(**kwargs)
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected Dataset for {dspec.path}, got {type(ds)}")
    if dspec.filter_fn is not None:
        ds = ds.filter(dspec.filter_fn)
    return ds


NLVR2_DIR_ENV = "PRISM_NLVR2_DIR"
_NLVR2_FORM = "https://github.com/lil-lab/nlvr/tree/master/nlvr2#direct-image-download"


def _load_nlvr2() -> Dataset:
    """NLVR2 balanced test-P from the copy the user obtained from its authors.

    ``$PRISM_NLVR2_DIR`` holds ``balanced_test1.json`` from
    github.com/lil-lab/nlvr (nlvr2/data/balanced/) and the test1 images, which
    the authors hand out on request. Rows keep the file's order, which is the
    order the published pool was sampled from, so item ids are unchanged.
    """
    root = os.environ.get(NLVR2_DIR_ENV)
    if not root:
        raise SourceNotProvidedError(
            f"NLVR2 images are distributed only on request. Request them at "
            f"{_NLVR2_FORM}, then set {NLVR2_DIR_ENV} (see README)."
        )
    root_path = Path(root).expanduser()
    ann = next(root_path.rglob("balanced_test1.json"), None)
    if ann is None:
        raise FileNotFoundError(f"balanced_test1.json not found under {root_path}")
    # Images are named after the pair: test1-0-0-img0.png for test1-0-0-0.
    images = {p.stem: str(p) for p in root_path.rglob("*-img[01].*")}
    rows = []
    for line in ann.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        pair = r["identifier"].rsplit("-", 1)[0]
        try:
            left, right = images[f"{pair}-img0"], images[f"{pair}-img1"]
        except KeyError as e:
            raise FileNotFoundError(f"NLVR2 image {e.args[0]} not found under {root_path}") from None
        rows.append({"question": r["sentence"], "answer": r["label"],
                     "left_image": left, "right_image": right})
    return Dataset.from_list(rows)


def build_loader(spec: BenchmarkSpec):  # type: ignore[no-untyped-def]
    if spec.id not in _BUILDERS:
        raise KeyError(f"No loader registered for benchmark '{spec.id}'")
    dspec = _BUILDERS[spec.id]

    # MMMU + BLINK: auto-populate all subjects / subtasks for concatenation.
    if spec.id == "mmmu" and dspec.configs is None:
        dspec.configs = _resolve_mmmu_subjects()
    if spec.id == "blink" and dspec.configs is None:
        dspec.configs = _resolve_blink_subjects()

    # GQA uses a two-config join; custom loader.
    if spec.id == "gqa":
        images_ds = load_dataset("lmms-lab/gqa", name=_GQA_CONFIG_IMAGES, split="testdev")
        instr_ds = load_dataset("lmms-lab/gqa", name=_GQA_CONFIG_INSTR, split="testdev")
        image_by_id = {str(r["id"]): r["image"] for r in images_ds}
        return _GQALoader(spec=spec, images_ds=images_ds, instructions_ds=instr_ds,
                          image_by_id=image_by_id)

    ds = _load_nlvr2() if spec.id == "mmt_bench" else _load_hf(dspec)
    grounding = BoundedMC4Adapter() if dspec.grounding else None
    return _GenericLoader(spec=spec, ds=ds, dspec=dspec, grounding=grounding)
