"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Offline smoke tests. No network, no API key: the build loop runs against a
stub judge, and everything else pins the determinism the benchmark rests on --
sampling seeds, content-keyed item ids, and the h sub-type rotation.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import pkgutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import prism_vlm
from prism_vlm.config.loader import (
    BenchmarkSpec,
    DomainsConfig,
    ModelsConfig,
    ModelSpec,
    RunConfig,
    load_domains,
    load_models,
    load_run_config,
)
from prism_vlm.data.sampling import deterministic_sample_indices
from prism_vlm.data.schemas import Item, Variant
from prism_vlm.models.base import GenConfig, build_multimodal_messages
from prism_vlm.runner import build_pipeline
from prism_vlm.scoring import AXES as AXES_ORDER
from prism_vlm.utils.hashing import deterministic_rng, item_id_for, variant_id_for
from prism_vlm.utils.io import read_json, write_json_atomic
from prism_vlm.utils.seeds import REACH_K_SEEDS
from prism_vlm.variants import SUB_TYPES

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# --------------------------------------------------------------------------
# package integrity
# --------------------------------------------------------------------------

def test_every_module_imports() -> None:
    """Catches an over-trimmed ``__init__`` re-exporting a module we dropped."""
    failures = []
    for mod in pkgutil.walk_packages(prism_vlm.__path__, prefix="prism_vlm."):
        try:
            importlib.import_module(mod.name)
        except Exception as e:  # noqa: BLE001
            failures.append(f"{mod.name}: {e!r}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


# --------------------------------------------------------------------------
# determinism of the item pool
# --------------------------------------------------------------------------

def test_reach_k_seeds_are_the_published_five() -> None:
    assert REACH_K_SEEDS == (0, 1, 42, 1234, 2026)


def test_item_id_is_content_keyed() -> None:
    a = item_id_for("okvqa", 17, "What season is it outside?")
    assert a == item_id_for("okvqa", 17, "What season is it outside?")
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)
    # Every component participates in the key.
    assert a != item_id_for("gqa", 17, "What season is it outside?")
    assert a != item_id_for("okvqa", 18, "What season is it outside?")
    assert a != item_id_for("okvqa", 17, "What season is it?")


def test_item_id_is_stable_across_processes() -> None:
    """The id is a cache key: a rebuild must land on the same entry."""
    code = (
        "from prism_vlm.utils.hashing import item_id_for;"
        "print(item_id_for('okvqa', 17, 'What season is it outside?'))"
    )
    got = {_run_child(code, hashseed) for hashseed in ("0", "1", "random")}
    assert len(got) == 1, f"item_id varied across hash seeds: {got}"


def test_variant_id_is_axis_and_slot_scoped() -> None:
    iid = item_id_for("okvqa", 17, "q?")
    assert variant_id_for(iid, "h", 0) != variant_id_for(iid, "mq", 0)
    assert variant_id_for(iid, "mq", 0) != variant_id_for(iid, "mq", 1)


def test_deterministic_rng_is_reproducible() -> None:
    draw = lambda: [deterministic_rng("abc", "h").random() for _ in range(3)]  # noqa: E731
    assert draw() == draw()
    assert draw() != [deterministic_rng("abd", "h").random() for _ in range(3)]


def test_deterministic_sample_indices() -> None:
    a = deterministic_sample_indices(500, 100, seed=42)
    assert a == deterministic_sample_indices(500, 100, seed=42)
    assert a == sorted(a) and len(set(a)) == 100
    assert a != deterministic_sample_indices(500, 100, seed=2026)
    with pytest.raises(ValueError):
        deterministic_sample_indices(50, 100, seed=0)


# --------------------------------------------------------------------------
# h sub-type rotation
# --------------------------------------------------------------------------

def test_h_sub_types_are_the_published_three() -> None:
    assert SUB_TYPES == ("absent_entity", "misleading_premise", "uncertain_image")


def test_h_sub_type_is_stable_across_processes() -> None:
    """Regression: a builtin ``hash`` here would re-roll on every rebuild."""
    code = (
        "from prism_vlm.runner.build_pipeline import _pick_h_sub_type;"
        "print(','.join(_pick_h_sub_type(f'item{i:04d}') for i in range(12)))"
    )
    got = {_run_child(code, hashseed) for hashseed in ("0", "1", "12345", "random")}
    assert len(got) == 1, f"h sub-type varied across hash seeds: {got}"


def test_take_sample_is_stable_across_processes() -> None:
    """build and run must land on the same debug subset.

    The sampler returns its union in set order, which CPython varies per
    process, so ``take_sample`` sorts before slicing.
    """
    code = (
        "from prism_vlm.data.sampling import take_sample;"
        "from prism_vlm.data.schemas import Item;"
        "items=[Item(item_id=f'id{i:03d}',domain='d',benchmark='b',"
        "question='q',images=(),gold='g') for i in range(20)];"
        "import random; random.shuffle(items);"
        "print(','.join(i.item_id for i in take_sample(items, 5)))"
    )
    got = {_run_child(code, hashseed) for hashseed in ("0", "1", "12345", "random")}
    assert len(got) == 1, f"debug subset varied across hash seeds: {got}"
    assert got.pop() == "id000,id001,id002,id003,id004"


def test_filtered_sampling_keeps_metadata_for_items_outside_the_filter() -> None:
    """Regression: a --benchmarks run used to truncate items_meta.jsonl.

    The next unfiltered run then took the fast path, intersected the full id
    list against the shrunken metadata, and silently returned fewer items.
    """
    import tempfile

    from prism_vlm.data.sampling import (
        _items_metadata_to_write,
        _read_items_metadata,
        _write_items_metadata,
    )

    def _item(i: str, bench: str) -> Item:
        return Item(item_id=i, domain="d", benchmark=bench,
                    question=f"q{i}", images=(), gold="g")

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        _write_items_metadata(base, {"a": _item("a", "one"), "b": _item("b", "two")})

        narrowed = {"a": _item("a", "one")}
        merged = _items_metadata_to_write(base, narrowed, is_partial=True)
        assert set(merged) == {"a", "b"}

        # An unfiltered call is the authority and replaces the file wholesale.
        assert _items_metadata_to_write(base, narrowed, is_partial=False) is narrowed

        _write_items_metadata(base, merged)
        assert set(_read_items_metadata(base)) == {"a", "b"}


def test_take_sample_passes_through_when_not_limiting() -> None:
    from prism_vlm.data.sampling import take_sample

    items = [Item(item_id=f"id{i}", domain="d", benchmark="b",
                  question="q", images=(), gold="g") for i in range(3)]
    assert take_sample(items, None) is items
    assert take_sample(items, 9) is items
    assert len(take_sample(items, 2)) == 2


def test_h_sub_type_covers_each_type_about_evenly() -> None:
    """The paper claims each sub-type covers ~N/3 of the pool."""
    n = 3_000
    counts = {s: 0 for s in SUB_TYPES}
    for i in range(n):
        counts[build_pipeline._pick_h_sub_type(item_id_for("okvqa", i, f"q{i}?"))] += 1
    assert set(counts) == set(SUB_TYPES)
    for sub_type, c in counts.items():
        assert abs(c - n / 3) < 0.1 * n, f"{sub_type} got {c} of {n}"


# --------------------------------------------------------------------------
# configs
# --------------------------------------------------------------------------

def test_domains_config_is_five_by_three() -> None:
    cfg = load_domains(CONFIGS / "domains.yaml")
    assert len(cfg.domains) == 5
    assert all(len(v) == 3 for v in cfg.domains.values())
    ids = [b.id for b in cfg.all_benchmarks]
    assert len(ids) == 15 and len(set(ids)) == 15
    assert cfg.sampling.items_per_benchmark == 100


def test_smoke_domains_mirrors_domains() -> None:
    """Item ids are keyed on the benchmark id, so the two pools must agree."""
    full = load_domains(CONFIGS / "domains.yaml")
    smoke = load_domains(CONFIGS / "smoke_domains.yaml")
    assert full.domains == smoke.domains
    assert smoke.sampling.items_per_benchmark == 1


def test_every_benchmark_id_has_a_loader() -> None:
    """The id is the only thing the config carries, so it must resolve."""
    from prism_vlm.data.hf_loaders import _BUILDERS

    ids = [b.id for b in load_domains(CONFIGS / "domains.yaml").all_benchmarks]
    assert [i for i in ids if i not in _BUILDERS] == []


def test_substituted_benchmarks_point_at_their_real_sources() -> None:
    """The two substituted slots keep their legacy ids (cache keys) but must
    resolve to the sources the paper discloses."""
    from prism_vlm.data.hf_loaders import _BUILDERS

    assert _BUILDERS["tallyqa"].path == "BLINK-Benchmark/BLINK"
    assert _BUILDERS["tallyqa"].config == "Counting"
    assert _BUILDERS["mmt_bench"].path == "lil-lab/nlvr"


def test_nlvr2_reads_the_users_copy_and_keeps_item_ids(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """NLVR2 comes from the authors' files, never a Hub mirror, and yields the
    same item ids the published pool was sampled under."""
    from prism_vlm.data.hf_loaders import build_loader

    rows = [{"identifier": "test1-0-0-0", "sentence": "There is an empty glass.", "label": "False"},
            {"identifier": "test1-0-2-0", "sentence": "Two dogs.", "label": "True"}]
    (tmp_path / "balanced_test1.json").write_text("\n".join(json.dumps(r) for r in rows))
    img_dir = tmp_path / "test1"
    img_dir.mkdir()
    for pair in ("test1-0-0", "test1-0-2"):
        for side in (0, 1):
            (img_dir / f"{pair}-img{side}.png").write_bytes(f"{pair}{side}".encode())
    monkeypatch.setenv("PRISM_NLVR2_DIR", str(tmp_path))

    loader = build_loader(BenchmarkSpec(id="mmt_bench"))
    item = loader.item_at("multi_image", 1)
    assert loader.pool_size() == 2
    assert item.images == (b"test1-0-20", b"test1-0-21")
    assert item.gold == "True"
    assert item.item_id == item_id_for("mmt_bench", 1, item.question)
    assert item.question.startswith("Claim: Two dogs.")


def test_sampling_skips_nlvr2_without_the_users_copy(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from prism_vlm.data.sampling import load_sampled_items

    monkeypatch.delenv("PRISM_NLVR2_DIR", raising=False)
    cfg = DomainsConfig(domains={"multi_image": [BenchmarkSpec(id="mmt_bench")]}, sampling={})
    items, _ = load_sampled_items(cfg, tmp_path, seeds=[0])
    assert items == []


def test_dataset_terms_cover_every_benchmark_and_match_notice() -> None:
    from prism_vlm.data.terms import SOURCES, terms_text

    ids = [b.id for b in load_domains(CONFIGS / "domains.yaml").all_benchmarks]
    assert sorted(SOURCES) == sorted(ids)
    notice = (CONFIGS.parent / "NOTICE").read_text()
    for bid, (name, url, _) in SOURCES.items():
        assert f"  {bid} " in notice and name in notice and url in notice
    assert "not warranted" in terms_text(ids)


def test_dataset_terms_must_be_accepted_once(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    from prism_vlm.cli import _confirm_dataset_terms

    cfg = SimpleNamespace(cache_root=tmp_path)
    domains_cfg = load_domains(CONFIGS / "smoke_domains.yaml")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    assert not _confirm_dataset_terms(cfg, domains_cfg, accepted=False)
    assert _confirm_dataset_terms(cfg, domains_cfg, accepted=True)
    assert _confirm_dataset_terms(cfg, domains_cfg, accepted=False)  # recorded


@pytest.mark.parametrize("name", ["main_run", "smoke_run"])
def test_run_config_loads(name: str) -> None:
    cfg = load_run_config(CONFIGS / f"{name}.yaml")
    assert cfg.judge.model == "gpt-5" and cfg.judge.effort == "low"
    assert cfg.R >= 1
    assert cfg.models_path is not None and cfg.models_path.exists()
    # The analysis output dir belongs to a later stage.
    assert cfg.output_dir is None


def test_model_registry() -> None:
    models = load_models(CONFIGS / "models.yaml").models
    assert {m.provider for m in models} == {"vllm", "openrouter", "openai"}
    assert len({m.id for m in models}) == len(models)


@pytest.mark.parametrize("bad", ["../../etc", "..", "/abs", "a/b", "a\\b", ".hidden", ""])
def test_names_that_become_directories_reject_traversal(bad: str) -> None:
    """run_id, model id, domain and benchmark id are joined into cache paths."""
    with pytest.raises(ValidationError):
        BenchmarkSpec(id=bad)
    with pytest.raises(ValidationError):
        ModelSpec(id=bad, provider="openai", model="m")
    with pytest.raises(ValidationError):
        RunConfig(run_id=bad, domains_path=Path("d.yaml"), variant_bundle=Path("v"))
    with pytest.raises(ValidationError):
        DomainsConfig(domains={bad: [BenchmarkSpec(id="mmstar")]}, sampling={})


def test_shipped_configs_use_plain_names() -> None:
    cfg = load_domains(CONFIGS / "domains.yaml")
    names = ([b.id for b in cfg.all_benchmarks] + list(cfg.domains)
             + [m.id for m in load_models(CONFIGS / "models.yaml").models]
             + [load_run_config(CONFIGS / f"{n}.yaml").run_id for n in ("main_run", "smoke_run")])
    assert all(n == Path(n).name and not n.startswith(".") for n in names)


def test_registry_ships_no_endpoints() -> None:
    """Endpoints are deployment-specific and must not be committed."""
    raw = yaml.safe_load((CONFIGS / "models.yaml").read_text(encoding="utf-8"))
    assert all("endpoint" not in m for m in raw["models"])


def test_endpoint_resolution_precedence(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "models.yaml").write_text(
        "models:\n"
        "  - {id: m, provider: vllm, model: M}\n",
        encoding="utf-8",
    )

    def endpoint() -> str | None:
        return load_models(cfg_dir / "models.yaml").by_id("m").endpoint

    monkeypatch.delenv("PRISM_VLLM_ENDPOINT", raising=False)
    assert endpoint() is None

    monkeypatch.setenv("PRISM_VLLM_ENDPOINT", "http://env:8000/v1")
    assert endpoint() == "http://env:8000/v1"

    (cfg_dir / "endpoints.yaml").write_text(
        "endpoints:\n  m: http://file:9000/v1\n", encoding="utf-8"
    )
    assert endpoint() == "http://file:9000/v1"


def test_vllm_client_names_both_endpoint_sources(monkeypatch) -> None:  # noqa: ANN001
    from prism_vlm.models.router import get_client

    monkeypatch.delenv("PRISM_VLLM_ENDPOINT", raising=False)
    cfg = ModelsConfig(
        models=[ModelSpec(id="m", provider="vllm", model="M")]
    )
    with pytest.raises(ValueError) as e:
        get_client("m", cfg)
    assert "endpoints.yaml" in str(e.value) and "PRISM_VLLM_ENDPOINT" in str(e.value)


def test_router_rejects_unknown_provider() -> None:
    from prism_vlm.models.router import get_client

    cfg = ModelsConfig(
        models=[ModelSpec(id="m", provider="nope", model="M")]
    )
    with pytest.raises(ValueError, match="Unknown provider"):
        get_client("m", cfg)


def test_generation_prompts_are_present_and_formattable() -> None:
    from prism_vlm.evaluation.judge import _gen_prompt

    assert _gen_prompt("h_pool").format(sub_type="absent_entity")
    assert _gen_prompt("multi_qa").format(question="q?", gold="a")
    assert _gen_prompt("sc_wrong").format(question="q?", gold="a")
    # These two are sent verbatim -- they must not need substitution.
    for name in ("counting", "vis_pair"):
        assert _gen_prompt(name).strip()


# --------------------------------------------------------------------------
# io + message shape
# --------------------------------------------------------------------------

def test_write_json_atomic_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "x.json"
    write_json_atomic(path, {"a": [1, 2], "b": "ü"})
    assert read_json(path) == {"a": [1, 2], "b": "ü"}
    assert not list(path.parent.glob("*.tmp*")), "tmp file left behind"


def test_build_multimodal_messages_shape() -> None:
    msgs = build_multimodal_messages("hello", (_PNG,))
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    kinds = [p["type"] for p in msgs[0]["content"]]
    assert "text" in kinds and "image_url" in kinds


def test_google_client_preserves_roles() -> None:
    """Regression: a flattened history hands sc the model's own answers as user text."""
    pytest.importorskip("google.genai")
    from prism_vlm.models.google_client import GoogleClient

    contents = GoogleClient._build_contents(
        None,
        [
            {"role": "user", "content": [{"type": "text", "text": "q?"}]},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "are you sure?"},
        ],
    )
    assert [c["role"] for c in contents] == ["user", "model", "user"]
    assert contents[1]["parts"] == [{"text": "first answer"}]


def test_anthropic_client_omits_temperature() -> None:
    """Claude 4.7+ rejects any non-default temperature, thinking on or off."""
    from prism_vlm.models.base import GenConfig

    captured: dict = {}

    class _Messages:
        async def create(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
                model_dump=dict,
            )

    client = _stub_anthropic(_Messages())
    asyncio.run(client.generate([{"role": "user", "content": "hi"}], GenConfig()))
    assert "temperature" not in captured


@pytest.mark.parametrize(
    ("reasoning", "budget", "expected"),
    [
        (None, None, None),
        ("off", None, None),
        # An effort level means adaptive thinking: Claude 4.6 and later.
        ("high", None, {"type": "adaptive"}),
        # A budget means the manual form: Claude 4.5 and earlier.
        (None, 2048, {"type": "enabled", "budget_tokens": 2048}),
        # A budget wins, so a spec carrying both is unambiguous.
        ("high", 2048, {"type": "enabled", "budget_tokens": 2048}),
    ],
)
def test_anthropic_thinking_shape_is_explicit(reasoning, budget, expected) -> None:  # noqa: ANN001
    """The two Anthropic thinking APIs do not overlap, so the spec picks one."""
    from prism_vlm.models.base import GenConfig

    captured: dict = {}

    class _Messages:
        async def create(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn",
                model_dump=dict,
            )

    client = _stub_anthropic(_Messages(), reasoning=reasoning, budget=budget)
    asyncio.run(client.generate([{"role": "user", "content": "hi"}], GenConfig()))
    assert captured.get("thinking") == expected
    if expected and expected["type"] == "adaptive":
        assert captured["output_config"] == {"effort": reasoning}
    else:
        assert "output_config" not in captured


# --------------------------------------------------------------------------
# the build loop, with a stub judge
# --------------------------------------------------------------------------

class _StubJudge:
    """Returns a well-formed artifact for every axis, with no network."""

    def __init__(self, cfg) -> None:  # noqa: ANN001
        self.cfg = cfg

    async def generate_h_pool_trap(self, image_bytes, sub_type):  # noqa: ANN001
        return {"sub_type": sub_type, "question": f"trap? ({sub_type})",
                "absent_entity": "dragon"}

    async def generate_multi_qa_extras(self, question, gold, image_bytes):  # noqa: ANN001
        return [{"question": f"extra {i}?", "gold": f"a{i}"} for i in range(3)]

    async def generate_counting_question(self, image_bytes):  # noqa: ANN001
        return {"question": "how many?", "gold_count": 4, "object": "bananas"}

    async def generate_vision_pair(self, image_bytes):  # noqa: ANN001
        return {"true_caption": "a black dog", "fake_caption": "a brown dog",
                "swapped_detail": {"original": "black", "fake": "brown"}}

    async def generate_sc_wrong_answer(self, question, gold, image_bytes):  # noqa: ANN001
        return {"wrong_answer": "a bus"}


def test_build_writes_five_artifact_families(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(build_pipeline, "JudgeClient", _StubJudge)

    items = [
        Item(
            item_id=item_id_for("okvqa", i, f"q{i}?"),
            domain="general_vqa",
            benchmark="okvqa",
            question=f"q{i}?",
            images=(_PNG,),
            gold=("a van", "van"),
        )
        for i in range(4)
    ]
    bundle = tmp_path / "variants" / "v1"
    run_cfg = load_run_config(CONFIGS / "smoke_run.yaml")
    domains_cfg = load_domains(CONFIGS / "smoke_domains.yaml")

    out = asyncio.run(
        build_pipeline.build_variant_bundle(
            run_cfg, domains_cfg, bundle_dir=bundle, items=items
        )
    )
    assert out == bundle

    for family in ("h", "multi_qa", "counting", "vis", "sc"):
        files = sorted((bundle / family).glob("*.json"))
        assert len(files) == len(items), f"{family}: {len(files)} files"

    manifest = read_json(bundle / "manifest.json")
    assert manifest["n_items"] == len(items)
    assert manifest["axes"] == ["q", "IF", "h", "mq", "sc", "c", "vis"]
    for axis in ("h", "mq", "c", "vis", "sc"):
        assert manifest[f"acceptance_rate_{axis}"] == 1.0

    # Artifact contents match what the evaluation stage reads back.
    one = items[0].item_id
    h = read_json(bundle / "h" / f"{one}.json")
    assert h["sub_type"] in SUB_TYPES and h["question"]
    mq = read_json(bundle / "multi_qa" / f"{one}.json")
    assert mq["questions"][0] == items[0].question
    assert len(mq["questions"]) == len(mq["golds"]) == 4
    assert read_json(bundle / "counting" / f"{one}.json")["gold_count"] == 4
    assert read_json(bundle / "sc" / f"{one}.json")["wrong_answer"] == "a bus"


def test_build_is_idempotent(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """Re-running must resume on file existence, not re-synthesise."""
    monkeypatch.setattr(build_pipeline, "JudgeClient", _StubJudge)
    items = [Item(item_id=item_id_for("okvqa", 0, "q?"), domain="general_vqa",
                  benchmark="okvqa", question="q?", images=(_PNG,), gold="a van")]
    bundle = tmp_path / "variants" / "v1"
    run_cfg = load_run_config(CONFIGS / "smoke_run.yaml")
    domains_cfg = load_domains(CONFIGS / "smoke_domains.yaml")

    args = (run_cfg, domains_cfg)
    asyncio.run(build_pipeline.build_variant_bundle(*args, bundle_dir=bundle, items=items))
    path = bundle / "h" / f"{items[0].item_id}.json"
    first = path.read_bytes()
    stamp = path.stat().st_mtime_ns

    asyncio.run(build_pipeline.build_variant_bundle(*args, bundle_dir=bundle, items=items))
    assert path.read_bytes() == first
    assert path.stat().st_mtime_ns == stamp, "artifact was rewritten on resume"


# --------------------------------------------------------------------------
# the run loop, with a stub client and judge
# --------------------------------------------------------------------------

class _StubClient:
    """Echoes the prompt text it was handed, with no network."""

    model_id = "stub"

    async def generate(self, messages, config):  # noqa: ANN001
        from prism_vlm.models.base import RawResponse

        text = next(p["text"] for p in messages[0]["content"] if p["type"] == "text")
        return RawResponse(text=f"answered: {text}", finish_reason="stop", usage={}, raw={})


class _RecordingJudge:
    """Captures the question each grader was grounded on."""

    def __init__(self) -> None:
        self.seen: dict[str, str] = {}

    async def grade_hallucination(self, sub_type, question, response, context):  # noqa: ANN001
        self.seen["h"] = question
        return {"hallucinated": 0}

    async def grade_counting(self, question, gold_count, response):  # noqa: ANN001
        self.seen["c"] = question
        return {"verdict": 1}


def test_call_cache_records_the_question_that_was_sent(tmp_path: Path) -> None:
    from prism_vlm.runner.cache import CacheLayout
    from prism_vlm.runner.concurrency import ConcurrencyGates
    from prism_vlm.runner.run_pipeline import _call_and_cache

    cache = CacheLayout(run_root=tmp_path)
    cache.ensure()
    variant = Variant(
        variant_id=variant_id_for("item1", "h", 0),
        item_id="item1",
        axis="h",
        slot=0,
        question="Is the dragon on the left or the right?",
        images=(_PNG,),
        meta={"sub_type": "absent_entity", "absent_entity": "dragon"},
    )

    async def _go():  # noqa: ANN202
        gates = ConcurrencyGates.build()
        return await _call_and_cache(_StubClient(), variant, 0, cache, gates.model_gate)

    payload = asyncio.run(_go())
    assert payload["question"] == variant.question
    assert read_json(cache.call_path("stub", "item1", "h", 0))["question"] == variant.question


def test_h_and_c_grade_against_the_question_the_model_saw(tmp_path: Path) -> None:
    """Regression: both graders used to fall back to the recycled original question."""
    from prism_vlm.runner.cache import CacheLayout
    from prism_vlm.runner.concurrency import ConcurrencyGates
    from prism_vlm.runner.run_pipeline import _score_c, _score_h

    cache = CacheLayout(run_root=tmp_path)
    cache.ensure()
    item = Item(
        item_id="item1",
        domain="general_vqa",
        benchmark="okvqa",
        question="What colour is the van?",
        images=(_PNG,),
        gold=("white",),
    )
    write_json_atomic(
        cache.call_path("stub", "item1", "h", 0),
        {"question": "How many dragons are in the image?",
         "text": "I do not see a dragon.",
         "meta": {"sub_type": "absent_entity", "absent_entity": "dragon"}},
    )
    write_json_atomic(
        cache.call_path("stub", "item1", "c", 0),
        {"question": "How many bananas are in the image?",
         "text": "Four.",
         "meta": {"gold_count": 4, "object": "bananas"}},
    )
    judge = _RecordingJudge()

    async def _go():  # noqa: ANN202
        gates = ConcurrencyGates.build()
        await _score_h(item, "stub", cache, gates, judge)
        await _score_c(item, "stub", cache, gates, judge)

    asyncio.run(_go())
    assert judge.seen["h"] == "How many dragons are in the image?"
    assert judge.seen["c"] == "How many bananas are in the image?"
    assert item.question not in judge.seen.values()


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------

# Two items chosen so the conditional sc rule changes the answer: item2 is
# wrong at turn 0, so it is outside the sc denominator entirely.
_SCORE_ROWS = [
    {"item_id": "i1", "domain": "d", "benchmark": "b",
     "q": 1.0, "IF": 1.0, "h": 1.0, "mq": 1.0, "sc": 1.0, "c": 1.0, "vis": 1.0},
    {"item_id": "i2", "domain": "d", "benchmark": "b",
     "q": 0.0, "IF": 0.5, "h": 1.0, "mq": 0.0, "sc": 0.0, "c": 1.0, "vis": None},
]


def test_axis_means_follow_the_published_definition() -> None:
    from prism_vlm.scoring import axis_means

    means = axis_means(_SCORE_ROWS)
    assert means["q"] == 0.5
    assert means["IF"] == 0.75
    assert means["mq"] == 0.5
    # vis: the second item carries no artifact, so it is dropped rather than zeroed.
    assert means["vis"] == 1.0
    # sc: only the turn-0-correct item counts, so 1.0 and not the plain mean 0.5.
    assert means["sc"] == 1.0


def test_pscore_is_the_harmonic_mean_of_the_axis_means() -> None:
    from prism_vlm.scoring import axis_means, pscore

    means = axis_means(_SCORE_ROWS)
    # 7 / (1/.5 + 1/.75 + 1/1 + 1/.5 + 1/1 + 1/1 + 1/1) == 7 / (28/3) == 0.75
    assert pscore(means) == pytest.approx(0.75)
    assert pscore(means, "min") == 0.5
    # means are .5 .75 1 .5 1 1 1 -> sum 5.75
    assert pscore(means, "arithmetic") == pytest.approx(5.75 / 7)
    assert pscore(means, "geometric") > pscore(means, "harmonic")


def test_empty_axis_does_not_divide_by_zero() -> None:
    from prism_vlm.scoring import axis_means, pscore

    means = axis_means([{**_SCORE_ROWS[0], "c": 0.0}])
    assert means["c"] == 0.0
    assert pscore(means) == pytest.approx(0.0, abs=1e-5)


def test_score_report_round_trip(tmp_path: Path) -> None:
    from prism_vlm.scoring import discover_models, format_table, score_models, write_report

    scores = tmp_path / "scores"
    for name, rows in (("strong", _SCORE_ROWS), ("weak", [{**_SCORE_ROWS[1], "q": 0.0}])):
        for i, row in enumerate(rows):
            path = scores / name / row["domain"] / f"item{i}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, row)

    assert discover_models(scores) == ["strong", "weak"]
    report = score_models(scores, discover_models(scores))
    assert [r["model_id"] for r in report] == ["strong", "weak"]  # best first
    assert report[0]["n_items"] == 2

    written = write_report(report, tmp_path / "out", "harmonic")
    assert [p.name for p in written] == ["scores.json", "scores.csv"]
    assert read_json(tmp_path / "out" / "scores.json")["aggregator"] == "harmonic"
    header = (tmp_path / "out" / "scores.csv").read_text().splitlines()[0]
    assert header.split(",") == ["model_id", "n_items", *AXES_ORDER, "PScore"]
    assert "PScore" in format_table(report, "harmonic")


def test_cli_aggregator_choices_match_the_registry() -> None:
    """The CLI keeps its own tuple so argparse needs no scoring import."""
    from prism_vlm.cli import AGGREGATOR_NAMES
    from prism_vlm.scoring import AGGREGATORS

    assert set(AGGREGATOR_NAMES) == set(AGGREGATORS)


# --------------------------------------------------------------------------
# judge output parsing
# --------------------------------------------------------------------------

def test_parse_json_tolerates_fences_and_prose() -> None:
    from prism_vlm.evaluation.judge import _parse_json

    assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert _parse_json('Sure! {"a": 1} hope that helps') == {"a": 1}
    assert _parse_json("not json at all") == {}
    assert _parse_json("[1, 2]") == {}


def test_parse_verdict_falls_back_explicitly() -> None:
    from prism_vlm.evaluation.judge import _parse_verdict

    assert _parse_verdict('{"verdict": 1}', default=0) == {"verdict": 1}
    out = _parse_verdict("garbage", default=0)
    assert out["verdict"] == 0 and "unparseable" in out["reason"]


def test_clean_verdicts_pads_and_truncates() -> None:
    from prism_vlm.evaluation.judge import _clean_verdicts

    assert _clean_verdicts([1, 1, 1], 2) == [1, 1]
    assert _clean_verdicts([1], 3) == [1, 0, 0]
    assert _clean_verdicts(["1", "0", "maybe"], 3) == [1, 0, 0]
    assert _clean_verdicts(None, 2) == [0, 0]


def test_grading_prompts_are_present_and_formattable() -> None:
    from prism_vlm.evaluation.judge import _judge_prompt

    assert _judge_prompt("if").format(constraints="1. lowercase", response="a van.")
    assert _judge_prompt("hallucination").format(
        sub_type="absent_entity", context="dog", question="How many dogs?", response="One."
    )
    assert _judge_prompt("multi_qa").format(n=2, qa_block="1. Q: a\n   gold: b", response="x")
    assert _judge_prompt("sc_change").format(question="q?", prev="a", curr="b")
    assert _judge_prompt("counting").format(question="how many?", gold_count=2, response="3")
    assert _judge_prompt("vis_pair").format(response="A")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

# 1x1 PNG.
_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae426082"
)

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)


@pytest.mark.parametrize("module,cls,key", [
    ("openai_client", "OpenAIClient", "max_completion_tokens"),
    ("openrouter_client", "OpenRouterClient", "max_tokens"),
    ("vllm_client", "VLLMClient", "max_tokens"),
])
def test_registry_max_tokens_reaches_every_openai_shaped_provider(
    module: str, cls: str, key: str, monkeypatch,  # noqa: ANN001
) -> None:
    """ModelSpec.max_tokens is documented as a per-model override for all of them."""
    mod = importlib.import_module(f"prism_vlm.models.{module}")
    captured: dict = {}

    class _Completions:
        async def create(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="ok"), finish_reason="stop")],
                usage=SimpleNamespace(model_dump=dict),
                model_dump=dict,
            )

    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    monkeypatch.setattr(mod, "AsyncOpenAI", lambda *a, **k: SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions())))
    client = getattr(mod, cls)(ModelSpec(
        id="stub", provider="p", model="m", max_tokens=99, endpoint="http://x/v1"))
    asyncio.run(client.generate([{"role": "user", "content": "hi"}], GenConfig()))
    assert captured[key] == 99


def test_anthropic_max_tokens_leaves_room_for_the_thinking_budget() -> None:
    """A reasoning_budget at or above the default budget is a 400 without it."""
    captured: dict = {}

    class _Messages:
        async def create(self, **kwargs):  # noqa: ANN003
            captured.update(kwargs)
            return SimpleNamespace(
                content=[SimpleNamespace(type="text", text="ok")],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1),
                stop_reason="end_turn", model_dump=dict)

    client = _stub_anthropic(_Messages(), budget=4096, max_tokens=8192)
    asyncio.run(client.generate([{"role": "user", "content": "hi"}], GenConfig()))
    assert captured["max_tokens"] == 8192
    assert captured["thinking"]["budget_tokens"] == 4096
    assert captured["max_tokens"] > captured["thinking"]["budget_tokens"]


@pytest.mark.parametrize("extra", ["anthropic", "google"])
def test_package_imports_without_the_optional_extras(extra: str) -> None:
    """README's test recipe installs [dev], which carries neither SDK."""
    code = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        f"        if name == {extra!r} or name.startswith({extra + '.'!r}):\n"
        "            raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import pkgutil, importlib, prism_vlm\n"
        "bad = []\n"
        "for m in pkgutil.walk_packages(prism_vlm.__path__, 'prism_vlm.'):\n"
        "    try:\n"
        "        importlib.import_module(m.name)\n"
        "    except Exception as e:\n"
        "        bad.append(f'{m.name}: {e!r}')\n"
        "print('|'.join(bad))"
    )
    assert _run_child(code, "0") == ""


def test_an_axis_with_no_data_is_named_not_just_scored_zero() -> None:
    """A partial `run --axes` otherwise reads as a model that failed everything."""
    from prism_vlm.scoring.aggregate import axis_means, empty_axes, format_table, pscore

    rows = [{"q": 1.0, "IF": None, "h": None, "mq": None,
             "sc": None, "c": None, "vis": None} for _ in range(5)]
    assert empty_axes(rows) == ["IF", "h", "mq", "sc", "c", "vis"]
    assert empty_axes([{a: 1.0 for a in AXES_ORDER}]) == []

    means = axis_means(rows)
    report = [{"model_id": "m", "n_items": 5, **means,
               "PScore": pscore(means), "empty_axes": empty_axes(rows)}]
    table = format_table(report, "harmonic")
    assert "no scored items on" in table and "vis" in table.split("note:")[1]


def _stub_anthropic(messages, reasoning=None, budget=None, max_tokens=None):  # noqa: ANN001, ANN202
    """AnthropicClient built from a real ModelSpec, SDK replaced — no key, no network.

    Constructed through ``__init__`` rather than ``object.__new__`` so the
    ModelSpec-to-client mapping is part of what these tests cover.
    """
    from prism_vlm.models import anthropic_client as mod

    # The SDK is imported inside __init__, so the stub goes in sys.modules:
    # that exercises the real import path instead of side-stepping it.
    real = sys.modules.get("anthropic")
    sys.modules["anthropic"] = SimpleNamespace(
        AsyncAnthropic=lambda *a, **k: SimpleNamespace(messages=messages))
    try:
        return mod.AnthropicClient(ModelSpec(
            id="stub", provider="anthropic", model="claude-x",
            reasoning=reasoning, reasoning_budget=budget, max_tokens=max_tokens,
        ))
    finally:
        if real is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = real


def _run_child(code: str, hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": _REPO_ROOT}
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()
