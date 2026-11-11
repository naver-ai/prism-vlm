"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

`prism-vlm build`: per-item bundle generation for the seven-axis design.

Five of the seven axes carry a judge-synthesised per-item artifact. Bundles
produced (under ``run_cfg.variant_bundle``):
  - ``h/<item_id>.json`` — one hallucination trap (sub-type rotated per item)
  - ``multi_qa/<item_id>.json`` — original Q + 2-4 judge-synthesised extras
  - ``counting/<item_id>.json`` — judge-synthesised counting question (1-100)
  - ``vis/<item_id>.json`` — true caption + fake caption with one detail swapped
  - ``sc/<item_id>.json`` — the plausible-but-wrong answer planted at turn Q3
  - ``manifest.json`` — generation model, item count, per-axis acceptance rates

The remaining two axes need no judge call and therefore no artifact: ``q``
reuses the recycled benchmark question and gold as-is, and ``if`` samples its
1-3 constraints deterministically from a fixed pool at evaluation time.

Each per-item bundle file is independent — re-runs are idempotent and resume
on file-existence checks.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from ..config.loader import DomainsConfig, RunConfig
from ..data.sampling import load_sampled_items, take_sample
from ..evaluation.judge import JudgeClient
from ..utils.hashing import deterministic_rng
from ..utils.io import write_json_atomic
from ..utils.seeds import REACH_K_SEEDS
from ..variants.h_pool import SUB_TYPES as H_SUB_TYPES


def _pick_h_sub_type(item_id: str) -> str:
    """Deterministic per-item sub-type rotation across the 3-type pool.

    Uses :func:`deterministic_rng` (SHA-256 of the item id) rather than the
    builtin ``hash``: CPython randomises string hashing per process, so a
    ``hash``-based rotation would hand the same item a different sub-type on
    every rebuild.
    """
    return H_SUB_TYPES[deterministic_rng(item_id, "h").randrange(len(H_SUB_TYPES))]


async def build_variant_bundle(
    run_cfg: RunConfig,
    domains_cfg: DomainsConfig,
    bundle_dir: Path | None = None,
    items: list | None = None,
    only_benchmarks: Iterable[str] | None = None,
    only_domains: Iterable[str] | None = None,
    limit: int | None = None,
) -> Path:
    bundle_dir = bundle_dir or run_cfg.variant_bundle
    bundle_dir.mkdir(parents=True, exist_ok=True)

    h_dir = bundle_dir / "h"
    mq_dir = bundle_dir / "multi_qa"
    c_dir = bundle_dir / "counting"
    vis_dir = bundle_dir / "vis"
    sc_dir = bundle_dir / "sc"
    for d in (h_dir, mq_dir, c_dir, vis_dir, sc_dir):
        d.mkdir(parents=True, exist_ok=True)

    if items is None:
        base_dir = bundle_dir.parent.parent / "base"
        items, _ = load_sampled_items(
            domains_cfg, base_dir, seeds=REACH_K_SEEDS[: run_cfg.R],
            only_benchmarks=only_benchmarks, only_domains=only_domains,
        )
        items = take_sample(items, limit)

    judge = JudgeClient(run_cfg.judge)
    sem = asyncio.Semaphore(run_cfg.judge_concurrency or run_cfg.concurrency)

    async def _build_h(item) -> bool:
        path = h_dir / f"{item.item_id}.json"
        if path.exists():
            return True
        sub_type = _pick_h_sub_type(item.item_id)
        async with sem:
            try:
                payload = await judge.generate_h_pool_trap(item.images[0], sub_type)
            except Exception as e:  # noqa: BLE001
                print(f"[build] h failed {item.item_id}: {e!r}", flush=True)
                return False
        question = payload.get("question")
        if not question:
            return False
        meta = {k: v for k, v in payload.items() if k not in ("question", "sub_type")}
        write_json_atomic(
            path,
            {
                "item_id": item.item_id,
                "sub_type": sub_type,
                "question": question,
                "meta": meta,
            },
        )
        return True

    async def _build_mq(item) -> bool:
        path = mq_dir / f"{item.item_id}.json"
        if path.exists():
            return True
        async with sem:
            try:
                extras = await judge.generate_multi_qa_extras(
                    item.question, item.gold, item.images[0]
                )
            except Exception as e:  # noqa: BLE001
                print(f"[build] mq failed {item.item_id}: {e!r}", flush=True)
                return False
        if len(extras) < 2:  # need at least 3 total (1 orig + 2 extras)
            return False
        gold0 = item.gold if isinstance(item.gold, str) else " | ".join(item.gold)
        questions = [item.question] + [e["question"] for e in extras]
        golds = [gold0] + [e["gold"] for e in extras]
        write_json_atomic(
            path,
            {"item_id": item.item_id, "questions": questions, "golds": golds},
        )
        return True

    async def _build_c(item) -> bool:
        path = c_dir / f"{item.item_id}.json"
        if path.exists():
            return True
        async with sem:
            try:
                payload = await judge.generate_counting_question(item.images[0])
            except Exception as e:  # noqa: BLE001
                print(f"[build] c failed {item.item_id}: {e!r}", flush=True)
                return False
        if not payload:
            return False
        write_json_atomic(
            path,
            {
                "item_id": item.item_id,
                "question": payload["question"],
                "gold_count": payload["gold_count"],
                "object": payload.get("object", ""),
            },
        )
        return True

    async def _build_vis(item) -> bool:
        path = vis_dir / f"{item.item_id}.json"
        if path.exists():
            return True
        async with sem:
            try:
                payload = await judge.generate_vision_pair(item.images[0])
            except Exception as e:  # noqa: BLE001
                print(f"[build] vis failed {item.item_id}: {e!r}", flush=True)
                return False
        if not payload:
            return False
        write_json_atomic(
            path,
            {
                "item_id": item.item_id,
                "true_caption": payload["true_caption"],
                "fake_caption": payload["fake_caption"],
                "swapped_detail": payload.get("swapped_detail", {}),
            },
        )
        return True

    async def _build_sc(item) -> bool:
        path = sc_dir / f"{item.item_id}.json"
        if path.exists():
            return True
        async with sem:
            try:
                payload = await judge.generate_sc_wrong_answer(
                    item.question, item.gold, item.images[0]
                )
            except Exception as e:  # noqa: BLE001
                print(f"[build] sc failed {item.item_id}: {e!r}", flush=True)
                return False
        if not payload:
            return False
        write_json_atomic(
            path,
            {"item_id": item.item_id, "wrong_answer": payload["wrong_answer"]},
        )
        return True

    print(f"[build] generating bundles for {len(items)} items "
          f"(judge={run_cfg.judge.model} effort={run_cfg.judge.effort})", flush=True)

    h_results = await asyncio.gather(*(_build_h(it) for it in items), return_exceptions=True)
    mq_results = await asyncio.gather(*(_build_mq(it) for it in items), return_exceptions=True)
    c_results = await asyncio.gather(*(_build_c(it) for it in items), return_exceptions=True)
    vis_results = await asyncio.gather(*(_build_vis(it) for it in items), return_exceptions=True)
    sc_results = await asyncio.gather(*(_build_sc(it) for it in items), return_exceptions=True)

    h_ok = sum(1 for r in h_results if r is True)
    mq_ok = sum(1 for r in mq_results if r is True)
    c_ok = sum(1 for r in c_results if r is True)
    vis_ok = sum(1 for r in vis_results if r is True)
    sc_ok = sum(1 for r in sc_results if r is True)

    manifest = {
        "version": bundle_dir.name,
        "generated_at": datetime.now(UTC).isoformat(),
        "generation_model": run_cfg.judge.model,
        "judge_effort": run_cfg.judge.effort,
        "n_items": len(items),
        "acceptance_rate_h": h_ok / max(len(items), 1),
        "acceptance_rate_mq": mq_ok / max(len(items), 1),
        "acceptance_rate_c": c_ok / max(len(items), 1),
        "acceptance_rate_vis": vis_ok / max(len(items), 1),
        "acceptance_rate_sc": sc_ok / max(len(items), 1),
        "axes": ["q", "IF", "h", "mq", "sc", "c", "vis"],
    }
    write_json_atomic(bundle_dir / "manifest.json", manifest)
    print(f"[build] done — h:{h_ok} mq:{mq_ok} c:{c_ok} vis:{vis_ok} sc:{sc_ok} / {len(items)}",
          flush=True)
    return bundle_dir
