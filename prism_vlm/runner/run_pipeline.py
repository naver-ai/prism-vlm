"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

`prism-vlm run`: orchestrates item × model × variant over the seven axes.

Axes: ``q`` (Quality), ``IF`` (Instruction following), ``h`` (Hallucination),
``mq`` (Multi-question), ``sc`` (Pressure/sycophancy), ``c`` (Counting),
``vis`` (Visual discrimination).

Phases per model:
  1.  Standard parallel inference for q/IF/h/mq/c/vis (one call per (item, axis)).
  2.  Pressure dialogue — reuses q's turn-0 response, issues 3 escalating turns
      with dynamic termination (stops as soon as the judge detects a change).
  3.  Scoring — judge-grade every cached call and write the per-item score row.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path

from ..config.loader import DomainsConfig, ModelsConfig, RunConfig
from ..data.sampling import hydrate_items, load_sampled_items, take_sample
from ..data.schemas import Item, Variant
from ..evaluation.judge import JudgeClient
from ..models.base import GenConfig, VLMClient, build_multimodal_messages
from ..models.router import get_client
from ..utils.io import read_json, write_json_atomic
from ..utils.seeds import REACH_K_SEEDS
from ..variants import all_generators
from ..variants.self_correction import pick_pressure_templates
from .cache import CacheLayout
from .concurrency import ConcurrencyGates

# Axes that take exactly one inference call per item (cached at call_path).
_SIMPLE_AXES: tuple[tuple[str, int], ...] = (
    ("q", 0), ("IF", 0), ("h", 0), ("mq", 0), ("c", 0), ("vis", 0),
)
# sc takes a multi-turn loop handled separately.
_ALL_AXIS_NAMES = ("q", "IF", "h", "mq", "sc", "c", "vis")


async def _call_and_cache(
    client: VLMClient,
    variant: Variant,
    seed: int,
    cache: CacheLayout,
    gate: asyncio.Semaphore,
) -> dict:
    path = cache.call_path(client.model_id, variant.item_id, variant.axis, variant.slot)
    if path.exists():
        return read_json(path)
    messages = build_multimodal_messages(variant.question, variant.images)
    cfg = GenConfig(seed=seed)
    async with gate:
        resp = await client.generate(messages, cfg)
    payload = {
        "variant_id": variant.variant_id,
        "item_id": variant.item_id,
        "axis": variant.axis,
        "slot": variant.slot,
        "seed": seed,
        "model_id": client.model_id,
        "question": variant.question,
        "text": resp.text,
        "finish_reason": resp.finish_reason,
        "usage": resp.usage,
        "meta": variant.meta,
    }
    write_json_atomic(path, payload)
    return payload


async def _run_sc_dialogue(
    client: VLMClient,
    item: Item,
    t0_text: str,
    bundle_dir: Path,
    cache: CacheLayout,
    gates: ConcurrencyGates,
    judge: JudgeClient,
) -> dict:
    """Run the 3-turn pressure dialogue and persist the verdict.

    Returns ``{"survived": 0|1, "broken_at": int|None, "transcript": [...]}``.
    Cached under ``judge_path(model, item, "sc")`` so re-runs are free.
    """
    sc_judge_path = cache.judge_path(client.model_id, item.item_id, "sc")
    if sc_judge_path.exists():
        return read_json(sc_judge_path)

    t1, t2, t3 = pick_pressure_templates(item.item_id)
    # T3 false-hint uses a bundle-precomputed plausible-but-wrong answer
    # (judge-generated at build time, image-aware). Falls back to a generic
    # placeholder if the bundle entry is missing.
    sc_bundle = bundle_dir / "sc" / f"{item.item_id}.json"
    if sc_bundle.exists():
        wrong_answer = read_json(sc_bundle).get("wrong_answer", "something different")
    else:
        wrong_answer = "something different"
    t3_filled = t3.format(wrong=wrong_answer)

    image_content = build_multimodal_messages("", item.images)[0]["content"][:-1]
    history: list[dict] = [
        {"role": "user", "content": [*image_content, {"type": "text", "text": item.question}]},
        {"role": "assistant", "content": t0_text},
    ]

    transcript: list[dict] = [{"turn": 0, "text": t0_text}]
    survived = 1
    broken_at = None
    prev_text = t0_text

    for turn_idx, (template, label) in enumerate(
        ((t1, "t1"), (t2, "t2"), (t3_filled, "t3")), start=1
    ):
        history.append({"role": "user", "content": template})
        cfg = GenConfig(seed=0)
        async with gates.model_gate:
            resp = await client.generate(history, cfg)
        history.append({"role": "assistant", "content": resp.text})
        transcript.append({"turn": turn_idx, "label": label, "prompt": template, "text": resp.text})

        async with gates.judge_gate:
            change = await judge.detect_sc_change(item.question, prev_text, resp.text)
        transcript[-1]["change"] = change
        if change.get("changed", 0) == 1:
            survived = 0
            broken_at = turn_idx
            break
        prev_text = resp.text

    payload = {
        "model_id": client.model_id,
        "item_id": item.item_id,
        "survived": survived,
        "broken_at": broken_at,
        "transcript": transcript,
    }
    write_json_atomic(sc_judge_path, payload)
    return payload


async def _score_q(item, model_id, q_payload, cache, gates, judge) -> float:
    judge_path = cache.judge_path(model_id, item.item_id, "q")
    if judge_path.exists():
        return float(read_json(judge_path).get("verdict", 0))
    async with gates.judge_gate:
        graded = await judge.grade_q(item.benchmark, item.question, item.gold, q_payload["text"])
    write_json_atomic(judge_path, graded)
    return float(graded.get("verdict", 0))


async def _score_if(item, model_id, cache, gates, judge) -> float:
    if_path = cache.call_path(model_id, item.item_id, "IF", 0)
    if not if_path.exists():
        return 0.0
    from ..variants.if_constraints import Constraint, format_constraints

    if_payload = read_json(if_path)
    raw_constraints = if_payload.get("meta", {}).get("constraints", [])
    n = len(raw_constraints)
    if n == 0:
        return 1.0
    judge_path = cache.judge_path(model_id, item.item_id, "IF")
    if judge_path.exists():
        verdicts = read_json(judge_path).get("verdicts", [])
    else:
        cobjs = tuple(Constraint(c["name"], c["params"]) for c in raw_constraints)
        rendered = format_constraints(cobjs)
        numbered = "\n".join(
            line for line in rendered.splitlines() if line.strip().startswith(tuple("123"))
        )
        async with gates.judge_gate:
            graded = await judge.grade_if(if_payload["text"], numbered, n)
        write_json_atomic(judge_path, graded)
        verdicts = graded.get("verdicts", [])
    return sum(verdicts) / max(n, 1)


async def _score_h(item, model_id, cache, gates, judge) -> float:
    h_path = cache.call_path(model_id, item.item_id, "h", 0)
    if not h_path.exists():
        return 0.0
    h_payload = read_json(h_path)
    judge_path = cache.judge_path(model_id, item.item_id, "h")
    if judge_path.exists():
        graded = read_json(judge_path)
    else:
        meta = h_payload.get("meta", {})
        sub_type = meta.get("sub_type", "absent_entity")
        # Compact context for the judge — pick the relevant meta field.
        context_field = {
            "absent_entity": meta.get("absent_entity", ""),
            "misleading_premise": meta.get("false_premise", ""),
            "uncertain_image": meta.get("ambiguity_source", ""),
        }.get(sub_type, "")
        async with gates.judge_gate:
            graded = await judge.grade_hallucination(
                sub_type=sub_type,
                question=h_payload.get("question") or item.question,
                response=h_payload["text"],
                context=context_field,
            )
        graded = {**graded, "sub_type": sub_type, "context": context_field}
        write_json_atomic(judge_path, graded)
    # h_score = 1 - hallucinated (resistance metric).
    return 1.0 - float(graded.get("hallucinated", 1))


async def _score_mq(item, model_id, cache, gates, judge) -> float:
    mq_path = cache.call_path(model_id, item.item_id, "mq", 0)
    if not mq_path.exists():
        return 0.0
    mq_payload = read_json(mq_path)
    meta = mq_payload.get("meta", {})
    questions = meta.get("questions", [])
    golds = meta.get("golds", [])
    n = len(questions)
    if n == 0 or len(golds) != n:
        return 0.0
    judge_path = cache.judge_path(model_id, item.item_id, "mq")
    if judge_path.exists():
        verdicts = read_json(judge_path).get("verdicts", [])
    else:
        async with gates.judge_gate:
            graded = await judge.grade_multi_qa(mq_payload["text"], questions, golds)
        write_json_atomic(judge_path, graded)
        verdicts = graded.get("verdicts", [])
    if not verdicts or len(verdicts) != n:
        return 0.0
    return 1.0 if all(int(v) == 1 for v in verdicts) else 0.0


async def _score_vis(item, model_id, cache, gates, judge) -> float | None:
    vis_path = cache.call_path(model_id, item.item_id, "vis", 0)
    if not vis_path.exists():
        return None
    vis_payload = read_json(vis_path)
    meta = vis_payload.get("meta", {})
    letter_correct = meta.get("letter_correct")
    if letter_correct not in ("A", "B"):
        return None
    judge_path = cache.judge_path(model_id, item.item_id, "vis")
    if judge_path.exists():
        graded = read_json(judge_path)
    else:
        async with gates.judge_gate:
            graded = await judge.grade_vis(vis_payload["text"])
        graded = {**graded, "letter_correct": letter_correct}
        write_json_atomic(judge_path, graded)
    return 1.0 if graded.get("choice") == letter_correct else 0.0


async def _score_c(item, model_id, cache, gates, judge) -> float:
    c_path = cache.call_path(model_id, item.item_id, "c", 0)
    if not c_path.exists():
        return 0.0
    c_payload = read_json(c_path)
    meta = c_payload.get("meta", {})
    gold_count = meta.get("gold_count")
    if gold_count is None:
        return 0.0
    judge_path = cache.judge_path(model_id, item.item_id, "c")
    if judge_path.exists():
        return float(read_json(judge_path).get("verdict", 0))
    question = c_payload.get("question") or item.question
    async with gates.judge_gate:
        graded = await judge.grade_counting(question, int(gold_count), c_payload["text"])
    write_json_atomic(judge_path, graded)
    return float(graded.get("verdict", 0))


def _score_sc_from_judge(model_id: str, item_id: str, cache: CacheLayout) -> float:
    judge_path = cache.judge_path(model_id, item_id, "sc")
    if not judge_path.exists():
        return 0.0
    return float(read_json(judge_path).get("survived", 0))


async def _score_one_item(
    item: Item,
    model_id: str,
    cache: CacheLayout,
    gates: ConcurrencyGates,
    judge: JudgeClient,
) -> dict | None:
    """Compute and persist the per-axis score row for one item."""
    q_path = cache.call_path(model_id, item.item_id, "q", 0)
    if not q_path.exists():
        return None
    q_payload = read_json(q_path)

    q_score = await _score_q(item, model_id, q_payload, cache, gates, judge)
    if_score = await _score_if(item, model_id, cache, gates, judge)
    h_score = await _score_h(item, model_id, cache, gates, judge)
    mq_score = await _score_mq(item, model_id, cache, gates, judge)
    c_score = await _score_c(item, model_id, cache, gates, judge)
    vis_score = await _score_vis(item, model_id, cache, gates, judge)
    sc_score = _score_sc_from_judge(model_id, item.item_id, cache)

    row = {
        "model_id": model_id,
        "item_id": item.item_id,
        "domain": item.domain,
        "benchmark": item.benchmark,
        "q": q_score,
        "IF": if_score,
        "h": h_score,
        "mq": mq_score,
        "sc": sc_score,
        "c": c_score,
        "vis": vis_score,
    }
    write_json_atomic(cache.score_path(model_id, item.domain, item.item_id), row)
    return row


async def _run_single_model(
    client: VLMClient,
    items: list[Item],
    run_cfg: RunConfig,
    domains_cfg: DomainsConfig,
    cache: CacheLayout,
    gates: ConcurrencyGates,
    judge: JudgeClient,
    only_axes: set[str] | None = None,
) -> None:
    generators = all_generators()
    if only_axes is not None:
        generators = [g for g in generators if g.axis in only_axes]

    requested_axes = {g.axis for g in generators}

    # ---- Phase 1: which simple-axis calls are missing? ----
    pending_by_item: dict[str, set[tuple[str, int]]] = {}
    for item in items:
        pending: set[tuple[str, int]] = set()
        for axis, slot in _SIMPLE_AXES:
            if axis not in requested_axes:
                continue
            if not cache.call_path(client.model_id, item.item_id, axis, slot).exists():
                pending.add((axis, slot))
        if pending:
            pending_by_item[item.item_id] = pending

    # sc multi-turn needs the image at every turn even when simple-axis calls
    # are fully cached. So include items whose sc judge is still missing in the
    # hydration set, regardless of whether their simple-axis calls are pending.
    items_for_hydration_ids: set[str] = set(pending_by_item.keys())
    if "sc" in requested_axes:
        for item in items:
            sc_judge = cache.judge_path(client.model_id, item.item_id, "sc")
            q_call = cache.call_path(client.model_id, item.item_id, "q", 0)
            if not sc_judge.exists() and q_call.exists():
                items_for_hydration_ids.add(item.item_id)

    items_needing_work = [it for it in items if it.item_id in items_for_hydration_ids]
    if items_needing_work:
        n_simple_pending = len(pending_by_item)
        n_sc_only = len(items_for_hydration_ids) - n_simple_pending
        print(f"[run] {client.model_id}: hydrating {len(items_needing_work)} items "
              f"({n_simple_pending} pending simple-axis, {n_sc_only} sc-only)", flush=True)
        hydrated = hydrate_items(items_needing_work, domains_cfg, image_cache_dir=cache.run_root)
        hydrated_by_id = {it.item_id: it for it in hydrated}
        print(f"[run] {client.model_id}: hydration done, starting inference",
              flush=True)
    else:
        hydrated_by_id = {it.item_id: it for it in items if it.images}

    # ---- Phase 1a: parallel single-call inference for q/IF/h/mq/c/vis ----
    tasks: list[asyncio.Task[dict]] = []
    for item_id, pending in pending_by_item.items():
        item = hydrated_by_id.get(item_id) or next(it for it in items if it.item_id == item_id)
        for gen in generators:
            if gen.axis == "sc":
                continue  # handled in Phase 1b
            for variant in gen.generate(item, run_cfg.variant_bundle):
                if (variant.axis, variant.slot) not in pending:
                    continue
                tasks.append(
                    asyncio.create_task(
                        _call_and_cache(client, variant, 0, cache, gates.model_gate)
                    )
                )
    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            print(f"[run] {client.model_id}: {len(failures)}/{len(tasks)} simple "
                  f"calls failed (run continues). First: {failures[0]!r}",
                  flush=True)

    # ---- Phase 1b: sc multi-turn pressure dialogue ----
    if "sc" in requested_axes:
        sc_tasks: list[asyncio.Task[dict]] = []
        for item in items:
            sc_judge = cache.judge_path(client.model_id, item.item_id, "sc")
            if sc_judge.exists():
                continue
            q_path = cache.call_path(client.model_id, item.item_id, "q", 0)
            if not q_path.exists():
                continue
            t0_text = read_json(q_path)["text"]
            hydr_item = hydrated_by_id.get(item.item_id) or item
            sc_tasks.append(
                asyncio.create_task(
                    _run_sc_dialogue(client, hydr_item, t0_text,
                                     run_cfg.variant_bundle, cache, gates, judge)
                )
            )
        if sc_tasks:
            print(f"[run] {client.model_id}: sc dialogues × {len(sc_tasks)}", flush=True)
            sc_results = await asyncio.gather(*sc_tasks, return_exceptions=True)
            sc_failures = [r for r in sc_results if isinstance(r, BaseException)]
            if sc_failures:
                print(f"[run] {client.model_id}: {len(sc_failures)}/{len(sc_tasks)} sc "
                      f"dialogues failed. First: {sc_failures[0]!r}", flush=True)

    # ---- Phase 2: scoring (judge calls + score rows) ----
    print(f"[run] {client.model_id}: judging × {len(items)}", flush=True)
    score_tasks = [
        asyncio.create_task(_score_one_item(item, client.model_id, cache, gates, judge))
        for item in items
    ]
    total = len(score_tasks)
    step = max(total // 20, 1)
    done = 0
    judge_failures: list[BaseException] = []
    for fut in asyncio.as_completed(score_tasks):
        try:
            await fut
        except BaseException as exc:
            judge_failures.append(exc)
        done += 1
        if done % step == 0 or done == total:
            print(f"[run] {client.model_id}: judged {done}/{total} "
                  f"({done * 100 // total}%)", flush=True)
    if judge_failures:
        print(f"[run] {client.model_id}: {len(judge_failures)}/{total} judge "
              f"items failed. First: {judge_failures[0]!r}", flush=True)


async def run_model_evaluation(
    run_cfg: RunConfig,
    domains_cfg: DomainsConfig,
    models_cfg: ModelsConfig,
    only_models: Iterable[str] | None = None,
    only_axes: Iterable[str] | None = None,
    only_benchmarks: Iterable[str] | None = None,
    only_domains: Iterable[str] | None = None,
    limit: int | None = None,
) -> None:
    """Top-level entrypoint for ``prism-vlm run``.

    ``only_axes`` filters which axes are processed: ``q, IF, h, mq, sc, c, vis``.
    ``only_benchmarks``, ``only_domains`` and ``limit`` narrow the item pool for
    debug runs, and must match the ones the bundle was built with. An item the
    build skipped carries no artifact: it leaves the mean on ``vis`` alone, and
    scores 0 on ``h``, ``mq`` and ``c``, so a mismatched filter depresses those
    three rather than shrinking them.
    """
    base_dir = run_cfg.variant_bundle.parent.parent / "base"
    items, _seed_to_item_ids = load_sampled_items(
        domains_cfg, base_dir, seeds=REACH_K_SEEDS[: run_cfg.R], load_images=False,
        only_benchmarks=only_benchmarks, only_domains=only_domains,
    )
    items = take_sample(items, limit)

    cache = CacheLayout(run_cfg.cache_run_dir)
    cache.ensure()

    judge = JudgeClient(run_cfg.judge)
    judge_concurrency = run_cfg.judge_concurrency or run_cfg.concurrency
    judge_gate = ConcurrencyGates.build(1, judge_concurrency).judge_gate

    axes_set = set(only_axes) if only_axes else None

    model_ids = list(only_models) if only_models else [m.id for m in models_cfg.models]
    for model_id in model_ids:
        client = get_client(model_id, models_cfg)
        model_spec = models_cfg.by_id(model_id)
        model_concurrency = run_cfg.resolve_concurrency(model_spec)
        gates = ConcurrencyGates(
            model_gate=asyncio.Semaphore(model_concurrency),
            judge_gate=judge_gate,
        )
        print(f"[run] {model_id}: provider={model_spec.provider} "
              f"concurrency={model_concurrency}", flush=True)
        await _run_single_model(client, items, run_cfg, domains_cfg, cache,
                                 gates, judge, only_axes=axes_set)
