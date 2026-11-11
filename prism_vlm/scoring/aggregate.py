"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Per-axis means and the PScore aggregate.

Three details decide the numbers:

* ``sc`` is conditional. A model that was already wrong at turn 0 has no
  answer to hold on to, so resistance is measured only over the items whose
  ``q`` verdict is 1. Every other axis is a plain mean.
* A declined item leaves the mean only on ``vis``. Where the build judge
  refused to synthesise an artifact the item was never asked, but only ``vis``
  records that as ``None``; ``h``, ``mq`` and ``c`` record ``0.0``, so those
  items count as failures. The published numbers were computed that way and
  this reproduces them rather than correcting them. The same items are
  declined for every model, so the effect moves every model's mean alike.
* PScore aggregates the seven per-axis means, not the per-item rows. The
  harmonic mean is the paper's aggregate: one weak axis pulls the total down
  instead of being averaged away. Axis means are floored at ``EPS`` so an
  empty axis yields a near-zero PScore rather than a division by zero.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from ..utils.io import read_json, write_json_atomic

AXES = ("q", "IF", "h", "mq", "sc", "c", "vis")
EPS = 1e-6


def _harmonic(values: Sequence[float]) -> float:
    return len(values) / sum(1.0 / max(v, EPS) for v in values)


def _geometric(values: Sequence[float]) -> float:
    return math.exp(sum(math.log(max(v, EPS)) for v in values) / len(values))


def _arithmetic(values: Sequence[float]) -> float:
    return sum(values) / len(values)


AGGREGATORS: dict[str, Callable[[Sequence[float]], float]] = {
    "harmonic": _harmonic,
    "geometric": _geometric,
    "arithmetic": _arithmetic,
    "min": min,
}


def load_score_rows(scores_dir: Path, model_id: str) -> list[dict]:
    """Read every per-item row ``prism-vlm run`` wrote for one model."""
    model_dir = scores_dir / model_id
    if not model_dir.is_dir():
        return []
    return [
        read_json(f)
        for domain_dir in sorted(model_dir.iterdir()) if domain_dir.is_dir()
        for f in sorted(domain_dir.iterdir()) if f.suffix == ".json"
    ]


def discover_models(scores_dir: Path) -> list[str]:
    if not scores_dir.is_dir():
        return []
    return sorted(d.name for d in scores_dir.iterdir() if d.is_dir())


def axis_mean(rows: Iterable[dict], axis: str) -> float:
    values = [r[axis] for r in rows if r.get(axis) is not None]
    return sum(values) / len(values) if values else 0.0


def sc_mean(rows: Iterable[dict]) -> float:
    """sc over the items answered correctly at turn 0 — see the module docstring."""
    eligible = [r for r in rows if r.get("q") == 1.0]
    if not eligible:
        return 0.0
    return sum(1 for r in eligible if r.get("sc") == 1.0) / len(eligible)


def axis_means(rows: Iterable[dict]) -> dict[str, float]:
    rows = list(rows)
    means = {axis: axis_mean(rows, axis) for axis in AXES}
    means["sc"] = sc_mean(rows)
    return means


def empty_axes(rows: Iterable[dict]) -> list[str]:
    """Axes no row carried a value for -- a mean of 0 there means "not run".

    Their mean is 0.0 like any other total failure, and the harmonic mean
    then floors PScore at EPS, so a partial ``run --axes`` reads as a model
    that failed everything. The arithmetic is the paper's; this only names
    the axes so the table cannot be misread.
    """
    rows = list(rows)
    return [a for a in AXES if not any(r.get(a) is not None for r in rows)]


def pscore(means: dict[str, float], aggregator: str = "harmonic") -> float:
    return AGGREGATORS[aggregator]([means[axis] for axis in AXES])


def score_models(
    scores_dir: Path,
    model_ids: Iterable[str],
    aggregator: str = "harmonic",
) -> list[dict]:
    """Per-axis means plus PScore for each model, best first."""
    report = []
    for model_id in model_ids:
        rows = load_score_rows(scores_dir, model_id)
        if not rows:
            continue
        means = axis_means(rows)
        report.append({
            "model_id": model_id,
            "n_items": len(rows),
            **means,
            "PScore": pscore(means, aggregator),
            "empty_axes": empty_axes(rows),
        })
    return sorted(report, key=lambda r: -r["PScore"])


def format_table(report: Sequence[dict], aggregator: str) -> str:
    if not report:
        return "no scored items found — run `prism-vlm run` first"
    width = max(len(r["model_id"]) for r in report)
    head = f"{'model':<{width}}  {'n':>6}  " + "  ".join(f"{a:>5}" for a in AXES)
    head += f"  {'PScore':>7}"
    lines = [f"aggregator: {aggregator}", "", head, "-" * len(head)]
    for r in report:
        cells = "  ".join(f"{r[a]:>5.3f}" for a in AXES)
        lines.append(f"{r['model_id']:<{width}}  {r['n_items']:>6}  {cells}  {r['PScore']:>7.3f}")
    for r in report:
        if r.get("empty_axes"):
            lines.append(
                f"note: {r['model_id']} has no scored items on "
                f"{', '.join(r['empty_axes'])} — those axes read 0 and pull PScore down"
            )
    return "\n".join(lines)


def write_report(report: Sequence[dict], out_dir: Path, aggregator: str) -> list[Path]:
    """Write scores.json and scores.csv, and return the paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "scores.json"
    csv_path = out_dir / "scores.csv"
    write_json_atomic(json_path, {"aggregator": aggregator, "models": list(report)})
    columns = ["model_id", "n_items", *AXES, "PScore"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows({c: r[c] for c in columns} for r in report)
    return [json_path, csv_path]
