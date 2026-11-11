"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

4-way bounded-output multiple-choice adapter for RefCOCOg / RefCOCO+.

Turns each ``(image, referring expression, bbox)`` row into a 4-way MC item:
the ground-truth box plus three distractor boxes whose IoU against the gold
box is < 0.3. Seeding uses ``item_id`` so the adapter is deterministic.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from ..utils.hashing import deterministic_rng
from .schemas import Item

_MC_LETTERS = ("A", "B", "C", "D")
_IOU_THRESHOLD = 0.3


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _to_xyxy(box: Any) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"Expected 4-tuple bbox, got {box!r}")
    x1, y1, x2_or_w, y2_or_h = (float(v) for v in box)
    # Common convention: [x, y, w, h]. We convert to x1y1x2y2 heuristically.
    if x2_or_w > x1 and y2_or_h > y1:
        return (x1, y1, x2_or_w, y2_or_h)
    return (x1, y1, x1 + x2_or_w, y1 + y2_or_h)


class BoundedMC4Adapter:
    def __init__(self, n_choices: int = 4, iou_threshold: float = _IOU_THRESHOLD) -> None:
        self.n_choices = n_choices
        self.iou_threshold = iou_threshold

    def wrap(self, item: Item, row: dict[str, Any]) -> Item:
        """Convert a bbox-grounded item into a 4-way MC item.

        We accept whatever bbox-ish field survived on ``item.gold`` / ``row``:
        the adapter keeps the original image, builds ``n_choices - 1``
        distractors via random jitter + IoU rejection, and appends a lettered
        choice list to the question text.
        """
        try:
            gold_box = _to_xyxy(item.gold if isinstance(item.gold, (list, tuple)) else row["bbox"])
        except Exception:
            return item  # leave untouched if bbox unavailable

        rng = deterministic_rng(item.item_id, "bounded_mc4")
        w = gold_box[2] - gold_box[0]
        h = gold_box[3] - gold_box[1]

        distractors: list[tuple[float, float, float, float]] = []
        attempts = 0
        while len(distractors) < self.n_choices - 1 and attempts < 200:
            attempts += 1
            shift_x = rng.uniform(-3 * w, 3 * w)
            shift_y = rng.uniform(-3 * h, 3 * h)
            scale = rng.uniform(0.5, 1.8)
            nw, nh = w * scale, h * scale
            x1 = max(0.0, gold_box[0] + shift_x)
            y1 = max(0.0, gold_box[1] + shift_y)
            cand = (x1, y1, x1 + nw, y1 + nh)
            if _iou(gold_box, cand) < self.iou_threshold:
                distractors.append(cand)

        if len(distractors) < self.n_choices - 1:
            return item

        ordering = list(distractors) + [gold_box]
        rng.shuffle(ordering)
        gold_letter = _MC_LETTERS[ordering.index(gold_box)]

        lines = [item.question, "", "Choose the region that best matches:"]
        for letter, box in zip(_MC_LETTERS, ordering, strict=True):
            lines.append(f"{letter}. box={tuple(round(v, 1) for v in box)}")
        lines.append("")
        lines.append("Respond with a single letter: A, B, C, or D.")
        new_q = "\n".join(lines)

        return replace(
            item,
            question=new_q,
            gold=gold_letter,
            meta={**item.meta, "grounding_adapter": "bounded_mc4", "choices": ordering},
        )
