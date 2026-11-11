"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

c axis (Counting): judge-synthesised counting question per item.

Bundle layout (`bundle_dir/counting/<item_id>.json`):
    {
      "item_id": "...",
      "question":  "How many <objects> are in the image?",
      "gold_count": <int 1-100>,
      "object":     "<what's being counted>",
    }

Synthesis happens at bundle build time via `judge.generate_counting_question`.
If the
judge cannot produce a confident integer count it omits the file (graceful
skip).
"""

from __future__ import annotations

from pathlib import Path

from ..data.schemas import Item, Variant
from ..utils.hashing import variant_id_for
from ..utils.io import read_json


class Counting:
    axis: str = "c"

    def generate(self, item: Item, bundle_dir: Path) -> list[Variant]:
        path = bundle_dir / "counting" / f"{item.item_id}.json"
        if not path.exists():
            return []
        payload = read_json(path)
        return [
            Variant(
                variant_id=variant_id_for(item.item_id, "c", 0),
                item_id=item.item_id,
                axis="c",
                slot=0,
                question=payload["question"],
                images=item.images,
                meta={
                    "gold_count": payload["gold_count"],
                    "object": payload.get("object", ""),
                },
            )
        ]
