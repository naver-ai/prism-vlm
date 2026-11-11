"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

h axis (hallucination): 3-subtype pool {absent_entity, misleading_premise, uncertain_image}.

Bundle layout (`bundle_dir/h/<item_id>.json`):
    {
      "item_id": "...",
      "sub_type": "absent_entity" | "misleading_premise" | "uncertain_image",
      "question": "<trap question text>",
      "meta": {...sub-type-specific fields...}
    }

Sub-type is chosen at bundle-build time (deterministic per item) so re-runs hit
the same trap. Judge grades the response with `grade_hallucination`.
"""

from __future__ import annotations

from pathlib import Path

from ..data.schemas import Item, Variant
from ..utils.hashing import variant_id_for
from ..utils.io import read_json

SUB_TYPES = ("absent_entity", "misleading_premise", "uncertain_image")


class HPool:
    axis: str = "h"

    def generate(self, item: Item, bundle_dir: Path) -> list[Variant]:
        path = bundle_dir / "h" / f"{item.item_id}.json"
        if not path.exists():
            return []
        payload = read_json(path)
        return [
            Variant(
                variant_id=variant_id_for(item.item_id, "h", 0),
                item_id=item.item_id,
                axis="h",
                slot=0,
                question=payload["question"],
                images=item.images,
                meta={
                    "sub_type": payload["sub_type"],
                    **payload.get("meta", {}),
                },
            )
        ]
