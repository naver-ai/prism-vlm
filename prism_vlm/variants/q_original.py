"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

q axis: original question, unchanged.
"""

from __future__ import annotations

from pathlib import Path

from ..data.schemas import Item, Variant
from ..utils.hashing import variant_id_for


class QOriginal:
    axis: str = "q"

    def generate(self, item: Item, bundle_dir: Path) -> list[Variant]:
        variant = Variant(
            variant_id=variant_id_for(item.item_id, "q", 0),
            item_id=item.item_id,
            axis="q",
            slot=0,
            question=item.question,
            images=item.images,
        )
        return [variant]
