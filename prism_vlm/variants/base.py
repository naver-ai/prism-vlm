"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Variant generator protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..data.schemas import Item, Variant


class VariantGenerator(Protocol):
    axis: str

    def generate(self, item: Item, bundle_dir: Path) -> list[Variant]:
        """Return the list of variants this axis contributes for ``item``.

        Returning ``[]`` means the axis has no new per-model call to make --
        either its build artifact is absent for this item, or the axis is
        driven elsewhere, as sc is by the run pipeline's dialogue loop.
        """
        ...
