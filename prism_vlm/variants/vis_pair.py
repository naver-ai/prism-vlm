"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

vis axis (Vision discrimination): true-vs-fake caption forced choice.

Bundle layout (`bundle_dir/vis/<item_id>.json`):
    {
      "item_id": "...",
      "true_caption":  "<accurate detailed description>",
      "fake_caption":  "<true caption with one image-only detail swapped>",
      "swapped_detail": {"original": "...", "fake": "..."}  # audit metadata
    }

Which caption is shown as A is not stored: ``shuffle_letter`` derives it from
the item id at evaluation time, so the bundle stays independent of it.

The two captions are designed to be equally plausible from text alone — a
model that doesn't actually look at the image cannot distinguish them. The
forced-choice answer (A or B) tests pure visual discrimination. Random
baseline = 0.5; higher = stronger vision use.
"""

from __future__ import annotations

from pathlib import Path

from ..data.schemas import Item, Variant
from ..utils.hashing import deterministic_rng, variant_id_for
from ..utils.io import read_json


def format_vis_prompt(caption_a: str, caption_b: str) -> str:
    return (
        "Below are two descriptions of the image. Exactly ONE is fully accurate; "
        "the other has a single detail changed.\n\n"
        f"A. {caption_a}\n"
        f"B. {caption_b}\n\n"
        "Look at the image and answer with just the letter (A or B) of the accurate description. "
        "Do not explain."
    )


def shuffle_letter(item_id: str, true_caption: str, fake_caption: str) -> tuple[str, str, str]:
    """Deterministic A/B shuffle by item_id. Returns (caption_a, caption_b, letter_correct)."""
    rng = deterministic_rng(item_id, "vis")
    if rng.random() < 0.5:
        return true_caption, fake_caption, "A"
    return fake_caption, true_caption, "B"


class VisPair:
    axis: str = "vis"

    def generate(self, item: Item, bundle_dir: Path) -> list[Variant]:
        path = bundle_dir / "vis" / f"{item.item_id}.json"
        if not path.exists():
            return []
        payload = read_json(path)
        true_cap = payload.get("true_caption")
        fake_cap = payload.get("fake_caption")
        if not true_cap or not fake_cap:
            return []
        cap_a, cap_b, letter = shuffle_letter(item.item_id, true_cap, fake_cap)
        prompt = format_vis_prompt(cap_a, cap_b)
        return [
            Variant(
                variant_id=variant_id_for(item.item_id, "vis", 0),
                item_id=item.item_id,
                axis="vis",
                slot=0,
                question=prompt,
                images=item.images,
                meta={
                    "letter_correct": letter,
                    "swapped_detail": payload.get("swapped_detail", {}),
                },
            )
        ]
