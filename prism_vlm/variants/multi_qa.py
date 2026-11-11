"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

mq axis (Multiple QA): 3-5 questions on same image, single prompt.

Bundle layout (`bundle_dir/multi_qa/<item_id>.json`):
    {
      "item_id": "...",
      "questions": ["<orig>", "<extra1>", "<extra2>", ...],   # 3-5 entries
      "golds":     ["<gold0>", "<gold1>", ...],               # parallel
    }

Position 0 is the original benchmark question/gold; positions 1+ are judge-
synthesised at bundle build time. The variant prompt concatenates them as a
numbered list; judge grades each answer independently via `grade_multi_qa`.
"""

from __future__ import annotations

from pathlib import Path

from ..data.schemas import Item, Variant
from ..utils.hashing import variant_id_for
from ..utils.io import read_json


def format_multi_qa_prompt(questions: list[str]) -> str:
    """Render N questions as a single prompt that asks the model to answer all."""
    lines = [
        "Please answer ALL of the following questions about the image. "
        "Number each answer to match the question number."
    ]
    for idx, q in enumerate(questions, start=1):
        lines.append(f"\n{idx}. {q}")
    return "\n".join(lines)


class MultiQA:
    axis: str = "mq"

    def generate(self, item: Item, bundle_dir: Path) -> list[Variant]:
        path = bundle_dir / "multi_qa" / f"{item.item_id}.json"
        if not path.exists():
            return []
        payload = read_json(path)
        questions = payload.get("questions", [])
        if len(questions) < 3:
            return []
        prompt = format_multi_qa_prompt(questions)
        return [
            Variant(
                variant_id=variant_id_for(item.item_id, "mq", 0),
                item_id=item.item_id,
                axis="mq",
                slot=0,
                question=prompt,
                images=item.images,
                meta={
                    "questions": questions,
                    "golds": payload.get("golds", []),
                    "n": len(questions),
                },
            )
        ]
