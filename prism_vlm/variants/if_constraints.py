"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

IF axis: deterministic 1–3 constraints drawn from a fixed 11-item pool.

The constraint set is seeded by ``item_id`` so every re-run picks the same
constraints for the same item. The prompt suffix is built here and appended
to the original question; grading happens in ``evaluation.judge.grade_if``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..data.schemas import Item, Variant
from ..utils.hashing import deterministic_rng, variant_id_for

_POOL = (
    # original 5 (formal text constraints)
    "word_count",
    "terminal_punct",
    "casing",
    "required_keyword",
    "output_format",
    # new formatting-focused constraints — judge-graded
    "boxed_answer",
    "letter_choice_only",
    "markdown_bold",
    "prefix_phrase",
    "code_block",
    "no_explanation",
)

_WORD_COUNT_LIMITS = (5, 10, 20)
_PUNCTUATION = (".", "?", "!")
_CASING = ("lower", "upper", "title")
_FORMATS = ("json_answer_key", "numbered_list_max_3")
_PREFIX_PHRASES = ("The answer is", "My answer:", "Final answer:")
_CODE_LANGS = ("text", "json", "")  # "" → bare ``` fence


# Tiny per-benchmark keyword pool (hand-authored).
_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mmstar":      ("answer", "image", "because", "therefore", "overall"),
    "gqa":         ("answer", "yes", "no", "because", "object"),
    "okvqa":       ("because", "therefore", "appears", "likely", "item"),
    "mmmu":        ("therefore", "thus", "hence", "because", "result"),
    "mathvista":   ("therefore", "equals", "value", "so", "hence"),
    "scienceqa":   ("because", "therefore", "hence", "reason", "result"),
    "ai2d":        ("diagram", "label", "arrow", "because", "shows"),
    "chartqa":     ("value", "chart", "therefore", "percent", "answer"),
    "docvqa":      ("document", "answer", "value", "table", "line"),
    "refcocog":    ("region", "box", "letter", "answer", "choice"),
    "refcocoplus": ("region", "box", "letter", "answer", "choice"),
    "tallyqa":     ("count", "number", "total", "therefore", "answer"),
    "mantis":      ("image", "comparison", "therefore", "because", "answer"),
    "mmt_bench":   ("image", "answer", "therefore", "because", "choice"),
    "blink":       ("image", "answer", "therefore", "because", "choice"),
}


@dataclass(frozen=True, slots=True)
class Constraint:
    name: str
    params: dict[str, Any]


def sample_constraints(item: Item) -> tuple[Constraint, ...]:
    """Draw 1–3 constraints for ``item`` deterministically.

    Seeded by the item id alone, so an item carries the same constraints
    under every reach-K seed.
    """
    rng = deterministic_rng(item.item_id, "IF")
    n = rng.randint(1, 3)
    pool = list(_POOL)
    rng.shuffle(pool)
    picks = pool[:n]
    out: list[Constraint] = []
    for name in picks:
        if name == "word_count":
            out.append(Constraint(name, {"limit": rng.choice(_WORD_COUNT_LIMITS)}))
        elif name == "terminal_punct":
            out.append(Constraint(name, {"punct": rng.choice(_PUNCTUATION)}))
        elif name == "casing":
            out.append(Constraint(name, {"case": rng.choice(_CASING)}))
        elif name == "required_keyword":
            kws = _KEYWORDS.get(item.benchmark, ("answer",))
            out.append(Constraint(name, {"keyword": rng.choice(list(kws))}))
        elif name == "output_format":
            out.append(Constraint(name, {"format": rng.choice(_FORMATS)}))
        elif name == "boxed_answer":
            out.append(Constraint(name, {}))
        elif name == "letter_choice_only":
            out.append(Constraint(name, {}))
        elif name == "markdown_bold":
            out.append(Constraint(name, {}))
        elif name == "prefix_phrase":
            out.append(Constraint(name, {"phrase": rng.choice(_PREFIX_PHRASES)}))
        elif name == "code_block":
            out.append(Constraint(name, {"lang": rng.choice(_CODE_LANGS)}))
        elif name == "no_explanation":
            out.append(Constraint(name, {}))
    return tuple(out)


def format_constraints(constraints: tuple[Constraint, ...]) -> str:
    lines = ["", "You must also satisfy ALL of the following constraints:"]
    for idx, c in enumerate(constraints, start=1):
        if c.name == "word_count":
            lines.append(f"  {idx}. Your full answer must be at most {c.params['limit']} words.")
        elif c.name == "terminal_punct":
            lines.append(f"  {idx}. Your answer must end with '{c.params['punct']}'.")
        elif c.name == "casing":
            case = c.params["case"]
            label = {"lower": "all lowercase", "upper": "ALL UPPERCASE", "title": "Title Case"}[case]
            lines.append(f"  {idx}. Your answer must be written in {label}.")
        elif c.name == "required_keyword":
            lines.append(f"  {idx}. Your answer must contain the word '{c.params['keyword']}'.")
        elif c.name == "output_format":
            fmt = c.params["format"]
            desc = {
                "json_answer_key": 'valid JSON with a single "answer" key, e.g. {"answer": "..."}',
                "numbered_list_max_3": "a numbered list with at most 3 items (1. ... 2. ... 3. ...)",
            }[fmt]
            lines.append(f"  {idx}. Your response format must be: {desc}.")
        elif c.name == "boxed_answer":
            lines.append(f"  {idx}. Wrap your final answer inside \\boxed{{...}}.")
        elif c.name == "letter_choice_only":
            lines.append(f"  {idx}. Respond with a single uppercase letter (A, B, C, or D) and nothing else.")
        elif c.name == "markdown_bold":
            lines.append(f"  {idx}. Format your final answer in **bold markdown** (surrounded by double asterisks).")
        elif c.name == "prefix_phrase":
            lines.append(f"  {idx}. Begin your response with the exact phrase: {c.params['phrase']!r}.")
        elif c.name == "code_block":
            lang = c.params.get("lang", "")
            fence = f"```{lang}" if lang else "```"
            lines.append(f"  {idx}. Wrap your entire answer in a single fenced code block: {fence} ... ```.")
        elif c.name == "no_explanation":
            lines.append(f"  {idx}. Provide ONLY the final answer, with no explanation, reasoning, or preamble.")
    return "\n".join(lines)


class IFConstraints:
    axis: str = "IF"

    def generate(self, item: Item, bundle_dir: Path) -> list[Variant]:
        constraints = sample_constraints(item)
        suffix = format_constraints(constraints)
        new_q = item.question + "\n" + suffix
        return [
            Variant(
                variant_id=variant_id_for(item.item_id, "IF", 0),
                item_id=item.item_id,
                axis="IF",
                slot=0,
                question=new_q,
                images=item.images,
                meta={
                    "constraints": [
                        {"name": c.name, "params": c.params} for c in constraints
                    ]
                },
            )
        ]
