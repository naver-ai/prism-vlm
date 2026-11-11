"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Item / Variant / Response dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    domain: str
    benchmark: str
    question: str
    images: tuple[bytes, ...]
    gold: str | tuple[str, ...]
    judge_gold: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Variant:
    variant_id: str
    item_id: str
    axis: str
    slot: int
    question: str
    images: tuple[bytes, ...]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Response:
    variant_id: str
    item_id: str
    axis: str
    slot: int
    seed: int
    model_id: str
    text: str
    finish_reason: str
    usage: dict[str, Any]
    raw: dict[str, Any]
