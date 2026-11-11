"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Deterministic hashing and RNG helpers.
"""

from __future__ import annotations

import hashlib
import json
from random import Random
from typing import Any


def _sha1_short(payload: str, length: int = 16) -> str:
    return hashlib.sha1(payload.encode("utf-8"), usedforsecurity=False).hexdigest()[:length]


def item_id_for(benchmark: str, orig_index: int, question: str) -> str:
    """Stable 16-hex-char id for a base item."""
    return _sha1_short(f"{benchmark}|{orig_index}|{question}")


def variant_id_for(item_id: str, axis: str, slot: int = 0) -> str:
    """Stable id for a per-item variant (axis-scoped, slot for multi-variant axes)."""
    return _sha1_short(f"{item_id}|{axis}|{slot}")


def deterministic_rng(*parts: str | int) -> Random:
    """Build a Random seeded from a stable join of the given parts.

    Using str(parts) means the caller controls what goes into the hash; we do
    not rely on Python's hash randomisation.
    """
    joined = "|".join(str(p) for p in parts)
    seed_int = int(hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16], 16)
    return Random(seed_int)


def config_sha256(payload: dict[str, Any]) -> str:
    """Canonical JSON hash of a resolved config (stable across runs)."""
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()
