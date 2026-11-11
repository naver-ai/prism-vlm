"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Global async gates: one semaphore for model calls, one for the judge.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class ConcurrencyGates:
    model_gate: asyncio.Semaphore
    judge_gate: asyncio.Semaphore

    @classmethod
    def build(cls, n_model: int = 5, n_judge: int = 5) -> ConcurrencyGates:
        return cls(
            model_gate=asyncio.Semaphore(n_model),
            judge_gate=asyncio.Semaphore(n_judge),
        )
