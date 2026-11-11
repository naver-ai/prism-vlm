"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

File-per-item cache tree for model calls, judge verdicts and scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CacheLayout:
    run_root: Path

    @property
    def config_sha_path(self) -> Path:
        return self.run_root / "config.sha256"

    def calls_dir(self, model_id: str) -> Path:
        return self.run_root / "calls" / model_id

    def call_path(self, model_id: str, item_id: str, axis: str, slot: int) -> Path:
        """Per-(item, axis, slot) call cache.

        Deliberately not keyed by seed: model calls are seed-invariant by
        design — seeds only affect reach-K ordering and the item-id-seeded IF
        constraints — so one call is cached and reused across the R seeds.
        """
        suffix = f"{axis}" if slot == 0 else f"{axis}_{slot}"
        return self.calls_dir(model_id) / item_id / f"{suffix}.json"

    def score_path(self, model_id: str, domain: str, item_id: str) -> Path:
        return self.run_root / "scores" / model_id / domain / f"{item_id}.json"

    def judge_path(self, model_id: str, item_id: str, axis: str) -> Path:
        return self.run_root / "judge" / model_id / f"{item_id}__{axis}.json"

    def ensure(self) -> None:
        self.run_root.mkdir(parents=True, exist_ok=True)
        (self.run_root / "calls").mkdir(exist_ok=True)
        (self.run_root / "scores").mkdir(exist_ok=True)
        (self.run_root / "judge").mkdir(exist_ok=True)
