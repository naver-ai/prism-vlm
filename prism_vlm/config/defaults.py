"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Default values for RunConfig fields not set explicitly in YAML.
"""

from __future__ import annotations

from typing import Any

DEFAULTS: dict[str, Any] = {
    "R": 5,
    "concurrency": 5,
    "judge": {"model": "gpt-5", "temperature": 0.0, "effort": "low"},
    "cache_dir": "cache/runs/",
    "cache_root": "cache/hf/",
}
