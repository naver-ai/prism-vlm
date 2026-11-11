"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Minimal .env loader (no python-dotenv dependency).

Only parses simple ``KEY=VALUE`` / ``KEY="VALUE"`` lines; quoted values keep
their interior whitespace. Comments (``#``) and blank lines are skipped.
Values that are already present in ``os.environ`` are never overridden, so
the shell-level export wins over ``.env``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

_LINE_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def parse_dotenv(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    if not path.exists():
        return result
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Allow `export KEY=VAL` for shell compatibility.
        if line.startswith("export "):
            line = line[len("export "):]
        match = _LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        # Strip inline comments (only if value is unquoted)
        if value and value[0] not in {'"', "'"}:
            hash_idx = value.find(" #")
            if hash_idx != -1:
                value = value[:hash_idx].strip()
        result[key] = _strip_quotes(value)
    return result


def load_dotenv(search_paths: list[Path]) -> list[Path]:
    """Populate ``os.environ`` from the first ``.env`` found in ``search_paths``.

    Returns the list of loaded paths (usually 0 or 1). Never overrides a key
    already present in ``os.environ``; multiple files can be layered by the
    caller if needed.
    """
    loaded: list[Path] = []
    for candidate in search_paths:
        if not candidate.exists():
            continue
        parsed = parse_dotenv(candidate)
        for key, value in parsed.items():
            os.environ.setdefault(key, value)
        loaded.append(candidate)
        break
    return loaded


def find_project_dotenv(start: Path) -> list[Path]:
    """Default search order: ``start/.env``, then parents up to 3 levels."""
    start = start.resolve()
    candidates = [start / ".env"]
    parent = start
    for _ in range(3):
        parent = parent.parent
        candidates.append(parent / ".env")
    return candidates
