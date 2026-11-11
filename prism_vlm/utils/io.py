"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

File I/O, image helpers, and third-party cache configuration.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image


def configure_cache_env(cache_root: str | os.PathLike[str]) -> Path:
    """Point HuggingFace / transformers / torch caches at ``cache_root``.

    This MUST be called before any ``datasets`` / ``transformers`` /
    ``huggingface_hub`` import. The CLI calls it from ``cli.main`` as the first
    line of every subcommand.

    Precedence:
      1. ``HF_HOME`` already exported in the environment — fully respected.
      2. ``HF_CACHE_DIR`` (custom convention, e.g. from ``.env``) — used as
         the HF root instead of the project-local ``cache/hf/``.
      3. ``cache_root`` argument (the ``cache_root`` field from ``main_run.yaml``).

    Returns the resolved HF root path.
    """
    existing_hf_home = os.environ.get("HF_HOME")
    hf_cache_override = os.environ.get("HF_CACHE_DIR")

    if existing_hf_home:
        hf_root = Path(existing_hf_home).expanduser().resolve()
        root = hf_root.parent
    elif hf_cache_override:
        hf_root = Path(hf_cache_override).expanduser().resolve()
        root = hf_root.parent
    else:
        root = Path(cache_root).expanduser().resolve()
        hf_root = root / "huggingface"

    (hf_root / "datasets").mkdir(parents=True, exist_ok=True)
    (hf_root / "hub").mkdir(parents=True, exist_ok=True)
    (hf_root / "transformers").mkdir(parents=True, exist_ok=True)
    (root / "torch").mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(hf_root))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_root / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_root / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(root / "torch"))
    return hf_root


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomic JSON write: tmp-file + rename so readers never see a partial file.

    The tmp filename includes the PID so parallel writers (e.g. two
    ``prism-vlm`` processes sampling the same seed map) don't race on
    the same tmp handle. The final rename is atomic on POSIX — last writer
    wins, and since all concurrent writers produce the same deterministic
    content the result is consistent.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"), default=str)
    tmp.replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_image_bytes(source: bytes | str | Path | Image.Image) -> bytes:
    """Normalise various image inputs to raw PNG/JPEG bytes."""
    if isinstance(source, bytes):
        return source
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, Image.Image):
        buf = io.BytesIO()
        fmt = source.format or "PNG"
        source.save(buf, format=fmt)
        return buf.getvalue()
    raise TypeError(f"Unsupported image source type: {type(source)!r}")


def make_blank_image(reference_bytes: bytes) -> bytes:
    """Generate a same-sized RGB(0,0,0) PNG matching the reference image's shape."""
    with Image.open(io.BytesIO(reference_bytes)) as src:
        size = src.size
        mode = src.mode if src.mode in {"RGB", "RGBA", "L"} else "RGB"
    blank = Image.new(mode, size, color=0)
    buf = io.BytesIO()
    blank.save(buf, format="PNG")
    return buf.getvalue()
