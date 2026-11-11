"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Benchmark loader registry: benchmark spec → concrete HF loader.
"""

from __future__ import annotations

from typing import Protocol

from ..config.loader import BenchmarkSpec
from .schemas import Item


class SourceNotProvidedError(RuntimeError):
    """The benchmark is not on the Hub and the user has not pointed us at a copy."""


class BenchmarkLoader(Protocol):
    def pool_size(self) -> int: ...
    def item_at(self, domain: str, orig_index: int) -> Item: ...


def get_loader(spec: BenchmarkSpec) -> BenchmarkLoader:
    """Return the concrete loader for ``spec``.

    The import is deferred so that loading ``hf_loaders`` (which imports
    ``datasets``) only happens after ``configure_cache_env`` has been called.
    """
    from . import hf_loaders

    return hf_loaders.build_loader(spec)
