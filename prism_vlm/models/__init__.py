"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Model-facing primitives: the shared message shape and the provider router.

``build_multimodal_messages`` is also used by the generation judge; the router
and the per-provider clients serve the evaluation stage.
"""

from .base import GenConfig, RawResponse, VLMClient, build_multimodal_messages
from .router import get_client

__all__ = [
    "GenConfig",
    "RawResponse",
    "VLMClient",
    "build_multimodal_messages",
    "get_client",
]
