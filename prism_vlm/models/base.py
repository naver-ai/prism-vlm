"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

VLMClient protocol and shared request/response types.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GenConfig:
    max_tokens: int = 2048
    stop: tuple[str, ...] = ()
    seed: int | None = None
    effort: str | None = None            # OpenAI: minimal/low/medium/high
    reasoning: str | None = None         # Anthropic: effort for adaptive thinking
    thinking: str | None = None          # Google: off/low/medium/high; OpenRouter, vLLM: off
    temperature: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RawResponse:
    text: str
    finish_reason: str
    usage: dict[str, Any]
    raw: dict[str, Any]


class VLMClient(Protocol):
    model_id: str

    async def generate(
        self,
        messages: list[dict[str, Any]],
        config: GenConfig,
    ) -> RawResponse: ...


def encode_image_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def sniff_image_mime(image_bytes: bytes) -> str:
    """Detect MIME type from magic bytes. Anthropic rejects mismatched media_type."""
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def build_multimodal_messages(question: str, images: tuple[bytes, ...]) -> list[dict[str, Any]]:
    """Return OpenAI-style multimodal ``messages``: image(s) first, then text."""
    content: list[dict[str, Any]] = []
    for img_bytes in images:
        b64 = encode_image_b64(img_bytes)
        mime = sniff_image_mime(img_bytes)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]
