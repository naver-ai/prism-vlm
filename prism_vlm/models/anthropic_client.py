"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Anthropic async client (Claude Haiku / Opus).
"""

from __future__ import annotations

import base64
import io
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config.loader import ModelSpec
from .base import GenConfig, RawResponse

_ANTHROPIC_IMG_LIMIT = 5 * 1024 * 1024  # 5 MB hard limit on Anthropic API


def _shrink_to_limit(img_bytes: bytes) -> tuple[bytes, str]:
    """Re-encode oversized images as JPEG until under Anthropic's 5MB limit."""
    from PIL import Image

    img = Image.open(io.BytesIO(img_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    quality = 90
    scale = 1.0
    while True:
        w, h = int(img.width * scale), int(img.height * scale)
        resized = img.resize((max(1, w), max(1, h)), Image.LANCZOS) if scale < 1.0 else img
        buf = io.BytesIO()
        resized.save(buf, format="JPEG", quality=quality, optimize=True)
        out = buf.getvalue()
        if len(out) <= _ANTHROPIC_IMG_LIMIT:
            return out, "image/jpeg"
        if quality > 60:
            quality -= 10
        else:
            scale *= 0.85
            quality = 80


def _openai_messages_to_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert the OpenAI multimodal format to the Anthropic one."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        parts: list[dict[str, Any]] = []
        content = msg["content"]
        if isinstance(content, str):
            parts.append({"type": "text", "text": content})
        else:
            for part in content:
                if part["type"] == "text":
                    parts.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    # data:image/png;base64,XXXX
                    header, _, b64 = url.partition(",")
                    media_type = header.split(";")[0].removeprefix("data:") or "image/png"
                    raw = base64.b64decode(b64)
                    if len(raw) > _ANTHROPIC_IMG_LIMIT:
                        shrunk, media_type = _shrink_to_limit(raw)
                        b64 = base64.b64encode(shrunk).decode("ascii")
                    parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        }
                    )
        out.append({"role": msg["role"], "content": parts})
    return out


class AnthropicClient:
    def __init__(self, spec: ModelSpec) -> None:
        from anthropic import AsyncAnthropic

        self.model_id = spec.id
        self._model = spec.model
        self._reasoning = spec.reasoning
        self._reasoning_budget = spec.reasoning_budget
        self._max_tokens = spec.max_tokens
        self._client = AsyncAnthropic()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
    )
    async def generate(self, messages: list[dict[str, Any]], config: GenConfig) -> RawResponse:
        anth_messages = _openai_messages_to_anthropic(messages)
        # No temperature for any Claude model. Claude 4.7 and later reject a
        # non-default value, thinking on or off, and a name-based version gate
        # would 400 every call the moment it guessed wrong -- so all Anthropic
        # calls run at the model's own default.
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": anth_messages,
            "max_tokens": self._max_tokens or config.max_tokens,
        }
        reasoning = config.reasoning or self._reasoning
        if self._reasoning_budget:
            # Manual extended thinking. Claude 4.7 and later reject this form.
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self._reasoning_budget}
        elif reasoning and reasoning != "off":
            # Adaptive thinking. Claude 4.5 and earlier reject this form.
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": reasoning}

        resp = await self._client.messages.create(**kwargs)
        text_parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        text = "".join(text_parts)
        usage = {
            "input_tokens": getattr(resp.usage, "input_tokens", 0),
            "output_tokens": getattr(resp.usage, "output_tokens", 0),
        }
        return RawResponse(
            text=text,
            finish_reason=resp.stop_reason or "stop",
            usage=usage,
            raw=resp.model_dump(),
        )
