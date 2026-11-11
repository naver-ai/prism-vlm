"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Google Gemini async client via google-genai.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config.loader import ModelSpec
from .base import GenConfig, RawResponse


class GoogleClient:
    def __init__(self, spec: ModelSpec) -> None:
        from google import genai

        self.model_id = spec.id
        self._model = spec.model
        self._thinking = spec.thinking
        self._max_tokens = spec.max_tokens
        self._client = genai.Client()

    def _build_contents(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style messages to Gemini ``contents``, one turn each.

        Roles must survive: the sc axis replays the model's own answers back to
        it, and flattening them into the user turn destroys the dialogue.
        """
        import base64 as _b64

        contents: list[dict[str, Any]] = []
        for msg in messages:
            parts: list[dict[str, Any]] = []
            content = msg["content"]
            if isinstance(content, str):
                parts.append({"text": content})
            else:
                for part in content:
                    if part["type"] == "text":
                        parts.append({"text": part["text"]})
                    elif part["type"] == "image_url":
                        url = part["image_url"]["url"]
                        header, _, b64 = url.partition(",")
                        mime = header.split(";")[0].removeprefix("data:") or "image/png"
                        parts.append(
                            {
                                "inline_data": {
                                    "mime_type": mime,
                                    "data": _b64.b64decode(b64),
                                }
                            }
                        )
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": parts})
        return contents

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
    )
    async def generate(self, messages: list[dict[str, Any]], config: GenConfig) -> RawResponse:
        contents = self._build_contents(messages)
        cfg: dict[str, Any] = {
            "max_output_tokens": self._max_tokens or config.max_tokens,
            "temperature": config.temperature,
        }
        thinking = config.thinking or self._thinking
        if thinking == "off":
            cfg["thinking_config"] = {"include_thoughts": False, "thinking_budget": 0}
        elif thinking:
            level_map = {"low": 512, "medium": 2048, "high": 8192}
            cfg["thinking_config"] = {
                "include_thoughts": False,
                "thinking_budget": level_map.get(thinking, 512),
            }

        def _sync_call() -> Any:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=cfg,
            )

        resp = await asyncio.to_thread(_sync_call)
        text = resp.text or ""
        usage_meta = getattr(resp, "usage_metadata", None)
        usage = {
            "prompt_token_count": getattr(usage_meta, "prompt_token_count", 0),
            "candidates_token_count": getattr(usage_meta, "candidates_token_count", 0),
            "thoughts_token_count": getattr(usage_meta, "thoughts_token_count", 0),
        }
        return RawResponse(
            text=text,
            finish_reason=getattr(resp, "finish_reason", "stop") or "stop",
            usage=usage,
            raw=resp.to_json_dict() if hasattr(resp, "to_json_dict") else {"text": text},
        )
