"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

OpenRouter async client (OpenAI-compatible).
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config.loader import ModelSpec
from .base import GenConfig, RawResponse


class OpenRouterClient:
    def __init__(self, spec: ModelSpec) -> None:
        self.model_id = spec.id
        self._model = spec.model
        self._thinking = spec.thinking
        self._max_tokens = spec.max_tokens
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
    )
    async def generate(self, messages: list[dict[str, Any]], config: GenConfig) -> RawResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens or config.max_tokens,
            "temperature": config.temperature,
        }
        if config.stop:
            kwargs["stop"] = list(config.stop)
        if config.seed is not None:
            kwargs["seed"] = config.seed
        # OpenRouter normalises reasoning across upstreams via a top-level
        # `reasoning` field — `{"enabled": false}` disables thinking on
        # models that expose it (Qwen3, Anthropic-thinking, etc.). We use
        # that instead of the vLLM-only `chat_template_kwargs.enable_thinking`,
        # which OpenRouter does not forward.
        thinking = config.thinking or self._thinking
        if thinking == "off":
            kwargs["extra_body"] = {"reasoning": {"enabled": False}}
        kwargs.update(config.extra)

        resp = await self._client.chat.completions.create(**kwargs)
        # Defensive parse: OpenRouter occasionally returns responses with
        # `choices=None`, an empty list, or `message=None` — usually upstream
        # content-filter rejections or transient routing hiccups. Treat these
        # as empty-text responses with finish_reason="error" so the pipeline
        # records a real result instead of crashing the per-item task.
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return RawResponse(
                text="",
                finish_reason="error",
                usage={},
                raw=resp.model_dump() if hasattr(resp, "model_dump") else {},
            )
        choice = choices[0]
        message = getattr(choice, "message", None)
        text = (message.content or "") if message else ""
        finish = getattr(choice, "finish_reason", None) or "stop"
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return RawResponse(
            text=text,
            finish_reason=finish,
            usage=usage,
            raw=resp.model_dump(),
        )
