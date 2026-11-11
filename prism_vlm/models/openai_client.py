"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

OpenAI async client (GPT-5 family, reference model).
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config.loader import ModelSpec
from .base import GenConfig, RawResponse


class OpenAIClient:
    def __init__(self, spec: ModelSpec) -> None:
        self.model_id = spec.id
        self._model = spec.model
        self._effort = spec.effort
        self._max_tokens = spec.max_tokens
        self._client = AsyncOpenAI()

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
    )
    async def generate(self, messages: list[dict[str, Any]], config: GenConfig) -> RawResponse:
        # GPT-5 family / o-series reasoning models only support the default
        # temperature (=1). Omitting the param lets the SDK use the default.
        is_reasoning = self._model.startswith(("gpt-5", "o1", "o3", "o4"))
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_completion_tokens": self._max_tokens or config.max_tokens,
        }
        if not is_reasoning:
            kwargs["temperature"] = config.temperature
        if config.stop:
            kwargs["stop"] = list(config.stop)
        if self._effort or config.effort:
            kwargs["reasoning_effort"] = config.effort or self._effort
        if config.seed is not None:
            kwargs["seed"] = config.seed
        kwargs.update(config.extra)

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        text = (choice.message.content or "") if choice.message else ""
        finish = choice.finish_reason or "stop"
        usage = resp.usage.model_dump() if resp.usage else {}
        return RawResponse(
            text=text,
            finish_reason=finish,
            usage=usage,
            raw=resp.model_dump(),
        )
