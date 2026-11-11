"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

vLLM async client (OpenAI-compatible, local fallback).
"""

from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config.loader import ModelSpec
from .base import GenConfig, RawResponse


class VLLMClient:
    def __init__(self, spec: ModelSpec) -> None:
        if not spec.endpoint:
            raise ValueError(
                f"No endpoint for vLLM model {spec.id!r}. Set it in "
                f"configs/endpoints.yaml, or point PRISM_VLLM_ENDPOINT at your server."
            )
        self.model_id = spec.id
        self._model = spec.model
        self._thinking = spec.thinking
        self._max_tokens = spec.max_tokens
        self._client = AsyncOpenAI(api_key="EMPTY", base_url=spec.endpoint)

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
        thinking = config.thinking or self._thinking
        if thinking == "off":
            kwargs["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": False},
            }
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
