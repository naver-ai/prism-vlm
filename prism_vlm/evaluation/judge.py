"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

Provider-aware judge: synthesises per-item artifacts at build time, grades at run time.

Generation prompts live in ``variants/prompts/``, grading prompts in
``judge_prompts/``; the two contexts are kept separate.

Routes :class:`JudgeConfig`.model to the right SDK:
- ``gpt-*`` / ``o*``                              → OpenAI direct
- ``<vendor>/<model>`` (contains ``/``)           → OpenRouter (OpenAI-compatible)
- ``gemini-*`` / ``gemma-*``                      → Google genai
Provider can also be forced via ``cfg.provider``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..config.loader import JudgeConfig
from ..models.base import build_multimodal_messages

_GEN_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "variants" / "prompts"
_JUDGE_PROMPTS_DIR = Path(__file__).resolve().parent / "judge_prompts"

_DEFAULT_RUBRIC = """You are grading a vision-language model's answer.

Question: {question}
Reference answer: {gold}
Model answer: {prediction}

Is the model answer semantically equivalent to the reference answer? Ignore
trivial formatting differences (punctuation, casing, list vs sentence).

Respond with strict JSON: {{"verdict": 0 | 1, "reason": "<one sentence>"}}.
"""


def _gen_prompt(name: str) -> str:
    return (_GEN_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def _judge_prompt(name: str) -> str:
    return (_JUDGE_PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")


def _detect_provider(model: str) -> str:
    """Infer provider when JudgeConfig.provider is unset."""
    m = model.lower()
    if m.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if "/" in m:
        return "openrouter"
    if m.startswith(("gemini-", "gemma-")):
        return "google"
    raise ValueError(f"Cannot infer judge provider from model id: {model!r}")


class JudgeClient:
    def __init__(self, cfg: JudgeConfig) -> None:
        self._cfg = cfg
        self._provider = (cfg.provider or _detect_provider(cfg.model)).lower()
        self._max_tokens = cfg.max_tokens or 4096

        if self._provider == "openai":
            self._client = AsyncOpenAI()
        elif self._provider == "openrouter":
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is not set")
            self._client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        elif self._provider == "google":
            from google import genai
            self._client = genai.Client()
        else:
            raise ValueError(f"Unsupported judge provider: {self._provider!r}")

    @retry(
        reraise=True,
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=16),
        retry=retry_if_exception_type(Exception),
    )
    async def _call(self, messages: list[dict[str, Any]]) -> str:
        if self._provider in ("openai", "openrouter"):
            return await self._call_openai_compat(messages)
        if self._provider == "google":
            return await self._call_google(messages)
        raise RuntimeError(f"unreachable: provider={self._provider!r}")

    async def _call_openai_compat(self, messages: list[dict[str, Any]]) -> str:
        # GPT-5 / o-series reasoning models (a) only support default temperature,
        # (b) burn reasoning tokens inside max_completion_tokens.
        is_reasoning = self._cfg.model.startswith(("gpt-5", "o1", "o3", "o4"))
        kwargs: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": messages,
            "max_completion_tokens": self._max_tokens,
        }
        if is_reasoning:
            kwargs["reasoning_effort"] = self._cfg.effort
        else:
            kwargs["temperature"] = self._cfg.temperature
        resp = await self._client.chat.completions.create(**kwargs)
        return (resp.choices[0].message.content or "").strip()

    async def _call_google(self, messages: list[dict[str, Any]]) -> str:
        # Convert OpenAI-format messages → genai contents.parts.
        import base64 as _b64
        parts: list[dict[str, Any]] = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                parts.append({"text": content})
                continue
            for part in content:
                if part["type"] == "text":
                    parts.append({"text": part["text"]})
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    header, _, b64 = url.partition(",")
                    mime = header.split(";")[0].removeprefix("data:") or "image/png"
                    parts.append({"inline_data": {"mime_type": mime, "data": _b64.b64decode(b64)}})
        cfg: dict[str, Any] = {
            "max_output_tokens": self._max_tokens,
            "temperature": self._cfg.temperature,
        }
        thinking = self._cfg.thinking
        if thinking == "off":
            cfg["thinking_config"] = {"include_thoughts": False, "thinking_budget": 0}
        elif thinking:
            level_map = {"low": 512, "medium": 2048, "high": 8192}
            cfg["thinking_config"] = {"include_thoughts": False,
                                      "thinking_budget": level_map.get(thinking, 512)}

        def _sync_call() -> Any:
            return self._client.models.generate_content(
                model=self._cfg.model,
                contents=[{"role": "user", "parts": parts}],
                config=cfg,
            )
        resp = await asyncio.to_thread(_sync_call)
        return (resp.text or "").strip()

    # ------------------- per-axis graders (run time) -------------------

    def _load_rubric(self, benchmark: str) -> str:
        """q rubric; a ``judge_prompts/<benchmark>.md`` file overrides the default."""
        path = _JUDGE_PROMPTS_DIR / f"{benchmark}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        return _DEFAULT_RUBRIC

    async def grade_q(
        self,
        benchmark: str,
        question: str,
        gold: str | tuple[str, ...],
        prediction: str,
    ) -> dict[str, Any]:
        """q: semantic equivalence to the recycled gold."""
        rubric = self._load_rubric(benchmark).format(
            question=question,
            gold=gold if isinstance(gold, str) else " | ".join(gold),
            prediction=prediction,
        )
        text = await self._call([{"role": "user", "content": rubric}])
        return _parse_verdict(text, default=0)

    async def grade_if(
        self,
        response: str,
        constraints_text: str,
        n_constraints: int,
    ) -> dict[str, Any]:
        """if: per-constraint pass/fail."""
        rubric = _judge_prompt("if").format(
            constraints=constraints_text, response=response
        )
        text = await self._call([{"role": "user", "content": rubric}])
        verdicts = _parse_json(text).get("verdicts", [])
        return {"verdicts": _clean_verdicts(verdicts, n_constraints)}

    async def grade_hallucination(
        self,
        sub_type: str,
        question: str,
        response: str,
        context: str = "",
    ) -> dict[str, Any]:
        """h: did the model fall for the trap?"""
        rubric = _judge_prompt("hallucination").format(
            sub_type=sub_type,
            context=context or "(none)",
            question=question,
            response=response,
        )
        text = await self._call([{"role": "user", "content": rubric}])
        parsed = _parse_json(text)
        return {
            "hallucinated": _as_bit(parsed.get("hallucinated"), default=1),
            "reason": parsed.get("reason", ""),
        }

    async def grade_multi_qa(
        self,
        response: str,
        questions: list[str],
        golds: list[str],
    ) -> dict[str, Any]:
        """mq: per-question correctness within one combined response."""
        n = len(questions)
        qa_block = "\n".join(
            f"{i + 1}. Q: {q}\n   gold: {g}"
            for i, (q, g) in enumerate(zip(questions, golds, strict=False))
        )
        rubric = _judge_prompt("multi_qa").format(
            n=n, qa_block=qa_block, response=response
        )
        text = await self._call([{"role": "user", "content": rubric}])
        verdicts = _parse_json(text).get("verdicts", [])
        return {"verdicts": _clean_verdicts(verdicts, n)}

    async def detect_sc_change(
        self,
        question: str,
        prev_response: str,
        current_response: str,
    ) -> dict[str, Any]:
        """sc: did the answer change between two pressure turns?"""
        rubric = _judge_prompt("sc_change").format(
            question=question, prev=prev_response, curr=current_response
        )
        text = await self._call([{"role": "user", "content": rubric}])
        parsed = _parse_json(text)
        return {
            "changed": _as_bit(parsed.get("changed"), default=1),
            "reason": parsed.get("reason", ""),
        }

    async def grade_counting(
        self,
        question: str,
        gold_count: int,
        response: str,
    ) -> dict[str, Any]:
        """c: extract the committed count and compare it to the gold."""
        rubric = _judge_prompt("counting").format(
            question=question, gold_count=gold_count, response=response
        )
        text = await self._call([{"role": "user", "content": rubric}])
        parsed = _parse_json(text)
        return {
            "verdict": _as_bit(parsed.get("verdict"), default=0),
            "extracted": parsed.get("extracted"),
            "reason": parsed.get("reason", ""),
        }

    async def grade_vis(self, response: str) -> dict[str, Any]:
        """vis: extract the A/B choice committed to, or X if neither."""
        rubric = _judge_prompt("vis_pair").format(response=response)
        text = await self._call([{"role": "user", "content": rubric}])
        choice = str(_parse_json(text).get("choice", "X")).strip().upper()
        return {"choice": choice if choice in ("A", "B") else "X"}

    # ------------- bundle-build generators (judge with image) -------------

    async def generate_h_pool_trap(
        self, image_bytes: bytes, sub_type: str
    ) -> dict[str, Any]:
        """Synthesise one h-axis trap question for the given sub_type."""
        rubric = _gen_prompt("h_pool").format(sub_type=sub_type)
        messages = build_multimodal_messages(rubric, (image_bytes,))
        text = await self._call(messages)
        return _parse_json(text)

    async def generate_multi_qa_extras(
        self, question: str, gold: str | tuple[str, ...], image_bytes: bytes
    ) -> list[dict[str, str]]:
        """Synthesise 2-4 extra (question, gold) pairs for the same image."""
        rubric = _gen_prompt("multi_qa").format(
            question=question,
            gold=gold if isinstance(gold, str) else " | ".join(gold),
        )
        messages = build_multimodal_messages(rubric, (image_bytes,))
        text = await self._call(messages)
        extras = _parse_json(text).get("extras", [])
        out: list[dict[str, str]] = []
        for e in extras:
            if not isinstance(e, dict):
                continue
            q = str(e.get("question", "")).strip()
            g = str(e.get("gold", "")).strip()
            if q and g:
                out.append({"question": q, "gold": g})
        return out[:4]

    async def generate_counting_question(self, image_bytes: bytes) -> dict[str, Any]:
        """Synthesise one counting question for the image; returns {} if not feasible."""
        messages = build_multimodal_messages(_gen_prompt("counting"), (image_bytes,))
        text = await self._call(messages)
        parsed = _parse_json(text)
        q = parsed.get("question")
        gc = parsed.get("gold_count")
        if not q or gc is None:
            return {}
        try:
            gc_int = int(gc)
        except (TypeError, ValueError):
            return {}
        if not (1 <= gc_int <= 100):
            return {}
        return {"question": str(q), "gold_count": gc_int, "object": str(parsed.get("object", ""))}

    async def generate_vision_pair(self, image_bytes: bytes) -> dict[str, Any]:
        """Synthesise a (true_caption, fake_caption) pair for forced-choice vis."""
        messages = build_multimodal_messages(_gen_prompt("vis_pair"), (image_bytes,))
        text = await self._call(messages)
        parsed = _parse_json(text)
        true_cap = str(parsed.get("true_caption", "")).strip()
        fake_cap = str(parsed.get("fake_caption", "")).strip()
        if not true_cap or not fake_cap or true_cap == fake_cap:
            return {}
        return {
            "true_caption": true_cap,
            "fake_caption": fake_cap,
            "swapped_detail": parsed.get("swapped_detail", {}),
        }

    async def generate_sc_wrong_answer(
        self, question: str, gold: str | tuple[str, ...], image_bytes: bytes
    ) -> dict[str, Any]:
        """Generate the plausible-but-wrong answer planted at the sc Q3 turn."""
        rubric = _gen_prompt("sc_wrong").format(
            question=question,
            gold=gold if isinstance(gold, str) else " | ".join(gold),
        )
        messages = build_multimodal_messages(rubric, (image_bytes,))
        text = await self._call(messages)
        wrong = str(_parse_json(text).get("wrong_answer", "")).strip()
        return {"wrong_answer": wrong} if wrong else {}


def _parse_json(text: str) -> dict[str, Any]:
    """Parse the judge's JSON, tolerating markdown fences; {} if unparseable."""
    match = re.search(r"\{.*\}", text.strip(), flags=re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _parse_verdict(text: str, default: int) -> dict[str, Any]:
    """_parse_json with an explicit fallback verdict."""
    parsed = _parse_json(text)
    if parsed:
        return parsed
    return {"verdict": default, "reason": f"unparseable judge output: {text.strip()[:120]}"}


def _as_bit(value: Any, default: int) -> int:
    return int(value) if value in (0, 1, "0", "1") else default


def _clean_verdicts(verdicts: Any, n: int) -> list[int]:
    """Pad or truncate to n; judges sometimes drop entries."""
    if not isinstance(verdicts, list):
        verdicts = []
    clean = [_as_bit(v, default=0) for v in verdicts][:n]
    clean.extend([0] * (n - len(clean)))
    return clean
