"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

YAML → frozen pydantic configuration models.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from .defaults import DEFAULTS


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


_PLAIN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def plain_name(value: str) -> str:
    if not _PLAIN_NAME.match(value):
        raise ValueError(
            "must be a plain name of letters, digits, dot, underscore or "
            f"hyphen, not {value!r}"
        )
    return value


#: A name the cache tree turns into a directory. Run, model, domain and
#: benchmark names are joined into paths such as
#: ``cache/runs/<run_id>/scores/<model_id>/<domain>/``, so a separator or a
#: leading dot in one of them would write outside the run directory.
PathName = Annotated[str, AfterValidator(plain_name)]


class JudgeConfig(_Frozen):
    model: str = "gpt-5-mini"
    temperature: float = 0.0
    effort: str | None = "minimal"
    # Optional fields needed when the judge is not OpenAI:
    # - provider: "openai" (default), "openrouter", "google". Auto-detected
    #   from model name when unset (gpt-*/o* → openai; */* → openrouter;
    #   gemini-* / gemma-* → google).
    provider: str | None = None
    # Anthropic-via-OpenRouter and Google thinking budget control.
    thinking: str | None = None       # "off" | "low" | "medium" | "high"
    max_tokens: int | None = None     # output budget; defaults to 4096


class BenchmarkSpec(_Frozen):
    """A benchmark is named by its id and nothing else.

    ``prism_vlm.data.hf_loaders`` owns the HuggingFace path, split, subset and
    per-benchmark preprocessing for each id. The id is also hashed into every
    item id, so it is a cache key: renaming one rebuilds the whole pool.
    """

    id: PathName


class DomainSpec(_Frozen):
    name: str
    benchmarks: list[BenchmarkSpec]


class SamplingSpec(_Frozen):
    items_per_benchmark: int = 100


class DomainsConfig(_Frozen):
    domains: dict[PathName, list[BenchmarkSpec]]
    sampling: SamplingSpec

    @property
    def all_benchmarks(self) -> list[BenchmarkSpec]:
        out: list[BenchmarkSpec] = []
        for bs in self.domains.values():
            out.extend(bs)
        return out


class ModelSpec(_Frozen):
    id: PathName
    provider: str
    model: str
    effort: str | None = None
    # Anthropic thinking. The two API shapes do not overlap, so say which one:
    # ``reasoning`` is an effort level for adaptive thinking (Claude 4.6+),
    # ``reasoning_budget`` a manual token budget (Claude 4.5 and earlier).
    reasoning: str | None = None
    reasoning_budget: int | None = None
    thinking: str | None = None
    endpoint: str | None = None
    # Per-model concurrency override — wins over concurrency_by_provider and
    # the top-level ``concurrency`` default. Useful for a slow reasoning model
    # in an otherwise fast provider group.
    concurrency: int | None = None
    # Per-model max output tokens override — needed for high-effort reasoning
    # models where reasoning tokens consume the budget before the answer, and
    # for an Anthropic ``reasoning_budget`` at or above the default budget.
    max_tokens: int | None = None


class ModelsConfig(_Frozen):
    models: list[ModelSpec]

    def by_id(self, model_id: str) -> ModelSpec:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(f"model_id not in registry: {model_id}")


class RunConfig(_Frozen):
    run_id: PathName
    R: int = DEFAULTS["R"]
    # Default concurrency used when neither model nor provider specifies one.
    concurrency: int = DEFAULTS["concurrency"]
    # Provider → concurrency overrides (e.g. openai: 32, google: 4, anthropic: 8).
    # Applies when a model doesn't set its own ``concurrency``.
    concurrency_by_provider: dict[str, int] = Field(default_factory=dict)
    # Concurrency for judge calls (shared across all models in a run). Defaults
    # to ``concurrency`` when unset.
    judge_concurrency: int | None = None
    # The evaluee pool and the analysis output directory belong to the
    # evaluation and scoring stages; a build-only config may omit both.
    domains_path: Path
    models_path: Path | None = None
    judge: JudgeConfig = JudgeConfig()
    variant_bundle: Path
    cache_dir: Path = Path(DEFAULTS["cache_dir"])
    cache_root: Path = Path(DEFAULTS["cache_root"])
    output_dir: Path | None = None

    @property
    def cache_run_dir(self) -> Path:
        return self.cache_dir / self.run_id

    def resolve_concurrency(self, model: ModelSpec) -> int:
        """Pick the effective concurrency for ``model``: per-model > per-provider > default."""
        if model.concurrency is not None:
            return model.concurrency
        return self.concurrency_by_provider.get(model.provider, self.concurrency)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _resolve(path_str: str, base: Path) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (base / p).resolve()


def load_domains(path: Path) -> DomainsConfig:
    raw = _load_yaml(path)
    domains = {
        name: [BenchmarkSpec(id=bid) for bid in ids]
        for name, ids in raw["domains"].items()
    }
    sampling = SamplingSpec(**raw.get("sampling", {}))
    return DomainsConfig(domains=domains, sampling=sampling)


def load_models(path: Path) -> ModelsConfig:
    """Load the model registry, filling vLLM endpoints from local config.

    Endpoints are deployment-specific, so they live outside the registry: an
    ``endpoints.yaml`` beside ``path`` maps model id to base URL, and
    ``PRISM_VLLM_ENDPOINT`` serves as a single-server fallback.
    """
    raw = _load_yaml(path)
    endpoints = _load_endpoints(path.parent / "endpoints.yaml")
    fallback = os.environ.get("PRISM_VLLM_ENDPOINT")

    models = []
    for m in raw["models"]:
        if m.get("provider") == "vllm" and not m.get("endpoint"):
            resolved = endpoints.get(m["id"]) or fallback
            if resolved:
                m = {**m, "endpoint": resolved}
        models.append(ModelSpec(**m))
    return ModelsConfig(models=models)


def _load_endpoints(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    raw = _load_yaml(path)
    return {str(k): str(v) for k, v in (raw.get("endpoints") or {}).items()}


def load_run_config(config_path: str | Path) -> RunConfig:
    """Parse a run YAML (e.g. ``configs/main_run.yaml``) into a frozen ``RunConfig``.

    Relative ``domains``, ``models``, and ``variant_bundle`` paths are
    resolved against the directory containing ``config_path``.
    """
    config_path = Path(config_path).resolve()
    base = config_path.parent
    raw = _load_yaml(config_path)

    judge_raw = raw.get("judge", {})
    judge = JudgeConfig(model=judge_raw.get("model", DEFAULTS["judge"]["model"]),
                        temperature=judge_raw.get("temperature",
                                                  DEFAULTS["judge"]["temperature"]),
                        effort=judge_raw.get("effort",
                                             DEFAULTS["judge"].get("effort", "minimal")),
                        provider=judge_raw.get("provider"),
                        thinking=judge_raw.get("thinking"),
                        max_tokens=judge_raw.get("max_tokens"))

    domains_path = _resolve(raw["domains"], base)
    models_path = _resolve(raw["models"], base) if raw.get("models") else None
    variant_bundle = _resolve(raw["variant_bundle"], base)
    cache_dir = _resolve(raw.get("cache_dir", DEFAULTS["cache_dir"]), base)
    cache_root = _resolve(raw.get("cache_root", DEFAULTS["cache_root"]), base)
    output_dir = _resolve(raw["output_dir"], base) if raw.get("output_dir") else None

    return RunConfig(
        run_id=raw["run_id"],
        R=raw.get("R", DEFAULTS["R"]),
        concurrency=raw.get("concurrency", DEFAULTS["concurrency"]),
        concurrency_by_provider=raw.get("concurrency_by_provider", {}),
        judge_concurrency=raw.get("judge_concurrency"),
        domains_path=domains_path,
        models_path=models_path,
        judge=judge,
        variant_bundle=variant_bundle,
        cache_dir=cache_dir,
        cache_root=cache_root,
        output_dir=output_dir,
    )
