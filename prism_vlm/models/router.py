"""PRISM-VLM
Copyright (c) 2026-present NAVER Cloud Corp.
Apache-2.0

model_id → VLMClient dispatch.
"""

from __future__ import annotations

from ..config.loader import ModelsConfig, ModelSpec
from .base import VLMClient

# provider → (module, class, pip extra needed to import it)
_PROVIDERS = {
    "openai": ("openai_client", "OpenAIClient", None),
    "openrouter": ("openrouter_client", "OpenRouterClient", None),
    "vllm": ("vllm_client", "VLLMClient", None),
    "anthropic": ("anthropic_client", "AnthropicClient", "anthropic"),
    "google": ("google_client", "GoogleClient", "google"),
}


def get_client(model_id: str, models_cfg: ModelsConfig) -> VLMClient:
    return _build(models_cfg.by_id(model_id))


def _build(spec: ModelSpec) -> VLMClient:
    try:
        module, cls_name, extra = _PROVIDERS[spec.provider]
    except KeyError:
        raise ValueError(
            f"Unknown provider {spec.provider!r} for model {spec.id!r}; "
            f"expected one of {', '.join(sorted(_PROVIDERS))}"
        ) from None

    import importlib

    try:
        mod = importlib.import_module(f".{module}", __package__)
    except ImportError as e:
        if extra:
            raise ImportError(
                f"Model {spec.id!r} needs the optional {extra!r} provider: "
                f'pip install "prism-vlm[{extra}]"'
            ) from e
        raise

    return getattr(mod, cls_name)(spec)  # type: ignore[no-any-return]
