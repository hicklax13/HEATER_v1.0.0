"""Tier -> model routing (admin-overridable) + a verified June-2026 price table.

Model strings are LiteLLM provider-prefixed ("anthropic/claude-...", "ollama/...").
The price table is the cost-cap fallback for when litellm.completion_cost() can't
price a newer model (it returns 0 for models outside its bundled map).
"""

from __future__ import annotations

import json

from src.app_settings import get_setting, set_setting

_TIER_MODELS_SETTING = "ai_tier_models"

_DEFAULT_TIER_MODELS = {
    "simple": "anthropic/claude-haiku-4-5",
    "moderate": "anthropic/claude-sonnet-4-6",
    "complex": "anthropic/claude-opus-4-8",
}

# USD per token (input, output). Verified June 2026. ollama/* is local = free.
_PRICE_PER_TOKEN = {
    "anthropic/claude-haiku-4-5": (1e-6, 5e-6),
    "anthropic/claude-sonnet-4-6": (3e-6, 15e-6),
    "anthropic/claude-opus-4-8": (5e-6, 25e-6),
    "anthropic/claude-fable-5": (10e-6, 50e-6),
}


def _overrides() -> dict:
    raw = get_setting(_TIER_MODELS_SETTING)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    # Guard against valid-but-non-dict JSON (corrupt/manual write): a list would
    # crash model_for_tier's {**defaults, **_overrides()} unpack on the hot path.
    return parsed if isinstance(parsed, dict) else {}


def model_for_tier(tier: str) -> str:
    merged = {**_DEFAULT_TIER_MODELS, **_overrides()}
    return merged.get(tier, _DEFAULT_TIER_MODELS["moderate"])


def set_tier_models(mapping: dict, admin_id: int) -> None:
    """Merge admin overrides into the tier->model map (MULTI_USER-gated via set_setting)."""
    current = _overrides()
    current.update({k: v for k, v in mapping.items() if v})
    set_setting(_TIER_MODELS_SETTING, json.dumps(current), admin_id)


def provider_of(model: str) -> str:
    """LiteLLM provider prefix; bare strings default to openai (litellm convention)."""
    return model.split("/", 1)[0] if "/" in model else "openai"


def price_per_token(model: str) -> tuple[float, float]:
    if model.startswith("ollama/"):
        return (0.0, 0.0)
    return _PRICE_PER_TOKEN.get(model, (0.0, 0.0))
