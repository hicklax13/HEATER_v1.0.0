"""Tier -> model routing (admin-overridable) + a verified June-2026 price table.

Model strings are LiteLLM provider-prefixed ("anthropic/claude-...", "ollama/...").
The price table is the cost-cap fallback for when litellm.completion_cost() can't
price a newer model (it returns 0 for models outside its bundled map).
"""

from __future__ import annotations

import json

from src.app_settings import get_setting, set_setting

_TIER_MODELS_SETTING = "ai_tier_models"

# Defaults point at DeepSeek V4 (the operator's configured shared provider) so the
# Simple/Moderate/Complex "Auto" tiers work out of the box with the DeepSeek shared
# key. Admins can repoint any tier to another provider in Admin Controls, and members
# can pick a specific model in the chat window's dropdown.
_DEFAULT_TIER_MODELS = {
    "simple": "deepseek/deepseek-v4-flash",
    "moderate": "deepseek/deepseek-v4-flash",
    "complex": "deepseek/deepseek-v4-pro",
}

# USD per token (input, output). Verified June 2026. ollama/* is local = free.
_PRICE_PER_TOKEN = {
    "anthropic/claude-haiku-4-5": (1e-6, 5e-6),
    "anthropic/claude-sonnet-4-6": (3e-6, 15e-6),
    "anthropic/claude-opus-4-8": (5e-6, 25e-6),
    "anthropic/claude-fable-5": (10e-6, 50e-6),
    # DeepSeek V4 (released 2026-04; replaced V3.2/R1). Both 1M ctx, tool-calling.
    "deepseek/deepseek-v4-flash": (0.14e-6, 0.28e-6),
    "deepseek/deepseek-v4-pro": (0.435e-6, 0.87e-6),
    # OpenAI GPT-5.x (June 2026).
    "openai/gpt-5.5": (5e-6, 30e-6),
    "openai/gpt-5.4": (2.5e-6, 15e-6),
    "openai/gpt-5.4-nano": (0.2e-6, 1.25e-6),
    # Google Gemini 3.x (June 2026). Pro Preview base tier (<=200k prompt).
    "gemini/gemini-3.1-pro-preview": (2e-6, 12e-6),
    "gemini/gemini-3.5-flash": (1.5e-6, 9e-6),
    "gemini/gemini-3.1-flash-lite": (0.25e-6, 1.5e-6),
}

# Models offered in the chat window's per-message picker. (label, model_string).
# Tier autos (Simple/Moderate/Complex) are added by the UI on top of these.
_MODEL_CATALOG = [
    ("DeepSeek V4 Flash", "deepseek/deepseek-v4-flash"),
    ("DeepSeek V4 Pro", "deepseek/deepseek-v4-pro"),
    ("Claude Haiku 4.5", "anthropic/claude-haiku-4-5"),
    ("Claude Sonnet 4.6", "anthropic/claude-sonnet-4-6"),
    ("Claude Opus 4.8", "anthropic/claude-opus-4-8"),
    ("GPT-5.5", "openai/gpt-5.5"),
    ("GPT-5.4", "openai/gpt-5.4"),
    ("GPT-5.4 Nano", "openai/gpt-5.4-nano"),
    ("Gemini 3.1 Pro", "gemini/gemini-3.1-pro-preview"),
    ("Gemini 3.5 Flash", "gemini/gemini-3.5-flash"),
    ("Gemini 3.1 Flash-Lite", "gemini/gemini-3.1-flash-lite"),
]


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


def model_catalog() -> list[tuple[str, str]]:
    """(label, model_string) pairs offered in the chat window's model picker."""
    return list(_MODEL_CATALOG)


def label_for_model(model: str) -> str:
    """Human label for a model string (falls back to the raw string)."""
    for label, m in _MODEL_CATALOG:
        if m == model:
            return label
    return model


def provider_of(model: str) -> str:
    """LiteLLM provider prefix; bare strings default to openai (litellm convention)."""
    return model.split("/", 1)[0] if "/" in model else "openai"


def price_per_token(model: str) -> tuple[float, float]:
    if model.startswith("ollama/"):
        return (0.0, 0.0)
    return _PRICE_PER_TOKEN.get(model, (0.0, 0.0))
