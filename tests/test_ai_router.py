"""Tier->model routing + admin overrides + the verified price table."""

import pytest

from src.database import init_db


@pytest.fixture(autouse=True)
def _multi_user(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("DELETE FROM app_settings WHERE key = 'ai_tier_models'")
        conn.commit()
    finally:
        conn.close()


def test_default_tier_models():
    from src.ai.router import model_for_tier

    assert model_for_tier("simple") == "anthropic/claude-haiku-4-5"
    assert model_for_tier("moderate") == "anthropic/claude-sonnet-4-6"
    assert model_for_tier("complex") == "anthropic/claude-opus-4-8"


def test_unknown_tier_falls_back_to_moderate():
    from src.ai.router import model_for_tier

    assert model_for_tier("nonsense") == "anthropic/claude-sonnet-4-6"


def test_admin_override(monkeypatch):
    from src.ai.router import model_for_tier, set_tier_models

    set_tier_models({"simple": "gemini/gemini-3-flash"}, admin_id=1)
    assert model_for_tier("simple") == "gemini/gemini-3-flash"
    # untouched tiers keep defaults
    assert model_for_tier("complex") == "anthropic/claude-opus-4-8"


def test_provider_of():
    from src.ai.router import provider_of

    assert provider_of("anthropic/claude-haiku-4-5") == "anthropic"
    assert provider_of("ollama/qwen2.5:7b") == "ollama"
    assert provider_of("bare-model") == "openai"  # litellm default convention


def test_price_table_known_models():
    from src.ai.router import price_per_token

    pin, pout = price_per_token("anthropic/claude-opus-4-8")
    assert pin == pytest.approx(5e-6)
    assert pout == pytest.approx(25e-6)
    # local model is free
    assert price_per_token("ollama/qwen2.5:7b") == (0.0, 0.0)
