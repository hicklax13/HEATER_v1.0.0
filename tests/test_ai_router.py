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


def test_deepseek_models_priced_and_in_catalog():
    from src.ai.router import label_for_model, model_catalog, price_per_token, provider_of

    assert price_per_token("deepseek/deepseek-v4-flash") == pytest.approx((0.14e-6, 0.28e-6))
    assert price_per_token("deepseek/deepseek-v4-pro") == pytest.approx((0.435e-6, 0.87e-6))
    assert provider_of("deepseek/deepseek-v4-flash") == "deepseek"
    catalog = dict((m, label) for label, m in model_catalog())
    assert "deepseek/deepseek-v4-flash" in catalog
    assert "deepseek/deepseek-v4-pro" in catalog
    assert label_for_model("deepseek/deepseek-v4-pro") == "DeepSeek V4 Pro"
    assert label_for_model("unknown/model") == "unknown/model"
