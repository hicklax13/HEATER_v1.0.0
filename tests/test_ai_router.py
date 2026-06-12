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

    assert model_for_tier("simple") == "deepseek/deepseek-v4-flash"
    assert model_for_tier("moderate") == "deepseek/deepseek-v4-flash"
    assert model_for_tier("complex") == "deepseek/deepseek-v4-pro"


def test_unknown_tier_falls_back_to_moderate():
    from src.ai.router import model_for_tier

    assert model_for_tier("nonsense") == "deepseek/deepseek-v4-flash"


def test_admin_override(monkeypatch):
    from src.ai.router import model_for_tier, set_tier_models

    set_tier_models({"simple": "gemini/gemini-3-flash"}, admin_id=1)
    assert model_for_tier("simple") == "gemini/gemini-3-flash"
    # untouched tiers keep defaults
    assert model_for_tier("complex") == "deepseek/deepseek-v4-pro"


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


def test_openai_and_gemini_models_priced_and_in_catalog():
    from src.ai.router import model_catalog, price_per_token, provider_of

    # OpenAI GPT-5.x
    assert price_per_token("openai/gpt-5.5") == pytest.approx((5e-6, 30e-6))
    assert price_per_token("openai/gpt-5.4") == pytest.approx((2.5e-6, 15e-6))
    assert price_per_token("openai/gpt-5.4-nano") == pytest.approx((0.2e-6, 1.25e-6))
    assert provider_of("openai/gpt-5.5") == "openai"
    # Google Gemini 3.x
    assert price_per_token("gemini/gemini-3.1-pro-preview") == pytest.approx((2e-6, 12e-6))
    assert price_per_token("gemini/gemini-3.5-flash") == pytest.approx((1.5e-6, 9e-6))
    assert price_per_token("gemini/gemini-3.1-flash-lite") == pytest.approx((0.25e-6, 1.5e-6))
    assert provider_of("gemini/gemini-3.5-flash") == "gemini"
    # all six in the picker catalog
    models = {m for _, m in model_catalog()}
    for m in (
        "openai/gpt-5.5",
        "openai/gpt-5.4",
        "openai/gpt-5.4-nano",
        "gemini/gemini-3.1-pro-preview",
        "gemini/gemini-3.5-flash",
        "gemini/gemini-3.1-flash-lite",
    ):
        assert m in models
