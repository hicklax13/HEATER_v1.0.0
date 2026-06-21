"""thinking_params_for_model maps the Off/Low/Med/High dial to per-provider
reasoning params. off/low -> {} (fastest); unsupported provider -> {} (no-op)."""

import pytest

from src.ai.router import thinking_params_for_model


@pytest.mark.parametrize("effort", [None, "off", "low"])
def test_off_and_low_are_noop_for_every_provider(effort):
    assert thinking_params_for_model("anthropic/claude-opus-4-8", effort) == {}
    assert thinking_params_for_model("openai/gpt-5.5", effort) == {}
    assert thinking_params_for_model("deepseek/deepseek-v4-pro", effort) == {}


def test_anthropic_medium_and_high_emit_thinking_budget():
    med = thinking_params_for_model("anthropic/claude-opus-4-8", "medium")
    high = thinking_params_for_model("anthropic/claude-opus-4-8", "high")
    assert med["thinking"]["type"] == "enabled" and med["thinking"]["budget_tokens"] > 0
    assert high["thinking"]["budget_tokens"] > med["thinking"]["budget_tokens"]


def test_openai_uses_reasoning_effort_passthrough():
    assert thinking_params_for_model("openai/gpt-5.5", "medium") == {"reasoning_effort": "medium"}
    assert thinking_params_for_model("openai/gpt-5.5", "high") == {"reasoning_effort": "high"}


def test_unsupported_provider_is_noop_even_on_high():
    assert thinking_params_for_model("deepseek/deepseek-v4-pro", "high") == {}
    assert thinking_params_for_model("gemini/gemini-3.5-flash", "high") == {}
