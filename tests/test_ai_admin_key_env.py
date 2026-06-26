"""Env-based managed (admin shared) AI key — M-5.

App B (the FastAPI API) has no Streamlit admin console to call set_admin_shared_key,
and `set_setting` (the DB-backed shared-key writer) is MULTI_USER-gated, so the
operator can't install a managed key on App B that way. Instead the operator sets
two env vars (HEATER_AI_ADMIN_PROVIDER + HEATER_AI_ADMIN_KEY); get_admin_shared_key
falls back to them and list_admin_shared_providers surfaces the provider. Additive +
dormant: unset env → byte-identical to today (DB/app_settings path only).
"""

from __future__ import annotations

import os

import pytest

import src.ai.keys as keys


@pytest.fixture(autouse=True)
def _clear_admin_env(monkeypatch):
    """Hermetic: clear any managed-key env (the single pair + every per-provider
    HEATER_AI_ADMIN_KEY_* var) so a real .env on the dev box can't leak in."""
    monkeypatch.delenv("HEATER_AI_ADMIN_PROVIDER", raising=False)
    monkeypatch.delenv("HEATER_AI_ADMIN_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("HEATER_AI_ADMIN_KEY_"):
            monkeypatch.delenv(name, raising=False)


def test_env_admin_key_dormant_when_unset(monkeypatch):
    monkeypatch.delenv("HEATER_AI_ADMIN_PROVIDER", raising=False)
    monkeypatch.delenv("HEATER_AI_ADMIN_KEY", raising=False)
    # app_settings empty → no shared key at all
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: None)
    assert keys.get_admin_shared_key("deepseek") is None
    assert keys.list_admin_shared_providers() == []


def test_env_admin_key_surfaces_provider(monkeypatch):
    monkeypatch.setenv("HEATER_AI_ADMIN_PROVIDER", "deepseek")
    monkeypatch.setenv("HEATER_AI_ADMIN_KEY", "sk-managed-test")
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: None)  # no DB-stored shared key
    assert keys.get_admin_shared_key("deepseek") == "sk-managed-test"
    # a DIFFERENT provider is not surfaced by the single env key
    assert keys.get_admin_shared_key("openai") is None
    assert keys.list_admin_shared_providers() == ["deepseek"]


def test_env_admin_key_is_case_insensitive_on_provider(monkeypatch):
    monkeypatch.setenv("HEATER_AI_ADMIN_PROVIDER", "DeepSeek")
    monkeypatch.setenv("HEATER_AI_ADMIN_KEY", "sk-x")
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: None)
    assert keys.get_admin_shared_key("deepseek") == "sk-x"


def test_db_shared_key_takes_precedence_over_env(monkeypatch):
    # When BOTH an app_settings ciphertext AND the env key exist for a provider,
    # the explicitly-configured DB key wins (env is a fallback only).
    import json

    monkeypatch.setenv("HEATER_AI_ADMIN_PROVIDER", "deepseek")
    monkeypatch.setenv("HEATER_AI_ADMIN_KEY", "sk-env")
    monkeypatch.setattr(keys, "_decrypt", lambda ct: "sk-db" if ct == "CT" else None)
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: json.dumps({"deepseek": "CT"}))
    assert keys.get_admin_shared_key("deepseek") == "sk-db"


def test_env_provider_merges_with_db_providers_in_listing(monkeypatch):
    import json

    monkeypatch.setenv("HEATER_AI_ADMIN_PROVIDER", "deepseek")
    monkeypatch.setenv("HEATER_AI_ADMIN_KEY", "sk-env")
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: json.dumps({"openai": "CT"}))
    # both the DB provider (openai) and the env provider (deepseek) are listed, sorted
    assert keys.list_admin_shared_providers() == ["deepseek", "openai"]


def test_env_admin_key_requires_both_vars(monkeypatch):
    monkeypatch.setenv("HEATER_AI_ADMIN_PROVIDER", "deepseek")
    monkeypatch.delenv("HEATER_AI_ADMIN_KEY", raising=False)
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: None)
    assert keys.get_admin_shared_key("deepseek") is None
    assert keys.list_admin_shared_providers() == []


def test_half_configured_env_warns_once(monkeypatch, caplog):
    """The most likely operator typo (one of the two vars set) must leave a
    breadcrumb — warn ONCE, not silently inert and not on every call (log spam)."""
    import logging

    monkeypatch.setattr(keys, "_half_config_warned", False)  # reset the once-flag
    monkeypatch.setenv("HEATER_AI_ADMIN_PROVIDER", "deepseek")
    monkeypatch.delenv("HEATER_AI_ADMIN_KEY", raising=False)  # forgot the key
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: None)
    with caplog.at_level(logging.WARNING, logger="src.ai.keys"):
        assert keys.get_admin_shared_key("deepseek") is None
        keys.get_admin_shared_key("deepseek")  # second call must NOT warn again
        keys.list_admin_shared_providers()
    half = [r for r in caplog.records if "half-configured" in r.getMessage()]
    assert len(half) == 1


# --- multi-provider: per-provider env vars (HEATER_AI_ADMIN_KEY_<PROVIDER>) ----


def test_per_provider_env_key_surfaces_only_that_provider(monkeypatch):
    monkeypatch.setenv("HEATER_AI_ADMIN_KEY_DEEPSEEK", "sk-ds")
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: None)
    assert keys.get_admin_shared_key("deepseek") == "sk-ds"
    assert keys.get_admin_shared_key("openai") is None
    assert keys.list_admin_shared_providers() == ["deepseek"]


def test_all_six_per_provider_env_keys_surface(monkeypatch):
    six = {
        "anthropic": "sk-ant-x",
        "openai": "sk-proj-x",
        "gemini": "g-key-x",
        "deepseek": "sk-ds-x",
        "xai": "xai-x",
        "openrouter": "sk-or-x",
    }
    for provider, key in six.items():
        monkeypatch.setenv(f"HEATER_AI_ADMIN_KEY_{provider.upper()}", key)
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: None)
    assert keys.list_admin_shared_providers() == sorted(six)
    for provider, key in six.items():
        assert keys.get_admin_shared_key(provider) == key


def test_per_provider_and_single_pair_coexist(monkeypatch):
    monkeypatch.setenv("HEATER_AI_ADMIN_PROVIDER", "anthropic")
    monkeypatch.setenv("HEATER_AI_ADMIN_KEY", "sk-ant-single")
    monkeypatch.setenv("HEATER_AI_ADMIN_KEY_OPENAI", "sk-proj-x")
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: None)
    assert keys.list_admin_shared_providers() == ["anthropic", "openai"]
    assert keys.get_admin_shared_key("anthropic") == "sk-ant-single"
    assert keys.get_admin_shared_key("openai") == "sk-proj-x"


def test_db_key_wins_over_per_provider_env(monkeypatch):
    import json

    monkeypatch.setenv("HEATER_AI_ADMIN_KEY_DEEPSEEK", "sk-env")
    monkeypatch.setattr(keys, "_decrypt", lambda ct: "sk-db" if ct == "CT" else None)
    monkeypatch.setattr(keys, "get_setting", lambda *_a, **_k: json.dumps({"deepseek": "CT"}))
    assert keys.get_admin_shared_key("deepseek") == "sk-db"
