"""BYOK key store: Fernet-encrypted at rest, admin-shared-key fallback."""

import os

import pytest
from cryptography.fernet import Fernet

from src.database import get_connection, init_db


@pytest.fixture(autouse=True)
def _fernet_key(monkeypatch):
    monkeypatch.setenv("HEATER_AI_KEY", Fernet.generate_key().decode())
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()
    # clean slate for the test user
    conn = get_connection()
    try:
        conn.execute("DELETE FROM ai_provider_keys WHERE user_id = 99")
        conn.execute("DELETE FROM app_settings WHERE key = 'ai_shared_key'")
        conn.commit()
    finally:
        conn.close()


def test_store_and_get_roundtrip():
    from src.ai.keys import get_key, store_key

    store_key(99, "anthropic", "sk-ant-secret", label="mine")
    assert get_key(99, "anthropic") == "sk-ant-secret"


def test_ciphertext_is_encrypted_at_rest():
    from src.ai.keys import store_key

    store_key(99, "openai", "sk-plaintext-should-not-appear")
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT encrypted_key FROM ai_provider_keys WHERE user_id = 99 AND provider = 'openai'"
        ).fetchone()
    finally:
        conn.close()
    assert "sk-plaintext-should-not-appear" not in row["encrypted_key"]


def test_falls_back_to_admin_shared_key():
    from src.ai.keys import get_key, set_admin_shared_key

    set_admin_shared_key("anthropic", "sk-admin-shared", admin_id=1)
    # user 99 has no anthropic key of their own → gets the shared one
    assert get_key(99, "anthropic") == "sk-admin-shared"


def test_user_key_overrides_shared():
    from src.ai.keys import get_key, set_admin_shared_key, store_key

    set_admin_shared_key("anthropic", "sk-admin-shared", admin_id=1)
    store_key(99, "anthropic", "sk-user-own")
    assert get_key(99, "anthropic") == "sk-user-own"


def test_no_fernet_key_disables_storage(monkeypatch):
    from src.ai import keys as keys_mod

    monkeypatch.delenv("HEATER_AI_KEY", raising=False)
    monkeypatch.delenv("HEATER_RELAY_KEY", raising=False)
    with pytest.raises(RuntimeError):
        keys_mod.store_key(99, "anthropic", "sk-whatever")


def test_list_and_delete():
    from src.ai.keys import delete_key, list_keys, store_key

    store_key(99, "openai", "k1", label="work")
    store_key(99, "gemini", "k2", label="home")
    listed = list_keys(99)
    assert {r["provider"] for r in listed} == {"openai", "gemini"}
    # list must NOT leak plaintext
    assert all("k1" not in str(r) and "k2" not in str(r) for r in listed)
    delete_key(99, "openai", "work")
    assert {r["provider"] for r in list_keys(99)} == {"gemini"}


def test_list_admin_shared_providers_presence_only():
    from src.ai.keys import list_admin_shared_providers, set_admin_shared_key

    assert list_admin_shared_providers() == []
    set_admin_shared_key("deepseek", "sk-ds-secret", admin_id=1)
    set_admin_shared_key("anthropic", "sk-ant-secret", admin_id=1)
    provs = list_admin_shared_providers()
    assert provs == ["anthropic", "deepseek"]  # sorted, presence only
    # never leaks the stored key material
    assert all("sk-" not in p for p in provs)


def test_probe_shared_key_no_key_set():
    from src.ai.keys import probe_shared_key

    ok, msg = probe_shared_key("anthropic")
    assert ok is False
    assert "No shared key" in msg


def test_probe_shared_key_success(monkeypatch):
    import sys
    import types

    from src.ai.keys import probe_shared_key, set_admin_shared_key

    set_admin_shared_key("anthropic", "sk-admin", admin_id=1)
    fake = types.SimpleNamespace(completion=lambda **kw: object(), suppress_debug_info=False)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    ok, msg = probe_shared_key("anthropic")
    assert ok is True
    assert "Working" in msg


def test_probe_shared_key_surfaces_provider_error(monkeypatch):
    import sys
    import types

    from src.ai.keys import probe_shared_key, set_admin_shared_key

    set_admin_shared_key("openai", "sk-bad", admin_id=1)

    def _boom(**kw):
        raise RuntimeError("invalid api key")

    fake = types.SimpleNamespace(completion=_boom, suppress_debug_info=False)
    monkeypatch.setitem(sys.modules, "litellm", fake)
    ok, msg = probe_shared_key("openai")
    assert ok is False
    assert "RuntimeError" in msg and "invalid api key" in msg
