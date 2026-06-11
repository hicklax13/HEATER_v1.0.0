"""BYOK provider-key store: Fernet-encrypted at rest, admin-shared-key fallback.

Mirrors src/token_relay.py's Fernet usage. The encryption key comes from
HEATER_AI_KEY (falling back to HEATER_RELAY_KEY so the operator can reuse one
Fernet value). Plaintext keys are never written to the DB or logged.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from cryptography.fernet import Fernet

from src.app_settings import get_setting, set_setting

_SHARED_KEY_SETTING = "ai_shared_key"  # JSON {provider: ciphertext}


def _fernet() -> Fernet:
    raw = (os.environ.get("HEATER_AI_KEY") or os.environ.get("HEATER_RELAY_KEY") or "").strip()
    if not raw:
        raise RuntimeError("AI key storage disabled: set HEATER_AI_KEY (or reuse HEATER_RELAY_KEY) to a Fernet key.")
    return Fernet(raw.encode())


def _encrypt(plaintext: str) -> str:
    return _fernet().encrypt(plaintext.encode()).decode()


def _decrypt(ciphertext: str) -> str:
    return _fernet().decrypt(ciphertext.encode()).decode()


def store_key(user_id: int, provider: str, api_key: str, label: str | None = None) -> None:
    """Encrypt + upsert a user's provider key. Raises if no Fernet key is set."""
    from src.database import get_connection

    ciphertext = _encrypt(api_key)  # raises RuntimeError before any DB write if unconfigured
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO ai_provider_keys (user_id, provider, label, encrypted_key, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, provider, label) DO UPDATE SET encrypted_key = excluded.encrypted_key
            """,
            (user_id, provider, label, ciphertext, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_key(user_id: int, provider: str) -> str | None:
    """The user's own key for the provider, else the admin shared key, else None."""
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT encrypted_key FROM ai_provider_keys WHERE user_id = ? AND provider = ? ORDER BY id DESC LIMIT 1",
            (user_id, provider),
        ).fetchone()
    finally:
        conn.close()
    if row is not None:
        try:
            return _decrypt(row["encrypted_key"])
        except Exception:
            return None
    return get_admin_shared_key(provider)


def list_keys(user_id: int) -> list[dict]:
    """Metadata for a user's keys (provider/label/created_at) — never the plaintext."""
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT provider, label, created_at FROM ai_provider_keys WHERE user_id = ? ORDER BY id DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def delete_key(user_id: int, provider: str, label: str | None = None) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        if label is None:
            conn.execute(
                "DELETE FROM ai_provider_keys WHERE user_id = ? AND provider = ? AND label IS NULL",
                (user_id, provider),
            )
        else:
            conn.execute(
                "DELETE FROM ai_provider_keys WHERE user_id = ? AND provider = ? AND label = ?",
                (user_id, provider, label),
            )
        conn.commit()
    finally:
        conn.close()


def set_admin_shared_key(provider: str, api_key: str, admin_id: int) -> None:
    """Encrypt + store the shared fallback key for a provider (admin only path).

    Stored as a JSON map {provider: ciphertext} in app_settings under one key.
    set_setting() is MULTI_USER-gated and audit-logs the action name only.
    """
    import json

    ciphertext = _encrypt(api_key)  # raises before any persistence if unconfigured
    raw = get_setting(_SHARED_KEY_SETTING)
    data = {}
    if raw:
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            data = {}
    data[provider] = ciphertext
    set_setting(_SHARED_KEY_SETTING, json.dumps(data), admin_id)


def get_admin_shared_key(provider: str) -> str | None:
    import json

    raw = get_setting(_SHARED_KEY_SETTING)
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    ct = data.get(provider)
    if not ct:
        return None
    try:
        return _decrypt(ct)
    except Exception:
        return None
