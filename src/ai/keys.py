"""BYOK provider-key store: Fernet-encrypted at rest, admin-shared-key fallback.

Mirrors src/token_relay.py's Fernet usage. The encryption key comes from
HEATER_AI_KEY (falling back to HEATER_RELAY_KEY so the operator can reuse one
Fernet value). Plaintext keys are never written to the DB or logged.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

from cryptography.fernet import Fernet

from src.app_settings import get_setting, set_setting

logger = logging.getLogger(__name__)

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
            # A stored key that won't decrypt (rotated HEATER_AI_KEY, corruption)
            # would otherwise silently look like "user has no key" — surface it so
            # the operator can see why a user's BYOK key stopped working. The Fernet
            # error carries no key material; user_id/provider are not secrets.
            logger.warning(
                "ai key decrypt failed for user_id=%s provider=%s (rotated HEATER_AI_KEY?)",
                user_id,
                provider,
            )
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
    set_setting() is MULTI_USER-gated and audit-logs the value it receives — which
    here is the ENCRYPTED ciphertext map, never the plaintext key. (The admin page
    additionally logs an "ai_shared_key_update" action with the provider only.)
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
        # Shared-key decrypt failure (rotated HEATER_AI_KEY) would silently disable
        # the fallback for every user — log it so the operator can re-set the key.
        logger.warning("admin shared key decrypt failed for provider=%s (rotated HEATER_AI_KEY?)", provider)
        return None


def list_admin_shared_providers() -> list[str]:
    """Provider NAMES that currently have a shared key configured (never the keys).

    Presence-only visibility for the Admin Console — the ciphertext is never
    returned. Empty when the flag is off / nothing is set / the map is corrupt.
    """
    import json

    raw = get_setting(_SHARED_KEY_SETTING)
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, dict):
        return []
    # only surface providers whose stored ciphertext is non-empty
    return sorted(p for p, ct in data.items() if ct)


def _probe_model_for_provider(provider: str) -> str | None:
    """Cheapest catalog model for a provider — used only for the live key probe."""
    from src.ai.router import model_catalog, price_per_token, provider_of

    candidates = [m for _label, m in model_catalog() if provider_of(m) == provider]
    if not candidates:
        return None
    return min(candidates, key=lambda m: sum(price_per_token(m)))


def probe_shared_key(provider: str) -> tuple[bool, str]:
    """Live 1-shot check of the admin shared key for a provider → (ok, message).

    Decrypts the stored shared key and runs a tiny completion through the cheapest
    catalog model for that provider. Success = the call returns without raising
    (a valid-but-thinking model may emit empty text on a tiny budget — that still
    proves the key authenticates). The key is never returned or logged.
    """
    key = get_admin_shared_key(provider)
    if not key:
        return False, "No shared key set."
    model = _probe_model_for_provider(provider)
    if not model:
        return False, f"No catalog model for {provider}."
    try:
        import litellm

        litellm.suppress_debug_info = True
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "Reply with: OK"}],
            api_key=key,
            max_tokens=32,
        )
        return True, f"Working — authenticated via {model}."
    except Exception as e:  # noqa: BLE001 — surface any provider error verbatim to the admin
        return False, f"{type(e).__name__}: {str(e)[:200]}"
