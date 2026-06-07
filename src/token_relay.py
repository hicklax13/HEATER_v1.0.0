"""Encrypted Yahoo-token relay (mini PC ⇄ GitHub gist ⇄ Railway server).

Root cause this solves: Yahoo blocks OAuth token refresh from Railway's datacenter
IP. The mini PC refreshes the token from a residential IP and uploads it (encrypted)
to a secret gist; the server pulls it here. All server-side behavior is DORMANT
unless HEATER_TOKEN_RELAY_URL + HEATER_RELAY_KEY are set (v1 byte-for-byte).
"""

from __future__ import annotations

import json
import logging
import os
import time

import requests as _requests
from cryptography.fernet import Fernet

from src.yahoo_api import _AUTH_DIR, _write_token_file

logger = logging.getLogger(__name__)

GIST_FILENAME = "heater_yahoo_token.enc"
_relay_healthy: bool | None = None  # for degrade/recover transition logging
# Warn when the gist's newest token ages past this (the mini-PC relay refreshes
# every 30 min and tokens live 60, so >45 min stale ⇒ the relay likely stopped).
_RELAY_STALE_WARN_MINUTES = 45


def _fernet() -> Fernet | None:
    key = os.environ.get("HEATER_RELAY_KEY", "").strip()
    if not key:
        return None
    try:
        return Fernet(key.encode())
    except Exception:
        logger.warning("token_relay: HEATER_RELAY_KEY is not a valid Fernet key.")
        return None


def encrypt_token(token_dict: dict, fernet: Fernet) -> str:
    return fernet.encrypt(json.dumps(token_dict).encode()).decode()


def decrypt_token(ciphertext: str, fernet: Fernet) -> dict:
    return json.loads(fernet.decrypt(ciphertext.encode()).decode())


def relay_enabled() -> bool:
    return bool(os.environ.get("HEATER_TOKEN_RELAY_URL") and os.environ.get("HEATER_RELAY_KEY"))


def _read_local_token() -> dict | None:
    p = _AUTH_DIR / "yahoo_token.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def pull_relayed_token() -> bool:
    """Fetch + decrypt the relayed token from the gist raw URL; write it to the
    volume if it is newer than what's on disk. No-op (returns False) when the relay
    is not configured. Never logs token contents.

    Observability: warns ONCE (healthy->stale transition) when the gist stops
    advancing past ``_RELAY_STALE_WARN_MINUTES`` — i.e. the mini-PC relay is likely
    down — and logs recovery once when it advances again. This keeps a dead relay
    from being silently invisible (the exact failure class this feature fixes)."""
    global _relay_healthy
    if not relay_enabled():
        return False
    url = os.environ.get("HEATER_TOKEN_RELAY_URL", "").strip()
    f = _fernet()
    if not url or f is None:
        return False

    try:
        resp = _requests.get(url, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        logger.warning("pull_relayed_token: could not fetch relay token: %s", type(exc).__name__)
        return False

    try:
        token = decrypt_token(resp.text.strip(), f)
    except Exception:
        logger.warning("pull_relayed_token: decrypt failed (HEATER_RELAY_KEY mismatch?).")
        return False

    if not (token.get("access_token") and token.get("refresh_token") and token.get("token_time")):
        logger.warning("pull_relayed_token: relayed token missing access/refresh token or token_time.")
        return False

    existing = _read_local_token()
    new_tt = float(token.get("token_time") or 0)
    old_tt = float((existing or {}).get("token_time") or 0)
    if existing and new_tt <= old_tt:
        # Gist not advancing. Normal between the relay's 30-min refreshes; only a
        # problem once the newest token ages past the warn threshold (relay down).
        age_min = int((time.time() - new_tt) / 60) if new_tt else -1
        if (age_min < 0 or age_min > _RELAY_STALE_WARN_MINUTES) and _relay_healthy is not False:
            logger.warning(
                "pull_relayed_token: relay gist not advancing — newest token is ~%dm old. "
                "Is the mini-PC relay still running?",
                age_min,
            )
            _relay_healthy = False
        return False

    if not _write_token_file(token):
        logger.warning("pull_relayed_token: failed to write the relayed token to disk.")
        return False

    if _relay_healthy is False:
        logger.info("pull_relayed_token: relay recovered (gist advancing again).")
    _relay_healthy = True
    age_min = max(0, int((time.time() - new_tt) / 60)) if new_tt else -1
    logger.info("pull_relayed_token: wrote relayed token (age ~%dm).", age_min)
    return True
