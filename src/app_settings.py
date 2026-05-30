"""Application settings (broadcast banner + maintenance mode). MULTI_USER-gated writes."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from src.audit import log_action
from src.auth import multi_user_enabled

_BROADCAST_KEY = "broadcast"
_MAINTENANCE_KEY = "maintenance"
_DEFAULT_TOGGLE = {"enabled": False, "message": ""}


def _put_setting(key: str, value: str, admin_id: int) -> None:
    """Upsert a raw setting value. No audit (callers add the semantic audit row)."""
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO app_settings (key, value, updated_by, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                           updated_by = excluded.updated_by,
                                           updated_at = excluded.updated_at
            """,
            (key, value, admin_id, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_setting(key: str, default: str | None = None) -> str | None:
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT value FROM app_settings WHERE key = ?", (key,)).fetchone()
        return row["value"] if row is not None else default
    finally:
        conn.close()


def set_setting(key: str, value: str, admin_id: int) -> None:
    if not multi_user_enabled():
        return
    _put_setting(key, value, admin_id)
    log_action(admin_id, "set_setting", target=key, detail={"value": value})


def _get_toggle(key: str) -> dict:
    raw = get_setting(key)
    if raw is None:
        return dict(_DEFAULT_TOGGLE)
    try:
        data = json.loads(raw)
        return {"enabled": bool(data.get("enabled", False)), "message": str(data.get("message", ""))}
    except (ValueError, TypeError):
        return dict(_DEFAULT_TOGGLE)


def get_broadcast() -> dict:
    return _get_toggle(_BROADCAST_KEY)


def get_maintenance() -> dict:
    return _get_toggle(_MAINTENANCE_KEY)


def set_broadcast(enabled: bool, message: str, admin_id: int) -> None:
    if not multi_user_enabled():
        return
    _put_setting(_BROADCAST_KEY, json.dumps({"enabled": bool(enabled), "message": message}), admin_id)
    log_action(admin_id, "set_broadcast", detail={"enabled": bool(enabled), "message": message})


def set_maintenance(enabled: bool, message: str, admin_id: int) -> None:
    if not multi_user_enabled():
        return
    _put_setting(_MAINTENANCE_KEY, json.dumps({"enabled": bool(enabled), "message": message}), admin_id)
    log_action(admin_id, "toggle_maintenance", detail={"enabled": bool(enabled), "message": message})
