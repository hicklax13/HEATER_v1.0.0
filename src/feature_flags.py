"""Per-page feature flags. Absence = enabled. MULTI_USER-gated writes; admins bypass gate."""

from __future__ import annotations

from datetime import UTC, datetime

from src.audit import log_action
from src.auth import multi_user_enabled


def is_page_enabled(key: str) -> bool:
    if not multi_user_enabled():
        return True
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT enabled FROM feature_flags WHERE key = ?", (key,)).fetchone()
        return row is None or row["enabled"] == 1
    finally:
        conn.close()


def set_page_flag(key: str, enabled: bool, admin_id: int) -> None:
    if not multi_user_enabled():
        return
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO feature_flags (key, enabled, updated_by, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET enabled = excluded.enabled,
                                           updated_by = excluded.updated_by,
                                           updated_at = excluded.updated_at
            """,
            (key, 1 if enabled else 0, admin_id, datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    log_action(admin_id, "toggle_flag", target=key, detail={"enabled": bool(enabled)})


def list_page_flags() -> dict[str, bool]:
    if not multi_user_enabled():
        return {}
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute("SELECT key, enabled FROM feature_flags").fetchall()
        return {r["key"]: r["enabled"] == 1 for r in rows}
    finally:
        conn.close()


def require_page_enabled(key: str) -> None:
    if not multi_user_enabled():
        return
    from src.auth import current_user

    user = current_user()
    if user and user.get("is_admin"):
        return
    if is_page_enabled(key):
        return
    import streamlit as st

    st.error("This page is currently disabled by the administrator.")
    st.stop()
