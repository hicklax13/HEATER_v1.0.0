"""Lightweight per-page usage logging (v2 Plan 2, MULTI_USER-gated).

Records one 'view' event per (page, action) per session so the admin can see
which features get used, and bumps users.last_seen_at. No-op in single-user v1
(flag off) or when there is no logged-in user.
"""

import uuid
from datetime import UTC, datetime

from src.auth import current_user, multi_user_enabled


def _session_state():
    """Seam over st.session_state so unit tests can inject a plain dict."""
    import streamlit as st

    return st.session_state


def log_page_view(page: str, action: str = "view") -> None:
    """Record a usage event once per (page, action) per session; bump last_seen_at.

    No-op when MULTI_USER is off or no user is logged in. The dedup key is added
    only after a successful commit, so a failed write is retried on the next run.
    """
    if not multi_user_enabled():
        return
    user = current_user()
    if user is None:
        return

    state = _session_state()
    logged = state.get("_usage_logged")
    if logged is None:
        logged = set()
        state["_usage_logged"] = logged
    dedup_key = (page, action)
    if dedup_key in logged:
        return

    session_id = state.get("_usage_session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        state["_usage_session_id"] = session_id

    from src.database import get_connection

    conn = get_connection()
    try:
        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO usage_events (user_id, page, action, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (user["user_id"], page, action, session_id, now),
        )
        conn.execute(
            "UPDATE users SET last_seen_at = ? WHERE user_id = ?",
            (now, user["user_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    logged.add(dedup_key)
