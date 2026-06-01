"""Per-session page-view logging + session/dwell tracking + analytics readers.

MULTI_USER-gated. When the flag is off, log_page_view() and bump_activity() are
no-ops (v1 byte-for-byte). When on: one deduped usage_event per (page, action)
per session, a sessions row per browser session, page_visits dwell tracking, and
a last_seen_at bump on the user. Readers power the admin analytics surfaces.
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import UTC, datetime, timedelta

from src.auth import current_user, multi_user_enabled


def _session_state():
    import streamlit as st

    return st.session_state


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _seconds_between(a_iso: str, b_iso: str) -> float:
    try:
        a = datetime.fromisoformat(a_iso)
        b = datetime.fromisoformat(b_iso)
    except (ValueError, TypeError):
        return 0.0
    return max(0.0, (b - a).total_seconds())


def _ensure_session_row(conn, session_id: str, user_id: int, now: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id, user_id, login_at, last_activity_at) VALUES (?, ?, ?, ?)",
        (session_id, user_id, now, now),
    )


def _close_open_visit(conn, session_id: str, exit_at: str) -> None:
    row = conn.execute(
        "SELECT id, enter_at FROM page_visits WHERE session_id = ? AND exit_at IS NULL ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is None:
        return
    dwell = _seconds_between(row["enter_at"], exit_at)
    conn.execute(
        "UPDATE page_visits SET exit_at = ?, dwell_seconds = ? WHERE id = ?",
        (exit_at, dwell, row["id"]),
    )


def _track_page_visit(conn, state, session_id: str, user_id: int, page: str, now: str) -> None:
    current = state.get("_current_page")
    if current == page:
        return
    if current is not None:
        _close_open_visit(conn, session_id, now)
    conn.execute(
        "INSERT INTO page_visits (session_id, user_id, page, enter_at, exit_at, dwell_seconds) "
        "VALUES (?, ?, ?, ?, NULL, NULL)",
        (session_id, user_id, page, now),
    )
    state["_current_page"] = page


def log_page_view(page: str, action: str = "view") -> None:
    if not multi_user_enabled():
        return
    user = current_user()
    if not user:
        return
    state = _session_state()
    session_id = state.get("_usage_session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
        state["_usage_session_id"] = session_id
    logged = state.get("_usage_logged")
    if logged is None:
        logged = set()
        state["_usage_logged"] = logged
    dedup_key = (page, action)

    from src.database import get_connection

    user_id = user["user_id"]
    now = _now_iso()
    conn = get_connection()
    try:
        _ensure_session_row(conn, session_id, user_id, now)
        _track_page_visit(conn, state, session_id, user_id, page, now)
        conn.execute("UPDATE sessions SET last_activity_at = ? WHERE session_id = ?", (now, session_id))
        inserted = False
        if dedup_key not in logged:
            conn.execute(
                "INSERT INTO usage_events (user_id, page, action, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, page, action, session_id, now),
            )
            conn.execute("UPDATE users SET last_seen_at = ? WHERE user_id = ?", (now, user_id))
            inserted = True
        conn.commit()
    finally:
        conn.close()
    if inserted:
        logged.add(dedup_key)


def bump_activity() -> None:
    if not multi_user_enabled():
        return
    user = current_user()
    if not user:
        return
    state = _session_state()
    session_id = state.get("_usage_session_id")
    if not session_id:
        return
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("UPDATE sessions SET last_activity_at = ? WHERE session_id = ?", (_now_iso(), session_id))
        conn.commit()
    except Exception:
        # Best-effort heartbeat: if the single-writer scheduler holds the DB at
        # this instant, skip silently — the next 60s tick self-heals. A transient
        # lock must never crash a member's page.
        pass
    finally:
        conn.close()


def _lazy_close_open_visits(conn) -> None:
    """Resolve dangling open visits using their session's last_activity_at."""
    rows = conn.execute(
        """
        SELECT pv.id AS id, pv.enter_at AS enter_at, s.last_activity_at AS last_activity_at
        FROM page_visits pv
        JOIN sessions s ON s.session_id = pv.session_id
        WHERE pv.exit_at IS NULL
        """
    ).fetchall()
    if not rows:
        return
    for r in rows:
        dwell = _seconds_between(r["enter_at"], r["last_activity_at"])
        conn.execute(
            "UPDATE page_visits SET exit_at = ?, dwell_seconds = ? WHERE id = ?",
            (r["last_activity_at"], dwell, r["id"]),
        )
    conn.commit()


def dau_series(days: int = 30) -> list[dict]:
    from src.database import get_connection

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT substr(created_at, 1, 10) AS day, COUNT(DISTINCT user_id) AS users
            FROM usage_events
            WHERE created_at >= ?
            GROUP BY day
            ORDER BY day
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def most_used_pages(days: int = 30) -> list[dict]:
    from src.database import get_connection

    cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT page, COUNT(*) AS views
            FROM usage_events
            WHERE created_at >= ?
            GROUP BY page
            ORDER BY views DESC, page
            """,
            (cutoff,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def per_user_activity() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT u.user_id AS user_id,
                   u.username AS username,
                   COUNT(e.id) AS events,
                   u.last_seen_at AS last_seen
            FROM users u
            LEFT JOIN usage_events e ON e.user_id = u.user_id
            GROUP BY u.user_id
            ORDER BY events DESC, u.username
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def last_seen_summary() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT username, last_seen_at AS last_seen FROM users ORDER BY last_seen_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def session_timeline(user_id: int | None = None) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        _lazy_close_open_visits(conn)
        sql = (
            "SELECT s.session_id AS session_id, s.user_id AS user_id, u.username AS username, "
            "s.login_at AS login_at, s.last_activity_at AS last_activity_at "
            "FROM sessions s LEFT JOIN users u ON u.user_id = s.user_id"
        )
        params: list = []
        if user_id is not None:
            sql += " WHERE s.user_id = ?"
            params.append(user_id)
        sql += " ORDER BY s.login_at DESC"
        out = []
        for r in conn.execute(sql, params).fetchall():
            d = dict(r)
            d["duration_seconds"] = _seconds_between(d["login_at"], d["last_activity_at"])
            out.append(d)
        return out
    finally:
        conn.close()


def page_dwell_summary(user_id: int | None = None) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        _lazy_close_open_visits(conn)
        sql = (
            "SELECT page, "
            "COALESCE(SUM(dwell_seconds), 0) AS total_seconds, "
            "COALESCE(AVG(dwell_seconds), 0) AS avg_seconds, "
            "COUNT(*) AS visits "
            "FROM page_visits WHERE dwell_seconds IS NOT NULL"
        )
        params: list = []
        if user_id is not None:
            sql += " AND user_id = ?"
            params.append(user_id)
        sql += " GROUP BY page ORDER BY total_seconds DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def usage_csv() -> str:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT e.id AS id, e.created_at AS created_at, e.user_id AS user_id,
                   u.username AS username, e.page AS page, e.action AS action,
                   e.session_id AS session_id
            FROM usage_events e
            LEFT JOIN users u ON u.user_id = e.user_id
            ORDER BY e.created_at DESC
            """
        ).fetchall()
    finally:
        conn.close()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "created_at", "user_id", "username", "page", "action", "session_id"])
    for r in rows:
        writer.writerow(
            [r["id"], r["created_at"], r["user_id"], r["username"], r["page"], r["action"], r["session_id"]]
        )
    return buf.getvalue()
