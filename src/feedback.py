"""Per-feature feedback capture + admin inbox (v2 Plan 2, MULTI_USER-gated).

Feedback is recorded against the logged-in user, tagged with the page and an
optional feature, stamped with the running app version, and snapshotted with
the current data-freshness state so an admin can reproduce what the user saw.
"""

import json
from datetime import UTC, datetime

from src.version import APP_VERSION

_VALID_STATUSES = ("new", "triaged", "resolved")


def _capture_data_state() -> str | None:
    """JSON snapshot of the refresh_log, or None if unavailable.

    Best-effort: a feedback submission must never fail because the snapshot did.
    Imported lazily so importing this module never drags in the database layer.
    """
    try:
        from src.database import get_refresh_log_snapshot

        snapshot = get_refresh_log_snapshot()
        if not snapshot:
            return None
        return json.dumps(snapshot)
    except Exception:
        return None


def submit_feedback(user_id: int, page: str, message: str, feature_tag: str | None = None) -> int:
    """Insert a feedback row; return the new row id.

    status defaults to 'new' at the DB layer. app_version + data_state are
    captured here so the admin inbox can reproduce the user's context.
    """
    from src.database import get_connection

    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO feedback (user_id, page, feature_tag, message, app_version, "
            "data_state, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                page,
                feature_tag,
                message,
                APP_VERSION,
                _capture_data_state(),
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_feedback(status: str | None = None) -> list[dict]:
    """Feedback rows joined with submitter username/team, newest first."""
    from src.database import get_connection

    conn = get_connection()
    try:
        sql = (
            "SELECT f.*, u.username AS username, u.team_name AS team_name "
            "FROM feedback f LEFT JOIN users u ON u.user_id = f.user_id"
        )
        params: tuple = ()
        if status is not None:
            sql += " WHERE f.status = ?"
            params = (status,)
        sql += " ORDER BY f.created_at DESC, f.id DESC"
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def set_feedback_status(feedback_id: int, status: str) -> None:
    """Update a feedback row's triage status. Validates against _VALID_STATUSES."""
    if status not in _VALID_STATUSES:
        raise ValueError(f"Invalid feedback status: {status!r}. Expected one of {_VALID_STATUSES}.")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("UPDATE feedback SET status = ? WHERE id = ?", (status, feedback_id))
        conn.commit()
    finally:
        conn.close()


def set_feedback_notes(feedback_id: int, notes: str) -> None:
    """Save an admin's triage notes onto a feedback row."""
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("UPDATE feedback SET admin_notes = ? WHERE id = ?", (notes, feedback_id))
        conn.commit()
    finally:
        conn.close()
