"""Per-user watchlists and saved views (lineups / trades).

Tables: user_watchlist, user_saved_views  (created by init_db, Phase 7).

Works in both v1 (MULTI_USER off, user_id=0) and v2 (MULTI_USER on, resolves
the logged-in user via src.auth.current_user()).  Never crashes on missing
identity — falls back to user_id=0 for single-user dev.
"""

import json
import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)

# Local user sentinel used when MULTI_USER is off or no session user exists.
_LOCAL_USER_ID = 0


def _current_user_id() -> int:
    """Return the active user_id for DB operations.

    v2 (MULTI_USER on): resolves from the authenticated session.
    v1 or unauthenticated: returns 0 (stable local sentinel).
    """
    try:
        from src.auth import current_user, multi_user_enabled

        if multi_user_enabled():
            user = current_user()
            if user is not None:
                return int(user["user_id"])
    except Exception:
        pass
    return _LOCAL_USER_ID


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write(sql: str, params: tuple) -> bool:
    """Best-effort per-user write (watchlist / saved views).

    Members write their OWN rows from session processes while the scheduler is
    the primary SQLite writer; on a transient 'database is locked' — or any
    error — log + swallow so it never crashes the member's page (mirrors
    ``usage.bump_activity``). ``get_connection()`` already waits up to
    ``busy_timeout`` for the lock before raising. Returns True on a committed
    write. (2026-06-16 hardening — Phase 7 added the first member-triggered
    writes; this gives them the resilience the refresh_log path already has.)
    """
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(sql, params)
        conn.commit()
        return True
    except Exception as exc:
        logger.warning("user_data write skipped (%s): %s", type(exc).__name__, exc)
        return False
    finally:
        conn.close()


# ── Watchlist ─────────────────────────────────────────────────────────


def add_to_watchlist(player_id: int) -> None:
    """Add *player_id* to the current user's watchlist. Idempotent."""
    _write(
        "INSERT OR IGNORE INTO user_watchlist (user_id, player_id, created_at) VALUES (?, ?, ?)",
        (_current_user_id(), player_id, _now()),
    )


def remove_from_watchlist(player_id: int) -> None:
    """Remove *player_id* from the current user's watchlist. Silent if absent."""
    _write(
        "DELETE FROM user_watchlist WHERE user_id = ? AND player_id = ?",
        (_current_user_id(), player_id),
    )


def get_watchlist() -> set[int]:
    """Return the set of player_ids on the current user's watchlist."""
    from src.database import get_connection

    uid = _current_user_id()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT player_id FROM user_watchlist WHERE user_id = ?",
            (uid,),
        ).fetchall()
        return {int(r[0]) for r in rows}
    finally:
        conn.close()


def is_watched(player_id: int) -> bool:
    """True iff *player_id* is on the current user's watchlist."""
    return player_id in get_watchlist()


def toggle_watchlist(player_id: int) -> bool:
    """Add if absent, remove if present. Returns the new watched state."""
    if is_watched(player_id):
        remove_from_watchlist(player_id)
        return False
    else:
        add_to_watchlist(player_id)
        return True


# ── Saved views ───────────────────────────────────────────────────────


def save_view(kind: str, name: str, payload: dict) -> None:
    """Persist *payload* as a named view of *kind* for the current user.

    Upserts on (user_id, kind, name) — calling again with a different payload
    updates the existing row (and refreshes created_at).
    """
    _write(
        "INSERT INTO user_saved_views (user_id, kind, name, payload_json, created_at)"
        " VALUES (?, ?, ?, ?, ?)"
        " ON CONFLICT(user_id, kind, name)"
        " DO UPDATE SET payload_json = excluded.payload_json,"
        "               created_at   = excluded.created_at",
        (_current_user_id(), kind, name, json.dumps(payload, ensure_ascii=False), _now()),
    )


def load_view(kind: str, name: str) -> dict | None:
    """Load a saved view by kind + name. Returns None if not found or corrupt."""
    from src.database import get_connection

    uid = _current_user_id()
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT payload_json FROM user_saved_views WHERE user_id = ? AND kind = ? AND name = ?",
            (uid, kind, name),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    try:
        result = json.loads(row[0])
        if not isinstance(result, dict):
            return None
        return result
    except (json.JSONDecodeError, TypeError):
        logger.warning("user_data.load_view: corrupt JSON for kind=%r name=%r", kind, name)
        return None


def list_views(kind: str) -> list[str]:
    """Return saved-view names for *kind*, newest first."""
    from src.database import get_connection

    uid = _current_user_id()
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT name FROM user_saved_views WHERE user_id = ? AND kind = ? ORDER BY created_at DESC, id DESC",
            (uid, kind),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def delete_view(kind: str, name: str) -> None:
    """Delete a saved view by kind + name. Silent if not found."""
    _write(
        "DELETE FROM user_saved_views WHERE user_id = ? AND kind = ? AND name = ?",
        (_current_user_id(), kind, name),
    )
