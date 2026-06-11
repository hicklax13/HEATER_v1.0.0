"""The AI's only write path: enqueue refresh requests; the single-writer
scheduler drains them. The AI never writes data directly (single-writer rule)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

logger = logging.getLogger(__name__)


def _bootstrap_all_data(**kwargs):
    # Indirection so tests can monkeypatch without importing the heavy module.
    from src.data_bootstrap import bootstrap_all_data

    return bootstrap_all_data(**kwargs)


def request_refresh(source: str, requested_by: int | None = None) -> int:
    from src.database import get_connection

    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO forced_refresh_queue (source, requested_by, status, created_at) VALUES (?, ?, 'pending', ?)",
            (source, requested_by, datetime.now(UTC).isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def status_of(request_id: int) -> str | None:
    from src.database import get_connection

    conn = get_connection()
    try:
        row = conn.execute("SELECT status FROM forced_refresh_queue WHERE id = ?", (request_id,)).fetchone()
        return row["status"] if row is not None else None
    finally:
        conn.close()


def drain_queue() -> int:
    """Process all pending requests. Called by the scheduler thread ONLY (sole writer).

    Returns the number of requests processed. force=True so the requested source
    refreshes regardless of staleness.
    """
    from src.database import get_connection

    conn = get_connection()
    try:
        pending = conn.execute(
            "SELECT id, source FROM forced_refresh_queue WHERE status = 'pending' ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    processed = 0
    for row in pending:
        rid = row["id"]
        _set_status(rid, "running")
        try:
            _bootstrap_all_data(force=True)
            _set_status(rid, "done", completed=True)
        except Exception as exc:
            logger.warning("forced_refresh_queue: request %s failed: %s", rid, exc)
            _set_status(rid, "error", completed=True, detail=str(exc)[:300])
        processed += 1
    return processed


def _set_status(request_id: int, status: str, completed: bool = False, detail: str | None = None) -> None:
    from src.database import get_connection

    conn = get_connection()
    try:
        completed_at = datetime.now(UTC).isoformat() if completed else None
        conn.execute(
            "UPDATE forced_refresh_queue SET status = ?, detail = COALESCE(?, detail), "
            "completed_at = COALESCE(?, completed_at) WHERE id = ?",
            (status, detail, completed_at, request_id),
        )
        conn.commit()
    finally:
        conn.close()
