"""request_refresh enqueues; the scheduler drains pending -> done (single-writer-safe)."""

import pytest

from src.database import get_connection, init_db


@pytest.fixture(autouse=True)
def _db():
    init_db()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM forced_refresh_queue")
        conn.commit()
    finally:
        conn.close()


def test_request_refresh_enqueues():
    from src.ai.refresh_queue import request_refresh

    rid = request_refresh("players", requested_by=99)
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM forced_refresh_queue WHERE id = ?", (rid,)).fetchone()
    finally:
        conn.close()
    assert row["source"] == "players"
    assert row["status"] == "pending"


def test_status_of():
    from src.ai.refresh_queue import request_refresh, status_of

    rid = request_refresh("all", requested_by=99)
    assert status_of(rid) == "pending"


def test_drain_runs_pending(monkeypatch):
    from src.ai import refresh_queue

    calls = {}

    def fake_bootstrap(**kwargs):
        calls["force"] = kwargs.get("force")
        return {"players": "Updated"}

    monkeypatch.setattr(refresh_queue, "_bootstrap_all_data", fake_bootstrap)

    rid = refresh_queue.request_refresh("players", requested_by=99)
    refresh_queue.drain_queue()
    assert calls["force"] is True
    assert refresh_queue.status_of(rid) == "done"


def test_drain_marks_error_on_exception(monkeypatch):
    from src.ai import refresh_queue

    def boom(**kwargs):
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(refresh_queue, "_bootstrap_all_data", boom)
    rid = refresh_queue.request_refresh("players", requested_by=99)
    refresh_queue.drain_queue()
    assert refresh_queue.status_of(rid) == "error"


def test_scheduler_calls_drain_queue():
    import inspect

    import src.scheduler as sched

    src = inspect.getsource(sched._refresh_once)
    assert "drain_queue" in src, "_refresh_once must drain the forced_refresh_queue"
