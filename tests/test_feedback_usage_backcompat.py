"""With MULTI_USER off, usage + feedback hooks are inert (v1 byte-for-byte)."""

import pytest

import src.feedback as feedback
import src.usage as usage


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "backcompat.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _usage_count() -> int:
    from src.database import get_connection

    conn = get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM usage_events").fetchone()[0]
    finally:
        conn.close()


def test_log_page_view_noop_when_flag_off(temp_db, monkeypatch):
    # Even with a user present, flag-off must write nothing.
    state: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: state)
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": 1})
    usage.log_page_view("My Team")
    assert _usage_count() == 0
    assert "_usage_session_id" not in state


def test_render_feedback_widget_noop_when_flag_off(monkeypatch):
    monkeypatch.setattr(feedback, "current_user", lambda: {"user_id": 1})
    monkeypatch.setattr(feedback.st, "popover", lambda *a, **k: pytest.fail("popover called"))
    feedback.render_feedback_widget("My Team")  # must return without rendering
