"""submit_feedback writes a row; _capture_data_state is best-effort JSON."""

import json

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "feedback.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _row(feedback_id):
    from src.database import get_connection

    conn = get_connection()
    try:
        return dict(conn.execute("SELECT * FROM feedback WHERE id = ?", (feedback_id,)).fetchone())
    finally:
        conn.close()


def test_submit_feedback_inserts_and_returns_id(temp_db):
    from src.feedback import submit_feedback
    from src.version import APP_VERSION

    fid = submit_feedback(user_id=1, page="My Team", message="Looks great", feature_tag="roster")
    assert isinstance(fid, int) and fid > 0
    row = _row(fid)
    assert row["user_id"] == 1
    assert row["page"] == "My Team"
    assert row["feature_tag"] == "roster"
    assert row["message"] == "Looks great"
    assert row["app_version"] == APP_VERSION
    assert row["status"] == "new"  # DB default
    assert row["created_at"]


def test_submit_feedback_allows_null_feature_tag(temp_db):
    from src.feedback import submit_feedback

    fid = submit_feedback(user_id=2, page="Leaders", message="No tag here")
    assert _row(fid)["feature_tag"] is None


def test_capture_data_state_returns_json(monkeypatch):
    monkeypatch.setattr(
        "src.database.get_refresh_log_snapshot",
        lambda: [{"source": "players", "status": "success"}],
    )
    from src.feedback import _capture_data_state

    captured = _capture_data_state()
    assert json.loads(captured) == [{"source": "players", "status": "success"}]


def test_capture_data_state_none_when_empty(monkeypatch):
    monkeypatch.setattr("src.database.get_refresh_log_snapshot", lambda: [])
    from src.feedback import _capture_data_state

    assert _capture_data_state() is None


def test_capture_data_state_swallows_errors(monkeypatch):
    def boom():
        raise RuntimeError("db down")

    monkeypatch.setattr("src.database.get_refresh_log_snapshot", boom)
    from src.feedback import _capture_data_state

    assert _capture_data_state() is None
