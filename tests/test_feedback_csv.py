"""feedback_csv exports the inbox as human-readable CSV (header always present)."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "feedback_csv.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _seed_feedback_row():
    from datetime import UTC, datetime

    from src.auth import approve_user, create_user, get_user
    from src.database import get_connection

    create_user("fb_amy", "pw")
    approve_user("fb_amy", team_name="Team Amy", approved_by="test")
    uid = get_user("fb_amy")["user_id"]
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO feedback (user_id, page, feature_tag, message, app_version, data_state, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (uid, "Trade Analyzer", "trade", "great tool", "v2.0", "{}", "new", datetime.now(UTC).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def test_feedback_csv_has_header_and_row(temp_db):
    from src.feedback import feedback_csv

    _seed_feedback_row()
    csv_text = feedback_csv()
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("id,created_at,username,team_name,page")
    assert "great tool" in csv_text
    assert "fb_amy" in csv_text
    # data_state must NOT be exported.
    assert "data_state" not in lines[0]


def test_feedback_csv_empty_inbox_is_header_only(temp_db):
    from src.feedback import feedback_csv

    lines = feedback_csv().strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("id,created_at,username,team_name,page")
