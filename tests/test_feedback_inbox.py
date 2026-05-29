"""Admin inbox helpers: list (joined with submitter), status, notes."""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "inbox.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _seed_user(username: str, team_name: str) -> int:
    from src.auth import approve_user, create_user, get_user

    create_user(username, "pw")
    approve_user(username, team_name=team_name, approved_by="test")
    return get_user(username)["user_id"]


def test_list_feedback_joins_username_and_team(temp_db):
    from src.feedback import list_feedback, submit_feedback

    uid = _seed_user("inbox_alice", "Team Alice")
    fid = submit_feedback(uid, "Trade Analyzer", "trade math looks off")
    match = [r for r in list_feedback() if r["id"] == fid]
    assert match, "submitted feedback must appear in the inbox"
    assert match[0]["username"] == "inbox_alice"
    assert match[0]["team_name"] == "Team Alice"


def test_list_feedback_filters_by_status(temp_db):
    from src.feedback import list_feedback, set_feedback_status, submit_feedback

    uid = _seed_user("inbox_bob", "Team Bob")
    fid = submit_feedback(uid, "Leaders", "needs a sort toggle")
    set_feedback_status(fid, "triaged")
    assert fid not in {r["id"] for r in list_feedback(status="new")}
    assert fid in {r["id"] for r in list_feedback(status="triaged")}


def test_set_feedback_status_rejects_invalid(temp_db):
    from src.feedback import set_feedback_status, submit_feedback

    uid = _seed_user("inbox_carol", "Team Carol")
    fid = submit_feedback(uid, "My Team", "x")
    with pytest.raises(ValueError):
        set_feedback_status(fid, "bogus")


def test_set_feedback_notes_persists(temp_db):
    from src.feedback import list_feedback, set_feedback_notes, submit_feedback

    uid = _seed_user("inbox_dave", "Team Dave")
    fid = submit_feedback(uid, "My Team", "y")
    set_feedback_notes(fid, "Investigated — projection cache was stale.")
    match = [r for r in list_feedback() if r["id"] == fid][0]
    assert match["admin_notes"] == "Investigated — projection cache was stale."
