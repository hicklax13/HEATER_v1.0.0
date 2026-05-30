"""AppTest smoke: the admin Feedback tab renders submitted feedback for an admin.

Uses a temp DB (DB_PATH monkeypatch) + a real bootstrap admin because
require_auth() re-validates the session identity against the DB on every page
load — a session-only stub would be logged straight back out.
"""

import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "admin_feedback.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def test_feedback_tab_renders_message_for_admin(temp_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.auth import approve_user, create_user, ensure_bootstrap_admin, get_user
    from src.feedback import submit_feedback

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("ADMIN_TEAM_NAME", "Team Hickey")
    ensure_bootstrap_admin()

    # A non-admin submits feedback that the admin should see in the inbox.
    create_user("submitter", "pw", display_name="Sam")
    approve_user("submitter", team_name="Team Sam", approved_by="test")
    submit_feedback(get_user("submitter")["user_id"], "Trade Analyzer", "ZNVECTOR sentinel message")

    at = AppTest.from_file("pages/_admin_console.py")
    at.session_state["auth_user"] = {
        "username": "connor",
        "status": "active",
        "is_admin": 1,
        "team_name": "Team Hickey",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()

    assert not at.exception, [str(e) for e in at.exception]
    # The submitted message must appear somewhere in the rendered markdown/text.
    blobs = [m.value for m in at.markdown] + [t.value for t in at.text]
    assert any("ZNVECTOR sentinel message" in b for b in blobs), blobs
