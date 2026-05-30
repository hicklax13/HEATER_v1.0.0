"""Smoke + guard tests for the admin usage-analytics page (nav-routed).

Mirrors test_admin_console_guarded.py: a real temp-DB user is seeded because
require_auth()/require_admin() re-validate the session identity against the DB
on every page load. The page omits set_page_config (nav owns it), so AppTest
can drive it directly.
"""

from pathlib import Path

import pytest

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "_admin_analytics.py"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "admin_analytics.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def test_admin_analytics_page_exists():
    assert _PAGE.exists(), "pages/_admin_analytics.py must exist"


def test_admin_analytics_smoke_renders_for_admin(temp_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.auth import ensure_bootstrap_admin

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("ADMIN_TEAM_NAME", "Team Hickey")
    ensure_bootstrap_admin()

    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "username": "connor",
        "status": "active",
        "is_admin": 1,
        "team_name": "Team Hickey",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert any("analytics" in m.value.lower() for m in at.title), [m.value for m in at.title]


def test_admin_analytics_smoke_blocks_non_admin(temp_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.auth import approve_user, create_user

    monkeypatch.setenv("MULTI_USER", "1")
    create_user("alice", "pw", display_name="Alice")
    approve_user("alice", team_name="Team Alice", approved_by="test")

    at = AppTest.from_file(str(_PAGE))
    at.session_state["auth_user"] = {
        "username": "alice",
        "status": "active",
        "is_admin": 0,
        "team_name": "Team Alice",
    }
    at.session_state["_auth_bootstrap_done"] = True
    at.run()
    assert any("access" in e.value.lower() for e in at.error), [e.value for e in at.error]
