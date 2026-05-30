"""The admin console must be guarded by require_admin and render its queue.

The two structural tests (exists, calls_require_admin) are the load-bearing
guards. The two AppTest smoke tests seed a temp DB with real users because
require_auth() re-validates the session identity against the DB on every page
load — a session-only stub would be logged straight back out to the login
screen, never reaching the require_admin hard-stop we want to assert.
"""

from pathlib import Path

import pytest

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "_admin_console.py"


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    """A fresh, isolated SQLite DB so smoke tests never touch the real one."""
    db = tmp_path / "admin_console.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def test_admin_console_exists():
    assert _PAGE.exists(), "pages/_admin_console.py must exist"


def test_admin_console_calls_require_admin():
    src = _PAGE.read_text(encoding="utf-8")
    assert "from src.auth import" in src
    assert "require_admin" in src
    # The guard must run before any lifecycle action is offered. Compare the
    # require_admin() *call* against the first approve_user( *call* — matching
    # the bare name "approve_user" would collide with the import line, which
    # always precedes the guard call.
    i_guard = src.index("require_admin()")
    i_approve = src.index("approve_user(")
    assert i_guard < i_approve, "require_admin() must gate the page before approve actions"


def test_admin_console_smoke_blocks_non_admin(temp_db, monkeypatch):
    """AppTest: a logged-in non-admin hits the require_admin hard-stop."""
    from streamlit.testing.v1 import AppTest

    from src.auth import approve_user, create_user

    monkeypatch.setenv("MULTI_USER", "1")
    # Seed a real active non-admin so require_auth's DB re-validation passes
    # and execution reaches the require_admin is_admin check.
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
    at.run(timeout=60)
    # require_admin() calls st.error + st.stop → an error is shown, page halts.
    assert any("access" in e.value.lower() for e in at.error), [e.value for e in at.error]


def test_admin_console_smoke_renders_for_admin(temp_db, monkeypatch):
    from streamlit.testing.v1 import AppTest

    from src.auth import ensure_bootstrap_admin

    monkeypatch.setenv("MULTI_USER", "1")
    # Seed a real active admin via the bootstrap path (is_admin=1).
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
    at.run(timeout=60)
    assert not at.exception, [str(e) for e in at.exception]
    # The page title should render for an admin.
    assert any("admin" in m.value.lower() for m in at.title), [m.value for m in at.title]
