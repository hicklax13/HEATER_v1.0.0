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


def _seed_league_teams(names):
    """Insert league_teams rows (team_key + team_name is all the picker needs)."""
    from src.database import get_connection

    conn = get_connection()
    try:
        for i, name in enumerate(names):
            conn.execute(
                "INSERT INTO league_teams (team_key, team_name) VALUES (?, ?)",
                (f"469.l.1.t.{i + 1}", name),
            )
        conn.commit()
    finally:
        conn.close()


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


def test_active_reassign_preselects_current_team(temp_db, monkeypatch):
    """Fix 1: an active user whose team isn't in league_teams must still see
    their OWN team preselected in the reassign dropdown — not the first team in
    the list, which (because the Save guard is `new_team != current`) would arm a
    silent mis-assign. Reproduces production: the admin's ADMIN_TEAM_NAME team is
    not a row in league_teams."""
    from streamlit.testing.v1 import AppTest

    from src.auth import ensure_bootstrap_admin

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("ADMIN_TEAM_NAME", "Team Hickey")
    ensure_bootstrap_admin()
    # Deliberately EXCLUDES "Team Hickey"; alphabetical first is "BUBBA CROSBY".
    _seed_league_teams(["BUBBA CROSBY", "Zebra Squad"])

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

    reassign = at.selectbox(key="reassign_connor")
    assert reassign.value == "Team Hickey", reassign.value
    assert "Team Hickey" in reassign.options


def test_pending_picker_defaults_to_sentinel_and_refuses_blank_approve(temp_db, monkeypatch):
    """Fix 2: the pending-approval team picker defaults to a non-team sentinel,
    and Approve refuses (warns, no activation) until a real team is chosen — so an
    admin can't silently assign the first team by clicking through."""
    from streamlit.testing.v1 import AppTest

    from src.auth import create_user, ensure_bootstrap_admin, list_users

    monkeypatch.setenv("MULTI_USER", "1")
    monkeypatch.setenv("ADMIN_USERNAME", "connor")
    monkeypatch.setenv("ADMIN_PASSWORD", "pw")
    monkeypatch.setenv("ADMIN_TEAM_NAME", "Team Hickey")
    ensure_bootstrap_admin()
    _seed_league_teams(["BUBBA CROSBY", "Zebra Squad"])
    create_user("bob", "pw", display_name="Bob")  # a pending registrant

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

    # Default is the sentinel (first option), NOT a real team.
    pending_pick = at.selectbox(key="team_bob")
    assert pending_pick.value not in {"BUBBA CROSBY", "Zebra Squad"}, pending_pick.value
    assert pending_pick.value == pending_pick.options[0], "default must be the sentinel option"

    # Approving while the sentinel is still selected must NOT activate bob.
    at.button(key="approve_bob").click()
    at.run(timeout=60)
    assert [u for u in list_users(status="pending") if u["username"] == "bob"], (
        "approving with no team chosen must leave bob pending"
    )
    assert not [u for u in list_users(status="active") if u["username"] == "bob"]
    assert any("pick a team" in w.value.lower() for w in at.warning), [w.value for w in at.warning]
