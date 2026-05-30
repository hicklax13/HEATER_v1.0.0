"""With MULTI_USER off, the whole Plan 3 admin surface is inert (v1 byte-for-byte).

Consolidated back-compat guard. A connection spy proves the flag-off admin
helpers never open a DB connection, the five Plan 3 tables stay empty, and
app.py keeps the automatic-nav (v1) entry path instead of st.navigation().
Mirrors tests/test_feedback_usage_backcompat.py and tests/test_auth_backcompat.py.
"""

import ast
import pathlib
import sqlite3

import pytest

import src.app_settings as app_settings
import src.audit as audit
import src.feature_flags as feature_flags
import src.usage as usage

_TABLES = ("feature_flags", "audit_log", "app_settings", "sessions", "page_visits")


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


class _ConnSpy:
    """Wraps get_connection and counts how many times it is opened."""

    def __init__(self, real):
        self._real = real
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self._real()


@pytest.fixture
def conn_spy(temp_db, monkeypatch):
    # All admin helpers do a function-local `from src.database import get_connection`,
    # so patching the source symbol catches every open at call time.
    import src.database as database

    spy = _ConnSpy(database.get_connection)
    monkeypatch.setattr(database, "get_connection", spy)
    return spy


def _row_counts(db) -> dict:
    conn = sqlite3.connect(db)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in _TABLES}
    finally:
        conn.close()


def test_is_page_enabled_true_without_db(conn_spy):
    assert feature_flags.is_page_enabled("page:1_My_Team") is True
    feature_flags.require_page_enabled("page:1_My_Team")  # must not raise, must not touch DB
    assert conn_spy.calls == 0


def test_log_action_noop(conn_spy):
    audit.log_action(admin_id=1, action="toggle_flag", target="page:1_My_Team", detail={"enabled": False})
    assert conn_spy.calls == 0


def test_set_setting_noop(conn_spy):
    app_settings.set_setting("broadcast", "hello league", admin_id=1)
    assert conn_spy.calls == 0


def test_session_dwell_logger_noop(conn_spy, monkeypatch):
    state: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: state)
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": 1})
    usage.log_page_view("My Team")
    usage.bump_activity()
    assert conn_spy.calls == 0
    assert "_usage_session_id" not in state


def test_five_tables_stay_empty(conn_spy, temp_db, monkeypatch):
    state: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: state)
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": 1})
    feature_flags.is_page_enabled("page:1_My_Team")
    feature_flags.set_page_flag("page:1_My_Team", enabled=False, admin_id=1)
    audit.log_action(admin_id=1, action="toggle_flag", target="page:1_My_Team", detail={"enabled": False})
    app_settings.set_setting("broadcast", "hi", admin_id=1)
    usage.log_page_view("My Team")
    usage.bump_activity()
    assert conn_spy.calls == 0
    assert _row_counts(temp_db) == dict.fromkeys(_TABLES, 0)


def test_app_main_returns_before_navigation_when_flag_off():
    """main()'s flag-off branch early-returns before it can reach st.navigation()."""
    src = pathlib.Path("app.py").read_text(encoding="utf-8")
    main = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "main")
    guard_lines = [
        n.lineno
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "multi_user_enabled"
    ]
    nav_lines = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Attribute) and n.attr == "navigation"]
    return_lines = [n.lineno for n in ast.walk(main) if isinstance(n, ast.Return)]
    assert guard_lines, "main() must branch on multi_user_enabled()"
    assert nav_lines, "main() must call st.navigation() on the flag-on path"
    assert return_lines, "main() must early-return on the flag-off path"
    assert min(guard_lines) < min(nav_lines), "the flag guard must precede st.navigation()"
    assert any(min(guard_lines) <= r < min(nav_lines) for r in return_lines), (
        "an early return must sit between the flag guard and st.navigation()"
    )
