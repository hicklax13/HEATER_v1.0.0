"""usage.py session-row creation + page-visit dwell tracking."""

import pytest

import src.usage as usage


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "usage_dwell.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def state(monkeypatch):
    s: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: s)
    monkeypatch.setattr(usage, "multi_user_enabled", lambda: True)
    return s


@pytest.fixture
def clock(monkeypatch):
    c = {"now": "2026-05-29T12:00:00+00:00"}
    monkeypatch.setattr(usage, "_now_iso", lambda: c["now"])
    return c


def _seed_user(username: str) -> int:
    from src.auth import approve_user, create_user, get_user

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    return get_user(username)["user_id"]


def _sessions() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM sessions").fetchall()]
    finally:
        conn.close()


def _visits() -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM page_visits ORDER BY id").fetchall()]
    finally:
        conn.close()


def test_first_view_creates_session_row(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_amy")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    rows = _sessions()
    assert len(rows) == 1
    assert rows[0]["user_id"] == uid


def test_page_visit_opened_on_first_view(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_ben")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    visits = _visits()
    assert len(visits) == 1
    assert visits[0]["page"] == "My Team"
    assert visits[0]["exit_at"] is None


def test_navigating_closes_prior_visit_with_dwell(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_cat")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    clock["now"] = "2026-05-29T12:00:30+00:00"
    usage.log_page_view("Leaders")
    visits = _visits()
    assert len(visits) == 2
    closed = visits[0]
    assert closed["page"] == "My Team"
    assert closed["exit_at"] == "2026-05-29T12:00:30+00:00"
    assert closed["dwell_seconds"] == 30.0


def test_repeat_same_page_does_not_open_new_visit(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_dan")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    usage.log_page_view("My Team")
    assert len(_visits()) == 1


def test_bump_activity_updates_last_activity(temp_db, state, clock, monkeypatch):
    uid = _seed_user("dwell_eve")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    clock["now"] = "2026-05-29T12:05:00+00:00"
    usage.bump_activity()
    assert _sessions()[0]["last_activity_at"] == "2026-05-29T12:05:00+00:00"


def test_session_row_noop_when_flag_off(temp_db, state, clock, monkeypatch):
    monkeypatch.setattr(usage, "multi_user_enabled", lambda: False)
    usage.log_page_view("My Team")
    assert _sessions() == []
