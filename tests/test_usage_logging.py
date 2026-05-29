"""log_page_view: one deduped event per (page, action) per session + last_seen bump."""

import pytest

import src.usage as usage


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "usage.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def state(monkeypatch):
    """A plain-dict stand-in for st.session_state + flag forced on."""
    s: dict = {}
    monkeypatch.setattr(usage, "_session_state", lambda: s)
    monkeypatch.setattr(usage, "multi_user_enabled", lambda: True)
    return s


def _seed_user(username: str) -> int:
    from src.auth import approve_user, create_user, get_user

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    return get_user(username)["user_id"]


def _events(user_id: int) -> list[dict]:
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM usage_events WHERE user_id = ? ORDER BY id", (user_id,)).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def test_log_inserts_one_event(temp_db, state, monkeypatch):
    uid = _seed_user("usage_amy")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    events = _events(uid)
    assert len(events) == 1
    assert events[0]["page"] == "My Team"
    assert events[0]["action"] == "view"
    assert events[0]["session_id"]


def test_log_dedups_same_page(temp_db, state, monkeypatch):
    uid = _seed_user("usage_ben")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("Leaders")
    usage.log_page_view("Leaders")
    usage.log_page_view("Leaders")
    assert len(_events(uid)) == 1


def test_log_distinct_pages(temp_db, state, monkeypatch):
    uid = _seed_user("usage_cat")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    usage.log_page_view("Leaders")
    assert {e["page"] for e in _events(uid)} == {"My Team", "Leaders"}


def test_session_id_stable_across_views(temp_db, state, monkeypatch):
    uid = _seed_user("usage_dan")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    usage.log_page_view("Leaders")
    assert len({e["session_id"] for e in _events(uid)}) == 1


def test_last_seen_at_bumped(temp_db, state, monkeypatch):
    uid = _seed_user("usage_eve")
    monkeypatch.setattr(usage, "current_user", lambda: {"user_id": uid})
    usage.log_page_view("My Team")
    from src.database import get_connection

    conn = get_connection()
    try:
        row = dict(conn.execute("SELECT last_seen_at FROM users WHERE user_id = ?", (uid,)).fetchone())
    finally:
        conn.close()
    assert row["last_seen_at"]
