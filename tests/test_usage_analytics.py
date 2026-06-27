"""usage.py analytics readers: DAU, top pages, per-user, sessions/dwell, CSV, lazy-close."""

import pytest

import src.usage as usage


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "usage_analytics.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


def _seed_user(username: str) -> int:
    from src.auth import approve_user, create_user, get_user

    create_user(username, "pw")
    approve_user(username, team_name="Team " + username, approved_by="test")
    return get_user(username)["user_id"]


def _event(uid, page, created_at, action="view", session_id="s1"):
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO usage_events (user_id, page, action, session_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (uid, page, action, session_id, created_at),
        )
        conn.commit()
    finally:
        conn.close()


def test_dau_series_counts_distinct_users_per_day(temp_db):
    from datetime import UTC, datetime, timedelta

    # dau_series filters by a `now - days` cutoff, so the seeded events must be dated
    # RELATIVE to today (two distinct recent days, safely inside the 30-day window) —
    # hardcoded calendar dates turn this into a time-bomb once they age past the window.
    today = datetime.now(UTC).date()
    d1 = (today - timedelta(days=2)).isoformat()
    d2 = (today - timedelta(days=1)).isoformat()
    amy = _seed_user("an_amy")
    ben = _seed_user("an_ben")
    _event(amy, "My Team", f"{d1}T10:00:00+00:00")
    _event(ben, "Leaders", f"{d1}T11:00:00+00:00")
    _event(amy, "Leaders", f"{d2}T09:00:00+00:00")
    series = usage.dau_series(days=30)
    by_day = {r["day"]: r["users"] for r in series}
    assert by_day[d1] == 2
    assert by_day[d2] == 1


def test_most_used_pages_orders_by_views(temp_db):
    from datetime import UTC, datetime, timedelta

    # most_used_pages also filters by a `now - days` cutoff — date events relative to today
    # (yesterday is safely inside the 30-day window) instead of a hardcoded calendar date.
    day = (datetime.now(UTC).date() - timedelta(days=1)).isoformat()
    amy = _seed_user("an_cat")
    _event(amy, "Leaders", f"{day}T09:00:00+00:00")
    _event(amy, "Leaders", f"{day}T09:05:00+00:00", session_id="s2")
    _event(amy, "My Team", f"{day}T09:10:00+00:00")
    rows = usage.most_used_pages(days=30)
    assert rows[0]["page"] == "Leaders"
    assert rows[0]["views"] == 2


def test_per_user_activity_keeps_zero_event_users(temp_db):
    _seed_user("an_dan")  # no events
    rows = usage.per_user_activity()
    dan = [r for r in rows if r["username"] == "an_dan"][0]
    assert dan["events"] == 0


def test_session_timeline_has_duration(temp_db):
    uid = _seed_user("an_eve")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, login_at, last_activity_at) VALUES (?, ?, ?, ?)",
            ("sess1", uid, "2026-05-29T12:00:00+00:00", "2026-05-29T12:10:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    rows = usage.session_timeline()
    assert rows[0]["duration_seconds"] == 600.0


def test_page_dwell_summary_aggregates(temp_db):
    uid = _seed_user("an_fay")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.executemany(
            "INSERT INTO page_visits (session_id, user_id, page, enter_at, exit_at, dwell_seconds) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("s1", uid, "My Team", "2026-05-29T12:00:00+00:00", "2026-05-29T12:00:10+00:00", 10.0),
                ("s1", uid, "My Team", "2026-05-29T12:01:00+00:00", "2026-05-29T12:01:30+00:00", 30.0),
            ],
        )
        conn.commit()
    finally:
        conn.close()
    rows = usage.page_dwell_summary()
    mt = [r for r in rows if r["page"] == "My Team"][0]
    assert mt["total_seconds"] == 40.0
    assert mt["visits"] == 2
    assert mt["avg_seconds"] == 20.0


def test_last_seen_summary(temp_db):
    uid = _seed_user("an_gus")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute("UPDATE users SET last_seen_at = ? WHERE user_id = ?", ("2026-05-29T12:00:00+00:00", uid))
        conn.commit()
    finally:
        conn.close()
    rows = usage.last_seen_summary()
    gus = [r for r in rows if r["username"] == "an_gus"][0]
    assert gus["last_seen"] == "2026-05-29T12:00:00+00:00"


def test_usage_csv_roundtrip(temp_db):
    amy = _seed_user("an_hal")
    _event(amy, "My Team", "2026-05-29T09:00:00+00:00")
    csv_text = usage.usage_csv()
    lines = csv_text.strip().splitlines()
    assert lines[0].split(",")[:4] == ["id", "created_at", "user_id", "username"]
    assert "My Team" in csv_text
    assert "an_hal" in csv_text


def test_lazy_close_resolves_open_visits(temp_db):
    uid = _seed_user("an_ivy")
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, login_at, last_activity_at) VALUES (?, ?, ?, ?)",
            ("os1", uid, "2026-05-29T12:00:00+00:00", "2026-05-29T12:05:00+00:00"),
        )
        conn.execute(
            "INSERT INTO page_visits (session_id, user_id, page, enter_at, exit_at, dwell_seconds) "
            "VALUES (?, ?, ?, ?, NULL, NULL)",
            ("os1", uid, "My Team", "2026-05-29T12:00:00+00:00"),
        )
        conn.commit()
    finally:
        conn.close()
    rows = usage.page_dwell_summary()
    mt = [r for r in rows if r["page"] == "My Team"][0]
    assert mt["total_seconds"] == 300.0
