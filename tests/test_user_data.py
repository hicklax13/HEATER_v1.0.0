"""TDD tests for src/user_data.py — watchlists + saved views.

Tables: user_watchlist, user_saved_views
All tests use an isolated tmp-path DB (monkeypatched DB_PATH) and run with
MULTI_USER off (default), exercising the local-user-0 fallback path.
"""

import json
import os

import pytest

# ── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_multiuser(monkeypatch):
    """Ensure MULTI_USER is off for all tests in this file."""
    monkeypatch.delenv("MULTI_USER", raising=False)


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db = tmp_path / "user_data_test.db"
    monkeypatch.setattr("src.database.DB_PATH", db)
    from src.database import init_db

    init_db()
    return db


@pytest.fixture
def ud(temp_db, monkeypatch):
    """Return the user_data module with a patched DB_PATH."""
    # Re-patch inside user_data's own import of get_connection so it uses tmp db
    monkeypatch.setattr("src.database.DB_PATH", temp_db)
    import importlib

    import src.user_data as m

    importlib.reload(m)
    return m


# ── table creation ────────────────────────────────────────────────────


def test_user_watchlist_table_created(temp_db):
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute("PRAGMA table_info(user_watchlist)").fetchall()
        cols = {r[1] for r in rows}
    finally:
        conn.close()
    assert {"id", "user_id", "player_id", "created_at"} <= cols


def test_user_saved_views_table_created(temp_db):
    from src.database import get_connection

    conn = get_connection()
    try:
        rows = conn.execute("PRAGMA table_info(user_saved_views)").fetchall()
        cols = {r[1] for r in rows}
    finally:
        conn.close()
    assert {"id", "user_id", "kind", "name", "payload_json", "created_at"} <= cols


def test_init_db_idempotent_for_user_data_tables(temp_db):
    """A second init_db() must not raise (CREATE TABLE IF NOT EXISTS)."""
    from src.database import init_db

    init_db()  # second call
    from src.database import get_connection

    conn = get_connection()
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(user_watchlist)").fetchall()}
    finally:
        conn.close()
    assert "player_id" in cols


# ── watchlist: add / get / remove ────────────────────────────────────


def test_get_watchlist_empty(ud):
    assert ud.get_watchlist() == set()


def test_add_to_watchlist(ud):
    ud.add_to_watchlist(42)
    assert ud.get_watchlist() == {42}


def test_add_multiple(ud):
    ud.add_to_watchlist(1)
    ud.add_to_watchlist(2)
    ud.add_to_watchlist(3)
    assert ud.get_watchlist() == {1, 2, 3}


def test_add_duplicate_is_idempotent(ud):
    ud.add_to_watchlist(7)
    ud.add_to_watchlist(7)  # second add — must not raise
    assert ud.get_watchlist() == {7}


def test_remove_from_watchlist(ud):
    ud.add_to_watchlist(10)
    ud.remove_from_watchlist(10)
    assert ud.get_watchlist() == set()


def test_remove_nonexistent_is_silent(ud):
    ud.remove_from_watchlist(999)  # must not raise


def test_is_watched_true(ud):
    ud.add_to_watchlist(5)
    assert ud.is_watched(5) is True


def test_is_watched_false(ud):
    assert ud.is_watched(5) is False


def test_toggle_watchlist_adds_when_absent(ud):
    new_state = ud.toggle_watchlist(88)
    assert new_state is True
    assert ud.is_watched(88)


def test_toggle_watchlist_removes_when_present(ud):
    ud.add_to_watchlist(88)
    new_state = ud.toggle_watchlist(88)
    assert new_state is False
    assert not ud.is_watched(88)


# ── saved views: save / load / list / delete ─────────────────────────


def test_load_view_missing_returns_none(ud):
    assert ud.load_view("lineup", "my_lineup") is None


def test_save_and_load_view_roundtrip(ud):
    payload = {"starters": [1, 2, 3], "bench": [4, 5]}
    ud.save_view("lineup", "Week 1", payload)
    loaded = ud.load_view("lineup", "Week 1")
    assert loaded == payload


def test_save_view_upserts_on_same_name(ud):
    ud.save_view("trade", "deal_a", {"gives": [1]})
    ud.save_view("trade", "deal_a", {"gives": [99]})  # update
    loaded = ud.load_view("trade", "deal_a")
    assert loaded == {"gives": [99]}


def test_list_views_returns_names_newest_first(ud):
    ud.save_view("lineup", "Alpha", {"x": 1})
    ud.save_view("lineup", "Beta", {"x": 2})
    ud.save_view("lineup", "Gamma", {"x": 3})
    names = ud.list_views("lineup")
    # newest first; exact insert order may vary but all three must be present
    assert set(names) == {"Alpha", "Beta", "Gamma"}
    assert names[0] == "Gamma"  # last saved = newest


def test_list_views_empty_kind(ud):
    ud.save_view("lineup", "A", {})
    assert ud.list_views("trade") == []


def test_delete_view(ud):
    ud.save_view("lineup", "Keeper", {"k": 1})
    ud.delete_view("lineup", "Keeper")
    assert ud.load_view("lineup", "Keeper") is None
    assert "Keeper" not in ud.list_views("lineup")


def test_delete_view_nonexistent_is_silent(ud):
    ud.delete_view("lineup", "ghost")  # must not raise


def test_save_view_json_payload_integrity(ud):
    """Complex nested payload survives the JSON round-trip."""
    payload = {
        "players": [{"id": 1, "name": "Acuna"}, {"id": 2, "name": "Freeman"}],
        "meta": {"week": 24, "score": 3.14},
    }
    ud.save_view("trade", "complex", payload)
    assert ud.load_view("trade", "complex") == payload


def test_kinds_are_isolated(ud):
    """A 'lineup' view and a 'trade' view with the same name don't collide."""
    ud.save_view("lineup", "shared_name", {"type": "lineup"})
    ud.save_view("trade", "shared_name", {"type": "trade"})
    assert ud.load_view("lineup", "shared_name") == {"type": "lineup"}
    assert ud.load_view("trade", "shared_name") == {"type": "trade"}


# ── graceful handling of corrupt DB data ─────────────────────────────


def test_load_view_bad_json_returns_none(ud, temp_db):
    """If the payload_json column is somehow corrupt, return None (no crash)."""
    from src.database import get_connection

    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO user_saved_views (user_id, kind, name, payload_json, created_at)"
            " VALUES (0, 'lineup', 'corrupt', 'NOT_JSON', '2026-01-01T00:00:00+00:00')"
        )
        conn.commit()
    finally:
        conn.close()
    assert ud.load_view("lineup", "corrupt") is None


# ── MULTI_USER off == local user 0, everything still works ───────────


def test_all_operations_work_without_multiuser(ud, monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    ud.add_to_watchlist(100)
    assert ud.is_watched(100)
    ud.save_view("lineup", "v1", {"a": 1})
    assert ud.load_view("lineup", "v1") == {"a": 1}
    assert ud.list_views("lineup") == ["v1"]
    ud.delete_view("lineup", "v1")
    assert ud.load_view("lineup", "v1") is None
