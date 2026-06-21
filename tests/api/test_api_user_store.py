"""UserStore tests — DB-free (in-memory fake) + tmp-file (sqlite, never the live
draft_tool.db). Proves get_or_create is idempotent and that the sqlite impl owns
its OWN separate file."""

import logging
import sqlite3

import pytest

from api.stores.user_store import _CHAT_USER_ID_OFFSET, AppUser, InMemoryUserStore, SqliteUserStore


def test_chat_user_id_is_offset_and_collision_safe():
    # src/ai chat keys on Streamlit users.user_id (small ints) in draft_tool.db;
    # AppUser.chat_user_id must be disjoint so API chat never mixes with Streamlit chat.
    u = AppUser(id=7, clerk_user_id="x", created_at="now")
    assert u.chat_user_id == _CHAT_USER_ID_OFFSET + 7
    assert u.chat_user_id > 100_000  # clear of the live Streamlit user range


def test_chat_user_id_is_not_a_serialized_field():
    # Plain property, not a pydantic field → never in model_dump()/the openapi schema
    # (so adding it does not touch the snapshot the CEO/Bubba track owns during B1).
    assert "chat_user_id" not in AppUser(id=1, clerk_user_id="x", created_at="now").model_dump()


def test_inmemory_get_or_create_is_idempotent():
    store = InMemoryUserStore()
    a = store.get_or_create("user_1")
    b = store.get_or_create("user_1")
    assert isinstance(a, AppUser)
    assert a.id == b.id
    assert a.clerk_user_id == "user_1"


def test_inmemory_distinct_users_get_distinct_ids():
    store = InMemoryUserStore()
    assert store.get_or_create("user_1").id != store.get_or_create("user_2").id


def test_sqlite_store_idempotent_in_separate_file(tmp_path):
    db = tmp_path / "api_state.db"
    store = SqliteUserStore(db_path=str(db))
    a = store.get_or_create("user_x")
    b = store.get_or_create("user_x")
    assert a.id == b.id
    assert db.exists()  # api owns its OWN file


def test_sqlite_store_persists_across_instances(tmp_path):
    db = tmp_path / "api_state.db"
    a = SqliteUserStore(db_path=str(db)).get_or_create("user_y")
    b = SqliteUserStore(db_path=str(db)).get_or_create("user_y")
    assert a.id == b.id


def test_sqlite_store_distinct_users(tmp_path):
    db = tmp_path / "api_state.db"
    store = SqliteUserStore(db_path=str(db))
    assert store.get_or_create("u1").id != store.get_or_create("u2").id


def test_sqlite_store_logs_and_propagates_and_closes_on_failure(tmp_path, monkeypatch, caplog):
    # A failure during the SELECT/INSERT must PROPAGATE (never silently no-op),
    # leave a WARNING breadcrumb, and still close the connection (finally).
    store = SqliteUserStore(db_path=str(tmp_path / "x.db"))

    class _BoomConn:
        closed = False

        def execute(self, *a, **k):
            raise sqlite3.OperationalError("disk I/O error")

        def commit(self):  # pragma: no cover - not reached
            pass

        def close(self):
            self.closed = True

    boom = _BoomConn()
    monkeypatch.setattr(store, "_connect", lambda: boom)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(sqlite3.OperationalError):
            store.get_or_create("u")
    assert "get_or_create failed" in caplog.text  # operator breadcrumb
    assert boom.closed is True  # connection released despite the error
