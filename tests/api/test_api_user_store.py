"""UserStore tests — DB-free (in-memory fake) + tmp-file (sqlite, never the live
draft_tool.db). Proves get_or_create is idempotent and that the sqlite impl owns
its OWN separate file."""

from api.stores.user_store import AppUser, InMemoryUserStore, SqliteUserStore


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
