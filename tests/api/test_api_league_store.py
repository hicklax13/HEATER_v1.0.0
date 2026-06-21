"""LeagueStore tests — DB-free in-memory fake + tmp-file sqlite (NEVER the live
draft_tool.db). Proves get_or_create_default is idempotent and api owns its own file."""

from api.stores.league_store import InMemoryLeagueStore, League, SqliteLeagueStore


def test_inmemory_default_league_is_idempotent():
    store = InMemoryLeagueStore()
    a = store.get_or_create_default()
    b = store.get_or_create_default()
    assert isinstance(a, League)
    assert a.id == b.id
    assert a.provider == "yahoo"


def test_inmemory_default_league_uses_overrides():
    store = InMemoryLeagueStore(provider="sleeper", external_league_id="abc", name="My League")
    lg = store.get_or_create_default()
    assert lg.provider == "sleeper"
    assert lg.external_league_id == "abc"
    assert lg.name == "My League"


def test_sqlite_default_league_idempotent_in_separate_file(tmp_path):
    db = tmp_path / "api_state.db"
    a = SqliteLeagueStore(db_path=str(db)).get_or_create_default()
    b = SqliteLeagueStore(db_path=str(db)).get_or_create_default()
    assert a.id == b.id
    assert db.exists()  # api owns its OWN file


def test_sqlite_get_by_id(tmp_path):
    store = SqliteLeagueStore(db_path=str(tmp_path / "api_state.db"))
    lg = store.get_or_create_default()
    assert store.get(lg.id).id == lg.id
    assert store.get(999) is None
