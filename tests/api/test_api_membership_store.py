"""MembershipStore tests — DB-free in-memory fake + tmp-file sqlite. Proves
assign upserts (one row per user+league), lookups, and api-owned separate file."""

from api.stores.membership_store import InMemoryMembershipStore, SqliteMembershipStore, UserTeam


def test_inmemory_assign_and_get():
    store = InMemoryMembershipStore()
    m = store.assign(user_id=1, league_id=1, team_name="Team Hickey", team_key="t1", assigned_by=9)
    assert isinstance(m, UserTeam)
    got = store.get_for_user(user_id=1, league_id=1)
    assert got is not None and got.team_name == "Team Hickey" and got.assigned_by == 9


def test_inmemory_assign_is_upsert_per_user_league():
    store = InMemoryMembershipStore()
    store.assign(user_id=1, league_id=1, team_name="Old", team_key=None, assigned_by=9)
    store.assign(user_id=1, league_id=1, team_name="New", team_key=None, assigned_by=9)
    assert store.get_for_user(1, 1).team_name == "New"
    assert len(store.list_for_league(1)) == 1  # upsert, not duplicate


def test_inmemory_get_missing_returns_none():
    assert InMemoryMembershipStore().get_for_user(1, 1) is None


def test_sqlite_assign_upsert_separate_file(tmp_path):
    db = tmp_path / "api_state.db"
    store = SqliteMembershipStore(db_path=str(db))
    store.assign(user_id=1, league_id=1, team_name="A", team_key=None, assigned_by=9)
    store.assign(user_id=1, league_id=1, team_name="B", team_key=None, assigned_by=9)
    assert SqliteMembershipStore(db_path=str(db)).get_for_user(1, 1).team_name == "B"
    assert len(SqliteMembershipStore(db_path=str(db)).list_for_league(1)) == 1
    assert db.exists()


def test_sqlite_list_for_league(tmp_path):
    store = SqliteMembershipStore(db_path=str(tmp_path / "api_state.db"))
    store.assign(user_id=1, league_id=1, team_name="A", team_key=None, assigned_by=9)
    store.assign(user_id=2, league_id=1, team_name="B", team_key=None, assigned_by=9)
    assert {m.team_name for m in store.list_for_league(1)} == {"A", "B"}
