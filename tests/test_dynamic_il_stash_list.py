"""PR7 (FA engine overhaul): IL_STASH_NAMES must derive dynamically
from league_rosters, not be hardcoded. The hardcoded list {Bieber,
Strider} went stale immediately — every week players move on and off IL
and the hardcoded list misses them."""

import pytest

from src.database import get_connection, init_db


@pytest.fixture
def db_with_il_players():
    """Seed league_rosters with 3 IL players + 1 active player.

    Creates 4 test players (if not already present) and 4 league_rosters
    entries under team_name 'TestDynIL'. Cleans up both tables on teardown
    so production data is not polluted.
    """
    init_db()
    conn = get_connection()
    try:
        cur = conn.cursor()
        # Clean stale test rows from previous runs (idempotent).
        cur.execute("DELETE FROM league_rosters WHERE team_name = 'TestDynIL'")

        # Use deterministic test-player names — re-use existing rows if a
        # prior run left them behind; otherwise insert. This keeps the
        # fixture robust against the empty-DB worktree case AND production
        # data without depending on whichever 4 players happen to sort first.
        test_player_names = [
            "PR7TestIL10Player",
            "PR7TestIL15Player",
            "PR7TestIL60Player",
            "PR7TestActivePlayer",
        ]
        rows: list[tuple[int, str]] = []
        for name in test_player_names:
            cur.execute(
                "SELECT player_id, name FROM players WHERE name = ? COLLATE NOCASE",
                (name,),
            )
            existing = cur.fetchone()
            if existing:
                rows.append((existing[0], existing[1]))
            else:
                cur.execute(
                    "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, 'TST', '3B', 1)",
                    (name,),
                )
                rows.append((cur.lastrowid, name))

        statuses = ["IL10", "IL15", "IL60", "active"]
        for (pid, _name), status in zip(rows, statuses):
            cur.execute(
                "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, "
                "is_user_team, status, selected_position) "
                "VALUES (?, 1, ?, '3B', 0, ?, '3B')",
                ("TestDynIL", pid, status),
            )
        conn.commit()
        yield rows, statuses
        # Cleanup — remove rosters first (FK), then players we inserted.
        cur.execute("DELETE FROM league_rosters WHERE team_name = 'TestDynIL'")
        cur.execute(
            "DELETE FROM players WHERE name IN ({})".format(",".join("?" * len(test_player_names))),
            tuple(test_player_names),
        )
        conn.commit()
    finally:
        conn.close()


def test_il_stash_names_derived_from_league_rosters(db_with_il_players):
    """IL_STASH_NAMES should include all 3 IL players from league_rosters."""
    rows, statuses = db_with_il_players
    il_names_expected = {rows[0][1], rows[1][1], rows[2][1]}  # first 3 are IL
    active_name = rows[3][1]  # 4th is active

    from src.alerts import get_il_stash_names

    derived = get_il_stash_names()
    # All IL players must be in the set
    for name in il_names_expected:
        assert name in derived, f"Expected IL player '{name}' in derived IL_STASH_NAMES"
    # Active player must NOT be in the set (just from our test data)
    # Note: there may be other IL players in DB beyond our test setup.
    assert active_name not in derived, f"Active player '{active_name}' should not be in IL_STASH_NAMES"


def test_get_il_stash_names_returns_set():
    """Function returns a set of strings."""
    from src.alerts import get_il_stash_names

    result = get_il_stash_names()
    assert isinstance(result, set)
    for name in result:
        assert isinstance(name, str)


def test_il_stash_names_backward_compat_import(db_with_il_players):
    """The legacy `IL_STASH_NAMES` import must still work — many callers
    use it (pages/14_Free_Agents.py imports it at line 8). It now returns
    the dynamic set."""
    from src.alerts import IL_STASH_NAMES

    # Should be a set (or equivalent), backward-compat
    assert hasattr(IL_STASH_NAMES, "__contains__")
