"""Regression guard: a partial Yahoo roster fetch must not wipe the cache.

2026-06-02 live incident. ``YahooFantasyClient.sync_to_db`` does
``clear_league_rosters()`` (``DELETE FROM league_rosters``) then re-inserts
whatever ``get_all_rosters()`` returned. The only guard was
``if not rosters_df.empty`` — which lets a *non-empty but incomplete* fetch
(e.g. Yahoo still reconnecting after a token re-paste, or rate-limited) replace
a full 12-team league with a partial one. The user's roster vanished
("No players on your roster yet") even though the league data was fine.

Fix: gate the destructive clear+reinsert on the fetch being *complete enough*
(``database.roster_fetch_is_complete``) — at least as many teams as the cache,
and not a large drop in total roster rows. The scheduler runs this same sync,
so an incomplete fetch could otherwise wipe rosters for every user.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database import (
    clear_league_rosters,
    load_league_rosters,
    roster_fetch_is_complete,
    upsert_league_roster_entry,
)
from src.yahoo_api import YahooFantasyClient

_TEAMS = ["Team Hickey", "Over the Rembow", "My Precious"]


def _seed_complete_league(players_per_team: int = 5) -> None:
    """Populate league_rosters with a known-complete 3-team league."""
    clear_league_rosters()
    for ti, team in enumerate(_TEAMS):
        for pid in range(1, players_per_team + 1):
            upsert_league_roster_entry(
                team_name=team,
                team_index=ti,
                player_id=ti * 100 + pid,
                roster_slot="BN",
            )


def _df(teams_with_counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for team, n in teams_with_counts.items():
        for i in range(n):
            rows.append(
                {
                    "team_name": team,
                    "team_key": f"k.{team}",
                    "player_name": f"{team}-{i}",
                    "editorial_team_abbr": "NYY",
                    "selected_position": "BN",
                }
            )
    return pd.DataFrame(rows)


# ── roster_fetch_is_complete (the decision) ──────────────────────────────


def test_partial_fetch_missing_a_team_is_incomplete():
    _seed_complete_league()
    try:
        # Yahoo returned only 2 of the 3 cached teams (Team Hickey missing).
        partial = _df({"Over the Rembow": 5, "My Precious": 5})
        assert roster_fetch_is_complete(partial) is False
    finally:
        clear_league_rosters()


def test_full_fetch_matching_cache_is_complete():
    _seed_complete_league()
    try:
        full = _df({t: 5 for t in _TEAMS})
        assert roster_fetch_is_complete(full) is True
    finally:
        clear_league_rosters()


def test_empty_cache_accepts_any_nonempty_fetch():
    """First-ever population: nothing to protect, so any real fetch is OK."""
    clear_league_rosters()
    assert load_league_rosters().empty
    assert roster_fetch_is_complete(_df({t: 5 for t in _TEAMS})) is True


def test_totally_empty_fetch_is_incomplete():
    _seed_complete_league()
    try:
        assert roster_fetch_is_complete(pd.DataFrame()) is False
    finally:
        clear_league_rosters()


def test_large_row_loss_same_teams_is_incomplete():
    """All teams present but most players missing → still a bad fetch."""
    _seed_complete_league(players_per_team=5)  # 15 rows cached
    try:
        # Same 3 teams, but only 1 player each → 3 rows (80%+ loss).
        shrunk = _df({t: 1 for t in _TEAMS})
        assert roster_fetch_is_complete(shrunk) is False
    finally:
        clear_league_rosters()


# ── sync_to_db wiring (the behavior) ─────────────────────────────────────


def test_sync_to_db_preserves_cache_on_partial_fetch(monkeypatch):
    """The live incident, end to end: a partial fetch must NOT wipe the cache."""
    _seed_complete_league()
    try:
        assert load_league_rosters()["team_name"].nunique() == 3

        client = YahooFantasyClient(league_id="12345", game_code="mlb", season=2026)
        partial = _df({"Over the Rembow": 5, "My Precious": 5})  # Team Hickey missing
        monkeypatch.setattr(client, "_ensure_auth", lambda: True)
        monkeypatch.setattr(client, "get_all_rosters", lambda: partial)
        monkeypatch.setattr(client, "get_league_standings", lambda: pd.DataFrame())
        monkeypatch.setattr(client, "_get_user_team_key", lambda: None)

        client.sync_to_db()

        after = load_league_rosters()
        assert "Team Hickey" in after["team_name"].values, (
            "partial fetch wiped the cached roster — the destructive clear ran without a completeness check"
        )
        assert after["team_name"].nunique() == 3
    finally:
        clear_league_rosters()
