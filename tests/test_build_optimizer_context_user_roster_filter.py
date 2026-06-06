"""PR18: Pin build_optimizer_context's behavior for multi-team rosters
and MLB-level filtering. These were the root cause of bad FA recs in
production validation 2026-05-20."""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.optimizer.shared_data_layer import build_optimizer_context
from src.valuation import LeagueConfig


def _make_multi_team_rosters():
    """Build a fake 3-team roster DataFrame (10 players each)."""
    rows = []
    for team_idx, team in enumerate(["Team A", "Team B", "Team C"]):
        for i in range(10):
            rows.append(
                {
                    "team_name": team,
                    "player_id": team_idx * 10 + i + 1,
                    "name": f"{team} Player {i}",
                    "positions": "OF",
                    "status": "active",
                    "is_user_team": 1 if team == "Team A" else 0,
                    "is_hitter": 1,
                }
            )
    return pd.DataFrame(rows)


def test_multi_team_roster_filtered_to_user_team_when_team_name_passed():
    """When a multi-team rosters DataFrame is passed AND user_team_name
    is set, ctx.user_roster_ids must contain ONLY the user team's player
    IDs — not all 30 league IDs.

    Root cause of FA-engine bug 2026-05-20: ctx.user_roster_ids contained
    all 317 league players because build_optimizer_context didn't filter
    when a roster was passed."""
    rosters = _make_multi_team_rosters()
    config = LeagueConfig()
    yds = MagicMock()
    yds.get_rosters.return_value = rosters
    yds.get_free_agents.return_value = pd.DataFrame()
    yds.get_matchup.return_value = None

    ctx = build_optimizer_context(
        scope="rest_of_season",
        yds=yds,
        config=config,
        user_team_name="Team A",
        roster=rosters,  # ← multi-team DataFrame passed in
    )

    # MUST be filtered to Team A's 10 players, not all 30
    assert len(ctx.user_roster_ids) == 10, (
        f"Expected 10 player IDs (Team A only), got {len(ctx.user_roster_ids)}. "
        f"build_optimizer_context must filter multi-team rosters by user_team_name."
    )
    # All IDs must be in Team A's range (1-10)
    for pid in ctx.user_roster_ids:
        assert 1 <= pid <= 10, f"Found non-Team-A pid {pid} in user_roster_ids"


def test_single_team_roster_unchanged_when_no_team_name():
    """Backward-compat: when a single-team roster is passed without
    user_team_name, behavior is unchanged."""
    rosters = _make_multi_team_rosters()
    team_a_only = rosters[rosters["team_name"] == "Team A"]
    config = LeagueConfig()
    yds = MagicMock()
    yds.get_rosters.return_value = team_a_only
    yds.get_free_agents.return_value = pd.DataFrame()
    yds.get_matchup.return_value = None

    ctx = build_optimizer_context(
        scope="rest_of_season",
        yds=yds,
        config=config,
        roster=team_a_only,
    )
    assert len(ctx.user_roster_ids) == 10


def test_filter_no_op_when_team_name_already_unique():
    """When the passed roster is already single-team, the filter is a no-op."""
    rosters = _make_multi_team_rosters()
    team_a_only = rosters[rosters["team_name"] == "Team A"]
    config = LeagueConfig()
    yds = MagicMock()
    yds.get_rosters.return_value = team_a_only
    yds.get_free_agents.return_value = pd.DataFrame()
    yds.get_matchup.return_value = None

    ctx = build_optimizer_context(
        scope="rest_of_season",
        yds=yds,
        config=config,
        user_team_name="Team A",
        roster=team_a_only,
    )
    assert len(ctx.user_roster_ids) == 10


def test_f5_no_team_name_and_no_roster_does_not_use_all_teams():
    """F5 defense-in-depth (2026-06-05): with NO user_team_name AND no roster
    passed, build_optimizer_context must NOT fall back to the full multi-team
    rosters (which silently produced league-wide nonsense recommendations). It
    leaves the roster empty so downstream degrades to a 'no team' state."""
    rosters = _make_multi_team_rosters()  # 3 teams, 30 players
    config = LeagueConfig()
    yds = MagicMock()
    yds.get_rosters.return_value = rosters
    yds.get_free_agents.return_value = pd.DataFrame()
    yds.get_matchup.return_value = None

    ctx = build_optimizer_context(
        scope="rest_of_season",
        yds=yds,
        config=config,
        user_team_name=None,  # falsy → viewer's team cannot be identified
        roster=None,  # no explicit roster
    )
    assert ctx.roster.empty, (
        "F5: multi-team rosters + no user_team_name must NOT populate the roster "
        f"with all teams; got {len(ctx.roster)} rows."
    )
    assert ctx.user_roster_ids == []


def test_f5_single_team_fetched_still_used_without_team_name():
    """Backward-compat for the F5 guard: a single-team fetched roster is
    unambiguous, so it is still used even without a user_team_name."""
    rosters = _make_multi_team_rosters()
    team_a_only = rosters[rosters["team_name"] == "Team A"]
    config = LeagueConfig()
    yds = MagicMock()
    yds.get_rosters.return_value = team_a_only
    yds.get_free_agents.return_value = pd.DataFrame()
    yds.get_matchup.return_value = None

    ctx = build_optimizer_context(
        scope="rest_of_season",
        yds=yds,
        config=config,
        user_team_name=None,
        roster=None,
    )
    assert len(ctx.user_roster_ids) == 10
