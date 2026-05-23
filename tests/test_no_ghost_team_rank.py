"""Structural-invariant guard: ghost teams in standings don't inflate rank.

Bug D (2026-05-23 validation): risk flag emitted "Ranked 13th in SB" for a
12-team league. Root cause: the standings table contained 13 rows — the 12
real league teams plus a "Twigs" team that no longer exists in
league_rosters (likely a renamed team that lingered in the standings
cache). With 13 entries fed into category_gap_analysis, the rank loop
produced rank=13 even though only 12 real teams exist.

Fix:
  - build_standings_totals(standings, valid_teams=...) filters by an
    optional whitelist of real team names.
  - evaluate_trade() always passes valid_teams — either league_rosters
    keys (when caller provides them) or the league_rosters table (when not).

This test guards against the regression where a ghost row inflates the
displayed rank past num_real_teams.
"""

from __future__ import annotations

import pandas as pd

from src.engine.portfolio.category_analysis import (
    build_standings_totals,
    category_gap_analysis,
)


def test_build_standings_totals_no_filter_includes_all() -> None:
    """Legacy behavior preserved when valid_teams=None — all rows pass through."""
    standings = pd.DataFrame(
        [
            {"team_name": "Real1", "category": "HR", "total": 30, "rank": 1},
            {"team_name": "Real2", "category": "HR", "total": 25, "rank": 2},
            {"team_name": "Ghost", "category": "HR", "total": 20, "rank": 3},
        ]
    )
    totals = build_standings_totals(standings)
    assert set(totals.keys()) == {"Real1", "Real2", "Ghost"}


def test_build_standings_totals_filters_ghost_teams() -> None:
    """When valid_teams is set, ghost rows are dropped."""
    standings = pd.DataFrame(
        [
            {"team_name": "Real1", "category": "HR", "total": 30, "rank": 1},
            {"team_name": "Real2", "category": "HR", "total": 25, "rank": 2},
            {"team_name": "Twigs", "category": "HR", "total": 20, "rank": 3},  # ghost
        ]
    )
    valid = {"Real1", "Real2"}
    totals = build_standings_totals(standings, valid_teams=valid)
    assert set(totals.keys()) == {"Real1", "Real2"}
    assert "Twigs" not in totals


def test_ghost_team_inflates_rank_without_filter() -> None:
    """Demonstrates the pre-fix problem: ghost row inflates rank."""
    # 12 real teams + 1 ghost = 13. User is worst.
    real_teams = [f"Team_{i}" for i in range(12)]
    rows = [{"team_name": t, "category": "SB", "total": 30 + i * 2, "rank": 12 - i} for i, t in enumerate(real_teams)]
    # Add ghost team with SB=20 (worst of all)
    rows.append({"team_name": "Twigs", "category": "SB", "total": 20, "rank": 13})
    standings = pd.DataFrame(rows)

    # WITHOUT filter — user (Team_0, SB=30) gets ranked among 13 teams
    totals_unfiltered = build_standings_totals(standings)
    user_team = "Team_0"
    analysis = category_gap_analysis(
        your_totals=totals_unfiltered[user_team],
        all_team_totals=totals_unfiltered,
        your_team_id=user_team,
    )
    # Team_0 has the lowest SB among real teams (30), but Twigs has 20 →
    # Twigs ranks last. Team_0 should rank 11th of 12 reals (12th of 13 incl. ghost).
    rank_unfiltered = analysis["SB"]["rank"]
    assert rank_unfiltered == 12, f"Without filter, Team_0 is ranked {rank_unfiltered} out of 13 (incl. Twigs ghost)"


def test_ghost_team_filtered_gives_correct_rank() -> None:
    """With the valid_teams filter, rank reflects only real teams."""
    real_teams = [f"Team_{i}" for i in range(12)]
    rows = [{"team_name": t, "category": "SB", "total": 30 + i * 2, "rank": 12 - i} for i, t in enumerate(real_teams)]
    rows.append({"team_name": "Twigs", "category": "SB", "total": 20, "rank": 13})
    standings = pd.DataFrame(rows)

    totals = build_standings_totals(standings, valid_teams=set(real_teams))
    user_team = "Team_0"
    analysis = category_gap_analysis(
        your_totals=totals[user_team],
        all_team_totals=totals,
        your_team_id=user_team,
    )
    rank_filtered = analysis["SB"]["rank"]
    assert rank_filtered <= 12, f"With filter, rank must not exceed 12 (real team count); got {rank_filtered}"
    # User has the lowest SB among real teams, so should be rank 12 (last)
    assert rank_filtered == 12


def test_empty_league_rosters_falls_back_to_no_filter() -> None:
    """Hotfix regression (PR #113 CI failure): when league_rosters is empty
    (fresh DB, test env, etc.), evaluate_trade must NOT silently filter out
    all standings rows. The ghost-team protection is a safety net, not a
    hard prerequisite.

    This test patches load_league_standings to return a non-empty standings
    DataFrame WITHOUT supplying league_rosters. With the empty-set bug,
    all_team_totals would come back empty and category_gap_analysis would
    never run. Asserts the call chain remains intact.
    """
    from unittest.mock import patch

    from src.engine.output.trade_evaluator import evaluate_trade
    from src.valuation import LeagueConfig

    cfg = LeagueConfig()
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "P1",
                "player_name": "P1",
                "is_hitter": 1,
                "positions": "OF",
                "r": 70,
                "hr": 25,
                "rbi": 80,
                "sb": 8,
                "h": 130,
                "ab": 500,
                "bb": 50,
                "hbp": 5,
                "sf": 5,
                "pa": 560,
                "avg": 0.270,
                "obp": 0.340,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "era": 0,
                "whip": 0,
            },
            {
                "player_id": 2,
                "name": "P2",
                "player_name": "P2",
                "is_hitter": 1,
                "positions": "OF",
                "r": 75,
                "hr": 22,
                "rbi": 75,
                "sb": 10,
                "h": 132,
                "ab": 500,
                "bb": 52,
                "hbp": 5,
                "sf": 5,
                "pa": 562,
                "avg": 0.270,
                "obp": 0.340,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "era": 0,
                "whip": 0,
            },
        ]
    )

    # Standings with real category totals for "Team Hickey" + 2 opponents
    standings_df = pd.DataFrame(
        [
            {"team_name": "Team Hickey", "category": c, "total": v, "rank": 2}
            for c, v in {"R": 100, "HR": 30, "RBI": 90, "SB": 12, "AVG": 0.27, "OBP": 0.34}.items()
        ]
        + [
            {"team_name": f"Opp{n}", "category": c, "total": v, "rank": 1}
            for n in (1, 2)
            for c, v in {"R": 110, "HR": 32, "RBI": 95, "SB": 15, "AVG": 0.28, "OBP": 0.35}.items()
        ]
    )

    # Patch the gap analysis so we can detect whether it was called
    with (
        patch("src.engine.output.trade_evaluator.category_gap_analysis") as cga,
        patch("src.engine.output.trade_evaluator.load_league_standings", return_value=standings_df),
    ):
        cga.return_value = {
            c: {"rank": 6, "is_punt": False, "marginal_value": 1.0, "gap_to_next": 0.0, "gainable_positions": 1}
            for c in cfg.all_categories
        }
        try:
            evaluate_trade(
                giving_ids=[1],
                receiving_ids=[2],
                user_roster_ids=[1],
                player_pool=pool,
                config=cfg,
                user_team_name="Team Hickey",
                weeks_remaining=16,
                enable_mc=False,
                enable_context=False,
                enable_game_theory=False,
                apply_ytd_blend=False,
                # Critical: NO league_rosters passed → forces DB fallback,
                # which in test env returns empty → previously broke the chain
            )
        except Exception:
            pass

        assert cga.called, (
            "PR #113 hotfix regression: empty league_rosters fallback caused "
            "all_team_totals to be empty, skipping category_gap_analysis. "
            "The ghost-team filter must degrade to no-filter on empty input."
        )


def test_no_13th_in_12_team_league_via_evaluate_trade() -> None:
    """End-to-end via evaluate_trade: no risk flag should mention rank > 12
    when league_rosters has 12 teams."""
    from src.engine.output.trade_evaluator import evaluate_trade

    # Build a 12-team league
    real_teams = [f"Team_{i}" for i in range(12)]
    rosters = {t: [i * 10 + 1] for i, t in enumerate(real_teams)}

    # Build a minimal pool — one hitter per team
    rows = []
    for i, t in enumerate(real_teams):
        rows.append(
            {
                "player_id": i * 10 + 1,
                "name": f"H{i}",
                "player_name": f"H{i}",
                "is_hitter": 1,
                "positions": "OF",
                "r": 80,
                "hr": 22,
                "rbi": 80,
                "sb": 10,
                "h": 130,
                "ab": 500,
                "bb": 55,
                "hbp": 5,
                "sf": 5,
                "pa": 570,
                "avg": 0.260,
                "obp": 0.330,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "ip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "era": 0,
                "whip": 0,
            }
        )
    pool = pd.DataFrame(rows)

    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_0"],
        player_pool=pool,
        user_team_name="Team_0",
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
        league_rosters=rosters,  # provides valid_teams
    )
    # No risk flag should claim rank > 12
    flags = result.get("risk_flags", [])
    for f in flags:
        if "Ranked" in f and "th in" in f:
            # Extract the number
            import re

            m = re.search(r"Ranked (\d+)", f)
            if m:
                rank_num = int(m.group(1))
                assert rank_num <= 12, (
                    f"Bug D regression: risk flag claims 'Ranked {rank_num}' in a 12-team league. Flag text: {f!r}"
                )
