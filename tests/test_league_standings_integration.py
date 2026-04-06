"""Integration tests for League Standings + Matchup Planner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.database import init_db
from src.valuation import LeagueConfig


@pytest.fixture(autouse=True)
def _fresh_db():
    init_db()


class TestEndToEndSimulation:
    """Test the full simulation pipeline with realistic data."""

    def test_full_pipeline(self):
        """Run category probs + season sim + magic numbers end-to-end."""
        from src.standings_engine import (
            compute_category_win_probabilities,
            compute_magic_numbers,
            simulate_season_enhanced,
        )

        config = LeagueConfig()
        n_teams = 4

        # Build minimal player pool
        rows = []
        pid = 1
        for team_idx in range(n_teams):
            for _ in range(5):  # 5 hitters per team
                rows.append(
                    {
                        "player_id": pid,
                        "name": f"H{pid}",
                        "team": f"T{team_idx}",
                        "positions": "OF",
                        "is_hitter": 1,
                        "is_injured": 0,
                        "pa": 600,
                        "ab": 550,
                        "h": 140 + team_idx * 10,
                        "r": 70 + team_idx * 5,
                        "hr": 20 + team_idx * 3,
                        "rbi": 75 + team_idx * 5,
                        "sb": 8 + team_idx * 2,
                        "avg": 0.255 + team_idx * 0.01,
                        "obp": 0.320 + team_idx * 0.01,
                        "bb": 50,
                        "hbp": 5,
                        "sf": 5,
                        "ip": 0,
                        "w": 0,
                        "l": 0,
                        "sv": 0,
                        "k": 0,
                        "era": 0,
                        "whip": 0,
                        "er": 0,
                        "bb_allowed": 0,
                        "h_allowed": 0,
                        "adp": 50 + pid,
                    }
                )
                pid += 1
            for _ in range(3):  # 3 pitchers per team
                rows.append(
                    {
                        "player_id": pid,
                        "name": f"P{pid}",
                        "team": f"T{team_idx}",
                        "positions": "SP",
                        "is_hitter": 0,
                        "is_injured": 0,
                        "pa": 0,
                        "ab": 0,
                        "h": 0,
                        "r": 0,
                        "hr": 0,
                        "rbi": 0,
                        "sb": 0,
                        "avg": 0,
                        "obp": 0,
                        "bb": 0,
                        "hbp": 0,
                        "sf": 0,
                        "ip": 180,
                        "w": 10 + team_idx,
                        "l": 10 - team_idx,
                        "sv": 2 * team_idx,
                        "k": 150 + team_idx * 10,
                        "era": 4.0 - team_idx * 0.3,
                        "whip": 1.3 - team_idx * 0.05,
                        "er": 80 - team_idx * 5,
                        "bb_allowed": 55,
                        "h_allowed": 170 - team_idx * 5,
                        "adp": 50 + pid,
                    }
                )
                pid += 1

        pool = pd.DataFrame(rows)
        teams = [f"Team {chr(65 + i)}" for i in range(n_teams)]
        roster_ids = {}
        for i, team in enumerate(teams):
            start = i * 8 + 1
            roster_ids[team] = list(range(start, start + 8))

        # Category win probabilities
        probs = compute_category_win_probabilities(roster_ids[teams[0]], roster_ids[teams[3]], pool, config)
        assert 0.0 <= probs["overall_win_pct"] <= 1.0
        assert len(probs["categories"]) == 12

        # Season simulation
        current = {t: {"W": 5, "L": 5, "T": 0} for t in teams}
        schedule = {w: [(teams[0], teams[1]), (teams[2], teams[3])] for w in range(6, 15)}

        from src.standings_engine import _estimate_team_weekly_stats

        team_totals = {}
        for team in teams:
            team_totals[team] = _estimate_team_weekly_stats(roster_ids[team], pool, config, weeks_remaining=9)

        sim = simulate_season_enhanced(current, team_totals, schedule, current_week=6, n_sims=50, seed=42)
        assert set(sim.keys()) == {
            "projected_records",
            "playoff_probability",
            "confidence_intervals",
            "strength_of_schedule",
        }

        # Magic numbers
        wins = {t: int(sim["projected_records"][t]["W"]) for t in teams}
        magic = compute_magic_numbers(wins, remaining_matchups=5, playoff_spots=2)
        assert all(t in magic for t in teams)

    def test_category_probs_all_categories_present(self):
        """Verify all 12 categories appear in output."""
        from src.standings_engine import compute_category_win_probabilities

        config = LeagueConfig()
        all_cats = config.hitting_categories + config.pitching_categories

        rows = []
        pid = 1
        for _ in range(2):  # 2 teams
            for _ in range(5):
                rows.append(
                    {
                        "player_id": pid,
                        "name": f"H{pid}",
                        "team": "TST",
                        "positions": "OF",
                        "is_hitter": 1,
                        "is_injured": 0,
                        "pa": 600,
                        "ab": 550,
                        "h": 150,
                        "r": 80,
                        "hr": 25,
                        "rbi": 85,
                        "sb": 10,
                        "avg": 0.273,
                        "obp": 0.340,
                        "bb": 50,
                        "hbp": 5,
                        "sf": 5,
                        "ip": 0,
                        "w": 0,
                        "l": 0,
                        "sv": 0,
                        "k": 0,
                        "era": 0,
                        "whip": 0,
                        "er": 0,
                        "bb_allowed": 0,
                        "h_allowed": 0,
                        "adp": 100,
                    }
                )
                pid += 1
            for _ in range(3):
                rows.append(
                    {
                        "player_id": pid,
                        "name": f"P{pid}",
                        "team": "TST",
                        "positions": "SP",
                        "is_hitter": 0,
                        "is_injured": 0,
                        "pa": 0,
                        "ab": 0,
                        "h": 0,
                        "r": 0,
                        "hr": 0,
                        "rbi": 0,
                        "sb": 0,
                        "avg": 0,
                        "obp": 0,
                        "bb": 0,
                        "hbp": 0,
                        "sf": 0,
                        "ip": 180,
                        "w": 12,
                        "l": 8,
                        "sv": 0,
                        "k": 180,
                        "era": 3.50,
                        "whip": 1.20,
                        "er": 70,
                        "bb_allowed": 55,
                        "h_allowed": 165,
                        "adp": 100,
                    }
                )
                pid += 1

        pool = pd.DataFrame(rows)
        user_ids = list(range(1, 9))
        opp_ids = list(range(9, 17))

        result = compute_category_win_probabilities(user_ids, opp_ids, pool, config)

        cat_names = {c["name"] for c in result["categories"]}
        for cat in all_cats:
            assert cat in cat_names, f"Missing category: {cat}"

    def test_simulation_preserves_current_wins(self):
        """Verify simulation respects current records (wins only go up)."""
        from src.standings_engine import simulate_season_enhanced

        teams = ["A", "B"]
        current = {"A": {"W": 10, "L": 5, "T": 1}, "B": {"W": 8, "L": 7, "T": 1}}
        team_totals = {
            "A": {
                "R": 5,
                "HR": 1.5,
                "RBI": 5,
                "SB": 0.5,
                "AVG": 0.260,
                "OBP": 0.330,
                "W": 0.8,
                "L": 0.5,
                "SV": 0.3,
                "K": 8,
                "ERA": 3.80,
                "WHIP": 1.25,
            },
            "B": {
                "R": 4.5,
                "HR": 1.2,
                "RBI": 4.5,
                "SB": 0.8,
                "AVG": 0.250,
                "OBP": 0.320,
                "W": 0.7,
                "L": 0.6,
                "SV": 0.2,
                "K": 7,
                "ERA": 4.00,
                "WHIP": 1.30,
            },
        }
        schedule = {w: [("A", "B")] for w in range(8, 12)}

        sim = simulate_season_enhanced(current, team_totals, schedule, current_week=8, n_sims=50, seed=42)

        # Projected wins should be >= current wins
        for team in teams:
            proj_w = sim["projected_records"][team]["W"]
            assert proj_w >= current[team]["W"], f"{team}: projected {proj_w} < current {current[team]['W']}"

    def test_magic_numbers_consistency(self):
        """Magic numbers should be consistent with standings."""
        from src.standings_engine import compute_magic_numbers

        wins = {"A": 15, "B": 12, "C": 10, "D": 8, "E": 6, "F": 4}
        magic = compute_magic_numbers(wins, remaining_matchups=8, playoff_spots=3)

        # Leader should have lowest (or 0) magic number
        assert magic["A"] is not None
        assert magic["A"] <= magic["B"] or magic["B"] is None

        # Last place may be eliminated
        # (with 4 wins and 8 remaining = 12 max, but 3rd place has 10 now + 8 = 18 max)
        # So F can still theoretically make it (12 vs 10's max of 18)

    def test_team_strength_profiles(self):
        """Verify team strength profiles compute for all teams."""
        from src.standings_engine import compute_team_strength_profiles

        team_totals = {
            "A": {
                "R": 5,
                "HR": 1.5,
                "RBI": 5,
                "SB": 0.5,
                "AVG": 0.260,
                "OBP": 0.330,
                "W": 0.8,
                "L": 0.5,
                "SV": 0.3,
                "K": 8,
                "ERA": 3.80,
                "WHIP": 1.25,
            },
            "B": {
                "R": 4.5,
                "HR": 1.2,
                "RBI": 4.5,
                "SB": 0.8,
                "AVG": 0.250,
                "OBP": 0.320,
                "W": 0.7,
                "L": 0.6,
                "SV": 0.2,
                "K": 7,
                "ERA": 4.00,
                "WHIP": 1.30,
            },
            "C": {
                "R": 6,
                "HR": 2.0,
                "RBI": 6,
                "SB": 0.3,
                "AVG": 0.275,
                "OBP": 0.345,
                "W": 0.9,
                "L": 0.4,
                "SV": 0.5,
                "K": 9,
                "ERA": 3.50,
                "WHIP": 1.15,
            },
        }

        profiles = compute_team_strength_profiles(team_totals)
        assert len(profiles) == 3
        for p in profiles:
            assert "team_name" in p
            assert "power_rating" in p
            assert 0.0 <= p["power_rating"] <= 100.0
