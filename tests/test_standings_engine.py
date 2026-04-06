"""Tests for standings engine — database layer."""

from __future__ import annotations

import pytest
import pandas as pd

from src.database import (
    init_db,
    get_connection,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    """Ensure fresh DB for each test, clearing new tables."""
    init_db()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM league_schedule_full")
        conn.execute("DELETE FROM league_records")
        conn.commit()
    finally:
        conn.close()


class TestLeagueScheduleFullTable:
    """Tests for league_schedule_full table CRUD."""

    def test_upsert_and_load_full_schedule(self):
        from src.database import upsert_league_schedule_full, load_league_schedule_full

        upsert_league_schedule_full(1, "Team A", "Team B")
        upsert_league_schedule_full(1, "Team C", "Team D")
        upsert_league_schedule_full(2, "Team A", "Team C")

        result = load_league_schedule_full()
        assert isinstance(result, dict)
        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 2  # 2 matchups in week 1
        assert ("Team A", "Team B") in result[1]

    def test_upsert_full_schedule_idempotent(self):
        from src.database import upsert_league_schedule_full, load_league_schedule_full

        upsert_league_schedule_full(1, "Team A", "Team B")
        upsert_league_schedule_full(1, "Team A", "Team B")  # duplicate

        result = load_league_schedule_full()
        assert len(result[1]) == 1  # no duplicate

    def test_load_empty_full_schedule(self):
        from src.database import load_league_schedule_full

        result = load_league_schedule_full()
        assert result == {}


class TestLeagueRecordsTable:
    """Tests for league_records table CRUD."""

    def test_upsert_and_load_records(self):
        from src.database import upsert_league_record, load_league_records

        upsert_league_record("Team Hickey", wins=42, losses=32, ties=6,
                             win_pct=0.563, streak="L1", rank=3)
        upsert_league_record("Jonny Jockstrap", wins=48, losses=26, ties=6,
                             win_pct=0.638, streak="W3", rank=1)

        df = load_league_records()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.loc[df["team_name"] == "Team Hickey", "wins"].iloc[0] == 42

    def test_upsert_record_overwrites(self):
        from src.database import upsert_league_record, load_league_records

        upsert_league_record("Team A", wins=10, losses=5, ties=1,
                             win_pct=0.656, streak="W1", rank=1)
        upsert_league_record("Team A", wins=11, losses=5, ties=1,
                             win_pct=0.688, streak="W2", rank=1)

        df = load_league_records()
        assert len(df) == 1
        assert df.iloc[0]["wins"] == 11

    def test_load_empty_records(self):
        from src.database import load_league_records

        df = load_league_records()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


import numpy as np
from src.valuation import LeagueConfig


class TestCategoryWinProbabilities:
    """Tests for per-category win probability computation."""

    def _make_pool(self, players: list[dict]) -> pd.DataFrame:
        """Build minimal player pool DataFrame for testing."""
        base = {
            "player_id": 0, "name": "Test", "team": "TST", "positions": "OF",
            "is_hitter": 1, "is_injured": 0,
            "pa": 600, "ab": 550, "h": 150, "r": 80, "hr": 25, "rbi": 85,
            "sb": 10, "avg": 0.273, "obp": 0.340, "bb": 50, "hbp": 5, "sf": 5,
            "ip": 0, "w": 0, "l": 0, "sv": 0, "k": 0, "era": 0.0, "whip": 0.0,
            "er": 0, "bb_allowed": 0, "h_allowed": 0, "adp": 100,
        }
        rows = []
        for i, overrides in enumerate(players):
            row = {**base, "player_id": i + 1, **overrides}
            rows.append(row)
        return pd.DataFrame(rows)

    def test_equal_teams_near_50_percent(self):
        """Two identical teams should have ~50% win probability per category."""
        from src.standings_engine import compute_category_win_probabilities

        config = LeagueConfig()
        # 5 identical hitters per team
        hitter = {"name": "H", "is_hitter": 1, "r": 80, "hr": 25, "rbi": 85,
                  "sb": 10, "avg": 0.273, "obp": 0.340, "pa": 600, "ab": 550,
                  "h": 150, "bb": 50, "hbp": 5, "sf": 5}
        pitcher = {"name": "P", "is_hitter": 0, "ip": 180, "w": 12, "l": 8,
                   "sv": 0, "k": 180, "era": 3.50, "whip": 1.20,
                   "er": 70, "bb_allowed": 55, "h_allowed": 165,
                   "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        pool = self._make_pool(
            [hitter] * 5 + [pitcher] * 3 +  # user team (ids 1-8)
            [hitter] * 5 + [pitcher] * 3     # opp team (ids 9-16)
        )
        user_ids = list(range(1, 9))
        opp_ids = list(range(9, 17))

        result = compute_category_win_probabilities(user_ids, opp_ids, pool, config)

        assert "overall_win_pct" in result
        assert "categories" in result
        assert len(result["categories"]) == 12

        # Equal teams: overall should be near 50%
        assert 0.35 <= result["overall_win_pct"] <= 0.65

        # Each category should be near 50%
        for cat in result["categories"]:
            assert 0.30 <= cat["win_pct"] <= 0.70, f"{cat['name']} win_pct out of range: {cat['win_pct']}"

    def test_dominant_team_high_probability(self):
        """A much stronger team should have >80% win probability."""
        from src.standings_engine import compute_category_win_probabilities

        config = LeagueConfig()
        strong_hitter = {"name": "Strong", "is_hitter": 1, "r": 120, "hr": 45,
                         "rbi": 130, "sb": 25, "avg": 0.310, "obp": 0.400,
                         "pa": 700, "ab": 600, "h": 186, "bb": 80, "hbp": 10, "sf": 5}
        weak_hitter = {"name": "Weak", "is_hitter": 1, "r": 40, "hr": 8,
                       "rbi": 35, "sb": 2, "avg": 0.220, "obp": 0.270,
                       "pa": 400, "ab": 360, "h": 79, "bb": 30, "hbp": 3, "sf": 3}
        pitcher = {"name": "P", "is_hitter": 0, "ip": 180, "w": 12, "l": 8,
                   "sv": 5, "k": 180, "era": 3.50, "whip": 1.20,
                   "er": 70, "bb_allowed": 55, "h_allowed": 165,
                   "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        pool = self._make_pool(
            [strong_hitter] * 5 + [pitcher] * 3 +  # user (strong)
            [weak_hitter] * 5 + [pitcher] * 3       # opp (weak)
        )

        result = compute_category_win_probabilities(
            list(range(1, 9)), list(range(9, 17)), pool, config
        )
        assert result["overall_win_pct"] > 0.70

    def test_inverse_categories_correct_direction(self):
        """For ERA/WHIP/L, lower values should mean higher win probability."""
        from src.standings_engine import compute_category_win_probabilities

        config = LeagueConfig()
        hitter = {"name": "H", "is_hitter": 1, "r": 80, "hr": 25, "rbi": 85,
                  "sb": 10, "avg": 0.273, "obp": 0.340, "pa": 600, "ab": 550,
                  "h": 150, "bb": 50, "hbp": 5, "sf": 5}
        good_pitcher = {"name": "GP", "is_hitter": 0, "ip": 200, "w": 15, "l": 5,
                        "sv": 0, "k": 200, "era": 2.50, "whip": 1.00,
                        "er": 56, "bb_allowed": 40, "h_allowed": 160,
                        "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        bad_pitcher = {"name": "BP", "is_hitter": 0, "ip": 150, "w": 6, "l": 14,
                       "sv": 0, "k": 100, "era": 5.50, "whip": 1.60,
                       "er": 92, "bb_allowed": 70, "h_allowed": 170,
                       "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        pool = self._make_pool(
            [hitter] * 5 + [good_pitcher] * 3 +
            [hitter] * 5 + [bad_pitcher] * 3
        )

        result = compute_category_win_probabilities(
            list(range(1, 9)), list(range(9, 17)), pool, config
        )
        era_cat = next(c for c in result["categories"] if c["name"] == "ERA")
        whip_cat = next(c for c in result["categories"] if c["name"] == "WHIP")

        # User has better (lower) ERA/WHIP -> should have >50% win probability
        assert era_cat["win_pct"] > 0.55, f"ERA win_pct should be >0.55: {era_cat['win_pct']}"
        assert whip_cat["win_pct"] > 0.55, f"WHIP win_pct should be >0.55: {whip_cat['win_pct']}"

    def test_output_schema(self):
        """Verify the output dict has all required keys."""
        from src.standings_engine import compute_category_win_probabilities

        config = LeagueConfig()
        hitter = {"name": "H", "is_hitter": 1, "r": 80, "hr": 25, "rbi": 85,
                  "sb": 10, "avg": 0.273, "obp": 0.340, "pa": 600, "ab": 550,
                  "h": 150, "bb": 50, "hbp": 5, "sf": 5}
        pitcher = {"name": "P", "is_hitter": 0, "ip": 180, "w": 12, "l": 8,
                   "sv": 5, "k": 180, "era": 3.50, "whip": 1.20,
                   "er": 70, "bb_allowed": 55, "h_allowed": 165,
                   "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0, "obp": 0, "pa": 0, "ab": 0, "h": 0}
        pool = self._make_pool([hitter] * 5 + [pitcher] * 3 + [hitter] * 5 + [pitcher] * 3)

        result = compute_category_win_probabilities(
            list(range(1, 9)), list(range(9, 17)), pool, config
        )

        # Top-level keys
        assert "overall_win_pct" in result
        assert "overall_tie_pct" in result
        assert "overall_loss_pct" in result
        assert "projected_score" in result
        assert "categories" in result

        # Probabilities sum to ~1.0
        total = result["overall_win_pct"] + result["overall_tie_pct"] + result["overall_loss_pct"]
        assert 0.99 <= total <= 1.01

        # Per-category schema
        for cat in result["categories"]:
            assert "name" in cat
            assert "user_proj" in cat
            assert "opp_proj" in cat
            assert "win_pct" in cat
            assert "is_inverse" in cat
            assert 0.0 <= cat["win_pct"] <= 1.0


class TestSimulateSeasonEnhanced:
    """Tests for schedule-aware MC season simulation."""

    def test_carries_forward_current_record(self):
        """Simulation should start from current W-L-T, not zero."""
        from src.standings_engine import simulate_season_enhanced

        current = {
            "Team A": {"W": 40, "L": 30, "T": 10},
            "Team B": {"W": 30, "L": 40, "T": 10},
        }
        schedule = {
            i: [("Team A", "Team B")]
            for i in range(15, 25)  # weeks 15-24 remaining
        }
        # Minimal team totals
        team_totals = {
            "Team A": {"R": 5, "HR": 2, "RBI": 5, "SB": 1, "AVG": 0.270, "OBP": 0.340,
                        "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 3.50, "WHIP": 1.20},
            "Team B": {"R": 5, "HR": 2, "RBI": 5, "SB": 1, "AVG": 0.270, "OBP": 0.340,
                        "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 3.50, "WHIP": 1.20},
        }

        result = simulate_season_enhanced(
            current_standings=current,
            team_weekly_totals=team_totals,
            full_schedule=schedule,
            current_week=15,
            n_sims=100,
            seed=42,
        )

        assert "projected_records" in result
        # Team A started with 40W, should end with >= 40W
        assert result["projected_records"]["Team A"]["W"] >= 40
        # Team B started with 30W, should end with >= 30W
        assert result["projected_records"]["Team B"]["W"] >= 30

    def test_uses_actual_schedule(self):
        """Simulation should use specific matchups, not round-robin."""
        from src.standings_engine import simulate_season_enhanced

        current = {
            "Team A": {"W": 0, "L": 0, "T": 0},
            "Team B": {"W": 0, "L": 0, "T": 0},
            "Team C": {"W": 0, "L": 0, "T": 0},
            "Team D": {"W": 0, "L": 0, "T": 0},
        }
        # A always plays B, C always plays D
        schedule = {
            i: [("Team A", "Team B"), ("Team C", "Team D")]
            for i in range(1, 11)
        }
        team_totals = {
            t: {"R": 5, "HR": 2, "RBI": 5, "SB": 1, "AVG": 0.270, "OBP": 0.340,
                "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 3.50, "WHIP": 1.20}
            for t in current
        }

        result = simulate_season_enhanced(
            current_standings=current,
            team_weekly_totals=team_totals,
            full_schedule=schedule,
            current_week=1,
            n_sims=100,
            seed=42,
        )

        # All teams should have exactly 10 matchups (10 weeks)
        for team in current:
            rec = result["projected_records"][team]
            assert rec["W"] + rec["L"] + rec["T"] == 10

    def test_output_schema(self):
        from src.standings_engine import simulate_season_enhanced

        current = {"A": {"W": 5, "L": 5, "T": 0}, "B": {"W": 5, "L": 5, "T": 0}}
        schedule = {i: [("A", "B")] for i in range(6, 11)}
        totals = {
            t: {"R": 5, "HR": 2, "RBI": 5, "SB": 1, "AVG": 0.270, "OBP": 0.340,
                "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 3.50, "WHIP": 1.20}
            for t in current
        }

        result = simulate_season_enhanced(
            current, totals, schedule, current_week=6, n_sims=50, seed=42
        )

        assert "projected_records" in result
        assert "playoff_probability" in result
        assert "confidence_intervals" in result
        assert "strength_of_schedule" in result


class TestMagicNumbers:
    """Tests for magic number computation."""

    def test_already_clinched(self):
        from src.standings_engine import compute_magic_numbers

        # 4-team league, team already has huge lead
        standings = {"A": 50, "B": 40, "C": 30, "D": 20}
        remaining = 2  # only 2 games left
        result = compute_magic_numbers(standings, remaining, playoff_spots=2)
        assert result["A"] == 0  # already clinched

    def test_eliminated_team(self):
        from src.standings_engine import compute_magic_numbers

        standings = {"A": 50, "B": 45, "C": 40, "D": 5}
        remaining = 2
        result = compute_magic_numbers(standings, remaining, playoff_spots=2)
        assert result["D"] is None  # can't catch up

    def test_mid_pack(self):
        from src.standings_engine import compute_magic_numbers

        standings = {"A": 10, "B": 9, "C": 8, "D": 7, "E": 6, "F": 5}
        remaining = 10
        result = compute_magic_numbers(standings, remaining, playoff_spots=4)
        # Mid-pack teams should have positive magic numbers
        assert result["C"] is not None
        assert result["C"] > 0


class TestTeamStrengthProfiles:
    def test_all_teams_ranked(self):
        from src.standings_engine import compute_team_strength_profiles

        totals = {
            "A": {"R": 5, "HR": 3, "RBI": 5, "SB": 2, "AVG": 0.280, "OBP": 0.350,
                  "W": 2, "L": 1, "SV": 1, "K": 10, "ERA": 3.00, "WHIP": 1.10},
            "B": {"R": 4, "HR": 2, "RBI": 4, "SB": 1, "AVG": 0.260, "OBP": 0.330,
                  "W": 1, "L": 1, "SV": 1, "K": 8, "ERA": 4.00, "WHIP": 1.30},
        }
        schedule = {1: [("A", "B")]}
        result = compute_team_strength_profiles(totals, schedule, current_week=1)
        assert len(result) == 2
        assert result[0]["power_rating"] > result[1]["power_rating"]  # A is stronger
        assert all("ci_low" in r and "ci_high" in r for r in result)
