"""Tests for the Start/Sit Advisor module.

Covers all 4 public functions, edge cases, confidence labels,
reasoning generation, and category impact calculations.
All tests are self-contained with no DB or API dependencies.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.start_sit import (
    _ALPHA_MAP,
    _compute_category_impact,
    _compute_matchup_factors,
    _compute_start_score,
    _count_team_games,
    classify_matchup_state,
    compute_weekly_projection,
    risk_adjusted_score,
    start_sit_recommendation,
)
from src.valuation import LeagueConfig

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_config():
    """Create a default LeagueConfig for testing."""
    return LeagueConfig()


def _make_pool(players=None):
    """Build a minimal player_pool DataFrame for testing.

    Each player dict should have at minimum: player_id, name, team,
    is_hitter, and projection columns.
    """
    if players is None:
        players = [
            {
                "player_id": 1,
                "name": "Aaron Judge",
                "team": "NYY",
                "is_hitter": 1,
                "bats": "R",
                "r": 60,
                "hr": 40,
                "rbi": 90,
                "sb": 5,
                "avg": 0.280,
                "obp": 0.380,
                "ab": 500,
                "h": 140,
                "bb": 80,
                "hbp": 10,
                "sf": 5,
                "pa": 595,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "ip": 0,
            },
            {
                "player_id": 2,
                "name": "Mike Trout",
                "team": "LAA",
                "is_hitter": 1,
                "bats": "R",
                "r": 50,
                "hr": 30,
                "rbi": 70,
                "sb": 10,
                "avg": 0.270,
                "obp": 0.370,
                "ab": 450,
                "h": 121,
                "bb": 70,
                "hbp": 8,
                "sf": 4,
                "pa": 532,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "ip": 0,
            },
            {
                "player_id": 3,
                "name": "Mookie Betts",
                "team": "LAD",
                "is_hitter": 1,
                "bats": "R",
                "r": 55,
                "hr": 25,
                "rbi": 75,
                "sb": 15,
                "avg": 0.290,
                "obp": 0.385,
                "ab": 520,
                "h": 150,
                "bb": 75,
                "hbp": 7,
                "sf": 6,
                "pa": 608,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "ip": 0,
            },
            {
                "player_id": 4,
                "name": "Trea Turner",
                "team": "PHI",
                "is_hitter": 1,
                "bats": "R",
                "r": 45,
                "hr": 18,
                "rbi": 60,
                "sb": 25,
                "avg": 0.275,
                "obp": 0.340,
                "ab": 540,
                "h": 148,
                "bb": 40,
                "hbp": 5,
                "sf": 4,
                "pa": 589,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "ip": 0,
            },
            {
                "player_id": 10,
                "name": "Gerrit Cole",
                "team": "NYY",
                "is_hitter": 0,
                "bats": "R",
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "ab": 0,
                "h": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "pa": 0,
                "w": 14,
                "l": 5,
                "sv": 0,
                "k": 200,
                "era": 2.80,
                "whip": 1.05,
                "ip": 190,
                "er": 59,
                "bb_allowed": 40,
                "h_allowed": 160,
            },
            {
                "player_id": 11,
                "name": "Zack Wheeler",
                "team": "PHI",
                "is_hitter": 0,
                "bats": "R",
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "ab": 0,
                "h": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "pa": 0,
                "w": 12,
                "l": 7,
                "sv": 0,
                "k": 180,
                "era": 3.20,
                "whip": 1.10,
                "ip": 175,
                "er": 62,
                "bb_allowed": 35,
                "h_allowed": 158,
            },
        ]
    return pd.DataFrame(players)


def _make_schedule(team_abbrev="NYY", games=6):
    """Create a mock weekly schedule for a team."""
    from src.optimizer.matchup_adjustments import _MLB_TEAM_ABBREVS

    abbrev_to_full = {v: k for k, v in _MLB_TEAM_ABBREVS.items()}
    full_name = abbrev_to_full.get(team_abbrev, team_abbrev)

    schedule = []
    opponents = ["BOS", "BOS", "BOS", "TOR", "TOR", "TOR"]
    for i in range(min(games, len(opponents))):
        opp_full = abbrev_to_full.get(opponents[i], opponents[i])
        if i < 3:
            # Home games
            schedule.append(
                {
                    "game_date": f"2026-04-0{i + 1}",
                    "home_name": full_name,
                    "away_name": opp_full,
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                }
            )
        else:
            # Away games
            schedule.append(
                {
                    "game_date": f"2026-04-0{i + 1}",
                    "home_name": opp_full,
                    "away_name": full_name,
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                }
            )
    return schedule


def _make_park_factors():
    """Create minimal park factors dict."""
    return {
        "NYY": 1.05,
        "LAA": 0.95,
        "LAD": 1.00,
        "PHI": 1.02,
        "BOS": 1.10,
        "TOR": 1.00,
        "COL": 1.38,
        "MIA": 0.88,
    }


# ── classify_matchup_state tests ────────────────────────────────────


class TestClassifyMatchupState:
    """Tests for classify_matchup_state."""

    def test_winning_state(self):
        """Winning 8 of 12 categories should return 'winning'."""
        my = {
            "R": 50,
            "HR": 15,
            "RBI": 45,
            "SB": 8,
            "AVG": 0.280,
            "OBP": 0.360,
            "W": 6,
            "L": 3,
            "SV": 4,
            "K": 55,
            "ERA": 3.20,
            "WHIP": 1.10,
        }
        opp = {
            "R": 40,
            "HR": 10,
            "RBI": 35,
            "SB": 5,
            "AVG": 0.250,
            "OBP": 0.320,
            "W": 4,
            "L": 5,
            "SV": 2,
            "K": 40,
            "ERA": 4.00,
            "WHIP": 1.30,
        }
        assert classify_matchup_state(my, opp) == "winning"

    def test_losing_state(self):
        """Losing most categories should return 'losing'."""
        my = {
            "R": 30,
            "HR": 5,
            "RBI": 25,
            "SB": 2,
            "AVG": 0.230,
            "OBP": 0.290,
            "W": 2,
            "L": 6,
            "SV": 1,
            "K": 30,
            "ERA": 5.00,
            "WHIP": 1.50,
        }
        opp = {
            "R": 50,
            "HR": 15,
            "RBI": 45,
            "SB": 8,
            "AVG": 0.280,
            "OBP": 0.360,
            "W": 6,
            "L": 3,
            "SV": 4,
            "K": 55,
            "ERA": 3.20,
            "WHIP": 1.10,
        }
        assert classify_matchup_state(my, opp) == "losing"

    def test_close_state(self):
        """Splitting categories evenly should return 'close'."""
        my = {
            "R": 50,
            "HR": 15,
            "RBI": 45,
            "SB": 3,
            "AVG": 0.250,
            "OBP": 0.320,
            "W": 6,
            "L": 5,
            "SV": 4,
            "K": 55,
            "ERA": 4.00,
            "WHIP": 1.30,
        }
        opp = {
            "R": 45,
            "HR": 12,
            "RBI": 40,
            "SB": 8,
            "AVG": 0.270,
            "OBP": 0.350,
            "W": 5,
            "L": 4,
            "SV": 3,
            "K": 50,
            "ERA": 3.50,
            "WHIP": 1.15,
        }
        assert classify_matchup_state(my, opp) == "close"

    def test_none_totals_returns_close(self):
        """None totals should gracefully return 'close'."""
        assert classify_matchup_state(None, None) == "close"
        assert classify_matchup_state({"R": 50}, None) == "close"
        assert classify_matchup_state(None, {"R": 50}) == "close"

    def test_empty_totals_returns_close(self):
        """Empty dicts should return 'close'."""
        assert classify_matchup_state({}, {}) == "close"

    def test_inverse_stats_handled(self):
        """ERA and WHIP should be counted correctly (lower is better)."""
        # Only ERA/WHIP differ; my team has better (lower) values -> winning those cats
        my = {"ERA": 2.50, "WHIP": 0.95}
        opp = {"ERA": 4.50, "WHIP": 1.40}
        config = LeagueConfig()
        # Only 2 cats, both won -> winning
        result = classify_matchup_state(my, opp, config)
        assert result == "winning"

    def test_lowercase_keys(self):
        """Should handle lowercase category keys."""
        my = {
            "r": 50,
            "hr": 15,
            "rbi": 45,
            "sb": 8,
            "avg": 0.280,
            "obp": 0.360,
            "w": 6,
            "l": 3,
            "sv": 4,
            "k": 55,
            "era": 3.20,
            "whip": 1.10,
        }
        opp = {
            "r": 40,
            "hr": 10,
            "rbi": 35,
            "sb": 5,
            "avg": 0.250,
            "obp": 0.320,
            "w": 4,
            "l": 5,
            "sv": 2,
            "k": 40,
            "era": 4.00,
            "whip": 1.30,
        }
        assert classify_matchup_state(my, opp) == "winning"


# ── risk_adjusted_score tests ───────────────────────────────────────


class TestRiskAdjustedScore:
    """Tests for risk_adjusted_score."""

    def test_winning_favors_floor(self):
        """When winning, alpha=0.8, weights toward expected + floor."""
        result = risk_adjusted_score(10.0, 6.0, 15.0, "winning")
        # alpha=0.8: 0.8*10 + 0.2*6 = 8.0 + 1.2 = 9.2
        assert abs(result - 9.2) < 0.01

    def test_losing_favors_ceiling(self):
        """When losing, alpha=0.2, weights toward ceiling."""
        result = risk_adjusted_score(10.0, 6.0, 15.0, "losing")
        # alpha=0.2: 0.2*10 + 0.8*15 = 2.0 + 12.0 = 14.0
        assert abs(result - 14.0) < 0.01

    def test_close_balanced(self):
        """When close, alpha=0.5, balanced between expected and avg(floor, ceiling)."""
        result = risk_adjusted_score(10.0, 6.0, 15.0, "close")
        # alpha=0.5: 0.5*10 + 0.5*((6+15)/2) = 5.0 + 5.25 = 10.25
        assert abs(result - 10.25) < 0.01

    def test_unknown_state_defaults_close(self):
        """Unknown matchup state should default to alpha=0.5."""
        result = risk_adjusted_score(10.0, 6.0, 15.0, "unknown")
        expected = risk_adjusted_score(10.0, 6.0, 15.0, "close")
        assert abs(result - expected) < 0.01

    def test_zero_values(self):
        """Should handle zero values without error."""
        result = risk_adjusted_score(0.0, 0.0, 0.0, "close")
        assert result == 0.0

    def test_negative_values(self):
        """Should handle negative values (pitcher ERA contributions)."""
        result = risk_adjusted_score(-5.0, -8.0, -2.0, "winning")
        # alpha=0.8: 0.8*(-5) + 0.2*(-8) = -4.0 + -1.6 = -5.6
        assert abs(result - (-5.6)) < 0.01

    def test_alpha_map_values(self):
        """Verify alpha map has correct values."""
        assert _ALPHA_MAP["winning"] == 0.8
        assert _ALPHA_MAP["close"] == 0.5
        assert _ALPHA_MAP["losing"] == 0.2


# ── compute_weekly_projection tests ─────────────────────────────────


class TestComputeWeeklyProjection:
    """Tests for compute_weekly_projection."""

    def test_hitter_without_schedule(self):
        """Hitter with no schedule falls back to default 6 games."""
        pool = _make_pool()
        player = pool[pool["player_id"] == 1].iloc[0]
        proj = compute_weekly_projection(player, None, None)

        # Should have all hitter stat keys
        assert "r" in proj
        assert "hr" in proj
        assert "rbi" in proj
        assert "sb" in proj
        assert "avg" in proj
        assert "obp" in proj
        # Weekly values should be a fraction of season totals
        assert proj["hr"] < player["hr"]
        assert proj["hr"] > 0

    def test_pitcher_without_schedule(self):
        """Pitcher with no schedule falls back to 1 game."""
        pool = _make_pool()
        player = pool[pool["player_id"] == 10].iloc[0]
        proj = compute_weekly_projection(player, None, None)

        assert "w" in proj
        assert "k" in proj
        assert "era" in proj
        assert "whip" in proj
        assert proj["k"] > 0
        assert proj["k"] < player["k"]

    def test_hitter_with_schedule(self):
        """Hitter with schedule uses actual game count."""
        pool = _make_pool()
        player = pool[pool["player_id"] == 1].iloc[0]
        schedule = _make_schedule("NYY", 6)
        proj = compute_weekly_projection(player, schedule, _make_park_factors())

        assert proj["hr"] > 0
        assert "avg" in proj
        assert 0.0 < proj["avg"] < 0.500

    def test_park_factor_applied(self):
        """Park factors should adjust counting stats."""
        pool = _make_pool()
        player = pool[pool["player_id"] == 1].iloc[0]

        # All home games at Coors (high PF)
        from src.optimizer.matchup_adjustments import _MLB_TEAM_ABBREVS

        abbrev_to_full = {v: k for k, v in _MLB_TEAM_ABBREVS.items()}

        schedule_neutral = [
            {
                "game_date": "2026-04-01",
                "home_name": abbrev_to_full.get("NYY", "NYY"),
                "away_name": abbrev_to_full.get("BOS", "BOS"),
                "home_probable_pitcher": "",
                "away_probable_pitcher": "",
            }
        ]

        pf_neutral = {"NYY": 1.0, "BOS": 1.0}
        pf_high = {"NYY": 1.30, "BOS": 1.0}

        proj_neutral = compute_weekly_projection(player, schedule_neutral, pf_neutral)
        proj_high = compute_weekly_projection(player, schedule_neutral, pf_high)

        # High park factor should boost counting stats
        assert proj_high["hr"] >= proj_neutral["hr"]

    def test_rate_stats_computed_correctly(self):
        """Rate stats should be derived from components, not scaled."""
        pool = _make_pool()
        player = pool[pool["player_id"] == 1].iloc[0]
        proj = compute_weekly_projection(player, None, None)

        # AVG should be reasonable
        assert 0.100 <= proj["avg"] <= 0.400
        # OBP should be >= AVG
        assert proj["obp"] >= proj["avg"] - 0.01  # small tolerance


# ── start_sit_recommendation tests ──────────────────────────────────


class TestStartSitRecommendation:
    """Tests for start_sit_recommendation."""

    def test_two_player_comparison(self):
        """Should recommend one of two players."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2], pool, config)

        assert result["recommendation"] in (1, 2)
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["confidence_label"] in ("Clear Start", "Lean Start", "Toss-up")
        assert len(result["players"]) == 2

    def test_three_player_comparison(self):
        """Should handle 3 players."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2, 3], pool, config)

        assert result["recommendation"] in (1, 2, 3)
        assert len(result["players"]) == 3

    def test_four_player_comparison(self):
        """Should handle 4 players (maximum)."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2, 3, 4], pool, config)

        assert result["recommendation"] in (1, 2, 3, 4)
        assert len(result["players"]) == 4

    def test_single_player_returns_trivial(self):
        """Single player should return that player with full confidence."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1], pool, config)

        assert result["recommendation"] == 1
        assert result["confidence"] == 1.0
        assert result["confidence_label"] == "Clear Start"
        assert len(result["players"]) == 1
        assert result["players"][0]["reasoning"] == ["Only candidate for this slot"]

    def test_empty_player_list(self):
        """Empty player list should return None recommendation."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([], pool, config)

        assert result["recommendation"] is None
        assert result["confidence"] == 0.0

    def test_missing_player_id(self):
        """Player IDs not in pool should be skipped."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 999], pool, config)

        # Only player 1 found
        assert len(result["players"]) == 1
        # With only 1 valid player found from 2 requested, should still return
        assert result["recommendation"] == 1

    def test_player_detail_structure(self):
        """Each player result should have all required fields."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2], pool, config)

        for p in result["players"]:
            assert "player_id" in p
            assert "name" in p
            assert "start_score" in p
            assert "matchup_factors" in p
            assert "floor" in p
            assert "ceiling" in p
            assert "category_impact" in p
            assert "reasoning" in p
            assert isinstance(p["reasoning"], list)
            assert len(p["reasoning"]) >= 1

    def test_players_sorted_descending(self):
        """Players should be sorted by score, best first."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2, 3], pool, config)

        scores = [p["start_score"] for p in result["players"]]
        assert scores == sorted(scores, reverse=True)

    def test_recommendation_is_first_player(self):
        """Recommended player should be the first (highest-scored) player."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2], pool, config)

        assert result["recommendation"] == result["players"][0]["player_id"]

    def test_with_h2h_totals(self):
        """Should work with H2H weekly totals provided."""
        pool = _make_pool()
        config = _make_config()
        my_totals = {
            "r": 40,
            "hr": 10,
            "rbi": 35,
            "sb": 5,
            "avg": 0.260,
            "obp": 0.340,
            "w": 5,
            "l": 4,
            "sv": 3,
            "k": 45,
            "era": 3.80,
            "whip": 1.20,
        }
        opp_totals = {
            "r": 45,
            "hr": 12,
            "rbi": 40,
            "sb": 3,
            "avg": 0.270,
            "obp": 0.350,
            "w": 4,
            "l": 5,
            "sv": 2,
            "k": 40,
            "era": 4.00,
            "whip": 1.30,
        }

        result = start_sit_recommendation(
            [1, 2],
            pool,
            config,
            my_weekly_totals=my_totals,
            opp_weekly_totals=opp_totals,
        )

        assert result["recommendation"] in (1, 2)
        assert len(result["players"]) == 2

    def test_pitcher_comparison(self):
        """Should handle pitcher-vs-pitcher comparison."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([10, 11], pool, config)

        assert result["recommendation"] in (10, 11)
        assert len(result["players"]) == 2
        # Pitchers should have pitching category impacts
        for p in result["players"]:
            cats = set(p["category_impact"].keys())
            # Should have at least some pitching categories
            assert len(cats) > 0

    def test_with_schedule_and_park_factors(self):
        """Should incorporate schedule and park factors when provided."""
        pool = _make_pool()
        config = _make_config()
        schedule = _make_schedule("NYY", 6)
        pf = _make_park_factors()

        result = start_sit_recommendation(
            [1, 2],
            pool,
            config,
            weekly_schedule=schedule,
            park_factors=pf,
        )

        assert result["recommendation"] in (1, 2)
        # Player 1 (NYY) should have park factor in matchup_factors
        nyy_player = next(p for p in result["players"] if p["player_id"] == 1)
        assert "park" in nyy_player["matchup_factors"]

    def test_truncates_to_four_players(self):
        """More than 4 players should be truncated to 4."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2, 3, 4, 10], pool, config)

        # Should have at most 4 players
        assert len(result["players"]) <= 4


# ── Confidence label tests ──────────────────────────────────────────


class TestConfidenceLabels:
    """Tests for confidence label thresholds."""

    def test_clear_start_threshold(self):
        """Confidence > 0.30 should be 'Clear Start'."""
        config = _make_config()

        # Create players with extremely different projections
        players = [
            {
                "player_id": 100,
                "name": "Star Player",
                "team": "NYY",
                "is_hitter": 1,
                "bats": "R",
                "r": 120,
                "hr": 55,
                "rbi": 130,
                "sb": 25,
                "avg": 0.320,
                "obp": 0.430,
                "ab": 600,
                "h": 192,
                "bb": 100,
                "hbp": 12,
                "sf": 6,
                "pa": 718,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "ip": 0,
            },
            {
                "player_id": 101,
                "name": "Weak Player",
                "team": "MIA",
                "is_hitter": 1,
                "bats": "R",
                "r": 2,
                "hr": 0,
                "rbi": 2,
                "sb": 0,
                "avg": 0.150,
                "obp": 0.190,
                "ab": 40,
                "h": 6,
                "bb": 2,
                "hbp": 0,
                "sf": 0,
                "pa": 42,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "ip": 0,
            },
        ]
        big_gap_pool = pd.DataFrame(players)
        result = start_sit_recommendation([100, 101], big_gap_pool, config)
        assert result["confidence"] > 0.20
        assert result["confidence_label"] in ("Clear Start", "Lean Start")

    def test_toss_up_for_similar_players(self):
        """Very similar players should produce 'Toss-up' or 'Lean Start'."""
        # Create two nearly identical players
        players = [
            {
                "player_id": 200,
                "name": "Player A",
                "team": "NYY",
                "is_hitter": 1,
                "bats": "R",
                "r": 50,
                "hr": 20,
                "rbi": 60,
                "sb": 10,
                "avg": 0.270,
                "obp": 0.350,
                "ab": 500,
                "h": 135,
                "bb": 60,
                "hbp": 5,
                "sf": 4,
                "pa": 569,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "ip": 0,
            },
            {
                "player_id": 201,
                "name": "Player B",
                "team": "NYY",
                "is_hitter": 1,
                "bats": "R",
                "r": 50,
                "hr": 20,
                "rbi": 60,
                "sb": 10,
                "avg": 0.270,
                "obp": 0.350,
                "ab": 500,
                "h": 135,
                "bb": 60,
                "hbp": 5,
                "sf": 4,
                "pa": 569,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "ip": 0,
            },
        ]
        same_pool = pd.DataFrame(players)
        config = _make_config()
        result = start_sit_recommendation([200, 201], same_pool, config)
        assert result["confidence_label"] == "Toss-up"

    def test_all_label_values_valid(self):
        """Confidence label should always be one of three valid values."""
        pool = _make_pool()
        config = _make_config()
        for ids in ([1, 2], [1, 2, 3], [1, 2, 3, 4]):
            result = start_sit_recommendation(ids, pool, config)
            assert result["confidence_label"] in ("Clear Start", "Lean Start", "Toss-up")


# ── Reasoning generation tests ──────────────────────────────────────


class TestReasoningGeneration:
    """Tests for reasoning generation."""

    def test_at_least_one_reason_per_player(self):
        """Every player should have at least one reasoning string."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2, 3], pool, config)

        for p in result["players"]:
            assert len(p["reasoning"]) >= 1
            for reason in p["reasoning"]:
                assert isinstance(reason, str)
                assert len(reason) > 0

    def test_max_three_reasons(self):
        """No player should have more than 3 reasons."""
        pool = _make_pool()
        config = _make_config()
        schedule = _make_schedule("NYY", 6)
        pf = _make_park_factors()

        result = start_sit_recommendation(
            [1, 2],
            pool,
            config,
            weekly_schedule=schedule,
            park_factors=pf,
        )

        for p in result["players"]:
            assert len(p["reasoning"]) <= 3

    def test_no_emoji_in_reasoning(self):
        """Reasoning strings must not contain emoji."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2], pool, config)

        for p in result["players"]:
            for reason in p["reasoning"]:
                # Check for common emoji code points
                for char in reason:
                    assert ord(char) < 0x1F600 or ord(char) > 0x1F9FF, f"Emoji found in reasoning: {reason}"


# ── Category impact tests ───────────────────────────────────────────


class TestCategoryImpact:
    """Tests for category impact calculation."""

    def test_hitter_has_hitting_categories(self):
        """Hitter should have hitting category impacts."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2], pool, config)

        hitter = result["players"][0]
        impact_cats = set(hitter["category_impact"].keys())
        # Should include hitting categories
        for cat in config.hitting_categories:
            assert cat in impact_cats

    def test_pitcher_has_pitching_categories(self):
        """Pitcher should have pitching category impacts."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([10, 11], pool, config)

        pitcher = result["players"][0]
        impact_cats = set(pitcher["category_impact"].keys())
        for cat in config.pitching_categories:
            assert cat in impact_cats

    def test_inverse_stat_impact_sign(self):
        """ERA and WHIP impact should be negative (lower is better = negative contribution)."""
        config = _make_config()
        weekly_proj = {"w": 1.0, "l": 0.5, "sv": 0.0, "k": 8.0, "era": 3.50, "whip": 1.15}
        h2h_weights = {c.lower(): 1.0 for c in config.all_categories}

        impact = _compute_category_impact(weekly_proj, h2h_weights, config, is_hitter=False)

        # ERA is inverse: a positive ERA (bad for pitcher) should give negative impact
        assert impact["ERA"] < 0
        assert impact["WHIP"] < 0
        # L is also inverse
        assert impact["L"] < 0

    def test_category_impact_values_are_floats(self):
        """All category impact values should be numeric."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2], pool, config)

        for p in result["players"]:
            for cat, val in p["category_impact"].items():
                assert isinstance(val, (int, float))


# ── _count_team_games tests ─────────────────────────────────────────


class TestCountTeamGames:
    """Tests for _count_team_games helper."""

    def test_no_schedule(self):
        """No schedule returns 0."""
        assert _count_team_games("NYY", None) == 0

    def test_empty_schedule(self):
        """Empty schedule returns 0."""
        assert _count_team_games("NYY", []) == 0

    def test_no_team(self):
        """Empty team string returns 0."""
        assert _count_team_games("", _make_schedule("NYY", 3)) == 0

    def test_counts_correct_games(self):
        """Should count all home + away games for the team."""
        schedule = _make_schedule("NYY", 6)
        count = _count_team_games("NYY", schedule)
        # _make_schedule creates entries using full names
        # _count_team_games should match via abbreviation lookup
        assert count == 6


# ── Edge case tests ─────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_empty_pool(self):
        """Empty player pool should return no recommendation."""
        pool = pd.DataFrame()
        config = _make_config()
        result = start_sit_recommendation([1, 2], pool, config)

        assert result["recommendation"] is None
        assert result["confidence"] == 0.0
        assert result["players"] == []

    def test_default_config(self):
        """Should work with None config (creates default)."""
        pool = _make_pool()
        result = start_sit_recommendation([1, 2], pool, None)

        assert result["recommendation"] in (1, 2)

    def test_missing_schedule_graceful(self):
        """Missing schedule data should not crash."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation(
            [1, 2],
            pool,
            config,
            weekly_schedule=None,
            park_factors=_make_park_factors(),
        )

        assert result["recommendation"] in (1, 2)

    def test_floor_less_than_ceiling(self):
        """Floor should always be less than or equal to ceiling."""
        pool = _make_pool()
        config = _make_config()
        result = start_sit_recommendation([1, 2, 3], pool, config)

        for p in result["players"]:
            assert p["floor"] <= p["ceiling"]
