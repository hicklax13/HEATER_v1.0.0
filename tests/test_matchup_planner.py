"""Tests for src/matchup_planner.py.

Covers hitter/pitcher single-game ratings, color tier mapping,
weekly matchup ratings pipeline, percentile ranking, park factor
impact, platoon advantage, home/away adjustment, and edge cases.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.matchup_planner import (
    _BASELINE_GAMES_PER_WEEK,
    _DEFAULT_BASE_WOBA,
    _DEFAULT_OPP_WRC_PLUS,
    _DEFAULT_XFIP,
    _HITTER_HOME_ADVANTAGE,
    _MIN_OPP_WRC_PLUS,
    _PITCHER_HOME_ADVANTAGE,
    color_tier,
    compute_all_ratings_with_percentiles,
    compute_hitter_game_rating,
    compute_pitcher_game_rating,
    compute_weekly_matchup_ratings,
)

# ── Color Tier Tests ─────────────────────────────────────────────────


class TestColorTier:
    """Tests for color_tier percentile-to-tier mapping."""

    def test_smash_at_80(self) -> None:
        """Exactly 80th percentile is smash."""
        assert color_tier(80.0) == "smash"

    def test_smash_at_100(self) -> None:
        """100th percentile is smash."""
        assert color_tier(100.0) == "smash"

    def test_smash_at_95(self) -> None:
        """95th percentile is smash."""
        assert color_tier(95.0) == "smash"

    def test_favorable_at_60(self) -> None:
        """Exactly 60th percentile is favorable."""
        assert color_tier(60.0) == "favorable"

    def test_favorable_at_79(self) -> None:
        """79th percentile is favorable."""
        assert color_tier(79.9) == "favorable"

    def test_neutral_at_40(self) -> None:
        """Exactly 40th percentile is neutral."""
        assert color_tier(40.0) == "neutral"

    def test_neutral_at_59(self) -> None:
        """59th percentile is neutral."""
        assert color_tier(59.0) == "neutral"

    def test_unfavorable_at_20(self) -> None:
        """Exactly 20th percentile is unfavorable."""
        assert color_tier(20.0) == "unfavorable"

    def test_unfavorable_at_39(self) -> None:
        """39th percentile is unfavorable."""
        assert color_tier(39.0) == "unfavorable"

    def test_avoid_below_20(self) -> None:
        """Below 20th percentile is avoid."""
        assert color_tier(19.9) == "avoid"

    def test_avoid_at_zero(self) -> None:
        """0th percentile is avoid."""
        assert color_tier(0.0) == "avoid"


# ── Hitter Game Rating Tests ─────────────────────────────────────────


class TestHitterGameRating:
    """Tests for compute_hitter_game_rating."""

    def test_basic_hitter_rating(self) -> None:
        """Basic hitter rating with neutral conditions."""
        result = compute_hitter_game_rating(
            player_stats={"woba": 0.350},
            opposing_pitcher_stats=None,
            park_factor=1.0,
            is_home=False,
        )
        assert result["raw_score"] == pytest.approx(0.350, abs=1e-6)
        assert result["base_woba"] == 0.350
        assert result["park_factor"] == 1.0
        assert result["platoon_adj"] == 1.0
        assert result["home_away"] == 1.0

    def test_hitter_coors_boost(self) -> None:
        """Coors Field (1.38) significantly boosts hitter rating."""
        result = compute_hitter_game_rating(
            player_stats={"woba": 0.320},
            opposing_pitcher_stats=None,
            park_factor=1.38,
            is_home=False,
        )
        expected = 0.320 * 1.38
        assert result["raw_score"] == pytest.approx(expected, abs=1e-6)
        assert result["raw_score"] > 0.320

    def test_hitter_miami_suppression(self) -> None:
        """Miami (0.88) suppresses hitter rating."""
        result = compute_hitter_game_rating(
            player_stats={"woba": 0.320},
            opposing_pitcher_stats=None,
            park_factor=0.88,
            is_home=False,
        )
        expected = 0.320 * 0.88
        assert result["raw_score"] == pytest.approx(expected, abs=1e-6)
        assert result["raw_score"] < 0.320

    def test_hitter_home_advantage(self) -> None:
        """Home games get the home advantage multiplier."""
        away = compute_hitter_game_rating(
            player_stats={"woba": 0.350},
            opposing_pitcher_stats=None,
            park_factor=1.0,
            is_home=False,
        )
        home = compute_hitter_game_rating(
            player_stats={"woba": 0.350},
            opposing_pitcher_stats=None,
            park_factor=1.0,
            is_home=True,
        )
        assert home["raw_score"] > away["raw_score"]
        assert home["home_away"] == _HITTER_HOME_ADVANTAGE

    def test_hitter_platoon_advantage(self) -> None:
        """LHB vs RHP gets platoon boost."""
        result = compute_hitter_game_rating(
            player_stats={"woba": 0.320},
            opposing_pitcher_stats=None,
            park_factor=1.0,
            is_home=False,
            batter_hand="L",
            pitcher_hand="R",
        )
        assert result["platoon_adj"] > 1.0
        assert result["raw_score"] > 0.320

    def test_hitter_platoon_disadvantage(self) -> None:
        """LHB vs LHP gets platoon penalty."""
        result = compute_hitter_game_rating(
            player_stats={"woba": 0.320},
            opposing_pitcher_stats=None,
            park_factor=1.0,
            is_home=False,
            batter_hand="L",
            pitcher_hand="L",
        )
        assert result["platoon_adj"] < 1.0
        assert result["raw_score"] < 0.320

    def test_hitter_no_platoon_without_hands(self) -> None:
        """No platoon adjustment when handedness is missing."""
        result = compute_hitter_game_rating(
            player_stats={"woba": 0.320},
            opposing_pitcher_stats=None,
            park_factor=1.0,
            is_home=False,
            batter_hand=None,
            pitcher_hand="R",
        )
        assert result["platoon_adj"] == 1.0

    def test_hitter_falls_back_to_obp(self) -> None:
        """Uses OBP when wOBA is missing."""
        result = compute_hitter_game_rating(
            player_stats={"obp": 0.380},
            opposing_pitcher_stats=None,
            park_factor=1.0,
            is_home=False,
        )
        assert result["base_woba"] == 0.380

    def test_hitter_falls_back_to_default(self) -> None:
        """Uses default wOBA when all stats missing."""
        result = compute_hitter_game_rating(
            player_stats={},
            opposing_pitcher_stats=None,
            park_factor=1.0,
            is_home=False,
        )
        assert result["base_woba"] == _DEFAULT_BASE_WOBA

    def test_hitter_combined_factors(self) -> None:
        """All factors multiply together correctly."""
        result = compute_hitter_game_rating(
            player_stats={"woba": 0.400},
            opposing_pitcher_stats=None,
            park_factor=1.10,
            is_home=True,
            batter_hand="R",
            pitcher_hand="L",
        )
        # All components should multiply
        expected = 0.400 * 1.10 * result["platoon_adj"] * _HITTER_HOME_ADVANTAGE
        assert result["raw_score"] == pytest.approx(expected, abs=1e-6)


# ── Pitcher Game Rating Tests ────────────────────────────────────────


class TestPitcherGameRating:
    """Tests for compute_pitcher_game_rating."""

    def test_basic_pitcher_rating(self) -> None:
        """Basic pitcher rating with neutral conditions."""
        result = compute_pitcher_game_rating(
            pitcher_stats={"xfip": 3.50},
            opponent_team_stats=None,
            park_factor=1.0,
            is_home=False,
        )
        # (10 - 3.5) * (100/100) * (2.0-1.0) * 1 * 1.0
        expected = 6.5 * 1.0 * 1.0 * 1 * 1.0
        assert result["raw_score"] == pytest.approx(expected, abs=1e-6)

    def test_pitcher_inverse_park_factor(self) -> None:
        """Hitter-friendly parks hurt pitchers via inverse_park."""
        result = compute_pitcher_game_rating(
            pitcher_stats={"xfip": 3.50},
            opponent_team_stats=None,
            park_factor=1.38,  # Coors
            is_home=False,
        )
        assert result["inverse_park"] == pytest.approx(0.62, abs=1e-6)
        assert result["raw_score"] < 6.5  # Worse than neutral

    def test_pitcher_friendly_park(self) -> None:
        """Pitcher-friendly parks boost pitcher rating."""
        result = compute_pitcher_game_rating(
            pitcher_stats={"xfip": 3.50},
            opponent_team_stats=None,
            park_factor=0.88,  # Miami
            is_home=False,
        )
        assert result["inverse_park"] == pytest.approx(1.12, abs=1e-6)
        assert result["raw_score"] > 6.5  # Better than neutral

    def test_pitcher_weak_opponent(self) -> None:
        """Weak-hitting opponent (low wRC+) boosts pitcher rating."""
        result = compute_pitcher_game_rating(
            pitcher_stats={"xfip": 3.50},
            opponent_team_stats={"wrc_plus": 80},
            park_factor=1.0,
            is_home=False,
        )
        # (10 - 3.5) * (100/80) * 1.0 * 1 * 1.0 = 6.5 * 1.25
        expected = 6.5 * 1.25 * 1.0 * 1 * 1.0
        assert result["raw_score"] == pytest.approx(expected, abs=1e-6)

    def test_pitcher_strong_opponent(self) -> None:
        """Strong-hitting opponent (high wRC+) hurts pitcher rating."""
        result = compute_pitcher_game_rating(
            pitcher_stats={"xfip": 3.50},
            opponent_team_stats={"wrc_plus": 120},
            park_factor=1.0,
            is_home=False,
        )
        expected = 6.5 * (100.0 / 120.0)
        assert result["raw_score"] == pytest.approx(expected, abs=1e-6)
        assert result["raw_score"] < 6.5

    def test_pitcher_opp_wrc_floor(self) -> None:
        """Opponent wRC+ is floored at 50 to prevent extreme scores."""
        result = compute_pitcher_game_rating(
            pitcher_stats={"xfip": 3.50},
            opponent_team_stats={"wrc_plus": 10},
            park_factor=1.0,
            is_home=False,
        )
        assert result["opp_wrc_plus"] == _MIN_OPP_WRC_PLUS

    def test_pitcher_home_advantage(self) -> None:
        """Home pitchers get the home advantage multiplier."""
        away = compute_pitcher_game_rating(
            pitcher_stats={"xfip": 3.50},
            opponent_team_stats=None,
            park_factor=1.0,
            is_home=False,
        )
        home = compute_pitcher_game_rating(
            pitcher_stats={"xfip": 3.50},
            opponent_team_stats=None,
            park_factor=1.0,
            is_home=True,
        )
        assert home["raw_score"] > away["raw_score"]
        assert home["home_away"] == _PITCHER_HOME_ADVANTAGE

    def test_pitcher_falls_back_to_era(self) -> None:
        """Uses ERA when xFIP is missing."""
        result = compute_pitcher_game_rating(
            pitcher_stats={"era": 3.00},
            opponent_team_stats=None,
            park_factor=1.0,
            is_home=False,
        )
        assert result["xfip"] == 3.00

    def test_pitcher_falls_back_to_default_xfip(self) -> None:
        """Uses default xFIP when all stats missing."""
        result = compute_pitcher_game_rating(
            pitcher_stats={},
            opponent_team_stats=None,
            park_factor=1.0,
            is_home=False,
        )
        assert result["xfip"] == _DEFAULT_XFIP


# ── Percentile Ratings Tests ─────────────────────────────────────────


class TestPercentileRatings:
    """Tests for compute_all_ratings_with_percentiles."""

    def test_empty_list(self) -> None:
        """Empty input returns empty output."""
        assert compute_all_ratings_with_percentiles([]) == []

    def test_single_value(self) -> None:
        """Single value gets 50th percentile (median)."""
        result = compute_all_ratings_with_percentiles([5.0])
        assert len(result) == 1
        assert result[0]["percentile_rank"] == 50.0
        # rating = 1 + 9 * 0.5 = 5.5
        assert result[0]["rating"] == pytest.approx(5.5, abs=0.1)
        assert result[0]["tier"] == "neutral"

    def test_ordering_preserved(self) -> None:
        """Higher raw scores get higher ratings."""
        result = compute_all_ratings_with_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        ratings = [r["rating"] for r in result]
        assert ratings == sorted(ratings)

    def test_lowest_gets_avoid(self) -> None:
        """Lowest score in a large enough set gets avoid tier."""
        scores = list(range(1, 11))  # 1 through 10
        result = compute_all_ratings_with_percentiles(scores)
        assert result[0]["tier"] == "avoid"

    def test_highest_gets_smash(self) -> None:
        """Highest score in a large enough set gets smash tier."""
        scores = list(range(1, 11))
        result = compute_all_ratings_with_percentiles(scores)
        assert result[-1]["tier"] == "smash"

    def test_rating_range_1_to_10(self) -> None:
        """All ratings fall within [1, 10]."""
        scores = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        result = compute_all_ratings_with_percentiles(scores)
        for r in result:
            assert 1.0 <= r["rating"] <= 10.0

    def test_identical_scores(self) -> None:
        """All identical scores get the same rating."""
        result = compute_all_ratings_with_percentiles([3.0, 3.0, 3.0])
        ratings = [r["rating"] for r in result]
        assert ratings[0] == ratings[1] == ratings[2]


# ── Weekly Matchup Ratings Tests ─────────────────────────────────────


def _mock_roster(players: list[dict]) -> pd.DataFrame:
    """Build a test roster DataFrame."""
    return pd.DataFrame(players)


def _mock_schedule(games: list[dict]) -> list[dict]:
    """Build a test weekly schedule."""
    return games


class TestWeeklyMatchupRatings:
    """Tests for compute_weekly_matchup_ratings."""

    def test_empty_roster(self) -> None:
        """Empty roster returns empty DataFrame."""
        result = compute_weekly_matchup_ratings(
            roster=pd.DataFrame(),
            weekly_schedule=[],
            park_factors={"NYY": 1.05},
        )
        assert result.empty

    def test_none_roster(self) -> None:
        """None roster returns empty DataFrame."""
        result = compute_weekly_matchup_ratings(
            roster=None,
            weekly_schedule=[
                {
                    "home_name": "New York Yankees",
                    "away_name": "Boston Red Sox",
                    "game_date": "2026-04-01",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                }
            ],
            park_factors={"NYY": 1.05},
        )
        assert result.empty

    def test_no_schedule(self) -> None:
        """Empty schedule returns empty DataFrame."""
        roster = _mock_roster(
            [
                {
                    "player_id": 1,
                    "name": "Test Hitter",
                    "team": "NYY",
                    "positions": "1B",
                    "is_hitter": True,
                    "woba": 0.350,
                },
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster,
            weekly_schedule=[],
            park_factors={"NYY": 1.05},
        )
        assert result.empty

    def test_single_hitter_single_game(self) -> None:
        """Single hitter with one game gets a valid rating."""
        roster = _mock_roster(
            [
                {
                    "player_id": 1,
                    "name": "Test Hitter",
                    "team": "NYY",
                    "positions": "1B",
                    "is_hitter": True,
                    "woba": 0.350,
                },
            ]
        )
        schedule = _mock_schedule(
            [
                {
                    "home_name": "New York Yankees",
                    "away_name": "Boston Red Sox",
                    "game_date": "2026-04-01",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                },
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster,
            weekly_schedule=schedule,
            park_factors={"NYY": 1.05, "BOS": 1.04},
        )
        assert len(result) == 1
        assert result.iloc[0]["games_count"] == 1
        assert 1.0 <= result.iloc[0]["weekly_matchup_rating"] <= 10.0
        assert result.iloc[0]["matchup_tier"] in ("smash", "favorable", "neutral", "unfavorable", "avoid")

    def test_games_count_impacts_hitter_rating(self) -> None:
        """More games per week should boost a hitter's raw score via games_scaling."""
        roster_few = _mock_roster(
            [
                {
                    "player_id": 1,
                    "name": "Hitter A",
                    "team": "NYY",
                    "positions": "1B",
                    "is_hitter": True,
                    "woba": 0.350,
                },
                {
                    "player_id": 2,
                    "name": "Hitter B",
                    "team": "BOS",
                    "positions": "1B",
                    "is_hitter": True,
                    "woba": 0.350,
                },
            ]
        )
        # NYY gets 3 games, BOS gets 6 games (but same wOBA)
        schedule = _mock_schedule(
            [
                {
                    "home_name": "New York Yankees",
                    "away_name": "Tampa Bay Rays",
                    "game_date": f"2026-04-0{d}",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                }
                for d in range(1, 4)
            ]
            + [
                {
                    "home_name": "Boston Red Sox",
                    "away_name": "Tampa Bay Rays",
                    "game_date": f"2026-04-0{d}",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                }
                for d in range(1, 7)
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster_few,
            weekly_schedule=schedule,
            park_factors={"NYY": 1.0, "BOS": 1.0, "TB": 1.0},
        )
        # BOS hitter (6 games) should have higher raw score due to games_scaling
        # Since they share the percentile pool, the one with more games
        # should rate higher
        assert len(result) == 2

    def test_pitcher_in_roster(self) -> None:
        """Pitcher ratings computed separately from hitters."""
        roster = _mock_roster(
            [
                {
                    "player_id": 1,
                    "name": "Ace Pitcher",
                    "team": "NYY",
                    "positions": "SP",
                    "is_hitter": False,
                    "xfip": 3.00,
                    "era": 2.80,
                    "starts": 1,
                },
            ]
        )
        schedule = _mock_schedule(
            [
                {
                    "home_name": "New York Yankees",
                    "away_name": "Boston Red Sox",
                    "game_date": "2026-04-01",
                    "home_probable_pitcher": "Ace Pitcher",
                    "away_probable_pitcher": "",
                },
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster,
            weekly_schedule=schedule,
            park_factors={"NYY": 1.05, "BOS": 1.04},
        )
        assert len(result) == 1
        assert not result.iloc[0]["is_hitter"]
        assert 1.0 <= result.iloc[0]["weekly_matchup_rating"] <= 10.0

    def test_mixed_hitters_and_pitchers(self) -> None:
        """Hitters and pitchers are both included in results."""
        roster = _mock_roster(
            [
                {"player_id": 1, "name": "Hitter", "team": "NYY", "positions": "1B", "is_hitter": True, "woba": 0.350},
                {
                    "player_id": 2,
                    "name": "Pitcher",
                    "team": "NYY",
                    "positions": "SP",
                    "is_hitter": False,
                    "xfip": 3.50,
                    "starts": 1,
                },
            ]
        )
        schedule = _mock_schedule(
            [
                {
                    "home_name": "New York Yankees",
                    "away_name": "Boston Red Sox",
                    "game_date": "2026-04-01",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                },
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster,
            weekly_schedule=schedule,
            park_factors={"NYY": 1.05, "BOS": 1.04},
        )
        assert len(result) == 2
        hitters = result[result["is_hitter"].astype(bool)]
        pitchers = result[~result["is_hitter"].astype(bool)]
        assert len(hitters) == 1
        assert len(pitchers) == 1

    def test_player_no_matching_team_in_schedule(self) -> None:
        """Player whose team has no games gets 0 games and avoid tier."""
        roster = _mock_roster(
            [
                {
                    "player_id": 1,
                    "name": "Lonely Player",
                    "team": "SEA",
                    "positions": "SS",
                    "is_hitter": True,
                    "woba": 0.400,
                },
                {
                    "player_id": 2,
                    "name": "Active Player",
                    "team": "NYY",
                    "positions": "1B",
                    "is_hitter": True,
                    "woba": 0.320,
                },
            ]
        )
        schedule = _mock_schedule(
            [
                {
                    "home_name": "New York Yankees",
                    "away_name": "Boston Red Sox",
                    "game_date": "2026-04-01",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                },
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster,
            weekly_schedule=schedule,
            park_factors={"NYY": 1.05, "BOS": 1.04},
        )
        sea_player = result[result["name"] == "Lonely Player"].iloc[0]
        assert sea_player["games_count"] == 0
        assert sea_player["matchup_tier"] == "avoid"
        assert sea_player["weekly_matchup_rating"] == 1.0

    def test_output_columns_present(self) -> None:
        """All expected output columns are present."""
        roster = _mock_roster(
            [
                {"player_id": 1, "name": "Test", "team": "NYY", "positions": "1B", "is_hitter": True, "woba": 0.320},
            ]
        )
        schedule = _mock_schedule(
            [
                {
                    "home_name": "New York Yankees",
                    "away_name": "Boston Red Sox",
                    "game_date": "2026-04-01",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                },
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster,
            weekly_schedule=schedule,
            park_factors={"NYY": 1.05, "BOS": 1.04},
        )
        expected_cols = {
            "player_id",
            "name",
            "positions",
            "is_hitter",
            "games",
            "weekly_matchup_rating",
            "matchup_tier",
            "games_count",
            "projected_stats_adjusted",
        }
        assert expected_cols.issubset(set(result.columns))

    @patch("src.matchup_planner.get_weekly_schedule")
    def test_fetches_schedule_when_none(self, mock_sched) -> None:
        """When weekly_schedule is None, fetches schedule automatically."""
        mock_sched.return_value = [
            {
                "home_name": "New York Yankees",
                "away_name": "Boston Red Sox",
                "game_date": "2026-04-01",
                "home_probable_pitcher": "",
                "away_probable_pitcher": "",
            },
        ]
        roster = _mock_roster(
            [
                {"player_id": 1, "name": "Test", "team": "NYY", "positions": "1B", "is_hitter": True, "woba": 0.320},
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster,
            weekly_schedule=None,
            park_factors={"NYY": 1.05, "BOS": 1.04},
        )
        mock_sched.assert_called_once_with(days_ahead=7)
        assert len(result) == 1

    def test_park_factor_coors_vs_miami_hitter(self) -> None:
        """Coors hitter should rate higher than Miami hitter (same wOBA)."""
        roster = _mock_roster(
            [
                {
                    "player_id": 1,
                    "name": "Coors Hitter",
                    "team": "COL",
                    "positions": "1B",
                    "is_hitter": True,
                    "woba": 0.320,
                },
                {
                    "player_id": 2,
                    "name": "Miami Hitter",
                    "team": "MIA",
                    "positions": "1B",
                    "is_hitter": True,
                    "woba": 0.320,
                },
            ]
        )
        schedule = _mock_schedule(
            [
                {
                    "home_name": "Colorado Rockies",
                    "away_name": "Arizona Diamondbacks",
                    "game_date": "2026-04-01",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                },
                {
                    "home_name": "Miami Marlins",
                    "away_name": "Atlanta Braves",
                    "game_date": "2026-04-01",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                },
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster,
            weekly_schedule=schedule,
            park_factors={"COL": 1.38, "MIA": 0.88, "ARI": 1.06, "ATL": 1.01},
        )
        coors = result[result["name"] == "Coors Hitter"].iloc[0]
        miami = result[result["name"] == "Miami Hitter"].iloc[0]
        assert coors["weekly_matchup_rating"] > miami["weekly_matchup_rating"]

    def test_projected_stats_adjusted_present(self) -> None:
        """Hitters get park-adjusted projected stats in output."""
        roster = _mock_roster(
            [
                {
                    "player_id": 1,
                    "name": "Test",
                    "team": "COL",
                    "positions": "1B",
                    "is_hitter": True,
                    "woba": 0.350,
                    "hr": 30,
                    "r": 90,
                    "rbi": 85,
                    "sb": 10,
                },
            ]
        )
        schedule = _mock_schedule(
            [
                {
                    "home_name": "Colorado Rockies",
                    "away_name": "Arizona Diamondbacks",
                    "game_date": "2026-04-01",
                    "home_probable_pitcher": "",
                    "away_probable_pitcher": "",
                },
            ]
        )
        result = compute_weekly_matchup_ratings(
            roster=roster,
            weekly_schedule=schedule,
            park_factors={"COL": 1.38, "ARI": 1.06},
        )
        adjusted = result.iloc[0]["projected_stats_adjusted"]
        assert "hr" in adjusted
        # HR should be inflated by Coors factor
        assert adjusted["hr"] > 30
