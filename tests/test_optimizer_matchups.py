"""Tests for src/optimizer/matchup_adjustments.py.

Covers platoon splits, park factors, weather adjustments,
schedule fetching, and the weekly adjustment pipeline.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.optimizer.matchup_adjustments import (
    DEFAULT_PLATOON_ADVANTAGE,
    PLATOON_REGRESSION_PA,
    _build_team_schedule,
    compute_weekly_matchup_adjustments,
    park_factor_adjustment,
    platoon_adjustment,
    weather_hr_adjustment,
)

# ── Platoon Tests ────────────────────────────────────────────────────


class TestPlatoonAdjustment:
    def test_platoon_lhb_advantage(self) -> None:
        """LHB vs RHP gets the default 8.6% platoon advantage."""
        factor = platoon_adjustment(batter_hand="L", pitcher_hand="R")
        expected = 1.0 + DEFAULT_PLATOON_ADVANTAGE["LHB"]
        assert factor == pytest.approx(expected, abs=1e-6)
        assert factor > 1.0

    def test_platoon_rhb_advantage(self) -> None:
        """RHB vs LHP gets the smaller 6.1% advantage."""
        factor = platoon_adjustment(batter_hand="R", pitcher_hand="L")
        expected = 1.0 + DEFAULT_PLATOON_ADVANTAGE["RHB"]
        assert factor == pytest.approx(expected, abs=1e-6)
        assert factor > 1.0
        # RHB advantage is smaller than LHB advantage
        lhb_factor = platoon_adjustment(batter_hand="L", pitcher_hand="R")
        assert factor < lhb_factor

    def test_platoon_same_hand_lhb(self) -> None:
        """LHB vs LHP: same hand = disadvantage."""
        factor = platoon_adjustment(batter_hand="L", pitcher_hand="L")
        assert factor < 1.0
        expected = 1.0 - DEFAULT_PLATOON_ADVANTAGE["LHB"]
        assert factor == pytest.approx(expected, abs=1e-6)

    def test_platoon_same_hand_rhb(self) -> None:
        """RHB vs RHP: same hand = disadvantage."""
        factor = platoon_adjustment(batter_hand="R", pitcher_hand="R")
        assert factor < 1.0
        expected = 1.0 - DEFAULT_PLATOON_ADVANTAGE["RHB"]
        assert factor == pytest.approx(expected, abs=1e-6)

    def test_platoon_with_individual_data(self) -> None:
        """Individual split data regresses toward overall average."""
        # Player with .300 split avg over 200 PA, .270 overall
        # LHB: regression_pa = 1000
        factor = platoon_adjustment(
            batter_hand="L",
            pitcher_hand="R",
            batter_split_avg=0.300,
            batter_overall_avg=0.270,
            sample_pa=200,
        )
        # regressed = (200*.300 + 1000*.270) / (200 + 1000)
        #           = (60 + 270) / 1200 = 330/1200 = 0.275
        # factor = 0.275 / 0.270 = 1.01852
        expected_regressed = (200 * 0.300 + 1000 * 0.270) / (200 + 1000)
        expected_factor = expected_regressed / 0.270
        assert factor == pytest.approx(expected_factor, abs=1e-4)
        # Should be between 1.0 and the raw split ratio (0.300/0.270)
        assert 1.0 < factor < 0.300 / 0.270

    def test_platoon_unknown_hand_neutral(self) -> None:
        """Unknown handedness returns neutral 1.0."""
        assert platoon_adjustment(batter_hand="", pitcher_hand="R") == 1.0
        assert platoon_adjustment(batter_hand="L", pitcher_hand="") == 1.0
        assert platoon_adjustment(batter_hand="S", pitcher_hand="R") == 1.0


# ── Park Factor Tests ────────────────────────────────────────────────


class TestParkFactorAdjustment:
    def test_park_factor_coors(self) -> None:
        """Opponent's park (venue) factor is used first."""
        pf = {"COL": 1.38, "NYM": 0.95}
        factor = park_factor_adjustment("COL", "NYM", pf, is_hitter=True)
        assert factor == pytest.approx(0.95, abs=1e-6)

    def test_park_factor_miami(self) -> None:
        """Opponent's park (venue) factor is used first."""
        pf = {"MIA": 0.88, "ATL": 1.01}
        factor = park_factor_adjustment("MIA", "ATL", pf, is_hitter=True)
        assert factor == pytest.approx(1.01, abs=1e-6)

    def test_park_factor_unknown_team(self) -> None:
        """Unknown team returns neutral 1.0."""
        pf = {"COL": 1.38}
        factor = park_factor_adjustment("XYZ", "ABC", pf, is_hitter=True)
        assert factor == 1.0

    def test_park_factor_pitcher_dampened(self) -> None:
        """C5: Pitcher park factor uses reciprocal (1/pf)."""
        pf = {"COL": 1.38}
        factor = park_factor_adjustment("COL", "COL", pf, is_hitter=False)
        # 1.0 / 1.38 = 0.7246
        assert factor == pytest.approx(1.0 / 1.38, abs=1e-3)

    def test_park_factor_pitcher_unknown_neutral(self) -> None:
        """Pitcher in unknown park gets neutral 1.0 (pf defaults to 1.0)."""
        pf = {"COL": 1.38}
        factor = park_factor_adjustment("XYZ", "ABC", pf, is_hitter=False)
        # pf = 1.0, so 1.0 / 1.0 = 1.0
        assert factor == pytest.approx(1.0, abs=1e-6)


# ── Weather Tests ────────────────────────────────────────────────────


class TestWeatherHrAdjustment:
    def test_weather_hot_day(self) -> None:
        """95F -> positive HR adjustment > 1.0."""
        factor = weather_hr_adjustment(temp_f=95.0)
        # 1.0 + 0.009 * (95 - 72) = 1.0 + 0.009 * 23 = 1.207
        expected = 1.0 + 0.009 * 23.0
        assert factor == pytest.approx(expected, abs=1e-6)
        assert factor > 1.0

    def test_weather_cold_day(self) -> None:
        """60F -> no adjustment (no penalty below 72)."""
        factor = weather_hr_adjustment(temp_f=60.0)
        assert factor == 1.0

    def test_weather_reference_temp(self) -> None:
        """72F -> exactly 1.0 (reference temperature)."""
        factor = weather_hr_adjustment(temp_f=72.0)
        assert factor == 1.0

    def test_weather_default_neutral(self) -> None:
        """Default (no arg) -> neutral 1.0."""
        factor = weather_hr_adjustment()
        assert factor == 1.0


# ── Schedule + Pipeline Tests ────────────────────────────────────────


class TestComputeWeeklyAdjustments:
    @pytest.fixture()
    def sample_roster(self) -> pd.DataFrame:
        """Small test roster with hitter and pitcher."""
        return pd.DataFrame(
            [
                {
                    "name": "Test Hitter",
                    "team": "COL",
                    "is_hitter": True,
                    "r": 10.0,
                    "hr": 5.0,
                    "rbi": 12.0,
                    "sb": 2.0,
                    "w": 0.0,
                    "sv": 0.0,
                    "k": 0.0,
                },
                {
                    "name": "Test Pitcher",
                    "team": "MIA",
                    "is_hitter": False,
                    "r": 0.0,
                    "hr": 0.0,
                    "rbi": 0.0,
                    "sb": 0.0,
                    "w": 2.0,
                    "sv": 0.0,
                    "k": 15.0,
                },
            ]
        )

    @pytest.fixture()
    def sample_schedule(self) -> list[dict]:
        return [
            {
                "game_date": "2026-03-16",
                "home_name": "Colorado Rockies",
                "away_name": "Miami Marlins",
                "home_probable_pitcher": "Some Pitcher",
                "away_probable_pitcher": "Other Pitcher",
            },
        ]

    def test_schedule_empty_graceful(self, sample_roster: pd.DataFrame) -> None:
        """Empty schedule returns unchanged roster with matchup_adjusted=False."""
        result = compute_weekly_matchup_adjustments(
            roster=sample_roster,
            week_schedule=[],
            park_factors={"COL": 1.38},
        )
        assert "matchup_adjusted" in result.columns
        assert not result["matchup_adjusted"].any()
        # Stats unchanged
        assert result.iloc[0]["hr"] == 5.0

    def test_compute_adjustments_preserves_schema(
        self, sample_roster: pd.DataFrame, sample_schedule: list[dict]
    ) -> None:
        """Output has same columns as input + matchup_adjusted."""
        result = compute_weekly_matchup_adjustments(
            roster=sample_roster,
            week_schedule=sample_schedule,
            park_factors={"COL": 1.38, "MIA": 0.88},
        )
        original_cols = set(sample_roster.columns)
        result_cols = set(result.columns)
        assert original_cols.issubset(result_cols)
        assert "matchup_adjusted" in result_cols

    def test_coors_hitter_adjusted_up(self, sample_roster: pd.DataFrame, sample_schedule: list[dict]) -> None:
        """COL hitter at home in Coors should have inflated stats."""
        result = compute_weekly_matchup_adjustments(
            roster=sample_roster,
            week_schedule=sample_schedule,
            park_factors={"COL": 1.38, "MIA": 0.88},
            enable_platoon=False,
            enable_weather=False,
            enable_opposing_pitcher=False,
        )
        hitter_row = result[result["name"] == "Test Hitter"].iloc[0]
        # HR should be adjusted by Coors park factor
        assert hitter_row["hr"] > 5.0
        assert hitter_row["matchup_adjusted"] == True  # noqa: E712

    def test_empty_roster_graceful(self, sample_schedule: list[dict]) -> None:
        """Empty roster returns empty DataFrame with matchup_adjusted column."""
        empty = pd.DataFrame(columns=["name", "team", "is_hitter", "hr"])
        result = compute_weekly_matchup_adjustments(
            roster=empty,
            week_schedule=sample_schedule,
            park_factors={"COL": 1.38},
        )
        assert "matchup_adjusted" in result.columns
        assert len(result) == 0


class TestBuildTeamSchedule:
    def test_both_teams_get_entries(self) -> None:
        """Both home and away teams appear in the grouped schedule."""
        schedule = [
            {
                "game_date": "2026-03-16",
                "home_name": "Colorado Rockies",
                "away_name": "Miami Marlins",
                "home_probable_pitcher": "",
                "away_probable_pitcher": "",
            }
        ]
        result = _build_team_schedule(schedule)
        assert "COL" in result
        assert "MIA" in result
        assert result["COL"][0]["is_home"] is True
        assert result["MIA"][0]["is_home"] is False
