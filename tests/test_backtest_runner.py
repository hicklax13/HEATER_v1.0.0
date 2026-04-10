"""Tests for the historical backtest runner.

Covers:
  - WeekResult and BacktestReport data structure integrity
  - Aggregation helpers (_aggregate_hitting_games, _aggregate_pitching_games)
  - IP parsing edge cases
  - Projection scaling (_build_projected_stats)
  - fetch_weekly_actuals graceful degradation when API unavailable
  - Single-week and multi-week backtest flow
  - Report formatting
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.optimizer.backtest_runner import (
    ALL_CATEGORIES,
    BACKTEST_PLAYER_IDS,
    BacktestReport,
    WeekResult,
    _aggregate_hitting_games,
    _aggregate_pitching_games,
    _build_projected_stats,
    _parse_ip,
    backtest_week,
    fetch_weekly_actuals,
    format_report,
    run_backtest,
)

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_roster():
    """Roster DataFrame with projected stats matching BACKTEST_PLAYER_IDS."""
    rng = np.random.default_rng(42)
    rows = []
    for pid, name in list(BACKTEST_PLAYER_IDS.items())[:10]:
        is_hitter = pid in [592450, 660271, 665742, 605141, 641355]
        rows.append(
            {
                "player_id": pid,
                "mlb_id": pid,
                "name": name,
                "positions": "OF" if is_hitter else "SP",
                "is_hitter": 1 if is_hitter else 0,
                "r": int(rng.integers(50, 110)) if is_hitter else 0,
                "hr": int(rng.integers(15, 45)) if is_hitter else 0,
                "rbi": int(rng.integers(50, 120)) if is_hitter else 0,
                "sb": int(rng.integers(5, 30)) if is_hitter else 0,
                "avg": float(rng.uniform(0.250, 0.310)) if is_hitter else 0.0,
                "obp": float(rng.uniform(0.320, 0.400)) if is_hitter else 0.0,
                "w": 0 if is_hitter else int(rng.integers(8, 18)),
                "l": 0 if is_hitter else int(rng.integers(4, 12)),
                "sv": 0,
                "k": 0 if is_hitter else int(rng.integers(150, 250)),
                "era": 0.0 if is_hitter else float(rng.uniform(2.50, 4.50)),
                "whip": 0.0 if is_hitter else float(rng.uniform(0.95, 1.30)),
                "ip": 0.0 if is_hitter else float(rng.uniform(150, 220)),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_actuals():
    """Simulated weekly actuals DataFrame."""
    rng = np.random.default_rng(99)
    rows = []
    for pid, name in list(BACKTEST_PLAYER_IDS.items())[:10]:
        is_hitter = pid in [592450, 660271, 665742, 605141, 641355]
        rows.append(
            {
                "player_id": pid,
                "player_name": name,
                "is_hitter": 1 if is_hitter else 0,
                "r": int(rng.integers(2, 8)) if is_hitter else 0,
                "hr": int(rng.integers(0, 4)) if is_hitter else 0,
                "rbi": int(rng.integers(2, 10)) if is_hitter else 0,
                "sb": int(rng.integers(0, 3)) if is_hitter else 0,
                "avg": float(rng.uniform(0.200, 0.350)) if is_hitter else 0.0,
                "obp": float(rng.uniform(0.280, 0.420)) if is_hitter else 0.0,
                "w": 0 if is_hitter else int(rng.integers(0, 2)),
                "l": 0 if is_hitter else int(rng.integers(0, 2)),
                "sv": 0,
                "k": 0 if is_hitter else int(rng.integers(5, 20)),
                "era": 0.0 if is_hitter else float(rng.uniform(1.50, 6.00)),
                "whip": 0.0 if is_hitter else float(rng.uniform(0.80, 1.60)),
                "ip": 0.0 if is_hitter else float(rng.uniform(5, 14)),
                "h": int(rng.integers(3, 10)) if is_hitter else 0,
                "ab": int(rng.integers(20, 30)) if is_hitter else 0,
                "games": float(rng.integers(4, 7)),
            }
        )
    return pd.DataFrame(rows)


# ── Data Structure Tests ──────────────────────────────────────────────


class TestWeekResult:
    def test_required_fields(self):
        """WeekResult should have all required fields."""
        wr = WeekResult(
            week_start=date(2025, 6, 2),
            week_end=date(2025, 6, 8),
            projection_rmse=5.2,
            rank_correlation=0.65,
            bust_rate=0.30,
            lineup_grade="B",
            n_players=15,
        )
        assert wr.week_start == date(2025, 6, 2)
        assert wr.week_end == date(2025, 6, 8)
        assert wr.projection_rmse == pytest.approx(5.2)
        assert wr.rank_correlation == pytest.approx(0.65)
        assert wr.bust_rate == pytest.approx(0.30)
        assert wr.lineup_grade == "B"
        assert wr.n_players == 15

    def test_category_rmse_default(self):
        """category_rmse should default to empty dict."""
        wr = WeekResult(
            week_start=date(2025, 6, 2),
            week_end=date(2025, 6, 8),
            projection_rmse=5.0,
            rank_correlation=0.5,
            bust_rate=0.2,
            lineup_grade="A",
            n_players=10,
        )
        assert wr.category_rmse == {}

    def test_category_rmse_populated(self):
        """category_rmse should accept per-category values."""
        cat_rmse = {"hr": 1.5, "rbi": 3.2, "era": 0.8}
        wr = WeekResult(
            week_start=date(2025, 6, 2),
            week_end=date(2025, 6, 8),
            projection_rmse=5.0,
            rank_correlation=0.5,
            bust_rate=0.2,
            lineup_grade="A",
            n_players=10,
            category_rmse=cat_rmse,
        )
        assert wr.category_rmse["hr"] == pytest.approx(1.5)
        assert len(wr.category_rmse) == 3


class TestBacktestReport:
    def test_aggregation(self):
        """BacktestReport should store aggregate metrics correctly."""
        weeks = [
            WeekResult(
                week_start=date(2025, 6, 2),
                week_end=date(2025, 6, 8),
                projection_rmse=4.0,
                rank_correlation=0.7,
                bust_rate=0.20,
                lineup_grade="A",
                n_players=10,
            ),
            WeekResult(
                week_start=date(2025, 6, 9),
                week_end=date(2025, 6, 15),
                projection_rmse=6.0,
                rank_correlation=0.5,
                bust_rate=0.40,
                lineup_grade="B",
                n_players=12,
            ),
        ]
        report = BacktestReport(
            weeks=weeks,
            avg_rmse=5.0,
            avg_rank_correlation=0.6,
            avg_bust_rate=0.30,
            grade_distribution={"A": 1, "B": 1, "C": 0},
        )
        assert report.avg_rmse == pytest.approx(5.0)
        assert report.avg_rank_correlation == pytest.approx(0.6)
        assert report.avg_bust_rate == pytest.approx(0.30)
        assert report.grade_distribution["A"] == 1
        assert report.grade_distribution["B"] == 1
        assert len(report.weeks) == 2

    def test_empty_report(self):
        """BacktestReport with zero weeks should be constructible."""
        report = BacktestReport(
            weeks=[],
            avg_rmse=float("inf"),
            avg_rank_correlation=0.0,
            avg_bust_rate=0.0,
            grade_distribution={"A": 0, "B": 0, "C": 0},
        )
        assert len(report.weeks) == 0
        assert report.grade_distribution["A"] == 0


# ── Helper Tests ──────────────────────────────────────────────────────


class TestParseIP:
    def test_normal(self):
        assert _parse_ip("6.0") == pytest.approx(6.0)

    def test_one_third(self):
        assert _parse_ip("6.1") == pytest.approx(6.0 + 1 / 3)

    def test_two_thirds(self):
        assert _parse_ip("6.2") == pytest.approx(6.0 + 2 / 3)

    def test_zero(self):
        assert _parse_ip("0") == pytest.approx(0.0)

    def test_empty_string(self):
        assert _parse_ip("") == pytest.approx(0.0)

    def test_invalid(self):
        assert _parse_ip("abc") == pytest.approx(0.0)


class TestAggregateHittingGames:
    def test_empty_games(self):
        assert _aggregate_hitting_games([]) == {}

    def test_single_game(self):
        games = [
            {
                "hits": 2,
                "atBats": 4,
                "homeRuns": 1,
                "rbi": 3,
                "stolenBases": 0,
                "runs": 2,
                "baseOnBalls": 1,
                "hitByPitch": 0,
                "sacFlies": 0,
            }
        ]
        result = _aggregate_hitting_games(games)
        assert result["hr"] == 1.0
        assert result["rbi"] == 3.0
        assert result["avg"] == pytest.approx(0.500)  # 2/4
        assert result["obp"] == pytest.approx(0.600)  # 3/5
        assert result["games"] == 1.0

    def test_multi_game_rate_stats(self):
        """Rate stats should be weighted averages, not simple averages."""
        games = [
            {
                "hits": 1,
                "atBats": 4,
                "homeRuns": 0,
                "rbi": 0,
                "stolenBases": 0,
                "runs": 0,
                "baseOnBalls": 0,
                "hitByPitch": 0,
                "sacFlies": 0,
            },
            {
                "hits": 3,
                "atBats": 4,
                "homeRuns": 1,
                "rbi": 2,
                "stolenBases": 1,
                "runs": 1,
                "baseOnBalls": 1,
                "hitByPitch": 0,
                "sacFlies": 0,
            },
        ]
        result = _aggregate_hitting_games(games)
        assert result["avg"] == pytest.approx(4 / 8)  # 4H / 8AB
        assert result["hr"] == 1.0
        assert result["games"] == 2.0


class TestAggregatePitchingGames:
    def test_empty_games(self):
        assert _aggregate_pitching_games([]) == {}

    def test_single_game(self):
        games = [
            {
                "inningsPitched": "7.0",
                "strikeOuts": 9,
                "wins": 1,
                "losses": 0,
                "saves": 0,
                "earnedRuns": 2,
                "baseOnBalls": 1,
                "hits": 5,
            }
        ]
        result = _aggregate_pitching_games(games)
        assert result["k"] == 9.0
        assert result["w"] == 1.0
        assert result["era"] == pytest.approx(2 * 9 / 7.0, abs=0.01)
        assert result["whip"] == pytest.approx((1 + 5) / 7.0, abs=0.01)

    def test_zero_ip_avoids_division(self):
        """ERA and WHIP should be 0 when IP is 0."""
        games = [
            {
                "inningsPitched": "0",
                "strikeOuts": 0,
                "wins": 0,
                "losses": 1,
                "saves": 0,
                "earnedRuns": 3,
                "baseOnBalls": 2,
                "hits": 1,
            }
        ]
        result = _aggregate_pitching_games(games)
        assert result["era"] == 0.0
        assert result["whip"] == 0.0
        assert result["l"] == 1.0


# ── Projection Scaling Tests ─────────────────────────────────────────


class TestBuildProjectedStats:
    def test_scales_counting_stats(self, sample_roster):
        """Counting stats should be divided by 26 weeks."""
        actual_ids = sample_roster["mlb_id"].tolist()
        result = _build_projected_stats(sample_roster, actual_ids)
        # Original HR for first player
        orig_hr = sample_roster.iloc[0]["hr"]
        if orig_hr > 0:
            assert result.iloc[0]["hr"] == pytest.approx(orig_hr / 26)

    def test_rate_stats_unchanged(self, sample_roster):
        """AVG and OBP should not be scaled."""
        actual_ids = sample_roster["mlb_id"].tolist()
        result = _build_projected_stats(sample_roster, actual_ids)
        for stat in ("avg", "obp", "era", "whip"):
            if stat in result.columns:
                orig = sample_roster.iloc[0][stat]
                assert result.iloc[0][stat] == pytest.approx(orig, abs=0.001)

    def test_filters_to_actual_ids(self, sample_roster):
        """Should only include players with matching actual IDs."""
        small_ids = sample_roster["mlb_id"].tolist()[:3]
        result = _build_projected_stats(sample_roster, small_ids)
        assert len(result) == 3

    def test_empty_roster(self):
        """Should handle empty roster gracefully."""
        empty = pd.DataFrame()
        result = _build_projected_stats(empty, [1, 2, 3])
        assert result.empty


# ── Fetch Tests ───────────────────────────────────────────────────────


class TestFetchWeeklyActuals:
    def test_returns_dataframe_when_api_unavailable(self):
        """Should return empty DataFrame when statsapi is not importable."""
        with patch.dict("sys.modules", {"statsapi": None}):
            result = fetch_weekly_actuals(date(2025, 6, 2), date(2025, 6, 8))
            assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self, sample_actuals):
        """Actuals DataFrame should have all category columns."""
        for cat in ALL_CATEGORIES:
            assert cat in sample_actuals.columns

    def test_player_ids_filter(self):
        """Should accept custom player IDs."""
        with patch.dict("sys.modules", {"statsapi": None}):
            result = fetch_weekly_actuals(
                date(2025, 6, 2),
                date(2025, 6, 8),
                player_ids=[592450, 660271],
            )
            assert isinstance(result, pd.DataFrame)


# ── Backtest Week Tests ───────────────────────────────────────────────


class TestBacktestWeek:
    def test_no_actuals_returns_default(self, sample_roster):
        """When actuals are empty, should return default WeekResult."""
        with patch(
            "src.optimizer.backtest_runner.fetch_weekly_actuals",
            return_value=pd.DataFrame(),
        ):
            result = backtest_week(
                date(2025, 6, 2),
                date(2025, 6, 8),
                sample_roster,
            )
            assert result.n_players == 0
            assert result.lineup_grade == "C"

    def test_with_mocked_actuals(self, sample_roster, sample_actuals):
        """Should produce valid metrics when actuals are provided."""
        with patch(
            "src.optimizer.backtest_runner.fetch_weekly_actuals",
            return_value=sample_actuals,
        ):
            result = backtest_week(
                date(2025, 6, 2),
                date(2025, 6, 8),
                sample_roster,
            )
            assert result.n_players > 0
            assert result.lineup_grade in ("A", "B", "C")
            assert isinstance(result.projection_rmse, float)
            assert -1.0 <= result.rank_correlation <= 1.0
            assert 0.0 <= result.bust_rate <= 1.0


# ── Multi-Week Runner Tests ───────────────────────────────────────────


class TestRunBacktest:
    def test_single_week(self, sample_roster, sample_actuals):
        """Run a single-week backtest and verify report structure."""
        weeks = [(date(2025, 6, 2), date(2025, 6, 8))]
        with patch(
            "src.optimizer.backtest_runner.fetch_weekly_actuals",
            return_value=sample_actuals,
        ):
            report = run_backtest(weeks, sample_roster)
            assert isinstance(report, BacktestReport)
            assert len(report.weeks) == 1
            assert isinstance(report.avg_rmse, float)
            assert isinstance(report.avg_rank_correlation, float)
            assert isinstance(report.avg_bust_rate, float)
            assert sum(report.grade_distribution.values()) == 1

    def test_multi_week(self, sample_roster, sample_actuals):
        """Run multi-week backtest and verify aggregation."""
        weeks = [
            (date(2025, 6, 2), date(2025, 6, 8)),
            (date(2025, 6, 9), date(2025, 6, 15)),
            (date(2025, 6, 16), date(2025, 6, 22)),
        ]
        with patch(
            "src.optimizer.backtest_runner.fetch_weekly_actuals",
            return_value=sample_actuals,
        ):
            report = run_backtest(weeks, sample_roster)
            assert len(report.weeks) == 3
            assert sum(report.grade_distribution.values()) == 3

    def test_all_empty_weeks(self, sample_roster):
        """All empty weeks should produce a valid but degenerate report."""
        weeks = [(date(2025, 6, 2), date(2025, 6, 8))]
        with patch(
            "src.optimizer.backtest_runner.fetch_weekly_actuals",
            return_value=pd.DataFrame(),
        ):
            report = run_backtest(weeks, sample_roster)
            assert len(report.weeks) == 1
            assert report.weeks[0].n_players == 0
            # avg_rmse should be inf when no valid weeks
            assert report.avg_rmse == float("inf")


# ── Report Formatting Tests ───────────────────────────────────────────


class TestFormatReport:
    def test_format_includes_header(self):
        """Formatted report should include the header."""
        report = BacktestReport(
            weeks=[],
            avg_rmse=5.0,
            avg_rank_correlation=0.6,
            avg_bust_rate=0.25,
            grade_distribution={"A": 0, "B": 0, "C": 0},
        )
        text = format_report(report)
        assert "BACKTEST REPORT" in text

    def test_format_includes_metrics(self, sample_roster, sample_actuals):
        """Formatted report should include accuracy metrics."""
        weeks = [(date(2025, 6, 2), date(2025, 6, 8))]
        with patch(
            "src.optimizer.backtest_runner.fetch_weekly_actuals",
            return_value=sample_actuals,
        ):
            report = run_backtest(weeks, sample_roster)
            text = format_report(report)
            assert "RMSE" in text
            assert "rank correlation" in text
            assert "bust rate" in text
            assert "Grade distribution" in text

    def test_format_per_week_detail(self):
        """Formatted report should include per-week lines."""
        weeks = [
            WeekResult(
                week_start=date(2025, 6, 2),
                week_end=date(2025, 6, 8),
                projection_rmse=4.5,
                rank_correlation=0.72,
                bust_rate=0.15,
                lineup_grade="A",
                n_players=10,
                category_rmse={"hr": 1.2, "rbi": 2.5},
            ),
        ]
        report = BacktestReport(
            weeks=weeks,
            avg_rmse=4.5,
            avg_rank_correlation=0.72,
            avg_bust_rate=0.15,
            grade_distribution={"A": 1, "B": 0, "C": 0},
        )
        text = format_report(report)
        assert "2025-06-02" in text
        assert "PER-WEEK DETAIL" in text
        assert "Category RMSE" in text


# ── Constants Tests ───────────────────────────────────────────────────


class TestConstants:
    def test_backtest_player_ids_count(self):
        """Should have 20 player IDs."""
        assert len(BACKTEST_PLAYER_IDS) == 20

    def test_all_categories(self):
        """ALL_CATEGORIES should have 12 categories."""
        assert len(ALL_CATEGORIES) == 12
        assert "hr" in ALL_CATEGORIES
        assert "era" in ALL_CATEGORIES
