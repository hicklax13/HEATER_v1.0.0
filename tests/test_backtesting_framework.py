"""Tests for backtesting framework."""

import math

import pandas as pd
import pytest

from src.backtesting_framework import BacktestRunner
from src.valuation import LeagueConfig


@pytest.fixture
def runner():
    return BacktestRunner()


@pytest.fixture
def config():
    return LeagueConfig()


def _make_actuals(rows: list[dict]) -> pd.DataFrame:
    """Build an actuals DataFrame from a list of stat dicts."""
    return pd.DataFrame(rows)


# ── score_trade_recommendation ──────────────────────────────────────


class TestScoreTradeRecommendation:
    def test_positive_outcome(self, runner):
        """Received player outperforms given player -> positive actual gain."""
        recommendation = {
            "giving_ids": [1],
            "receiving_ids": [2],
            "predicted_sgp_gain": 2.0,
        }
        # Player 2 has better counting stats than player 1
        actuals = _make_actuals(
            [
                {
                    "player_id": 1,
                    "week": 1,
                    "r": 5,
                    "hr": 1,
                    "rbi": 4,
                    "sb": 0,
                    "avg": 0.250,
                    "obp": 0.300,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 20,
                    "ip": 0,
                    "h": 5,
                    "bb": 1,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
                {
                    "player_id": 2,
                    "week": 1,
                    "r": 10,
                    "hr": 4,
                    "rbi": 10,
                    "sb": 2,
                    "avg": 0.300,
                    "obp": 0.370,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 20,
                    "ip": 0,
                    "h": 6,
                    "bb": 2,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
            ]
        )
        result = runner.score_trade_recommendation(recommendation, actuals)
        assert result["actual_sgp_gain"] > 0
        assert "error" in result
        assert "pct_error" in result
        assert result["predicted_sgp_gain"] == 2.0

    def test_negative_outcome(self, runner):
        """Given player outperforms received -> negative actual gain."""
        recommendation = {
            "giving_ids": [1],
            "receiving_ids": [2],
            "predicted_sgp_gain": 1.0,
        }
        # Player 1 is better than player 2
        actuals = _make_actuals(
            [
                {
                    "player_id": 1,
                    "week": 1,
                    "r": 15,
                    "hr": 5,
                    "rbi": 12,
                    "sb": 3,
                    "avg": 0.310,
                    "obp": 0.380,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 20,
                    "ip": 0,
                    "h": 6,
                    "bb": 2,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
                {
                    "player_id": 2,
                    "week": 1,
                    "r": 2,
                    "hr": 0,
                    "rbi": 1,
                    "sb": 0,
                    "avg": 0.200,
                    "obp": 0.250,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 20,
                    "ip": 0,
                    "h": 4,
                    "bb": 1,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
            ]
        )
        result = runner.score_trade_recommendation(recommendation, actuals)
        assert result["actual_sgp_gain"] < 0
        # Prediction was +1.0 but actual was negative, so error > 0
        assert result["error"] > 0

    def test_weeks_forward_filter(self, runner):
        """Only weeks within the window are included."""
        recommendation = {
            "giving_ids": [1],
            "receiving_ids": [2],
            "predicted_sgp_gain": 0.5,
        }
        actuals = _make_actuals(
            [
                {
                    "player_id": 1,
                    "week": 1,
                    "r": 5,
                    "hr": 1,
                    "rbi": 3,
                    "sb": 0,
                    "avg": 0.0,
                    "obp": 0.0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 0,
                    "ip": 0,
                    "h": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
                {
                    "player_id": 2,
                    "week": 1,
                    "r": 6,
                    "hr": 2,
                    "rbi": 5,
                    "sb": 1,
                    "avg": 0.0,
                    "obp": 0.0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 0,
                    "ip": 0,
                    "h": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
                # Week 10 should be excluded with weeks_forward=4
                {
                    "player_id": 2,
                    "week": 10,
                    "r": 100,
                    "hr": 50,
                    "rbi": 100,
                    "sb": 20,
                    "avg": 0.0,
                    "obp": 0.0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 0,
                    "ip": 0,
                    "h": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
            ]
        )
        result_4w = runner.score_trade_recommendation(recommendation, actuals, weeks_forward=4)
        result_all = runner.score_trade_recommendation(recommendation, actuals, weeks_forward=10)
        # With week 10 included, actual gain should be much larger
        assert result_all["actual_sgp_gain"] > result_4w["actual_sgp_gain"]

    def test_empty_actuals(self, runner):
        """Empty actuals returns zero gain."""
        recommendation = {
            "giving_ids": [1],
            "receiving_ids": [2],
            "predicted_sgp_gain": 1.0,
        }
        actuals = pd.DataFrame(columns=["player_id", "week", "r", "hr"])
        result = runner.score_trade_recommendation(recommendation, actuals)
        assert result["actual_sgp_gain"] == 0.0
        assert result["pct_error"] == float("inf")


# ── score_start_sit_decision ────────────────────────────────────────


class TestScoreStartSitDecision:
    def test_correct_decision(self, runner):
        """Recommended player outscores alternative."""
        decision = {"recommended_player_id": 1, "alternative_id": 2}
        actual_stats = _make_actuals(
            [
                {
                    "player_id": 1,
                    "r": 3,
                    "hr": 2,
                    "rbi": 5,
                    "sb": 1,
                    "avg": 0.300,
                    "obp": 0.370,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 10,
                    "ip": 0,
                    "h": 3,
                    "bb": 1,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
                {
                    "player_id": 2,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0.100,
                    "obp": 0.150,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 10,
                    "ip": 0,
                    "h": 1,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
            ]
        )
        result = runner.score_start_sit_decision(decision, actual_stats)
        assert result["correct"] is True
        assert result["margin"] > 0

    def test_incorrect_decision(self, runner):
        """Alternative outscores recommended player."""
        decision = {"recommended_player_id": 1, "alternative_id": 2}
        actual_stats = _make_actuals(
            [
                {
                    "player_id": 1,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0.0,
                    "obp": 0.0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 5,
                    "ip": 0,
                    "h": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
                {
                    "player_id": 2,
                    "r": 5,
                    "hr": 3,
                    "rbi": 8,
                    "sb": 2,
                    "avg": 0.350,
                    "obp": 0.400,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 20,
                    "ip": 0,
                    "h": 7,
                    "bb": 2,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
            ]
        )
        result = runner.score_start_sit_decision(decision, actual_stats)
        assert result["correct"] is False
        assert result["margin"] < 0

    def test_missing_player(self, runner):
        """Missing player gets 0 SGP."""
        decision = {"recommended_player_id": 1, "alternative_id": 999}
        actual_stats = _make_actuals(
            [
                {
                    "player_id": 1,
                    "r": 3,
                    "hr": 1,
                    "rbi": 3,
                    "sb": 0,
                    "avg": 0.0,
                    "obp": 0.0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0.0,
                    "whip": 0.0,
                    "ab": 0,
                    "ip": 0,
                    "h": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                },
            ]
        )
        result = runner.score_start_sit_decision(decision, actual_stats)
        assert result["correct"] is True  # 999 not found -> 0 SGP


# ── score_category_prediction ───────────────────────────────────────


class TestScoreCategoryPrediction:
    def test_perfect_prediction(self, runner):
        predicted = {"R": True, "HR": False, "RBI": True, "SB": False}
        actual = {"R": True, "HR": False, "RBI": True, "SB": False}
        result = runner.score_category_prediction(predicted, actual)
        assert result["accuracy"] == 1.0
        assert result["correct_categories"] == 4
        assert result["total"] == 4

    def test_half_correct(self, runner):
        predicted = {"R": True, "HR": True, "RBI": True, "SB": True}
        actual = {"R": True, "HR": False, "RBI": True, "SB": False}
        result = runner.score_category_prediction(predicted, actual)
        assert result["accuracy"] == 0.5
        assert result["correct_categories"] == 2

    def test_empty_predictions(self, runner):
        result = runner.score_category_prediction({}, {})
        assert result["accuracy"] == 0.0
        assert result["total"] == 0

    def test_partial_overlap(self, runner):
        """Only shared categories are scored."""
        predicted = {"R": True, "HR": True, "SV": False}
        actual = {"R": True, "HR": False, "K": True}
        result = runner.score_category_prediction(predicted, actual)
        assert result["total"] == 2  # R and HR only
        assert result["correct_categories"] == 1  # R correct, HR wrong


# ── calibration_report ──────────────────────────────────────────────


class TestCalibrationReport:
    def test_basic_calibration(self, runner):
        predictions = [
            {"predicted": 0.1, "actual": False},
            {"predicted": 0.1, "actual": False},
            {"predicted": 0.5, "actual": True},
            {"predicted": 0.5, "actual": False},
            {"predicted": 0.9, "actual": True},
            {"predicted": 0.9, "actual": True},
        ]
        result = runner.calibration_report(predictions)
        assert len(result["buckets"]) == 5
        assert "brier_score" in result

        # 0-20% bucket: two items at 0.1, both False -> actual_rate = 0.0
        low_bucket = result["buckets"][0]
        assert low_bucket["count"] == 2
        assert low_bucket["actual_rate"] == 0.0

        # 80-100% bucket: two items at 0.9, both True -> actual_rate = 1.0
        high_bucket = result["buckets"][4]
        assert high_bucket["count"] == 2
        assert high_bucket["actual_rate"] == 1.0

    def test_empty_predictions(self, runner):
        result = runner.calibration_report([])
        assert result["buckets"] == []
        assert math.isnan(result["brier_score"])

    def test_single_prediction(self, runner):
        result = runner.calibration_report([{"predicted": 0.75, "actual": True}])
        assert result["brier_score"] == pytest.approx(0.0625, abs=0.001)
        # Should be in 60-80% bucket
        bucket_60_80 = result["buckets"][3]
        assert bucket_60_80["count"] == 1

    def test_brier_score_perfect(self, runner):
        """Perfect predictions -> brier score = 0."""
        predictions = [
            {"predicted": 1.0, "actual": True},
            {"predicted": 0.0, "actual": False},
        ]
        result = runner.calibration_report(predictions)
        assert result["brier_score"] == 0.0


# ── aggregate_report ────────────────────────────────────────────────


class TestAggregateReport:
    def test_trade_scores(self, runner):
        scores = [
            {"predicted_sgp_gain": 2.0, "actual_sgp_gain": 1.5, "error": 0.5, "pct_error": 0.33},
            {"predicted_sgp_gain": -1.0, "actual_sgp_gain": -0.5, "error": -0.5, "pct_error": 1.0},
        ]
        result = runner.aggregate_report(scores)
        assert result["mean_error"] == 0.0  # 0.5 + (-0.5) / 2
        assert result["rmse"] == pytest.approx(0.5, abs=0.01)
        assert result["n_scored"] == 2

    def test_start_sit_scores(self, runner):
        scores = [
            {"correct": True, "margin": 1.5},
            {"correct": False, "margin": -0.3},
            {"correct": True, "margin": 2.1},
        ]
        result = runner.aggregate_report(scores)
        assert result["correct_rate"] == pytest.approx(0.6667, abs=0.01)
        assert result["n_scored"] == 3

    def test_mixed_scores(self, runner):
        """Mix of trade and start/sit scores."""
        scores = [
            {"error": 0.5, "pct_error": 0.25},
            {"correct": True, "margin": 1.0},
        ]
        result = runner.aggregate_report(scores)
        assert result["mean_error"] == 0.5
        assert result["correct_rate"] == 1.0

    def test_empty_scores(self, runner):
        result = runner.aggregate_report([])
        assert math.isnan(result["mean_error"])
        assert result["n_scored"] == 0

    def test_category_prediction_in_aggregate(self, runner):
        scores = [
            {"correct_categories": 8, "total": 12, "accuracy": 0.6667},
            {"correct_categories": 10, "total": 12, "accuracy": 0.8333},
        ]
        result = runner.aggregate_report(scores)
        assert result["correct_rate"] == pytest.approx(0.75, abs=0.01)
