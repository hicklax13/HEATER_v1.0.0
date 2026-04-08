"""Backtesting framework: score engine recommendations against actual outcomes.

Pure computation module. Takes pre-computed predictions and actuals, returns
accuracy metrics. No data fetching, no API calls, no Streamlit dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.valuation import LeagueConfig, SGPCalculator


@dataclass
class BacktestRunner:
    """Score engine recommendations against actual outcomes.

    Supports: trade recommendations, start/sit decisions,
    category win predictions. Each has a specific scoring method.
    """

    config: LeagueConfig = field(default_factory=LeagueConfig)
    sgp_calc: SGPCalculator | None = None

    def __post_init__(self) -> None:
        if self.sgp_calc is None:
            self.sgp_calc = SGPCalculator(self.config)

    # ── Trade Recommendation Scoring ────────────────────────────────

    def score_trade_recommendation(
        self,
        recommendation: dict,
        actuals: pd.DataFrame,
        weeks_forward: int = 4,
    ) -> dict:
        """Score a trade recommendation against actual stats.

        Args:
            recommendation: dict with giving_ids, receiving_ids, predicted_sgp_gain.
            actuals: DataFrame with player_id, week, and per-stat columns.
            weeks_forward: number of weeks to evaluate after the trade.

        Returns:
            {predicted_sgp_gain, actual_sgp_gain, error, pct_error}
        """
        giving_ids = set(recommendation["giving_ids"])
        receiving_ids = set(recommendation["receiving_ids"])
        predicted = recommendation.get("predicted_sgp_gain", 0.0)

        # Filter actuals to the evaluation window
        week_col = "week" if "week" in actuals.columns else actuals.columns[1]
        eval_df = actuals[actuals[week_col] <= weeks_forward]

        # Compute actual SGP for received players minus given players
        recv_sgp = self._sum_player_sgp(eval_df, receiving_ids)
        give_sgp = self._sum_player_sgp(eval_df, giving_ids)
        actual_gain = recv_sgp - give_sgp

        error = predicted - actual_gain
        pct_error = abs(error / actual_gain) if abs(actual_gain) > 1e-9 else float("inf")

        return {
            "predicted_sgp_gain": predicted,
            "actual_sgp_gain": actual_gain,
            "error": error,
            "pct_error": pct_error,
        }

    def _sum_player_sgp(self, df: pd.DataFrame, player_ids: set) -> float:
        """Sum total SGP for a set of players from actual stats."""
        subset = df[df["player_id"].isin(player_ids)]
        if subset.empty:
            return 0.0

        # Aggregate stats across weeks per player, then sum SGP
        total_sgp = 0.0
        for pid, grp in subset.groupby("player_id"):
            agg = grp.sum(numeric_only=True)
            player_series = agg.rename(lambda c: c.lower())
            total_sgp += self.sgp_calc.total_sgp(player_series)
        return total_sgp

    # ── Start/Sit Decision Scoring ──────────────────────────────────

    def score_start_sit_decision(
        self,
        decision: dict,
        actual_stats: pd.DataFrame,
    ) -> dict:
        """Score a start/sit recommendation against actual game stats.

        Args:
            decision: dict with recommended_player_id, alternative_id.
            actual_stats: DataFrame with player_id and stat columns.

        Returns:
            {correct: bool, margin: float}
        """
        rec_id = decision["recommended_player_id"]
        alt_id = decision["alternative_id"]

        rec_sgp = self._player_sgp_from_row(actual_stats, rec_id)
        alt_sgp = self._player_sgp_from_row(actual_stats, alt_id)

        return {
            "correct": bool(rec_sgp >= alt_sgp),
            "margin": float(rec_sgp - alt_sgp),
        }

    def _player_sgp_from_row(self, df: pd.DataFrame, player_id: int) -> float:
        """Get total SGP for a single player from actual stats."""
        row = df[df["player_id"] == player_id]
        if row.empty:
            return 0.0
        agg = row.sum(numeric_only=True)
        player_series = agg.rename(lambda c: c.lower())
        return self.sgp_calc.total_sgp(player_series)

    # ── Category Prediction Scoring ─────────────────────────────────

    @staticmethod
    def score_category_prediction(
        predicted_wins: dict[str, bool],
        actual_wins: dict[str, bool],
    ) -> dict:
        """Score predicted category W/L against actual results.

        Args:
            predicted_wins: {category: True/False} for predicted win per category.
            actual_wins: {category: True/False} for actual win per category.

        Returns:
            {correct_categories, total, accuracy}
        """
        categories = set(predicted_wins.keys()) & set(actual_wins.keys())
        if not categories:
            return {"correct_categories": 0, "total": 0, "accuracy": 0.0}

        correct = sum(1 for cat in categories if predicted_wins[cat] == actual_wins[cat])
        total = len(categories)

        return {
            "correct_categories": correct,
            "total": total,
            "accuracy": correct / total,
        }

    # ── Calibration Report ──────────────────────────────────────────

    @staticmethod
    def calibration_report(predictions: list[dict]) -> dict:
        """Compute bucket-based calibration from probability predictions.

        Args:
            predictions: list of {predicted: float (0-1), actual: bool}.

        Returns:
            {buckets: [{range, predicted_mean, actual_rate, count}], brier_score}
        """
        if not predictions:
            return {"buckets": [], "brier_score": float("nan")}

        boundaries = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        buckets = []

        brier_sum = 0.0
        for pred in predictions:
            brier_sum += (pred["predicted"] - (1.0 if pred["actual"] else 0.0)) ** 2

        for i, label in enumerate(labels):
            lo, hi = boundaries[i], boundaries[i + 1]
            items = [p for p in predictions if lo <= p["predicted"] < hi]
            if not items:
                buckets.append({"range": label, "predicted_mean": 0.0, "actual_rate": 0.0, "count": 0})
                continue
            pred_mean = sum(p["predicted"] for p in items) / len(items)
            actual_rate = sum(1 for p in items if p["actual"]) / len(items)
            buckets.append(
                {
                    "range": label,
                    "predicted_mean": round(pred_mean, 4),
                    "actual_rate": round(actual_rate, 4),
                    "count": len(items),
                }
            )

        return {
            "buckets": buckets,
            "brier_score": round(brier_sum / len(predictions), 4),
        }

    # ── Aggregate Report ────────────────────────────────────────────

    @staticmethod
    def aggregate_report(scores: list[dict]) -> dict:
        """Aggregate individual score dicts into summary statistics.

        Handles output from score_trade_recommendation (has 'error'),
        score_start_sit_decision (has 'correct'), and
        score_category_prediction (has 'accuracy').

        Returns:
            {mean_error, rmse, correct_rate, n_scored}
        """
        if not scores:
            return {"mean_error": float("nan"), "rmse": float("nan"), "correct_rate": float("nan"), "n_scored": 0}

        # Collect errors (from trade scoring)
        errors = [s["error"] for s in scores if "error" in s and math.isfinite(s["error"])]

        # Collect correctness (from start/sit and category prediction)
        correct_flags = []
        for s in scores:
            if "correct" in s:
                correct_flags.append(1.0 if s["correct"] else 0.0)
            elif "accuracy" in s:
                correct_flags.append(s["accuracy"])

        mean_error = float(np.mean(errors)) if errors else float("nan")
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2))) if errors else float("nan")
        correct_rate = float(np.mean(correct_flags)) if correct_flags else float("nan")

        return {
            "mean_error": round(mean_error, 4) if math.isfinite(mean_error) else mean_error,
            "rmse": round(rmse, 4) if math.isfinite(rmse) else rmse,
            "correct_rate": round(correct_rate, 4) if math.isfinite(correct_rate) else correct_rate,
            "n_scored": len(scores),
        }
