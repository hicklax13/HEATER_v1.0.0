"""Guard: Section G historical-backtest error metrics + ingestion path.

Item 2 (2026-05-25): extends the self-validation shipped in PR #117 with
the report's five Section G error metrics (rank MAE, cat-win RMSE, trade
Spearman, projection MAE) and a data-agnostic ingestion path that grades
each computed metric against the report's targets. No historical data ships
with the repo, so the path operates on caller-supplied records.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.engine.output.backtest_calibration import (
    cat_win_rmse,
    projection_mae,
    rank_mae,
    run_historical_backtest,
    trade_prediction_spearman,
)


def test_rank_mae_perfect_is_zero() -> None:
    assert rank_mae(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0.0
    assert rank_mae(np.array([1, 2, 3]), np.array([2, 2, 3])) == (1 / 3)


def test_cat_win_rmse() -> None:
    assert cat_win_rmse(np.array([100, 100]), np.array([100, 100])) == 0.0
    val = cat_win_rmse(np.array([100, 100]), np.array([110, 90]))
    assert abs(val - 10.0) < 1e-9


def test_trade_spearman_monotonic() -> None:
    """Perfectly rank-correlated predictions → ρ = 1."""
    pred = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    realized = np.array([1, 2, 3, 4, 5])
    assert abs(trade_prediction_spearman(pred, realized) - 1.0) < 1e-9


def test_trade_spearman_too_few_points_is_nan() -> None:
    assert np.isnan(trade_prediction_spearman(np.array([0.1, 0.2]), np.array([1, 2])))


def test_projection_mae() -> None:
    assert projection_mae(np.array([20, 30]), np.array([22, 27])) == 2.5


def test_run_historical_backtest_grades_against_targets() -> None:
    """A near-perfect backtest passes every available Section G target."""
    records = pd.DataFrame(
        {
            "predicted_rank": [1, 2, 3, 4],
            "actual_rank": [1, 2, 3, 4],
            "predicted_cat_wins": [160, 150, 140, 130],
            "actual_cat_wins": [161, 150, 139, 131],
            "predicted_playoff_prob": [0.95, 0.80, 0.20, 0.05],
            "made_playoffs": [1, 1, 0, 0],
            "predicted_trade_delta": [0.1, 0.2, 0.3, 0.4],
            "realized_rank_change": [1, 2, 3, 4],
            "predicted_hr": [30, 25, 20, 15],
            "actual_hr": [31, 24, 21, 14],
            "predicted_avg": [0.300, 0.270, 0.250, 0.240],
            "actual_avg": [0.305, 0.268, 0.252, 0.241],
        }
    )
    report = run_historical_backtest(records)
    assert report["n_records"] == 4
    metrics = report["metrics"]
    for key in ("rank_mae", "cat_win_rmse", "brier", "trade_spearman", "hr_mae", "avg_mae"):
        assert key in metrics, f"{key} should be computed"
    # All these synthetic predictions are near-perfect → all targets pass.
    for name, info in report["targets"].items():
        assert info["passing"] is True, f"{name} should pass: {info}"


def test_run_historical_backtest_partial_columns() -> None:
    """Only computes metrics for columns that are present; no crash."""
    records = pd.DataFrame({"predicted_rank": [1, 2], "actual_rank": [1, 3]})
    report = run_historical_backtest(records)
    assert "rank_mae" in report["metrics"]
    assert "cat_win_rmse" not in report["metrics"]
    assert report["n_records"] == 2


def test_run_historical_backtest_empty() -> None:
    report = run_historical_backtest(pd.DataFrame())
    assert report["n_records"] == 0
    assert report["metrics"] == {}
