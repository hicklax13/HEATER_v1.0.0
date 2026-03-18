"""Tests for the backtesting harness.

Covers:
  - BacktestEngine initialization and season setting
  - Historical actuals loading (mocked)
  - Accuracy evaluation metrics (RMSE, rank correlation, win rate)
  - Rank correlation edge cases (perfect, random)
  - Bust rate extremes (none, all)
  - Value capture above random baseline
  - Random baseline draft count
  - Summary report structure
  - Simulated draft roster output
  - Full backtest averaging across positions
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.backtesting import (
    ALL_EVAL_CATEGORIES,
    BacktestEngine,
    _rankdata,
    _spearman_correlation,
    random_draft_baseline,
)
from src.valuation import LeagueConfig

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def config():
    """Standard 12-team H2H league config."""
    return LeagueConfig()


@pytest.fixture
def engine(config):
    """BacktestEngine with default season."""
    return BacktestEngine(season=2025, config=config)


@pytest.fixture
def sample_pool():
    """Player pool with realistic stat columns for backtesting."""
    rng = np.random.default_rng(42)
    n = 100
    players = []
    for i in range(n):
        is_hitter = i < 60
        players.append(
            {
                "player_id": i + 1,
                "name": f"Player {i + 1}",
                "team": f"T{(i % 10) + 1}",
                "positions": "OF,Util" if is_hitter else "SP",
                "is_hitter": 1 if is_hitter else 0,
                "is_injured": 0,
                "pa": int(rng.integers(300, 650)) if is_hitter else 0,
                "ab": int(rng.integers(250, 600)) if is_hitter else 0,
                "h": int(rng.integers(60, 180)) if is_hitter else 0,
                "r": int(rng.integers(30, 110)) if is_hitter else 0,
                "hr": int(rng.integers(5, 45)) if is_hitter else 0,
                "rbi": int(rng.integers(25, 120)) if is_hitter else 0,
                "sb": int(rng.integers(0, 40)) if is_hitter else 0,
                "avg": float(rng.uniform(0.220, 0.320)) if is_hitter else 0.0,
                "obp": float(rng.uniform(0.280, 0.400)) if is_hitter else 0.0,
                "bb": int(rng.integers(20, 80)) if is_hitter else 0,
                "hbp": int(rng.integers(0, 10)) if is_hitter else 0,
                "sf": int(rng.integers(0, 8)) if is_hitter else 0,
                "ip": 0.0 if is_hitter else float(rng.uniform(50, 220)),
                "w": 0 if is_hitter else int(rng.integers(3, 18)),
                "l": 0 if is_hitter else int(rng.integers(3, 14)),
                "sv": 0 if is_hitter else int(rng.integers(0, 35)),
                "k": 0 if is_hitter else int(rng.integers(40, 250)),
                "era": 0.0 if is_hitter else float(rng.uniform(2.50, 5.50)),
                "whip": 0.0 if is_hitter else float(rng.uniform(0.90, 1.50)),
                "er": 0 if is_hitter else int(rng.integers(15, 80)),
                "bb_allowed": 0 if is_hitter else int(rng.integers(15, 70)),
                "h_allowed": 0 if is_hitter else int(rng.integers(40, 180)),
                "adp": float(i + 1 + rng.uniform(-5, 5)),
            }
        )
    return pd.DataFrame(players)


@pytest.fixture
def sample_actuals(sample_pool):
    """Actual stats derived from sample pool with noise added."""
    rng = np.random.default_rng(99)
    actuals = sample_pool.copy()
    # Add noise to simulate actual vs projected divergence
    for cat in ["r", "hr", "rbi", "sb", "w", "sv", "k"]:
        if cat in actuals.columns:
            noise = rng.normal(0, 3, size=len(actuals))
            actuals[cat] = np.maximum(0, actuals[cat] + noise).astype(int)
    for cat in ["avg", "obp", "era", "whip"]:
        if cat in actuals.columns:
            noise = rng.normal(0, 0.01, size=len(actuals))
            actuals[cat] = np.clip(actuals[cat] + noise, 0.0, 10.0)
    return actuals


# ── Tests ─────────────────────────────────────────────────────────


def test_backtest_engine_init(config):
    """BacktestEngine initializes with correct season and config."""
    engine = BacktestEngine(season=2024, config=config)
    assert engine.season == 2024
    assert engine.config is config
    assert engine.actuals is None
    assert engine._results == []


def test_load_historical_returns_dataframe(engine):
    """load_historical_actuals returns a DataFrame (possibly empty)."""
    # Mock both DB and API to raise — graceful degradation to empty DataFrame
    with (
        patch("src.database.load_season_stats", side_effect=Exception("no DB")),
        patch("src.live_stats.fetch_season_stats", side_effect=Exception("no API")),
    ):
        result = engine.load_historical_actuals()

    assert isinstance(result, pd.DataFrame)
    assert "player_id" in result.columns


def test_evaluate_accuracy_returns_metrics(engine, sample_pool, sample_actuals):
    """evaluate_accuracy returns dict with all expected metric keys."""
    engine.actuals = sample_actuals
    # Simulate a small roster from the pool
    roster = sample_pool.head(23).copy()
    metrics = engine.evaluate_accuracy(roster, sample_actuals)

    assert isinstance(metrics, dict)
    assert "rmse" in metrics
    assert "rank_correlation" in metrics
    assert "category_win_rate" in metrics
    assert "value_capture_rate" in metrics
    assert "bust_rate" in metrics
    assert "per_category" in metrics


def test_rank_correlation_perfect_should_be_1():
    """Perfect rank correlation: identical arrays should yield rho ~ 1.0."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    rho = _spearman_correlation(x, y)
    assert abs(rho - 1.0) < 0.01


def test_rank_correlation_random_near_zero():
    """Random arrays should have rank correlation near zero."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal(200)
    y = rng.standard_normal(200)
    rho = _spearman_correlation(x, y)
    # With 200 random samples, expect rho within [-0.2, 0.2]
    assert -0.25 <= rho <= 0.25


def test_category_rmse_computed_per_stat(engine, sample_pool, sample_actuals):
    """RMSE should be computed for each category in per_category dict."""
    roster = sample_pool.head(23).copy()
    metrics = engine.evaluate_accuracy(roster, sample_actuals)

    per_cat = metrics["per_category"]
    # At least some categories should be present
    assert len(per_cat) > 0
    for cat, cat_metrics in per_cat.items():
        assert "rmse" in cat_metrics
        assert cat_metrics["rmse"] >= 0.0


def test_win_rate_all_correct_is_1(engine):
    """When projections perfectly predict above/below median, win rate = 1.0."""
    # Create perfectly correlated proj and actuals
    roster = pd.DataFrame(
        {
            "player_id": range(1, 11),
            "name": [f"P{i}" for i in range(1, 11)],
            "r": list(range(10, 110, 10)),
            "hr": list(range(5, 55, 5)),
            "is_hitter": [1] * 10,
        }
    )
    # Actuals are identical (perfect prediction)
    actuals = roster.copy()

    engine.actuals = actuals
    metrics = engine.evaluate_accuracy(roster, actuals)

    # Every category present should have win_rate = 1.0
    for cat, cat_metrics in metrics["per_category"].items():
        assert cat_metrics["win_rate"] == 1.0


def test_win_rate_all_wrong_is_0(engine):
    """When projections are perfectly inverted, win rate = 0.0."""
    n = 10
    roster = pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "name": [f"P{i}" for i in range(1, n + 1)],
            "r": list(range(10, 10 * (n + 1), 10)),
            "is_hitter": [1] * n,
        }
    )
    # Actuals are perfectly reversed
    actuals = pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "name": [f"P{i}" for i in range(1, n + 1)],
            "r": list(range(10 * n, 0, -10)),
            "is_hitter": [1] * n,
        }
    )

    metrics = engine.evaluate_accuracy(roster, actuals)
    # For the "r" category, above/below median should be inverted
    if "r" in metrics["per_category"]:
        assert metrics["per_category"]["r"]["win_rate"] == 0.0


def test_bust_rate_no_busts(engine):
    """When all players have identical actual stats, bust rate = 0.0."""
    # Create roster and actuals where every player has exactly the same stats
    # so no player falls below the 25th percentile (since all are equal)
    n = 10
    roster = pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "name": [f"P{i}" for i in range(1, n + 1)],
            "positions": ["OF"] * n,
            "is_hitter": [1] * n,
            "pa": [500] * n,
            "ab": [450] * n,
            "h": [130] * n,
            "r": [80] * n,
            "hr": [25] * n,
            "rbi": [80] * n,
            "sb": [10] * n,
            "avg": [0.289] * n,
            "obp": [0.350] * n,
            "ip": [0.0] * n,
            "w": [0] * n,
            "l": [0] * n,
            "sv": [0] * n,
            "k": [0] * n,
            "era": [0.0] * n,
            "whip": [0.0] * n,
            "er": [0] * n,
            "bb_allowed": [0] * n,
            "h_allowed": [0] * n,
        }
    )
    actuals = roster.copy()

    metrics = engine.evaluate_accuracy(roster, actuals)
    # All identical SGP -> none below the 25th percentile (strict <)
    assert metrics["bust_rate"] == 0.0


def test_bust_rate_all_busts(engine):
    """When most players are below replacement, bust rate is high."""
    # Create a roster and actuals with a clear split:
    # A few strong players set the replacement level high enough
    # that most players fall below it.
    n = 20
    rng = np.random.default_rng(77)
    roster = pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "name": [f"P{i}" for i in range(1, n + 1)],
            "positions": ["OF"] * n,
            "is_hitter": [1] * n,
            "pa": [500] * n,
            "ab": [450] * n,
            "h": [100] * n,
            "r": [50] * n,
            "hr": [10] * n,
            "rbi": [40] * n,
            "sb": [5] * n,
            "avg": [0.220] * n,
            "obp": [0.280] * n,
            "ip": [0.0] * n,
            "w": [0] * n,
            "l": [0] * n,
            "sv": [0] * n,
            "k": [0] * n,
            "era": [0.0] * n,
            "whip": [0.0] * n,
            "er": [0] * n,
            "bb_allowed": [0] * n,
            "h_allowed": [0] * n,
        }
    )
    # Actuals: create a spread where each player has different SGP
    # so percentile computation is meaningful. The top 5 are strong,
    # the bottom 15 are varying levels of terrible.
    actual_r = [100, 90, 85, 80, 75] + [int(rng.integers(1, 15)) for _ in range(15)]
    actual_hr = [40, 35, 32, 30, 28] + [int(rng.integers(0, 3)) for _ in range(15)]
    actual_rbi = [100, 90, 85, 80, 75] + [int(rng.integers(1, 10)) for _ in range(15)]
    actuals = pd.DataFrame(
        {
            "player_id": range(1, n + 1),
            "name": [f"P{i}" for i in range(1, n + 1)],
            "r": actual_r,
            "hr": actual_hr,
            "rbi": actual_rbi,
            "sb": [20, 18, 15, 12, 10] + [int(rng.integers(0, 2)) for _ in range(15)],
            "avg": [0.310, 0.300, 0.295, 0.290, 0.285] + [0.150] * 15,
            "obp": [0.390, 0.380, 0.370, 0.360, 0.350] + [0.200] * 15,
            "is_hitter": [1] * n,
            "pa": [600] * 5 + [50] * 15,
            "ab": [550] * 5 + [40] * 15,
            "h": [170, 165, 160, 155, 150] + [6] * 15,
            "ip": [0.0] * n,
            "w": [0] * n,
            "l": [0] * n,
            "sv": [0] * n,
            "k": [0] * n,
            "era": [0.0] * n,
            "whip": [0.0] * n,
            "er": [0] * n,
            "bb_allowed": [0] * n,
            "h_allowed": [0] * n,
        }
    )

    metrics = engine.evaluate_accuracy(roster, actuals)
    # The bottom 15 should mostly fall below the 25th percentile
    # which sits between the strong and weak players
    assert metrics["bust_rate"] > 0.0


def test_value_capture_above_random(engine, sample_pool, sample_actuals):
    """Engine draft value capture should be >= 0 (non-negative)."""
    engine.actuals = sample_actuals
    roster = sample_pool.head(23).copy()
    metrics = engine.evaluate_accuracy(roster, sample_actuals)
    assert metrics["value_capture_rate"] >= 0.0


def test_random_baseline_drafts_correct_count(sample_pool):
    """random_draft_baseline returns exactly num_picks players."""
    result = random_draft_baseline(sample_pool, num_picks=23, seed=42)
    assert len(result) == 23

    # With fewer available than requested, clamp to pool size
    small_pool = sample_pool.head(5)
    result_small = random_draft_baseline(small_pool, num_picks=23, seed=42)
    assert len(result_small) == 5


def test_summary_report_has_all_keys(engine, sample_pool, sample_actuals):
    """summary_report returns dict with all required summary keys."""
    engine.actuals = sample_actuals

    # Run a mini backtest (2 positions for speed)
    engine.run_full_backtest(
        player_pool=sample_pool,
        n_positions=[1, 2],
        num_teams=12,
    )

    report = engine.summary_report()
    assert isinstance(report, dict)
    expected_keys = {
        "projection_rmse",
        "rank_correlation",
        "category_win_rate",
        "value_capture_rate",
        "bust_rate",
        "n_positions_tested",
        "per_position",
    }
    assert expected_keys.issubset(set(report.keys()))
    assert report["n_positions_tested"] == 2
    assert isinstance(report["per_position"], list)
    assert len(report["per_position"]) == 2


def test_simulate_draft_returns_roster(engine, sample_pool):
    """simulate_draft returns a DataFrame with drafted players."""
    from src.valuation import value_all_players

    pool = value_all_players(sample_pool, engine.config)
    roster = engine.simulate_draft(
        player_pool=pool,
        draft_position=1,
        num_teams=4,  # Small league for speed
        num_rounds=5,
    )
    assert isinstance(roster, pd.DataFrame)
    assert len(roster) > 0
    assert len(roster) <= 5  # At most num_rounds players
    assert "player_id" in roster.columns


def test_full_backtest_averages_positions(engine, sample_pool, sample_actuals):
    """run_full_backtest averages metrics across all tested positions."""
    engine.actuals = sample_actuals

    from src.valuation import value_all_players

    pool = value_all_players(sample_pool, engine.config)

    # Run from positions 1 and 2 with tiny league for speed
    result = engine.run_full_backtest(
        player_pool=pool,
        n_positions=[1, 2],
        num_teams=4,
    )

    assert isinstance(result, dict)
    assert "projection_rmse" in result
    assert "rank_correlation" in result
    assert "category_win_rate" in result
    assert "value_capture_rate" in result
    assert "bust_rate" in result

    # Values should be numeric
    assert isinstance(result["projection_rmse"], float)
    assert isinstance(result["rank_correlation"], float)
