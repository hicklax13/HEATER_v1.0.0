"""Tests for Statcast XGBoost regression model in src/ml_ensemble.py.

Covers: train_ensemble, predict_corrections, graceful fallback.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.ml_ensemble import (
    MAX_CORRECTION,
    MIN_TRAINING_SAMPLES,
    STATCAST_FEATURES,
    XGBOOST_AVAILABLE,
    predict_corrections,
    train_ensemble,
)

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_historical_data(n: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic historical stats and projections for testing."""
    rng = np.random.RandomState(seed)
    player_ids = list(range(1, n + 1))

    # Historical actual stats (with Statcast features)
    stats = pd.DataFrame(
        {
            "player_id": player_ids,
            "is_hitter": [1] * n,
            "R": rng.randint(40, 100, n).astype(float),
            "HR": rng.randint(5, 40, n).astype(float),
            "RBI": rng.randint(30, 100, n).astype(float),
            "SB": rng.randint(0, 30, n).astype(float),
            "AVG": rng.uniform(0.220, 0.320, n),
            "OBP": rng.uniform(0.290, 0.400, n),
            "ab": rng.randint(300, 600, n).astype(float),
            "h": rng.randint(80, 180, n).astype(float),
            "bb": rng.randint(20, 80, n).astype(float),
            "hbp": rng.randint(0, 10, n).astype(float),
            "sf": rng.randint(0, 8, n).astype(float),
            # Statcast features
            "ev_mean": rng.uniform(85.0, 95.0, n),
            "barrel_pct": rng.uniform(3.0, 18.0, n),
            "hard_hit_pct": rng.uniform(30.0, 50.0, n),
            "xwoba": rng.uniform(0.280, 0.400, n),
            "xba": rng.uniform(0.220, 0.310, n),
            "sprint_speed": rng.uniform(24.0, 30.0, n),
            "age": rng.randint(22, 38, n).astype(float),
            "health_score": rng.uniform(0.5, 1.0, n),
        }
    )

    # Projected stats (slightly different from actuals)
    projections = pd.DataFrame(
        {
            "player_id": player_ids,
            "is_hitter": [1] * n,
            "R": stats["R"] + rng.normal(0, 10, n),
            "HR": stats["HR"] + rng.normal(0, 5, n),
            "RBI": stats["RBI"] + rng.normal(0, 10, n),
            "SB": stats["SB"] + rng.normal(0, 3, n),
            "AVG": stats["AVG"] + rng.normal(0, 0.015, n),
            "OBP": stats["OBP"] + rng.normal(0, 0.015, n),
            "ab": stats["ab"].copy(),
            "h": stats["h"].copy(),
            "bb": stats["bb"].copy(),
            "hbp": stats["hbp"].copy(),
            "sf": stats["sf"].copy(),
        }
    )

    return stats, projections


def _make_current_pool(n: int = 50, seed: int = 99) -> pd.DataFrame:
    """Generate synthetic current player pool."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "player_id": list(range(1, n + 1)),
            "name": [f"Player {i}" for i in range(1, n + 1)],
            "is_hitter": [1] * n,
            "ev_mean": rng.uniform(85.0, 95.0, n),
            "barrel_pct": rng.uniform(3.0, 18.0, n),
            "hard_hit_pct": rng.uniform(30.0, 50.0, n),
            "xwoba": rng.uniform(0.280, 0.400, n),
            "xba": rng.uniform(0.220, 0.310, n),
            "sprint_speed": rng.uniform(24.0, 30.0, n),
            "age": rng.randint(22, 38, n).astype(float),
            "health_score": rng.uniform(0.5, 1.0, n),
        }
    )


# ── Tests requiring XGBoost ──────────────────────────────────────────


@pytest.mark.skipif(not XGBOOST_AVAILABLE, reason="xgboost not installed")
class TestTrainEnsembleWithXGBoost:
    """Tests that require actual XGBoost installation."""

    def test_train_returns_model_with_sufficient_data(self):
        stats, projections = _make_historical_data(n=100)
        model = train_ensemble(stats, projections)
        assert model is not None
        # Should have a predict method (XGBRegressor)
        assert hasattr(model, "predict")

    def test_train_returns_none_with_insufficient_data(self):
        stats, projections = _make_historical_data(n=30)
        model = train_ensemble(stats, projections)
        assert model is None

    def test_train_returns_none_at_boundary(self):
        stats, projections = _make_historical_data(n=MIN_TRAINING_SAMPLES - 1)
        model = train_ensemble(stats, projections)
        assert model is None

    def test_train_succeeds_at_exact_min_samples(self):
        stats, projections = _make_historical_data(n=MIN_TRAINING_SAMPLES)
        model = train_ensemble(stats, projections)
        assert model is not None

    def test_predict_returns_series_with_correct_index(self):
        stats, projections = _make_historical_data(n=100)
        model = train_ensemble(stats, projections)
        pool = _make_current_pool(n=50)
        corrections = predict_corrections(model, pool)
        assert isinstance(corrections, pd.Series)
        assert corrections.name == "ml_correction"
        assert len(corrections) == 50
        # Index should match player_id column
        assert list(corrections.index) == list(pool["player_id"])

    def test_corrections_clamped_to_max(self):
        stats, projections = _make_historical_data(n=100)
        model = train_ensemble(stats, projections)
        pool = _make_current_pool(n=50)
        corrections = predict_corrections(model, pool)
        assert corrections.max() <= MAX_CORRECTION
        assert corrections.min() >= -MAX_CORRECTION

    def test_predict_handles_missing_statcast_columns(self):
        """Pool missing some Statcast columns should still work."""
        stats, projections = _make_historical_data(n=100)
        model = train_ensemble(stats, projections)
        # Pool with only some features
        pool = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "ev_mean": [90.0, 88.0, 92.0],
                "age": [25.0, 30.0, 28.0],
                # Missing: barrel_pct, hard_hit_pct, xwoba, xba, sprint_speed, health_score
            }
        )
        corrections = predict_corrections(model, pool)
        assert len(corrections) == 3
        assert not corrections.isna().any()

    def test_predict_with_nan_values(self):
        """NaN values in features should be filled with medians."""
        stats, projections = _make_historical_data(n=100)
        model = train_ensemble(stats, projections)
        pool = _make_current_pool(n=10)
        pool.loc[0, "ev_mean"] = np.nan
        pool.loc[1, "barrel_pct"] = np.nan
        corrections = predict_corrections(model, pool)
        assert len(corrections) == 10
        assert not corrections.isna().any()

    def test_train_with_custom_config(self):
        from src.valuation import LeagueConfig

        config = LeagueConfig(num_teams=10)
        stats, projections = _make_historical_data(n=100)
        model = train_ensemble(stats, projections, config=config)
        assert model is not None


# ── Tests that work without XGBoost ──────────────────────────────────


class TestPredictCorrectionsNoModel:
    """Tests that do not require XGBoost."""

    def test_returns_zeros_when_model_is_none(self):
        pool = _make_current_pool(n=20)
        corrections = predict_corrections(None, pool)
        assert isinstance(corrections, pd.Series)
        assert len(corrections) == 20
        assert (corrections == 0.0).all()
        assert corrections.name == "ml_correction"

    def test_returns_empty_series_for_none_pool(self):
        corrections = predict_corrections(None, None)
        assert isinstance(corrections, pd.Series)
        assert len(corrections) == 0


class TestGracefulFallbackNoXGBoost:
    """Test graceful fallback when xgboost is not installed."""

    def test_train_returns_none_when_xgboost_unavailable(self):
        stats, projections = _make_historical_data(n=100)
        with patch("src.ml_ensemble.XGBOOST_AVAILABLE", False):
            model = train_ensemble(stats, projections)
        assert model is None

    def test_predict_returns_zeros_when_xgboost_unavailable(self):
        pool = _make_current_pool(n=10)
        # Even with a non-None "model", should return zeros if flag is False
        with patch("src.ml_ensemble.XGBOOST_AVAILABLE", False):
            corrections = predict_corrections("fake_model", pool)
        assert isinstance(corrections, pd.Series)
        assert len(corrections) == 10
        assert (corrections == 0.0).all()


class TestTrainEnsembleEdgeCases:
    """Edge case tests that do not require XGBoost."""

    def test_train_returns_none_for_missing_player_id(self):
        stats = pd.DataFrame({"name": ["A", "B"]})
        projections = pd.DataFrame({"name": ["A", "B"]})
        result = train_ensemble(stats, projections)
        assert result is None

    def test_train_returns_none_for_none_inputs(self):
        result = train_ensemble(None, pd.DataFrame())
        assert result is None
        result2 = train_ensemble(pd.DataFrame(), None)
        assert result2 is None

    def test_statcast_features_list(self):
        """Verify expected features are defined."""
        assert "ev_mean" in STATCAST_FEATURES
        assert "barrel_pct" in STATCAST_FEATURES
        assert "hard_hit_pct" in STATCAST_FEATURES
        assert "xwoba" in STATCAST_FEATURES
        assert "xba" in STATCAST_FEATURES
        assert "sprint_speed" in STATCAST_FEATURES
        assert "age" in STATCAST_FEATURES
        assert "health_score" in STATCAST_FEATURES
        assert len(STATCAST_FEATURES) == 8
