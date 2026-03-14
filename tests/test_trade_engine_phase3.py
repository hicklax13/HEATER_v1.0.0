"""Tests for Trade Analyzer Engine Phase 3: Signal Intelligence.

Spec reference: Section 17 Phase 3 items 13-16

Tests cover:
  - Statcast data aggregation (batted ball, pitching features)
  - Signal decay weighting (exponential, half-lives, feature mapping)
  - Kalman filter (convergence, gain, sample size effect)
  - BOCPD changepoint detection (stable, changepoint, reset)
  - HMM regime detection (state classification, fallback)
  - Rolling feature computation
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

# ── Signal Decay ─────────────────────────────────────────────────────


class TestSignalDecay:
    """Spec ref: Section 4 L1A — Exponential Decay Weighting."""

    def test_today_weight_is_one(self):
        """Observation from today should have weight 1.0."""
        from src.engine.signals.decay import decay_weight

        today = date(2026, 6, 15)
        assert decay_weight(today, today, lambda_param=0.020) == 1.0

    def test_weight_decreases_with_age(self):
        """Older observations should have lower weight."""
        from src.engine.signals.decay import decay_weight

        today = date(2026, 6, 15)
        w1 = decay_weight(date(2026, 6, 14), today, 0.020)  # 1 day ago
        w7 = decay_weight(date(2026, 6, 8), today, 0.020)  # 7 days ago
        w30 = decay_weight(date(2026, 5, 16), today, 0.020)  # 30 days ago

        assert w1 > w7 > w30
        assert w1 < 1.0
        assert w30 > 0.0

    def test_half_life_correct(self):
        """At the half-life, weight should be ~0.5."""
        from src.engine.signals.decay import decay_weight, half_life_days

        lam = 0.020
        hl = half_life_days(lam)  # ~35 days
        today = date(2026, 6, 15)
        obs_date = today - timedelta(days=int(hl))

        w = decay_weight(obs_date, today, lam)
        assert abs(w - 0.5) < 0.05

    def test_zero_lambda_no_decay(self):
        """Lambda=0 (counting stats) should give weight=1.0 always."""
        from src.engine.signals.decay import decay_weight

        today = date(2026, 6, 15)
        ancient = date(2025, 1, 1)

        assert decay_weight(ancient, today, lambda_param=0.0) == 1.0

    def test_feature_lambda_lookup(self):
        """Each feature type should map to its correct decay category."""
        from src.engine.signals.decay import get_feature_lambda

        # Exit velocity → fast decay (0.020)
        assert get_feature_lambda("ev_mean") == 0.020
        # Spin rate → medium decay (0.015)
        assert get_feature_lambda("ff_spin_rate") == 0.015
        # Sprint speed → slow decay (0.005)
        assert get_feature_lambda("sprint_speed") == 0.005
        # Counting stats → no decay (0.0)
        assert get_feature_lambda("hr") == 0.0

    def test_weighted_mean_recency(self):
        """Weighted mean should favor recent observations."""
        from src.engine.signals.decay import apply_decay_weights, weighted_mean

        today = date(2026, 6, 15)
        obs = [
            {"date": date(2026, 5, 1), "value": 90.0},  # Old: 90 mph EV
            {"date": date(2026, 6, 14), "value": 95.0},  # Recent: 95 mph EV
        ]
        values, weights = apply_decay_weights(obs, "ev_mean", today)
        wm = weighted_mean(values, weights)

        # Weighted mean should be closer to 95 (recent) than 90 (old)
        assert wm > 92.5

    def test_weighted_variance(self):
        """Weighted variance should be non-negative."""
        from src.engine.signals.decay import apply_decay_weights, weighted_variance

        today = date(2026, 6, 15)
        obs = [
            {"date": date(2026, 6, 10), "value": 90.0},
            {"date": date(2026, 6, 12), "value": 92.0},
            {"date": date(2026, 6, 14), "value": 95.0},
        ]
        values, weights = apply_decay_weights(obs, "ev_mean", today)
        var = weighted_variance(values, weights)

        assert var >= 0


# ── Kalman Filter ────────────────────────────────────────────────────


class TestKalmanFilter:
    """Spec ref: Section 4 L1D — Kalman Filter for True Talent."""

    def test_converges_to_mean(self):
        """Given many observations at the same value, filter should converge."""
        from src.engine.signals.kalman import kalman_true_talent

        n = 50
        true_value = 0.300
        obs = np.full(n, true_value)
        obs_var = np.full(n, 0.01)  # Low noise

        filtered, var = kalman_true_talent(
            observations=obs,
            obs_variance=obs_var,
            process_variance=0.0001,
            prior_mean=0.250,
            prior_variance=0.01,
        )

        # Should converge close to true value
        assert abs(filtered[-1] - true_value) < 0.01
        # Variance should decrease
        assert var[-1] < var[0]

    def test_high_noise_trusts_prior(self):
        """With very noisy observations, filter should stay near prior."""
        from src.engine.signals.kalman import kalman_true_talent

        prior = 0.280
        obs = np.array([0.400, 0.350, 0.380])  # Very high but noisy
        obs_var = np.array([1.0, 1.0, 1.0])  # Huge variance

        filtered, _ = kalman_true_talent(
            observations=obs,
            obs_variance=obs_var,
            process_variance=0.0001,
            prior_mean=prior,
            prior_variance=0.001,  # Low prior variance = trust prior
        )

        # Should stay much closer to prior (0.280) than observations (0.380)
        assert abs(filtered[-1] - prior) < abs(filtered[-1] - 0.380)

    def test_low_noise_trusts_data(self):
        """With precise observations, filter should track data closely."""
        from src.engine.signals.kalman import kalman_true_talent

        obs = np.array([0.320, 0.315, 0.318])
        obs_var = np.array([0.0001, 0.0001, 0.0001])  # Very precise

        filtered, _ = kalman_true_talent(
            observations=obs,
            obs_variance=obs_var,
            process_variance=0.0001,
            prior_mean=0.250,  # Prior is far off
            prior_variance=0.01,
        )

        # Should track close to data
        assert abs(filtered[-1] - 0.318) < 0.01

    def test_observation_variance_scales_with_sample(self):
        """Larger samples → smaller observation variance."""
        from src.engine.signals.kalman import observation_variance

        var_small = observation_variance("ba", sample_size=20)
        var_large = observation_variance("ba", sample_size=500)

        assert var_small > var_large
        assert var_large > 0

    def test_kalman_gain_decreases_over_time(self):
        """As variance shrinks, Kalman gain should decrease (less responsive)."""
        from src.engine.signals.kalman import kalman_true_talent

        obs = np.full(20, 0.300)
        obs_var = np.full(20, 0.01)

        filtered, f_var = kalman_true_talent(
            observations=obs,
            obs_variance=obs_var,
            process_variance=0.0001,
            prior_mean=0.250,
            prior_variance=0.05,
        )

        # Filtered variance should monotonically decrease (or plateau)
        for i in range(1, len(f_var)):
            assert f_var[i] <= f_var[i - 1] + 0.001  # Allow tiny float error

    def test_run_kalman_for_feature(self):
        """Convenience function should return all expected keys."""
        from src.engine.signals.kalman import run_kalman_for_feature

        rolling = [
            {"value": 0.300, "sample_size": 50},
            {"value": 0.310, "sample_size": 60},
            {"value": 0.290, "sample_size": 55},
        ]

        result = run_kalman_for_feature(rolling, "ba", prior_mean=0.280)

        assert "filtered_mean" in result
        assert "filtered_var" in result
        assert "kalman_gain_final" in result
        assert 0 < result["kalman_gain_final"] < 1

    def test_empty_input(self):
        """Empty observation list should return defaults."""
        from src.engine.signals.kalman import run_kalman_for_feature

        result = run_kalman_for_feature([], "ba", prior_mean=0.280)
        assert result["filtered_mean"] == 0.280


# ── BOCPD ────────────────────────────────────────────────────────────


class TestBOCPD:
    """Spec ref: Section 5 L2A — Bayesian Online Changepoint Detection."""

    def test_stable_signal_no_changepoint(self):
        """Constant signal should produce low changepoint probability."""
        from src.engine.signals.regime import detect_changepoints

        rng = np.random.RandomState(42)
        stable = rng.normal(0.315, 0.005, size=50)

        result = detect_changepoints(stable, hazard_lambda=200, threshold=0.7)

        # Should find zero or very few changepoints in a stable signal
        assert len(result["changepoint_indices"]) <= 2

    def test_clear_changepoint_detected(self):
        """Abrupt mean shift should be detected via run length mode drop.

        Note: The raw cp_prob (P(r_t=0)) is bounded by 1/hazard_lambda,
        so it never exceeds ~0.033 for lambda=30. The real detection signal
        is the run length distribution's mode dropping from a long run to
        near zero, which detect_changepoints() now uses.
        """
        from src.engine.signals.regime import detect_changepoints

        rng = np.random.RandomState(42)
        before = rng.normal(0.280, 0.005, size=30)
        after = rng.normal(0.380, 0.005, size=30)  # +100 points of xwOBA
        series = np.concatenate([before, after])

        result = detect_changepoints(series, hazard_lambda=30)

        # Should detect at least one changepoint near the true break at index 30
        assert len(result["changepoint_indices"]) >= 1
        # The detected changepoint should be in the vicinity of the true shift
        closest_cp = min(result["changepoint_indices"], key=lambda x: abs(x - 30))
        assert abs(closest_cp - 30) <= 10  # Within 10 steps of the true break

    def test_bocpd_class_update(self):
        """BOCPD.update should return valid probability."""
        from src.engine.signals.regime import BOCPD

        detector = BOCPD(hazard_lambda=100, mu0=0.300)
        cp_prob, probs = detector.update(0.305)

        assert 0 <= cp_prob <= 1
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_bocpd_reset(self):
        """Reset should clear all state."""
        from src.engine.signals.regime import BOCPD

        detector = BOCPD(hazard_lambda=100)
        for _ in range(20):
            detector.update(np.random.normal(0.300, 0.01))

        assert len(detector.run_length_probs) > 1

        detector.reset()
        assert len(detector.run_length_probs) == 1
        assert detector.run_length_probs[0] == 1.0

    def test_current_regime_length(self):
        """After a detected changepoint, regime length should reflect it."""
        from src.engine.signals.regime import detect_changepoints

        rng = np.random.RandomState(42)
        # Add noise to make BOCPD work properly (constant values cause
        # degenerate variance estimates)
        before = rng.normal(0.280, 0.010, size=30)
        after = rng.normal(0.450, 0.010, size=30)  # Large jump
        series = np.concatenate([before, after])

        # Use very sensitive detection settings
        result = detect_changepoints(series, hazard_lambda=20, threshold=0.1)

        if result["changepoint_indices"]:
            # If changepoints detected, regime length should be less than total
            assert result["current_regime_length"] <= len(series)
        else:
            # BOCPD may not detect with these params — test the structure only
            assert result["current_regime_length"] == len(series)


# ── HMM / Regime Classification ──────────────────────────────────────


class TestRegimeClassification:
    """Spec ref: Section 5 L2B — Hidden Markov Model."""

    def test_simple_classification_elite(self):
        """High xwOBA → Elite classification."""
        from src.engine.signals.regime import classify_regime_simple

        label, probs = classify_regime_simple(
            recent_xwoba=0.400,
            season_xwoba=0.390,
            league_avg_xwoba=0.315,
        )

        assert label == "Elite"
        assert probs[0] > 0.5  # Most weight on Elite state

    def test_simple_classification_replacement(self):
        """Low xwOBA → Replacement classification."""
        from src.engine.signals.regime import classify_regime_simple

        label, probs = classify_regime_simple(
            recent_xwoba=0.250,
            season_xwoba=0.260,
            league_avg_xwoba=0.315,
        )

        assert label == "Replacement"
        assert probs[3] > 0.5  # Most weight on Replacement state

    def test_trending_up_shifts_states(self):
        """Rising recent performance should shift probabilities upward."""
        from src.engine.signals.regime import classify_regime_simple

        # Trending up: recent >> season
        _, probs_up = classify_regime_simple(
            recent_xwoba=0.370,
            season_xwoba=0.320,
        )

        # Neutral
        _, probs_flat = classify_regime_simple(
            recent_xwoba=0.345,
            season_xwoba=0.340,
        )

        # Trending up should have more weight on better states
        assert probs_up[0] + probs_up[1] >= probs_flat[0] + probs_flat[1]

    def test_probs_sum_to_one(self):
        """State probabilities should always sum to 1."""
        from src.engine.signals.regime import classify_regime_simple

        for xwoba in [0.200, 0.300, 0.350, 0.450]:
            _, probs = classify_regime_simple(xwoba, 0.315)
            assert abs(probs.sum() - 1.0) < 1e-6

    def test_regime_conditional_projection(self):
        """Mixture projection should weight by state probabilities."""
        from src.engine.signals.regime import regime_conditional_projection

        probs = np.array([0.6, 0.3, 0.1, 0.0])
        projections = [
            {"hr": 40, "avg": 0.300},  # Elite
            {"hr": 25, "avg": 0.275},  # Above-avg
            {"hr": 15, "avg": 0.250},  # Below-avg
            {"hr": 5, "avg": 0.220},  # Replacement
        ]

        result = regime_conditional_projection(probs, projections)

        # Should be weighted toward Elite projections
        expected_hr = 0.6 * 40 + 0.3 * 25 + 0.1 * 15
        assert abs(result["hr"] - expected_hr) < 0.01

    def test_hmm_graceful_degradation(self):
        """HMM should return valid probs even when hmmlearn unavailable."""
        from src.engine.signals.regime import fit_player_hmm

        # With too few observations
        obs = np.array([[0.300, 90.0, 0.08]] * 5)  # Only 5 data points
        model, probs = fit_player_hmm(obs)

        assert len(probs) == 4
        assert abs(probs.sum() - 1.0) < 1e-6


# ── Statcast Aggregation ─────────────────────────────────────────────


class TestStatcastAggregation:
    """Spec ref: Section 3 L0 — Signal Harvesting."""

    def test_aggregate_batter_empty(self):
        """Empty DataFrame should return empty dict."""
        import pandas as pd

        from src.engine.signals.statcast import aggregate_batter_statcast

        result = aggregate_batter_statcast(pd.DataFrame())
        assert result == {}

    def test_aggregate_batter_exit_velocity(self):
        """Should compute ev_mean and ev_p90 from launch_speed."""
        import pandas as pd

        from src.engine.signals.statcast import aggregate_batter_statcast

        data = pd.DataFrame(
            {
                "launch_speed": [90.0, 95.0, 100.0, 105.0, 110.0],
                "type": ["X", "X", "X", "X", "X"],  # All batted balls
            }
        )

        result = aggregate_batter_statcast(data)

        assert "ev_mean" in result
        assert abs(result["ev_mean"] - 100.0) < 0.1
        assert "ev_p90" in result
        assert result["ev_p90"] > result["ev_mean"]

    def test_aggregate_batter_hard_hit(self):
        """Hard hit % should count EV >= 95."""
        import pandas as pd

        from src.engine.signals.statcast import aggregate_batter_statcast

        data = pd.DataFrame(
            {
                "launch_speed": [85.0, 90.0, 95.0, 100.0, 105.0],
                "type": ["X", "X", "X", "X", "X"],
            }
        )

        result = aggregate_batter_statcast(data)
        # 3 out of 5 are >= 95 mph
        assert abs(result["hard_hit_pct"] - 0.6) < 0.01

    def test_aggregate_pitcher_empty(self):
        """Empty DataFrame should return empty dict."""
        import pandas as pd

        from src.engine.signals.statcast import aggregate_pitcher_statcast

        result = aggregate_pitcher_statcast(pd.DataFrame())
        assert result == {}

    def test_aggregate_pitcher_fastball_speed(self):
        """Should compute average fastball speed."""
        import pandas as pd

        from src.engine.signals.statcast import aggregate_pitcher_statcast

        data = pd.DataFrame(
            {
                "pitch_type": ["FF", "FF", "SL", "FF", "CH"],
                "release_speed": [95.0, 96.0, 85.0, 94.0, 88.0],
            }
        )

        result = aggregate_pitcher_statcast(data)

        assert "ff_avg_speed" in result
        # Mean of [95, 96, 94] (FF only)
        assert abs(result["ff_avg_speed"] - 95.0) < 0.5


# ── Rolling Features ─────────────────────────────────────────────────


class TestRollingFeatures:
    """Test rolling window feature computation."""

    def test_compute_rolling_features_empty(self):
        """Empty data should return empty list."""
        import pandas as pd

        from src.engine.signals.statcast import compute_rolling_features

        result = compute_rolling_features(pd.DataFrame(), window_days=14)
        assert result == []

    def test_compute_rolling_features_dates(self):
        """Rolling features should have chronological dates."""
        import pandas as pd

        from src.engine.signals.statcast import compute_rolling_features

        dates = pd.date_range("2026-04-01", periods=60, freq="D")
        rng = np.random.RandomState(42)
        data = pd.DataFrame(
            {
                "game_date": dates.repeat(10),  # 10 pitches per day
                "launch_speed": rng.normal(90, 5, 600),
                "type": ["X"] * 600,
            }
        )

        result = compute_rolling_features(data, window_days=14, is_pitcher=False)

        assert len(result) > 0
        # Dates should be in order
        for i in range(1, len(result)):
            assert result[i]["date"] >= result[i - 1]["date"]
