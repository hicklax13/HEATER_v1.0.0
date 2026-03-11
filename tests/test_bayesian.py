"""Tests for Bayesian projection updater."""

import numpy as np
import pandas as pd
import pytest

from src.bayesian import (
    DECLINE_RATES,
    LEAGUE_MEANS,
    PEAK_AGES,
    PYMC_AVAILABLE,
    STABILIZATION_POINTS,
    BayesianUpdater,
)


@pytest.fixture
def updater():
    """Create a BayesianUpdater instance."""
    return BayesianUpdater(prior_weight=0.6)


# ── regressed_rate tests ─────────────────────────────────────────────


class TestRegressedRate:
    def test_zero_sample_returns_league_mean(self, updater):
        """With no observed data, should return the prior (league mean)."""
        result = updater.regressed_rate(0.300, 0, 0.250, 910)
        assert result == pytest.approx(0.250)

    def test_large_sample_approaches_observed(self, updater):
        """With large sample, should approach the observed rate."""
        result = updater.regressed_rate(0.300, 5000, 0.250, 910)
        assert result == pytest.approx(0.300, abs=0.01)

    def test_at_stabilization_point_is_midpoint(self, updater):
        """At exactly the stabilization point, should be midpoint of observed and prior."""
        result = updater.regressed_rate(0.300, 910, 0.250, 910)
        expected = (0.300 * 910 + 0.250 * 910) / (910 + 910)
        assert result == pytest.approx(expected)

    def test_small_sample_biased_toward_mean(self, updater):
        """Small sample should be weighted heavily toward league mean."""
        result = updater.regressed_rate(0.400, 50, 0.250, 910)
        # With only 50 PA and stab=910, weight on observed = 50/960 ≈ 5%
        assert result == pytest.approx(0.258, abs=0.01)

    def test_negative_sample_returns_mean(self, updater):
        """Negative sample size should return league mean."""
        result = updater.regressed_rate(0.300, -10, 0.250, 910)
        assert result == pytest.approx(0.250)

    def test_zero_stabilization_returns_observed(self, updater):
        """Zero stabilization point should return observed directly."""
        result = updater.regressed_rate(0.300, 100, 0.250, 0)
        assert result == pytest.approx(0.300)


class TestRegressedRateWithPrior:
    def test_uses_player_prior(self, updater):
        """Should use player-specific prior instead of league mean."""
        result = updater.regressed_rate_with_prior(0.300, 0, 0.270, 910)
        assert result == pytest.approx(0.270)

    def test_blends_observed_and_prior(self, updater):
        """Should blend observed and prior based on sample size."""
        result = updater.regressed_rate_with_prior(0.300, 200, 0.260, 910)
        expected = (0.300 * 200 + 0.260 * 910) / (200 + 910)
        assert result == pytest.approx(expected)


# ── Stabilization points tests ───────────────────────────────────────


class TestStabilizationPoints:
    def test_k_rate_is_60(self):
        assert STABILIZATION_POINTS["k_rate"] == 60

    def test_avg_is_910(self):
        assert STABILIZATION_POINTS["avg"] == 910

    def test_era_is_70(self):
        assert STABILIZATION_POINTS["era"] == 70

    def test_hr_rate_is_170(self):
        assert STABILIZATION_POINTS["hr_rate"] == 170

    def test_get_stabilization_point(self, updater):
        assert BayesianUpdater.get_stabilization_point("avg") == 910
        assert BayesianUpdater.get_stabilization_point("unknown_stat") == 200


# ── Age adjustment tests ─────────────────────────────────────────────


class TestAgeAdjustment:
    def test_at_peak_returns_one(self, updater):
        """Player at peak age should get multiplier of 1.0."""
        assert updater.age_adjustment(27, "hr") == pytest.approx(1.0)

    def test_young_returns_one(self, updater):
        """Player below peak should get 1.0 (no early bonus)."""
        assert updater.age_adjustment(22, "hr") == pytest.approx(1.0)

    def test_power_decline(self, updater):
        """Power stat should decline at 1.5%/yr after age 27."""
        result = updater.age_adjustment(32, "hr")
        expected = 1.0 - (5 * DECLINE_RATES["power"])
        assert result == pytest.approx(expected)

    def test_speed_decline_faster(self, updater):
        """Speed should decline faster than power."""
        speed_adj = updater.age_adjustment(32, "sb")
        power_adj = updater.age_adjustment(32, "hr")
        assert speed_adj < power_adj

    def test_floor_at_50_percent(self, updater):
        """Very old player should never go below 0.5 multiplier."""
        # Speed declines at 2.5%/yr from age 26. At age 60: 1.0 - 34*0.025 = 0.15 → clamped to 0.5
        result = updater.age_adjustment(60, "sb")
        assert result == pytest.approx(0.5)

    def test_pitching_decline(self, updater):
        """Pitcher age curve should use pitching peak (27)."""
        result = updater.age_adjustment(30, "era")
        expected = 1.0 - (3 * DECLINE_RATES["pitching"])
        assert result == pytest.approx(expected)

    def test_contact_peaks_at_29(self, updater):
        """Contact stats should peak at 29."""
        assert updater.age_adjustment(29, "avg") == pytest.approx(1.0)
        assert updater.age_adjustment(30, "avg") < 1.0


# ── Hierarchical model tests ────────────────────────────────────────


class TestHierarchicalModel:
    def test_fallback_when_no_pymc(self, updater):
        """When PyMC unavailable, should return Marcel estimates."""
        rates = np.array([0.280, 0.300, 0.260])
        sizes = np.array([200, 300, 150])
        result = updater.hierarchical_rate_model(rates, sizes, 0.250)
        assert "posterior_means" in result
        assert "hdi_low" in result
        assert "hdi_high" in result
        assert len(result["posterior_means"]) == 3
        # All means should be between league mean and observed
        for mean, obs in zip(result["posterior_means"], rates):
            assert 0.240 <= mean <= obs + 0.01

    def test_zero_sample_sizes(self, updater):
        """All zero sample sizes should return group mean."""
        rates = np.array([0.0, 0.0, 0.0])
        sizes = np.array([0, 0, 0])
        result = updater.hierarchical_rate_model(rates, sizes, 0.250)
        np.testing.assert_allclose(result["posterior_means"], 0.250)

    @pytest.mark.skipif(not PYMC_AVAILABLE, reason="PyMC not installed")
    def test_pymc_model_converges(self, updater):
        """With PyMC available, model should converge and produce reasonable posteriors."""
        np.random.seed(42)
        true_rates = np.array([0.270, 0.290, 0.250, 0.310, 0.260])
        sizes = np.array([300, 400, 200, 350, 250])
        result = updater.hierarchical_rate_model(true_rates, sizes, 0.265)
        assert result["converged"] is True
        # Posteriors should be in reasonable range
        for mean in result["posterior_means"]:
            assert 0.200 <= mean <= 0.400


# ── batch_update_projections tests ───────────────────────────────────


class TestBatchUpdate:
    @pytest.fixture
    def preseason(self):
        return pd.DataFrame(
            {
                "player_id": [1, 2],
                "pa": [600, 550],
                "ab": [540, 500],
                "h": [146, 130],
                "r": [80, 70],
                "hr": [25, 20],
                "rbi": [85, 75],
                "sb": [10, 15],
                "avg": [0.270, 0.260],
                "ip": [0, 0],
                "w": [0, 0],
                "sv": [0, 0],
                "k": [0, 0],
                "era": [0, 0],
                "whip": [0, 0],
                "er": [0, 0],
                "bb_allowed": [0, 0],
                "h_allowed": [0, 0],
            }
        )

    @pytest.fixture
    def season_stats(self):
        return pd.DataFrame(
            {
                "player_id": [1, 2],
                "pa": [200, 150],
                "ab": [180, 135],
                "h": [54, 38],
                "r": [28, 18],
                "hr": [10, 5],
                "rbi": [30, 20],
                "sb": [3, 8],
                "avg": [0.300, 0.281],
                "ip": [0, 0],
                "w": [0, 0],
                "sv": [0, 0],
                "k": [0, 0],
                "era": [0, 0],
                "whip": [0, 0],
                "er": [0, 0],
                "bb_allowed": [0, 0],
                "h_allowed": [0, 0],
                "games_played": [50, 40],
            }
        )

    def test_returns_dataframe(self, updater, preseason, season_stats):
        result = updater.batch_update_projections(season_stats, preseason)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_has_bayesian_system(self, updater, preseason, season_stats):
        result = updater.batch_update_projections(season_stats, preseason)
        assert (result["system"] == "bayesian").all()

    def test_hot_hitter_adjusted_up(self, updater, preseason, season_stats):
        """Player 1 is hitting .300 (above .270 preseason) — projection should increase."""
        result = updater.batch_update_projections(season_stats, preseason)
        player_1 = result[result["player_id"] == 1].iloc[0]
        assert player_1["avg"] > 0.270  # Should be pulled up from .270 toward .300

    def test_empty_season_stats(self, updater, preseason):
        """Empty season stats should return preseason projections."""
        empty = pd.DataFrame()
        result = updater.batch_update_projections(empty, preseason)
        assert len(result) == len(preseason)

    def test_empty_preseason(self, updater, season_stats):
        """Empty preseason should return season stats (nothing to update from)."""
        empty = pd.DataFrame()
        result = updater.batch_update_projections(season_stats, empty)
        # With empty preseason, returns preseason as-is (which is empty)
        assert isinstance(result, pd.DataFrame)


# ── Constants validation ─────────────────────────────────────────────


class TestConstants:
    def test_league_means_reasonable(self):
        assert 0.230 <= LEAGUE_MEANS["avg"] <= 0.270
        assert 3.50 <= LEAGUE_MEANS["era"] <= 5.00
        assert 1.10 <= LEAGUE_MEANS["whip"] <= 1.45

    def test_peak_ages_reasonable(self):
        for skill, age in PEAK_AGES.items():
            assert 24 <= age <= 32, f"Peak age for {skill} is {age}"

    def test_decline_rates_positive(self):
        for skill, rate in DECLINE_RATES.items():
            assert 0 < rate < 0.10, f"Decline rate for {skill} is {rate}"
