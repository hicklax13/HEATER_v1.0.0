"""Tests for Trade Analyzer Engine Phase 2: BMA, KDE, Copula, Monte Carlo.

Spec reference: Section 17 Phase 2 items 9-12

Tests cover:
  - Bayesian Model Averaging (posterior weights, blended projections, variance)
  - KDE marginals (ppf inverse CDF, Normal fallback, stat bounds)
  - Gaussian copula (correlated sampling, positive definiteness)
  - Paired Monte Carlo (variance reduction, distributional metrics)
  - Integration: evaluate_trade with enable_mc=True
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Bayesian Model Averaging ─────────────────────────────────────────


class TestBayesianModelAveraging:
    """Spec ref: Section 6 L3A — Bayesian Model Averaging."""

    def test_uniform_priors_equal_projections(self):
        """When all systems project the same value, posterior weights are nearly equal.

        Note: weights differ slightly because forecast sigmas differ by system
        (steamer HR sigma=5.2 vs zips sigma=5.5). Tighter sigma → higher
        likelihood density at the observed value → slightly more weight.
        """
        from src.engine.projections.bayesian_blend import bayesian_model_average

        ytd = {"hr": 20}
        projections = {
            "steamer": {"hr": 20},
            "zips": {"hr": 20},
        }
        pw, blended, var = bayesian_model_average(ytd, projections)

        # Equal projections → roughly equal weights (within ~5% due to sigma differences)
        assert abs(pw["steamer"] - pw["zips"]) < 0.05
        assert abs(blended["hr"] - 20.0) < 0.01

    def test_closer_system_gets_higher_weight(self):
        """System whose projection is closer to YTD gets higher posterior weight."""
        from src.engine.projections.bayesian_blend import bayesian_model_average

        ytd = {"hr": 25, "rbi": 80}
        projections = {
            "steamer": {"hr": 24, "rbi": 78},  # Close to YTD
            "zips": {"hr": 15, "rbi": 50},  # Far from YTD
        }
        pw, blended, _ = bayesian_model_average(ytd, projections)

        assert pw["steamer"] > pw["zips"]
        # Blended should be closer to Steamer's projection
        assert abs(blended["hr"] - 24) < abs(blended["hr"] - 15)

    def test_weights_sum_to_one(self):
        """Posterior weights must be a valid probability distribution."""
        from src.engine.projections.bayesian_blend import bayesian_model_average

        ytd = {"hr": 20, "avg": 0.280}
        projections = {
            "steamer": {"hr": 22, "avg": 0.275},
            "zips": {"hr": 18, "avg": 0.290},
            "depthcharts": {"hr": 21, "avg": 0.270},
        }
        pw, _, _ = bayesian_model_average(ytd, projections)

        assert abs(sum(pw.values()) - 1.0) < 1e-10

    def test_variance_includes_between_model(self):
        """Total variance = within-model + between-model.

        When systems disagree, between-model variance should be positive.
        """
        from src.engine.projections.bayesian_blend import bayesian_model_average

        ytd = {"hr": 20}
        # Systems that agree → low between-model variance
        agree = {"steamer": {"hr": 20}, "zips": {"hr": 20}}
        _, _, var_agree = bayesian_model_average(ytd, agree)

        # Systems that disagree → high between-model variance
        disagree = {"steamer": {"hr": 30}, "zips": {"hr": 10}}
        _, _, var_disagree = bayesian_model_average(ytd, disagree)

        assert var_disagree["hr"] > var_agree["hr"]

    def test_single_system_returns_that_system(self):
        """With only one system, posterior weight is 1.0 and blend = that system."""
        from src.engine.projections.bayesian_blend import bayesian_model_average

        ytd = {"hr": 20}
        projections = {"steamer": {"hr": 22}}
        pw, blended, _ = bayesian_model_average(ytd, projections)

        assert pw["steamer"] == 1.0
        assert blended["hr"] == 22

    def test_empty_projections(self):
        """Empty projections returns empty results."""
        from src.engine.projections.bayesian_blend import bayesian_model_average

        pw, blended, var = bayesian_model_average({"hr": 20}, {})
        assert pw == {}


# ── KDE Marginals ────────────────────────────────────────────────────


class TestKDEMarginals:
    """Spec ref: Section 17 Phase 2 item 10 — KDE marginals."""

    def test_normal_fallback_with_few_samples(self):
        """Fewer than MIN_KDE_SAMPLES → Normal distribution fallback."""
        from src.engine.projections.marginals import PlayerMarginal

        m = PlayerMarginal(
            stat_name="hr",
            projected_value=25.0,
            variance=25.0,
            historical_values=np.array([20, 22, 28]),  # Only 3 samples
        )
        assert not m.uses_kde

    def test_kde_with_enough_samples(self):
        """>= MIN_KDE_SAMPLES → KDE is fitted."""
        from src.engine.projections.marginals import PlayerMarginal

        rng = np.random.RandomState(42)
        hist = rng.normal(25, 5, size=50)

        m = PlayerMarginal(
            stat_name="hr",
            projected_value=25.0,
            variance=25.0,
            historical_values=hist,
        )
        assert m.uses_kde

    def test_ppf_returns_reasonable_values(self):
        """ppf(0.5) should be near the projected value."""
        from src.engine.projections.marginals import PlayerMarginal

        m = PlayerMarginal(
            stat_name="hr",
            projected_value=25.0,
            variance=25.0,
        )
        median = m.ppf(0.5)
        assert abs(median - 25.0) < 1.0

    def test_ppf_monotonically_increasing(self):
        """ppf should be monotonically increasing (higher quantile → higher value)."""
        from src.engine.projections.marginals import PlayerMarginal

        m = PlayerMarginal(
            stat_name="hr",
            projected_value=25.0,
            variance=25.0,
        )
        q10 = m.ppf(0.10)
        q50 = m.ppf(0.50)
        q90 = m.ppf(0.90)

        assert q10 < q50 < q90

    def test_stat_bounds_clipping(self):
        """AVG should be bounded between 0.100 and 0.400."""
        from src.engine.projections.marginals import PlayerMarginal

        m = PlayerMarginal(
            stat_name="avg",
            projected_value=0.280,
            variance=0.001,
        )
        # Very low quantile should be clipped to 0.100
        low = m.ppf(0.001)
        assert low >= 0.100

        # Very high quantile should be clipped to 0.400
        high = m.ppf(0.999)
        assert high <= 0.400

    def test_sample_returns_correct_shape(self):
        """sample(n) should return array of shape (n,)."""
        from src.engine.projections.marginals import PlayerMarginal

        m = PlayerMarginal(stat_name="hr", projected_value=25, variance=25)
        samples = m.sample(100)
        assert samples.shape == (100,)

    def test_build_player_marginals(self):
        """build_player_marginals returns a marginal for each stat."""
        from src.engine.projections.marginals import build_player_marginals

        stats = {"hr": 25.0, "avg": 0.280, "era": 3.50}
        variances = {"hr": 25.0, "avg": 0.001, "era": 0.5}

        marginals = build_player_marginals(stats, variances)
        assert set(marginals.keys()) == {"hr", "avg", "era"}
        assert marginals["hr"].projected_value == 25.0


# ── Gaussian Copula ──────────────────────────────────────────────────


class TestGaussianCopula:
    """Spec ref: Section 7 L4A — Vine/Gaussian Copula."""

    def test_sample_shape(self):
        """Copula samples should have shape (n, 12) for 12 categories."""
        from src.engine.portfolio.copula import GaussianCopula

        copula = GaussianCopula()
        samples = copula.sample(100)
        assert samples.shape == (100, 12)

    def test_samples_in_unit_interval(self):
        """All copula samples should be in (0, 1)."""
        from src.engine.portfolio.copula import GaussianCopula

        copula = GaussianCopula()
        samples = copula.sample(1000, rng=np.random.RandomState(42))

        assert np.all(samples > 0)
        assert np.all(samples < 1)

    def test_correlation_preserved(self):
        """HR and RBI columns should be positively correlated in samples."""
        from src.engine.portfolio.copula import CATEGORIES, GaussianCopula

        copula = GaussianCopula()
        samples = copula.sample(5000, rng=np.random.RandomState(42))

        hr_idx = CATEGORIES.index("HR")
        rbi_idx = CATEGORIES.index("RBI")
        corr = np.corrcoef(samples[:, hr_idx], samples[:, rbi_idx])[0, 1]

        # HR-RBI should be strongly positively correlated
        assert corr > 0.5

    def test_sb_hr_negative_correlation(self):
        """SB and HR should be weakly negatively correlated."""
        from src.engine.portfolio.copula import CATEGORIES, GaussianCopula

        copula = GaussianCopula()
        samples = copula.sample(5000, rng=np.random.RandomState(42))

        sb_idx = CATEGORIES.index("SB")
        hr_idx = CATEGORIES.index("HR")
        corr = np.corrcoef(samples[:, sb_idx], samples[:, hr_idx])[0, 1]

        # Should be negative
        assert corr < 0

    def test_positive_definite_correction(self):
        """Non-PD matrix should be corrected to nearest PD."""
        from src.engine.portfolio.copula import GaussianCopula

        # Create a non-positive-definite matrix
        bad_corr = np.array(
            [
                [1.0, 0.9, 0.9],
                [0.9, 1.0, -0.9],  # This combo is not PD
                [0.9, -0.9, 1.0],
            ]
        )

        # Should not raise — nearest PD correction kicks in
        copula = GaussianCopula(bad_corr)
        samples = copula.sample(10)
        assert samples.shape == (10, 3)

    def test_reproducibility_with_seed(self):
        """Same rng seed should produce identical samples."""
        from src.engine.portfolio.copula import GaussianCopula

        copula = GaussianCopula()
        s1 = copula.sample(10, rng=np.random.RandomState(123))
        s2 = copula.sample(10, rng=np.random.RandomState(123))

        np.testing.assert_array_equal(s1, s2)


class TestCorrelatedStatSampling:
    """Test copula + marginals integration."""

    def test_sample_correlated_stats_shape(self):
        """sample_correlated_stats returns (n, 12) array."""
        from src.engine.portfolio.copula import GaussianCopula, sample_correlated_stats
        from src.engine.projections.marginals import PlayerMarginal

        copula = GaussianCopula()
        marginals = {
            "r": PlayerMarginal("r", 80, 100),
            "hr": PlayerMarginal("hr", 25, 25),
            "rbi": PlayerMarginal("rbi", 80, 100),
            "sb": PlayerMarginal("sb", 10, 16),
            "avg": PlayerMarginal("avg", 0.280, 0.001),
            "w": PlayerMarginal("w", 10, 9),
            "k": PlayerMarginal("k", 180, 400),
            "sv": PlayerMarginal("sv", 0, 0.01),
            "era": PlayerMarginal("era", 3.50, 0.5),
            "whip": PlayerMarginal("whip", 1.20, 0.01),
        }

        stats = sample_correlated_stats(copula, marginals, n=100)
        assert stats.shape == (100, 12)

    def test_sampled_stats_reasonable(self):
        """Sampled HR values should be in a reasonable range around projection."""
        from src.engine.portfolio.copula import CATEGORIES, GaussianCopula, sample_correlated_stats
        from src.engine.projections.marginals import PlayerMarginal

        copula = GaussianCopula()
        marginals = {
            "r": PlayerMarginal("r", 80, 100),
            "hr": PlayerMarginal("hr", 25, 25),
            "rbi": PlayerMarginal("rbi", 80, 100),
            "sb": PlayerMarginal("sb", 10, 16),
            "avg": PlayerMarginal("avg", 0.280, 0.001),
            "w": PlayerMarginal("w", 10, 9),
            "k": PlayerMarginal("k", 180, 400),
            "sv": PlayerMarginal("sv", 0, 0.01),
            "era": PlayerMarginal("era", 3.50, 0.5),
            "whip": PlayerMarginal("whip", 1.20, 0.01),
        }

        rng = np.random.RandomState(42)
        stats = sample_correlated_stats(copula, marginals, n=1000, rng=rng)
        hr_idx = CATEGORIES.index("HR")

        mean_hr = np.mean(stats[:, hr_idx])
        # Mean should be near projected value (25)
        assert 15 < mean_hr < 35


# ── Paired Monte Carlo ───────────────────────────────────────────────


class TestPairedMonteCarlo:
    """Spec ref: Section 13 L10 — Paired MC trade evaluation."""

    def _make_roster_stats(self, players: dict[str, dict[str, float]]):
        return players

    def test_identical_rosters_zero_surplus(self):
        """When before == after roster, mean surplus should be ~0."""
        from src.engine.monte_carlo.trade_simulator import run_paired_monte_carlo

        roster = {
            "1": {
                "hr": 25,
                "r": 80,
                "rbi": 80,
                "sb": 10,
                "avg": 0.280,
                "w": 10,
                "k": 180,
                "sv": 0,
                "era": 3.50,
                "whip": 1.20,
            },
            "2": {
                "hr": 15,
                "r": 60,
                "rbi": 55,
                "sb": 5,
                "avg": 0.260,
                "w": 8,
                "k": 150,
                "sv": 0,
                "era": 4.00,
                "whip": 1.30,
            },
        }

        result = run_paired_monte_carlo(
            before_roster_stats=roster,
            after_roster_stats=roster,  # Same roster
            n_sims=1000,
        )

        # Paired comparison with identical rosters → surplus exactly 0
        assert abs(result["mc_mean"]) < 0.001

    def test_better_roster_positive_surplus(self):
        """Trading for a clearly better player should show positive surplus."""
        from src.engine.monte_carlo.trade_simulator import run_paired_monte_carlo

        before = {
            "1": {
                "hr": 10,
                "r": 40,
                "rbi": 40,
                "sb": 5,
                "avg": 0.240,
                "w": 5,
                "k": 80,
                "sv": 0,
                "era": 5.00,
                "whip": 1.50,
            },
        }
        after = {
            "1": {
                "hr": 35,
                "r": 100,
                "rbi": 110,
                "sb": 15,
                "avg": 0.300,
                "w": 15,
                "k": 250,
                "sv": 0,
                "era": 2.80,
                "whip": 1.00,
            },
        }

        result = run_paired_monte_carlo(
            before_roster_stats=before,
            after_roster_stats=after,
            n_sims=1000,
        )

        assert result["mc_mean"] > 0
        assert result["prob_positive"] > 0.5

    def test_result_contains_all_metrics(self):
        """MC result should contain all distributional metrics."""
        from src.engine.monte_carlo.trade_simulator import run_paired_monte_carlo

        roster = {
            "1": {
                "hr": 20,
                "r": 60,
                "rbi": 60,
                "sb": 8,
                "avg": 0.270,
                "w": 8,
                "k": 150,
                "sv": 0,
                "era": 3.80,
                "whip": 1.25,
            }
        }

        result = run_paired_monte_carlo(
            before_roster_stats=roster,
            after_roster_stats=roster,
            n_sims=200,
        )

        required_keys = [
            "mc_mean",
            "mc_std",
            "mc_median",
            "p5",
            "p25",
            "p75",
            "p95",
            "prob_positive",
            "var5",
            "cvar5",
            "sharpe",
            "grade",
            "verdict",
            "confidence_pct",
            "confidence_interval",
            "surplus_distribution",
            "n_sims",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_variance_reduction_from_pairing(self):
        """Paired sims should have lower variance than unpaired.

        This tests the fundamental property: identical seeds for
        before/after cancel out common randomness.
        """
        from src.engine.monte_carlo.trade_simulator import run_paired_monte_carlo

        before = {
            "1": {
                "hr": 20,
                "r": 70,
                "rbi": 65,
                "sb": 8,
                "avg": 0.270,
                "w": 8,
                "k": 150,
                "sv": 0,
                "era": 3.80,
                "whip": 1.25,
            },
        }
        after = {
            "1": {
                "hr": 22,
                "r": 72,
                "rbi": 68,
                "sb": 9,
                "avg": 0.275,
                "w": 9,
                "k": 155,
                "sv": 0,
                "era": 3.70,
                "whip": 1.22,
            },
        }

        result = run_paired_monte_carlo(
            before_roster_stats=before,
            after_roster_stats=after,
            n_sims=1000,
        )

        # The std of the surplus distribution should be relatively small
        # compared to the absolute SGP values (variance reduction in action)
        assert result["mc_std"] < 10.0  # Reasonable bound for small trade

    def test_percentile_ordering(self):
        """p5 < p25 < median < p75 < p95."""
        from src.engine.monte_carlo.trade_simulator import run_paired_monte_carlo

        before = {
            "1": {
                "hr": 20,
                "r": 60,
                "rbi": 60,
                "sb": 8,
                "avg": 0.270,
                "w": 8,
                "k": 150,
                "sv": 0,
                "era": 3.80,
                "whip": 1.25,
            }
        }
        after = {
            "1": {
                "hr": 25,
                "r": 70,
                "rbi": 70,
                "sb": 10,
                "avg": 0.280,
                "w": 10,
                "k": 180,
                "sv": 0,
                "era": 3.50,
                "whip": 1.15,
            }
        }

        result = run_paired_monte_carlo(
            before_roster_stats=before,
            after_roster_stats=after,
            n_sims=1000,
        )

        assert result["p5"] <= result["p25"]
        assert result["p25"] <= result["mc_median"]
        assert result["mc_median"] <= result["p75"]
        assert result["p75"] <= result["p95"]

    def test_prob_positive_range(self):
        """prob_positive should be between 0 and 1."""
        from src.engine.monte_carlo.trade_simulator import run_paired_monte_carlo

        roster = {
            "1": {
                "hr": 20,
                "r": 60,
                "rbi": 60,
                "sb": 8,
                "avg": 0.270,
                "w": 8,
                "k": 150,
                "sv": 0,
                "era": 3.80,
                "whip": 1.25,
            }
        }

        result = run_paired_monte_carlo(
            before_roster_stats=roster,
            after_roster_stats=roster,
            n_sims=200,
        )

        assert 0.0 <= result["prob_positive"] <= 1.0

    def test_cvar5_less_than_var5(self):
        """CVaR5 (expected loss in worst 5%) should be <= VaR5."""
        from src.engine.monte_carlo.trade_simulator import run_paired_monte_carlo

        before = {
            "1": {
                "hr": 20,
                "r": 60,
                "rbi": 60,
                "sb": 8,
                "avg": 0.270,
                "w": 8,
                "k": 150,
                "sv": 0,
                "era": 3.80,
                "whip": 1.25,
            }
        }
        after = {
            "1": {
                "hr": 25,
                "r": 70,
                "rbi": 70,
                "sb": 10,
                "avg": 0.280,
                "w": 10,
                "k": 180,
                "sv": 0,
                "era": 3.50,
                "whip": 1.15,
            }
        }

        result = run_paired_monte_carlo(
            before_roster_stats=before,
            after_roster_stats=after,
            n_sims=1000,
        )

        # CVaR is the mean of the worst tail — should be <= VaR threshold
        assert result["cvar5"] <= result["var5"] + 0.01  # Small tolerance


# ── Build Roster Stats ───────────────────────────────────────────────


class TestBuildRosterStats:
    """Test the helper that converts player pool → MC stat dicts."""

    def test_from_dataframe(self):
        """Should extract stat columns from a DataFrame."""
        from src.engine.monte_carlo.trade_simulator import build_roster_stats

        pool = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "hr": 25,
                    "r": 80,
                    "rbi": 80,
                    "sb": 10,
                    "avg": 0.280,
                    "w": 0,
                    "k": 0,
                    "sv": 0,
                    "era": 0,
                    "whip": 0,
                },
                {
                    "player_id": 2,
                    "hr": 30,
                    "r": 90,
                    "rbi": 95,
                    "sb": 5,
                    "avg": 0.290,
                    "w": 0,
                    "k": 0,
                    "sv": 0,
                    "era": 0,
                    "whip": 0,
                },
            ]
        )

        result = build_roster_stats([1, 2], pool)

        assert "1" in result
        assert result["1"]["hr"] == 25.0
        assert "2" in result
        assert result["2"]["hr"] == 30.0

    def test_missing_player_skipped(self):
        """Players not in pool should be silently skipped."""
        from src.engine.monte_carlo.trade_simulator import build_roster_stats

        pool = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "hr": 25,
                    "r": 80,
                    "rbi": 80,
                    "sb": 10,
                    "avg": 0.280,
                    "w": 0,
                    "k": 0,
                    "sv": 0,
                    "era": 0,
                    "whip": 0,
                },
            ]
        )

        result = build_roster_stats([1, 999], pool)
        assert "1" in result
        assert "999" not in result


# ── Integration: evaluate_trade with MC ──────────────────────────────


class TestEvaluateTradeWithMC:
    """Test evaluate_trade() with enable_mc=True."""

    def _make_pool(self):
        """Create a minimal player pool for testing."""
        return pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "name": "Player A",
                    "team": "NYY",
                    "positions": "1B",
                    "is_hitter": 1,
                    "is_injured": 0,
                    "r": 80,
                    "hr": 25,
                    "rbi": 80,
                    "sb": 10,
                    "avg": 0.280,
                    "pa": 550,
                    "ab": 500,
                    "h": 140,
                    "w": 0,
                    "k": 0,
                    "sv": 0,
                    "era": 0,
                    "whip": 0,
                    "ip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "adp": 50,
                },
                {
                    "player_id": 2,
                    "name": "Player B",
                    "team": "BOS",
                    "positions": "OF",
                    "is_hitter": 1,
                    "is_injured": 0,
                    "r": 90,
                    "hr": 30,
                    "rbi": 95,
                    "sb": 15,
                    "avg": 0.290,
                    "pa": 600,
                    "ab": 540,
                    "h": 157,
                    "w": 0,
                    "k": 0,
                    "sv": 0,
                    "era": 0,
                    "whip": 0,
                    "ip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "adp": 30,
                },
                {
                    "player_id": 3,
                    "name": "Player C",
                    "team": "LAD",
                    "positions": "SP",
                    "is_hitter": 0,
                    "is_injured": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "w": 14,
                    "k": 220,
                    "sv": 0,
                    "era": 3.20,
                    "whip": 1.10,
                    "ip": 190,
                    "er": 67,
                    "bb_allowed": 55,
                    "h_allowed": 150,
                    "adp": 25,
                },
            ]
        )

    def test_mc_keys_present_when_enabled(self):
        """When enable_mc=True, result should have MC-specific keys."""
        from src.engine.output.trade_evaluator import evaluate_trade

        pool = self._make_pool()
        result = evaluate_trade(
            giving_ids=[1],
            receiving_ids=[2],
            user_roster_ids=[1, 3],
            player_pool=pool,
            enable_mc=True,
            n_sims=200,
        )

        assert "mc_mean" in result
        assert "mc_std" in result
        assert "prob_positive" in result
        assert "sharpe" in result

    def test_mc_disabled_by_default(self):
        """By default, MC is not run (mc_std should be 0.0 from Phase 1)."""
        from src.engine.output.trade_evaluator import evaluate_trade

        pool = self._make_pool()
        result = evaluate_trade(
            giving_ids=[1],
            receiving_ids=[2],
            user_roster_ids=[1, 3],
            player_pool=pool,
        )

        # Phase 1 sets mc_std = 0.0 (no simulation)
        assert result["mc_std"] == 0.0

    def test_mc_does_not_break_phase1_keys(self):
        """MC overlay should preserve all Phase 1 result keys."""
        from src.engine.output.trade_evaluator import evaluate_trade

        pool = self._make_pool()
        result = evaluate_trade(
            giving_ids=[1],
            receiving_ids=[2],
            user_roster_ids=[1, 3],
            player_pool=pool,
            enable_mc=True,
            n_sims=200,
        )

        phase1_keys = [
            "grade",
            "surplus_sgp",
            "category_impact",
            "category_analysis",
            "punt_categories",
            "bench_cost",
            "risk_flags",
            "verdict",
            "confidence_pct",
            "before_totals",
            "after_totals",
            "giving_players",
            "receiving_players",
        ]
        for key in phase1_keys:
            assert key in result, f"Missing Phase 1 key: {key}"
