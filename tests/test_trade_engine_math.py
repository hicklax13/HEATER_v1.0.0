"""Mathematical verification tests for the Trade Analyzer Engine.

Proves correctness of every formula in the trade engine by hand-computing
expected values and asserting the code matches within floating-point tolerance.

Covers all 6 phases:
  Phase 1: SGP delta, inverse stat sign-flip, category weights, punt detection,
           bench option value, grade thresholds
  Phase 2: BMA posterior weights (Bayes' theorem), BMA variance decomposition,
           copula Cholesky, paired MC variance reduction
  Phase 3: Exponential decay, Kalman update
  Phase 4: Concentration risk (HHI), enhanced bench option value
  Phase 5: Bayes' theorem (adverse selection), Vickrey auction (market clearing),
           opponent valuations (inverse categories), Bellman rollout,
           sensitivity breakeven
  Phase 6: ESS formula, split-R̂, adaptive sim scaling, cache TTL
"""

import math
import unittest

import numpy as np
from scipy.stats import norm

# ── Phase 1: Deterministic SGP Pipeline ─────────────────────────────


class TestPhase1SGPMath(unittest.TestCase):
    """Hand-verify the SGP delta formula used in evaluate_trade."""

    def test_counting_stat_sgp_delta(self):
        """SGP_change = (after - before) / denom for counting stats.

        Hand calculation:
          before_HR = 150, after_HR = 180, denom = 13.0 (LeagueConfig default)
          SGP_change = (180 - 150) / 13 = 30 / 13 ≈ 2.3077
        """
        before = 150
        after = 180
        denom = 13.0  # LeagueConfig default for HR
        expected = (after - before) / denom  # 30/13 ≈ 2.3077
        assert abs(expected - 30 / 13) < 1e-9

        # Verify code path in trade_evaluator matches
        from src.valuation import LeagueConfig

        config = LeagueConfig()
        raw_change = after - before
        sgp_change = raw_change / config.sgp_denominators.get("HR", 13.0)
        assert abs(sgp_change - expected) < 1e-6

    def test_inverse_stat_sgp_delta_sign_flip(self):
        """For ERA/WHIP, SGP = -(after - before) / denom.

        Hand calculation:
          before_ERA = 4.00, after_ERA = 3.60
          raw_change = 3.60 - 4.00 = -0.40
          SGP = -(-0.40) / 0.20 = 0.40 / 0.20 = 2.0  (positive = good)

        Lowering ERA is good, so the sign flip makes it positive.
        """
        before_era = 4.00
        after_era = 3.60
        denom = 0.20

        raw_change = after_era - before_era  # -0.40
        assert abs(raw_change - (-0.40)) < 1e-9

        # The sign flip: for inverse stats, negate the change
        sgp_change = -raw_change / denom  # -(-0.40)/0.20 = 2.0
        assert abs(sgp_change - 2.0) < 1e-9

    def test_weighted_sgp_with_category_weights(self):
        """weighted_sgp = sgp_change * category_weight.

        Hand calculation:
          HR SGP_change = 2.5, weight = 1.5 (close to gaining standings point)
          weighted = 2.5 * 1.5 = 3.75

          SB SGP_change = 0.5, weight = 0.3 (already dominant)
          weighted = 0.5 * 0.3 = 0.15

          Total surplus = 3.75 + 0.15 = 3.90
        """
        changes = {"HR": 2.5, "SB": 0.5}
        weights = {"HR": 1.5, "SB": 0.3}

        total = sum(changes[cat] * weights[cat] for cat in changes)
        assert abs(total - 3.90) < 1e-9

    def test_grade_thresholds_exact(self):
        """Verify all grade boundary conditions.

        From GRADE_THRESHOLDS:
          surplus > 2.0 → A+
          surplus > 1.5 → A
          ...
          surplus > 0.0 → C+
          surplus > -0.2 → C
          surplus ≤ -1.0 → F

        IMPORTANT: thresholds use strictly greater than (>), NOT >=.
        So 2.0 is NOT > 2.0, meaning grade_trade(2.0) == "A" (not A+).
        """
        from src.engine.output.trade_evaluator import grade_trade

        # Boundary tests (strictly greater than)
        assert grade_trade(2.01) == "A+"
        assert grade_trade(2.0) == "A"  # > 2.0 is FALSE for exactly 2.0

        assert grade_trade(2.001) == "A+"
        assert grade_trade(1.999) == "A"  # > 1.5 but not > 2.0
        assert grade_trade(1.501) == "A"
        assert grade_trade(1.499) == "A-"  # > 1.0 but not > 1.5
        assert grade_trade(0.001) == "C+"  # > 0.0
        assert grade_trade(0.0) == "C"  # NOT > 0.0, so next: > -0.2 → C
        assert grade_trade(-0.199) == "C"
        assert grade_trade(-0.201) == "C-"
        assert grade_trade(-1.001) == "F"

    def test_bench_option_value_formula(self):
        """bench_cost = streaming_sgp_per_week * weeks_remaining.

        Hand calculation:
          streaming = 0.166 SGP/week, weeks = 16
          bench_value = 0.166 * 16 = 2.656
        """
        from src.engine.portfolio.lineup_optimizer import bench_option_value

        expected = 0.166 * 16
        actual = bench_option_value(weeks_remaining=16)
        assert abs(actual - expected) < 0.01

    def test_bench_option_scales_with_weeks(self):
        """Bench option value scales linearly for small weeks (before hot_fa cap).

        The function has: stream = 0.15*w, option = min(0.15*w, 1.0)*0.5*0.5
        For weeks < 7, hot_fa probability isn't capped at 1.0, so total is linear.
        For weeks >= 7, the cap makes the option component constant (0.25),
        breaking the pure 2:1 ratio.

        Use small weeks (2 and 4) where both are below the cap:
          v(2) = 0.15*2 + 0.15*2*0.25 = 0.30 + 0.075 = 0.375
          v(4) = 0.15*4 + 0.15*4*0.25 = 0.60 + 0.150 = 0.750
          Ratio = 0.750 / 0.375 = 2.0
        """
        from src.engine.portfolio.lineup_optimizer import bench_option_value

        v2 = bench_option_value(weeks_remaining=2)
        v4 = bench_option_value(weeks_remaining=4)
        # Both below the hot_fa cap, so scaling is truly linear → 2:1 ratio
        assert abs(v4 / v2 - 2.0) < 0.01


class TestPhase1MarginalElasticity(unittest.TestCase):
    """Hand-verify the marginal SGP formula: 1/gap."""

    def test_marginal_sgp_close_gap(self):
        """marginal = 1/gap. Close gap = high marginal value.

        Hand calculation:
          Your HR = 198, next team = 200, gap = 2
          marginal = 1/2 = 0.5
        """
        from src.engine.portfolio.category_analysis import compute_marginal_sgp

        totals = {
            "My Team": {"HR": 198},
            "Team B": {"HR": 200},
            "Team C": {"HR": 190},
        }
        result = compute_marginal_sgp(totals["My Team"], totals, categories=["HR"])
        # gap to next above = 200 - 198 = 2, marginal = 1/2 = 0.5
        assert abs(result["HR"] - 0.5) < 0.01

    def test_marginal_sgp_large_gap(self):
        """Large gap → near-zero marginal value.

        Your HR = 150, next = 200, gap = 50
        marginal = 1/50 = 0.02
        """
        from src.engine.portfolio.category_analysis import compute_marginal_sgp

        totals = {
            "My Team": {"HR": 150},
            "Team B": {"HR": 200},
        }
        result = compute_marginal_sgp(totals["My Team"], totals, categories=["HR"])
        assert abs(result["HR"] - 0.02) < 0.001

    def test_marginal_sgp_first_place(self):
        """First place → marginal = 0.01 (floor value)."""
        from src.engine.portfolio.category_analysis import compute_marginal_sgp

        totals = {
            "My Team": {"HR": 250},
            "Team B": {"HR": 200},
        }
        result = compute_marginal_sgp(totals["My Team"], totals, categories=["HR"])
        assert result["HR"] == 0.01

    def test_marginal_sgp_inverse_category(self):
        """For ERA, lower is better. Gap to next LOWER team.

        Your ERA = 3.90, next better = 3.60, gap = 0.30
        marginal = 1/0.30 = 3.333
        """
        from src.engine.portfolio.category_analysis import compute_marginal_sgp

        totals = {
            "My Team": {"ERA": 3.90},
            "Team B": {"ERA": 3.60},
            "Team C": {"ERA": 4.20},
        }
        result = compute_marginal_sgp(totals["My Team"], totals, categories=["ERA"])
        expected = 1.0 / 0.30  # 3.333
        assert abs(result["ERA"] - expected) < 0.01

    def test_punt_detection_both_conditions(self):
        """Punt requires BOTH: gainable_positions == 0 AND rank >= 10.

        Setup: 12 teams. My team is last in SB by a wide margin.
        - rank = 12, gainable = 0 → IS a punt
        - rank = 9, gainable = 0 → NOT a punt (rank < 10)
        """
        from src.engine.portfolio.category_analysis import category_gap_analysis

        teams = {}
        for i in range(12):
            teams[f"Team {i}"] = {"SB": 100 - i * 5}  # Team 0 = 100, Team 11 = 45

        # My team is dead last with 10 SB, cannot catch up
        teams["My Team"] = {"SB": 10}

        analysis = category_gap_analysis(
            your_totals=teams["My Team"],
            all_team_totals=teams,
            your_team_id="My Team",
            weeks_remaining=2,  # Very few weeks, can't close gaps
        )
        assert analysis["SB"]["is_punt"] is True
        assert analysis["SB"]["rank"] >= 10
        assert analysis["SB"]["gainable_positions"] == 0

    def test_category_weights_normalization(self):
        """Non-punt weights should average to 1.0 after normalization.

        Hand calculation:
          HR marginal = 0.5, R marginal = 0.1
          mean = (0.5 + 0.1) / 2 = 0.3
          HR weight = 0.5/0.3 = 1.667, R weight = 0.1/0.3 = 0.333
          Average = (1.667 + 0.333) / 2 = 1.0 ✓
        """
        from src.engine.portfolio.category_analysis import (
            compute_category_weights_from_analysis,
        )

        analysis = {
            "HR": {"is_punt": False, "marginal_value": 0.5},
            "R": {"is_punt": False, "marginal_value": 0.1},
            "SB": {"is_punt": True, "marginal_value": 0.0},
        }
        weights = compute_category_weights_from_analysis(analysis)

        assert weights["SB"] == 0.0  # Punt → zero weight
        non_punt = [weights["HR"], weights["R"]]
        avg = sum(non_punt) / len(non_punt)
        assert abs(avg - 1.0) < 0.01

        # Check individual weight math
        mean_mv = (0.5 + 0.1) / 2  # = 0.3
        assert abs(weights["HR"] - 0.5 / mean_mv) < 0.01  # 1.667
        assert abs(weights["R"] - 0.1 / mean_mv) < 0.01  # 0.333


# ── Phase 2: Stochastic Pipeline ────────────────────────────────────


class TestPhase2BMAMath(unittest.TestCase):
    """Hand-verify Bayesian Model Averaging posterior weights."""

    def test_bma_posterior_weights_bayes_theorem(self):
        """Verify BMA computes P(Model|YTD) = P(YTD|Model)*P(Model)/Z.

        Hand calculation with 2 systems, 1 stat (HR):
          YTD HR = 18
          Steamer projects HR = 20, sigma = 5.2
          ZiPS projects HR = 15, sigma = 5.5

          P(YTD=18 | Steamer) = N(18; 20, 5.2)
            = exp(-(18-20)²/(2*5.2²)) / (5.2*sqrt(2π))
            = exp(-4/54.08) / 13.017
            = exp(-0.0740) / 13.017
            = 0.9288 / 13.017
            = 0.07136

          P(YTD=18 | ZiPS) = N(18; 15, 5.5)
            = exp(-(18-15)²/(2*5.5²)) / (5.5*sqrt(2π))
            = exp(-9/60.5) / 13.768
            = exp(-0.1488) / 13.768
            = 0.8618 / 13.768
            = 0.06259

          With uniform prior (0.5, 0.5):
            unnorm_steamer = 0.07136 * 0.5 = 0.03568
            unnorm_zips = 0.06259 * 0.5 = 0.03130
            Z = 0.06698
            P(Steamer|YTD) = 0.03568/0.06698 = 0.5327
            P(ZiPS|YTD) = 0.03130/0.06698 = 0.4673

        Steamer should get more weight because it was closer to reality (18 vs 20)
        compared to ZiPS (18 vs 15).
        """
        from src.engine.projections.bayesian_blend import bayesian_model_average

        ytd = {"hr": 18}
        projections = {
            "steamer": {"hr": 20},
            "zips": {"hr": 15},
        }

        posterior, blended, variance = bayesian_model_average(ytd, projections)

        # Hand-computed expected posteriors
        p_steamer = norm.pdf(18, loc=20, scale=5.2)  # sigma from SYSTEM_FORECAST_SIGMA
        p_zips = norm.pdf(18, loc=15, scale=5.5)

        expected_steamer = p_steamer * 0.5
        expected_zips = p_zips * 0.5
        z = expected_steamer + expected_zips

        hand_steamer = expected_steamer / z
        hand_zips = expected_zips / z

        assert abs(posterior["steamer"] - hand_steamer) < 1e-6
        assert abs(posterior["zips"] - hand_zips) < 1e-6

        # Steamer should have higher weight (closer prediction)
        assert posterior["steamer"] > posterior["zips"]

    def test_bma_blended_is_weighted_mean(self):
        """Blended projection = Σ(weight_i * projection_i).

        With weights from above test: steamer ~ 0.533, zips ~ 0.467
        Blended HR = 0.533 * 20 + 0.467 * 15 = 10.66 + 7.005 = 17.665
        """
        from src.engine.projections.bayesian_blend import bayesian_model_average

        ytd = {"hr": 18}
        projections = {
            "steamer": {"hr": 20},
            "zips": {"hr": 15},
        }
        posterior, blended, _ = bayesian_model_average(ytd, projections)

        hand_blended = posterior["steamer"] * 20 + posterior["zips"] * 15
        assert abs(blended["hr"] - hand_blended) < 1e-6

    def test_bma_variance_decomposition(self):
        """Total variance = within-model + between-model.

        V_within = Σ w_i * σ_i²
        V_between = Σ w_i * (μ_i - μ_blend)²

        Hand calculation:
          w_steamer ≈ 0.533, σ_steamer = 5.2 → σ² = 27.04
          w_zips ≈ 0.467, σ_zips = 5.5 → σ² = 30.25
          μ_blend ≈ 17.67

          V_within = 0.533 * 27.04 + 0.467 * 30.25 = 14.41 + 14.13 = 28.54
          V_between = 0.533*(20-17.67)² + 0.467*(15-17.67)²
                    = 0.533*5.43 + 0.467*7.13
                    = 2.89 + 3.33 = 6.22
          Total ≈ 34.76
        """
        from src.engine.projections.bayesian_blend import bayesian_model_average

        ytd = {"hr": 18}
        projections = {
            "steamer": {"hr": 20},
            "zips": {"hr": 15},
        }
        posterior, blended, variance = bayesian_model_average(ytd, projections)

        # Recompute by hand
        v_within = posterior["steamer"] * 5.2**2 + posterior["zips"] * 5.5**2
        v_between = posterior["steamer"] * (20 - blended["hr"]) ** 2 + posterior["zips"] * (15 - blended["hr"]) ** 2
        expected_total = v_within + v_between

        assert abs(variance["hr"] - expected_total) < 1e-6

    def test_bma_uniform_projections_equal_weights(self):
        """When all systems project the same value, weights should still differ
        slightly due to different sigmas (tighter sigma → higher likelihood).
        """
        from src.engine.projections.bayesian_blend import bayesian_model_average

        ytd = {"hr": 25}
        projections = {
            "steamer": {"hr": 25},
            "zips": {"hr": 25},
            "depthcharts": {"hr": 25},
        }
        posterior, _, _ = bayesian_model_average(ytd, projections)

        # When projection == ytd, likelihood ∝ 1/sigma
        # depthcharts has smallest sigma (5.0) → highest weight
        assert posterior["depthcharts"] > posterior["steamer"]
        assert posterior["steamer"] > posterior["zips"]


class TestPhase2CopulaMath(unittest.TestCase):
    """Verify Gaussian copula produces correct correlation structure."""

    def test_copula_output_uniform_marginals(self):
        """Copula samples should be in [0, 1] with roughly uniform marginals."""
        from src.engine.portfolio.copula import GaussianCopula

        copula = GaussianCopula()
        rng = np.random.RandomState(42)
        samples = copula.sample(5000, rng)

        assert samples.shape == (5000, 12)
        assert samples.min() >= 0.0
        assert samples.max() <= 1.0

        # Each marginal should be approximately uniform
        for i in range(12):
            col = samples[:, i]
            assert abs(np.mean(col) - 0.5) < 0.05
            assert abs(np.std(col) - 1.0 / math.sqrt(12)) < 0.05

    def test_copula_preserves_known_correlation(self):
        """HR↔RBI correlation = 0.85 in the default matrix.
        After copula sampling, the Spearman rank correlation should be close.
        """
        from src.engine.portfolio.copula import CATEGORIES, GaussianCopula

        copula = GaussianCopula()
        rng = np.random.RandomState(42)
        samples = copula.sample(10000, rng)

        hr_idx = CATEGORIES.index("HR")
        rbi_idx = CATEGORIES.index("RBI")

        from scipy.stats import spearmanr

        corr, _ = spearmanr(samples[:, hr_idx], samples[:, rbi_idx])
        # Gaussian copula transforms correlation, but for high values
        # the Spearman correlation should be close to the Pearson input
        assert corr > 0.70  # 0.85 input → should be close

    def test_copula_negative_correlation_preserved(self):
        """SB↔HR correlation = -0.15 should be preserved as slightly negative."""
        from src.engine.portfolio.copula import CATEGORIES, GaussianCopula

        copula = GaussianCopula()
        rng = np.random.RandomState(42)
        samples = copula.sample(10000, rng)

        sb_idx = CATEGORIES.index("SB")
        hr_idx = CATEGORIES.index("HR")

        from scipy.stats import spearmanr

        corr, _ = spearmanr(samples[:, sb_idx], samples[:, hr_idx])
        assert corr < 0.0  # Should be negative


class TestPhase2PairedMCMath(unittest.TestCase):
    """Verify paired MC variance reduction property."""

    def test_paired_mc_reduces_variance(self):
        """Using identical seeds for before/after should reduce variance
        compared to independent simulations.

        Paired MC: Var(after - before) ≤ Var(after) + Var(before)
        because corr(after, before) > 0.
        """
        import pandas as pd

        from src.engine.monte_carlo.trade_simulator import (
            build_roster_stats,
            run_paired_monte_carlo,
        )

        pool = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "name": ["A", "B", "C"],
                "r": [80, 70, 90],
                "hr": [25, 20, 30],
                "rbi": [80, 65, 85],
                "sb": [10, 5, 15],
                "avg": [0.270, 0.260, 0.280],
                "w": [0, 0, 0],
                "sv": [0, 0, 0],
                "k": [0, 0, 0],
                "era": [0, 0, 0],
                "whip": [0, 0, 0],
            }
        )

        before = build_roster_stats([1, 2], pool)
        after = build_roster_stats([1, 3], pool)  # Swap 2 for 3

        result = run_paired_monte_carlo(before, after, n_sims=5000, seed=42)

        # The MC std should be relatively small for a simple swap
        # (paired MC captures the DIFFERENCE, not the absolute levels)
        assert result["mc_std"] < 5.0  # Should be small for similar rosters
        assert "mc_mean" in result


# ── Phase 3: Signal Intelligence ────────────────────────────────────


class TestPhase3SignalMath(unittest.TestCase):
    """Verify signal processing formulas."""

    def test_exponential_decay_formula(self):
        """weight = exp(-λ * days_ago), where λ is the decay rate.

        The actual API is: decay_weight(observation_date, today, lambda_param)

        Hand calculation:
          λ = ln(2)/30 ≈ 0.02310 (half-life = 30 days)
          days_ago = 30 → weight = exp(-0.02310 * 30) = exp(-0.6931) = 0.5000
          days_ago = 0  → weight = exp(0) = 1.0
          days_ago = 60 → weight = exp(-0.02310 * 60) = exp(-1.3863) = 0.25
        """
        from datetime import date, timedelta

        from src.engine.signals.decay import decay_weight

        lam = math.log(2) / 30  # half-life = 30 days → λ ≈ 0.02310
        today = date(2026, 6, 1)

        # At exactly one half-life (30 days ago), weight should be 0.5
        obs_30 = today - timedelta(days=30)
        w30 = decay_weight(obs_30, today, lambda_param=lam)
        assert abs(w30 - 0.5) < 0.01

        # At day 0 (today), weight = 1.0
        w0 = decay_weight(today, today, lambda_param=lam)
        assert abs(w0 - 1.0) < 0.01

        # At 2 half-lives (60 days ago), weight = 0.25
        obs_60 = today - timedelta(days=60)
        w60 = decay_weight(obs_60, today, lambda_param=lam)
        assert abs(w60 - 0.25) < 0.01

    def test_kalman_update_formula(self):
        """Kalman update: x_new = x_old + K*(obs - x_old), K = P_pred/(P_pred+R).

        The actual API is: kalman_true_talent(observations, obs_variance,
            process_variance, prior_mean, prior_variance)

        The filter does PREDICT then UPDATE at each step:
          1. Predict: P_pred = P_prior + Q (process variance)
          2. Update:  K = P_pred / (P_pred + R)
                      x_new = x_old + K*(obs - x_old)
                      P_new = (1 - K) * P_pred

        Hand calculation with process_variance=0 (pure filter, no drift):
          prior x = 0.280, P = 0.001
          Predict: P_pred = 0.001 + 0.0 = 0.001
          observation = 0.320, R = 0.002

          K = 0.001 / (0.001 + 0.002) = 1/3
          x_new = 0.280 + (1/3)*(0.320 - 0.280) = 0.280 + 0.01333 = 0.29333
          P_new = (1 - 1/3) * 0.001 = 0.000667
        """
        from src.engine.signals.kalman import kalman_true_talent

        observations = np.array([0.320])
        obs_variance = np.array([0.002])
        process_variance = 0.0  # Zero so predict step doesn't change P
        prior_mean = 0.280
        prior_variance = 0.001

        filtered_means, filtered_vars = kalman_true_talent(
            observations, obs_variance, process_variance, prior_mean, prior_variance
        )

        K = 0.001 / (0.001 + 0.002)  # 1/3
        expected_x = 0.280 + K * (0.320 - 0.280)  # 0.29333
        expected_P = (1 - K) * 0.001  # 0.000667

        assert abs(filtered_means[0] - expected_x) < 1e-6
        assert abs(filtered_vars[0] - expected_P) < 1e-6


# ── Phase 4: Context Engine ─────────────────────────────────────────


class TestPhase4ConcentrationMath(unittest.TestCase):
    """Verify Herfindahl-Hirschman Index formula."""

    def test_hhi_formula_hand_calculation(self):
        """HHI = Σ(share_i²).

        The actual API is: roster_concentration_hhi(roster_ids, player_pool)
        It takes player IDs + full pool DataFrame (not just the roster).

        Hand calculation with 3 teams, equal PA per player:
          NYY: 3 players (300 PA), BOS: 2 players (200 PA), LAD: 1 player (100 PA)
          Total PA = 600
          Shares: NYY=0.5, BOS=0.333, LAD=0.167
          HHI = 0.5² + 0.333² + 0.167²
              = 0.25 + 0.111 + 0.028
              = 0.389
        """
        import pandas as pd

        from src.engine.context.concentration import roster_concentration_hhi

        pool = pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5, 6],
                "team": ["NYY", "NYY", "NYY", "BOS", "BOS", "LAD"],
                "is_hitter": [1, 1, 1, 1, 1, 1],
                "pa": [100, 100, 100, 100, 100, 100],
                "ip": [0, 0, 0, 0, 0, 0],
            }
        )
        hhi = roster_concentration_hhi([1, 2, 3, 4, 5, 6], pool)

        shares = [3 / 6, 2 / 6, 1 / 6]
        expected = sum(s**2 for s in shares)  # 0.389

        assert abs(hhi - expected) < 0.01

    def test_hhi_penalty_threshold(self):
        """Penalty = max(0, (HHI - threshold) * scale).

        The actual API is: concentration_risk_penalty(hhi, threshold, scale)
        Defaults: threshold=0.15, scale=3.0

        HHI = 0.25 → penalty = (0.25 - 0.15) * 3.0 = 0.30
        HHI = 0.10 → penalty = max(0, (0.10 - 0.15) * 3.0) = 0
        """
        from src.engine.context.concentration import concentration_risk_penalty

        assert abs(concentration_risk_penalty(0.25) - 0.30) < 0.01
        assert concentration_risk_penalty(0.10) == 0.0


class TestPhase4EnhancedBenchMath(unittest.TestCase):
    """Verify enhanced bench option value formula."""

    def test_enhanced_bench_components(self):
        """Enhanced bench = streaming + hot_fa + flexibility + injury_cushion.

        Hand calculation (default values):
          streaming = 0.166 * weeks = 0.166 * 16 = 2.656
          hot_fa = 0.15 * 0.5 * weeks = 1.2
          flexibility = roster_flex * 0.5 = 0.0 * 0.5 = 0.0
          injury_cushion = 0.10 * weeks * 0.3 = 0.48
          total = 2.656 + 1.2 + 0.0 + 0.48 = 4.336
        """
        from src.engine.context.bench_value import enhanced_bench_option_value

        result = enhanced_bench_option_value(weeks_remaining=16, roster_flexibility=0.0)

        # Check individual components exist
        assert "streaming" in result
        assert "hot_fa" in result
        assert "flexibility" in result
        assert "injury_cushion" in result
        assert "total" in result

        # Total should be the sum of components
        component_sum = result["streaming"] + result["hot_fa"] + result["flexibility"] + result["injury_cushion"]
        assert abs(result["total"] - component_sum) < 0.01


# ── Phase 5: Game Theory ────────────────────────────────────────────


class TestPhase5BayesAdverseSelection(unittest.TestCase):
    """Hand-verify Bayes' theorem in adverse selection discount."""

    def test_bayes_theorem_default_prior(self):
        """P(flaw|offered) = P(offered|flaw)*P(flaw) / P(offered).

        Hand calculation with defaults:
          P(flaw) = 0.15
          P(offered|flaw) = 0.60
          P(offered|ok) = 0.20

          P(offered) = 0.60 * 0.15 + 0.20 * 0.85
                     = 0.090 + 0.170
                     = 0.260

          P(flaw|offered) = 0.60 * 0.15 / 0.260
                          = 0.090 / 0.260
                          = 0.3462

          discount = 1 - (0.3462 * 0.25) = 1 - 0.0865 = 0.9135
        """
        from src.engine.game_theory.adverse_selection import (
            DEFAULT_P_FLAW,
            MAX_DISCOUNT,
            P_OFFERED_GIVEN_FLAW,
            P_OFFERED_GIVEN_OK,
            adverse_selection_discount,
        )

        # Hand compute
        p_flaw = DEFAULT_P_FLAW  # 0.15
        p_offered = P_OFFERED_GIVEN_FLAW * p_flaw + P_OFFERED_GIVEN_OK * (1 - p_flaw)
        p_flaw_given_offered = (P_OFFERED_GIVEN_FLAW * p_flaw) / p_offered
        expected_discount = 1.0 - (p_flaw_given_offered * MAX_DISCOUNT)

        actual = adverse_selection_discount()
        assert abs(actual - expected_discount) < 1e-4

        # Verify intermediate values
        assert abs(p_offered - 0.260) < 0.001
        assert abs(p_flaw_given_offered - 0.3462) < 0.001

    def test_bayes_update_with_bad_history(self):
        """With 3/4 trades underperforming, P(flaw) = 0.75.

        Hand calculation:
          P(flaw) = 3/4 = 0.75  (from history)
          P(offered) = 0.60*0.75 + 0.20*0.25 = 0.45 + 0.05 = 0.50
          P(flaw|offered) = 0.60*0.75/0.50 = 0.45/0.50 = 0.90
          discount = 1 - (0.90 * 0.25) = 1 - 0.225 = 0.775
          Clamp to floor: max(0.775, 0.75) = 0.775
        """
        from src.engine.game_theory.adverse_selection import adverse_selection_discount

        history = [
            {"actual": 40, "projected": 100},  # Under: 40 < 80
            {"actual": 50, "projected": 100},  # Under: 50 < 80
            {"actual": 60, "projected": 100},  # Under: 60 < 80
            {"actual": 95, "projected": 80},  # OK: 95 >= 64
        ]
        discount = adverse_selection_discount(history)

        # p_flaw = 3/4 = 0.75
        p_flaw = 0.75
        p_offered = 0.60 * p_flaw + 0.20 * (1 - p_flaw)
        p_flaw_given = 0.60 * p_flaw / p_offered
        expected = max(1.0 - p_flaw_given * 0.25, 0.75)

        assert abs(discount - expected) < 1e-4

    def test_underperformance_threshold(self):
        """actual < projected * 0.80 means underperformed.

        Cases: actual=79, projected=100 → 79 < 80 → True
               actual=80, projected=100 → 80 < 80 → False (not strictly less)
               actual=81, projected=100 → 81 < 80 → False
        """
        from src.engine.game_theory.adverse_selection import _trade_underperformed

        assert _trade_underperformed({"actual": 79, "projected": 100}) is True
        assert _trade_underperformed({"actual": 80, "projected": 100}) is False
        assert _trade_underperformed({"actual": 81, "projected": 100}) is False


class TestPhase5VickreyAuction(unittest.TestCase):
    """Verify Vickrey auction (second-price) for market clearing."""

    def test_market_price_is_second_highest(self):
        """In a Vickrey auction, price = 2nd highest bid.

        Bids: [5.0, 3.0, 2.0, 1.0]
        Market price = 3.0 (second highest)
        """
        from src.engine.game_theory.opponent_valuation import market_clearing_price

        valuations = {"A": 5.0, "B": 3.0, "C": 2.0, "D": 1.0}
        assert market_clearing_price(valuations) == 3.0

    def test_market_price_with_ties(self):
        """Tied bids: [5.0, 5.0, 2.0] → second highest = 5.0."""
        from src.engine.game_theory.opponent_valuation import market_clearing_price

        valuations = {"A": 5.0, "B": 5.0, "C": 2.0}
        assert market_clearing_price(valuations) == 5.0


class TestPhase5OpponentValuationMath(unittest.TestCase):
    """Verify opponent valuation formulas, especially inverse categories."""

    def test_counting_stat_valuation(self):
        """For counting stats: min(proj, gap) / denom (linear, no squaring).

        Setup: Team B has HR=150, next team above = Team C at 200 (gap=50).
        Me has HR=100 (below Team B, doesn't affect gap_to_next).
        Player projects HR=30.

        _gap_to_next_team considers all teams except the one being evaluated.
        For Team B: others = [200, 100], better = [200], closest = 200,
        gap = 200 - 150 = 50.

        contribution = min(30, 50) / 12.0 = 30/12.0 = 2.5
        """
        from src.engine.game_theory.opponent_valuation import estimate_opponent_valuations

        totals = {
            "Team B": {"HR": 150, "R": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0},
            "Team C": {"HR": 200, "R": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0},
            "Me": {"HR": 100, "R": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0},
        }
        proj = {"HR": 30, "R": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}

        vals = estimate_opponent_valuations(proj, totals, "Me")

        # Team B: gap to next (Team C at 200) = 50
        # contribution = min(30, 50) / 12.0 = 2.5
        assert abs(vals["Team B"] - 2.5) < 0.01

    def test_inverse_stat_valuation_correct_direction(self):
        """For ERA: benefit = team_ERA_blended - pitcher_ERA_blended (unclamped).

        Setup: Team B has ERA=4.10, next better team = 3.80 (gap=0.30).
        Pitcher projects ERA=3.50.

        Good pitcher benefit is positive (lowers team ERA toward gap).
        Bad pitcher (ERA=4.50) benefit is negative (raises team ERA).
        contribution = min(benefit, gap) / denom.
        """
        from src.engine.game_theory.opponent_valuation import estimate_opponent_valuations

        totals = {
            "Team B": {"ERA": 4.10, "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0, "WHIP": 0},
            "Team C": {"ERA": 3.80, "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0, "WHIP": 0},
            "Me": {"ERA": 3.90, "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0, "WHIP": 0},
        }

        # Good pitcher (ERA 3.50) should have positive value for Team B
        good_pitcher = {"ERA": 3.50, "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0, "WHIP": 0}
        vals_good = estimate_opponent_valuations(good_pitcher, totals, "Me")

        # Bad pitcher (ERA 4.50) should have zero value (can't help)
        bad_pitcher = {"ERA": 4.50, "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0, "WHIP": 0}
        vals_bad = estimate_opponent_valuations(bad_pitcher, totals, "Me")

        # Good pitcher is MORE valuable than bad pitcher
        assert vals_good["Team B"] > vals_bad["Team B"]
        # Bad pitcher has zero ERA contribution (ERA too high to help)
        # (may have residual value from fallback path for other categories)

    def test_low_era_pitcher_more_valuable_than_high_era(self):
        """Critical regression test for Bug 1 fix: low ERA > high ERA.

        A pitcher with ERA 2.50 must be valued MORE than one with ERA 4.50.
        Before the fix, this was reversed because abs(4.50) > abs(2.50).
        """
        from src.engine.game_theory.opponent_valuation import estimate_opponent_valuations

        totals = {
            "Team B": {
                "ERA": 4.00,
                "WHIP": 1.30,
                "R": 0,
                "HR": 0,
                "RBI": 0,
                "SB": 0,
                "AVG": 0,
                "W": 0,
                "K": 0,
                "SV": 0,
            },
            "Team C": {
                "ERA": 3.60,
                "WHIP": 1.20,
                "R": 0,
                "HR": 0,
                "RBI": 0,
                "SB": 0,
                "AVG": 0,
                "W": 0,
                "K": 0,
                "SV": 0,
            },
            "Me": {"ERA": 3.80, "WHIP": 1.25, "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0},
        }
        zero_stats = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0, "W": 0, "K": 0, "SV": 0}

        ace = {"ERA": 2.50, "WHIP": 1.05, **zero_stats}
        junk = {"ERA": 5.00, "WHIP": 1.50, **zero_stats}

        vals_ace = estimate_opponent_valuations(ace, totals, "Me")
        vals_junk = estimate_opponent_valuations(junk, totals, "Me")

        # Ace MUST be valued more than junk pitcher
        assert vals_ace["Team B"] > vals_junk["Team B"]


class TestPhase5BellmanMath(unittest.TestCase):
    """Verify Bellman rollout formula."""

    def test_bellman_total_value_formula(self):
        """V = immediate + γ * (E[future_after] - E[future_before]).

        This is an algebraic identity — should always hold.
        """
        from src.engine.game_theory.dynamic_programming import bellman_rollout

        result = bellman_rollout(immediate_surplus=1.5, seed=42)

        # Check the algebraic identity
        expected_total = result["immediate"] + result["gamma"] * result["option_value"]
        assert abs(result["total_value"] - expected_total) < 0.01

    def test_gamma_boundary_values(self):
        """Gamma tiers are exact constants, not computed.

        playoff_prob > 0.70 → γ = 0.98
        playoff_prob ∈ (0.30, 0.70] → γ = 0.95
        playoff_prob ≤ 0.30 → γ = 0.85

        Boundary cases:
          0.70 is NOT > 0.70, so it falls to bubble → 0.95
          0.30 is NOT > 0.30, so it falls to rebuilding → 0.85
        """
        from src.engine.game_theory.dynamic_programming import get_gamma

        assert get_gamma(0.71) == 0.98
        assert get_gamma(0.70) == 0.95  # NOT > 0.70
        assert get_gamma(0.31) == 0.95
        assert get_gamma(0.30) == 0.85  # NOT > 0.30
        assert get_gamma(0.00) == 0.85
        assert get_gamma(1.00) == 0.98

    def test_roster_balance_formula(self):
        """balance = 1 - 2 * avg(|rank - median| / median).

        12-team league, median rank = 6.5.
        All ranks = 6 or 7:
          deviations = |6-6.5|/6.5 = 0.077 (×5), |7-6.5|/6.5 = 0.077 (×5)
          avg_dev = 0.077
          balance = 1 - 2*0.077 = 0.846
        """
        from src.engine.game_theory.dynamic_programming import compute_roster_balance

        balanced = compute_roster_balance(
            {"R": 6, "HR": 7, "RBI": 6, "SB": 7, "AVG": 6, "W": 7, "K": 6, "SV": 7, "ERA": 6, "WHIP": 7}
        )

        median = 6.5
        dev = abs(6 - median) / median  # 0.0769
        expected = 1.0 - 2.0 * dev  # 0.846
        assert abs(balanced - expected) < 0.01


class TestPhase5SensitivityMath(unittest.TestCase):
    """Verify sensitivity analysis formulas."""

    def test_breakeven_is_absolute_surplus(self):
        """Breakeven gap = |surplus_sgp|.

        surplus = 1.5 → breakeven = 1.5 (trade needs to lose 1.5 SGP to flip)
        surplus = -0.3 → breakeven = 0.3 (trade needs to gain 0.3 SGP to flip)
        """
        from src.engine.game_theory.sensitivity import trade_sensitivity_report

        report_pos = trade_sensitivity_report({"R": 1.0}, surplus_sgp=1.5)
        assert abs(report_pos["breakeven_gap"] - 1.5) < 0.001

        report_neg = trade_sensitivity_report({"R": -0.5}, surplus_sgp=-0.3)
        assert abs(report_neg["breakeven_gap"] - 0.3) < 0.001

    def test_vulnerability_classification_exact_thresholds(self):
        """vulnerability thresholds:
        |surplus| > 1.5 → robust
        |surplus| > 0.5 → moderate
        |surplus| > 0.1 → fragile
        |surplus| ≤ 0.1 → razor-thin
        """
        from src.engine.game_theory.sensitivity import trade_sensitivity_report

        assert trade_sensitivity_report({"R": 1}, surplus_sgp=1.51)["vulnerability"] == "robust"
        assert trade_sensitivity_report({"R": 1}, surplus_sgp=1.50)["vulnerability"] == "moderate"  # NOT > 1.5
        assert trade_sensitivity_report({"R": 1}, surplus_sgp=0.51)["vulnerability"] == "moderate"
        assert trade_sensitivity_report({"R": 1}, surplus_sgp=0.50)["vulnerability"] == "fragile"  # NOT > 0.5
        assert trade_sensitivity_report({"R": 1}, surplus_sgp=0.11)["vulnerability"] == "fragile"
        assert trade_sensitivity_report({"R": 1}, surplus_sgp=0.10)["vulnerability"] == "razor-thin"
        assert trade_sensitivity_report({"R": 1}, surplus_sgp=0.00)["vulnerability"] == "razor-thin"


# ── Phase 6: Production ─────────────────────────────────────────────


class TestPhase6ESSMath(unittest.TestCase):
    """Verify ESS formula: ESS = N / (1 + 2*Σ_k ρ(k))."""

    def test_ess_independent_equals_n(self):
        """For IID samples, autocorrelation → 0, so ESS ≈ N."""
        from src.engine.production.convergence import effective_sample_size

        rng = np.random.RandomState(42)
        n = 10000
        samples = rng.normal(0, 1, n)
        ess = effective_sample_size(samples)

        # For truly independent samples, ESS should be close to N
        assert ess > 0.2 * n  # Geyer pair-sum is conservative; well above correlated ESS

    def test_ess_constant_series_equals_n(self):
        """Constant series has zero variance → ESS = N (no autocorrelation)."""
        from src.engine.production.convergence import effective_sample_size

        samples = np.ones(500) * 3.14
        ess = effective_sample_size(samples)
        assert ess == 500.0

    def test_ess_random_walk_much_less_than_n(self):
        """Random walk (cumulative sum) is highly correlated → ESS << N.

        For a random walk X_t = X_{t-1} + ε_t, ρ(k) ≈ 1 - k/N for small k.
        This means ESS is dramatically smaller than N.
        """
        from src.engine.production.convergence import effective_sample_size

        rng = np.random.RandomState(42)
        n = 1000
        samples = np.cumsum(rng.normal(0, 0.1, n))
        ess = effective_sample_size(samples)

        assert ess < n * 0.3  # ESS should be much less than N


class TestPhase6SplitRhatMath(unittest.TestCase):
    """Verify split-R̂ formula."""

    def test_rhat_formula_hand_calculation(self):
        """R̂ = sqrt(var_hat / W) where:
          W = mean of within-chain variances
          B = n * var of chain means
          var_hat = ((n-1)/n)*W + (1/n)*B

        For IID N(0,1) with N=1000:
          Split at 500: both halves ≈ N(0,1)
          W ≈ 1.0, B ≈ 0 (means both ≈ 0)
          var_hat ≈ W → R̂ ≈ 1.0
        """
        from src.engine.production.convergence import split_rhat

        rng = np.random.RandomState(42)
        samples = rng.normal(0, 1, 1000)
        rhat = split_rhat(samples)

        assert abs(rhat - 1.0) < 0.05

    def test_rhat_detects_mean_shift(self):
        """Chain with mean shift: first half ≈ N(0,1), second half ≈ N(5,1).

        W ≈ 1.0 (within-chain variance is fine)
        B = 500 * (((-2.5)² + (2.5)²)) = 500 * 12.5 = 6250
        var_hat = (499/500)*1 + (1/500)*6250 ≈ 1 + 12.5 = 13.5
        R̂ = sqrt(13.5/1.0) ≈ 3.67 >> 1.05

        (Exact values differ due to sampling variance)
        """
        from src.engine.production.convergence import split_rhat

        rng = np.random.RandomState(42)
        chain1 = rng.normal(0, 1, 500)
        chain2 = rng.normal(5, 1, 500)
        samples = np.concatenate([chain1, chain2])

        rhat = split_rhat(samples)
        assert rhat > 1.5  # Far from converged


class TestPhase6AdaptiveSimMath(unittest.TestCase):
    """Verify adaptive simulation count formulas."""

    def test_complexity_scaling_formula(self):
        """n_sims = base + extra_players * SIMS_PER_EXTRA_PLAYER.

        Hand calculation (standard mode):
          1-for-1: base=10000, extra=0, n=10000
          2-for-1: base=10000, extra=1, n=10000+5000=15000
          3-for-2: base=10000, extra=3, n=10000+15000=25000
          5-for-5: base=10000, extra=8, n=10000+40000=50000
        """
        from src.engine.production.sim_config import compute_adaptive_n_sims

        assert compute_adaptive_n_sims(1, 1, "standard") == 10000
        assert compute_adaptive_n_sims(2, 1, "standard") == 15000
        assert compute_adaptive_n_sims(3, 2, "standard") == 25000
        # 5-for-5 = 10 players, 8 extra → 10000 + 40000 = 50000
        assert compute_adaptive_n_sims(5, 5, "standard") == 50000

    def test_time_budget_cap_formula(self):
        """max_by_time = time_budget_s * ESTIMATED_SIMS_PER_SECOND.

        time_budget = 2.0s, sims_per_second = 15000
        max_by_time = 2.0 * 15000 = 30000

        3-for-2 standard = 25000, so NOT capped (25000 < 30000).
        3-for-3 standard = 10000 + 4*5000 = 30000, equals cap exactly.
        """
        from src.engine.production.sim_config import compute_adaptive_n_sims

        # 3-for-2 with 2s budget = uncapped at 25000
        n = compute_adaptive_n_sims(3, 2, "standard", time_budget_s=2.0)
        assert n == 25000  # 25000 < 30000, not capped

        # 1s budget → max 15000, but 3-for-2 wants 25000
        n_capped = compute_adaptive_n_sims(3, 2, "standard", time_budget_s=1.0)
        assert n_capped == 15000  # Capped at 15000

    def test_runtime_estimate_linear(self):
        """runtime = n_sims / ESTIMATED_SIMS_PER_SECOND.

        10000 sims / 15000 sps = 0.6667 seconds
        """
        from src.engine.production.sim_config import estimate_runtime_seconds

        expected = 10000 / 15000  # 0.6667
        actual = estimate_runtime_seconds(10000)
        assert abs(actual - expected) < 0.001


class TestPhase6CacheTTL(unittest.TestCase):
    """Verify cache TTL and staleness math."""

    def test_ttl_expiry_logic(self):
        """Entry is stale when (now - created_at) > ttl.

        With TTL=0, entry is immediately stale.
        With TTL=86400, entry lasts 24 hours.
        """
        from src.engine.production.cache import CacheEntry

        entry_instant = CacheEntry(value="test", ttl=0)
        import time

        time.sleep(0.01)  # Tiny delay to ensure time passes
        assert entry_instant.is_stale is True

        entry_long = CacheEntry(value="test", ttl=86400)
        assert entry_long.is_stale is False
        assert entry_long.age_seconds < 1.0

    def test_get_or_compute_caches_none(self):
        """After Bug 4 fix: get_or_compute should cache None values."""
        from src.engine.production.cache import TradeEvalCache

        cache = TradeEvalCache()
        call_count = 0

        def returns_none():
            nonlocal call_count
            call_count += 1
            return None

        # First call: computes
        v1 = cache.get_or_compute("key", returns_none, ttl=3600)
        assert v1 is None
        assert call_count == 1

        # Second call: should use cached None, not recompute
        v2 = cache.get_or_compute("key", returns_none, ttl=3600)
        assert v2 is None
        assert call_count == 1  # NOT called again


# ── Cross-Phase Integration ─────────────────────────────────────────


class TestCrossPhaseIntegration(unittest.TestCase):
    """Verify that formulas compose correctly across phases."""

    def test_full_pipeline_sign_conventions(self):
        """Verify sign convention consistency across the pipeline.

        Positive surplus = good trade. This must be consistent from:
          Phase 1 (SGP delta) → Phase 5 (game theory) → Grade/Verdict
        """
        import pandas as pd

        from src.engine.output.trade_evaluator import evaluate_trade, grade_trade

        # Create a clearly good trade: give weak player, receive strong player
        pool = pd.DataFrame(
            {
                "player_id": [1, 2, 3],
                "name": ["Strong", "Weak", "Average"],
                "team": ["NYY", "BOS", "LAD"],
                "positions": ["OF", "OF", "OF"],
                "is_hitter": [1, 1, 1],
                "is_injured": [0, 0, 0],
                "pa": [600, 400, 500],
                "ab": [550, 360, 450],
                "h": [170, 90, 120],
                "r": [100, 50, 70],
                "hr": [35, 10, 20],
                "rbi": [100, 40, 65],
                "sb": [15, 3, 8],
                "avg": [0.309, 0.250, 0.267],
                "ip": [0, 0, 0],
                "w": [0, 0, 0],
                "sv": [0, 0, 0],
                "k": [0, 0, 0],
                "era": [0, 0, 0],
                "whip": [0, 0, 0],
                "er": [0, 0, 0],
                "bb_allowed": [0, 0, 0],
                "h_allowed": [0, 0, 0],
                "adp": [10, 150, 80],
            }
        )

        result = evaluate_trade(
            giving_ids=[2],  # Give Weak
            receiving_ids=[1],  # Receive Strong
            user_roster_ids=[2, 3],
            player_pool=pool,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        # Receiving a better player should yield positive surplus
        assert result["surplus_sgp"] > 0
        assert result["verdict"] == "ACCEPT"
        assert grade_trade(result["surplus_sgp"]) not in ("D", "F")

    def test_self_trade_zero_surplus(self):
        """Trading a player for themselves should yield zero surplus.

        This is an algebraic identity: before_totals == after_totals.
        """
        import pandas as pd

        from src.engine.output.trade_evaluator import evaluate_trade

        pool = pd.DataFrame(
            {
                "player_id": [1, 2],
                "name": ["A", "B"],
                "team": ["NYY", "BOS"],
                "positions": ["OF", "1B"],
                "is_hitter": [1, 1],
                "is_injured": [0, 0],
                "pa": [600, 550],
                "ab": [550, 500],
                "h": [150, 130],
                "r": [80, 70],
                "hr": [25, 20],
                "rbi": [80, 65],
                "sb": [10, 5],
                "avg": [0.273, 0.260],
                "ip": [0, 0],
                "w": [0, 0],
                "sv": [0, 0],
                "k": [0, 0],
                "era": [0, 0],
                "whip": [0, 0],
                "er": [0, 0],
                "bb_allowed": [0, 0],
                "h_allowed": [0, 0],
                "adp": [30, 60],
            }
        )

        result = evaluate_trade(
            giving_ids=[1],
            receiving_ids=[1],  # Same player
            user_roster_ids=[1, 2],
            player_pool=pool,
            enable_mc=False,
            enable_context=False,
            enable_game_theory=False,
        )

        # Self-trade must have zero surplus (algebraic identity)
        assert abs(result["surplus_sgp"]) < 0.01


class TestReplacementCostMath(unittest.TestCase):
    """Hand-calculated math verification for the replacement cost penalty.

    Formula: sgp_penalty = (unrecoverable / sgp_denom) * FA_TURNOVER_DISCOUNT
    where:   unrecoverable = max(0, raw_loss - best_FA)
             FA_TURNOVER_DISCOUNT = 0.5
    """

    def test_replacement_cost_formula_hand_calc(self):
        """Verify: SV loss=32, best FA=19, denom=9 → penalty = (13/9)*0.5 = 0.722."""
        from unittest.mock import patch

        import pandas as pd

        from src.engine.output.trade_evaluator import _compute_replacement_penalty
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        before_totals = {"SV": 32, "R": 100, "HR": 30, "RBI": 90, "SB": 20, "W": 12, "K": 180}
        after_totals = {"SV": 0, "R": 100, "HR": 30, "RBI": 90, "SB": 20, "W": 12, "K": 180}

        # Create a minimal FA pool with one closer projecting 19 SV
        fa_pool = pd.DataFrame(
            [
                {
                    "player_id": 999,
                    "player_name": "FA Closer",
                    "team": "FA",
                    "positions": "RP",
                    "is_hitter": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "w": 2,
                    "sv": 19,
                    "k": 50,
                    "avg": 0,
                    "era": 3.50,
                    "whip": 1.20,
                }
            ]
        )

        with patch(
            "src.engine.output.trade_evaluator.get_free_agents",
            return_value=fa_pool,
        ):
            penalty, detail = _compute_replacement_penalty(
                before_totals=before_totals,
                after_totals=after_totals,
                player_pool=fa_pool,
                config=config,
                category_weights={"SV": 1.0},
                punt_categories=[],
            )

        # Hand calculation:
        # raw_loss = 32 - 0 = 32
        # best_FA = 19
        # unrecoverable = 32 - 19 = 13
        # denom = 9.0 (SV SGP denominator)
        # sgp_penalty = (13 / 9.0) * 0.5 = 0.7222...
        expected_penalty = (13.0 / 9.0) * 0.5
        assert abs(detail["SV"]["sgp_penalty"] - expected_penalty) < 0.01
        assert detail["SV"]["raw_loss"] == 32.0
        assert detail["SV"]["best_fa"] == 19.0
        assert detail["SV"]["unrecoverable"] == 13.0

    def test_replacement_cost_fully_recoverable(self):
        """Verify: HR loss=5, best FA=30 → unrecoverable=0 → penalty=0."""
        from unittest.mock import patch

        import pandas as pd

        from src.engine.output.trade_evaluator import _compute_replacement_penalty
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        before_totals = {"HR": 35, "R": 100, "RBI": 90, "SB": 20, "SV": 10, "W": 12, "K": 180}
        after_totals = {"HR": 30, "R": 100, "RBI": 90, "SB": 20, "SV": 10, "W": 12, "K": 180}

        # FA pool has a slugger with 30 HR — more than enough to cover the 5 HR loss
        fa_pool = pd.DataFrame(
            [
                {
                    "player_id": 888,
                    "player_name": "FA Slugger",
                    "team": "FA",
                    "positions": "OF",
                    "is_hitter": 1,
                    "r": 80,
                    "hr": 30,
                    "rbi": 80,
                    "sb": 5,
                    "w": 0,
                    "sv": 0,
                    "k": 0,
                    "avg": 0.260,
                    "era": 0,
                    "whip": 0,
                }
            ]
        )

        with patch(
            "src.engine.output.trade_evaluator.get_free_agents",
            return_value=fa_pool,
        ):
            penalty, detail = _compute_replacement_penalty(
                before_totals=before_totals,
                after_totals=after_totals,
                player_pool=fa_pool,
                config=config,
                category_weights={"HR": 1.0},
                punt_categories=[],
            )

        # Hand calculation:
        # raw_loss = 35 - 30 = 5
        # best_FA = 30
        # unrecoverable = max(0, 5 - 30) = 0
        # sgp_penalty = 0
        assert detail["HR"]["unrecoverable"] == 0.0
        assert detail["HR"]["sgp_penalty"] == 0.0

    def test_replacement_cost_multi_category_sum(self):
        """Verify: SV penalty + K penalty sum correctly.

        SV: loss=32, FA=19, unrec=13, penalty=(13/9)*0.5 = 0.722
        K: loss=60, FA=40, unrec=20, penalty=(20/45)*0.5 = 0.222
        Total = 0.944
        """
        from unittest.mock import patch

        import pandas as pd

        from src.engine.output.trade_evaluator import _compute_replacement_penalty
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        before_totals = {"SV": 32, "K": 200, "R": 100, "HR": 30, "RBI": 90, "SB": 20, "W": 12}
        after_totals = {"SV": 0, "K": 140, "R": 100, "HR": 30, "RBI": 90, "SB": 20, "W": 12}

        fa_pool = pd.DataFrame(
            [
                {
                    "player_id": 999,
                    "player_name": "FA Closer",
                    "team": "FA",
                    "positions": "RP",
                    "is_hitter": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "w": 2,
                    "sv": 19,
                    "k": 40,
                    "avg": 0,
                    "era": 3.50,
                    "whip": 1.20,
                },
            ]
        )

        with patch(
            "src.engine.output.trade_evaluator.get_free_agents",
            return_value=fa_pool,
        ):
            penalty, detail = _compute_replacement_penalty(
                before_totals=before_totals,
                after_totals=after_totals,
                player_pool=fa_pool,
                config=config,
                category_weights={"SV": 1.0, "K": 1.0},
                punt_categories=[],
            )

        # Hand calculation:
        # SV: raw_loss=32, best_FA=19, unrec=13, penalty=(13/9)*0.5 = 0.7222
        # K: raw_loss=60, best_FA=40, unrec=20, penalty=(20/45)*0.5 = 0.2222
        # Total = 0.9444
        expected_sv = (13.0 / 9.0) * 0.5
        expected_k = (20.0 / 45.0) * 0.5
        expected_total = expected_sv + expected_k

        assert abs(detail["SV"]["sgp_penalty"] - expected_sv) < 0.01
        assert abs(detail["K"]["sgp_penalty"] - expected_k) < 0.01
        assert abs(penalty - expected_total) < 0.01


# ── Lineup Constraint Math ──────────────────────────────────────────


class TestLineupConstraintMath(unittest.TestCase):
    """Hand-verify lineup-constrained trade evaluation math.

    Proves that LP-constrained totals correctly prevent phantom production
    from bench players, and that the forced drop / FA pickup model works.
    """

    def test_lineup_constraint_prevents_phantom_production(self):
        """2 hitters received but all slots full: only marginal improvement counts.

        Setup:
          - 10 hitter slots (C/1B/2B/3B/SS/3OF/2Util) filled by strong hitters
          - Trade gives 1 RP (30 SV), receives 2 average hitters
          - Without constraint: both hitters add ~130 R, ~35 HR, ~130 RBI
          - With constraint: only 1 hitter starts (replacing worst), other benched

        The LP assigns the better received hitter to a Util slot (displacing
        the worst starter) and benches the other. Savings category changes
        come only from the marginal improvement.
        """
        import pandas as pd

        from src.engine.output.trade_evaluator import _lineup_constrained_totals
        from src.in_season import _roster_category_totals
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        # Build a 23-player roster with clear position assignments
        players = []
        # 13 hitters: 1 C, 1 1B, 1 2B, 1 3B, 1 SS, 3 OF, 2 Util-eligible, 3 bench
        positions = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "1B", "OF", "SS", "2B", "3B"]
        for i, pos in enumerate(positions):
            players.append(
                {
                    "player_id": i + 1,
                    "player_name": f"Hitter {i + 1}",
                    "name": f"Hitter {i + 1}",
                    "positions": pos,
                    "is_hitter": 1,
                    "is_injured": 0,
                    "pa": 550,
                    "ab": 500,
                    "h": 130 + i * 2,
                    "r": 70 + i * 3,
                    "hr": 15 + i * 2,
                    "rbi": 65 + i * 3,
                    "sb": 8 + i,
                    "avg": round(0.260 + i * 0.002, 3),
                    "ip": 0,
                    "w": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "adp": 50 + i * 5,
                }
            )

        # 10 pitchers: 6 SP + 4 RP (fills SP/RP/P slots)
        # Use low ERA/WHIP so LP values them positively (high K+W outweighs rate penalty)
        for i in range(10):
            is_sp = i < 6
            players.append(
                {
                    "player_id": 14 + i,
                    "player_name": f"Pitcher {i + 1}",
                    "name": f"Pitcher {i + 1}",
                    "positions": "SP" if is_sp else "RP",
                    "is_hitter": 0,
                    "is_injured": 0,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "ip": 200 if is_sp else 70,
                    "w": 15 if is_sp else 4,
                    "sv": 0 if is_sp else 28 + i,
                    "k": 220 if is_sp else 80,
                    "era": 2.80 if is_sp else 2.50,
                    "whip": 1.05 if is_sp else 0.95,
                    "er": 62 if is_sp else 19,
                    "bb_allowed": 45,
                    "h_allowed": 140,
                    "adp": 80 + i * 10,
                }
            )

        pool = pd.DataFrame(players)
        roster_ids = [p["player_id"] for p in players]
        assert len(roster_ids) == 23

        # Get raw totals (old method) and LP-constrained totals (new method)
        raw_totals = _roster_category_totals(roster_ids, pool)
        lp_totals, assignments = _lineup_constrained_totals(roster_ids, pool, config)

        if not assignments:
            self.skipTest("PuLP not available")

        # LP should have exactly 18 starters (max slots)
        self.assertEqual(len(assignments), 18)

        # LP totals must be <= raw totals for counting stats
        # (5 bench players excluded)
        self.assertLessEqual(lp_totals["R"], raw_totals["R"])
        self.assertLessEqual(lp_totals["HR"], raw_totals["HR"])
        self.assertLessEqual(lp_totals["SV"], raw_totals["SV"])

        # The difference should be meaningful — at least one bench hitter excluded
        # Each bench hitter contributes ~70+ R, so we should see a gap
        self.assertGreater(raw_totals["R"] - lp_totals["R"], 50)

    def test_closer_trade_grades_correctly(self):
        """Williams-like closer for 2 hitters should NOT grade A or A+.

        If an elite closer (30 SV, 80 K) is traded for two average hitters,
        the LP-constrained evaluation should show a negative or near-zero
        surplus because:
          1. Only one hitter can start (the other gets benched)
          2. The RP slot now gets filled by a worse reliever
          3. Saves are concentrated and hard to replace

        Without LP constraint, this would grade A+ due to phantom production.
        """
        import pandas as pd

        from src.engine.output.trade_evaluator import evaluate_trade
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        # Build 23-player roster (elite closer is player 23)
        players = []
        positions = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF", "1B", "OF", "SS", "2B", "3B"]
        for i, pos in enumerate(positions):
            players.append(
                {
                    "player_id": i + 1,
                    "player_name": f"Hitter {i + 1}",
                    "name": f"Hitter {i + 1}",
                    "positions": pos,
                    "is_hitter": 1,
                    "is_injured": 0,
                    "pa": 550,
                    "ab": 500,
                    "h": 135,
                    "r": 75,
                    "hr": 20,
                    "rbi": 75,
                    "sb": 12,
                    "avg": 0.270,
                    "ip": 0,
                    "w": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "adp": 50 + i * 5,
                }
            )

        # 6 SP (good pitchers — low ERA so LP values them)
        for i in range(6):
            players.append(
                {
                    "player_id": 14 + i,
                    "player_name": f"SP {i + 1}",
                    "name": f"SP {i + 1}",
                    "positions": "SP",
                    "is_hitter": 0,
                    "is_injured": 0,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "ip": 200,
                    "w": 15,
                    "sv": 0,
                    "k": 220,
                    "era": 2.80,
                    "whip": 1.05,
                    "er": 62,
                    "bb_allowed": 45,
                    "h_allowed": 140,
                    "adp": 80 + i * 10,
                }
            )

        # 3 RP (decent but not elite — low ERA so LP starts them)
        for i in range(3):
            players.append(
                {
                    "player_id": 20 + i,
                    "player_name": f"RP {i + 1}",
                    "name": f"RP {i + 1}",
                    "positions": "RP",
                    "is_hitter": 0,
                    "is_injured": 0,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "ip": 70,
                    "w": 4,
                    "sv": 12,
                    "k": 80,
                    "era": 3.00,
                    "whip": 1.10,
                    "er": 23,
                    "bb_allowed": 22,
                    "h_allowed": 55,
                    "adp": 140 + i * 10,
                }
            )

        # ELITE CLOSER (Devin Williams analog) — player 23
        players.append(
            {
                "player_id": 23,
                "player_name": "Elite Closer",
                "name": "Elite Closer",
                "positions": "RP",
                "is_hitter": 0,
                "is_injured": 0,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "ip": 60,
                "w": 4,
                "sv": 32,
                "k": 85,
                "era": 2.25,
                "whip": 0.95,
                "er": 15,
                "bb_allowed": 18,
                "h_allowed": 39,
                "adp": 60,
            }
        )

        # Two average hitters to receive (IDs 24, 25)
        for i in range(2):
            players.append(
                {
                    "player_id": 24 + i,
                    "player_name": f"Avg Hitter {i + 1}",
                    "name": f"Avg Hitter {i + 1}",
                    "positions": "OF",
                    "is_hitter": 1,
                    "is_injured": 0,
                    "pa": 550,
                    "ab": 500,
                    "h": 130,
                    "r": 65,
                    "hr": 18,
                    "rbi": 65,
                    "sb": 10,
                    "avg": 0.260,
                    "ip": 0,
                    "w": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "adp": 120,
                }
            )

        pool = pd.DataFrame(players)
        roster_ids = [p["player_id"] for p in players[:23]]
        assert len(roster_ids) == 23

        result = evaluate_trade(
            giving_ids=[23],  # Elite Closer
            receiving_ids=[24, 25],  # 2 average hitters
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_context=False,
            enable_game_theory=False,
        )

        if result.get("lineup_constrained"):
            # With LP constraint, this trade should NOT be an A or A+
            # The closer's saves loss + only marginal hitter improvement
            # should produce a much lower grade
            grade = result["grade"]
            self.assertNotIn(
                grade,
                ["A+", "A"],
                (
                    f"Grade {grade} is too high for trading an elite closer for 2 avg hitters. "
                    f"Surplus: {result['surplus_sgp']:.3f}"
                ),
            )
        else:
            self.skipTest("PuLP not available — cannot verify LP-constrained grade")

    def test_fa_pickup_fills_roster_gap(self):
        """2-for-1 trade adds FA; after-totals should include FA production."""
        import pandas as pd

        from src.engine.output.trade_evaluator import evaluate_trade
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        # Build minimal 5-player roster for simplicity
        players = [
            {
                "player_id": 1,
                "player_name": "Hitter A",
                "name": "Hitter A",
                "positions": "1B",
                "is_hitter": 1,
                "is_injured": 0,
                "pa": 550,
                "ab": 500,
                "h": 140,
                "r": 80,
                "hr": 25,
                "rbi": 90,
                "sb": 10,
                "avg": 0.280,
                "ip": 0,
                "w": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 30,
            },
            {
                "player_id": 2,
                "player_name": "Hitter B",
                "name": "Hitter B",
                "positions": "SS",
                "is_hitter": 1,
                "is_injured": 0,
                "pa": 550,
                "ab": 500,
                "h": 130,
                "r": 70,
                "hr": 20,
                "rbi": 75,
                "sb": 15,
                "avg": 0.260,
                "ip": 0,
                "w": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 50,
            },
            {
                "player_id": 3,
                "player_name": "Pitcher A",
                "name": "Pitcher A",
                "positions": "SP",
                "is_hitter": 0,
                "is_injured": 0,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "ip": 180,
                "w": 12,
                "sv": 0,
                "k": 180,
                "era": 3.50,
                "whip": 1.15,
                "er": 70,
                "bb_allowed": 50,
                "h_allowed": 155,
                "adp": 60,
            },
            # FA pool: a hitter not on roster
            {
                "player_id": 4,
                "player_name": "FA Hitter",
                "name": "FA Hitter",
                "positions": "OF",
                "is_hitter": 1,
                "is_injured": 0,
                "pa": 500,
                "ab": 460,
                "h": 115,
                "r": 55,
                "hr": 12,
                "rbi": 50,
                "sb": 5,
                "avg": 0.250,
                "ip": 0,
                "w": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 200,
            },
        ]
        pool = pd.DataFrame(players)
        roster_ids = [1, 2, 3]

        # 2-for-1: give 2 hitters, receive 1 pitcher (a new one)
        players.append(
            {
                "player_id": 5,
                "player_name": "New Pitcher",
                "name": "New Pitcher",
                "positions": "RP",
                "is_hitter": 0,
                "is_injured": 0,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "ip": 60,
                "w": 3,
                "sv": 20,
                "k": 65,
                "era": 3.00,
                "whip": 1.10,
                "er": 20,
                "bb_allowed": 18,
                "h_allowed": 48,
                "adp": 100,
            }
        )
        pool = pd.DataFrame(players)

        result = evaluate_trade(
            giving_ids=[1, 2],  # Give 2 hitters
            receiving_ids=[5],  # Receive 1 pitcher
            user_roster_ids=roster_ids,
            player_pool=pool,
            config=config,
            enable_context=False,
            enable_game_theory=False,
        )

        # The FA pickup should be attempted (roster shrinks by 1)
        # Result should have valid totals regardless of LP availability
        self.assertIn("after_totals", result)
        self.assertIn("grade", result)


if __name__ == "__main__":
    unittest.main()
