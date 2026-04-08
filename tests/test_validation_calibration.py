"""Tests for ROADMAP B6, B8, C2 — validation and calibration constants.

B6: Category urgency k-calibration constants
B8: Trade Finder composite weight constants
C2: Dual objective alpha with playoff probability
"""

from __future__ import annotations

# ── B6: Category Urgency k-Calibration ──────────────────────────────────


class TestCategoryUrgencyConstants:
    """B6: Verify sigmoid steepness constants are importable and valid."""

    def test_constants_importable(self):
        from src.optimizer.category_urgency import COUNTING_STAT_K, RATE_STAT_K

        assert isinstance(COUNTING_STAT_K, float)
        assert isinstance(RATE_STAT_K, float)

    def test_k_values_positive(self):
        from src.optimizer.category_urgency import COUNTING_STAT_K, RATE_STAT_K

        assert COUNTING_STAT_K > 0.0
        assert RATE_STAT_K > 0.0

    def test_rate_stat_k_higher_than_counting(self):
        """Rate stats should be steeper (noisier, need sharper sigmoid)."""
        from src.optimizer.category_urgency import COUNTING_STAT_K, RATE_STAT_K

        assert RATE_STAT_K > COUNTING_STAT_K

    def test_constants_used_in_urgency_computation(self):
        """Verify the constants actually affect urgency output."""
        from src.optimizer.category_urgency import (
            COUNTING_STAT_K,
            RATE_STAT_K,
            compute_category_urgency,
        )

        my_totals = {"HR": 10, "ERA": 3.50}
        opp_totals = {"HR": 15, "ERA": 4.00}

        result = compute_category_urgency(my_totals, opp_totals)

        # HR is counting stat, losing by 5 HR -> high urgency
        assert result["HR"] > 0.5
        # ERA is rate/inverse stat, winning (lower ERA) -> low urgency
        assert result["ERA"] < 0.5

    def test_sigmoid_at_zero_gap_equals_half(self):
        """When tied, urgency should be exactly 0.5 regardless of k."""
        from src.optimizer.category_urgency import compute_category_urgency

        my_totals = {"HR": 10, "ERA": 3.50}
        opp_totals = {"HR": 10, "ERA": 3.50}

        result = compute_category_urgency(my_totals, opp_totals)
        assert abs(result["HR"] - 0.5) < 1e-6
        assert abs(result["ERA"] - 0.5) < 1e-6


# ── B8: Trade Finder Composite Weights ──────────────────────────────────


class TestCompositeWeightConstants:
    """B8: Verify composite weight constants are importable and sum to 1.0."""

    def test_constants_importable(self):
        from src.trade_finder import (
            COMPOSITE_W_ACCEPT,
            COMPOSITE_W_ADP,
            COMPOSITE_W_CAT_FIT,
            COMPOSITE_W_ECR,
            COMPOSITE_W_OPP_NEED,
            COMPOSITE_W_SGP,
        )

        for w in [
            COMPOSITE_W_SGP,
            COMPOSITE_W_ADP,
            COMPOSITE_W_ECR,
            COMPOSITE_W_ACCEPT,
            COMPOSITE_W_CAT_FIT,
            COMPOSITE_W_OPP_NEED,
        ]:
            assert isinstance(w, float)

    def test_weights_sum_to_one(self):
        from src.trade_finder import (
            COMPOSITE_W_ACCEPT,
            COMPOSITE_W_ADP,
            COMPOSITE_W_CAT_FIT,
            COMPOSITE_W_ECR,
            COMPOSITE_W_OPP_NEED,
            COMPOSITE_W_SGP,
        )

        total = (
            COMPOSITE_W_SGP
            + COMPOSITE_W_ADP
            + COMPOSITE_W_ECR
            + COMPOSITE_W_ACCEPT
            + COMPOSITE_W_CAT_FIT
            + COMPOSITE_W_OPP_NEED
        )
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_all_weights_positive(self):
        from src.trade_finder import (
            COMPOSITE_W_ACCEPT,
            COMPOSITE_W_ADP,
            COMPOSITE_W_CAT_FIT,
            COMPOSITE_W_ECR,
            COMPOSITE_W_OPP_NEED,
            COMPOSITE_W_SGP,
        )

        for name, w in [
            ("SGP", COMPOSITE_W_SGP),
            ("ADP", COMPOSITE_W_ADP),
            ("ECR", COMPOSITE_W_ECR),
            ("ACCEPT", COMPOSITE_W_ACCEPT),
            ("CAT_FIT", COMPOSITE_W_CAT_FIT),
            ("OPP_NEED", COMPOSITE_W_OPP_NEED),
        ]:
            assert w > 0.0, f"COMPOSITE_W_{name} must be positive, got {w}"

    def test_acceptance_is_highest_weight(self):
        """Acceptance should be the highest weight (rejected trades are worthless)."""
        from src.trade_finder import (
            COMPOSITE_W_ACCEPT,
            COMPOSITE_W_ADP,
            COMPOSITE_W_CAT_FIT,
            COMPOSITE_W_ECR,
            COMPOSITE_W_OPP_NEED,
            COMPOSITE_W_SGP,
        )

        others = [
            COMPOSITE_W_SGP,
            COMPOSITE_W_ADP,
            COMPOSITE_W_ECR,
            COMPOSITE_W_CAT_FIT,
            COMPOSITE_W_OPP_NEED,
        ]
        assert all(COMPOSITE_W_ACCEPT > w for w in others)


# ── C2: Dual Objective Alpha with Playoff Probability ───────────────────


class TestPlayoffProbabilityAlpha:
    """C2: Verify playoff-probability-aware alpha recommendation."""

    def test_high_playoff_odds_gives_high_alpha(self):
        from src.optimizer.dual_objective import recommend_alpha

        alpha = recommend_alpha(weeks_remaining=12, playoff_probability=0.90)
        assert abs(alpha - 0.80) < 0.01, f"Expected ~0.80, got {alpha}"

    def test_mid_playoff_odds_gives_balanced_alpha(self):
        from src.optimizer.dual_objective import recommend_alpha

        alpha = recommend_alpha(weeks_remaining=12, playoff_probability=0.55)
        assert abs(alpha - 0.60) < 0.01, f"Expected ~0.60, got {alpha}"

    def test_low_playoff_odds_gives_low_alpha(self):
        from src.optimizer.dual_objective import recommend_alpha

        alpha = recommend_alpha(weeks_remaining=12, playoff_probability=0.20)
        assert abs(alpha - 0.35) < 0.01, f"Expected ~0.35, got {alpha}"

    def test_fallback_to_time_based_when_none(self):
        """When playoff_probability is None, use legacy time-based path."""
        from src.optimizer.dual_objective import recommend_alpha

        # Mid-season, no playoff prob -> should use time-based (0.65)
        alpha = recommend_alpha(weeks_remaining=12, playoff_probability=None)
        assert abs(alpha - 0.65) < 0.01, f"Expected ~0.65, got {alpha}"

    def test_playoff_prob_boundary_85(self):
        """Boundary: exactly 0.85 should be in the middle tier."""
        from src.optimizer.dual_objective import recommend_alpha

        alpha = recommend_alpha(weeks_remaining=12, playoff_probability=0.85)
        assert abs(alpha - 0.60) < 0.01, f"At 0.85 boundary, expected ~0.60, got {alpha}"

    def test_playoff_prob_boundary_40(self):
        """Boundary: exactly 0.40 should be in the middle tier."""
        from src.optimizer.dual_objective import recommend_alpha

        alpha = recommend_alpha(weeks_remaining=12, playoff_probability=0.40)
        assert abs(alpha - 0.60) < 0.01, f"At 0.40 boundary, expected ~0.60, got {alpha}"

    def test_playoff_prob_clamped_above_one(self):
        """Values > 1.0 should be clamped to 1.0 (contender tier)."""
        from src.optimizer.dual_objective import recommend_alpha

        alpha = recommend_alpha(weeks_remaining=12, playoff_probability=1.5)
        assert abs(alpha - 0.80) < 0.01

    def test_playoff_prob_clamped_below_zero(self):
        """Values < 0.0 should be clamped to 0.0 (long shot tier)."""
        from src.optimizer.dual_objective import recommend_alpha

        alpha = recommend_alpha(weeks_remaining=12, playoff_probability=-0.5)
        assert abs(alpha - 0.35) < 0.01

    def test_playoff_prob_with_desperation(self):
        """Desperation boost still applies on top of playoff-prob alpha."""
        from src.optimizer.dual_objective import recommend_alpha

        base = recommend_alpha(weeks_remaining=12, playoff_probability=0.50)
        boosted = recommend_alpha(weeks_remaining=12, playoff_probability=0.50, desperation_level=0.5)
        assert boosted > base

    def test_legacy_path_unchanged(self):
        """Existing behavior without playoff_probability should not change."""
        from src.optimizer.dual_objective import recommend_alpha

        # Early season
        assert recommend_alpha(weeks_remaining=20) == 0.55
        # Mid season
        assert recommend_alpha(weeks_remaining=12) == 0.65
        # Late season
        assert recommend_alpha(weeks_remaining=5) == 0.85
        # Playoffs
        assert recommend_alpha(weeks_remaining=2) == 0.85
