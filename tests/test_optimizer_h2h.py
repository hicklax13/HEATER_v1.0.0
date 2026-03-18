"""Tests for src/optimizer/h2h_engine.py.

Covers H2H category weights, win probability estimation,
default variances, and edge cases.
"""

from __future__ import annotations

import math

import pytest
from scipy.stats import norm

from src.optimizer.h2h_engine import (
    ALL_CATEGORIES,
    INVERSE_CATS,
    compute_h2h_category_weights,
    default_category_variances,
    estimate_h2h_win_probability,
)

# ── Helper: build tied totals ────────────────────────────────────────


def _tied_totals() -> dict[str, float]:
    """Return category totals for a 'perfectly average' team."""
    return {
        "r": 80.0,
        "hr": 20.0,
        "rbi": 75.0,
        "sb": 10.0,
        "avg": 0.265,
        "w": 8.0,
        "sv": 5.0,
        "k": 70.0,
        "era": 3.80,
        "whip": 1.20,
    }


def _dominant_totals() -> dict[str, float]:
    """Return totals that dominate in every category."""
    return {
        "r": 120.0,
        "hr": 40.0,
        "rbi": 115.0,
        "sb": 25.0,
        "avg": 0.310,
        "w": 14.0,
        "sv": 12.0,
        "k": 110.0,
        "era": 2.50,
        "whip": 0.95,
    }


def _weak_totals() -> dict[str, float]:
    """Return totals that lose in every category."""
    return {
        "r": 50.0,
        "hr": 8.0,
        "rbi": 45.0,
        "sb": 3.0,
        "avg": 0.230,
        "w": 3.0,
        "sv": 1.0,
        "k": 40.0,
        "era": 5.20,
        "whip": 1.50,
    }


# ── Category Weights Tests ───────────────────────────────────────────


class TestComputeH2hCategoryWeights:
    def test_tied_categories_maximum_weight(self) -> None:
        """When gap=0 with equal variances, all weights should be equal (1.0)."""
        totals = _tied_totals()
        # Use equal variance for all categories so the raw phi(0)/sigma
        # is the same for every category, and normalization yields 1.0.
        equal_var = {cat: 100.0 for cat in ALL_CATEGORIES}
        weights = compute_h2h_category_weights(totals, totals, category_variances=equal_var)
        for cat, w in weights.items():
            assert w == pytest.approx(1.0, abs=0.01), f"{cat} weight should be ~1.0 when tied with equal variance"

    def test_large_lead_low_weight(self) -> None:
        """Categories where we lead by a lot should have low weight."""
        my = _tied_totals()
        opp = _tied_totals()
        # Give ourselves a massive HR lead
        my["hr"] = 40.0
        opp["hr"] = 10.0
        weights = compute_h2h_category_weights(my, opp)
        # HR weight should be well below the mean (1.0) because
        # we're already winning that category decisively
        assert weights["hr"] < 0.5

    def test_large_deficit_low_weight(self) -> None:
        """Categories where we trail by a lot should also have low weight."""
        my = _tied_totals()
        opp = _tied_totals()
        # Give opponent a massive SB lead
        my["sb"] = 3.0
        opp["sb"] = 25.0
        weights = compute_h2h_category_weights(my, opp)
        # SB weight should be low because we can't realistically win it
        assert weights["sb"] < 0.5

    def test_inverse_cat_era_flip(self) -> None:
        """ERA gap properly flipped (lower is better)."""
        my = _tied_totals()
        opp = _tied_totals()
        # I have better (lower) ERA
        my["era"] = 2.50
        opp["era"] = 4.50
        weights = compute_h2h_category_weights(my, opp)
        # ERA should have LOW weight since we're winning easily
        # (large positive gap after flip)
        assert weights["era"] < 1.0

    def test_inverse_cat_whip_flip(self) -> None:
        """WHIP gap properly flipped (lower is better)."""
        my = _tied_totals()
        opp = _tied_totals()
        # I have better (lower) WHIP
        my["whip"] = 0.95
        opp["whip"] = 1.45
        weights = compute_h2h_category_weights(my, opp)
        # WHIP should be lower weight since we're well ahead
        assert weights["whip"] < 1.0

    def test_weights_normalized(self) -> None:
        """Mean of all weights should be approximately 1.0."""
        my = _tied_totals()
        opp = _tied_totals()
        my["hr"] = 30.0  # Slightly different to create variation
        opp["sb"] = 15.0
        weights = compute_h2h_category_weights(my, opp)
        mean_weight = sum(weights.values()) / len(weights)
        assert mean_weight == pytest.approx(1.0, abs=0.01)

    def test_empty_totals(self) -> None:
        """Empty totals return empty weights."""
        weights = compute_h2h_category_weights({}, {})
        assert weights == {}


# ── Win Probability Tests ────────────────────────────────────────────


class TestEstimateH2hWinProbability:
    def test_win_prob_dominant(self) -> None:
        """P(win) near 1.0 when far ahead in all categories."""
        result = estimate_h2h_win_probability(_dominant_totals(), _weak_totals())
        per_cat = result["per_category"]
        # Each category should be strongly favored
        for cat, p in per_cat.items():
            assert p > 0.8, f"P(win {cat}) should be >0.8 when dominant"
        assert result["overall_win_prob"] > 0.95

    def test_win_prob_behind(self) -> None:
        """P(win) near 0 when far behind in all categories."""
        result = estimate_h2h_win_probability(_weak_totals(), _dominant_totals())
        per_cat = result["per_category"]
        for cat, p in per_cat.items():
            assert p < 0.2, f"P(win {cat}) should be <0.2 when behind"
        assert result["overall_win_prob"] < 0.05

    def test_win_prob_balanced(self) -> None:
        """50/50 when tied in all categories."""
        totals = _tied_totals()
        result = estimate_h2h_win_probability(totals, totals)
        per_cat = result["per_category"]
        for cat, p in per_cat.items():
            assert p == pytest.approx(0.5, abs=0.01), f"P(win {cat}) should be 0.5 when tied"
        assert result["expected_wins"] == pytest.approx(5.0, abs=0.1)
        assert result["overall_win_prob"] == pytest.approx(0.5, abs=0.05)

    def test_expected_wins_sum(self) -> None:
        """Expected wins = sum of individual P(win) values."""
        my = _tied_totals()
        opp = _tied_totals()
        my["hr"] = 30.0
        my["era"] = 2.80
        result = estimate_h2h_win_probability(my, opp)
        per_cat = result["per_category"]
        expected_sum = sum(per_cat.values())
        assert result["expected_wins"] == pytest.approx(expected_sum, abs=1e-10)

    def test_inverse_cat_era_win_prob(self) -> None:
        """Lower ERA -> higher P(win) for ERA category."""
        my = _tied_totals()
        opp = _tied_totals()
        my["era"] = 2.50  # Better (lower)
        opp["era"] = 4.50  # Worse (higher)
        result = estimate_h2h_win_probability(my, opp)
        assert result["per_category"]["era"] > 0.5

    def test_inverse_cat_whip_win_prob(self) -> None:
        """Lower WHIP -> higher P(win) for WHIP category."""
        my = _tied_totals()
        opp = _tied_totals()
        my["whip"] = 0.95
        opp["whip"] = 1.45
        result = estimate_h2h_win_probability(my, opp)
        assert result["per_category"]["whip"] > 0.5


# ── Default Variances Tests ──────────────────────────────────────────


class TestDefaultCategoryVariances:
    def test_default_variances_positive(self) -> None:
        """All default variances are positive."""
        variances = default_category_variances()
        for cat, v in variances.items():
            assert v > 0, f"Variance for {cat} must be positive"

    def test_all_categories_present(self) -> None:
        """Default variances cover all 12 H2H categories."""
        variances = default_category_variances()
        for cat in ALL_CATEGORIES:
            assert cat in variances, f"Missing default variance for {cat}"


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_zero_variance_handled(self) -> None:
        """sigma=0 doesn't cause division by zero."""
        # Set all variances to 0
        zero_var = {cat: 0.0 for cat in ALL_CATEGORIES}
        my = _tied_totals()
        opp = _tied_totals()
        # Should not raise
        weights = compute_h2h_category_weights(my, opp, category_variances=zero_var)
        assert isinstance(weights, dict)

        result = estimate_h2h_win_probability(my, opp, category_variances=zero_var)
        assert "per_category" in result

    def test_partial_categories(self) -> None:
        """Works when only some categories are present."""
        my = {"r": 80.0, "hr": 20.0}
        opp = {"r": 75.0, "hr": 25.0}
        weights = compute_h2h_category_weights(my, opp)
        assert "r" in weights
        assert "hr" in weights
        assert len(weights) == 2

    def test_win_prob_returns_all_keys(self) -> None:
        """Result dict always has per_category, expected_wins, overall_win_prob."""
        totals = _tied_totals()
        result = estimate_h2h_win_probability(totals, totals)
        assert "per_category" in result
        assert "expected_wins" in result
        assert "overall_win_prob" in result
