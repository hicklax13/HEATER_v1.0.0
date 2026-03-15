"""Tests for dual H2H/roto objective blending.

Covers:
  - Pure roto (alpha=0), pure H2H (alpha=1), balanced (alpha=0.5)
  - Normalization of blended weights
  - Alpha recommendation based on weeks remaining, rank, and record
  - Clamping alpha to [0.0, 1.0]
"""

from __future__ import annotations

import pytest

from src.optimizer.dual_objective import (
    blend_h2h_roto_weights,
    recommend_alpha,
)

# ── Blend Weights Tests ──────────────────────────────────────────────


class TestBlendH2hRotoWeights:
    """Tests for blend_h2h_roto_weights()."""

    def _make_weights(self, h2h_factor: float = 1.0, roto_factor: float = 1.0):
        """Helper: create H2H and roto weight dicts with known values."""
        h2h = {"r": 2.0 * h2h_factor, "hr": 0.5 * h2h_factor, "era": 1.5 * h2h_factor}
        roto = {"r": 0.5 * roto_factor, "hr": 2.0 * roto_factor, "era": 0.5 * roto_factor}
        return h2h, roto

    def test_alpha_zero_returns_pure_roto(self):
        """alpha=0 should return pure roto weights (after normalization)."""
        h2h, roto = self._make_weights()

        result = blend_h2h_roto_weights(h2h, roto, alpha=0.0)

        # With alpha=0, blended = 0*h2h + 1*roto = roto
        # After normalization, relative proportions should match roto
        # roto: r=0.5, hr=2.0, era=0.5 -> mean=1.0 -> already normalized
        assert result["r"] == pytest.approx(result["era"], abs=1e-6)
        assert result["hr"] > result["r"]  # HR had the highest roto weight

    def test_alpha_one_returns_pure_h2h(self):
        """alpha=1 should return pure H2H weights (after normalization)."""
        h2h, roto = self._make_weights()

        result = blend_h2h_roto_weights(h2h, roto, alpha=1.0)

        # With alpha=1, blended = 1*h2h + 0*roto = h2h
        # h2h: r=2.0, hr=0.5, era=1.5
        assert result["r"] > result["hr"]  # R had the highest H2H weight
        assert result["r"] > result["era"]

    def test_alpha_half_returns_average(self):
        """alpha=0.5 should produce the arithmetic mean of both weights."""
        h2h = {"r": 2.0, "hr": 0.0}
        roto = {"r": 0.0, "hr": 2.0}

        result = blend_h2h_roto_weights(h2h, roto, alpha=0.5)

        # Raw: r = 0.5*2 + 0.5*0 = 1.0, hr = 0.5*0 + 0.5*2 = 1.0
        # After normalization: both should be 1.0
        assert result["r"] == pytest.approx(1.0, abs=1e-6)
        assert result["hr"] == pytest.approx(1.0, abs=1e-6)

    def test_blended_weights_normalized_mean_one(self):
        """Blended weights should have mean approximately 1.0."""
        h2h, roto = self._make_weights(h2h_factor=3.0, roto_factor=0.5)

        result = blend_h2h_roto_weights(h2h, roto, alpha=0.6)

        mean_w = sum(result.values()) / len(result)
        assert abs(mean_w - 1.0) < 1e-6

    def test_missing_category_in_one_source(self):
        """Categories missing from one source should use the other."""
        h2h = {"r": 1.5, "hr": 0.8}
        roto = {"r": 0.5, "sb": 2.0}  # HR missing from roto, SB missing from H2H

        result = blend_h2h_roto_weights(h2h, roto, alpha=0.5)

        # All three categories should be present
        assert "r" in result
        assert "hr" in result  # from H2H only
        assert "sb" in result  # from roto only

    def test_empty_inputs_returns_empty(self):
        """Empty weight dicts should produce empty result."""
        result = blend_h2h_roto_weights({}, {}, alpha=0.5)
        assert result == {}

    def test_alpha_clamped_below_zero(self):
        """Alpha below 0.0 should be treated as 0.0."""
        h2h = {"r": 2.0, "hr": 0.5}
        roto = {"r": 0.5, "hr": 2.0}

        result = blend_h2h_roto_weights(h2h, roto, alpha=-0.5)

        # Should behave like alpha=0 (pure roto)
        assert result["hr"] > result["r"]

    def test_alpha_clamped_above_one(self):
        """Alpha above 1.0 should be treated as 1.0."""
        h2h = {"r": 2.0, "hr": 0.5}
        roto = {"r": 0.5, "hr": 2.0}

        result = blend_h2h_roto_weights(h2h, roto, alpha=1.5)

        # Should behave like alpha=1 (pure H2H)
        assert result["r"] > result["hr"]


# ── Recommend Alpha Tests ────────────────────────────────────────────


class TestRecommendAlpha:
    """Tests for recommend_alpha()."""

    def test_early_season_low_alpha(self):
        """Early season (>12 weeks) should produce low alpha (roto lean)."""
        alpha = recommend_alpha(weeks_remaining=20)
        assert alpha == pytest.approx(0.3, abs=0.01)

    def test_mid_season_balanced(self):
        """Mid season (6-12 weeks) should produce balanced alpha."""
        alpha = recommend_alpha(weeks_remaining=10)
        assert alpha == pytest.approx(0.5, abs=0.01)

    def test_late_season_high_alpha(self):
        """Late season (<6 weeks) should lean H2H."""
        alpha = recommend_alpha(weeks_remaining=4)
        assert alpha == pytest.approx(0.7, abs=0.01)

    def test_playoff_weeks_very_high_alpha(self):
        """Playoff weeks (<3 weeks) should be nearly pure H2H."""
        alpha = recommend_alpha(weeks_remaining=2)
        assert alpha == pytest.approx(0.85, abs=0.01)

    def test_bad_roto_rank_increases_alpha(self):
        """Bottom-3 roto rank should increase alpha by 0.1."""
        # Without roto rank
        base_alpha = recommend_alpha(weeks_remaining=10)
        # With bad roto rank (12th out of 12)
        adj_alpha = recommend_alpha(weeks_remaining=10, roto_rank=12, num_teams=12)
        assert adj_alpha == pytest.approx(base_alpha + 0.1, abs=0.01)

    def test_good_roto_rank_no_adjustment(self):
        """Good roto rank (top half) should not adjust alpha."""
        base_alpha = recommend_alpha(weeks_remaining=10)
        adj_alpha = recommend_alpha(weeks_remaining=10, roto_rank=3, num_teams=12)
        assert adj_alpha == pytest.approx(base_alpha, abs=0.01)

    def test_bad_h2h_record_decreases_alpha(self):
        """Below .400 H2H record should decrease alpha by 0.1."""
        base_alpha = recommend_alpha(weeks_remaining=10)
        adj_alpha = recommend_alpha(weeks_remaining=10, h2h_record_wins=3, h2h_record_losses=10)
        # win_rate = 3/13 = 0.231 < 0.4
        assert adj_alpha == pytest.approx(base_alpha - 0.1, abs=0.01)

    def test_good_h2h_record_no_adjustment(self):
        """Good H2H record should not adjust alpha."""
        base_alpha = recommend_alpha(weeks_remaining=10)
        adj_alpha = recommend_alpha(weeks_remaining=10, h2h_record_wins=8, h2h_record_losses=5)
        # win_rate = 8/13 = 0.615 > 0.4
        assert adj_alpha == pytest.approx(base_alpha, abs=0.01)

    def test_result_always_in_zero_one(self):
        """Alpha should always be clamped to [0.0, 1.0]."""
        # Try to push alpha very high: playoff weeks + bad roto
        alpha_high = recommend_alpha(weeks_remaining=1, roto_rank=12, num_teams=12)
        assert 0.0 <= alpha_high <= 1.0

        # Try to push alpha very low: early season + bad H2H
        alpha_low = recommend_alpha(weeks_remaining=25, h2h_record_wins=0, h2h_record_losses=20)
        assert 0.0 <= alpha_low <= 1.0

    def test_both_adjustments_combine(self):
        """Bad roto + bad H2H adjustments can cancel each other out."""
        # Bad roto (+0.1) and bad H2H (-0.1) should roughly cancel
        base_alpha = recommend_alpha(weeks_remaining=10)
        adj_alpha = recommend_alpha(
            weeks_remaining=10,
            roto_rank=12,
            num_teams=12,
            h2h_record_wins=2,
            h2h_record_losses=10,
        )
        # +0.1 (roto) - 0.1 (h2h) = 0.0 net adjustment
        assert adj_alpha == pytest.approx(base_alpha, abs=0.01)

    def test_edge_12_weeks_is_mid_season(self):
        """Exactly 12 weeks remaining should be mid-season (alpha=0.5)."""
        alpha = recommend_alpha(weeks_remaining=12)
        assert alpha == pytest.approx(0.5, abs=0.01)

    def test_edge_6_weeks_is_mid_season(self):
        """Exactly 6 weeks remaining should be mid-season (alpha=0.5)."""
        alpha = recommend_alpha(weeks_remaining=6)
        assert alpha == pytest.approx(0.5, abs=0.01)

    def test_edge_3_weeks_is_late_season(self):
        """Exactly 3 weeks remaining should be late season (alpha=0.7)."""
        alpha = recommend_alpha(weeks_remaining=3)
        assert alpha == pytest.approx(0.7, abs=0.01)
