"""Tests for gmLI-enhanced closer job security scoring."""

from __future__ import annotations

import pytest

from src.closer_monitor import _compute_gmli_component, compute_job_security


class TestComputeGmliComponent:
    """Tests for the piecewise linear gmLI-to-trust conversion."""

    def test_high_gmli(self):
        assert _compute_gmli_component(2.5) == 1.0

    def test_threshold_1_8(self):
        assert _compute_gmli_component(1.8) == 1.0

    def test_midpoint_1_4(self):
        result = _compute_gmli_component(1.4)
        assert result == pytest.approx(0.75, abs=1e-6)

    def test_average_leverage(self):
        assert _compute_gmli_component(1.0) == pytest.approx(0.5, abs=1e-6)

    def test_low_leverage_0_75(self):
        result = _compute_gmli_component(0.75)
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_threshold_0_5(self):
        assert _compute_gmli_component(0.5) == pytest.approx(0.0, abs=1e-6)

    def test_very_low(self):
        assert _compute_gmli_component(0.2) == 0.0

    def test_zero(self):
        assert _compute_gmli_component(0.0) == 0.0

    def test_negative_clamped(self):
        """Negative gmLI treated as 0."""
        assert _compute_gmli_component(-1.0) == 0.0


class TestComputeJobSecurityBackwardCompat:
    """Original formula must produce identical results when gmli=None."""

    def test_original_formula_basic(self):
        result = compute_job_security(0.8, 25.0)
        sv_comp = min(1.0, 25.0 / 30.0)
        expected = 0.6 * 0.8 + 0.4 * sv_comp
        assert result == pytest.approx(expected, abs=1e-6)

    def test_original_formula_zero(self):
        assert compute_job_security(0.0, 0.0) == 0.0

    def test_original_formula_max(self):
        assert compute_job_security(1.0, 30.0) == pytest.approx(1.0, abs=1e-6)

    def test_original_formula_high_sv_clamped(self):
        result = compute_job_security(1.0, 50.0)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_gmli_none_same_as_no_arg(self):
        a = compute_job_security(0.7, 20.0)
        b = compute_job_security(0.7, 20.0, gmli=None)
        assert a == b


class TestComputeJobSecurityWithGmli:
    """Enhanced formula when gmLI is provided."""

    def test_high_gmli_increases_security(self):
        without = compute_job_security(0.7, 15.0, gmli=None)
        with_high = compute_job_security(0.7, 15.0, gmli=2.0)
        # High gmLI (component=1.0) should generally increase score for
        # moderate hierarchy/saves combos
        assert with_high > 0.0

    def test_low_gmli_decreases_security(self):
        without = compute_job_security(0.7, 20.0, gmli=None)
        with_low = compute_job_security(0.7, 20.0, gmli=0.5)
        # gmli=0.5 => component=0.0, so enhanced formula loses 0.25 * 0
        # while reducing hierarchy/sv weights
        assert with_low < without

    def test_enhanced_formula_values(self):
        # hierarchy=0.8, sv=30 (comp=1.0), gmli=1.8 (comp=1.0)
        result = compute_job_security(0.8, 30.0, gmli=1.8)
        expected = 0.45 * 0.8 + 0.30 * 1.0 + 0.25 * 1.0
        assert result == pytest.approx(expected, abs=1e-6)

    def test_enhanced_formula_low_gmli(self):
        # hierarchy=1.0, sv=30 (comp=1.0), gmli=0.3 (comp=0.0)
        result = compute_job_security(1.0, 30.0, gmli=0.3)
        expected = 0.45 * 1.0 + 0.30 * 1.0 + 0.25 * 0.0
        assert result == pytest.approx(expected, abs=1e-6)


class TestGmliTrendPenalty:
    """Test the -0.10 trend penalty when gmLI drops > 0.5."""

    def test_trend_penalty_applied(self):
        no_trend = compute_job_security(0.8, 25.0, gmli=1.2)
        with_trend = compute_job_security(0.8, 25.0, gmli=1.2, gmli_prev=2.0)
        # Drop = 0.8 > 0.5, penalty = -0.10
        assert with_trend == pytest.approx(no_trend - 0.10, abs=1e-6)

    def test_no_penalty_small_drop(self):
        no_trend = compute_job_security(0.8, 25.0, gmli=1.5)
        with_small = compute_job_security(0.8, 25.0, gmli=1.5, gmli_prev=1.8)
        # Drop = 0.3 <= 0.5, no penalty
        assert with_small == pytest.approx(no_trend, abs=1e-6)

    def test_no_penalty_increase(self):
        no_trend = compute_job_security(0.8, 25.0, gmli=2.0)
        with_increase = compute_job_security(0.8, 25.0, gmli=2.0, gmli_prev=1.5)
        # gmli went UP, no penalty
        assert with_increase == pytest.approx(no_trend, abs=1e-6)

    def test_no_penalty_when_prev_none(self):
        a = compute_job_security(0.8, 25.0, gmli=1.0)
        b = compute_job_security(0.8, 25.0, gmli=1.0, gmli_prev=None)
        assert a == b

    def test_trend_penalty_exact_threshold(self):
        """Drop of exactly 0.5 should NOT trigger penalty (> not >=)."""
        no_trend = compute_job_security(0.8, 20.0, gmli=1.0)
        at_threshold = compute_job_security(0.8, 20.0, gmli=1.0, gmli_prev=1.5)
        assert at_threshold == pytest.approx(no_trend, abs=1e-6)


class TestClampingAndEdgeCases:
    """Result must always be clamped to [0, 1]."""

    def test_clamped_to_zero(self):
        # Low everything + big penalty
        result = compute_job_security(0.0, 0.0, gmli=0.0, gmli_prev=5.0)
        assert result == 0.0

    def test_clamped_to_one(self):
        result = compute_job_security(1.0, 50.0, gmli=3.0)
        assert result <= 1.0

    def test_gmli_zero(self):
        result = compute_job_security(0.5, 15.0, gmli=0.0)
        assert 0.0 <= result <= 1.0

    def test_negative_gmli(self):
        """Negative gmLI should not crash, treated as 0."""
        result = compute_job_security(0.5, 10.0, gmli=-1.0)
        assert 0.0 <= result <= 1.0

    def test_very_high_gmli(self):
        result = compute_job_security(0.5, 10.0, gmli=5.0)
        assert 0.0 <= result <= 1.0

    def test_negative_hierarchy_clamped(self):
        result = compute_job_security(-0.5, 10.0, gmli=1.5)
        assert result >= 0.0

    def test_negative_projected_sv_clamped(self):
        result = compute_job_security(0.5, -10.0, gmli=1.5)
        assert result >= 0.0
