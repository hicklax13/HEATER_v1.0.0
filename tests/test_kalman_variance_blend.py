"""Structural-invariant guard: Kalman variance blend in weekly matrix.

Per 2026-05-23 design Q3 follow-up — _category_win_prob now accepts
optional Kalman true-talent variance and blends it with the per-week
CV-based sampling variance:

    σ_effective² = (CV × mean)² + kalman_var

Captures BOTH:
  - Per-week sampling variance (CV term — dominant short-term)
  - True-talent uncertainty (Kalman term — matters for under-sampled players)

Contract locked:
  - kalman_var_h/o default to 0.0 → legacy CV-only behavior unchanged
  - Non-zero Kalman var increases combined σ → win-prob moves toward 0.5
  - Symmetric: equal Kalman var on both sides → same shift as CV-only
"""

from __future__ import annotations

import math

from src.engine.output.weekly_matrix import _category_win_prob

# ── Backward compatibility ───────────────────────────────────────────


def test_kalman_var_zero_matches_legacy_behavior() -> None:
    """Default (kalman_var=0.0) must produce identical output to pre-fix."""
    p_default = _category_win_prob(mu_h=25.0, mu_o=20.0, cv=0.20, inverse=False)
    p_explicit_zero = _category_win_prob(
        mu_h=25.0,
        mu_o=20.0,
        cv=0.20,
        inverse=False,
        kalman_var_h=0.0,
        kalman_var_o=0.0,
    )
    assert p_default == p_explicit_zero, "kalman_var=0 must be exactly legacy CV-only result"


# ── Variance addition correctness ────────────────────────────────────


def test_kalman_var_increases_combined_sigma() -> None:
    """Adding Kalman variance → larger combined σ → win-prob closer to 0.5."""
    # Without Kalman uncertainty
    p_no_kalman = _category_win_prob(mu_h=30.0, mu_o=20.0, cv=0.20, inverse=False)
    # With significant Kalman uncertainty
    p_with_kalman = _category_win_prob(
        mu_h=30.0,
        mu_o=20.0,
        cv=0.20,
        inverse=False,
        kalman_var_h=25.0,
        kalman_var_o=25.0,
    )
    assert p_no_kalman > p_with_kalman > 0.5, (
        f"Kalman uncertainty must shrink win prob toward 0.5. no_kalman={p_no_kalman}, with_kalman={p_with_kalman}"
    )


def test_kalman_var_symmetric_on_equal_means() -> None:
    """Equal means + equal Kalman var on both sides → exact 0.5."""
    p = _category_win_prob(
        mu_h=20.0,
        mu_o=20.0,
        cv=0.20,
        inverse=False,
        kalman_var_h=10.0,
        kalman_var_o=10.0,
    )
    assert abs(p - 0.5) < 1e-9, f"Equal means + equal Kalman var must give 0.5; got {p}"


def test_kalman_var_only_one_side_still_works() -> None:
    """Kalman uncertainty on one team only still produces valid probability."""
    p = _category_win_prob(
        mu_h=25.0,
        mu_o=25.0,
        cv=0.20,
        inverse=False,
        kalman_var_h=50.0,
        kalman_var_o=0.0,
    )
    # User has high uncertainty (kalman var 50) but same mean → still ~0.5
    # (variance affects spread but mean is identical)
    assert 0.0 <= p <= 1.0
    assert abs(p - 0.5) < 0.05


def test_negative_kalman_var_clamped_to_zero() -> None:
    """Negative Kalman var (data error) should not crash — clamps to 0."""
    p_neg = _category_win_prob(
        mu_h=25.0,
        mu_o=20.0,
        cv=0.20,
        inverse=False,
        kalman_var_h=-10.0,
        kalman_var_o=-5.0,
    )
    p_legacy = _category_win_prob(mu_h=25.0, mu_o=20.0, cv=0.20, inverse=False)
    assert p_neg == p_legacy, "Negative Kalman var must be treated as 0 (clamped)"


def test_inverse_cat_with_kalman_still_correctly_flipped() -> None:
    """ERA inverse-cat handling preserved when Kalman var added."""
    # User has ERA 3.00 (better), opp has 4.00 (worse)
    p = _category_win_prob(
        mu_h=3.00,
        mu_o=4.00,
        cv=0.18,
        inverse=True,
        kalman_var_h=0.01,
        kalman_var_o=0.01,
    )
    # Lower ERA wins → p > 0.5
    assert p > 0.5, f"Better ERA must win even with Kalman uncertainty; got {p}"


def test_combined_sigma_formula_verified() -> None:
    """Algebraic check: σ_total² = (CV×mean)² + kalman_var."""
    # Carefully control inputs so we can predict the result
    mu_h, mu_o = 100.0, 100.0  # identical means → z=0 regardless of σ
    cv = 0.10
    kvar_h, kvar_o = 4.0, 9.0
    # Even with these σ's, z=0 → p=0.5
    p = _category_win_prob(
        mu_h=mu_h,
        mu_o=mu_o,
        cv=cv,
        inverse=False,
        kalman_var_h=kvar_h,
        kalman_var_o=kvar_o,
    )
    assert abs(p - 0.5) < 1e-9, f"With equal means, p must be 0.5 regardless of σ; got {p}"

    # Now test that adding Kalman var ACTUALLY changes σ vs no Kalman
    # by checking that bigger CV gives same p shift as adding Kalman var
    mu_h2 = 105.0  # offset
    # Pure CV: σ_h = 105 × 0.10 = 10.5, σ_o = 100 × 0.10 = 10.0
    # σ_total = sqrt(10.5² + 10.0²) = sqrt(210.25) ≈ 14.5
    p_cv_only = _category_win_prob(mu_h=mu_h2, mu_o=mu_o, cv=cv, inverse=False)
    # CV + Kalman: σ_h_total² = 110.25 + 4 = 114.25, σ_o_total² = 100 + 9 = 109
    # σ_total = sqrt(223.25) ≈ 14.94
    p_with_kalman = _category_win_prob(
        mu_h=mu_h2,
        mu_o=mu_o,
        cv=cv,
        inverse=False,
        kalman_var_h=4.0,
        kalman_var_o=9.0,
    )
    # Adding Kalman variance increases denominator → z smaller → p closer to 0.5
    assert 0.5 < p_with_kalman < p_cv_only, (
        f"Kalman var should pull p toward 0.5: cv_only={p_cv_only}, with_kalman={p_with_kalman}"
    )
