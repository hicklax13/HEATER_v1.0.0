"""Tests for swing-category emphasis in H2H category weights.

Validates that the emphasis multiplier correctly amplifies toss-up
categories (30-70% P(win)) and deprioritizes locked/lost categories
(<15% or >85% P(win)).

Reference: Haugh & Singal 2019, Management Science/Columbia.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from src.optimizer.h2h_engine import (
    ALL_CATEGORIES,
    INVERSE_CATS,
    compute_h2h_category_weights,
    default_category_variances,
)

# ── Helpers ─────────────────────────────────────────────────────────


def _base_totals() -> dict[str, float]:
    """Return baseline category totals for a typical team."""
    return {
        "r": 80.0,
        "hr": 20.0,
        "rbi": 75.0,
        "sb": 10.0,
        "avg": 0.265,
        "obp": 0.330,
        "w": 8.0,
        "l": 4.0,
        "sv": 5.0,
        "k": 70.0,
        "era": 3.80,
        "whip": 1.20,
    }


def _make_opponent(base: dict[str, float], overrides: dict[str, float]) -> dict[str, float]:
    """Clone base totals with specific overrides."""
    opp = dict(base)
    opp.update(overrides)
    return opp


def _compute_p_win(my_val: float, opp_val: float, cat: str, var: float) -> float:
    """Compute P(win) for a single category given values and variance."""
    sigma = math.sqrt(2.0 * var)
    if cat in INVERSE_CATS:
        gap = opp_val - my_val
    else:
        gap = my_val - opp_val
    z = gap / sigma
    return float(norm.cdf(z))


# ── Test: tied category gets 1.5x emphasis ──────────────────────────


def test_tied_category_gets_swing_emphasis():
    """When gap=0, P(win)=0.50, the emphasis should be 1.5x.

    With uniform variance across all categories, all tied categories
    get the same emphasis and normalize to 1.0 each.
    """
    my = _base_totals()
    opp = dict(my)  # Identical -> all tied

    # Use uniform variance so all categories have same raw PDF/sigma
    uniform_var = {cat: 1.0 for cat in my}
    weights = compute_h2h_category_weights(my, opp, uniform_var)

    # All categories tied with same variance -> all weights equal -> 1.0
    for cat, w in weights.items():
        assert abs(w - 1.0) < 0.01, f"{cat} weight should be ~1.0 when all tied, got {w}"


# ── Test: dominating category gets 0.5x emphasis ───────────────────


def test_dominating_category_gets_low_emphasis():
    """P(win) > 0.85 should receive 0.5x emphasis (deprioritized)."""
    my = _base_totals()
    variances = default_category_variances()

    # Make HR massively ahead so P(win) > 0.95
    # HR variance is ~16, sigma = sqrt(32) ~ 5.66, need gap >> sigma
    # gap = 30 -> z = 30/5.66 ~ 5.3 -> P(win) ~ 1.0
    opp = _make_opponent(my, {"hr": my["hr"] - 30})

    # Verify P(win) for HR is indeed > 0.85
    p_win_hr = _compute_p_win(my["hr"], opp["hr"], "hr", variances["hr"])
    assert p_win_hr > 0.85, f"Expected P(win) > 0.85 for HR, got {p_win_hr}"

    weights = compute_h2h_category_weights(my, opp, variances)

    # HR should have lower relative weight than tied categories
    # All other cats are tied (1.5x emphasis), HR gets 0.5x emphasis
    # So HR weight should be notably below mean (1.0)
    assert weights["hr"] < 0.5, f"Dominating HR should have low weight, got {weights['hr']}"


# ── Test: losing badly gets 0.5x emphasis ──────────────────────────


def test_losing_badly_gets_low_emphasis():
    """P(win) < 0.15 should receive 0.5x emphasis (deprioritized)."""
    my = _base_totals()
    variances = default_category_variances()

    # Make HR massively behind so P(win) < 0.05
    opp = _make_opponent(my, {"hr": my["hr"] + 30})

    p_win_hr = _compute_p_win(my["hr"], opp["hr"], "hr", variances["hr"])
    assert p_win_hr < 0.15, f"Expected P(win) < 0.15 for HR, got {p_win_hr}"

    weights = compute_h2h_category_weights(my, opp, variances)

    # HR should have lower relative weight than tied categories
    assert weights["hr"] < 0.5, f"Badly losing HR should have low weight, got {weights['hr']}"


# ── Test: moderate category gets 1.0x emphasis ─────────────────────


def test_moderate_category_gets_standard_emphasis():
    """P(win) around 0.75 should receive 1.0x emphasis (standard).

    Use uniform variances so the emphasis multiplier difference
    is the dominant factor in relative weight, not variance scaling.
    """
    my = _base_totals()
    uniform_var = {cat: 1.0 for cat in my}
    sigma = math.sqrt(2.0)

    # Want P(win) ~ 0.75 for HR -> z ~ 0.674 -> gap ~ 0.674 * sigma
    target_z = norm.ppf(0.75)  # ~0.674
    gap = target_z * sigma

    opp = _make_opponent(my, {"hr": my["hr"] - gap})

    p_win_hr = _compute_p_win(my["hr"], opp["hr"], "hr", uniform_var["hr"])
    assert 0.70 < p_win_hr <= 0.85, f"Expected P(win) in (0.70, 0.85), got {p_win_hr}"

    weights = compute_h2h_category_weights(my, opp, uniform_var)

    # HR at 1.0x emphasis among 11 cats at 1.5x emphasis
    # HR weight should be below mean but above the 0.5x locked floor
    assert weights["hr"] < 1.0, f"Moderate HR weight should be below mean, got {weights['hr']}"
    # With uniform variance, the PDF at z=0.674 is ~0.318 vs PDF at z=0 is ~0.399
    # After emphasis: HR raw = 0.318 * 1.0, others raw ~ 0.399 * 1.5
    # So HR should be roughly 0.318 / (mean of all) which is > 0.4
    assert weights["hr"] > 0.4, f"Moderate HR weight should not be too low, got {weights['hr']}"


# ── Test: weights still normalize to mean=1.0 ──────────────────────


def test_weights_normalize_to_mean_one():
    """After swing emphasis, weights should still have mean=1.0."""
    my = _base_totals()
    variances = default_category_variances()

    # Create varied scenario: some ahead, some behind, some tied
    opp = _make_opponent(my, {
        "hr": my["hr"] - 25,   # dominating
        "sb": my["sb"] - 15,   # dominating
        "k": my["k"] + 40,     # losing badly
        "era": my["era"] + 2.0,  # losing badly (inverse: higher ERA = worse)
    })

    weights = compute_h2h_category_weights(my, opp, variances)
    mean_w = np.mean(list(weights.values()))

    assert abs(mean_w - 1.0) < 0.01, f"Mean weight should be ~1.0, got {mean_w}"


# ── Test: inverse stats flip correctly ──────────────────────────────


def test_inverse_stats_flip_for_p_win():
    """For ERA/WHIP, being ahead means my_val < opp_val.

    A team with ERA=2.50 vs opponent ERA=5.00 should be dominating
    (P(win) >> 0.85) and get 0.5x emphasis.
    """
    my = _base_totals()
    variances = default_category_variances()

    # Dominating ERA: my ERA much lower (better) than opponent
    opp = _make_opponent(my, {"era": my["era"] + 3.0})

    # Verify P(win) for ERA > 0.85 (inverse: gap = opp - my = +3.0)
    p_win_era = _compute_p_win(my["era"], opp["era"], "era", variances["era"])
    assert p_win_era > 0.85, f"Expected P(win) > 0.85 for ERA, got {p_win_era}"

    weights = compute_h2h_category_weights(my, opp, variances)

    # ERA should be deprioritized (0.5x emphasis) relative to tied cats
    assert weights["era"] < 0.5, (
        f"Dominating ERA (inverse) should have low weight, got {weights['era']}"
    )


def test_inverse_stats_losing_gets_low_emphasis():
    """For ERA, having a much higher ERA (worse) means P(win) < 0.15."""
    my = _base_totals()
    variances = default_category_variances()

    # Losing ERA badly: my ERA much higher (worse) than opponent
    opp = _make_opponent(my, {"era": my["era"] - 3.0})

    p_win_era = _compute_p_win(my["era"], opp["era"], "era", variances["era"])
    assert p_win_era < 0.15, f"Expected P(win) < 0.15 for ERA, got {p_win_era}"

    weights = compute_h2h_category_weights(my, opp, variances)

    assert weights["era"] < 0.5, (
        f"Badly losing ERA (inverse) should have low weight, got {weights['era']}"
    )


# ── Test: swing emphasis relative ordering ──────────────────────────


def test_swing_category_weighted_higher_than_locked():
    """A tied category should get higher weight than a locked one."""
    my = _base_totals()
    variances = default_category_variances()

    # HR dominating, RBI tied
    opp = _make_opponent(my, {"hr": my["hr"] - 30})

    weights = compute_h2h_category_weights(my, opp, variances)

    # RBI tied -> 1.5x emphasis, HR locked -> 0.5x emphasis
    # RBI weight should be higher than HR weight
    assert weights["rbi"] > weights["hr"], (
        f"Tied RBI ({weights['rbi']:.3f}) should outweigh locked HR ({weights['hr']:.3f})"
    )
