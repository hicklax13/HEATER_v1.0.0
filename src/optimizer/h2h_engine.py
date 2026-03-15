"""Head-to-head weekly matchup analysis engine.

Answers the question: "How should I weight categories to beat THIS
WEEK's specific opponent?" by computing per-category win probabilities
and marginal-value-based category weights.

Core math:
  For each category c with gap = my_total - opp_total and
  sigma = sqrt(var_my + var_opp):

    weight_c = phi(gap / sigma) / sigma    (Normal PDF — peaks at tie)
    p_win_c  = Phi(gap / sigma)            (Normal CDF)

  For inverse categories (ERA, WHIP) where lower is better,
  the gap is flipped: gap = opp_total - my_total.

This module has no Streamlit dependency and no external API calls.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────

# Categories where lower is better
INVERSE_CATS: set[str] = {"era", "whip"}

# All 10 standard 5x5 roto categories
ALL_CATEGORIES: list[str] = [
    "r",
    "hr",
    "rbi",
    "sb",
    "avg",
    "w",
    "sv",
    "k",
    "era",
    "whip",
]

# Small epsilon to avoid division by zero
_EPSILON: float = 1e-12


# ── Default Variances ────────────────────────────────────────────────


def default_category_variances() -> dict[str, float]:
    """Return default weekly variance estimates for each fantasy category.

    These represent the typical weekly fluctuation in each category
    for a single fantasy team. Derived from historical weekly data
    of 12-team roto leagues.

    Returns:
        Dict mapping category name to variance (sigma-squared).
        All values are positive.
    """
    return {
        "r": 15.0**2,  # ~225
        "hr": 4.0**2,  # ~16
        "rbi": 14.0**2,  # ~196
        "sb": 3.0**2,  # ~9
        "avg": 0.015**2,  # ~0.000225
        "w": 1.5**2,  # ~2.25
        "sv": 1.5**2,  # ~2.25
        "k": 12.0**2,  # ~144
        "era": 0.5**2,  # ~0.25
        "whip": 0.05**2,  # ~0.0025
    }


# ── H2H Category Weights ────────────────────────────────────────────


def compute_h2h_category_weights(
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
    category_variances: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute category weights optimized for beating a specific H2H opponent.

    The weight for each category is based on the Normal PDF of the gap
    divided by sigma. This peaks when the category is tied (maximum
    marginal value of improvement) and drops off as the gap grows
    (already winning or losing by too much to affect outcome).

    Formula per category c:
        gap = my_total - opp_total  (flipped for ERA/WHIP)
        sigma = sqrt(var_my + var_opp)
        weight = phi(gap / sigma) / sigma

    Weights are normalized so their mean equals 1.0.

    Args:
        my_totals: Dict of my team's projected totals per category.
        opp_totals: Dict of opponent's projected totals per category.
        category_variances: Per-category variance for one team. Both
            teams are assumed to have equal variance, so total
            variance = 2 * category_variance. If None, uses defaults.

    Returns:
        Dict mapping category name to normalized weight (mean = 1.0).
        Categories not present in both totals are omitted.
    """
    if category_variances is None:
        category_variances = default_category_variances()

    raw_weights: dict[str, float] = {}

    for cat in ALL_CATEGORIES:
        my_val = my_totals.get(cat)
        opp_val = opp_totals.get(cat)

        if my_val is None or opp_val is None:
            continue

        # Combined variance (both teams contribute uncertainty)
        single_var = category_variances.get(cat, 1.0)
        combined_var = 2.0 * single_var
        sigma = math.sqrt(combined_var) if combined_var > _EPSILON else _EPSILON

        # Gap: positive = I'm ahead, negative = I'm behind
        # For inverse cats (ERA, WHIP): lower is better, so flip
        if cat in INVERSE_CATS:
            gap = float(opp_val) - float(my_val)
        else:
            gap = float(my_val) - float(opp_val)

        z = gap / sigma
        # Normal PDF at z, divided by sigma (marginal value density)
        raw_weights[cat] = float(norm.pdf(z)) / sigma

    if not raw_weights:
        return {}

    # Normalize so mean weight = 1.0
    mean_weight = np.mean(list(raw_weights.values()))
    if mean_weight < _EPSILON:
        # All weights near zero (extreme gaps everywhere) -> uniform
        return {cat: 1.0 for cat in raw_weights}

    return {cat: w / mean_weight for cat, w in raw_weights.items()}


# ── H2H Win Probability ─────────────────────────────────────────────


def estimate_h2h_win_probability(
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
    category_variances: dict[str, float] | None = None,
) -> dict[str, object]:
    """Estimate probability of winning each category and overall matchup.

    For each category, uses the Normal CDF to compute P(win):
        P(win_c) = Phi(gap / sigma)

    For inverse categories (ERA, WHIP), the gap is flipped since
    lower values are better.

    The overall win probability is approximated by treating the number
    of category wins as approximately Normal (sum of Bernoulli-ish
    variables) and computing P(total_wins > 5).

    Args:
        my_totals: Dict of my team's projected totals per category.
        opp_totals: Dict of opponent's projected totals per category.
        category_variances: Per-category variance for one team.
            If None, uses defaults.

    Returns:
        Dict with:
          - per_category: dict[str, float] — P(win) for each category
          - expected_wins: float — sum of P(win) across all categories
          - overall_win_prob: float — P(expected_wins > 5) via Normal approx
    """
    if category_variances is None:
        category_variances = default_category_variances()

    per_category: dict[str, float] = {}

    for cat in ALL_CATEGORIES:
        my_val = my_totals.get(cat)
        opp_val = opp_totals.get(cat)

        if my_val is None or opp_val is None:
            continue

        single_var = category_variances.get(cat, 1.0)
        combined_var = 2.0 * single_var
        sigma = math.sqrt(combined_var) if combined_var > _EPSILON else _EPSILON

        # For inverse cats: lower is better, so P(win) = P(opp - my > 0)
        if cat in INVERSE_CATS:
            gap = float(opp_val) - float(my_val)
        else:
            gap = float(my_val) - float(opp_val)

        z = gap / sigma
        per_category[cat] = float(norm.cdf(z))

    # Expected wins = sum of individual win probabilities
    expected_wins = sum(per_category.values())

    # Overall win probability: P(wins > 5) where wins ~ sum of Bernoullis
    # Approximate with Normal: mean = E[wins], var = sum(p*(1-p))
    n_cats = len(per_category)
    if n_cats > 0:
        win_variance = sum(p * (1.0 - p) for p in per_category.values())
        win_sigma = math.sqrt(win_variance) if win_variance > _EPSILON else _EPSILON
        # Win if > half the categories (>5 for 10 cats)
        threshold = n_cats / 2.0
        overall_z = (expected_wins - threshold) / win_sigma
        overall_win_prob = float(norm.cdf(overall_z))
    else:
        overall_win_prob = 0.5  # No data -> coin flip

    return {
        "per_category": per_category,
        "expected_wins": expected_wins,
        "overall_win_prob": overall_win_prob,
    }
