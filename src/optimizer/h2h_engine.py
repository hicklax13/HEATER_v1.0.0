"""Head-to-head weekly matchup analysis engine.

Answers the question: "How should I weight categories to beat THIS
WEEK's specific opponent?" by computing per-category win probabilities
and marginal-value-based category weights.

Variances can be calibrated via ConstantSet ("h2h_variance_r", "h2h_variance_hr").

Core math:
  For each category c with gap = my_total - opp_total and
  sigma = sqrt(var_my + var_opp):

    weight_c = phi(gap / sigma) / sigma    (Normal PDF — peaks at tie)
    p_win_c  = Phi(gap / sigma)            (Normal CDF)

  For inverse categories (ERA, WHIP) where lower is better,
  the gap is flipped: gap = opp_total - my_total.

Per-category win probability (LO-E3 / BR-6): low-count COUNTING categories
(SB, SV, W, L) use the Skellam model (the exact difference of two Poissons,
shared with the trade engine's weekly matrix) which captures the right-skew
the Gaussian approximation misses; rate cats (AVG/OBP/ERA/WHIP) and high-count
counting cats (R/HR/RBI/K) use the Normal model.

Overall matchup win probability (LO-E3 / BR-6): instead of treating the 12
category outcomes as INDEPENDENT Bernoullis (which made the lineup optimizer's
weekly win-prob ~3.4x more extreme than the Matchup Planner's copula estimate),
the overall P(win majority of categories) is computed from a CORRELATED joint
distribution via the same Gaussian copula the Matchup Planner and trade engine
use (``src.engine.portfolio.copula.DEFAULT_CORRELATION``). Positive correlation
among the hitting cluster and among the pitching cluster widens the win-count
distribution, pulling an over-confident independent estimate back toward the
copula ballpark.

This module has no Streamlit dependency and no external API calls.
"""

from __future__ import annotations

import logging
import math
from typing import TypedDict

import numpy as np
from scipy.stats import norm

from src.engine.output.weekly_matrix import _category_win_prob_skellam
from src.engine.portfolio.copula import (
    CAT_ORDER as _COPULA_CAT_ORDER,
)
from src.engine.portfolio.copula import (
    DEFAULT_CORRELATION as _COPULA_CORRELATION,
)
from src.engine.portfolio.copula import (
    GaussianCopula,
)
from src.validation.constant_optimizer import load_constants
from src.valuation import LeagueConfig as _LC_Class

logger = logging.getLogger(__name__)


class H2HWinProbability(TypedDict):
    """Return shape of :func:`estimate_h2h_win_probability`.

    Wave 8c (audit D4D-008): the original return was annotated
    ``dict[str, object]`` — accurate but opaque. Consumers had to cast.
    """

    per_category: dict[str, float]
    expected_wins: float
    overall_win_prob: float


# ── Constants ────────────────────────────────────────────────────────

_LC_ONCE = _LC_Class()
# Categories where lower is better
INVERSE_CATS: set[str] = {c.lower() for c in _LC_ONCE.inverse_stats}

# All 12 H2H categories
ALL_CATEGORIES: list[str] = [c.lower() for c in _LC_ONCE.all_categories]

# Low-count COUNTING categories that route to the Skellam (Poisson-difference)
# model for per-category win probability (LO-E3 / BR-6). These weekly totals are
# small enough (single digits / low teens) that the Gaussian approximation
# understates the spread and mis-handles the right skew — the same set the trade
# engine's weekly matrix flags (report C.5 / H.9). High-count counting cats
# (R/HR/RBI/K) and all rate cats keep the Normal model. Derived from the
# counting (non-rate) cats so it cannot drift from LeagueConfig's category set.
_LOW_COUNT_SKELLAM_NAMES: frozenset[str] = frozenset({"sb", "sv", "w", "l"})
_COUNTING_CATS: set[str] = {c.lower() for c in _LC_ONCE.all_categories} - {c.lower() for c in _LC_ONCE.rate_stats}
SKELLAM_CATS: frozenset[str] = frozenset(_LOW_COUNT_SKELLAM_NAMES & _COUNTING_CATS)

# Copula category order, lowercased to match this module's totals keys.
_COPULA_CATS_LOWER: list[str] = [c.lower() for c in _COPULA_CAT_ORDER]

del _LC_ONCE

# Small epsilon to avoid division by zero
_EPSILON: float = 1e-12

# Default Monte-Carlo draw count for the correlated overall win-probability.
_DEFAULT_OVERALL_SIMS: int = 10000

_CONSTANTS = load_constants()

# ── Default Variances ────────────────────────────────────────────────


def default_category_variances() -> dict[str, float]:
    """Return default weekly variance estimates for each fantasy category.

    These represent the typical weekly fluctuation in each category
    for a single fantasy team. Derived from historical weekly data
    of 12-team H2H leagues.

    Returns:
        Dict mapping category name to variance (sigma-squared).
        All values are positive.
    """
    return {
        "r": _CONSTANTS.get("h2h_variance_r"),  # ~225 default
        "hr": _CONSTANTS.get("h2h_variance_hr"),  # ~16 default
        "rbi": 14.0**2,  # ~196
        "sb": 3.0**2,  # ~9
        "avg": 0.015**2,  # ~0.000225
        "obp": 0.012**2,  # ~0.000144
        "w": 1.5**2,  # ~2.25
        "l": 1.5**2,  # ~2.25
        "sv": 1.5**2,  # ~2.25
        "k": 12.0**2,  # ~144
        "era": 0.5**2,  # ~0.25
        "whip": 0.05**2,  # ~0.0025
    }


def default_weekly_sigmas() -> dict[str, float]:
    """Return the CANONICAL per-category weekly standard deviation for each
    fantasy category, keyed by the league's uppercase category names.

    This is the single source of truth (MS-E1) for weekly per-category SDs used
    by every win-probability surface — standings_engine, standings_projection,
    and playoff_sim all resolve their per-category weekly SD from here so the
    same matchup yields the same win-prob across the Season Projections, Playoff
    Odds, and standings-engine pages.

    The values are simply ``sqrt(default_category_variances())``, re-keyed from
    the lowercase stat names to the league's canonical uppercase category names.
    Provenance: FanGraphs community research on 12-team H2H weekly category
    variance + the G-Score weekly-tau analysis (arXiv:2307.02188). A realistic
    one-team-sigma edge through the consumers' ``sqrt(2)*sd`` difference SD lands
    a win-prob at ~0.76 — neither compressed toward 0.5 nor saturated near 0.99.

    Returns:
        Dict mapping uppercase category name (R, HR, ..., ERA, WHIP) to the
        per-team weekly standard deviation. All values are positive.
    """
    variances = default_category_variances()
    cfg = _LC_Class()
    sigmas: dict[str, float] = {}
    for cat in cfg.all_categories:
        key = cfg.STAT_MAP.get(cat, cat.lower())
        var = variances.get(key, 1.0)
        sigmas[cat] = float(math.sqrt(max(float(var), 0.0)))
    return sigmas


# ── H2H Category Weights ────────────────────────────────────────────


def compute_h2h_category_weights(
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
    category_variances: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute category weights optimized for beating a specific H2H opponent.

    The weight for each category is based on the Normal PDF of the gap
    divided by sigma, with a swing-category emphasis multiplier
    (Haugh & Singal 2019). This peaks when the category is tied
    (maximum marginal value of improvement) and drops off as the gap
    grows (already winning or losing by too much to affect outcome).

    Formula per category c:
        gap = my_total - opp_total  (flipped for ERA/WHIP)
        sigma = sqrt(var_my + var_opp)
        p_win = Phi(gap / sigma)
        emphasis = 1.5 if swing (0.30-0.70), 1.0 if moderate, 0.5 if locked/lost
        weight = phi(gap / sigma) / sigma * emphasis

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

        # Swing-category emphasis (Haugh & Singal 2019):
        # Maximize investment in "swing" categories (30-70% P(win))
        # where marginal roster improvement has highest ROI.
        p_win = float(norm.cdf(z))
        if 0.30 <= p_win <= 0.70:
            emphasis = 1.5  # Toss-up: high marginal value
        elif 0.15 <= p_win < 0.30 or 0.70 < p_win <= 0.85:
            emphasis = 1.0  # Moderate: standard weight
        else:
            emphasis = 0.5  # Locked (>85%) or lost (<15%): deprioritize

        # Normal PDF at z, divided by sigma (marginal value density)
        raw_weights[cat] = float(norm.pdf(z)) / sigma * emphasis

    if not raw_weights:
        return {}

    # Normalize so mean weight = 1.0
    mean_weight = np.mean(list(raw_weights.values()))
    if mean_weight < _EPSILON:
        # All weights near zero (extreme gaps everywhere) -> uniform
        return {cat: 1.0 for cat in raw_weights}

    return {cat: w / mean_weight for cat, w in raw_weights.items()}


# ── H2H Win Probability ─────────────────────────────────────────────


def _per_category_win_prob(
    cat: str,
    my_val: float,
    opp_val: float,
    category_variances: dict[str, float],
) -> float:
    """Single-category P(win) with model dispatch by stat type (LO-E3).

    Low-count counting cats (``SKELLAM_CATS`` = SB/SV/W/L) use the Skellam
    (Poisson-difference) model shared with the trade engine, which captures the
    right-skew the Gaussian approximation misses for small weekly counts. Every
    other category (rate cats AVG/OBP/ERA/WHIP and high-count counting cats
    R/HR/RBI/K) uses the Normal CDF on the gap. Inverse cats (L/ERA/WHIP) are
    handled by each model (Skellam flips inside its helper; the Normal branch
    flips the gap here).

    Args:
        cat: Lowercase category name (e.g. ``"sb"``, ``"era"``).
        my_val: My team's projected total for the matchup period.
        opp_val: Opponent's projected total for the matchup period.
        category_variances: Per-category one-team variance table.

    Returns:
        P(win the category) in [0, 1].
    """
    inverse = cat in INVERSE_CATS
    if cat in SKELLAM_CATS:
        # Skellam consumes the per-week mean counts directly (the totals are the
        # matchup-period means) and handles the inverse flip internally.
        return _category_win_prob_skellam(float(my_val), float(opp_val), inverse=inverse)

    single_var = category_variances.get(cat, 1.0)
    combined_var = 2.0 * single_var
    sigma = math.sqrt(combined_var) if combined_var > _EPSILON else _EPSILON

    # For inverse cats: lower is better, so P(win) = P(opp - my > 0)
    if inverse:
        gap = float(opp_val) - float(my_val)
    else:
        gap = float(my_val) - float(opp_val)

    return float(norm.cdf(gap / sigma))


def _overall_win_prob_copula(
    per_category: dict[str, float],
    n_sims: int,
    rng: np.random.RandomState | None,
) -> float:
    """P(win the majority of categories) from a CORRELATED joint draw (LO-E3).

    Replaces the legacy independent-Bernoulli Normal approximation
    (``P(sum_of_independent_wins > n/2)``). Independence ignored the strong
    within-cluster correlations (HR↔RBI, ERA↔WHIP, …), which compressed the
    win-count variance and pushed the overall win-prob toward the extremes —
    the lineup optimizer landed ~3.4x off the Matchup Planner's copula estimate
    for the same matchup (BR-6).

    Method: draw ``n_sims`` correlated uniform variates over the categories
    present (Gaussian copula on the sub-matrix of ``DEFAULT_CORRELATION``), mark
    category ``c`` won when ``u_c <= p_c``, count wins per draw, and return
    ``P(wins > n/2)`` with a tie at exactly ``n/2`` scored as half a win (Yahoo
    splits an even category tie). The per-category marginals ``p_c`` already
    encode every team/projection difference, so the copula only injects the
    dependency structure between category outcomes.

    Args:
        per_category: ``{lowercase_cat: p_win}`` for the categories present.
        n_sims: Number of correlated Monte-Carlo draws.
        rng: Optional RandomState for deterministic results (tests pass a seed).

    Returns:
        Overall matchup win probability in [0, 1].
    """
    cats = [c for c in _COPULA_CATS_LOWER if c in per_category]
    n_cats = len(cats)
    if n_cats == 0:
        return 0.5  # No data -> coin flip

    if rng is None:
        rng = np.random.RandomState()

    probs = np.array([per_category[c] for c in cats], dtype=float)

    # Sub-matrix of the canonical correlation in the order of `cats`.
    idx = [_COPULA_CATS_LOWER.index(c) for c in cats]
    sub_corr = _COPULA_CORRELATION[np.ix_(idx, idx)]

    copula = GaussianCopula(sub_corr)
    u = copula.sample(n_sims, rng)  # (n_sims, n_cats) correlated uniforms

    wins = (u <= probs).sum(axis=1).astype(float)
    half = n_cats / 2.0
    won = float((wins > half).sum())
    tied = float((wins == half).sum())
    return (won + 0.5 * tied) / float(n_sims)


def estimate_h2h_win_probability(
    my_totals: dict[str, float],
    opp_totals: dict[str, float],
    category_variances: dict[str, float] | None = None,
    n_sims: int = _DEFAULT_OVERALL_SIMS,
    rng: np.random.RandomState | None = None,
) -> H2HWinProbability:
    """Estimate probability of winning each category and overall matchup.

    Per-category (LO-E3 / BR-6): low-count counting cats (SB/SV/W/L) use the
    Skellam (Poisson-difference) model; every other cat uses the Normal CDF on
    the gap. Inverse categories (L/ERA/WHIP) are flipped since lower is better.

    Overall (LO-E3 / BR-6): the matchup win probability is computed from a
    CORRELATED joint distribution of the category outcomes via the shared
    Gaussian copula (``DEFAULT_CORRELATION``), NOT a sum of independent
    Bernoullis. Correlation among the hitting and pitching clusters widens the
    win-count distribution and keeps the lineup optimizer's estimate in the same
    ballpark as the Matchup Planner's copula simulation.

    Args:
        my_totals: Dict of my team's projected totals per category.
        opp_totals: Dict of opponent's projected totals per category.
        category_variances: Per-category variance for one team.
            If None, uses defaults.
        n_sims: Correlated Monte-Carlo draw count for the overall win-prob.
        rng: Optional RandomState for deterministic overall win-prob (tests
            pass a seed; production leaves it None for a fresh draw).

    Returns:
        Dict with:
          - per_category: dict[str, float] — P(win) for each category
          - expected_wins: float — sum of P(win) across all categories
          - overall_win_prob: float — P(win majority) via Gaussian copula MC
    """
    if category_variances is None:
        category_variances = default_category_variances()

    per_category: dict[str, float] = {}

    for cat in ALL_CATEGORIES:
        my_val = my_totals.get(cat)
        opp_val = opp_totals.get(cat)

        if my_val is None or opp_val is None:
            continue

        per_category[cat] = _per_category_win_prob(cat, float(my_val), float(opp_val), category_variances)

    # Expected wins = sum of individual win probabilities
    expected_wins = sum(per_category.values())

    # Overall win probability via the correlated Gaussian-copula joint draw.
    overall_win_prob = _overall_win_prob_copula(per_category, n_sims=n_sims, rng=rng)

    return {
        "per_category": per_category,
        "expected_wins": expected_wins,
        "overall_win_prob": overall_win_prob,
    }
