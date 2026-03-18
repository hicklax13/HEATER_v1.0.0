"""Non-linear SGP (Standings Gained Points) category targeting.

Replaces the primitive ``1/(gap + 0.01)`` formula with bell-curve
marginal SGP based on the Normal PDF.  Each opponent contributes
proportional to proximity -- high value at standings boundaries,
near-zero when far from opponents.

Also provides slope-based SGP denominators via linear regression on
standings totals vs. ranks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.valuation import LeagueConfig as _LC_Class

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

_LC = _LC_Class()
ALL_CATS: list[str] = [c.lower() for c in _LC.all_categories]
INVERSE_CATS: set[str] = {c.lower() for c in _LC.inverse_stats}

# Fallback SGP denominators when regression is infeasible.
_DEFAULT_SGP_DENOMS: dict[str, float] = {c.lower(): v for c, v in _LC.sgp_denominators.items()}


# ── Core Functions ───────────────────────────────────────────────────


def default_category_sigmas() -> dict[str, float]:
    """Weekly standard deviation estimates for a competitive 12-team league.

    These represent expected season-end variance across the league in
    each scoring category.

    Returns:
        dict mapping category name to sigma (positive float).
    """
    return {
        "r": 25.0,
        "hr": 6.0,
        "rbi": 24.0,
        "sb": 5.0,
        "avg": 0.010,
        "obp": 0.008,
        "w": 2.5,
        "l": 2.0,
        "sv": 2.0,
        "k": 18.0,
        "era": 0.30,
        "whip": 0.03,
    }


def nonlinear_marginal_sgp(
    your_total: float,
    opponent_totals: list[float] | np.ndarray,
    sigma: float,
    *,
    is_inverse: bool = False,
) -> float:
    """Compute expected marginal SGP via Normal-PDF proximity weighting.

    For each opponent *i*, the contribution is::

        phi((your_total - opp_i) / sigma) / sigma

    For inverse categories (ERA, WHIP) the sign flips so that *lower*
    totals are better::

        phi((opp_i - your_total) / sigma) / sigma

    Args:
        your_total: Your team's current season total for this category.
        opponent_totals: Array of all other teams' totals.
        sigma: Standard deviation of the stat across the league.
            If ``<= 0``, returns ``0.0`` (no variance = no marginal value).
        is_inverse: ``True`` for ERA / WHIP where lower is better.

    Returns:
        Marginal SGP value (float).  Higher means one additional raw unit
        of this stat gains more standings points.
    """
    if sigma <= 0:
        return 0.0

    opp = np.asarray(opponent_totals, dtype=float)
    if opp.size == 0:
        return 0.0

    if is_inverse:
        z = (opp - your_total) / sigma
    else:
        z = (your_total - opp) / sigma

    return float(np.sum(norm.pdf(z) / sigma))


def compute_nonlinear_weights(
    standings: pd.DataFrame,
    team_name: str,
    sigmas: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute per-category weights using non-linear marginal SGP.

    Args:
        standings: DataFrame with columns ``team_name``, ``category``,
            ``total``, ``rank``.  One row per team per category.
        team_name: The user's team name (must appear in standings).
        sigmas: Optional per-category sigma overrides.  Defaults to
            :func:`default_category_sigmas`.

    Returns:
        dict mapping each category to a weight (mean normalised to 1.0,
        capped at 3.0).  Returns equal weights if standings are empty
        or the team is not found.
    """
    equal = {cat: 1.0 for cat in ALL_CATS}

    if standings is None or standings.empty:
        return equal

    if team_name not in standings["team_name"].values:
        logger.warning("Team '%s' not found in standings; returning equal weights", team_name)
        return equal

    sigs = sigmas or default_category_sigmas()
    raw_weights: dict[str, float] = {}

    for cat in ALL_CATS:
        cat_rows = standings[standings["category"] == cat]
        if cat_rows.empty:
            raw_weights[cat] = 1.0
            continue

        your_row = cat_rows[cat_rows["team_name"] == team_name]
        if your_row.empty:
            raw_weights[cat] = 1.0
            continue

        your_total = float(your_row["total"].iloc[0])
        opp_totals = cat_rows[cat_rows["team_name"] != team_name]["total"].values
        sig = sigs.get(cat, 1.0)

        raw_weights[cat] = nonlinear_marginal_sgp(
            your_total,
            opp_totals,
            sig,
            is_inverse=(cat in INVERSE_CATS),
        )

    # Normalise so mean weight = 1.0, cap at 3.0
    vals = list(raw_weights.values())
    mean_w = np.mean(vals) if vals else 1.0
    if mean_w <= 0:
        return equal

    weights = {}
    for cat in ALL_CATS:
        w = raw_weights.get(cat, 1.0) / mean_w
        weights[cat] = min(w, 3.0)

    return weights


def slope_sgp_denominators(standings: pd.DataFrame) -> dict[str, float]:
    """Derive SGP denominators from linear regression of standings totals.

    For each category, fits ``rank = alpha + beta * total`` and returns
    ``|1 / beta|`` as the denominator (raw stat units per standings
    position).

    For inverse categories (ERA, WHIP) a positive beta is expected
    (higher total = worse rank).

    Args:
        standings: DataFrame with columns ``category``, ``total``,
            ``rank``.  Must have at least 3 teams per category for a
            meaningful regression.

    Returns:
        dict mapping category name to SGP denominator.  Falls back to
        :data:`_DEFAULT_SGP_DENOMS` for categories with insufficient
        data.
    """
    if standings is None or standings.empty:
        return dict(_DEFAULT_SGP_DENOMS)

    result: dict[str, float] = {}

    for cat in ALL_CATS:
        cat_rows = standings[standings["category"] == cat]

        if len(cat_rows) < 3:
            result[cat] = _DEFAULT_SGP_DENOMS.get(cat, 1.0)
            continue

        totals = cat_rows["total"].values.astype(float)
        ranks = cat_rows["rank"].values.astype(float)

        # Zero variance in totals -- regression is meaningless
        if np.std(totals) < 1e-12:
            result[cat] = _DEFAULT_SGP_DENOMS.get(cat, 1.0)
            continue

        # polyfit: rank = beta*total + alpha  →  coefficients [beta, alpha]
        try:
            beta, _ = np.polyfit(totals, ranks, 1)
        except (np.linalg.LinAlgError, ValueError):
            result[cat] = _DEFAULT_SGP_DENOMS.get(cat, 1.0)
            continue

        if abs(beta) < 1e-12:
            result[cat] = _DEFAULT_SGP_DENOMS.get(cat, 1.0)
            continue

        result[cat] = abs(1.0 / beta)

    return result
