"""Sensitivity analysis and counter-offer generation.

Spec reference: Section 14 L11 (Decision Output)
               Section 17 Phase 5 item 24

Provides two critical decision-support features:
  1. Sensitivity analysis: which categories/players most influence the trade grade?
  2. Counter-offer suggestions: if the trade is bad, what modifications would fix it?

The key insight: a "C+" trade might become a "B+" if you swap one player in the
package. Rather than just rejecting, show the user HOW to make it work.

Wires into:
  - src/engine/output/trade_evaluator.py: sensitivity report + counter-offers
"""

from __future__ import annotations

import logging

from src.valuation import LeagueConfig as _LC_Class

logger = logging.getLogger(__name__)

# Resolve once at import; do not store as a long-lived module singleton.
# (BUG-010: SF-21 architectural directive.)
_LC_ONCE = _LC_Class()
# SF-26 follow-up: derive CATEGORIES from LeagueConfig.
# This constant is currently unused inside the module (imports only pull
# functions, not constants) and ordering is purely cosmetic — no matrix
# indexing, sort, or correlation step depends on category position. Use the
# canonical R/HR/RBI/SB/AVG/OBP/W/L/SV/K/ERA/WHIP order so future readers
# who happen to import it see the same order as the rest of the engine.
CATEGORIES: list[str] = list(_LC_ONCE.all_categories)
INVERSE_CATEGORIES: set[str] = set(_LC_ONCE.inverse_stats)
del _LC_ONCE

# Minimum improvement to suggest a swap (SGP)
MIN_SWAP_IMPROVEMENT: float = 0.2

# Stat column mapping
STAT_MAP: dict[str, str] = {
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg",
    "OBP": "obp",
    "W": "w",
    "L": "l",
    "SV": "sv",
    "K": "k",
    "ERA": "era",
    "WHIP": "whip",
}


def category_sensitivity(
    category_impact: dict[str, float],
    category_weights: dict[str, float] | None = None,
) -> list[dict[str, float | str]]:
    """Rank categories by their contribution to trade surplus.

    Shows which categories drive the trade value — both positive
    contributors ("you gain a lot in HR") and negative contributors
    ("you lose significantly in SB").

    Args:
        category_impact: Per-category weighted SGP change from evaluate_trade.
        category_weights: Per-category marginal weights (None = equal weights).

    Returns:
        List of dicts sorted by absolute impact (highest first):
          [{category, impact, weight, direction}, ...]
        where direction is "helps" or "hurts".
    """
    result = []
    for cat, impact in category_impact.items():
        weight = category_weights.get(cat, 1.0) if category_weights else 1.0
        result.append(
            {
                "category": cat,
                "impact": round(impact, 3),
                "weight": round(weight, 3),
                "direction": "helps" if impact >= 0 else "hurts",
            }
        )

    # Sort by absolute impact (most influential first)
    result.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return result


def trade_sensitivity_report(
    category_impact: dict[str, float],
    category_weights: dict[str, float] | None = None,
    surplus_sgp: float = 0.0,
) -> dict:
    """Generate a complete sensitivity report for the trade.

    Combines category sensitivity with breakeven analysis: how much
    would the worst-performing category need to improve to flip
    the trade verdict?

    Args:
        category_impact: Per-category SGP impact.
        category_weights: Category weights.
        surplus_sgp: Total trade surplus.

    Returns:
        Dict with:
          - category_ranking: sorted list of category impacts
          - biggest_driver: the most impactful category
          - biggest_drag: the most negative category
          - breakeven_gap: how much surplus needs to change to flip verdict
          - vulnerability: how fragile the verdict is ("robust" / "fragile" / "razor-thin")
    """
    cat_ranking = category_sensitivity(category_impact, category_weights)

    biggest_driver = cat_ranking[0] if cat_ranking else None
    biggest_drag = None
    for entry in cat_ranking:
        if entry["direction"] == "hurts":
            biggest_drag = entry
            break

    # Breakeven analysis
    breakeven_gap = abs(surplus_sgp)

    # Vulnerability classification
    if breakeven_gap > 1.5:
        vulnerability = "robust"
    elif breakeven_gap > 0.5:
        vulnerability = "moderate"
    elif breakeven_gap > 0.1:
        vulnerability = "fragile"
    else:
        vulnerability = "razor-thin"

    return {
        "category_ranking": cat_ranking,
        "biggest_driver": biggest_driver,
        "biggest_drag": biggest_drag,
        "breakeven_gap": round(breakeven_gap, 3),
        "vulnerability": vulnerability,
    }
