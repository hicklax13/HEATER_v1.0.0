"""Multi-period rolling horizon planning for lineup optimization.

Extends single-week optimization to a multi-week horizon by discounting
future weeks and computing season-balance urgency weights.  The key
insight is that sitting a pitcher THIS week to preserve starts for NEXT
week can be +EV when viewed through a multi-week lens.

Core formulas:

  Rolling horizon value:
    V = sum_{w=0..H-1} gamma^w * V_w
  where gamma is the discount factor and V_w is the projected value of
  week w's lineup decisions.

  Season-balance urgency:
    urgency_c = remaining_needed_c / remaining_available_c
  For inverse categories (ERA, WHIP), urgency is inverted: being "behind"
  means your stat is too HIGH.

This module has no Streamlit dependency and no external API calls.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ── Constants ────────────────────────────────────────────────────────

ALL_CATEGORIES: list[str] = [
    "r",
    "hr",
    "rbi",
    "sb",
    "avg",
    "obp",
    "w",
    "l",
    "sv",
    "k",
    "era",
    "whip",
]

INVERSE_CATS: set[str] = {"l", "era", "whip"}

# Maximum per-category urgency weight to prevent a single category from
# dominating the entire objective function.
_MAX_URGENCY: float = 3.0

# Minimum urgency floor so that categories where you are ahead of pace
# still receive nonzero weight.
_MIN_URGENCY: float = 0.1

# Small epsilon to avoid division by zero.
_EPSILON: float = 1e-12


# ── Rolling Horizon Optimization ─────────────────────────────────────


def rolling_horizon_optimization(
    weekly_projections: list[dict[str, float]],
    category_weights: dict[str, float],
    horizon_weeks: int = 3,
    discount: float = 0.95,
) -> dict[str, object]:
    """Compute multi-week blended category weights using a rolling horizon.

    Takes per-week projected stat totals and blends them with exponential
    time discounting, so that this week's decisions account for their
    downstream impact on future weeks.

    Args:
        weekly_projections: List of dicts, one per week, mapping category
            name to projected total for that week.  Length may exceed
            ``horizon_weeks``; only the first ``horizon_weeks`` entries
            are used.
        category_weights: Base per-category weights (e.g. from H2H or
            category targeting).
        horizon_weeks: Number of weeks to look ahead.  Must be >= 1.
        discount: Per-week discount factor gamma in (0, 1].
            1.0 = equal weight on all weeks.
            0.5 = heavy recency bias.

    Returns:
        Dict with:
          - ``week_weights``: list[float] of per-week discount multipliers
          - ``blended_category_weights``: dict[str, float] of blended
            weights normalized so mean = 1.0
          - ``horizon_value``: float total discounted value across the
            horizon
    """
    horizon_weeks = max(1, horizon_weeks)
    discount = max(0.0, min(1.0, discount))

    # Build discount weights for each week in the horizon
    n_weeks = min(horizon_weeks, len(weekly_projections))
    if n_weeks == 0:
        # No projections at all -- return base weights unchanged
        return {
            "week_weights": [],
            "blended_category_weights": dict(category_weights),
            "horizon_value": 0.0,
        }

    week_weights = [discount**w for w in range(n_weeks)]

    # Sum of discount weights for normalization
    total_discount = sum(week_weights)
    if total_discount < _EPSILON:
        total_discount = _EPSILON

    # Compute blended category values across the horizon:
    # For each category, blend the per-week projected values weighted by
    # the discount factor, then multiply by the base category weight.
    blended: dict[str, float] = {}
    horizon_value = 0.0

    # Gather all categories present in any week
    all_cats = set()
    for wp in weekly_projections[:n_weeks]:
        all_cats.update(wp.keys())

    for cat in all_cats:
        base_w = category_weights.get(cat, 1.0)
        discounted_total = 0.0
        for w_idx in range(n_weeks):
            val = weekly_projections[w_idx].get(cat, 0.0)
            discounted_total += week_weights[w_idx] * val
        # The blended weight incorporates both the base weight and the
        # discounted projection magnitude
        blended[cat] = base_w * (discounted_total / total_discount)
        horizon_value += base_w * discounted_total

    # Normalize blended weights so mean = 1.0
    if blended:
        vals = list(blended.values())
        mean_val = sum(vals) / len(vals)
        if mean_val > _EPSILON:
            blended = {cat: v / mean_val for cat, v in blended.items()}
        else:
            blended = {cat: 1.0 for cat in blended}

    return {
        "week_weights": week_weights,
        "blended_category_weights": blended,
        "horizon_value": horizon_value,
    }


# ── Season Balance Weights ───────────────────────────────────────────


def season_balance_weights(
    ytd_totals: dict[str, float],
    target_totals: dict[str, float],
    weeks_remaining: int,
    weekly_rates: dict[str, float],
) -> dict[str, float]:
    """Compute per-category urgency weights based on season pace.

    For each category:
      remaining_needed = target - ytd
      remaining_available = weekly_rate * weeks_remaining
      urgency = remaining_needed / remaining_available

    High urgency (>1.0) means you are behind pace and need to prioritize
    that category.  Low urgency (<1.0) means you are ahead of pace.

    For inverse categories (ERA, WHIP) where lower is better, the logic
    is inverted: being "behind" means your ERA is too HIGH relative to
    the target.

    Args:
        ytd_totals: Year-to-date accumulated totals per category.
        target_totals: Season-end target totals per category (e.g. the
            projection total or the total needed for a certain standings rank).
        weeks_remaining: Number of weeks left in the season.  If <= 0,
            returns equal weights (no time to adjust).
        weekly_rates: Expected per-week production rate for each category
            (e.g. weekly HR rate = 3.5).

    Returns:
        Dict mapping category name to urgency weight.  Normalized so the
        mean weight across all categories equals 1.0.  Individual weights
        are capped at ``_MAX_URGENCY`` and floored at ``_MIN_URGENCY``.
    """
    if weeks_remaining <= 0:
        # No time left to course-correct -- equal weights
        cats = set(ytd_totals.keys()) | set(target_totals.keys())
        if not cats:
            return {}
        return {cat: 1.0 for cat in cats}

    raw_urgency: dict[str, float] = {}

    # Use the union of categories present in any input
    all_cats = set(ytd_totals.keys()) | set(target_totals.keys())

    for cat in all_cats:
        ytd = ytd_totals.get(cat, 0.0)
        target = target_totals.get(cat, 0.0)
        rate = weekly_rates.get(cat, 0.0)

        remaining_available = rate * weeks_remaining

        if cat in INVERSE_CATS:
            # For ERA/WHIP: lower is better.
            # "remaining_needed" is how much you need your stat to DROP
            # (or not rise).
            # If ytd ERA > target ERA, you are BEHIND (urgency > 1).
            # If ytd ERA < target ERA, you are AHEAD (urgency < 1).
            remaining_needed = ytd - target  # positive = behind pace

            if abs(remaining_available) < _EPSILON:
                # No rate info; use a heuristic based on whether we are
                # behind or ahead.
                if remaining_needed > _EPSILON:
                    raw_urgency[cat] = _MAX_URGENCY
                else:
                    raw_urgency[cat] = _MIN_URGENCY
            elif remaining_needed > _EPSILON:
                # Behind pace — proportional urgency same as normal cats.
                urgency = remaining_needed / max(abs(remaining_available), _EPSILON)
                raw_urgency[cat] = urgency
            else:
                # For inverse stats, "available" is how much the stat
                # might move (weekly rate * weeks).  Higher urgency when
                # the remaining gap is a large fraction of what is left.
                urgency = remaining_needed / max(abs(remaining_available), _EPSILON)
                raw_urgency[cat] = urgency
        else:
            # Standard categories (R, HR, RBI, SB, AVG, W, SV, K):
            # higher is better.
            remaining_needed = target - ytd  # positive = behind pace

            if abs(remaining_available) < _EPSILON:
                if remaining_needed > _EPSILON:
                    raw_urgency[cat] = _MAX_URGENCY
                elif remaining_needed < -_EPSILON:
                    raw_urgency[cat] = _MIN_URGENCY
                else:
                    raw_urgency[cat] = 1.0
            elif remaining_available < 0 and remaining_needed > _EPSILON:
                # Behind pace AND rate is declining — maximum urgency
                raw_urgency[cat] = _MAX_URGENCY
            elif remaining_available < 0 and remaining_needed < -_EPSILON:
                # Ahead of pace AND rate is declining — moderate urgency
                raw_urgency[cat] = 1.0
            else:
                urgency = remaining_needed / remaining_available
                raw_urgency[cat] = urgency

    if not raw_urgency:
        return {}

    # Clamp to [_MIN_URGENCY, _MAX_URGENCY]
    clamped = {cat: max(_MIN_URGENCY, min(_MAX_URGENCY, u)) for cat, u in raw_urgency.items()}

    # Normalize so mean = 1.0
    vals = list(clamped.values())
    mean_val = sum(vals) / len(vals)
    if mean_val < _EPSILON:
        return {cat: 1.0 for cat in clamped}

    return {cat: v / mean_val for cat, v in clamped.items()}
