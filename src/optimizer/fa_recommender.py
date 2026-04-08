"""Post-optimization FA add/drop recommendation engine.

After the Line-up Optimizer runs (LP or DCV), this module evaluates whether
any free agent swaps would improve the roster given the current matchup state.
Enforces AVIS league rules: weekly add budget, closer minimum, IL stash
protection, and category worsening limits.

Usage:
    ctx = build_optimizer_context("rest_of_week", yds, config)
    moves = recommend_fa_moves(ctx, max_moves=3)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.optimizer.shared_data_layer import OptimizerDataContext
from src.waiver_wire import (
    compute_drop_cost,
    compute_net_swap_value,
    compute_sustainability_score,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_CLOSERS = 2
_CLOSER_SV_THRESHOLD = 5
_MAX_DROP_CANDIDATES = 5
_MAX_FA_CANDIDATES = 10
_CATEGORY_WORSEN_THRESHOLD = -0.1
_MAX_WORSENED_CATEGORIES = 3
_HITTER_SLOTS = 10  # C/1B/2B/3B/SS/3OF/2Util = 10 starting hitter slots
_CROSS_TYPE_SGP_MIN = 0.3
_OWNERSHIP_BOOST_DELTA = 5.0
_OWNERSHIP_BOOST_MULT = 1.10
_FLOOR_PENALTY_MULT = 0.85
_FLOOR_PA_MIN = 50
_FLOOR_IP_MIN = 20
_IL_EXCLUDE_STATUSES = {"il", "il10", "il15", "il60", "dtd", "day-to-day", "na", "out", "suspended"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def recommend_fa_moves(
    ctx: OptimizerDataContext,
    max_moves: int = 3,
) -> list[dict[str, Any]]:
    """Recommend FA add/drop swaps that improve the roster.

    Parameters
    ----------
    ctx : OptimizerDataContext
        Fully-built optimizer data context from the shared data layer.
    max_moves : int
        Maximum number of moves to recommend.

    Returns
    -------
    list[dict]
        Recommended swaps sorted by net SGP delta descending.
        Each dict contains add/drop info, category impact, reasoning, etc.
    """
    # ── AVIS budget check ────────────────────────────────────────────
    if ctx.adds_remaining_this_week <= 0:
        logger.info("No adds remaining this week — skipping FA recommendations")
        return []

    if not ctx.user_roster_ids or ctx.player_pool.empty:
        return []

    effective_max = min(max_moves, ctx.adds_remaining_this_week)

    # ── Step 1: Score drop candidates ────────────────────────────────
    drop_candidates = _score_drop_candidates(ctx)
    if not drop_candidates:
        logger.info("No viable drop candidates found")
        return []

    # ── Step 2: Score FA candidates ──────────────────────────────────
    fa_candidates = _score_fa_candidates(ctx)
    if not fa_candidates:
        logger.info("No viable FA candidates found")
        return []

    # ── Step 3: Evaluate all (drop, add) pairs ───────────────────────
    swap_results = _evaluate_swaps(ctx, drop_candidates, fa_candidates)

    # ── Step 4: Deduplicate and limit ────────────────────────────────
    final = _deduplicate_and_limit(swap_results, effective_max)

    return final


# ---------------------------------------------------------------------------
# Drop candidate scoring
# ---------------------------------------------------------------------------


def _score_drop_candidates(ctx: OptimizerDataContext) -> list[dict]:
    """Score rostered players as drop candidates.

    Hard filters:
    - Never drop IL stash players.
    - Never drop a closer if it would reduce closer count below minimum.
    """
    candidates: list[dict] = []

    for pid in ctx.user_roster_ids:
        # Hard filter: IL stash protection
        if pid in ctx.il_stash_ids:
            continue

        # Hard filter: closer minimum
        if _is_closer(pid, ctx) and ctx.closer_count <= _MIN_CLOSERS:
            continue

        cost = compute_drop_cost(pid, ctx.user_roster_ids, ctx.player_pool, ctx.config)

        # Look up player info
        match = ctx.player_pool[ctx.player_pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]

        candidates.append(
            {
                "player_id": pid,
                "name": str(row.get("name", row.get("player_name", "?"))),
                "positions": str(row.get("positions", "")),
                "is_hitter": bool(int(row.get("is_hitter", 1))),
                "drop_cost": cost,
            }
        )

    # Sort by cost ascending (cheapest to drop first), take top N
    candidates.sort(key=lambda x: x["drop_cost"])
    return candidates[:_MAX_DROP_CANDIDATES]


# ---------------------------------------------------------------------------
# FA candidate scoring
# ---------------------------------------------------------------------------


def _score_fa_candidates(ctx: OptimizerDataContext) -> list[dict]:
    """Score free agents by composite value.

    Filters out unhealthy players (health_score < 0.65 or IL/DTD/NA status).
    Scores by: base value, urgency boost, ownership trend, sustainability,
    and floor preference.
    """
    if ctx.free_agents.empty:
        return []

    candidates: list[dict] = []

    for _, fa_row in ctx.free_agents.iterrows():
        fa_id = fa_row.get("player_id")
        if fa_id is None:
            continue
        fa_id = int(fa_id)

        # Health filter: exclude injured players
        health = ctx.health_scores.get(fa_id, 1.0)
        if health < 0.65:
            continue

        # Status filter
        status = str(fa_row.get("status", "")).lower().strip()
        if status in _IL_EXCLUDE_STATUSES:
            continue

        # Also check player_pool for status
        pool_match = ctx.player_pool[ctx.player_pool["player_id"] == fa_id]
        if not pool_match.empty:
            pool_status = str(pool_match.iloc[0].get("status", "")).lower().strip()
            if pool_status in _IL_EXCLUDE_STATUSES:
                continue
            fa_data = pool_match.iloc[0]
        else:
            fa_data = fa_row

        # Base value
        base_value = _compute_base_value(fa_data, ctx)

        # Urgency boost: sum category weights for categories this FA contributes to
        urgency_boost = _compute_urgency_boost(fa_data, ctx)

        # Ownership trend boost
        ownership_mult = 1.0
        trend = ctx.ownership_trends.get(fa_id, {})
        delta_7d = trend.get("delta_7d", 0.0)
        if delta_7d > _OWNERSHIP_BOOST_DELTA:
            ownership_mult = _OWNERSHIP_BOOST_MULT

        # Sustainability
        sustainability = compute_sustainability_score(fa_data)

        # Floor preference penalty
        floor_mult = 1.0
        is_hitter = bool(int(fa_data.get("is_hitter", 1)))
        if is_hitter:
            pa = float(fa_data.get("pa", 0) or 0)
            if pa < _FLOOR_PA_MIN:
                floor_mult = _FLOOR_PENALTY_MULT
        else:
            ip = float(fa_data.get("ip", 0) or 0)
            if ip < _FLOOR_IP_MIN:
                floor_mult = _FLOOR_PENALTY_MULT

        composite = base_value * sustainability * ownership_mult * floor_mult + urgency_boost

        # Ownership trend label
        pct_owned = trend.get("pct_owned", 0.0)
        if delta_7d > _OWNERSHIP_BOOST_DELTA:
            trend_label = f"Rising ({pct_owned:.0f}%, +{delta_7d:.1f}%)"
        elif delta_7d < -_OWNERSHIP_BOOST_DELTA:
            trend_label = f"Falling ({pct_owned:.0f}%, {delta_7d:.1f}%)"
        else:
            trend_label = f"Stable ({pct_owned:.0f}%)"

        candidates.append(
            {
                "player_id": fa_id,
                "name": str(fa_data.get("name", fa_data.get("player_name", "?"))),
                "positions": str(fa_data.get("positions", "")),
                "is_hitter": is_hitter,
                "composite_score": composite,
                "sustainability": round(sustainability, 3),
                "ownership_trend": trend_label,
                "ownership_delta_7d": delta_7d,
            }
        )

    # Sort by composite descending, take top N
    candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    return candidates[:_MAX_FA_CANDIDATES]


# ---------------------------------------------------------------------------
# Swap evaluation
# ---------------------------------------------------------------------------


def _evaluate_swaps(
    ctx: OptimizerDataContext,
    drop_candidates: list[dict],
    fa_candidates: list[dict],
) -> list[dict]:
    """Evaluate all (drop, add) pairs and filter by AVIS rules."""
    results: list[dict] = []
    losing_cats = _get_losing_categories(ctx)
    tied_cats = _get_tied_categories(ctx)
    target_cats = set(losing_cats) | set(tied_cats)

    for fa in fa_candidates:
        fa_id = fa["player_id"]
        fa_is_hitter = fa["is_hitter"]

        for drop in drop_candidates:
            drop_id = drop["player_id"]
            drop_is_hitter = drop["is_hitter"]

            # Same-type check (cross-type only when surplus + improves losing/tied cat)
            if fa_is_hitter != drop_is_hitter:
                if not _allow_cross_type(ctx, fa, drop, target_cats):
                    continue

            # Compute net swap value
            swap = compute_net_swap_value(fa_id, drop_id, ctx.user_roster_ids, ctx.player_pool, ctx.config)

            net_sgp = swap["net_sgp"]
            cat_deltas = swap["category_deltas"]

            # Skip non-positive swaps
            if net_sgp <= 0:
                continue

            # AVIS Rule #3: reject if 3+ categories worsen
            worsened = sum(1 for v in cat_deltas.values() if v < _CATEGORY_WORSEN_THRESHOLD)
            if worsened >= _MAX_WORSENED_CATEGORIES:
                continue

            # Build reasoning
            reasoning = _build_reasoning(fa, drop, swap, ctx)

            # Urgency categories: which losing/tied cats does this swap help?
            urgency_cats = [cat for cat in target_cats if cat_deltas.get(cat, 0) > 0.01]

            # News warning
            news_warning = ctx.news_flags.get(fa_id)

            results.append(
                {
                    "add_id": fa_id,
                    "add_name": fa["name"],
                    "add_positions": fa["positions"],
                    "drop_id": drop_id,
                    "drop_name": drop["name"],
                    "drop_positions": drop["positions"],
                    "net_sgp_delta": round(net_sgp, 4),
                    "category_impact": {k: round(v, 4) for k, v in cat_deltas.items()},
                    "reasoning": reasoning,
                    "urgency_categories": urgency_cats,
                    "news_warning": news_warning,
                    "ownership_trend": fa["ownership_trend"],
                    "sustainability": fa["sustainability"],
                }
            )

    # Sort by net SGP descending
    results.sort(key=lambda x: x["net_sgp_delta"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Deduplication and limiting
# ---------------------------------------------------------------------------


def _deduplicate_and_limit(results: list[dict], max_moves: int) -> list[dict]:
    """Deduplicate so each FA and each drop candidate is used at most once."""
    used_adds: set[int] = set()
    used_drops: set[int] = set()
    final: list[dict] = []

    for swap in results:
        if len(final) >= max_moves:
            break
        add_id = swap["add_id"]
        drop_id = swap["drop_id"]
        if add_id in used_adds or drop_id in used_drops:
            continue
        used_adds.add(add_id)
        used_drops.add(drop_id)
        final.append(swap)

    return final


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_closer(player_id: int, ctx: OptimizerDataContext) -> bool:
    """Check if a player is a closer (is_closer flag or projected SV >= 5)."""
    if ctx.player_pool.empty:
        return False
    match = ctx.player_pool[ctx.player_pool["player_id"] == player_id]
    if match.empty:
        return False
    row = match.iloc[0]
    if row.get("is_closer", False):
        return True
    sv = float(row.get("sv", 0) or 0)
    return sv >= _CLOSER_SV_THRESHOLD


def _compute_base_value(fa_data: pd.Series, ctx: OptimizerDataContext) -> float:
    """Compute base value for an FA candidate.

    Uses marginal_value column if present, otherwise sums counting stat
    projections weighted by category weights.
    """
    if "marginal_value" in fa_data.index and pd.notna(fa_data.get("marginal_value")):
        return float(fa_data["marginal_value"])

    # Fallback: weighted sum of counting stats
    value = 0.0
    stat_map = ctx.config.STAT_MAP
    for cat_display, col in stat_map.items():
        if col in fa_data.index:
            stat_val = float(fa_data.get(col, 0) or 0)
            weight = ctx.category_weights.get(cat_display, ctx.category_weights.get(cat_display.lower(), 1.0))
            denom = ctx.config.sgp_denominators.get(cat_display, 1.0)
            if abs(denom) < 1e-9:
                denom = 1.0
            value += (stat_val / denom) * weight
    return value


def _compute_urgency_boost(fa_data: pd.Series, ctx: OptimizerDataContext) -> float:
    """Sum category weights for categories this FA contributes to."""
    boost = 0.0
    is_hitter = bool(int(fa_data.get("is_hitter", 1)))
    stat_map = ctx.config.STAT_MAP

    if is_hitter:
        relevant_cats = ctx.config.hitting_categories
    else:
        relevant_cats = ctx.config.pitching_categories

    for cat in relevant_cats:
        col = stat_map.get(cat, cat.lower())
        if col in fa_data.index:
            stat_val = float(fa_data.get(col, 0) or 0)
            if abs(stat_val) > 0.001:
                weight = ctx.category_weights.get(cat, ctx.category_weights.get(cat.lower(), 1.0))
                boost += weight

    return boost


def _get_losing_categories(ctx: OptimizerDataContext) -> list[str]:
    """Get list of categories the user is currently losing."""
    summary = ctx.urgency_weights.get("summary", {})
    return list(summary.get("losing", []))


def _get_tied_categories(ctx: OptimizerDataContext) -> list[str]:
    """Get list of categories the user is currently tied in."""
    summary = ctx.urgency_weights.get("summary", {})
    return list(summary.get("tied", []))


def _allow_cross_type(
    ctx: OptimizerDataContext,
    fa: dict,
    drop: dict,
    target_cats: set[str],
) -> bool:
    """Allow cross-type swap only when user has positional surplus AND
    the swap improves a losing/tied category by >= 0.3 SGP.
    """
    # Check positional surplus
    hitter_count = 0
    for pid in ctx.user_roster_ids:
        match = ctx.player_pool[ctx.player_pool["player_id"] == pid]
        if not match.empty and bool(int(match.iloc[0].get("is_hitter", 1))):
            hitter_count += 1

    fa_is_hitter = fa["is_hitter"]

    # If adding a hitter and dropping a pitcher, need pitcher surplus (hitters < slots)
    # If adding a pitcher and dropping a hitter, need hitter surplus
    if fa_is_hitter and hitter_count >= _HITTER_SLOTS:
        # Already at or above hitter capacity, no room for another hitter
        return False
    if not fa_is_hitter and hitter_count <= _HITTER_SLOTS:
        # No hitter surplus — can't drop a hitter for a pitcher
        return False

    # Check if swap improves a target category by >= 0.3 SGP
    swap = compute_net_swap_value(
        fa["player_id"],
        drop["player_id"],
        ctx.user_roster_ids,
        ctx.player_pool,
        ctx.config,
    )
    cat_deltas = swap["category_deltas"]
    for cat in target_cats:
        delta = cat_deltas.get(cat, 0)
        if delta >= _CROSS_TYPE_SGP_MIN:
            return True

    return False


def _build_reasoning(
    fa: dict,
    drop: dict,
    swap: dict,
    ctx: OptimizerDataContext,
) -> list[str]:
    """Build human-readable reasoning list for the recommendation."""
    reasons: list[str] = []
    cat_deltas = swap["category_deltas"]

    # Best category improvement
    if cat_deltas:
        best_cat = max(cat_deltas, key=lambda c: cat_deltas[c])
        best_val = cat_deltas[best_cat]
        if best_val > 0:
            reasons.append(f"{fa['name']} adds +{best_val:.2f} SGP in {best_cat}")

    # Worst category cost
    if cat_deltas:
        worst_cat = min(cat_deltas, key=lambda c: cat_deltas[c])
        worst_val = cat_deltas[worst_cat]
        if worst_val < _CATEGORY_WORSEN_THRESHOLD:
            reasons.append(f"Costs {worst_val:.2f} SGP in {worst_cat}")

    # Urgency mention
    losing = _get_losing_categories(ctx)
    helped_losing = [c for c in losing if cat_deltas.get(c, 0) > 0.01]
    if helped_losing:
        reasons.append(f"Helps in losing categories: {', '.join(helped_losing)}")

    # Sustainability
    sust = fa["sustainability"]
    if sust < 0.4:
        reasons.append("Caution: current stats may not be sustainable")
    elif sust > 0.7:
        reasons.append("Strong underlying metrics support continued production")

    # Ownership trend
    if fa.get("ownership_delta_7d", 0) > _OWNERSHIP_BOOST_DELTA:
        reasons.append(f"Ownership trending up: {fa['ownership_trend']}")

    # News flag
    news = ctx.news_flags.get(fa["player_id"])
    if news:
        reasons.append(f"Recent news: {news}")

    # Net SGP summary
    reasons.append(f"Net team improvement: +{swap['net_sgp']:.2f} SGP")

    return reasons
