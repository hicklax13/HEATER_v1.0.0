"""Opponent Trade Analysis — models trades from the opponent's perspective.

Reuses category_gap_analysis() to compute opponent's weak/strong categories,
then evaluates whether a proposed trade fills their gaps. Combines with
opponent profiles (archetypes) and ADP fairness to predict acceptance.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def compute_opponent_needs(
    opponent_team_name: str,
    all_team_totals: dict[str, dict[str, float]],
    weeks_remaining: int = 16,
) -> dict[str, dict]:
    """Run category_gap_analysis from the opponent's perspective.

    Returns same structure as category_gap_analysis(): per-category
    rank, is_punt, marginal_value, gap_to_next, gainable_positions.

    Note: config is not accepted here because category_gap_analysis()
    uses its own module-level LeagueConfig internally.
    """

    opp_totals = all_team_totals.get(opponent_team_name, {})
    if not opp_totals:
        return {}

    try:
        from src.engine.portfolio.category_analysis import category_gap_analysis

        return category_gap_analysis(
            your_totals=opp_totals,
            all_team_totals=all_team_totals,
            your_team_id=opponent_team_name,
            weeks_remaining=weeks_remaining,
        )
    except Exception:
        logger.debug(
            "Could not compute opponent needs for %s",
            opponent_team_name,
            exc_info=True,
        )
        return {}


def analyze_from_opponent_view(
    trade: dict,
    opponent_team_name: str,
    all_team_totals: dict[str, dict[str, float]],
    opponent_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config=None,
) -> dict:
    """Compute per-category impact of a trade from the opponent's view.

    Returns dict with:
        opp_category_deltas: {cat: sgp_change}
        opp_weak_cats_helped: list of weak cats improved
        opp_strong_cats_hurt: list of strong cats damaged
        opp_need_match: float 0-1 (how well trade fills their gaps)
    """
    if config is None:
        from src.valuation import LeagueConfig

        config = LeagueConfig()

    from src.in_season import _roster_category_totals

    # Get opponent's current needs
    opp_needs = compute_opponent_needs(opponent_team_name, all_team_totals)
    if not opp_needs:
        return {
            "opp_category_deltas": {},
            "opp_weak_cats_helped": [],
            "opp_strong_cats_hurt": [],
            "opp_need_match": 0.5,
        }

    # Compute pre/post trade totals for opponent
    giving_ids = trade.get("giving_ids", [])
    receiving_ids = trade.get("receiving_ids", [])

    # Opponent gives receiving_ids, gets giving_ids (inverse of user)
    opp_before = _roster_category_totals(opponent_roster_ids, player_pool)
    opp_after_ids = [pid for pid in opponent_roster_ids if pid not in receiving_ids] + list(giving_ids)
    opp_after = _roster_category_totals(opp_after_ids, player_pool)

    # Per-category deltas
    deltas = {}
    weak_helped = []
    strong_hurt = []
    for cat in config.all_categories:
        before_val = opp_before.get(cat, 0)
        after_val = opp_after.get(cat, 0)
        raw_delta = after_val - before_val
        improvement = -raw_delta if cat in config.inverse_stats else raw_delta
        deltas[cat] = round(raw_delta, 3)  # raw stat change

        # Check if this helps a weak category
        cat_info = opp_needs.get(cat, {})
        rank = cat_info.get("rank", 6)
        if rank >= 8 and improvement > 0:
            weak_helped.append(cat)
        elif rank <= 4 and improvement < 0:
            strong_hurt.append(cat)

    # Need match: fraction of opponent's weak categories that improve
    total_weak = sum(1 for info in opp_needs.values() if info.get("rank", 6) >= 8)
    need_match = len(weak_helped) / max(total_weak, 1)

    return {
        "opp_category_deltas": deltas,
        "opp_weak_cats_helped": weak_helped,
        "opp_strong_cats_hurt": strong_hurt,
        "opp_need_match": round(need_match, 3),
    }


def get_opponent_archetype(
    opponent_team_name: str,
) -> dict:
    """Get opponent's archetype with trade willingness from profiles.

    Returns dict with tier, threat, trade_willingness (0-1).
    """
    try:
        from src.opponent_intel import OPPONENT_PROFILES

        profile = OPPONENT_PROFILES.get(opponent_team_name, {})
    except ImportError:
        profile = {}

    tier = profile.get("tier", 3)
    # Map tier to trade willingness
    # Tier 1 = active skilled managers (willing to trade smartly)
    # Tier 4 = passive/inactive (rarely trades)
    willingness_map = {1: 0.7, 2: 0.5, 3: 0.4, 4: 0.3}
    trade_willingness = willingness_map.get(tier, 0.4)

    return {
        "tier": tier,
        "threat": profile.get("threat", "Unknown"),
        "manager": profile.get("manager", "Unknown"),
        "trade_willingness": trade_willingness,
        "strengths": profile.get("strengths", []),
        "weaknesses": profile.get("weaknesses", []),
    }
