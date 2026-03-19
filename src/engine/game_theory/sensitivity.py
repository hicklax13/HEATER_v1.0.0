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

logger = logging.getLogger(__name__)

CATEGORIES: list[str] = [
    "R",
    "HR",
    "RBI",
    "SB",
    "AVG",
    "OBP",
    "W",
    "L",
    "K",
    "SV",
    "ERA",
    "WHIP",
]
INVERSE_CATEGORIES: set[str] = {"L", "ERA", "WHIP"}

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


def player_sensitivity(
    giving_ids: list[int],
    receiving_ids: list[int],
    user_roster_ids: list[int],
    player_pool,
    evaluate_fn,
    base_surplus: float,
    config=None,
    user_team_name: str | None = None,
    weeks_remaining: int = 16,
) -> list[dict]:
    """Measure each player's marginal contribution to the trade.

    For each player in the trade, temporarily removes them and
    re-evaluates. The difference shows that player's impact.

    Args:
        giving_ids: Players being given away.
        receiving_ids: Players being received.
        user_roster_ids: Current roster IDs.
        player_pool: Full player pool DataFrame.
        evaluate_fn: The evaluate_trade function to call.
        base_surplus: The original trade's surplus SGP.
        config: League configuration.
        user_team_name: Team name.
        weeks_remaining: Weeks remaining.

    Returns:
        List of dicts sorted by absolute impact:
          [{player_id, name, side, marginal_impact, makes_trade}, ...]
        where side is "giving" or "receiving" and makes_trade is
        "better" or "worse" when removed.
    """
    name_col = "player_name" if "player_name" in player_pool.columns else "name"
    results = []

    # Test removing each player from the giving side
    for pid in giving_ids:
        if len(giving_ids) <= 1:
            # Can't have a trade with 0 players given
            continue

        reduced_giving = [g for g in giving_ids if g != pid]
        try:
            alt_result = evaluate_fn(
                giving_ids=reduced_giving,
                receiving_ids=receiving_ids,
                user_roster_ids=user_roster_ids,
                player_pool=player_pool,
                config=config,
                user_team_name=user_team_name,
                weeks_remaining=weeks_remaining,
                enable_mc=False,
                enable_context=False,
                enable_game_theory=False,
            )
            alt_surplus = alt_result.get("surplus_sgp", 0.0)
        except Exception:
            continue

        marginal = alt_surplus - base_surplus

        match = player_pool[player_pool["player_id"] == pid]
        pname = str(match.iloc[0][name_col]) if not match.empty else f"ID:{pid}"

        results.append(
            {
                "player_id": pid,
                "name": pname,
                "side": "giving",
                "marginal_impact": round(marginal, 3),
                "makes_trade": "better" if marginal > 0 else "worse",
            }
        )

    # Test removing each player from the receiving side
    for pid in receiving_ids:
        if len(receiving_ids) <= 1:
            continue

        reduced_receiving = [r for r in receiving_ids if r != pid]
        try:
            alt_result = evaluate_fn(
                giving_ids=giving_ids,
                receiving_ids=reduced_receiving,
                user_roster_ids=user_roster_ids,
                player_pool=player_pool,
                config=config,
                user_team_name=user_team_name,
                weeks_remaining=weeks_remaining,
                enable_mc=False,
                enable_context=False,
                enable_game_theory=False,
            )
            alt_surplus = alt_result.get("surplus_sgp", 0.0)
        except Exception:
            continue

        marginal = alt_surplus - base_surplus

        match = player_pool[player_pool["player_id"] == pid]
        pname = str(match.iloc[0][name_col]) if not match.empty else f"ID:{pid}"

        results.append(
            {
                "player_id": pid,
                "name": pname,
                "side": "receiving",
                "marginal_impact": round(marginal, 3),
                "makes_trade": "better" if marginal > 0 else "worse",
            }
        )

    results.sort(key=lambda x: abs(x["marginal_impact"]), reverse=True)
    return results


def suggest_counter_offers(
    giving_ids: list[int],
    receiving_ids: list[int],
    user_roster_ids: list[int],
    player_pool,
    evaluate_fn,
    base_surplus: float,
    config=None,
    user_team_name: str | None = None,
    weeks_remaining: int = 16,
    max_suggestions: int = 3,
) -> list[dict]:
    """Generate counter-offer suggestions to improve a trade.

    Tries swapping each player on the giving side with other roster
    players to find trades with better surplus. Only suggests swaps
    that improve the trade by at least MIN_SWAP_IMPROVEMENT.

    Args:
        giving_ids: Players being given away.
        receiving_ids: Players being received.
        user_roster_ids: Current roster IDs.
        player_pool: Full player pool DataFrame.
        evaluate_fn: The evaluate_trade function.
        base_surplus: Original trade surplus.
        config: League configuration.
        user_team_name: Team name.
        weeks_remaining: Weeks remaining.
        max_suggestions: Maximum number of counter-offers to return.

    Returns:
        List of counter-offer dicts sorted by improvement:
          [{swap_out, swap_out_name, swap_in, swap_in_name,
            new_surplus, improvement, new_grade}, ...]
    """
    name_col = "player_name" if "player_name" in player_pool.columns else "name"
    suggestions = []

    # Get roster players NOT in the current trade
    available_swaps = [pid for pid in user_roster_ids if pid not in giving_ids and pid not in receiving_ids]

    # For each player being given, try swapping with a roster player
    for give_pid in giving_ids:
        for swap_pid in available_swaps:
            new_giving = [swap_pid if g == give_pid else g for g in giving_ids]

            try:
                alt_result = evaluate_fn(
                    giving_ids=new_giving,
                    receiving_ids=receiving_ids,
                    user_roster_ids=user_roster_ids,
                    player_pool=player_pool,
                    config=config,
                    user_team_name=user_team_name,
                    weeks_remaining=weeks_remaining,
                    enable_mc=False,
                    enable_context=False,
                    enable_game_theory=False,
                )
                new_surplus = alt_result.get("surplus_sgp", 0.0)
            except Exception:
                continue

            improvement = new_surplus - base_surplus

            if improvement >= MIN_SWAP_IMPROVEMENT:
                give_match = player_pool[player_pool["player_id"] == give_pid]
                swap_match = player_pool[player_pool["player_id"] == swap_pid]
                give_name = str(give_match.iloc[0][name_col]) if not give_match.empty else f"ID:{give_pid}"
                swap_name = str(swap_match.iloc[0][name_col]) if not swap_match.empty else f"ID:{swap_pid}"

                suggestions.append(
                    {
                        "swap_out": give_pid,
                        "swap_out_name": give_name,
                        "swap_in": swap_pid,
                        "swap_in_name": swap_name,
                        "new_surplus": round(new_surplus, 3),
                        "improvement": round(improvement, 3),
                        "new_grade": alt_result.get("grade", "?"),
                    }
                )

    # Sort by improvement (best first) and limit
    suggestions.sort(key=lambda x: x["improvement"], reverse=True)
    return suggestions[:max_suggestions]


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
