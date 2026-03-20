"""Opponent valuation and market clearing price.

Spec reference: Section 11 L8A (Game Theory Layer)
               Section 17 Phase 5 item 21

Estimates each opponent's willingness-to-pay for a given player based
on THEIR category needs, then derives the Nash equilibrium price
(second-highest bidder) as the market clearing value.

The key insight: a player's trade value is NOT their raw SGP. It's the
maximum another manager would pay — which depends on THEIR category
gaps, not yours. A closer with 30 projected saves is worth more to
the team ranked 11th in SV (desperate) than to the team ranked 1st.

Wires into:
  - src/engine/portfolio/category_analysis.py: compute_marginal_sgp
  - src/engine/output/trade_evaluator.py: market context for trades
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

# Default SGP denominators for when standings unavailable
DEFAULT_SGP_DENOMS: dict[str, float] = {
    "R": 30.0,
    "HR": 12.0,
    "RBI": 30.0,
    "SB": 8.0,
    "AVG": 0.005,
    "OBP": 0.005,
    "W": 3.0,
    "L": 3.0,
    "K": 25.0,
    "SV": 5.0,
    "ERA": 0.20,
    "WHIP": 0.015,
}

# Stat column mapping (uppercase category → lowercase pool column)
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


def estimate_opponent_valuations(
    player_projections: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    your_team_id: str,
    sgp_denominators: dict[str, float] | None = None,
) -> dict[str, float]:
    """Estimate each opponent's willingness-to-pay for a player.

    Spec ref: Section 11 L8A — opponent valuations.

    For each opponent team, computes marginal SGP gain from adding
    the player's projected stats. Teams with bigger category gaps
    value the player less; teams near a standings jump value the
    player more.

    Args:
        player_projections: Player's projected stat totals, keyed by
            category name (e.g., {"R": 85, "HR": 30, ...}).
        all_team_totals: Dict of {team_name: {category: total}}.
        your_team_id: Your team name (excluded from valuations).
        sgp_denominators: SGP conversion denominators. If None, uses defaults.

    Returns:
        Dict of {team_name: valuation_sgp} for all opponents.
    """
    denoms = sgp_denominators or DEFAULT_SGP_DENOMS
    valuations: dict[str, float] = {}

    for team_id, team_totals in all_team_totals.items():
        if team_id == your_team_id:
            continue

        # Compute this opponent's marginal value for the player
        team_val = 0.0
        for cat in CATEGORIES:
            proj = player_projections.get(cat, 0.0)
            if abs(proj) < 1e-9:
                continue

            denom = denoms.get(cat, 1.0)
            if abs(denom) < 1e-9:
                denom = 1.0

            # Compute marginal value: how many standings points
            # could this team gain from +proj in this category?
            team_cat_total = team_totals.get(cat, 0.0)

            # Count how many teams they'd overtake
            gap_to_next = _gap_to_next_team(team_cat_total, cat, team_id, all_team_totals)

            if gap_to_next > 0:
                if cat in INVERSE_CATEGORIES and cat in ("ERA", "WHIP"):
                    # Rate stat: need IP-weighted blend to compute benefit
                    player_ip = player_projections.get("ip", player_projections.get("IP", 0))
                    team_ip = team_totals.get("ip", team_totals.get("IP", 0))
                    # Default IP when not provided: 150 for player, 1200 for team
                    if player_ip <= 0 and proj > 0:
                        player_ip = 150.0
                    if team_ip <= 0 and team_cat_total > 0:
                        team_ip = 1200.0
                    if player_ip > 0 and team_ip > 0:
                        blended_rate = (team_cat_total * team_ip + proj * player_ip) / (team_ip + player_ip)
                        benefit = max(0.0, team_cat_total - blended_rate)
                    else:
                        benefit = 0.0
                    marginal = min(benefit / gap_to_next, 1.0) if gap_to_next > 0 else 0.0
                    team_val += marginal * min(benefit, gap_to_next) / denom
                elif cat in INVERSE_CATEGORIES:
                    # L: inverse counting stat — more losses always hurts.
                    # Acquiring a pitcher's projected losses provides no benefit.
                    continue
                else:
                    # Counting stats: projected contribution / gap
                    marginal = min(proj / gap_to_next, 1.0)
                    team_val += marginal * (proj / denom)
            else:
                if cat in INVERSE_CATEGORIES and cat in ("ERA", "WHIP"):
                    # Rate stat: IP-weighted blend even when already best
                    player_ip = player_projections.get("ip", player_projections.get("IP", 0))
                    team_ip = team_totals.get("ip", team_totals.get("IP", 0))
                    if player_ip <= 0 and proj > 0:
                        player_ip = 150.0
                    if team_ip <= 0 and team_cat_total > 0:
                        team_ip = 1200.0
                    if player_ip > 0 and team_ip > 0:
                        blended_rate = (team_cat_total * team_ip + proj * player_ip) / (team_ip + player_ip)
                        benefit = max(0.0, team_cat_total - blended_rate)
                    else:
                        benefit = 0.0
                    team_val += benefit / denom * 0.5
                elif cat in INVERSE_CATEGORIES:
                    # L: inverse counting stat — more losses always hurts.
                    # Acquiring a pitcher's projected losses provides no benefit.
                    continue
                else:
                    # Team is already last; any help has marginal value
                    team_val += proj / denom * 0.5

        valuations[team_id] = round(team_val, 3)

    return valuations


def _gap_to_next_team(
    team_total: float,
    category: str,
    team_id: str,
    all_team_totals: dict[str, dict[str, float]],
) -> float:
    """Find the gap to the next team above in standings for a category.

    For counting stats (R, HR, etc.), "above" means the team with more.
    For inverse stats (ERA, WHIP), "above" means the team with less.

    Args:
        team_total: This team's category total.
        category: Category name.
        team_id: This team's ID (excluded from comparison).
        all_team_totals: All team totals.

    Returns:
        Gap to next better team (positive = catchable). 0 if already best.
    """
    others = [totals.get(category, 0.0) for tid, totals in all_team_totals.items() if tid != team_id]

    if not others:
        return 0.0

    if category in INVERSE_CATEGORIES:
        # Lower is better — find teams with lower (better) values
        better = [v for v in others if v < team_total]
        if not better:
            return 0.0
        closest_better = max(better)  # Closest below
        return abs(team_total - closest_better)
    else:
        # Higher is better — find teams with higher values
        better = [v for v in others if v > team_total]
        if not better:
            return 0.0
        closest_better = min(better)  # Closest above
        return abs(closest_better - team_total)


def market_clearing_price(valuations: dict[str, float]) -> float:
    """Compute Nash equilibrium price = second-highest bidder.

    Spec ref: Section 11 L8A — market clearing price.

    In a competitive trade market, the fair price is set by the
    second-highest bidder. The top bidder wins but pays only what
    the second bidder would have offered — classic Vickrey auction.

    Args:
        valuations: Dict of {team_name: valuation_sgp}.

    Returns:
        Market clearing price in SGP. Zero if fewer than 2 bidders.
    """
    if not valuations:
        return 0.0

    sorted_vals = sorted(valuations.values(), reverse=True)

    if len(sorted_vals) >= 2:
        return sorted_vals[1]
    return sorted_vals[0]


def player_market_value(
    player_projections: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    your_team_id: str,
    sgp_denominators: dict[str, float] | None = None,
) -> dict[str, float | dict]:
    """Full market analysis for a single player.

    Combines opponent valuations with market clearing price and
    demand analysis.

    Args:
        player_projections: Player's projected stats.
        all_team_totals: All team totals.
        your_team_id: Your team name.
        sgp_denominators: SGP denominators.

    Returns:
        Dict with:
          - valuations: per-team valuations
          - market_price: Nash equilibrium (2nd highest)
          - max_bidder: team with highest valuation
          - max_bid: highest valuation
          - demand: number of teams that value player > 0.5 SGP
    """
    valuations = estimate_opponent_valuations(
        player_projections=player_projections,
        all_team_totals=all_team_totals,
        your_team_id=your_team_id,
        sgp_denominators=sgp_denominators,
    )

    market_price = market_clearing_price(valuations)

    # Find top bidder
    max_bidder = ""
    max_bid = 0.0
    if valuations:
        max_bidder = max(valuations, key=valuations.get)  # type: ignore[arg-type]
        max_bid = valuations[max_bidder]

    # Count demand (teams valuing player above threshold)
    demand = sum(1 for v in valuations.values() if v > 0.5)

    return {
        "valuations": valuations,
        "market_price": round(market_price, 3),
        "max_bidder": max_bidder,
        "max_bid": round(max_bid, 3),
        "demand": demand,
    }


def get_player_projections_from_pool(
    player_id: int,
    player_pool,
) -> dict[str, float]:
    """Extract a player's projections in category format from the pool.

    Convenience wrapper for building the projections dict that
    estimate_opponent_valuations expects.

    Args:
        player_id: Player ID to look up.
        player_pool: Player pool DataFrame.

    Returns:
        Dict of {category: projected_value}.
    """
    match = player_pool[player_pool["player_id"] == player_id]
    if match.empty:
        return {}

    row = match.iloc[0]
    projections: dict[str, float] = {}

    for cat, col in STAT_MAP.items():
        val = row.get(col, 0.0)
        try:
            projections[cat] = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            projections[cat] = 0.0

    return projections
