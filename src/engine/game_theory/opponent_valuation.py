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
from typing import TypedDict

from src.valuation import LeagueConfig as _LC_Class

logger = logging.getLogger(__name__)


class PlayerMarketValue(TypedDict):
    """Return shape of :func:`player_market_value`.

    Wave 8c (audit D4D-008): the original return was
    ``dict[str, float | dict]`` — accurate union but consumers had to
    type-narrow before keying. This TypedDict documents the contract.
    """

    valuations: dict[str, float]
    market_price: float
    max_bidder: str
    max_bid: float
    demand: int


# C6/C7 cleanup: module-level _LC singleton AND DEFAULT_SGP_DENOMS fallback
# removed (both were causing stale/wrong denominator reads when callers
# expected live LeagueConfig values). Module constants below capture
# immutable category metadata at import time. Callers MUST now pass
# ``sgp_denominators`` (typically from ``config.sgp_denominators``) — there
# is no silent fallback, so test/production parity is enforced.
CATEGORIES: list[str] = list(_LC_Class().all_categories)
INVERSE_CATEGORIES: set[str] = set(_LC_Class().inverse_stats)

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

# Bug C (2026-05-23 validation): defense-in-depth cap per category.
# A single player cannot realistically contribute more than ~2 SGP to a
# single opponent's category. Any computation producing > MAX_PER_CAT_SGP
# is almost certainly a math bug (the original Bug C produced ~29 SGP per
# rate category for the "already best" path). The cap clamps to a safe
# bound; if it fires, the value flows through but a logger.warning surfaces
# the anomaly for operators.
MAX_PER_CAT_SGP: float = 3.0

# Bug C: roster-baseline defaults from LeagueConfig (CLAUDE.md):
#   AB_0 = 5500, PA_0 = 6100, IP_0 = 1300
# Used when team totals don't include component stats (e.g. standings
# table only has AVG/OBP/ERA/WHIP rates, not the underlying AB/PA/IP).
_DEFAULT_TEAM_AB: float = 5500.0
_DEFAULT_TEAM_PA: float = 6100.0
_DEFAULT_TEAM_IP: float = 1200.0  # existing value preserved for ERA/WHIP back-compat
# Sensible per-player defaults when projection lacks component stats:
_DEFAULT_PLAYER_AB: float = 500.0  # full-season everyday hitter
_DEFAULT_PLAYER_PA: float = 600.0
_DEFAULT_PLAYER_IP: float = 150.0  # existing value preserved


def _rate_stat_marginal_sgp(
    cat: str,
    player_rate: float,
    team_rate: float,
    player_volume: float,
    team_volume: float,
    denom: float,
    gap_to_next: float,
    already_best_mult: float = 0.5,
) -> float:
    """Marginal SGP contribution for a rate-stat addition.

    Computes how a player's rate stat shifts a team's existing rate via
    volume-weighted blending, then converts the shift to SGP units.

    For inverse rates (ERA, WHIP): benefit = team_rate − blended_rate
        (positive when player LOWERS the team's rate, which is good).
    For non-inverse rates (AVG, OBP): benefit = blended_rate − team_rate
        (positive when player RAISES the team's rate, which is good).

    Args:
        cat: Category name (AVG/OBP/ERA/WHIP).
        player_rate: Player's projected rate (e.g. AVG=0.215, ERA=2.50).
        team_rate: Team's current rate.
        player_volume: Player's volume in matching units (AB for AVG,
            PA for OBP, IP for ERA/WHIP).
        team_volume: Team's volume in matching units.
        denom: SGP denominator for the category.
        gap_to_next: Standings gap to the next-ranked team in raw rate
            units. 0 means team is already best.
        already_best_mult: Multiplier applied when gap_to_next == 0
            (default 0.5 for half-marginal value).

    Returns:
        Marginal SGP contribution (clamped to [0, MAX_PER_CAT_SGP]).
    """
    if player_volume <= 0 or team_volume <= 0 or abs(denom) < 1e-9:
        return 0.0

    blended = (team_rate * team_volume + player_rate * player_volume) / (team_volume + player_volume)

    if cat in ("ERA", "WHIP"):
        benefit = team_rate - blended  # lower rate is good
    else:  # AVG, OBP
        benefit = blended - team_rate  # higher rate is good

    if benefit <= 0:
        return 0.0

    if gap_to_next > 0:
        # Bound the benefit by the achievable gap
        raw_sgp = min(benefit, gap_to_next) / denom
    else:
        # Team is already best — half-marginal value
        raw_sgp = (benefit / denom) * already_best_mult

    # Defense-in-depth: clamp to MAX_PER_CAT_SGP. A single player should
    # never plausibly contribute > 3 SGP to a single opponent's category.
    if raw_sgp > MAX_PER_CAT_SGP:
        logger.warning(
            "opponent_valuation: rate-stat SGP %.2f for category %s exceeded "
            "cap %.1f (player_rate=%.3f team_rate=%.3f, volumes=%.1f/%.1f). "
            "Clamping. If this fires often, investigate the upstream rate / "
            "volume data.",
            raw_sgp,
            cat,
            MAX_PER_CAT_SGP,
            player_rate,
            team_rate,
            player_volume,
            team_volume,
        )
        raw_sgp = MAX_PER_CAT_SGP

    return raw_sgp


def estimate_opponent_valuations(
    player_projections: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    your_team_id: str,
    sgp_denominators: dict[str, float],
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
        sgp_denominators: SGP conversion denominators (REQUIRED — no
            fallback). Typically ``config.sgp_denominators`` derived
            from live league standings.

    Returns:
        Dict of {team_name: valuation_sgp} for all opponents.
    """
    if not sgp_denominators:
        raise ValueError(
            "sgp_denominators is required (no fallback). "
            "Pass config.sgp_denominators or LeagueConfig().sgp_denominators."
        )
    denoms = sgp_denominators
    valuations: dict[str, float] = {}

    # Bug C: rate stats (AVG, OBP, ERA, WHIP) require volume-weighted
    # blending. Component volumes (AB/PA/IP) come from the projections
    # dict when available, otherwise fall back to LeagueConfig roster-
    # baseline defaults. Pull once per player.
    player_ab = float(player_projections.get("ab", player_projections.get("AB", 0)) or 0)
    player_pa = float(player_projections.get("pa", player_projections.get("PA", 0)) or 0)
    player_ip = float(player_projections.get("ip", player_projections.get("IP", 0)) or 0)

    for team_id, team_totals in all_team_totals.items():
        if team_id == your_team_id:
            continue

        team_ab = float(team_totals.get("ab", team_totals.get("AB", 0)) or 0)
        team_pa = float(team_totals.get("pa", team_totals.get("PA", 0)) or 0)
        team_ip = float(team_totals.get("ip", team_totals.get("IP", 0)) or 0)

        # Compute this opponent's marginal value for the player
        team_val = 0.0
        for cat in CATEGORIES:
            proj = player_projections.get(cat, 0.0)
            if abs(proj) < 1e-9:
                continue

            denom = denoms.get(cat, 1.0)
            if abs(denom) < 1e-9:
                denom = 1.0

            team_cat_total = team_totals.get(cat, 0.0)
            gap_to_next = _gap_to_next_team(team_cat_total, cat, team_id, all_team_totals)

            # Bug C: route AVG/OBP/ERA/WHIP through the rate-stat helper.
            # Previously, AVG/OBP fell through to the generic counting-stat
            # `proj / denom * 0.5` path in the "already best" branch,
            # producing absurd SGP values (~29 SGP per rate cat for a
            # below-average hitter — see scripts/diag_trade_engine_validation.py
            # 2026-05-23 baseline run).
            if cat in ("AVG", "OBP", "ERA", "WHIP"):
                # Pick the matching volume per category
                if cat == "AVG":
                    pv = player_ab if player_ab > 0 else _DEFAULT_PLAYER_AB
                    tv = team_ab if team_ab > 0 else _DEFAULT_TEAM_AB
                elif cat == "OBP":
                    pv = player_pa if player_pa > 0 else _DEFAULT_PLAYER_PA
                    tv = team_pa if team_pa > 0 else _DEFAULT_TEAM_PA
                else:  # ERA, WHIP
                    pv = player_ip if player_ip > 0 else _DEFAULT_PLAYER_IP
                    tv = team_ip if team_ip > 0 else _DEFAULT_TEAM_IP
                team_val += _rate_stat_marginal_sgp(
                    cat=cat,
                    player_rate=proj,
                    team_rate=team_cat_total,
                    player_volume=pv,
                    team_volume=tv,
                    denom=denom,
                    gap_to_next=gap_to_next,
                    already_best_mult=0.5,
                )
            elif cat == "L":
                # L: inverse counting stat — more losses hurts the team.
                # Half-weight when team is already best in L (low L count).
                multiplier = 1.0 if gap_to_next > 0 else 0.5
                proj_l = float(player_projections.get("L", player_projections.get("l", 0.0)) or 0)
                contribution = proj_l / denom * multiplier
                # Clamp to cap (defense-in-depth)
                contribution = min(contribution, MAX_PER_CAT_SGP)
                team_val -= contribution
            else:
                # Counting stats (R, HR, RBI, SB, W, SV, K)
                if gap_to_next > 0:
                    contribution = min(proj, gap_to_next) / denom
                else:
                    contribution = proj / denom * 0.5
                # Clamp to cap (defense-in-depth)
                if contribution > MAX_PER_CAT_SGP:
                    logger.warning(
                        "opponent_valuation: counting-stat SGP %.2f for category %s "
                        "exceeded cap %.1f (proj=%.1f gap=%.1f denom=%.2f). Clamping.",
                        contribution,
                        cat,
                        MAX_PER_CAT_SGP,
                        proj,
                        gap_to_next,
                        denom,
                    )
                    contribution = MAX_PER_CAT_SGP
                team_val += contribution

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
    sgp_denominators: dict[str, float],
) -> PlayerMarketValue:
    """Full market analysis for a single player.

    Combines opponent valuations with market clearing price and
    demand analysis.

    Args:
        player_projections: Player's projected stats.
        all_team_totals: All team totals.
        your_team_id: Your team name.
        sgp_denominators: SGP denominators (REQUIRED — no fallback).
            Typically ``config.sgp_denominators``.

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

    # Bug C: also surface component volumes (AB/PA/IP) so the rate-stat
    # blender in estimate_opponent_valuations can compute volume-weighted
    # marginal SGP for AVG/OBP (parallel to existing ERA/WHIP handling).
    # Falls back to LeagueConfig roster-baseline defaults when missing.
    for vol_col in ("ab", "pa", "ip"):
        val = row.get(vol_col, 0.0)
        try:
            projections[vol_col] = float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            projections[vol_col] = 0.0

    return projections
