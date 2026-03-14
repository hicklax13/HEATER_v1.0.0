"""Roster concentration risk via Herfindahl-Hirschman Index.

Spec reference: Section 7 L4B (Roster Concentration Risk)
               Section 17 Phase 4 item 20

Measures how concentrated a fantasy roster is across MLB teams.
High concentration = correlated downside risk: if one MLB team
slumps, gets injuries, or has a brutal schedule, multiple fantasy
players suffer simultaneously.

Example:
  - 5 Yankees players: HHI ≈ 0.25 (very concentrated, dangerous)
  - 1 player per team: HHI ≈ 0.04 (well-diversified)

The Herfindahl-Hirschman Index (HHI) = Σ(share_i²) where share_i
is team i's fraction of total roster production capacity (PA for
hitters, IP for pitchers).

Wires into:
  - src/engine/output/trade_evaluator.py: concentration risk penalty
  - src/database.py: player_pool DataFrame (team column)
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# HHI threshold above which a penalty applies
# 0.15 ≈ roughly 3 players from the same team on a 23-man roster
HHI_PENALTY_THRESHOLD: float = 0.15

# SGP penalty scale factor
# Calibrated so 5 players from one team ≈ 0.3 SGP penalty
PENALTY_SCALE: float = 3.0

# Minimum weight for a player (prevents div-by-zero for zero PA/IP)
MIN_PLAYER_WEIGHT: float = 1.0

# Team abbreviation aliases for normalization
# MLB API and FanGraphs sometimes use different abbreviations
TEAM_ALIASES: dict[str, str] = {
    "WSN": "WSH",
    "AZ": "ARI",
    "CHW": "CWS",
    "TB": "TBR",
    "KC": "KCR",
    "SD": "SDP",
    "SF": "SFG",
}


def _normalize_team(team: str) -> str:
    """Normalize team abbreviation to a canonical form."""
    if not isinstance(team, str):
        return "UNK"
    t = team.strip().upper()
    return TEAM_ALIASES.get(t, t)


def roster_concentration_hhi(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
) -> float:
    """Compute the Herfindahl-Hirschman Index for roster team exposure.

    Spec ref: Section 7 L4B — roster concentration risk.

    HHI = Σ(share_i²) where share_i = team_i's fraction of total
    production capacity. Uses PA for hitters, IP for pitchers.

    Args:
        roster_ids: List of player IDs on the roster.
        player_pool: Full player pool DataFrame with columns:
            player_id, team, is_hitter, pa, ip.

    Returns:
        HHI value in [0, 1]. Lower = more diversified.
        0 for empty rosters.
    """
    if not roster_ids:
        return 0.0

    roster = player_pool[player_pool["player_id"].isin(roster_ids)]
    if roster.empty:
        return 0.0

    # Compute weight per player: PA for hitters, IP for pitchers
    team_weights: dict[str, float] = {}

    for _, player in roster.iterrows():
        team = _normalize_team(str(player.get("team", "UNK")))
        is_hitter = bool(player.get("is_hitter", True))

        if is_hitter:
            weight = max(float(player.get("pa", 0) or 0), MIN_PLAYER_WEIGHT)
        else:
            weight = max(float(player.get("ip", 0) or 0), MIN_PLAYER_WEIGHT)

        team_weights[team] = team_weights.get(team, 0.0) + weight

    total_weight = sum(team_weights.values())
    if total_weight <= 0:
        return 0.0

    # HHI = sum of squared shares
    hhi = sum((w / total_weight) ** 2 for w in team_weights.values())

    return float(hhi)


def concentration_risk_penalty(
    hhi: float,
    threshold: float = HHI_PENALTY_THRESHOLD,
    scale: float = PENALTY_SCALE,
) -> float:
    """Convert an HHI value into an SGP penalty.

    Below the threshold (well-diversified), no penalty.
    Above, the penalty grows linearly:
        penalty = (hhi - threshold) * scale

    Args:
        hhi: Herfindahl-Hirschman Index value.
        threshold: HHI level below which no penalty applies.
        scale: Penalty multiplier per unit of excess HHI.

    Returns:
        SGP penalty (non-negative). Higher = worse.
    """
    if hhi <= threshold:
        return 0.0
    return (hhi - threshold) * scale


def compute_concentration_delta(
    before_roster_ids: list[int],
    after_roster_ids: list[int],
    player_pool: pd.DataFrame,
) -> dict[str, float]:
    """Compute the change in concentration risk from a trade.

    Args:
        before_roster_ids: Pre-trade roster player IDs.
        after_roster_ids: Post-trade roster player IDs.
        player_pool: Full player pool DataFrame.

    Returns:
        Dict with:
          - hhi_before: Pre-trade HHI
          - hhi_after: Post-trade HHI
          - hhi_delta: Change in HHI (positive = more concentrated)
          - penalty_before: Pre-trade SGP penalty
          - penalty_after: Post-trade SGP penalty
          - penalty_delta: Change in penalty (positive = worse)
    """
    hhi_before = roster_concentration_hhi(before_roster_ids, player_pool)
    hhi_after = roster_concentration_hhi(after_roster_ids, player_pool)

    penalty_before = concentration_risk_penalty(hhi_before)
    penalty_after = concentration_risk_penalty(hhi_after)

    return {
        "hhi_before": round(hhi_before, 4),
        "hhi_after": round(hhi_after, 4),
        "hhi_delta": round(hhi_after - hhi_before, 4),
        "penalty_before": round(penalty_before, 3),
        "penalty_after": round(penalty_after, 3),
        "penalty_delta": round(penalty_after - penalty_before, 3),
    }


def team_exposure_breakdown(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
) -> dict[str, dict[str, float | int]]:
    """Break down roster exposure by MLB team.

    Useful for UI display: "You have 4 Yankees after this trade."

    Args:
        roster_ids: Player IDs on the roster.
        player_pool: Full player pool DataFrame.

    Returns:
        Dict mapping team abbreviation to:
          - count: number of players from that team
          - share: fraction of total production weight
    """
    if not roster_ids:
        return {}

    roster = player_pool[player_pool["player_id"].isin(roster_ids)]
    if roster.empty:
        return {}

    team_data: dict[str, dict[str, float | int]] = {}
    total_weight = 0.0

    for _, player in roster.iterrows():
        team = _normalize_team(str(player.get("team", "UNK")))
        is_hitter = bool(player.get("is_hitter", True))

        if is_hitter:
            weight = max(float(player.get("pa", 0) or 0), MIN_PLAYER_WEIGHT)
        else:
            weight = max(float(player.get("ip", 0) or 0), MIN_PLAYER_WEIGHT)

        if team not in team_data:
            team_data[team] = {"count": 0, "weight": 0.0}
        team_data[team]["count"] += 1
        team_data[team]["weight"] += weight
        total_weight += weight

    # Convert weights to shares
    result: dict[str, dict[str, float | int]] = {}
    for team, data in sorted(team_data.items(), key=lambda x: -x[1]["weight"]):
        result[team] = {
            "count": int(data["count"]),
            "share": round(data["weight"] / max(total_weight, 1.0), 3),
        }

    return result
