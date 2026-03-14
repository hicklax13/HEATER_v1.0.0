"""Game-level matchup adjustments via Log5 and park factors.

Spec reference: Section 9 L6 (Granular Matchup Engine)
               Section 17 Phase 4 item 17

Implements the Log5 odds-ratio method for batter-pitcher matchup
prediction, park factor adjustments, and lineup slot PA estimates.

The key idea: a .300 hitter vs a league-average pitcher in a
neutral park should hit ~.300. But in Coors (PF=1.38) against
a weak pitcher (.360 wOBA against), that same hitter projects
significantly higher.

Wires into:
  - src/data_bootstrap.py: PARK_FACTORS dict (30 teams)
  - src/engine/output/trade_evaluator.py: optional matchup context
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────

# Plate appearances per game by lineup slot
# Source: Historical MLB averages (top of order gets more PA)
LINEUP_SLOT_PA: dict[int, float] = {
    1: 4.65,
    2: 4.55,
    3: 4.45,
    4: 4.35,
    5: 4.25,
    6: 4.15,
    7: 4.05,
    8: 3.95,
    9: 3.85,
}

# Default league average rates (2024 MLB)
DEFAULT_LEAGUE_AVG_WOBA: float = 0.315
DEFAULT_LEAGUE_AVG_BA: float = 0.248
DEFAULT_LEAGUE_AVG_HR_RATE: float = 0.033  # HR per PA

# Minimum/maximum rate bounds to avoid division by zero
RATE_FLOOR: float = 0.001
RATE_CEILING: float = 0.999


# ── Log5 Matchup ────────────────────────────────────────────────────


def log5_matchup(
    batter_rate: float,
    pitcher_rate: float,
    league_avg: float,
) -> float:
    """Compute expected rate using the Log5 odds-ratio method.

    Spec ref: Section 9 L6 — odds ratio method for specific matchups.

    The Log5 method combines batter ability, pitcher ability, and
    league context to produce a matchup-specific expected rate.

    Formula:
        B_odds = batter_rate / (1 - batter_rate)
        P_odds = pitcher_rate / (1 - pitcher_rate)
        L_odds = league_avg / (1 - league_avg)
        M_odds = (B_odds * P_odds) / L_odds
        result = M_odds / (1 + M_odds)

    Args:
        batter_rate: Batter's rate stat (e.g., BA, wOBA). Must be in (0, 1).
        pitcher_rate: Pitcher's rate-against stat. Must be in (0, 1).
        league_avg: League average for this stat. Must be in (0, 1).

    Returns:
        Expected matchup rate in (0, 1).
    """
    # Clamp to avoid division by zero
    br = np.clip(batter_rate, RATE_FLOOR, RATE_CEILING)
    pr = np.clip(pitcher_rate, RATE_FLOOR, RATE_CEILING)
    la = np.clip(league_avg, RATE_FLOOR, RATE_CEILING)

    b_odds = br / (1.0 - br)
    p_odds = pr / (1.0 - pr)
    l_odds = la / (1.0 - la)

    m_odds = (b_odds * p_odds) / l_odds
    return float(m_odds / (1.0 + m_odds))


# ── Park Adjustment ─────────────────────────────────────────────────


def park_adjust(stat_value: float, park_factor: float) -> float:
    """Apply a park factor adjustment to a stat value.

    Park factors are centered at 1.0:
      - Coors Field: ~1.38 (inflates stats)
      - Marlins Park: ~0.88 (deflates stats)

    Args:
        stat_value: Raw stat value to adjust.
        park_factor: Multiplicative factor (1.0 = neutral).

    Returns:
        Park-adjusted stat value.
    """
    return stat_value * park_factor


# ── Stat Rate Estimators ────────────────────────────────────────────


def estimate_hr_rate(xwoba: float) -> float:
    """Estimate HR rate from expected wOBA.

    Simple linear regression approximation from MLB data:
    HR/PA ≈ 0.12 * xwOBA - 0.005

    Args:
        xwoba: Expected weighted on-base average.

    Returns:
        Estimated HR per PA rate.
    """
    return max(0.0, 0.12 * xwoba - 0.005)


def estimate_hit_rate(xwoba: float) -> float:
    """Estimate batting average from expected wOBA.

    BA ≈ 0.72 * xwOBA + 0.02

    Args:
        xwoba: Expected weighted on-base average.

    Returns:
        Estimated batting average.
    """
    return np.clip(0.72 * xwoba + 0.02, 0.100, 0.400)


def estimate_r_rate(xwoba: float, lineup_slot: int = 5) -> float:
    """Estimate runs scored rate from xwOBA and lineup position.

    Top-of-order batters score more runs. Leadoff and #2 batters
    get 15-20% more run-scoring opportunities.

    Args:
        xwoba: Expected weighted on-base average.
        lineup_slot: Position in batting order (1-9).

    Returns:
        Estimated runs per PA rate.
    """
    base = 0.15 * xwoba - 0.01
    slot_mult = 1.0 + max(0, (5 - lineup_slot)) * 0.04
    return max(0.0, base * slot_mult)


def estimate_rbi_rate(xwoba: float, lineup_slot: int = 5) -> float:
    """Estimate RBI rate from xwOBA and lineup position.

    3-4-5 hitters drive in the most runs due to lineup protection.

    Args:
        xwoba: Expected weighted on-base average.
        lineup_slot: Position in batting order (1-9).

    Returns:
        Estimated RBI per PA rate.
    """
    base = 0.18 * xwoba - 0.015
    # Cleanup hitters (3-5 spot) get ~10% more RBI opportunities
    if 3 <= lineup_slot <= 5:
        base *= 1.10
    return max(0.0, base)


# ── Game Projection ─────────────────────────────────────────────────


def game_projection(
    batter_xwoba: float,
    pitcher_xwoba_against: float | None = None,
    park_factor: float = 1.0,
    lineup_slot: int = 5,
    league_avg_woba: float = DEFAULT_LEAGUE_AVG_WOBA,
    temp_f: float | None = None,
    wind_out_mph: float | None = None,
) -> dict[str, float]:
    """Compute per-game stat projections for one batter.

    Spec ref: Section 9 L6 — full game-level stat projection.

    Combines Log5 matchup, park factor, and optional weather
    adjustments into per-game stat projections.

    Args:
        batter_xwoba: Batter's expected wOBA.
        pitcher_xwoba_against: Pitcher's xwOBA against (None = league avg).
        park_factor: Park factor multiplier (default 1.0 = neutral).
        lineup_slot: Batting order position (1-9).
        league_avg_woba: League average wOBA.
        temp_f: Temperature in Fahrenheit (None = no adjustment).
        wind_out_mph: Outgoing wind speed in mph (None = no adjustment).

    Returns:
        Dict with projected stats for one game:
          hr, r, rbi, sb, h, ab, pa
    """
    # Determine effective wOBA for this matchup
    if pitcher_xwoba_against is not None:
        effective_woba = log5_matchup(batter_xwoba, pitcher_xwoba_against, league_avg_woba)
    else:
        effective_woba = batter_xwoba

    # Park adjustment
    effective_woba = park_adjust(effective_woba, park_factor)

    # Weather adjustments (neutral when data unavailable)
    if temp_f is not None:
        temp_adj = 1.0 + 0.002 * (temp_f - 72.0)
        effective_woba *= temp_adj
    if wind_out_mph is not None:
        wind_adj = 1.0 + 0.001 * wind_out_mph
        effective_woba *= wind_adj

    # PA for this lineup slot
    pa = LINEUP_SLOT_PA.get(lineup_slot, 4.0)

    # Derive counting stat rates
    hr_rate = estimate_hr_rate(effective_woba)
    hit_rate = estimate_hit_rate(effective_woba)
    r_rate = estimate_r_rate(effective_woba, lineup_slot)
    rbi_rate = estimate_rbi_rate(effective_woba, lineup_slot)

    # Estimate AB from PA (subtract walks, ~8-10% of PA)
    bb_rate = 0.09  # League average walk rate
    ab = pa * (1.0 - bb_rate)

    return {
        "hr": hr_rate * pa,
        "r": r_rate * pa,
        "rbi": rbi_rate * pa,
        "sb": 0.0,  # SB requires baserunning data, not derivable from wOBA
        "h": hit_rate * ab,
        "ab": ab,
        "pa": pa,
    }


# ── Matchup Adjustment Factor ──────────────────────────────────────


def matchup_adjustment_factor(
    player_xwoba: float | None = None,
    opponent_xwoba_against: float | None = None,
    park_factor: float = 1.0,
    league_avg: float = DEFAULT_LEAGUE_AVG_WOBA,
) -> float:
    """Compute a multiplicative adjustment factor for a player's projections.

    This is the high-level function the trade evaluator calls. Without
    opponent data, returns 1.0 (neutral adjustment) — graceful fallback.

    Args:
        player_xwoba: Player's expected wOBA (None = no adjustment).
        opponent_xwoba_against: Opponent pitcher's xwOBA against (None = neutral).
        park_factor: Park factor (default 1.0).
        league_avg: League average wOBA.

    Returns:
        Multiplicative factor (1.0 = no change, >1 = upgrade, <1 = downgrade).
    """
    if player_xwoba is None:
        return 1.0

    # If no opponent data, just apply park factor
    if opponent_xwoba_against is None:
        return park_factor

    # Log5 matchup-adjusted rate vs neutral expectation
    matchup_rate = log5_matchup(player_xwoba, opponent_xwoba_against, league_avg)
    neutral_rate = player_xwoba

    if neutral_rate <= RATE_FLOOR:
        return 1.0

    return (matchup_rate / neutral_rate) * park_factor
