"""Enhanced bench option value with flexibility and injury replacement.

Spec reference: Section 10 L7D (Bench Option Value)
               Section 17 Phase 4 item 19

Extends the simple bench_option_value() from Phase 1 with:
  1. Roster flexibility premium (multi-position eligibility)
  2. Injury replacement cushion (value of having bench depth)
  3. Hot FA pickup option value (improved model)

The key insight: a bench slot isn't just "streaming value." A player
who can play 4 positions on the bench is worth more than a player who
only plays 1B, because they can fill in at any position when injuries
hit. This module captures that optionality.

Wires into:
  - src/engine/portfolio/lineup_optimizer.py: existing bench_option_value
  - src/engine/output/trade_evaluator.py: enhanced value in trade surplus
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Default SGP per week from streaming (same as existing Phase 1 value)
DEFAULT_STREAMING_SGP: float = 0.15

# Flexibility premium per unit of normalized flexibility score
FLEXIBILITY_PREMIUM_PER_UNIT: float = 0.1

# Hot FA pickup parameters
HOT_FA_WEEKLY_PROB: float = 0.15  # Probability per week of a breakout FA
HOT_FA_VALUE: float = 0.5  # Expected SGP from a breakout pickup


def enhanced_bench_option_value(
    weeks_remaining: int = 16,
    streaming_sgp_per_week: float = DEFAULT_STREAMING_SGP,
    roster_flexibility: float = 0.0,
    injury_replacement_value: float = 0.0,
) -> dict[str, float]:
    """Compute the total value of one empty roster slot.

    Spec ref: Section 10 L7D — bench option value (extended).

    Returns a breakdown dict with component values and total:
      - streaming: weekly streaming pickup value
      - hot_fa: option value of breakout free agents
      - flexibility: premium for multi-position bench players
      - injury_cushion: value of injury replacement depth
      - total: sum of all components

    Args:
        weeks_remaining: Weeks left in the season.
        streaming_sgp_per_week: Expected SGP from weekly streaming moves.
        roster_flexibility: Normalized flexibility score [0, 1].
            Higher = bench players have more positional eligibility.
        injury_replacement_value: Expected SGP saved by bench injury coverage.

    Returns:
        Dict with component breakdown and total.
    """
    # 1. Streaming value: reliable weekly pickup SGP
    streaming = streaming_sgp_per_week * weeks_remaining

    # 2. Hot FA option value: probability of breakout × expected value
    hot_fa_prob = 1.0 - (1.0 - HOT_FA_WEEKLY_PROB) ** weeks_remaining
    hot_fa = hot_fa_prob * HOT_FA_VALUE * 0.5  # Discount for uncertainty

    # 3. Roster flexibility premium
    flexibility = roster_flexibility * FLEXIBILITY_PREMIUM_PER_UNIT * weeks_remaining

    # 4. Injury replacement cushion
    injury_cushion = injury_replacement_value

    total = streaming + hot_fa + flexibility + injury_cushion

    return {
        "streaming": round(streaming, 3),
        "hot_fa": round(hot_fa, 3),
        "flexibility": round(flexibility, 3),
        "injury_cushion": round(injury_cushion, 3),
        "total": round(total, 3),
    }


def compute_roster_flexibility(roster_df: pd.DataFrame) -> float:
    """Score a roster's positional flexibility.

    Counts total positional eligibility across all bench players.
    More eligible positions = more flexibility to cover injuries
    and exploit favorable matchups.

    Scoring:
      - Each position beyond the first adds 1 unit
      - Normalized by max possible (8 positions × 5 bench slots = 40)

    Args:
        roster_df: Roster DataFrame with 'positions' column.
            Positions should be a comma-separated string or list.

    Returns:
        Normalized flexibility score in [0, 1].
    """
    if roster_df is None or roster_df.empty:
        return 0.0

    total_extra_positions = 0

    for _, player in roster_df.iterrows():
        positions = player.get("positions", "")
        if isinstance(positions, str):
            pos_list = [p.strip() for p in positions.split(",") if p.strip()]
        elif isinstance(positions, list):
            pos_list = positions
        else:
            pos_list = []

        # Each position beyond the first adds flexibility
        if len(pos_list) > 1:
            total_extra_positions += len(pos_list) - 1

    # Normalize: assume max ~40 extra positions across a 23-man roster
    max_extra = 40.0
    return min(total_extra_positions / max_extra, 1.0)


def compute_injury_replacement_value(
    roster_df: pd.DataFrame,
    bench_count: int = 5,
    avg_health_score: float = 0.85,
    weeks_remaining: int = 16,
) -> float:
    """Estimate SGP saved by having bench depth for injury coverage.

    The idea: if a starter gets injured and you have a bench player
    who can fill in, you avoid losing production. The value depends
    on how likely injuries are and how much production a replacement
    can provide vs an empty slot.

    Args:
        roster_df: Roster DataFrame with stat projections.
        bench_count: Number of bench players available.
        avg_health_score: Average health score of starters.
        weeks_remaining: Weeks left in season.

    Returns:
        Expected SGP saved from bench injury coverage.
    """
    if roster_df is None or roster_df.empty or bench_count <= 0:
        return 0.0

    # Expected number of injury events for active roster
    n_starters = max(len(roster_df) - bench_count, 1)
    injury_prob_per_starter = (1.0 - avg_health_score) * (weeks_remaining / 16.0)

    # Expected injuries across the roster
    expected_injuries = n_starters * injury_prob_per_starter

    # Each bench player can cover ~0.3 SGP per injury event
    # (partial replacement value vs losing the slot entirely)
    replacement_sgp_per_event = 0.3

    # Value scales with both expected injuries and bench depth
    # But diminishing returns: 5th bench player is less valuable than 1st
    effective_bench = min(bench_count, expected_injuries + 1)
    coverage_fraction = effective_bench / max(expected_injuries, 1)
    coverage_fraction = min(coverage_fraction, 1.0)

    return expected_injuries * replacement_sgp_per_event * coverage_fraction
