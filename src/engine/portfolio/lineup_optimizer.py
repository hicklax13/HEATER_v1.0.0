"""Integer LP lineup optimizer wrapper with bench option value.

Spec reference: Section 10 L7C (integer LP lineup optimizer)
               Section 10 L7D (bench option value)

Wires into existing:
  - src/lineup_optimizer.py: LineupOptimizer, ROSTER_SLOTS

This module:
  1. Wraps the existing LP optimizer for trade-specific lineup analysis
  2. Computes optimal lineup value for pre-trade and post-trade rosters
  3. Calculates bench option value (streaming + hot FA potential)
  4. Returns the lineup delta — how the trade changes your optimal output
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from src.lineup_optimizer import ROSTER_SLOTS, LineupOptimizer

if TYPE_CHECKING:
    from src.valuation import LeagueConfig


def compute_optimal_lineup_value(
    roster: pd.DataFrame,
    config: LeagueConfig | None = None,
    category_weights: dict[str, float] | None = None,
) -> dict:
    """Compute the optimal lineup assignment and projected stats.

    Spec ref: Section 10 L7C — maximize value subject to positional eligibility,
    slot counts, and active roster limit.

    Wraps the existing LineupOptimizer from src/lineup_optimizer.py.

    Args:
        roster: Player roster DataFrame with columns: player_id, player_name,
                positions, plus stat columns (r, hr, rbi, sb, avg, etc.).
        config: League configuration for SGP denominators.
        category_weights: Per-category weights for the LP objective.

    Returns:
        Dict with keys:
          - assignments: list of {slot, player_name, player_id}
          - bench: list of benched player names
          - projected_stats: dict of total stats for optimal starters
          - bench_count: number of players on the bench
          - status: solver status string
    """
    if roster.empty:
        return {
            "assignments": [],
            "bench": [],
            "projected_stats": {},
            "bench_count": 0,
            "status": "empty_roster",
        }

    optimizer = LineupOptimizer(roster, config=config, roster_slots=ROSTER_SLOTS)
    result = optimizer.optimize_lineup(category_weights=category_weights)
    result["bench_count"] = len(result.get("bench", []))
    return result


def compute_lineup_delta(
    before_roster: pd.DataFrame,
    after_roster: pd.DataFrame,
    config: LeagueConfig | None = None,
    category_weights: dict[str, float] | None = None,
) -> dict:
    """Compare optimal lineups before and after a trade.

    Args:
        before_roster: User's current roster DataFrame.
        after_roster: User's roster after applying the trade.
        config: League configuration.
        category_weights: Per-category weights.

    Returns:
        Dict with keys:
          - before: optimal lineup result (pre-trade)
          - after: optimal lineup result (post-trade)
          - stat_deltas: per-category change in projected stats
          - bench_delta: change in bench size (negative = lost a bench slot)
    """
    before = compute_optimal_lineup_value(before_roster, config, category_weights)
    after = compute_optimal_lineup_value(after_roster, config, category_weights)

    stat_deltas: dict[str, float] = {}
    all_cats = list(set(before.get("projected_stats", {}).keys()) | set(after.get("projected_stats", {}).keys()))
    for cat in all_cats:
        before_val = before.get("projected_stats", {}).get(cat, 0.0)
        after_val = after.get("projected_stats", {}).get(cat, 0.0)
        stat_deltas[cat] = after_val - before_val

    return {
        "before": before,
        "after": after,
        "stat_deltas": stat_deltas,
        "bench_delta": after.get("bench_count", 0) - before.get("bench_count", 0),
    }


def bench_option_value(
    weeks_remaining: int = 16,
    streaming_sgp_per_week: float = 0.15,
) -> float:
    """Value of an empty roster slot (streaming + hot FA potential).

    Spec ref: Section 10 L7D — bench option value.

    An empty bench slot has value because:
      1. Weekly streaming adds ~0.15 SGP/week (2-start pitchers, hot matchups)
      2. Hot free agent pickups (15% chance/week × 0.5 SGP expected value)

    For 2-for-1 trades that cost a bench slot, this value must be subtracted
    from the trade surplus.

    Args:
        weeks_remaining: Weeks left in the season.
        streaming_sgp_per_week: Expected SGP from weekly streaming moves.

    Returns:
        Total SGP value of one empty roster slot over remaining season.
    """
    # Streaming value: reliable weekly pickup SGP
    stream = streaming_sgp_per_week * weeks_remaining

    # Hot FA option value: probability of a breakout FA appearing × expected value
    hot_pickup_prob = min(0.15 * weeks_remaining, 1.0)  # Cap at 100%
    hot_pickup_val = 0.5  # Expected SGP from a breakout pickup
    option = hot_pickup_prob * hot_pickup_val * 0.5  # Discount for uncertainty

    return stream + option
