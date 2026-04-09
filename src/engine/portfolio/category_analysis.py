"""Marginal SGP + category gap analysis with punt detection.

Spec reference: Section 10 L7A (marginal category elasticity)
               Section 10 L7B (category gap analysis + punt detection)

Wires into existing:
  - src/database.py: load_league_standings
  - src/valuation.py: LeagueConfig

This module answers: "How much does one additional unit of stat X move my
team in the standings?" — and identifies categories where improvement is
impossible (punts), so the trade evaluator can zero-weight them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.validation.constant_optimizer import load_constants
from src.valuation import LeagueConfig as _LC_Class

_CONSTANTS = load_constants()
_LC = _LC_Class()
CATEGORIES: list[str] = _LC.all_categories
INVERSE_CATEGORIES: set[str] = _LC.inverse_stats

# Approximate weekly production rates per roster, used for punt estimation.
# Based on a competitive 12-team roster in a full season (~22 weeks).
WEEKLY_RATE_DEFAULTS: dict[str, float] = {
    "R": 35.0,
    "HR": 9.0,
    "RBI": 34.0,
    "SB": 5.0,
    "AVG": 0.0,  # Rate stat — handled differently
    "OBP": 0.0,  # Rate stat — handled differently
    "W": 3.0,
    "L": 3.0,
    "SV": 2.7,
    "K": 50.0,
    "ERA": 0.0,  # Rate stat — handled differently
    "WHIP": 0.0,  # Rate stat — handled differently
}


def compute_marginal_sgp(
    your_totals: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    categories: list[str] | None = None,
    your_team_name: str | None = None,
) -> dict[str, float]:
    """Compute marginal SGP: dE[standings_points] / d(your_stat).

    Spec ref: Section 10 L7A — non-linear marginal value depends on exact
    proximity to adjacent teams in the standings.

    The marginal value is 1/gap to the next team. If you're 2 HR from
    gaining a standings point, each HR is worth ~0.5 standings points.
    If you're 40 HR ahead, each additional HR is nearly worthless.

    Args:
        your_totals: Your team's category totals (e.g. {"R": 780, "HR": 200, ...}).
        all_team_totals: Dict mapping team_id/name -> category totals.
            Must include your team.
        categories: Which categories to compute. Defaults to all 10.
        your_team_name: If provided, exclude this team from the opponent list.

    Returns:
        Dict mapping category -> marginal SGP value (float).
    """
    if categories is None:
        categories = CATEGORIES

    # Exclude user's own team from opponent totals to avoid comparing against self
    opponent_totals = (
        {k: v for k, v in all_team_totals.items() if k != your_team_name} if your_team_name else all_team_totals
    )

    marginal: dict[str, float] = {}

    for cat in categories:
        # Use .get() to gracefully handle teams missing this category
        totals_sorted = sorted([team_totals.get(cat, 0.0) for team_totals in opponent_totals.values()])
        your_val = your_totals.get(cat, 0.0)

        if cat in INVERSE_CATEGORIES:
            # Lower is better: find team with next lower total
            better = sorted([t for t in totals_sorted if t < your_val], reverse=True)
        else:
            # Higher is better: find team with next higher total
            better = sorted([t for t in totals_sorted if t > your_val])

        if not better:
            # Already in 1st place in this category
            marginal[cat] = 0.01
        else:
            gap = max(abs(better[0] - your_val), 0.001)
            # 1 standings point per gap unit of stat
            marginal[cat] = 1.0 / gap

    return marginal


def category_gap_analysis(
    your_totals: dict[str, float],
    all_team_totals: dict[str, dict[str, float]],
    your_team_id: str,
    weeks_remaining: int = 16,
    weekly_rates: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Per-category analysis: rank, gaps, achievability, punt flags.

    Spec ref: Section 10 L7B — punt detection.

    A category is a PUNT if:
      - You cannot gain any standings position in the remaining weeks
      - You're ranked 10th or worse in that category

    Args:
        your_totals: Your team's category totals.
        all_team_totals: All teams' totals keyed by team ID/name.
        your_team_id: Your team's key in all_team_totals.
        weeks_remaining: Weeks left in the season.
        weekly_rates: Expected weekly production per category.
            Defaults to WEEKLY_RATE_DEFAULTS.

    Returns:
        Dict mapping category -> analysis dict with keys:
          rank, is_punt, marginal_value, gap_to_next, gainable_positions.
    """
    if weekly_rates is None:
        weekly_rates = dict(WEEKLY_RATE_DEFAULTS)

    analysis: dict[str, dict] = {}

    for cat in CATEGORIES:
        # Rank teams in this category
        if cat in INVERSE_CATEGORIES:
            # Lower is better: ascending sort puts best first
            ranked = sorted(
                all_team_totals.items(),
                key=lambda x: x[1].get(cat, 999),
            )
        else:
            # Higher is better: descending sort puts best first
            ranked = sorted(
                all_team_totals.items(),
                key=lambda x: x[1].get(cat, 0),
                reverse=True,
            )

        # Find your rank (1-indexed)
        your_rank = 1
        for i, (tid, _) in enumerate(ranked):
            if tid == your_team_id:
                your_rank = i + 1
                break

        # Count how many positions you can gain
        weekly_rate = weekly_rates.get(cat, 0.0)
        gainable = 0

        for target_rank in range(1, your_rank):
            target_total = ranked[target_rank - 1][1].get(cat, 0)
            gap = abs(target_total - your_totals.get(cat, 0))

            if cat in INVERSE_CATEGORIES:
                # Can't really "produce" lower ERA/WHIP at a fixed weekly rate.
                # For rate stats, check if the gap is closeable given remaining IP.
                # Simplified: if gap < 0.5 for ERA, consider it achievable.
                if gap < 0.5:
                    gainable += 1
            elif weekly_rate > 0:
                weeks_needed = gap / weekly_rate
                # Allow 20% buffer: if you can close the gap in 1.2x remaining weeks
                if weeks_needed <= weeks_remaining * _CONSTANTS.get("punt_weeks_buffer"):
                    gainable += 1

        # Punt detection: cannot gain any position AND ranked 10th or worse
        is_punt = gainable == 0 and your_rank >= _CONSTANTS.get("punt_rank_threshold")

        # Compute marginal value (0 if punt)
        if is_punt:
            marginal_value = 0.0
        else:
            marginal_value = compute_marginal_sgp(your_totals, all_team_totals, [cat], your_team_name=your_team_id).get(
                cat, 0.0
            )

        # Gap to next rank above
        gap_to_next = 0.0
        if your_rank > 1:
            next_above = ranked[your_rank - 2][1].get(cat, 0)
            gap_to_next = abs(next_above - your_totals.get(cat, 0))

        analysis[cat] = {
            "rank": your_rank,
            "is_punt": is_punt,
            "marginal_value": marginal_value,
            "gap_to_next": round(gap_to_next, 3),
            "gainable_positions": gainable,
        }

    return analysis


def build_standings_totals(
    standings: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """Convert standings DataFrame to the all_team_totals dict format.

    Args:
        standings: DataFrame with columns: team_name, category, total.

    Returns:
        Dict mapping team_name -> {category: total}.
    """
    if standings.empty:
        return {}

    totals: dict[str, dict[str, float]] = {}
    for _, row in standings.iterrows():
        team = str(row["team_name"])
        cat = str(row["category"])
        total = float(row.get("total", 0) or 0)
        if team not in totals:
            totals[team] = {}
        totals[team][cat] = total

    return totals


def compute_category_weights_from_analysis(
    analysis: dict[str, dict],
    max_weight: float = 3.0,
) -> dict[str, float]:
    """Convert gap analysis into trade evaluation category weights.

    Categories with high marginal value get high weight.
    Punted categories get zero weight.
    Normalizes so the average non-punt weight is ~1.0.

    After normalization, individual weights are capped at *max_weight* to
    prevent a single category with a tiny gap (e.g. AVG gap of 0.01 →
    marginal 100) from dominating the trade evaluation.  Without the cap,
    LP-solver lineup reshuffling noise in rate stats can be amplified into
    multi-SGP swings that dwarf the actual player swap.

    Args:
        analysis: Output from category_gap_analysis().
        max_weight: Upper bound for any single category weight after
            normalization.  Default ``3.0`` means no category can
            contribute more than 3× the average non-punt category.

    Returns:
        Dict mapping category -> weight (float, 0.0 for punts).
    """
    weights: dict[str, float] = {}
    non_punt_values: list[float] = []

    for cat, info in analysis.items():
        if info["is_punt"]:
            weights[cat] = 0.0
        else:
            mv = info["marginal_value"]
            weights[cat] = mv
            non_punt_values.append(mv)

    # Normalize non-punt weights so mean is 1.0
    if non_punt_values:
        mean_mv = float(np.mean(non_punt_values))
        if mean_mv > 0:
            for cat in weights:
                if weights[cat] > 0:
                    weights[cat] = weights[cat] / mean_mv

    # Cap individual weights to prevent runaway amplification.
    # Rate stats with tiny standings gaps (e.g. AVG gap = 0.01 →
    # marginal = 100) can produce weights 100× the mean.  Capping at
    # max_weight keeps trade evaluations grounded in real player value.
    if max_weight > 0:
        for cat in weights:
            if weights[cat] > max_weight:
                weights[cat] = max_weight

    return weights
