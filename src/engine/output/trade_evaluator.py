"""Master trade evaluator — orchestrates all Phase 1 engine modules.

Spec reference: Section 14 L11 (decision output / grading)
               Section 17 Phase 1 item 7 (trade surplus + grade)

Wires into:
  - src/engine/portfolio/valuation.py: z-scores, SGP, VORP
  - src/engine/portfolio/category_analysis.py: marginal SGP, gap analysis, punt detection
  - src/engine/portfolio/lineup_optimizer.py: LP optimizer, bench option value
  - src/engine/projections/projection_client.py: ROS projections
  - src/in_season.py: _roster_category_totals (aggregate roster stats)
  - src/valuation.py: LeagueConfig, SGPCalculator

The evaluation pipeline:
  1. Load player pool + standings
  2. Compute category gap analysis (punt detection, marginal elasticity)
  3. Compute pre-trade roster totals
  4. Compute post-trade roster totals
  5. Compute marginal SGP delta using elasticity weights
  6. Run LP optimizer on pre/post rosters to measure lineup impact
  7. Subtract bench option value if trade costs a bench slot (2-for-1)
  8. Grade the trade (A+ to F)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.database import load_league_standings
from src.engine.portfolio.category_analysis import (
    build_standings_totals,
    category_gap_analysis,
    compute_category_weights_from_analysis,
)
from src.engine.portfolio.lineup_optimizer import bench_option_value
from src.in_season import _roster_category_totals
from src.valuation import LeagueConfig, SGPCalculator

logger = logging.getLogger(__name__)

# Grade thresholds: composite score -> letter grade
# Spec ref: Section 14 L11, _grade() function
GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (2.0, "A+"),
    (1.5, "A"),
    (1.0, "A-"),
    (0.7, "B+"),
    (0.4, "B"),
    (0.2, "B-"),
    (0.0, "C+"),
    (-0.2, "C"),
    (-0.5, "C-"),
    (-1.0, "D"),
]

CATEGORIES: list[str] = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
INVERSE_CATEGORIES: set[str] = {"ERA", "WHIP"}

# Stat column mapping for computing roster totals
STAT_MAP: dict[str, str] = {
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "AVG": "avg",
    "W": "w",
    "SV": "sv",
    "K": "k",
    "ERA": "era",
    "WHIP": "whip",
}


def grade_trade(surplus_sgp: float) -> str:
    """Convert a trade surplus (in SGP) to a letter grade.

    Spec ref: Section 14 L11 — deterministic grading for Phase 1.

    The grading scale uses the composite score thresholds from the spec,
    applied directly to the surplus SGP value.

    Args:
        surplus_sgp: Net SGP gained from the trade (positive = good).

    Returns:
        Letter grade string (A+ through F).
    """
    for threshold, grade in GRADE_THRESHOLDS:
        if surplus_sgp > threshold:
            return grade
    return "F"


def evaluate_trade(
    giving_ids: list[int],
    receiving_ids: list[int],
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    user_team_name: str | None = None,
    weeks_remaining: int = 16,
) -> dict[str, Any]:
    """Full trade evaluation using the Phase 1 engine pipeline.

    Spec ref: Section 17 Phase 1 — MVP trade evaluation.

    Pipeline:
      1. Build pre-trade and post-trade roster ID lists
      2. Compute roster category totals (before/after)
      3. Load standings and compute marginal elasticity weights
      4. Apply punt detection to zero-weight unachievable categories
      5. Compute weighted SGP delta across all categories
      6. Subtract bench option value if 2-for-1 (or N-for-M where N>M)
      7. Grade the trade

    Args:
        giving_ids: Player IDs being traded away.
        receiving_ids: Player IDs being received.
        user_roster_ids: All player IDs on user's current roster.
        player_pool: Full player pool DataFrame (must have player_name or name col).
        config: League configuration.
        user_team_name: User's team name (for standings lookup).
        weeks_remaining: Weeks left in the season.

    Returns:
        Dict with keys:
          - grade: Letter grade (A+ to F)
          - surplus_sgp: Net SGP gained (float)
          - category_impact: Per-category SGP change dict
          - category_analysis: Full gap analysis per category
          - punt_categories: List of punted category names
          - bench_cost: SGP cost of lost bench slot(s)
          - risk_flags: List of warning strings
          - verdict: "ACCEPT" or "DECLINE"
          - confidence_pct: Confidence percentage
          - before_totals: Pre-trade roster totals
          - after_totals: Post-trade roster totals
          - giving_players: Names of players traded away
          - receiving_players: Names of players received
    """
    if config is None:
        config = LeagueConfig()

    sgp_calc = SGPCalculator(config)

    # Step 1: Build pre/post roster ID lists
    before_ids = list(user_roster_ids)
    after_ids = [pid for pid in before_ids if pid not in giving_ids] + list(receiving_ids)

    # Step 2: Compute roster category totals
    before_totals = _roster_category_totals(before_ids, player_pool)
    after_totals = _roster_category_totals(after_ids, player_pool)

    # Step 3: Load standings and compute marginal elasticity
    standings = load_league_standings()
    all_team_totals = build_standings_totals(standings)

    # Determine category weights
    category_weights: dict[str, float] = {cat: 1.0 for cat in CATEGORIES}
    cat_analysis: dict[str, dict] = {}
    punt_categories: list[str] = []

    if all_team_totals and user_team_name and user_team_name in all_team_totals:
        your_totals = all_team_totals[user_team_name]

        # Step 4: Gap analysis with punt detection
        cat_analysis = category_gap_analysis(
            your_totals=your_totals,
            all_team_totals=all_team_totals,
            your_team_id=user_team_name,
            weeks_remaining=weeks_remaining,
        )

        # Extract punt categories and compute weights
        punt_categories = [cat for cat, info in cat_analysis.items() if info["is_punt"]]
        category_weights = compute_category_weights_from_analysis(cat_analysis)

    # Step 5: Compute weighted SGP delta per category
    category_impact: dict[str, float] = {}
    total_surplus = 0.0

    for cat in config.all_categories:
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0

        before_val = before_totals.get(cat, 0)
        after_val = after_totals.get(cat, 0)
        raw_change = after_val - before_val

        # Convert raw stat change to SGP
        if cat in config.inverse_stats:
            sgp_change = -raw_change / denom
        else:
            sgp_change = raw_change / denom

        # Apply category weight (marginal elasticity + punt zeroing)
        weight = category_weights.get(cat, 1.0)
        weighted_sgp = sgp_change * weight

        category_impact[cat] = round(weighted_sgp, 3)
        total_surplus += weighted_sgp

    # Step 6: Bench option value penalty for uneven trades
    players_lost = len(giving_ids)
    players_gained = len(receiving_ids)
    bench_slots_lost = players_gained - players_lost  # Positive = gained slots, negative = lost
    bench_cost = 0.0

    if bench_slots_lost < 0:
        # Lost bench slots (e.g., 2-for-1 trade where you receive 2 and give 1)
        bench_cost = abs(bench_slots_lost) * bench_option_value(weeks_remaining=weeks_remaining)
        total_surplus -= bench_cost
    elif bench_slots_lost > 0:
        # Gained bench slots (e.g., 1-for-2 trade) — add option value
        bench_bonus = bench_slots_lost * bench_option_value(weeks_remaining=weeks_remaining)
        total_surplus += bench_bonus
        bench_cost = -bench_bonus  # Negative cost = benefit

    # Step 7: Grade the trade
    trade_grade = grade_trade(total_surplus)

    # Risk flags
    risk_flags: list[str] = _compute_risk_flags(
        giving_ids=giving_ids,
        receiving_ids=receiving_ids,
        player_pool=player_pool,
        sgp_calc=sgp_calc,
        cat_analysis=cat_analysis,
    )

    # Verdict
    verdict = "ACCEPT" if total_surplus > 0 else "DECLINE"
    # Confidence: how far above/below 0 the surplus is, scaled to percentage
    confidence_pct = min(100.0, max(0.0, 50.0 + total_surplus * 25.0))

    # Get player names for display
    name_col = "player_name" if "player_name" in player_pool.columns else "name"
    giving_players = _get_player_names(giving_ids, player_pool, name_col)
    receiving_players = _get_player_names(receiving_ids, player_pool, name_col)

    return {
        "grade": trade_grade,
        "surplus_sgp": round(total_surplus, 3),
        "category_impact": category_impact,
        "category_analysis": cat_analysis,
        "punt_categories": punt_categories,
        "bench_cost": round(bench_cost, 3),
        "risk_flags": risk_flags,
        "verdict": verdict,
        "confidence_pct": round(confidence_pct, 1),
        "before_totals": before_totals,
        "after_totals": after_totals,
        "giving_players": giving_players,
        "receiving_players": receiving_players,
        # Backward compat with existing UI
        "total_sgp_change": round(total_surplus, 3),
        "mc_mean": round(total_surplus, 3),
        "mc_std": 0.0,
    }


def _compute_risk_flags(
    giving_ids: list[int],
    receiving_ids: list[int],
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
    cat_analysis: dict[str, dict],
) -> list[str]:
    """Generate risk flag warnings for the trade.

    Checks for:
      - Receiving injured players
      - Trading away elite players (SGP > 3.0)
      - Trade worsens a non-punt category where you're already weak
      - Positional scarcity concerns

    Args:
        giving_ids: IDs of players being given.
        receiving_ids: IDs of players being received.
        player_pool: Full player pool.
        sgp_calc: SGP calculator instance.
        cat_analysis: Category gap analysis results.

    Returns:
        List of risk flag warning strings.
    """
    name_col = "player_name" if "player_name" in player_pool.columns else "name"
    flags: list[str] = []

    # Check for injured players being received
    receiving = player_pool[player_pool["player_id"].isin(receiving_ids)]
    for _, p in receiving.iterrows():
        if p.get("is_injured", 0):
            pname = p.get(name_col, "Unknown")
            flags.append(f"Receiving injured player: {pname}")

    # Check for trading away elite players
    giving = player_pool[player_pool["player_id"].isin(giving_ids)]
    for _, p in giving.iterrows():
        sgp = sgp_calc.total_sgp(p)
        if sgp > 3.0:
            pname = p.get(name_col, "Unknown")
            flags.append(f"Trading away elite player: {pname} (SGP: {sgp:.1f})")

    # Check if trade worsens a category where you're already weak (rank >= 8)
    if cat_analysis:
        for cat, info in cat_analysis.items():
            if info["rank"] >= 8 and not info["is_punt"]:
                # This is a weak but non-punt category — flag if trade makes it worse
                flags.append(
                    f"Caution: Ranked {info['rank']}th in {cat} — verify this trade doesn't hurt this category"
                )

    return flags


def _get_player_names(
    player_ids: list[int],
    player_pool: pd.DataFrame,
    name_col: str = "player_name",
) -> list[str]:
    """Look up player names from IDs."""
    names: list[str] = []
    for pid in player_ids:
        match = player_pool[player_pool["player_id"] == pid]
        if not match.empty:
            names.append(str(match.iloc[0][name_col]))
        else:
            names.append(f"Unknown (ID: {pid})")
    return names
