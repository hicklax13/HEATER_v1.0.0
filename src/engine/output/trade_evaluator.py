"""Master trade evaluator — orchestrates Phase 1-5 engine modules.

Spec reference: Section 14 L11 (decision output / grading)
               Section 17 Phase 1 item 7 (trade surplus + grade)
               Section 17 Phase 2 items 9-12 (BMA, KDE, copula, MC)
               Section 17 Phase 4 items 17-20 (context engine)
               Section 17 Phase 5 items 21-24 (game theory + optimization)

Wires into:
  - src/engine/portfolio/valuation.py: z-scores, SGP, VORP
  - src/engine/portfolio/category_analysis.py: marginal SGP, gap analysis, punt detection
  - src/engine/portfolio/lineup_optimizer.py: LP optimizer, bench option value
  - src/engine/portfolio/copula.py: GaussianCopula for correlated sampling
  - src/engine/projections/projection_client.py: ROS projections
  - src/engine/projections/bayesian_blend.py: BMA for projection weighting
  - src/engine/monte_carlo/trade_simulator.py: paired MC simulation
  - src/engine/context/concentration.py: roster concentration risk (HHI)
  - src/engine/context/bench_value.py: enhanced bench option value
  - src/engine/context/injury_process.py: injury availability modeling
  - src/engine/game_theory/opponent_valuation.py: opponent market analysis
  - src/engine/game_theory/adverse_selection.py: Bayesian flaw discount
  - src/engine/game_theory/dynamic_programming.py: Bellman rollout
  - src/engine/game_theory/sensitivity.py: sensitivity + counter-offers
  - src/in_season.py: _roster_category_totals (aggregate roster stats)
  - src/valuation.py: LeagueConfig, SGPCalculator

The evaluation pipeline:
  Phase 1 (deterministic):
    1. Load player pool + standings
    2. Compute category gap analysis (punt detection, marginal elasticity)
    3. Compute pre-trade roster totals
    4. Compute post-trade roster totals
    5. Compute marginal SGP delta using elasticity weights
    6. Subtract bench option value if trade costs a bench slot (2-for-1)
    7. Grade the trade (A+ to F)

  Phase 2 (stochastic, when enabled):
    8. Build player marginals (KDE or Gaussian)
    9. Run paired Monte Carlo (10K sims) with copula-correlated sampling
    10. Overlay MC distribution metrics onto result

  Phase 4 (context, when enabled):
    11. Compute roster concentration risk (HHI delta + penalty)
    12. Enhanced bench option value (flexibility + injury cushion)
    13. Concentration risk flags

  Phase 5 (game theory, when enabled):
    14. Adverse selection discount on received players
    15. Opponent valuation + market clearing price
    16. Sensitivity analysis (category + player-level)
    17. Counter-offer suggestions for sub-optimal trades
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.database import load_league_standings
from src.engine.context.concentration import (
    compute_concentration_delta,
    team_exposure_breakdown,
)
from src.engine.game_theory.adverse_selection import compute_discount_for_trade
from src.engine.game_theory.opponent_valuation import (
    get_player_projections_from_pool,
    player_market_value,
)
from src.engine.game_theory.sensitivity import (
    trade_sensitivity_report,
)
from src.engine.monte_carlo.trade_simulator import (
    build_roster_stats,
    run_paired_monte_carlo,
)
from src.engine.portfolio.category_analysis import (
    build_standings_totals,
    category_gap_analysis,
    compute_category_weights_from_analysis,
)
from src.engine.portfolio.copula import GaussianCopula
from src.in_season import _roster_category_totals
from src.league_manager import get_free_agents
from src.valuation import LeagueConfig, SGPCalculator

try:
    from src.lineup_optimizer import PULP_AVAILABLE, ROSTER_SLOTS, LineupOptimizer
except ImportError:
    PULP_AVAILABLE = False
    ROSTER_SLOTS = {}
    LineupOptimizer = None  # type: ignore[assignment,misc]

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

_LC = LeagueConfig()
CATEGORIES: list[str] = _LC.all_categories
INVERSE_CATEGORIES: set[str] = _LC.inverse_stats

# Stat column mapping for computing roster totals
STAT_MAP: dict[str, str] = _LC.STAT_MAP

# Categories where replacement cost penalty applies (counting stats only).
# Rate stats (AVG, OBP, ERA, WHIP) are excluded — they are roster-aggregate
# and "best FA replacement" doesn't map cleanly to a counting-stat gap.
COUNTING_CATEGORIES: set[str] = _LC.counting_stats

# Discount factor for FA pool turnover (drops, injuries, call-ups, role changes).
# 0.5 means we assume half the unrecoverable gap will eventually become
# recoverable as the FA pool changes over the season.
FA_TURNOVER_DISCOUNT: float = 0.5

# Roster cap — 23 slots (18 starting + 5 bench) for Yahoo H2H categories.
ROSTER_CAP: int = 23


# ── Lineup-Constrained Helpers ───────────────────────────────────────


def _lineup_constrained_totals(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> tuple[dict[str, float], list[dict]]:
    """Compute category totals using only LP-optimized starters.

    Runs the PuLP-based LineupOptimizer to assign the best players to the
    18 starting slots (C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P).  Only the
    assigned starters' stats feed the category totals.

    Falls back to ``_roster_category_totals()`` (raw sum of all players)
    when PuLP is unavailable or the LP solver fails.

    Args:
        roster_ids: Player IDs on the roster.
        player_pool: Full player pool DataFrame.
        config: League configuration.

    Returns:
        Tuple of (totals_dict, assignments_list) where:
          - totals_dict: Same format as ``_roster_category_totals()``
          - assignments_list: LP assignment details; empty on fallback
    """
    if not PULP_AVAILABLE or LineupOptimizer is None:
        logger.warning("PuLP unavailable — falling back to raw roster totals")
        return _roster_category_totals(roster_ids, player_pool), []

    # Build roster DataFrame for the LP
    roster_df = player_pool[player_pool["player_id"].isin(roster_ids)].copy()
    if roster_df.empty:
        return _roster_category_totals(roster_ids, player_pool), []

    # LP expects 'player_name' column; player_pool from DB has 'name'
    if "player_name" not in roster_df.columns and "name" in roster_df.columns:
        roster_df = roster_df.rename(columns={"name": "player_name"})

    try:
        optimizer = LineupOptimizer(roster_df, config=config, roster_slots=ROSTER_SLOTS)
        result = optimizer.optimize_lineup()

        if result.get("status") != "Optimal":
            logger.warning(
                "LP solver status '%s' — falling back to raw roster totals",
                result.get("status"),
            )
            return _roster_category_totals(roster_ids, player_pool), []

        assignments = result.get("assignments", [])
        starter_ids = [a["player_id"] for a in assignments]

        # Compute totals from starters only
        totals = _roster_category_totals(starter_ids, player_pool)
        return totals, assignments

    except Exception as exc:
        logger.warning("LP optimizer failed (%s) — falling back to raw roster totals", exc)
        return _roster_category_totals(roster_ids, player_pool), []


def _find_drop_candidate(
    bench_ids: list[int],
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
) -> int | None:
    """Find the worst bench player to drop (lowest total SGP).

    Args:
        bench_ids: Player IDs of bench players.
        player_pool: Full player pool DataFrame.
        sgp_calc: SGP calculator for ranking.

    Returns:
        ``player_id`` of the drop candidate, or ``None`` if bench is empty.
    """
    if not bench_ids:
        return None

    worst_id: int | None = None
    worst_sgp = float("inf")

    for pid in bench_ids:
        match = player_pool[player_pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]
        sgp = sgp_calc.total_sgp(row)
        if sgp < worst_sgp:
            worst_sgp = sgp
            worst_id = int(pid)

    return worst_id


def _find_fa_pickup(
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
    need_hitter: bool = True,
    median_sgp_cap: float | None = None,
    exclude_ids: set[int] | None = None,
) -> int | None:
    """Find the best FA pickup to fill an open roster slot.

    Queries the free agent pool and returns the player with the highest
    total SGP that matches the needed type (hitter or pitcher).

    Args:
        player_pool: Full player pool DataFrame.
        sgp_calc: SGP calculator for ranking.
        need_hitter: ``True`` to pick a hitter, ``False`` for a pitcher.
        median_sgp_cap: When set, FAs above this SGP are excluded
            (prevents picking elite "FAs" when no roster data is loaded).
        exclude_ids: Player IDs to exclude (e.g., already picked up in
            this evaluation).

    Returns:
        ``player_id`` of the best FA, or ``None`` if none found.
    """
    fa_pool = get_free_agents(player_pool)
    if fa_pool.empty:
        return None

    # Filter by hitter/pitcher type
    if "is_hitter" in fa_pool.columns:
        if need_hitter:
            fa_pool = fa_pool[fa_pool["is_hitter"] == 1]
        else:
            fa_pool = fa_pool[fa_pool["is_hitter"] == 0]

    if fa_pool.empty:
        return None

    # Exclude already-picked IDs (for multi-pickup scenarios)
    if exclude_ids:
        fa_pool = fa_pool[~fa_pool["player_id"].isin(exclude_ids)]
        if fa_pool.empty:
            return None

    # Score each FA
    best_id: int | None = None
    best_sgp = float("-inf")

    for idx, row in fa_pool.iterrows():
        sgp = sgp_calc.total_sgp(row)
        if median_sgp_cap is not None and sgp > median_sgp_cap:
            continue
        if sgp > best_sgp:
            best_sgp = sgp
            best_id = int(row["player_id"])

    return best_id


def _compute_median_sgp_cap(
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
    is_hitter: bool,
) -> float:
    """Compute the median total SGP for a player subset.

    Used as a cap on FA pickups when no league roster data is loaded,
    to prevent picking up elite players who are actually rostered.

    Args:
        player_pool: Full player pool DataFrame.
        sgp_calc: SGP calculator.
        is_hitter: ``True`` for hitter median, ``False`` for pitcher median.

    Returns:
        Median total SGP of the subset (0.0 if empty).
    """
    if "is_hitter" in player_pool.columns:
        subset = player_pool[player_pool["is_hitter"] == (1 if is_hitter else 0)]
    else:
        subset = player_pool

    if subset.empty:
        return 0.0

    sgp_values = [sgp_calc.total_sgp(row) for _, row in subset.iterrows()]
    sgp_values.sort()
    mid = len(sgp_values) // 2
    if len(sgp_values) % 2 == 0 and len(sgp_values) > 1:
        return (sgp_values[mid - 1] + sgp_values[mid]) / 2
    return sgp_values[mid]


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
    enable_mc: bool = False,
    n_sims: int = 10_000,
    enable_context: bool = True,
    enable_game_theory: bool = True,
) -> dict[str, Any]:
    """Full trade evaluation using Phase 1-5 engine pipeline.

    Spec ref: Section 17 Phase 1 — MVP trade evaluation.
              Section 17 Phase 2 — MC simulation overlay.
              Section 17 Phase 5 — Game theory + optimization.

    Phase 1 Pipeline (always runs):
      1. Build pre-trade and post-trade roster ID lists
      2. Compute LP-constrained roster totals (only starters count)
         - Roster grows (1-for-2): auto-drop worst bench player
         - Roster shrinks (2-for-1): auto-pickup best FA
      3. Load standings and compute marginal elasticity weights
      4. Apply punt detection to zero-weight unachievable categories
      5. Compute weighted SGP delta across all categories
      6. Grade the trade

    Phase 2 Pipeline (when enable_mc=True):
      8. Build roster stat dicts for MC simulation
      9. Run paired Monte Carlo (10K sims) with Gaussian copula
      10. Overlay MC distribution metrics (mc_mean, mc_std, percentiles, etc.)

    Phase 5 Pipeline (when enable_game_theory=True):
      14. Adverse selection discount on received players
      15. Opponent valuation + market clearing price (when standings available)
      16. Sensitivity analysis (category ranking + breakeven)

    Args:
        giving_ids: Player IDs being traded away.
        receiving_ids: Player IDs being received.
        user_roster_ids: All player IDs on user's current roster.
        player_pool: Full player pool DataFrame (must have player_name or name col).
        config: League configuration.
        user_team_name: User's team name (for standings lookup).
        weeks_remaining: Weeks left in the season.
        enable_mc: Whether to run Phase 2 Monte Carlo simulation.
        n_sims: Number of MC simulations (default 10K).
        enable_context: Whether to run Phase 4 context analysis
            (concentration risk, enhanced bench value). Default True.
        enable_game_theory: Whether to run Phase 5 game theory analysis
            (adverse selection, opponent valuations, sensitivity). Default True.

    Returns:
        Dict with keys:
          - grade: Letter grade (A+ to F)
          - surplus_sgp: Net SGP gained (float)
          - category_impact: Per-category SGP change dict
          - category_analysis: Full gap analysis per category
          - punt_categories: List of punted category names
          - bench_cost: Always 0.0 (replaced by lineup constraint model)
          - lineup_constrained: True if LP was used, False if fallback
          - drop_candidate: Name of auto-dropped player (1-for-2) or None
          - fa_pickup: Name of auto-picked-up FA (2-for-1) or None
          - risk_flags: List of warning strings
          - verdict: "ACCEPT" or "DECLINE"
          - confidence_pct: Confidence percentage
          - before_totals: Pre-trade roster totals
          - after_totals: Post-trade roster totals
          - giving_players: Names of players traded away
          - receiving_players: Names of players received
          When enable_mc=True, also includes:
          - mc_mean, mc_std, mc_median, p5, p25, p75, p95
          - prob_positive, var5, cvar5, sharpe, confidence_interval
          When enable_game_theory=True, also includes:
          - adverse_selection: discount analysis dict
          - market_values: per-player market analysis
          - sensitivity_report: category sensitivity + breakeven
    """
    if config is None:
        config = LeagueConfig()

    sgp_calc = SGPCalculator(config)

    # Step 1: Build pre/post roster ID lists
    before_ids = list(user_roster_ids)
    after_ids = [pid for pid in before_ids if pid not in giving_ids] + list(receiving_ids)

    # Step 2: Compute LP-constrained roster category totals
    # Only starters' stats count — bench players produce zero standings value.
    # Also enforce roster cap: auto-drop for oversized, auto-pickup for undersized.
    lineup_constrained = False
    drop_candidate_name: str | None = None
    fa_pickup_name: str | None = None
    name_col = "player_name" if "player_name" in player_pool.columns else "name"

    players_lost = len(giving_ids)
    players_gained = len(receiving_ids)
    net_roster_growth = players_gained - players_lost

    # --- Before totals (always straightforward — current roster) ---
    before_totals, before_assignments = _lineup_constrained_totals(before_ids, player_pool, config)
    if before_assignments:
        lineup_constrained = True

    # --- After totals with roster cap enforcement ---
    if net_roster_growth > 0 and PULP_AVAILABLE and LineupOptimizer is not None:
        # Roster grows (e.g., 1-for-2): run LP on oversized roster, then drop
        after_totals_raw, after_assignments = _lineup_constrained_totals(after_ids, player_pool, config)
        if after_assignments:
            lineup_constrained = True
            starter_ids = {a["player_id"] for a in after_assignments}
            bench_ids = [pid for pid in after_ids if pid not in starter_ids]

            # Drop the worst bench player(s) to return to roster cap
            dropped_ids: list[int] = []
            for _ in range(net_roster_growth):
                remaining_bench = [b for b in bench_ids if b not in dropped_ids]
                drop_id = _find_drop_candidate(remaining_bench, player_pool, sgp_calc)
                if drop_id is not None:
                    dropped_ids.append(drop_id)

            # Record first drop candidate name for display
            if dropped_ids:
                match = player_pool[player_pool["player_id"] == dropped_ids[0]]
                if not match.empty:
                    drop_candidate_name = str(match.iloc[0].get(name_col, "Unknown"))

            # Remove dropped players from after_ids (for concentration risk etc.)
            after_ids = [pid for pid in after_ids if pid not in dropped_ids]

            # Starters unchanged — dropping bench doesn't affect LP solution
            after_totals = after_totals_raw
        else:
            # LP failed — fallback already returned raw totals
            after_totals = after_totals_raw

    elif net_roster_growth < 0 and PULP_AVAILABLE and LineupOptimizer is not None:
        # Roster shrinks (e.g., 2-for-1): pick up best FA(s) to restore cap
        fa_pool_check = get_free_agents(player_pool)
        use_median_cap = fa_pool_check.shape[0] > 500  # heuristic: no real rosters loaded

        picked_up_ids: set[int] = set()
        for i in range(abs(net_roster_growth)):
            # Determine type needed: check what type of player was lost
            lost_players = player_pool[player_pool["player_id"].isin(giving_ids)]
            if not lost_players.empty:
                # Try to replace with same type — count hitters vs pitchers lost
                hitters_lost = int(lost_players["is_hitter"].sum()) if "is_hitter" in lost_players.columns else 0
                pitchers_lost = len(lost_players) - hitters_lost
                # For each open slot, alternate hitter/pitcher based on what was lost more
                need_hitter = (hitters_lost - i) > (pitchers_lost - max(0, i - hitters_lost))
            else:
                need_hitter = True

            cap = None
            if use_median_cap:
                cap = _compute_median_sgp_cap(player_pool, sgp_calc, is_hitter=need_hitter)

            # Exclude both already-picked-up FAs and the current post-trade roster
            all_exclude = picked_up_ids | set(after_ids)
            fa_id = _find_fa_pickup(
                player_pool,
                sgp_calc,
                need_hitter=need_hitter,
                median_sgp_cap=cap,
                exclude_ids=all_exclude,
            )
            if fa_id is not None:
                after_ids.append(fa_id)
                picked_up_ids.add(fa_id)

                # Record first FA pickup name for display
                if fa_pickup_name is None:
                    match = player_pool[player_pool["player_id"] == fa_id]
                    if not match.empty:
                        fa_pickup_name = str(match.iloc[0].get(name_col, "Unknown"))

        # Now run LP on the restored roster
        after_totals, after_assignments = _lineup_constrained_totals(after_ids, player_pool, config)
        if after_assignments:
            lineup_constrained = True

    else:
        # Equal trade (or PuLP unavailable) — just run LP directly
        after_totals, after_assignments = _lineup_constrained_totals(after_ids, player_pool, config)
        if after_assignments:
            lineup_constrained = True

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

    # Step 6: Bench cost is zero — replaced by lineup-constrained LP
    # + roster cap enforcement (drop/pickup) in Step 2 above.
    bench_cost = 0.0

    # Step 6a: Concentration risk (Phase 4)
    concentration_data: dict[str, float] | None = None
    if enable_context:
        try:
            concentration_data = compute_concentration_delta(before_ids, after_ids, player_pool)
            # Apply concentration penalty delta to surplus
            penalty_delta = concentration_data.get("penalty_delta", 0.0)
            if penalty_delta > 0:
                total_surplus -= penalty_delta
        except Exception as exc:
            logger.warning("Concentration risk calculation failed: %s", exc)

    # Step 6b: Category replacement cost penalty
    # Scans the FA pool to penalize trades where lost production in counting
    # stats (especially scarce categories like SV) cannot be replaced.
    replacement_penalty = 0.0
    replacement_detail: dict[str, dict] = {}
    try:
        replacement_penalty, replacement_detail = _compute_replacement_penalty(
            before_totals=before_totals,
            after_totals=after_totals,
            player_pool=player_pool,
            config=config,
            category_weights=category_weights,
            punt_categories=punt_categories,
        )
        if replacement_penalty > 0:
            total_surplus -= replacement_penalty
    except Exception as exc:
        logger.warning("Replacement cost penalty calculation failed: %s", exc)

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

    # Phase 4 risk flags: concentration risk
    if enable_context and concentration_data:
        if concentration_data.get("hhi_delta", 0) > 0.05:
            # Significant increase in concentration
            after_exposure = team_exposure_breakdown(after_ids, player_pool)
            worst_team = max(after_exposure.items(), key=lambda x: x[1]["count"]) if after_exposure else None
            if worst_team and worst_team[1]["count"] >= 4:
                risk_flags.append(f"High roster concentration: {worst_team[1]['count']} players from {worst_team[0]}")

    # Verdict
    verdict = "ACCEPT" if total_surplus > 0 else "DECLINE"
    # Confidence: how far above/below 0 the surplus is, scaled to percentage
    confidence_pct = min(100.0, max(0.0, 50.0 + total_surplus * 25.0))

    # Get player names for display
    giving_players = _get_player_names(giving_ids, player_pool, name_col)
    receiving_players = _get_player_names(receiving_ids, player_pool, name_col)

    result = {
        "grade": trade_grade,
        "surplus_sgp": round(total_surplus, 3),
        "category_impact": category_impact,
        "category_analysis": cat_analysis,
        "punt_categories": punt_categories,
        "bench_cost": round(bench_cost, 3),
        "replacement_penalty": round(replacement_penalty, 3),
        "replacement_detail": replacement_detail,
        "risk_flags": risk_flags,
        "verdict": verdict,
        "confidence_pct": round(confidence_pct, 1),
        "before_totals": before_totals,
        "after_totals": after_totals,
        "giving_players": giving_players,
        "receiving_players": receiving_players,
        # Lineup constraint metadata
        "lineup_constrained": lineup_constrained,
        "drop_candidate": drop_candidate_name,
        "fa_pickup": fa_pickup_name,
        # Backward compat with existing UI
        "total_sgp_change": round(total_surplus, 3),
        "mc_mean": round(total_surplus, 3),
        "mc_std": 0.0,
    }

    # Phase 4: Context analysis keys
    if enable_context:
        if concentration_data:
            result["concentration_hhi_before"] = concentration_data["hhi_before"]
            result["concentration_hhi_after"] = concentration_data["hhi_after"]
            result["concentration_delta"] = concentration_data["hhi_delta"]
            result["concentration_penalty"] = concentration_data.get("penalty_delta", 0.0)
        # bench_option_detail retained as None for backward compat
        result["bench_option_detail"] = None

    # Phase 5: Game theory analysis
    if enable_game_theory:
        try:
            _apply_game_theory(
                result=result,
                giving_ids=giving_ids,
                receiving_ids=receiving_ids,
                player_pool=player_pool,
                all_team_totals=all_team_totals,
                user_team_name=user_team_name,
                category_impact=category_impact,
                category_weights=category_weights,
                config=config,
            )
        except Exception as exc:
            logger.warning("Game theory analysis failed: %s", exc)

    # Phase 2: Monte Carlo simulation overlay
    if enable_mc:
        try:
            mc_result = _run_mc_overlay(
                before_ids=before_ids,
                after_ids=after_ids,
                player_pool=player_pool,
                config=config,
                all_team_totals=all_team_totals,
                weeks_remaining=weeks_remaining,
                n_sims=n_sims,
            )
            # Overlay MC metrics onto result
            result.update(mc_result)
            # Use MC grade and verdict when available
            if "grade" in mc_result:
                result["grade"] = mc_result["grade"]
            if "verdict" in mc_result:
                result["verdict"] = mc_result["verdict"]
            if "confidence_pct" in mc_result:
                result["confidence_pct"] = mc_result["confidence_pct"]
        except Exception as exc:
            logger.warning("MC simulation failed, using Phase 1 result: %s", exc)
            result["mc_error"] = str(exc)

    return result


def _run_mc_overlay(
    before_ids: list[int],
    after_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    all_team_totals: dict[str, dict[str, float]],
    weeks_remaining: int,
    n_sims: int,
) -> dict[str, Any]:
    """Run Phase 2 Monte Carlo simulation and return overlay metrics.

    Spec ref: Section 17 Phase 2 — 10K paired MC sims.

    Builds roster stat dicts, creates a Gaussian copula for correlated
    sampling, then runs paired simulations to produce distributional
    metrics (mean, std, percentiles, VaR, Sharpe, etc.).

    Args:
        before_ids: Pre-trade roster player IDs.
        after_ids: Post-trade roster player IDs.
        player_pool: Full player pool DataFrame.
        config: League configuration.
        all_team_totals: Standings totals for context.
        weeks_remaining: Weeks left in season.
        n_sims: Number of MC simulations.

    Returns:
        Dict with MC metrics to overlay onto Phase 1 result.
    """
    # Build roster stat dicts for pre/post
    before_stats = build_roster_stats(before_ids, player_pool)
    after_stats = build_roster_stats(after_ids, player_pool)

    # Create copula (uses default correlation matrix)
    copula = GaussianCopula()

    # Run paired MC simulation
    mc_result = run_paired_monte_carlo(
        before_roster_stats=before_stats,
        after_roster_stats=after_stats,
        copula=copula,
        all_team_totals=all_team_totals if all_team_totals else None,
        sgp_denominators=dict(config.sgp_denominators),
        n_sims=n_sims,
        weeks_remaining=weeks_remaining,
    )

    return mc_result


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


def _compute_replacement_penalty(
    before_totals: dict[str, float],
    after_totals: dict[str, float],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    category_weights: dict[str, float],
    punt_categories: list[str],
) -> tuple[float, dict[str, dict]]:
    """Compute SGP penalty for categories where lost production is unrecoverable.

    For each counting-stat category where the trade makes the user worse:
      1. Compute raw loss (before - after)
      2. Find best available FA replacement in that category
      3. Unrecoverable gap = max(0, raw_loss - best_FA)
      4. SGP penalty = (unrecoverable / sgp_denom) * FA_TURNOVER_DISCOUNT

    Rate stats (AVG, ERA, WHIP) are skipped — they are roster-aggregate
    and the single-player replacement concept doesn't apply cleanly.

    Punted categories (weight=0) are skipped — no penalty for losing
    production you've strategically abandoned.

    Args:
        before_totals: Pre-trade roster category totals.
        after_totals: Post-trade roster category totals.
        player_pool: Full player pool DataFrame.
        config: League configuration with sgp_denominators.
        category_weights: Category weights (0 for punted categories).
        punt_categories: List of punted category names.

    Returns:
        Tuple of (total_penalty, detail) where:
          - total_penalty: Total SGP penalty (float, >= 0).
          - detail: Per-category breakdown dict. Each entry has keys:
              raw_loss, best_fa, best_fa_name, unrecoverable, sgp_penalty
              OR skipped (str) explaining why the category was excluded.
    """
    # Get the free agent pool (unrostered players)
    fa_pool = get_free_agents(player_pool)
    if fa_pool.empty:
        return 0.0, {}

    # Detect name column (player_pool may use "name" or "player_name")
    name_col = "player_name" if "player_name" in fa_pool.columns else "name"

    detail: dict[str, dict] = {}
    total_penalty = 0.0

    for cat in config.all_categories:
        # Skip rate stats — roster-aggregate, replacement concept doesn't apply
        if cat not in COUNTING_CATEGORIES:
            detail[cat] = {"skipped": "rate_stat"}
            continue

        # Skip punted categories — no penalty for strategic abandonment
        if cat in punt_categories:
            detail[cat] = {"skipped": "punted"}
            continue

        # Compute raw loss in this category
        raw_loss = before_totals.get(cat, 0) - after_totals.get(cat, 0)
        if raw_loss <= 0:
            detail[cat] = {"skipped": "no_loss"}
            continue

        # Map category to DataFrame column name
        col = STAT_MAP.get(cat)
        if col is None or col not in fa_pool.columns:
            detail[cat] = {"skipped": "no_data"}
            continue

        # Find best FA replacement in this category
        fa_vals = pd.to_numeric(fa_pool[col], errors="coerce").fillna(0)
        if fa_vals.empty or fa_vals.max() <= 0:
            best_fa_val = 0.0
            best_fa_name = "None available"
        else:
            best_fa_idx = fa_vals.idxmax()
            best_fa_val = float(fa_vals.loc[best_fa_idx])
            best_fa_name = str(fa_pool.loc[best_fa_idx].get(name_col, "Unknown"))

        # Compute unrecoverable gap
        unrecoverable = max(0.0, raw_loss - best_fa_val)

        # Convert to SGP and apply discount
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0
        sgp_penalty = (unrecoverable / denom) * FA_TURNOVER_DISCOUNT

        detail[cat] = {
            "raw_loss": round(float(raw_loss), 1),
            "best_fa": round(best_fa_val, 1),
            "best_fa_name": best_fa_name,
            "unrecoverable": round(unrecoverable, 1),
            "sgp_penalty": round(sgp_penalty, 3),
        }
        total_penalty += sgp_penalty

    return round(total_penalty, 3), detail


def _apply_game_theory(
    result: dict[str, Any],
    giving_ids: list[int],
    receiving_ids: list[int],
    player_pool: pd.DataFrame,
    all_team_totals: dict[str, dict[str, float]],
    user_team_name: str | None,
    category_impact: dict[str, float],
    category_weights: dict[str, float],
    config: LeagueConfig | None = None,
) -> None:
    """Apply Phase 5 game theory analysis and add keys to result.

    Spec ref: Section 17 Phase 5 — game theory + optimization.

    Runs three sub-analyses:
      1. Adverse selection discount (Bayesian flaw probability)
      2. Opponent valuation + market clearing price (when standings available)
      3. Sensitivity report (category ranking + breakeven analysis)

    Modifies result dict in-place by adding Phase 5 keys.

    Args:
        result: The result dict from Phase 1 (modified in-place).
        giving_ids: Players being given away.
        receiving_ids: Players being received.
        player_pool: Full player pool.
        all_team_totals: Standings totals (may be empty).
        user_team_name: User's team name.
        category_impact: Per-category SGP change.
        category_weights: Category weights.
        config: League configuration.
    """
    # 1. Adverse selection discount
    adverse_data = compute_discount_for_trade(
        receiving_player_count=len(receiving_ids),
    )
    result["adverse_selection"] = adverse_data

    # Add adverse selection risk flag if high risk
    if adverse_data["risk_level"] == "high":
        result["risk_flags"].append(
            f"High adverse selection risk: {adverse_data['p_flaw_given_offered']:.0%} probability of hidden flaw"
        )

    # 2. Opponent valuation + market clearing price
    if all_team_totals and user_team_name:
        sgp_denoms = dict(config.sgp_denominators) if config else None
        market_values: dict[str, dict] = {}

        for pid in giving_ids:
            proj = get_player_projections_from_pool(pid, player_pool)
            if proj:
                mv = player_market_value(
                    player_projections=proj,
                    all_team_totals=all_team_totals,
                    your_team_id=user_team_name,
                    sgp_denominators=sgp_denoms,
                )
                market_values[str(pid)] = mv

        result["market_values"] = market_values

    # 3. Sensitivity report (category-level)
    sensitivity = trade_sensitivity_report(
        category_impact=category_impact,
        category_weights=category_weights,
        surplus_sgp=result.get("surplus_sgp", 0.0),
    )
    result["sensitivity_report"] = sensitivity
