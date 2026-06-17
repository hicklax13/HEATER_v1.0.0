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
    14. Opponent valuation + market clearing price
    15. Sensitivity analysis (category + player-level)
    16. Counter-offer suggestions for sub-optimal trades
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.analytics_context import AnalyticsContext, DataQuality, ModuleStatus
from src.database import get_connection, load_league_standings
from src.engine.context.concentration import (
    compute_concentration_delta,
    team_exposure_breakdown,
)
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
from src.engine.output.types import GradeRange, MCOverlayResult, TradeResult
from src.engine.portfolio.category_analysis import (
    build_standings_totals,
    category_gap_analysis,
    compute_category_weights_from_analysis,
)
from src.engine.portfolio.copula import GaussianCopula
from src.engine.production import convergence as _convergence_diagnostics
from src.in_season import _roster_category_totals
from src.league_manager import get_free_agents
from src.valuation import LeagueConfig, SGPCalculator

try:
    from src.lineup_optimizer import PULP_AVAILABLE, ROSTER_SLOTS, LineupOptimizer
except ImportError:
    PULP_AVAILABLE = False
    ROSTER_SLOTS = {}
    LineupOptimizer = None  # type: ignore[assignment,misc]

from src.validation.constant_optimizer import load_constants

logger = logging.getLogger(__name__)

_CONSTANTS = load_constants()

# Grade thresholds: composite score -> letter grade
# Spec ref: Section 14 L11, _grade() function
GRADE_THRESHOLDS: list[tuple[float, str]] = [
    (_CONSTANTS.get("trade_grade_a_plus"), "A+"),
    (_CONSTANTS.get("trade_grade_a"), "A"),
    (1.0, "A-"),
    (0.7, "B+"),
    (0.4, "B"),
    (0.2, "B-"),
    (0.0, "C+"),
    (-0.2, "C"),
    (-0.5, "C-"),
    (-1.0, "D"),
]

# Resolve once at import; do not store as a long-lived module singleton.
# (BUG-010: SF-21 architectural directive.)
_LC_ONCE = LeagueConfig()
CATEGORIES: list[str] = _LC_ONCE.all_categories
INVERSE_CATEGORIES: set[str] = _LC_ONCE.inverse_stats

# Stat column mapping for computing roster totals
STAT_MAP: dict[str, str] = _LC_ONCE.STAT_MAP

# Categories where replacement cost penalty applies (counting stats only).
# Rate stats (AVG, OBP, ERA, WHIP) are excluded — they are roster-aggregate
# and "best FA replacement" doesn't map cleanly to a counting-stat gap.
COUNTING_CATEGORIES: set[str] = _LC_ONCE.counting_stats
del _LC_ONCE

# Discount factor for FA pool turnover (drops, injuries, call-ups, role changes).
# 0.5 means we assume half the unrecoverable gap will eventually become
# recoverable as the FA pool changes over the season.
FA_TURNOVER_DISCOUNT: float = _CONSTANTS.get("fa_turnover_discount")

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


def _compute_reshuffle_transparency(
    before_assignments: list[dict],
    after_assignments: list[dict],
    giving_ids: list[int],
    receiving_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    category_weights: dict[str, float],
) -> dict[str, Any]:
    """Decompose LP lineup changes into trade-swap vs reshuffle components.

    When the LP solver re-optimizes after a trade, it may bench or promote
    players who weren't involved in the trade.  This function identifies
    those reshuffles and quantifies how much of the total category change
    comes from lineup rearrangement vs the actual player swap.

    Args:
        before_assignments: LP starter assignments pre-trade.
        after_assignments: LP starter assignments post-trade.
        giving_ids: Player IDs traded away.
        receiving_ids: Player IDs received.
        player_pool: Full player pool DataFrame.
        config: League configuration.
        category_weights: Per-category marginal weights.

    Returns:
        Dict with keys:
          - promoted: List of {player_id, name} promoted from bench.
          - demoted: List of {player_id, name} demoted to bench.
          - reshuffle_sgp: SGP attributable to lineup reshuffling (float).
          - trade_swap_sgp: SGP attributable to actual player swap (float).
          - reshuffle_pct: What % of total surplus comes from reshuffling.
          - slot_changes: Per-slot breakdown of what changed.
    """
    if not before_assignments or not after_assignments:
        return {}

    name_col = "player_name" if "player_name" in player_pool.columns else "name"

    def _name(pid: int) -> str:
        match = player_pool[player_pool["player_id"] == pid]
        if match.empty:
            return f"Unknown ({pid})"
        return str(match.iloc[0].get(name_col, f"ID:{pid}"))

    traded_ids = set(giving_ids) | set(receiving_ids)
    before_starter_ids = {a["player_id"] for a in before_assignments}
    after_starter_ids = {a["player_id"] for a in after_assignments}

    # Identify promotions/demotions (excluding traded players)
    promoted_ids = (after_starter_ids - before_starter_ids) - set(receiving_ids)
    demoted_ids = (before_starter_ids - after_starter_ids) - set(giving_ids)

    promoted = [{"player_id": pid, "name": _name(pid)} for pid in promoted_ids]
    demoted = [{"player_id": pid, "name": _name(pid)} for pid in demoted_ids]

    # Compute reshuffle SGP: sum weighted SGP for promoted minus demoted
    # across all categories.  This isolates the LP rearrangement effect.
    sgp_denoms = config.sgp_denominators
    reshuffle_sgp = 0.0
    slot_changes: list[dict[str, Any]] = []

    # Build before/after slot maps: slot -> player_id
    before_slot_map = {a.get("slot", ""): a["player_id"] for a in before_assignments}
    after_slot_map = {a.get("slot", ""): a["player_id"] for a in after_assignments}

    for slot in set(list(before_slot_map.keys()) + list(after_slot_map.keys())):
        b_pid = before_slot_map.get(slot)
        a_pid = after_slot_map.get(slot)
        if b_pid == a_pid:
            continue  # unchanged slot

        b_name = _name(b_pid) if b_pid else "empty"
        a_name = _name(a_pid) if a_pid else "empty"

        # Classify: trade-swap (involves traded player) or reshuffle
        involves_trade = (b_pid in traded_ids) or (a_pid in traded_ids)
        change_type = "trade_swap" if involves_trade else "reshuffle"

        slot_changes.append(
            {
                "slot": slot,
                "before": b_name,
                "after": a_name,
                "type": change_type,
            }
        )

    # Compute per-category SGP from reshuffle moves only.
    # Reshuffle effect = stats of promoted players minus stats of demoted players,
    # converted to weighted SGP.
    reshuffle_sgp = 0.0
    for cat in config.all_categories:
        col = STAT_MAP.get(cat)
        if col is None:
            continue
        weight = category_weights.get(cat, 1.0)
        denom = sgp_denoms.get(cat, 1.0)
        if abs(denom) < 1e-9:
            continue

        # Sum stats for promoted players
        promoted_stat = 0.0
        for pid in promoted_ids:
            match = player_pool[player_pool["player_id"] == pid]
            if not match.empty:
                promoted_stat += float(match.iloc[0].get(col, 0) or 0)

        # Sum stats for demoted players
        demoted_stat = 0.0
        for pid in demoted_ids:
            match = player_pool[player_pool["player_id"] == pid]
            if not match.empty:
                demoted_stat += float(match.iloc[0].get(col, 0) or 0)

        # Rate stats: promoted/demoted affect the team average differently.
        # For simplicity, skip rate stats in the reshuffle computation —
        # the counting stat breakdown is most useful for transparency.
        if cat in config.rate_stats:
            continue

        raw_diff = promoted_stat - demoted_stat
        if cat in config.inverse_stats:
            raw_diff = -raw_diff  # fewer L = good

        cat_sgp = (raw_diff / denom) * weight
        reshuffle_sgp += cat_sgp

    reshuffle_sgp = round(reshuffle_sgp, 3)

    return {
        "promoted": promoted,
        "demoted": demoted,
        "reshuffle_sgp": reshuffle_sgp,
        "slot_changes": slot_changes,
    }


def _find_drop_candidate(
    bench_ids: list[int],
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
    receiving_hitters: bool | None = None,
) -> int | None:
    """Find the best player to drop from the bench.

    Uses a multi-factor scoring system instead of pure SGP:
    1. Roster balance: when receiving hitters, prefer dropping a hitter
    2. Positional value: DH/Util-only players are more droppable
    3. Category dead weight: 0 in a counting category = penalty
    4. Rate stat drag: below-average AVG/OBP = penalty
    5. Base SGP: lower SGP = more droppable

    Args:
        bench_ids: Player IDs of bench players.
        player_pool: Full player pool DataFrame.
        sgp_calc: SGP calculator for ranking.
        receiving_hitters: True if the trade receives hitters, False for pitchers,
            None if unknown. Used for roster balance preference.

    Returns:
        ``player_id`` of the drop candidate, or ``None`` if bench is empty.
    """
    if not bench_ids:
        return None

    # Score each bench player — LOWER score = more droppable
    scores: list[tuple[int, float]] = []

    for pid in bench_ids:
        match = player_pool[player_pool["player_id"] == pid]
        if match.empty:
            scores.append((int(pid), -999.0))
            continue
        row = match.iloc[0]

        # Base SGP
        sgp = sgp_calc.total_sgp(row)

        # Factor 1: Roster balance penalty
        # When receiving hitters, hitters on the bench get a drop bonus (lower score)
        is_hitter = int(row.get("is_hitter", 0)) == 1
        balance_adj = 0.0
        if receiving_hitters is True and is_hitter:
            balance_adj = -2.0  # Prefer dropping hitters when receiving hitters
        elif receiving_hitters is False and not is_hitter:
            balance_adj = -2.0  # Prefer dropping pitchers when receiving pitchers

        # Factor 2: Positional flexibility penalty
        # DH-only or single-position players are more droppable
        positions = str(row.get("positions", "")).upper()
        pos_list = [p.strip() for p in positions.split(",") if p.strip()]
        pos_adj = 0.0
        if is_hitter:
            if positions in ("DH", "UTIL", "") or (len(pos_list) == 1 and pos_list[0] == "DH"):
                pos_adj = -3.0  # DH-only = very droppable
            elif len(pos_list) == 1:
                pos_adj = -0.5  # Single position = slightly more droppable
            elif len(pos_list) >= 3:
                pos_adj = 1.0  # Multi-position = keep (flexibility)

        # Factor 3: Category dead weight
        # 0 in a counting category = dead weight in that entire category
        dead_cat_adj = 0.0
        if is_hitter:
            sb = float(row.get("sb", 0) or 0)
            if sb < 1:
                dead_cat_adj -= 1.5  # 0 SB = dead in stolen bases
            hr = float(row.get("hr", 0) or 0)
            if hr < 5:
                dead_cat_adj -= 0.5  # Very low HR

        # Factor 4: Rate stat drag
        # Below-average AVG/OBP hurts team totals — dropping improves the team
        rate_adj = 0.0
        if is_hitter:
            avg = float(row.get("avg", 0) or 0)
            obp = float(row.get("obp", 0) or 0)
            if 0 < avg < 0.245:
                rate_adj -= 1.0  # Below average AVG = drag
            if 0 < obp < 0.310:
                rate_adj -= 0.5  # Below average OBP = drag

        # Combined droppability score
        # Higher = keep, Lower = drop
        total_score = sgp + balance_adj + pos_adj + dead_cat_adj + rate_adj
        scores.append((int(pid), total_score))

    if not scores:
        return None

    # Drop the player with the LOWEST score
    scores.sort(key=lambda x: x[1])
    return scores[0][0]


# Markov FA discount (report Section B.7) — geometric convergence of
# top-available FA quality to a stationary level over the season.
_MARKOV_ALPHA: float = 0.85
_MARKOV_STATIONARY_FRACTION: float = 0.70


def _markov_fa_discount(
    week_index: int,
    alpha: float = _MARKOV_ALPHA,
    stationary_fraction: float = _MARKOV_STATIONARY_FRACTION,
) -> float:
    """Markov-decayed FA pool quality factor per report Section B.7.

    The FA pool's top-available quality starts at 1.0 on draft day and
    geometrically converges to ``stationary_fraction`` as good players
    get rostered over the season:

        factor = α^w + (1 − α^w) × stationary_fraction

    With α=0.85, stationary=0.70:
      Week 1:   1.000 (full draft-day quality)
      Week 5:   0.857 (still pretty fresh)
      Week 10:  0.759 (mid-season depletion)
      Week 18:  0.715 (near stationary)
      Week 26:  0.704 (essentially stationary)

    Used to discount FA pickups in 2-for-1 trade evaluation so trades
    made later in the season get less inflated by FA replacement value.
    """
    week = max(int(week_index), 0)
    geometric = alpha**week
    return float(geometric * 1.0 + (1.0 - geometric) * stationary_fraction)


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
    try:
        fa_pool = get_free_agents(player_pool)
    except Exception:
        # DB not initialized or league_rosters table missing — treat all as FA
        fa_pool = player_pool
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


# ── Secondary valuation diagnostics (report B.8 + B.5/C.2) ───────────


def _compute_vorp_delta(
    giving_ids: list[int],
    receiving_ids: list[int],
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
    config: LeagueConfig,
    name_col: str,
) -> tuple[float, dict[str, Any]]:
    """Secondary league-wide VORP/PRP sanity check per report Section B.8.

    The primary engine signal is the roster-context SGP surplus. This is the
    SECONDARY check the report calls for: a league-wide positional
    replacement-player delta that answers "is this trade fair league-wide?"
    independent of the user's specific roster. Computed as
    Σ VORP(receiving) − Σ VORP(giving) using the standard positional
    replacement levels (N_p-th ranked player per position).

    Returns (delta_vorp, detail). Fails safe to (0.0, {}) if replacement
    levels can't be computed (e.g. empty pool).
    """
    from src.valuation import compute_replacement_levels, compute_vorp

    try:
        replacement_levels = compute_replacement_levels(player_pool, config, sgp_calc)
    except Exception:
        logger.warning("VORP secondary check: replacement levels failed", exc_info=True)
        return 0.0, {}

    def _sum_vorp(ids: list[int]) -> tuple[float, dict[str, float]]:
        total = 0.0
        per: dict[str, float] = {}
        for pid in ids:
            match = player_pool[player_pool["player_id"] == pid]
            if match.empty:
                continue
            row = match.iloc[0]
            v = float(compute_vorp(row, sgp_calc, replacement_levels))
            per[str(row.get(name_col, pid))] = round(v, 3)
            total += v
        return total, per

    recv_total, recv_per = _sum_vorp(receiving_ids)
    give_total, give_per = _sum_vorp(giving_ids)
    delta = recv_total - give_total
    detail = {
        "receiving_vorp": recv_per,
        "giving_vorp": give_per,
        "receiving_total": round(recv_total, 3),
        "giving_total": round(give_total, 3),
        "method": "league-wide PRP/VORP (report B.8 secondary check)",
    }
    return round(delta, 3), detail


def _compute_gscore_delta(
    giving_ids: list[int],
    receiving_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    name_col: str,
) -> tuple[float, dict[str, Any]]:
    """Variance-aware G-score (Rosenof) trade delta per report B.5 / C.2.

    Surfaces the G-score valuation alongside the SGP surplus as a
    diagnostic. G-score adds the team-level weekly-variance term to the
    z-score denominator, so volatile boom-bust profiles get discounted
    relative to steady producers — exactly the H2H-Cats correction the
    report calls for. Computed as Σ g_composite(receiving) − Σ g_composite(giving).

    Returns (delta_g, detail). Fails safe to (0.0, {}).
    """
    from src.engine.portfolio.valuation import compute_player_gscores

    try:
        scored = compute_player_gscores(player_pool, config)
    except Exception:
        logger.warning("G-score diagnostic failed", exc_info=True)
        return 0.0, {}
    if "g_composite" not in scored.columns or "player_id" not in scored.columns:
        return 0.0, {}

    gmap = dict(zip(scored["player_id"], scored["g_composite"], strict=False))

    def _sum_g(ids: list[int]) -> tuple[float, dict[str, float]]:
        total = 0.0
        per: dict[str, float] = {}
        for pid in ids:
            g = float(gmap.get(pid, 0.0) or 0.0)
            match = player_pool[player_pool["player_id"] == pid]
            label = str(match.iloc[0].get(name_col, pid)) if not match.empty else str(pid)
            per[label] = round(g, 3)
            total += g
        return total, per

    recv_total, recv_per = _sum_g(receiving_ids)
    give_total, give_per = _sum_g(giving_ids)
    delta = recv_total - give_total
    detail = {
        "receiving_g": recv_per,
        "giving_g": give_per,
        "receiving_total": round(recv_total, 3),
        "giving_total": round(give_total, 3),
        "method": "rosenof-gscore (report B.5/C.2)",
    }
    return round(delta, 3), detail


def _build_drl_replacement_chain(
    dropped_ids: list[int],
    picked_up_ids: set[int],
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
    name_col: str,
) -> list[dict[str, Any]]:
    """Assemble the named DRL replacement chain per report Section B.7 / pseudocode.

    The report's pseudocode outputs a ``DRL_replacement_chain`` — the named
    bench/FA players that actually fill the slots a trade vacates. The
    roster-cap branches already perform the greedy priority-order fill (best
    FA first for roster-shrink trades; worst bench first for roster-grow
    trades); this just records the result so the UI can show "you'll pick up
    X off waivers / drop Y to the bench" rather than an abstract scalar.
    """
    chain: list[dict[str, Any]] = []
    for pid in picked_up_ids:
        match = player_pool[player_pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]
        chain.append(
            {
                "action": "pickup",
                "player_id": int(pid),
                "name": str(row.get(name_col, "Unknown")),
                "positions": str(row.get("positions", "")),
                "sgp": round(float(sgp_calc.total_sgp(row)), 3),
                "source": "FCFS waiver",
            }
        )
    for pid in dropped_ids:
        match = player_pool[player_pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]
        chain.append(
            {
                "action": "drop",
                "player_id": int(pid),
                "name": str(row.get(name_col, "Unknown")),
                "positions": str(row.get("positions", "")),
                "sgp": round(float(sgp_calc.total_sgp(row)), 3),
                "source": "bench",
            }
        )
    return chain


# Specialist cap (report Section H.2): no single received player should be
# credited with more than this fraction of a full category's standings range
# in any one category. Prevents SB-only / SV-only specialists from inflating
# a trade's surplus through one narrow category. 0.25 × (T-1) standings
# points = the report's recommended 25%-of-category-target cap.
_SPECIALIST_CAP_FRACTION: float = 0.25


def _compute_specialist_cap_penalty(
    receiving_ids: list[int],
    after_starter_ids: list[int],
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
    config: LeagueConfig,
    category_weights: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    """Per report Section H.2 — cap a single specialist's per-category credit.

    For each received player who actually starts post-trade, compute their
    per-category SGP. If a single category's weighted contribution exceeds
    ``_SPECIALIST_CAP_FRACTION × (num_teams − 1)`` standings points, the
    excess is "phantom value" the trade shouldn't be credited for — it is
    summed into a penalty subtracted from the surplus.

    Only received STARTERS are considered: a benched specialist contributes
    nothing to category totals, so penalising it would double-count.

    Returns (penalty, detail).
    """
    n_teams = int(getattr(config, "num_teams", 12))
    cap = _SPECIALIST_CAP_FRACTION * max(n_teams - 1, 1)
    starter_set = set(after_starter_ids)
    penalty = 0.0
    capped: dict[str, dict[str, dict[str, float]]] = {}

    for pid in receiving_ids:
        if pid not in starter_set:
            continue
        match = player_pool[player_pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]
        try:
            per_cat = sgp_calc.player_sgp(row)
        except Exception:
            continue
        name = str(row.get("player_name", row.get("name", pid)))
        for cat, raw_sgp in per_cat.items():
            weight = category_weights.get(cat, 1.0)
            weighted = float(raw_sgp) * weight
            if weighted > cap:
                excess = weighted - cap
                penalty += excess
                capped.setdefault(name, {})[cat] = {
                    "weighted_sgp": round(weighted, 2),
                    "cap": round(cap, 2),
                    "excess": round(excess, 2),
                }

    detail = {
        "cap_per_category": round(cap, 3),
        "fraction": _SPECIALIST_CAP_FRACTION,
        "capped": capped,
    }
    return round(penalty, 3), detail


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


def _compute_grade_range(surplus_sgp: float, uncertainty_sd: float = 0.8) -> GradeRange:
    """Compute grade range based on projection uncertainty.

    A surplus of 1.5 SGP with SD 0.8 means the true surplus could be
    0.7 to 2.3, spanning B to A+. Show the range, not just the point estimate.

    Args:
        surplus_sgp: Net SGP from the trade (positive = good).
        uncertainty_sd: Standard deviation of projection uncertainty.

    Returns:
        :class:`GradeRange` TypedDict with grade, grade_low, grade_high,
        confidence keys.
    """
    grade_center = grade_trade(surplus_sgp)
    grade_low = grade_trade(surplus_sgp - uncertainty_sd)
    grade_high = grade_trade(surplus_sgp + uncertainty_sd)

    # Confidence based on how narrow the range is
    if grade_center == grade_low == grade_high:
        confidence = "high"
    elif grade_low == grade_center or grade_high == grade_center:
        confidence = "medium"
    else:
        confidence = "low"

    return GradeRange(
        grade=grade_center,
        grade_low=grade_low,
        grade_high=grade_high,
        confidence=confidence,
    )


def evaluate_trade(
    giving_ids: list[int],
    receiving_ids: list[int],
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    user_team_name: str | None = None,
    weeks_remaining: int | None = None,
    enable_mc: bool = False,
    n_sims: int = 10_000,
    enable_injury_mc: bool = True,
    enable_context: bool = True,
    enable_game_theory: bool = True,
    apply_ytd_blend: bool = True,
    enable_weekly_matrix: bool = False,
    weekly_schedule: dict[int, str] | None = None,
    league_rosters: dict[str, list[int]] | None = None,
    enable_playoff_sim: bool = False,
    current_wins: dict[str, int] | None = None,
    playoff_n_sims: int = 20_000,
    full_league_schedule: dict[int, list[tuple[str, str]]] | None = None,
    correlate_playoff_categories: bool = True,
) -> TradeResult:
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
        weeks_remaining: Weeks left in the season. If None, computed
            dynamically from today's date.
        enable_mc: Whether to run Phase 2 Monte Carlo simulation.
        n_sims: Number of MC simulations (default 10K).
        enable_injury_mc: TE-E5 — when the Phase 2 MC runs (enable_mc=True),
            model per-player season availability (Weibull injury durations)
            so fragile/IL players widen the downside risk tail (CVaR5). Only
            takes effect when enable_mc=True; default True for deep single-
            trade evaluation. No effect on the bulk-scan path (enable_mc=False).
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
          - market_values: per-player market analysis
          - sensitivity_report: category sensitivity + breakeven
    """
    if weeks_remaining is None:
        from datetime import datetime, timedelta, timezone

        # 2026-05-17 Section 2 L1 fix: was 24 weeks. FourzynBurn = 26 weeks
        # (SF-42 fix missed this site). Every trade evaluation that doesn't
        # pass weeks_remaining explicitly was understating remaining-season
        # value by ~8%. 2026-05-17 Section 3 D5: source from LeagueConfig.
        _ET = timezone(timedelta(hours=-4))
        _season_start = datetime(2026, 3, 25, tzinfo=_ET)
        _now = datetime.now(_ET)
        _weeks_elapsed = max(0, (_now - _season_start).days // 7)
        weeks_remaining = max(1, (config or LeagueConfig()).season_weeks - _weeks_elapsed)

    if config is None:
        config = LeagueConfig()

    sgp_calc = SGPCalculator(config)

    # Blend pre-season projections with YTD actuals. Without this, a player's
    # role change (e.g., Suarez moving from closer to setup man) or hot/cold
    # start (Crochet's 7.58 ERA, Cruz's .345 hot start) is ignored when
    # computing SGP — trade evaluation uses stale pre-season numbers.
    if apply_ytd_blend:
        from src.projection_blending import apply_ytd_corrections

        player_pool = apply_ytd_corrections(player_pool)

    # --- AnalyticsContext: transparency spine for this evaluation ---
    ctx = AnalyticsContext(pipeline="trade_engine")
    ctx.stamp_data(
        "player_pool",
        DataQuality.LIVE if len(player_pool) > 100 else DataQuality.SAMPLE,
        record_count=len(player_pool),
    )

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

    # DRL replacement bookkeeping (report B.7 / C.9). Hoisted before the
    # roster-cap branches so they are always defined (equal and roster-grow
    # trades never enter the FA-pickup branch) and so the DRL replacement
    # chain can be assembled after the branches regardless of trade shape.
    dropped_ids: list[int] = []
    picked_up_ids: set[int] = set()

    # --- After totals with roster cap enforcement ---
    if net_roster_growth > 0 and PULP_AVAILABLE and LineupOptimizer is not None:
        # Roster grows (e.g., 1-for-2): run LP on oversized roster, then drop
        after_totals_raw, after_assignments = _lineup_constrained_totals(after_ids, player_pool, config)
        if after_assignments:
            lineup_constrained = True
            starter_ids = {a["player_id"] for a in after_assignments}
            bench_ids = [pid for pid in after_ids if pid not in starter_ids]

            # Drop the worst bench player(s) to return to roster cap
            # Determine if we're receiving hitters or pitchers
            _recv_hitters = None
            for _rid in receiving_ids:
                _rmatch = player_pool[player_pool["player_id"] == _rid]
                if not _rmatch.empty:
                    _recv_hitters = int(_rmatch.iloc[0].get("is_hitter", 0)) == 1
                    break  # Use first received player's type

            for _ in range(net_roster_growth):
                remaining_bench = [b for b in bench_ids if b not in dropped_ids]
                drop_id = _find_drop_candidate(remaining_bench, player_pool, sgp_calc, receiving_hitters=_recv_hitters)
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
        try:
            fa_pool_check = get_free_agents(player_pool)
        except Exception:
            # DB not initialized or league_rosters table missing — treat all as FA
            fa_pool_check = player_pool
        use_median_cap = fa_pool_check.shape[0] > 500  # heuristic: no real rosters loaded

        # Determine type needed: check what type of player was lost
        lost_players = player_pool[player_pool["player_id"].isin(giving_ids)]
        if not lost_players.empty:
            hitters_lost = int(lost_players["is_hitter"].sum()) if "is_hitter" in lost_players.columns else 0
            pitchers_lost = len(lost_players) - hitters_lost
        else:
            hitters_lost = abs(net_roster_growth)
            pitchers_lost = 0
        hitters_needed = hitters_lost
        pitchers_needed = pitchers_lost
        for i in range(abs(net_roster_growth)):
            # Track remaining hitters and pitchers needed separately
            need_hitter = hitters_needed > 0 and (pitchers_needed <= 0 or hitters_needed >= pitchers_needed)

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

            # Decrement the appropriate counter
            if need_hitter:
                hitters_needed -= 1
            else:
                pitchers_needed -= 1

        # Now run LP on the restored roster
        after_totals, after_assignments = _lineup_constrained_totals(after_ids, player_pool, config)
        if after_assignments:
            lineup_constrained = True

    else:
        # Equal trade (or PuLP unavailable) — just run LP directly
        after_totals, after_assignments = _lineup_constrained_totals(after_ids, player_pool, config)
        if after_assignments:
            lineup_constrained = True

    # Stamp LP module status
    if lineup_constrained:
        ctx.stamp_module("phase1_lineup_lp", ModuleStatus.EXECUTED, influence=0.9)
    else:
        ctx.stamp_module(
            "phase1_lineup_lp",
            ModuleStatus.FALLBACK,
            reason="LP unavailable — using raw roster totals",
            influence=0.3,
        )

    # Step 3: Load standings and compute marginal elasticity
    standings = load_league_standings()
    # Bug D (2026-05-23): when the caller supplies league_rosters, filter out
    # ghost teams (renamed/abandoned teams that linger in the standings cache).
    # When league_rosters is None we do NOT auto-query the DB — that approach
    # broke caller-supplied test scenarios where standings team names don't
    # match the DB's authoritative list (PR #113 CI failure). Pages that want
    # the ghost-team protection must pass league_rosters explicitly. The
    # Trade Analyzer page already does this (validated by
    # test_trade_analyzer_features_wired.py::test_page_loads_league_rosters).
    _valid_teams: set[str] | None = None
    if league_rosters:
        _candidate = set(league_rosters.keys())
        # Empty caller-supplied dict → no-filter (don't silently zero out
        # all_team_totals). The user-supplied None / empty is a "skip filter"
        # signal, not an "everything is ghost" signal.
        _valid_teams = _candidate if _candidate else None
    all_team_totals = build_standings_totals(standings, valid_teams=_valid_teams)

    # Validate that standings contain actual stat categories (R, HR, etc.)
    # not just W/L records. If they only have matchup records, fall back to
    # computing category totals from league rosters + season stats.
    _stat_cats = set(config.all_categories)
    _has_real_cats = False
    if all_team_totals:
        for _team_cats in all_team_totals.values():
            if _stat_cats & set(_team_cats.keys()):
                _has_real_cats = True
                break
    if not _has_real_cats:
        # Standings only have W/L records — compute from rosters instead
        try:
            from src.in_season import _roster_category_totals

            _conn = get_connection()
            try:
                _lr = pd.read_sql_query("SELECT team_name, player_id FROM league_rosters", _conn)
            finally:
                _conn.close()
            all_team_totals = {}
            if not _lr.empty:
                for _tn, _grp in _lr.groupby("team_name"):
                    _pids = _grp["player_id"].dropna().astype(int).tolist()
                    _totals = _roster_category_totals(_pids, player_pool)
                    if _totals:
                        all_team_totals[str(_tn)] = _totals
        except Exception as exc:
            # D4B-005: was logger.error; lowered to WARNING because this is
            # recoverable — evaluate_trade still produces a result with
            # uniform category weights (see _mod_cat.fallback_reason and the
            # DataQuality.MISSING stamp below). Operators see WARN; UI gets
            # the degraded signal via the analytics_context.
            logger.warning(
                "Failed to compute team totals from league_rosters (%s) — "
                "no strategic context, falling back to uniform category weights",
                exc,
                exc_info=True,
            )
            all_team_totals = {}

    # Stamp standings data quality
    if all_team_totals and _has_real_cats:
        ctx.stamp_data("league_standings", DataQuality.LIVE, record_count=len(all_team_totals))
    elif all_team_totals:
        ctx.stamp_data(
            "league_standings",
            DataQuality.SAMPLE,
            notes="Computed from roster projections (no category standings in DB)",
            record_count=len(all_team_totals),
        )
    else:
        ctx.stamp_data("league_standings", DataQuality.MISSING, notes="No standings loaded")

    # Determine category weights
    category_weights: dict[str, float] = {cat: 1.0 for cat in CATEGORIES}
    cat_analysis: dict[str, dict] = {}
    punt_categories: list[str] = []

    with ctx.track_module("phase1_category_analysis") as _mod_cat:
        if all_team_totals and user_team_name and user_team_name in all_team_totals:
            your_totals = all_team_totals[user_team_name]

            # Step 4: Gap analysis with punt detection.
            # SF-21: thread the live config so category_gap_analysis uses
            # standings-derived sgp_denominators (or whatever the caller
            # passed in) instead of the stale module-level singleton.
            cat_analysis = category_gap_analysis(
                your_totals=your_totals,
                all_team_totals=all_team_totals,
                your_team_id=user_team_name,
                weeks_remaining=weeks_remaining,
                config=config,
            )

            # Extract punt categories and compute weights
            punt_categories = [cat for cat, info in cat_analysis.items() if info["is_punt"]]
            category_weights = compute_category_weights_from_analysis(cat_analysis)
            _mod_cat.influence = 0.8
        else:
            _mod_cat.status = ModuleStatus.FALLBACK
            _mod_cat.fallback_reason = "No standings or team name — using equal weights"

    # Step 5: Compute weighted SGP delta per category
    # SF-25: per-category math routed through SGPCalculator.totals_sgp
    # (the V1-V6 unified entry point). totals_sgp({cat: val}, weights={cat: w})
    # is mathematically equivalent to (val/denom)*sign*w with sign=-1 for
    # inverse cats (L, ERA, WHIP) and pathological zero-denoms contributing 0.
    # Loop preserved because category_impact[cat] feeds downstream trade
    # explanation UI (counter-offer suggestions, sensitivity, grade output).
    category_impact: dict[str, float] = {}
    total_surplus = 0.0

    for cat in config.all_categories:
        before_val = before_totals.get(cat, 0)
        after_val = after_totals.get(cat, 0)
        raw_change = after_val - before_val

        # Convert raw stat change to SGP and apply category weight (marginal
        # elasticity + punt zeroing). totals_sgp handles sign-flip for
        # inverse cats and zero-denom skip internally.
        weighted_sgp = sgp_calc.totals_sgp(
            {cat: raw_change},
            weights={cat: category_weights.get(cat, 1.0)},
        )

        category_impact[cat] = round(weighted_sgp, 3)
        total_surplus += weighted_sgp

    # Step 6: Bench cost is zero — replaced by lineup-constrained LP
    # + roster cap enforcement (drop/pickup) in Step 2 above.
    bench_cost = 0.0
    ctx.stamp_module("phase1_sgp_delta", ModuleStatus.EXECUTED, influence=1.0)

    # Step 6a: Concentration risk (Phase 4)
    concentration_data: dict[str, float] | None = None
    with ctx.track_module("phase4_concentration") as _mod_conc:
        if enable_context:
            concentration_data = compute_concentration_delta(before_ids, after_ids, player_pool)
            # Apply concentration penalty delta to surplus
            penalty_delta = concentration_data.get("penalty_delta", 0.0)
            if penalty_delta > 0:
                total_surplus -= penalty_delta
            _mod_conc.influence = 0.3
        else:
            _mod_conc.status = ModuleStatus.DISABLED
            _mod_conc.fallback_reason = "enable_context=False"

    # Step 6b: Category replacement cost penalty
    # Scans the FA pool to penalize trades where lost production in counting
    # stats (especially scarce categories like SV) cannot be replaced.
    replacement_penalty = 0.0
    replacement_detail: dict[str, dict] = {}
    with ctx.track_module("phase1_replacement_penalty") as _mod_repl:
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
        _mod_repl.influence = 0.3

    # Step 6c: Positional flexibility penalty
    # Penalises trades that lose multi-position eligibility slots.
    flexibility_penalty = 0.0
    flexibility_detail: dict[str, Any] = {}
    with ctx.track_module("phase1_flexibility_penalty") as _mod_flex:
        flexibility_penalty, flexibility_detail = _compute_flexibility_penalty(
            giving_ids=giving_ids,
            receiving_ids=receiving_ids,
            player_pool=player_pool,
        )
        if flexibility_penalty > 0:
            total_surplus -= flexibility_penalty
        _mod_flex.influence = 0.2

    # Step 6d: IP-floor soft penalty
    # Per report Section B.6 + 2026-05-23 design Q1: Yahoo's 20 IP/week floor
    # converts shortfall to losses across pitching cats. Apply quadratic SGP
    # penalty kappa*(20 - IP_w)^2 when projected weekly IP falls below floor.
    # Delta semantics: only the post-trade marginal increase in penalty counts
    # against this trade. If user was already at 18 IP and stays at 18, no
    # additional penalty — that situation existed before the trade.
    ip_floor_penalty_delta = 0.0
    ip_floor_detail: dict[str, Any] = {}
    with ctx.track_module("phase1_ip_floor") as _mod_ip:
        before_starter_ids = [a["player_id"] for a in (before_assignments or [])]
        after_starter_ids = [a["player_id"] for a in (after_assignments or [])]
        # Bug F (2026-05-23): pass weeks_remaining (forward-looking horizon),
        # not season_weeks. The pool's `ip` column is full-season blended
        # projection; the helper subtracts ytd_ip to get ROS, then divides
        # by remaining weeks to produce the per-week IP that the floor checks.
        ip_floor_penalty_delta, ip_floor_detail = _compute_ip_floor_penalty(
            before_starter_ids=before_starter_ids,
            after_starter_ids=after_starter_ids,
            player_pool=player_pool,
            weeks_remaining=weeks_remaining,
        )
        if ip_floor_penalty_delta > 0:
            total_surplus -= ip_floor_penalty_delta
        _mod_ip.influence = 0.3

    # Step 6e: Markov FA discount (report Section B.7)
    # When the trade is 2-for-1, the engine fills the empty roster slot
    # with the current best FA. The current code captures that FA's full
    # static SGP, implicitly assuming draft-day-quality pool. But the
    # pool degrades over the season (good FAs get rostered). Apply a
    # Markov-decayed discount so trades made later get appropriately less
    # credit for FA replacement value.
    markov_fa_penalty = 0.0
    markov_fa_detail: dict[str, Any] = {}
    with ctx.track_module("phase1_markov_fa_discount") as _mod_markov:
        if picked_up_ids:
            # Compute current week index from weeks_remaining + total season
            # (LeagueConfig.season_weeks=26 for FourzynBurn)
            current_week = max(1, config.season_weeks - weeks_remaining)
            markov_factor = _markov_fa_discount(current_week)
            # Discount = (1 - markov_factor) × (sum of FA SGPs)
            # E.g. week 10 → factor 0.76 → discount 24% of FA SGP value
            fa_sgp_total = 0.0
            for pid in picked_up_ids:
                match = player_pool[player_pool["player_id"] == pid]
                if not match.empty:
                    fa_sgp_total += sgp_calc.total_sgp(match.iloc[0])
            markov_fa_penalty = max(0.0, (1.0 - markov_factor) * fa_sgp_total)
            markov_fa_detail = {
                "current_week": int(current_week),
                "markov_factor": round(markov_factor, 4),
                "alpha": _MARKOV_ALPHA,
                "stationary_fraction": _MARKOV_STATIONARY_FRACTION,
                "fa_pickup_count": len(picked_up_ids),
                "fa_sgp_total_static": round(fa_sgp_total, 3),
                "discount_penalty": round(markov_fa_penalty, 3),
            }
            if markov_fa_penalty > 0:
                total_surplus -= markov_fa_penalty
            _mod_markov.influence = 0.2
        else:
            # Gated module: the FA-discount premise only applies when the trade
            # shrinks the roster and the engine backfills with free agents. On
            # an equal/roster-grow trade it computes nothing — that's "doesn't
            # apply to this trade," NOT "required inputs missing." NOT_APPLICABLE
            # keeps it out of the quality score (vs DISABLED scoring it 0.0).
            _mod_markov.status = ModuleStatus.NOT_APPLICABLE
            _mod_markov.fallback_reason = "No FA pickups (not a roster-shrink trade)"

    # Step 6f: Specialist cap (report Section H.2)
    # A single received specialist (SB-only, SV-only) can rack up huge SGP in
    # one narrow category. Cap any single received starter's per-category
    # credit at 25% of a full category's standings range; subtract the excess.
    specialist_cap_penalty = 0.0
    specialist_cap_detail: dict[str, Any] = {}
    with ctx.track_module("phase1_specialist_cap") as _mod_spec:
        specialist_cap_penalty, specialist_cap_detail = _compute_specialist_cap_penalty(
            receiving_ids=receiving_ids,
            after_starter_ids=after_starter_ids,
            player_pool=player_pool,
            sgp_calc=sgp_calc,
            config=config,
            category_weights=category_weights,
        )
        if specialist_cap_penalty > 0:
            total_surplus -= specialist_cap_penalty
        # Always-run analysis, like replacement/flexibility/ip_floor: it
        # iterates every received starter against the cap. Finding no excess is
        # a ran-and-found-nothing result, so it stays EXECUTED (default) with a
        # fixed influence weight — not DISABLED ("required inputs missing").
        _mod_spec.influence = 0.2

    # Step 7: Grade the trade
    trade_grade = grade_trade(total_surplus)

    # Count categories that worsen (informational only — no auto-reject)
    worsened_cats = [
        cat
        for cat, delta in category_impact.items()
        if delta < -0.1  # meaningful worsening threshold
    ]

    # Risk flags
    risk_flags: list[str] = _compute_risk_flags(
        giving_ids=giving_ids,
        receiving_ids=receiving_ids,
        player_pool=player_pool,
        sgp_calc=sgp_calc,
        cat_analysis=cat_analysis,
        category_impact=category_impact,
    )

    # IP-floor risk flag: when post-trade weekly IP is below the Yahoo floor.
    # Fires regardless of whether the trade caused it (informational), but the
    # SGP penalty above only counts the trade-caused marginal worsening.
    if ip_floor_detail.get("below_floor"):
        _post_ip = ip_floor_detail.get("after_weekly_ip", 0.0)
        _delta = ip_floor_detail.get("delta_penalty", 0.0)
        if _delta > 0:
            risk_flags.append(
                f"Post-trade weekly IP ({_post_ip:.1f}) below {_IP_FLOOR_PER_WEEK:.0f} IP/week "
                f"Yahoo floor — pitching cats at risk; trade adds {_delta:.2f} SGP penalty"
            )
        else:
            risk_flags.append(
                f"Weekly IP ({_post_ip:.1f}) already below {_IP_FLOOR_PER_WEEK:.0f} IP/week "
                f"Yahoo floor; trade doesn't worsen it but pitching cats are at risk"
            )

    # Phase 4 risk flags: concentration risk
    if enable_context and concentration_data:
        if concentration_data.get("hhi_delta", 0) > _CONSTANTS.get("concentration_hhi_threshold"):
            # Significant increase in concentration
            after_exposure = team_exposure_breakdown(after_ids, player_pool)
            worst_team = max(after_exposure.items(), key=lambda x: x[1]["count"]) if after_exposure else None
            if worst_team and worst_team[1]["count"] >= 4:
                risk_flags.append(f"High roster concentration: {worst_team[1]['count']} players from {worst_team[0]}")

    # Verdict
    verdict = "ACCEPT" if total_surplus > 0 else "DECLINE"
    # Confidence: blend of surplus magnitude (how far from 0) and category
    # agreement (what fraction of non-zero hitting categories agree with
    # the verdict direction).  Pure surplus scaling can hit 100% on a
    # trade that wins AVG/OBP but loses HR/SB — the category disagreement
    # should temper confidence.
    _surplus_conf = min(100.0, max(0.0, 50.0 + total_surplus * _CONSTANTS.get("trade_confidence_scale")))
    # Count non-zero hitting/pitching categories that agree vs disagree
    _agree = 0
    _disagree = 0
    for _cat, _impact in category_impact.items():
        if abs(_impact) < 0.001:
            continue  # skip zero-impact categories
        if (total_surplus > 0 and _impact > 0) or (total_surplus <= 0 and _impact <= 0):
            _agree += 1
        else:
            _disagree += 1
    _total_nonzero = _agree + _disagree
    _agreement_ratio = _agree / _total_nonzero if _total_nonzero > 0 else 1.0
    # Blend: 70% surplus magnitude, 30% category agreement
    confidence_pct = _surplus_conf * (0.7 + 0.3 * _agreement_ratio)

    # Get player names for display
    giving_players = _get_player_names(giving_ids, player_pool, name_col)
    receiving_players = _get_player_names(receiving_ids, player_pool, name_col)

    # Add category-worsening warning to risk flags (informational, not blocking)
    if len(worsened_cats) >= 3:
        risk_flags.append(f"Caution: Trade worsens {len(worsened_cats)} categories ({', '.join(worsened_cats)}).")

    grade_range = _compute_grade_range(total_surplus)

    # LP reshuffle transparency: decompose surplus into trade-swap vs
    # lineup rearrangement components.
    reshuffle_info: dict[str, Any] = {}
    if lineup_constrained and before_assignments and after_assignments:
        reshuffle_info = _compute_reshuffle_transparency(
            before_assignments=before_assignments,
            after_assignments=after_assignments,
            giving_ids=giving_ids,
            receiving_ids=receiving_ids,
            player_pool=player_pool,
            config=config,
            category_weights=category_weights,
        )
        # Bug B (2026-05-23 validation): the previous condition
        # `if demoted and abs(total_surplus) > 0.01:` skipped the warning
        # whenever the LP did a PROMOTE-only reshuffle (a bench player got
        # elevated to a starting slot to replace a traded starter, no
        # existing starter demoted). In live validation, this hid a trade
        # whose reshuffle SGP (+3.97) dwarfed its actual surplus (-1.58):
        # most of the "trade value" was really the engine finally deploying
        # a bench bat (Max Muncy) that the user could promote without
        # trading at all. The warning now fires whenever the reshuffle is
        # large in EITHER direction, with text adapted per case.
        reshuffle_sgp = reshuffle_info.get("reshuffle_sgp", 0.0)
        promoted = reshuffle_info.get("promoted", [])
        demoted = reshuffle_info.get("demoted", [])
        if (promoted or demoted) and abs(total_surplus) > 0.01:
            reshuffle_pct = abs(reshuffle_sgp) / abs(total_surplus)
            reshuffle_info["reshuffle_pct"] = round(reshuffle_pct * 100, 1)
            if reshuffle_pct > 0.3:
                promoted_names = ", ".join(p["name"] for p in promoted)
                demoted_names = ", ".join(d["name"] for d in demoted)
                if promoted and demoted:
                    # Both directions — LP did a full swap
                    risk_flags.append(
                        f"Lineup reshuffle accounts for {reshuffle_pct:.0%} of surplus — "
                        f"promoting {promoted_names}, benching {demoted_names}"
                    )
                elif demoted:
                    # Demote-only (rare — usually means trade gave you a clear upgrade)
                    risk_flags.append(
                        f"Lineup reshuffle accounts for {reshuffle_pct:.0%} of surplus — benching {demoted_names}"
                    )
                else:
                    # Promote-only — the most decision-relevant case:
                    # "the trade looks good, but most of the value is the LP
                    # finally using a bench bat you could deploy without trading."
                    risk_flags.append(
                        f"Lineup reshuffle accounts for {reshuffle_pct:.0%} of surplus — "
                        f"promoting {promoted_names} from bench. "
                        f"You may capture most of this value by setting your lineup, "
                        f"without making the trade."
                    )

    # Secondary diagnostics (report B.8 + B.5/C.2 + B.7 pseudocode output).
    # These do not feed the grade — they are the report's sanity-check
    # outputs alongside the primary roster-context surplus.
    delta_vorp_prp, vorp_detail = _compute_vorp_delta(
        giving_ids, receiving_ids, player_pool, sgp_calc, config, name_col
    )
    delta_g_score, gscore_detail = _compute_gscore_delta(giving_ids, receiving_ids, player_pool, config, name_col)
    drl_replacement_chain = _build_drl_replacement_chain(dropped_ids, picked_up_ids, player_pool, sgp_calc, name_col)

    result: TradeResult = {
        "grade": trade_grade,
        "grade_range": grade_range,
        "surplus_sgp": round(total_surplus, 3),
        "category_impact": category_impact,
        "category_analysis": cat_analysis,
        "punt_categories": punt_categories,
        "bench_cost": round(bench_cost, 3),
        "replacement_penalty": round(replacement_penalty, 3),
        "replacement_detail": replacement_detail,
        "flexibility_penalty": round(flexibility_penalty, 3),
        "flexibility_detail": flexibility_detail,
        "ip_floor_penalty": round(ip_floor_penalty_delta, 3),
        "ip_floor_detail": ip_floor_detail,
        "markov_fa_penalty": round(markov_fa_penalty, 3),
        "markov_fa_detail": markov_fa_detail,
        "specialist_cap_penalty": round(specialist_cap_penalty, 3),
        "specialist_cap_detail": specialist_cap_detail,
        # Secondary valuation diagnostics (report B.8 + B.5/C.2 + B.7).
        "delta_vorp_prp": delta_vorp_prp,
        "vorp_detail": vorp_detail,
        "delta_g_score": delta_g_score,
        "gscore_detail": gscore_detail,
        "drl_replacement_chain": drl_replacement_chain,
        "risk_flags": risk_flags,
        "verdict": verdict,
        "compliant": True,  # Legacy field — no longer enforced
        "confidence_pct": round(confidence_pct, 1),
        "before_totals": before_totals,
        "after_totals": after_totals,
        "giving_players": giving_players,
        "receiving_players": receiving_players,
        # Lineup constraint metadata
        "lineup_constrained": lineup_constrained,
        "drop_candidate": drop_candidate_name,
        "fa_pickup": fa_pickup_name,
        "reshuffle": reshuffle_info,
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
    with ctx.track_module("phase5_game_theory") as _mod_gt:
        if enable_game_theory:
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
            _mod_gt.influence = 0.2
        else:
            _mod_gt.status = ModuleStatus.DISABLED
            _mod_gt.fallback_reason = "enable_game_theory=False"

    # Phase 2: Monte Carlo simulation overlay
    #
    # Bug A (2026-05-23 validation): Phase 1 weighted SGP is the AUTHORITY for
    # grade/verdict/confidence_pct. Prior code blindly overrode these from MC,
    # producing user-facing contradictions like grade=B alongside surplus_sgp=-1.58
    # (Phase 1 said DECLINE, MC said ACCEPT, user only saw the MC verdict).
    #
    # _run_mc_overlay now renames its grade/verdict/confidence_pct →
    # mc_grade/mc_verdict/mc_confidence_pct BEFORE returning, so the
    # result.update(mc_result) call is safe and cannot clobber Phase 1.
    # The MC contribution is exclusively a RISK DIAGNOSTIC
    # (mc_mean/std/percentiles/CVaR5/sharpe/prob_positive) — never the grade.
    with ctx.track_module("phase2_monte_carlo") as _mod_mc:
        if enable_mc:
            mc_result = _run_mc_overlay(
                before_ids=before_ids,
                after_ids=after_ids,
                player_pool=player_pool,
                config=config,
                all_team_totals=all_team_totals,
                weeks_remaining=weeks_remaining,
                n_sims=n_sims,
                enable_injury_mc=enable_injury_mc,
            )
            # Overlay MC risk diagnostics onto result.  By contract,
            # _run_mc_overlay has stripped grade/verdict/confidence_pct and
            # exposed them as mc_grade/mc_verdict/mc_confidence_pct.
            result.update(mc_result)
            _mod_mc.influence = 0.7
        else:
            _mod_mc.status = ModuleStatus.DISABLED
            _mod_mc.fallback_reason = "enable_mc=False (default)"

    # Feature 2 (2026-05-23): Weekly H2H win-probability matrix.
    # Per report Section B.5 — for each remaining matchup week, the
    # probability of winning each category against the scheduled opponent.
    # Surfaces playoff-week asymmetries that the surplus_sgp scalar can't
    # capture (e.g. "this trade helps weeks 8-12 but hurts weeks 24-26").
    # Opt-in via enable_weekly_matrix=True; requires weekly_schedule
    # (Yahoo schedule lookup) and league_rosters (for opponent means).
    with ctx.track_module("phase6_weekly_matrix") as _mod_wm:
        if enable_weekly_matrix:
            if weekly_schedule and league_rosters:
                from src.engine.output.weekly_matrix import compute_trade_weekly_delta

                wm = compute_trade_weekly_delta(
                    before_roster_ids=before_ids,
                    after_roster_ids=after_ids,
                    player_pool=player_pool,
                    schedule=weekly_schedule,
                    all_team_rosters=league_rosters,
                    config=config,
                )
                result["weekly_matrix"] = wm
                _mod_wm.influence = 0.6
            else:
                _mod_wm.status = ModuleStatus.FALLBACK
                _mod_wm.fallback_reason = (
                    "enable_weekly_matrix=True but weekly_schedule or league_rosters is None — skipping matrix"
                )
        else:
            _mod_wm.status = ModuleStatus.DISABLED
            _mod_wm.fallback_reason = "enable_weekly_matrix=False (default)"

    # Feature 3 (2026-05-23): Playoff + championship probability bracket sim.
    # Per report Section B.10 + Q(a) — the engine's PRIMARY objective per
    # the report. ΔΠ_playoff and ΔΠ_champ tell the user how this trade
    # changes their actual title odds, not just SGP-flavored scalars.
    # Opt-in via enable_playoff_sim=True; requires weekly_schedule,
    # league_rosters, AND current_wins. Default off so Trade Finder
    # bulk scans stay fast (20K sims ~0.5-2s per trade).
    with ctx.track_module("phase7_playoff_sim") as _mod_ps:
        if enable_playoff_sim:
            if weekly_schedule and league_rosters and current_wins and user_team_name:
                from src.engine.output.playoff_sim import simulate_trade_playoff_delta

                # TE-E3: make the schedule-aware playoff sim (Path A) the DEFAULT.
                # When the caller didn't supply full_league_schedule, auto-load it
                # from the Yahoo-cache DB reader so Path A (per-team-per-week
                # opponent matchups, strength-of-schedule aware) runs whenever the
                # cache has a schedule. Path B (Binomial league-average) remains
                # the documented fallback when no full schedule is available.
                if full_league_schedule is None:
                    try:
                        from src.database import load_league_schedule_full

                        full_league_schedule = load_league_schedule_full() or None
                    except Exception:
                        logger.warning(
                            "TE-E3: load_league_schedule_full() failed — "
                            "playoff sim falls back to Binomial league-average (Path B)",
                            exc_info=True,
                        )
                        full_league_schedule = None

                # #S1365 (2026-06-16): the playoff sim must simulate only the
                # REMAINING weeks. load_league_schedule() returns the user's FULL
                # season schedule (~24 matchup weeks), so passing it straight
                # through set n_weeks_remaining = the whole season — projecting
                # impossible 30+-win finals and burying a competitive user at
                # ~0% playoff odds. Cap to the last `weeks_remaining` weeks (the
                # authoritative forward horizon). Path A opponent weeks derive
                # from user_schedule's keys, so this corrects both sides at once.
                _ps_schedule = weekly_schedule
                if weekly_schedule and weeks_remaining and len(weekly_schedule) > int(weeks_remaining):
                    _future_weeks = sorted(weekly_schedule.keys())[-int(weeks_remaining) :]
                    _ps_schedule = {w: weekly_schedule[w] for w in _future_weeks}

                ps = simulate_trade_playoff_delta(
                    before_roster_ids=before_ids,
                    after_roster_ids=after_ids,
                    user_team_name=user_team_name,
                    all_team_rosters=league_rosters,
                    user_schedule=_ps_schedule,
                    current_wins=current_wins,
                    player_pool=player_pool,
                    config=config,
                    n_sims=playoff_n_sims,
                    full_league_schedule=full_league_schedule,
                    correlate_categories=correlate_playoff_categories,
                )
                result["playoff_sim"] = ps
                # Also surface delta fields at top level for headline display
                result["delta_playoff_prob"] = ps["delta_playoff_prob"]
                result["delta_champ_prob"] = ps["delta_champ_prob"]
                _mod_ps.influence = 0.8  # primary objective per report
            else:
                _mod_ps.status = ModuleStatus.FALLBACK
                _mod_ps.fallback_reason = (
                    "enable_playoff_sim=True but missing one of: "
                    "weekly_schedule, league_rosters, current_wins, user_team_name"
                )
        else:
            _mod_ps.status = ModuleStatus.DISABLED
            _mod_ps.fallback_reason = "enable_playoff_sim=False (default)"

    # Attach transparency context for UI badge rendering
    result["analytics_context"] = ctx
    return result


def _run_mc_overlay(
    before_ids: list[int],
    after_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    all_team_totals: dict[str, dict[str, float]],
    weeks_remaining: int,
    n_sims: int,
    enable_injury_mc: bool = False,
) -> MCOverlayResult:
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
        enable_injury_mc=enable_injury_mc,
    )

    # BUG-007 fix: assess MC convergence quality and attach to result.
    # check_convergence returns a dict with effective sample size, R-hat,
    # running-mean stability, and a categorical `quality` field.  Without
    # this, a 10K-sim run with effective sample size 50 (highly
    # autocorrelated paired antithetic) would still appear "sharp" via
    # mc_std alone, and the trade UI would have no way to flag it.
    try:
        import numpy as np

        surplus_dist = mc_result.get("surplus_distribution")
        if surplus_dist is not None and len(surplus_dist) > 0:
            conv = _convergence_diagnostics.check_convergence(np.asarray(surplus_dist))
            mc_result["convergence_quality"] = conv.get("quality", "not_assessed")
            mc_result["convergence_ess"] = float(conv.get("ess", float("nan")))
            mc_result["convergence_rhat"] = float(conv.get("rhat", float("nan")))
            if mc_result["convergence_quality"] in ("marginal", "poor"):
                mc_result.setdefault("risk_flags", []).append(
                    f"MC convergence: {mc_result['convergence_quality']} "
                    f"(ESS={mc_result['convergence_ess']:.0f}, "
                    f"R-hat={mc_result['convergence_rhat']:.3f})"
                )
        else:
            mc_result["convergence_quality"] = "not_assessed"
            mc_result["convergence_ess"] = float("nan")
            mc_result["convergence_rhat"] = float("nan")
    except Exception:
        logger.warning("MC convergence check failed", exc_info=True)
        mc_result["convergence_quality"] = "not_assessed"
        mc_result["convergence_ess"] = float("nan")
        mc_result["convergence_rhat"] = float("nan")

    # Bug A (2026-05-23): rename grade/verdict/confidence_pct so the
    # caller's result.update(mc_result) cannot clobber Phase 1's
    # authoritative grade. These keys remain accessible as
    # mc_grade/mc_verdict/mc_confidence_pct for diagnostics, comparison
    # against the Phase 1 grade, and future hybrid graders.
    if "grade" in mc_result:
        mc_result["mc_grade"] = mc_result.pop("grade")
    if "verdict" in mc_result:
        mc_result["mc_verdict"] = mc_result.pop("verdict")
    if "confidence_pct" in mc_result:
        mc_result["mc_confidence_pct"] = mc_result.pop("confidence_pct")

    return mc_result


def _compute_risk_flags(
    giving_ids: list[int],
    receiving_ids: list[int],
    player_pool: pd.DataFrame,
    sgp_calc: SGPCalculator,
    cat_analysis: dict[str, dict],
    category_impact: dict[str, float] | None = None,
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
        category_impact: Per-category SGP change from the trade.  When
            provided, weak-category warnings are only emitted for categories
            where the trade actually hurts (negative impact).

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
        if sgp > 6.0:
            pname = p.get(name_col, "Unknown")
            flags.append(f"Trading away elite player: {pname} (SGP: {sgp:.2f})")

    # Check if trade worsens a category where you're already weak (rank >= 8)
    if cat_analysis:
        for cat, info in cat_analysis.items():
            if info["rank"] >= 8 and not info["is_punt"]:
                # Only flag if the trade actually hurts this category
                if category_impact and category_impact.get(cat, 0) >= 0:
                    continue  # trade doesn't hurt this category
                flags.append(f"Caution: Ranked {info['rank']}th in {cat} — this trade hurts an already weak category")

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

        # Skip L — it's an inverse counting stat.  Losing L production
        # (fewer losses) is *good*, so penalizing the "loss" is wrong.
        if cat == "L":
            detail[cat] = {"skipped": "inverse_counting_stat"}
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


# ── IP-floor soft penalty (report Section B.6) ──────────────────────

# FourzynBurn / Yahoo H2H Categories default: 20 IP/week. Below this,
# Yahoo converts the shortfall to losses across pitching counting cats
# (W, K, SV) and inflates ERA/WHIP. Confirmed via 2026-05-23 design Q1.
_IP_FLOOR_PER_WEEK: float = 20.0

# Quadratic penalty coefficient. Per report calibration: at 50% shortfall
# (10 IP/week, i.e. 10 IP below floor), penalty = 0.05 * 10^2 = 5.0 SGP
# — "forfeit-equivalent" against the typical ±1-3 SGP trade-surplus scale.
# At 30% shortfall (14 IP/week), penalty = 0.05 * 6^2 = 1.8 SGP — meaningful
# but not catastrophic. Steep at extremes, gentle near the threshold.
_IP_FLOOR_KAPPA: float = 0.05


def _ip_floor_penalty(weekly_ip: float) -> float:
    """Quadratic SGP penalty per report Section B.6.

    Pen_IP(IP_w) = 0                          if IP_w >= 20
                 = kappa * (20 - IP_w)^2      if IP_w <  20
    """
    if weekly_ip >= _IP_FLOOR_PER_WEEK:
        return 0.0
    shortfall = _IP_FLOOR_PER_WEEK - weekly_ip
    return _IP_FLOOR_KAPPA * shortfall * shortfall


def _compute_weekly_ip(
    starter_ids: list[int],
    player_pool: pd.DataFrame,
    weeks_remaining: int = 26,
) -> float:
    """ROS-aware average IP per remaining matchup week.

    Bug F (2026-05-23 validation): the prior implementation divided the
    full-season ``ip`` projection by ``season_weeks=26`` and got 8.11 IP/week
    mid-season — wrong because the ``ip`` column INCLUDES YTD already accrued.
    The correct semantic is "expected IP for the REMAINING weeks":

        ROS_IP = max(0, ip - ytd_ip)
        weekly = sum(ROS_IP across pitchers) / weeks_remaining

    Subtracting ``ytd_ip`` removes IP that was already accumulated (and
    won't recur), then dividing by ``weeks_remaining`` gives the forward-
    looking per-week IP. Falls back to ``ip / weeks_remaining`` when the
    ``ytd_ip`` column is missing (e.g., off-season / preseason mode where
    the full ``ip`` IS the ROS projection).

    Hitters and zero-IP rows contribute nothing.
    """
    if not starter_ids:
        return 0.0
    matches = player_pool[player_pool["player_id"].isin(starter_ids)]
    if matches.empty:
        return 0.0
    if "is_hitter" in matches.columns:
        pitchers = matches[matches["is_hitter"] == 0]
    else:
        pitchers = matches
    if pitchers.empty or "ip" not in pitchers.columns:
        return 0.0
    full_ip = pd.to_numeric(pitchers["ip"], errors="coerce").fillna(0)
    if "ytd_ip" in pitchers.columns:
        ytd_ip = pd.to_numeric(pitchers["ytd_ip"], errors="coerce").fillna(0)
        # ROS = full - ytd, floored at 0 in case mid-season YTD exceeded the
        # blended projection (over-performers can show ytd > ip).
        ros_ip = (full_ip - ytd_ip).clip(lower=0)
    else:
        # No ytd column — assume `ip` is already ROS (preseason mode).
        ros_ip = full_ip
    return float(ros_ip.sum()) / max(weeks_remaining, 1)


def _compute_ip_floor_penalty(
    before_starter_ids: list[int],
    after_starter_ids: list[int],
    player_pool: pd.DataFrame,
    weeks_remaining: int = 26,
) -> tuple[float, dict[str, Any]]:
    """SGP delta penalty when a trade pushes weekly IP below the floor.

    Bug F: caller must pass ``weeks_remaining`` (the forward-looking
    horizon), NOT ``season_weeks`` (the full season). The IP-floor check
    is about per-week IP for the rest of the season, not the season average.

    Delta semantics: returns max(0, after_penalty - before_penalty). If the
    user was already below the floor pre-trade and the trade doesn't worsen
    the situation, this returns 0 (the prior under-floor state is not the
    trade's fault). If the trade IMPROVES weekly IP, this also returns 0
    (negative deltas don't become bonuses here).

    Returns:
        (penalty_delta, detail_dict). penalty_delta >= 0.
    """
    before_weekly = _compute_weekly_ip(before_starter_ids, player_pool, weeks_remaining)
    after_weekly = _compute_weekly_ip(after_starter_ids, player_pool, weeks_remaining)
    before_pen = _ip_floor_penalty(before_weekly)
    after_pen = _ip_floor_penalty(after_weekly)
    # Only count the marginal increase in penalty caused by the trade.
    # A trade that maintains or improves the IP situation contributes 0.
    delta = max(0.0, after_pen - before_pen)

    detail: dict[str, Any] = {
        "threshold_ip_per_week": _IP_FLOOR_PER_WEEK,
        "kappa": _IP_FLOOR_KAPPA,
        "weeks_remaining": int(weeks_remaining),
        "before_weekly_ip": round(before_weekly, 2),
        "after_weekly_ip": round(after_weekly, 2),
        "before_penalty": round(before_pen, 3),
        "after_penalty": round(after_pen, 3),
        "delta_penalty": round(delta, 3),
        "below_floor": after_weekly < _IP_FLOOR_PER_WEEK,
    }
    return round(delta, 3), detail


# ── Positional flexibility penalty ──────────────────────────────────

# Positions that matter for lineup construction (excludes Util/DH/BN).
_REAL_POSITIONS = {"C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"}

# SGP penalty per net position-slot lost.  Calibrated conservatively:
# losing one positional slot is roughly equivalent to losing ~0.15 SGP
# of lineup flexibility (based on typical LP solver re-optimization).
_FLEXIBILITY_SGP_PER_SLOT = 0.15


def _compute_flexibility_penalty(
    giving_ids: list[int],
    receiving_ids: list[int],
    player_pool: pd.DataFrame,
) -> tuple[float, dict[str, Any]]:
    """Compute SGP penalty for net loss of positional eligibility slots.

    Multi-position players (e.g., Harper with 1B/OF) give the LP solver
    more degrees of freedom.  Trading them for single-position players
    reduces lineup construction flexibility.

    Computes the *net* change in total position-slot count across giving
    and receiving sides.  Only real roster positions count (C, 1B, 2B,
    3B, SS, OF, SP, RP) — Util/DH/BN are excluded because every player
    is eligible for those.

    Returns:
        Tuple of (penalty_sgp, detail_dict) where:
          - penalty_sgp: SGP penalty (>= 0).  Zero if flexibility improves.
          - detail_dict: Breakdown with giving/receiving position counts.
    """

    def _count_slots(player_ids: list[int]) -> tuple[int, dict[str, int]]:
        """Count total real-position slots and per-position breakdown."""
        total = 0
        breakdown: dict[str, int] = {}
        for pid in player_ids:
            match = player_pool[player_pool["player_id"] == pid]
            if match.empty:
                continue
            positions = str(match.iloc[0].get("positions", "")).upper()
            pos_list = [p.strip() for p in positions.split(",") if p.strip()]
            real = [p for p in pos_list if p in _REAL_POSITIONS]
            total += len(real)
            for p in real:
                breakdown[p] = breakdown.get(p, 0) + 1
        return total, breakdown

    giving_slots, giving_breakdown = _count_slots(giving_ids)
    receiving_slots, receiving_breakdown = _count_slots(receiving_ids)
    net_change = receiving_slots - giving_slots  # positive = gained flexibility

    detail: dict[str, Any] = {
        "giving_slots": giving_slots,
        "giving_breakdown": giving_breakdown,
        "receiving_slots": receiving_slots,
        "receiving_breakdown": receiving_breakdown,
        "net_slot_change": net_change,
    }

    if net_change >= 0:
        detail["penalty"] = 0.0
        return 0.0, detail

    # Net loss of positional slots — apply penalty
    penalty = abs(net_change) * _FLEXIBILITY_SGP_PER_SLOT
    detail["penalty"] = round(penalty, 3)
    return round(penalty, 3), detail


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

    Runs two sub-analyses:
      1. Opponent valuation + market clearing price (when standings available)
      2. Sensitivity report (category ranking + breakeven analysis)

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
    # 1. Opponent valuation + market clearing price
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
