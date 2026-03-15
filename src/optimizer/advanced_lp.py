"""Advanced LP formulations for lineup optimization.

Provides three standalone LP formulations that complement the base
LineupOptimizer in src/lineup_optimizer.py:

1. Maximin lineup — maximizes the WORST category contribution for a
   balanced roster (no single category tanks your week).
2. Epsilon-constraint lineup — multi-objective LP that maximizes one
   primary category while ensuring minimum thresholds on all others.
3. Stochastic MIP — unified mean + CVaR optimization over scenario
   arrays from scenario_generator.py.

All functions build and solve their own PuLP LP problems. They do NOT
modify or subclass LineupOptimizer.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

try:
    from pulp import (
        PULP_CBC_CMD,
        LpBinary,
        LpContinuous,
        LpMaximize,
        LpProblem,
        LpStatus,
        LpVariable,
        lpSum,
    )

    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Category definitions ─────────────────────────────────────────────

ALL_CATEGORIES: list[str] = [
    "r",
    "hr",
    "rbi",
    "sb",
    "avg",
    "w",
    "sv",
    "k",
    "era",
    "whip",
]
INVERSE_CATS: set[str] = {"era", "whip"}
HITTER_CATS: list[str] = ["r", "hr", "rbi", "sb", "avg"]
PITCHER_CATS: list[str] = ["w", "sv", "k", "era", "whip"]

# ── Helpers ──────────────────────────────────────────────────────────


def _get_player_name(row: pd.Series) -> str:
    """Extract player name from a roster row, checking both column names."""
    for col in ("name", "player_name"):
        val = row.get(col, None)
        if val is not None and str(val).strip():
            return str(val)
    return f"Player_{row.name}"


def _extract_stat(roster: pd.DataFrame, cat: str) -> np.ndarray:
    """Safely extract a stat column as a float array."""
    if cat not in roster.columns:
        return np.zeros(len(roster), dtype=float)
    return pd.to_numeric(roster[cat], errors="coerce").fillna(0.0).values.astype(float)


def _count_by_type(roster: pd.DataFrame) -> tuple[int, int]:
    """Count hitters and pitchers on the roster.

    Returns:
        (n_hitters, n_pitchers)
    """
    if "is_hitter" not in roster.columns:
        return len(roster), 0
    is_hitter = roster["is_hitter"].astype(bool)
    return int(is_hitter.sum()), int((~is_hitter).sum())


def _starter_limits(roster: pd.DataFrame) -> tuple[int, int]:
    """Compute how many hitters / pitchers to start.

    Yahoo 5x5 defaults: up to 13 hitter slots, up to 10 pitcher slots.
    Capped by actual roster size in each group.
    """
    n_hit, n_pit = _count_by_type(roster)
    return min(n_hit, 13), min(n_pit, 10)


def _unavailable_result(reason: str = "PuLP not available") -> dict:
    """Return a graceful empty result when the solver cannot run."""
    return {
        "status": reason,
        "objective_value": 0.0,
        "assignments": {},
    }


# ── Formulation 1: Maximin ───────────────────────────────────────────


def maximin_lineup(
    roster: pd.DataFrame,
    scale_factors: dict[str, float],
    category_weights: dict[str, float] | None = None,
    active_categories: list[str] | None = None,
) -> dict:
    """Maximize the worst (minimum) category contribution.

    LP formulation::

        maximize z
        subject to:
            sum_i(x_i * v_ic) / scale_c  >=  z     for normal cats c
            -sum_i(x_i * v_ic) / scale_c  >=  z     for inverse cats c (ERA, WHIP)
            sum(x_i for hitters)  ==  max_hitters
            sum(x_i for pitchers) ==  max_pitchers
            x_i in {0, 1}

    Args:
        roster: Player pool DataFrame with stat columns and ``is_hitter``.
        scale_factors: Per-category normalizers (e.g. SGP denominators).
            Every active category must have a positive entry.
        category_weights: Optional per-category multiplier applied to
            each constraint.  Categories with weight 0 are skipped.
        active_categories: Subset of categories to include.  Defaults
            to all 10.  Use this to exclude punted categories.

    Returns:
        Dict with keys: ``status``, ``z_value``, ``assignments``
        (player_name -> 0/1), ``objective_value``.
    """
    if not PULP_AVAILABLE:
        return _unavailable_result()

    if roster.empty:
        return {
            "status": "empty_roster",
            "z_value": 0.0,
            "assignments": {},
            "objective_value": 0.0,
        }

    cats = active_categories or list(ALL_CATEGORIES)
    weights = category_weights or {}
    max_hit, max_pit = _starter_limits(roster)
    n = len(roster)

    # Pre-extract stats and metadata
    is_hitter = roster["is_hitter"].astype(bool).values if "is_hitter" in roster.columns else np.ones(n, dtype=bool)
    names = [_get_player_name(roster.iloc[i]) for i in range(n)]
    stat_vals: dict[str, np.ndarray] = {cat: _extract_stat(roster, cat) for cat in cats}

    # Build LP
    prob = LpProblem("maximin_lineup", LpMaximize)

    # Decision variables
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]
    z = LpVariable("z_min", cat=LpContinuous)

    # Objective: maximize the floor
    prob += z

    # Per-category constraints: scaled total >= z  (or flipped for inverse)
    for cat in cats:
        w = weights.get(cat, 1.0)
        if w == 0:
            continue  # skip punted category
        sf = scale_factors.get(cat, 1.0)
        if sf <= 0:
            sf = 1.0  # safety

        vals = stat_vals[cat]

        if cat in INVERSE_CATS:
            # Lower is better.  We want the "contribution" to also be
            # in a "higher is better" frame so z captures the true worst.
            # Contribution = -sum(x_i * val_ic) / scale_c
            # Constraint:  -sum(x_i * val_ic) / scale_c  >=  z
            prob += (
                lpSum(-x[i] * float(vals[i]) * w / sf for i in range(n)) >= z,
                f"maximin_{cat}",
            )
        else:
            prob += (
                lpSum(x[i] * float(vals[i]) * w / sf for i in range(n)) >= z,
                f"maximin_{cat}",
            )

    # Roster type constraints
    prob += (
        lpSum(x[i] for i in range(n) if is_hitter[i]) == max_hit,
        "hitter_count",
    )
    prob += (
        lpSum(x[i] for i in range(n) if not is_hitter[i]) == max_pit,
        "pitcher_count",
    )

    # Solve
    solver = PULP_CBC_CMD(msg=0, timeLimit=10)
    prob.solve(solver)
    status = LpStatus[prob.status]

    assignments: dict[str, int] = {}
    z_val = 0.0

    if status == "Optimal":
        z_val = float(z.varValue) if z.varValue is not None else 0.0
        for i in range(n):
            val = x[i].varValue
            assignments[names[i]] = 1 if (val is not None and val > 0.5) else 0
    else:
        for i in range(n):
            assignments[names[i]] = 0

    return {
        "status": status,
        "z_value": z_val,
        "assignments": assignments,
        "objective_value": z_val,
    }


# ── Formulation 2: Epsilon-constraint ────────────────────────────────


def epsilon_constraint_lineup(
    roster: pd.DataFrame,
    scale_factors: dict[str, float],
    primary_category: str,
    epsilon_bounds: dict[str, float],
    category_weights: dict[str, float] | None = None,
) -> dict:
    """Multi-objective LP via epsilon-constraint method.

    Maximizes ``primary_category`` subject to minimum (or maximum for
    inverse stats) thresholds on every other category.

    Example question this answers: *"What is the best HR lineup that
    keeps AVG above .270 and ERA below 3.50?"*

    LP formulation::

        maximize  sum_i(x_i * v_i_primary) / scale_primary
        subject to:
            sum_i(x_i * v_ic) / scale_c  >=  epsilon_bounds[c]   (normal cats)
            sum_i(x_i * v_ic) / scale_c  <=  epsilon_bounds[c]   (inverse cats)
            roster-type constraints
            x_i in {0, 1}

    Args:
        roster: Player pool DataFrame.
        scale_factors: Per-category normalizers.
        primary_category: The category to maximize (e.g. ``"hr"``).
        epsilon_bounds: Minimum (or maximum for ERA/WHIP) scaled
            thresholds for every other category.
        category_weights: Optional multiplier for the primary objective.

    Returns:
        Dict with keys: ``status``, ``primary_value``,
        ``constraint_satisfaction`` (cat -> actual scaled value),
        ``assignments`` (player_name -> 0/1), ``objective_value``.
    """
    if not PULP_AVAILABLE:
        result = _unavailable_result()
        result["primary_value"] = 0.0
        result["constraint_satisfaction"] = {}
        return result

    if roster.empty:
        return {
            "status": "empty_roster",
            "primary_value": 0.0,
            "constraint_satisfaction": {},
            "assignments": {},
            "objective_value": 0.0,
        }

    weights = category_weights or {}
    max_hit, max_pit = _starter_limits(roster)
    n = len(roster)

    is_hitter = roster["is_hitter"].astype(bool).values if "is_hitter" in roster.columns else np.ones(n, dtype=bool)
    names = [_get_player_name(roster.iloc[i]) for i in range(n)]

    # Build LP
    prob = LpProblem("epsilon_constraint_lineup", LpMaximize)
    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n)]

    # Objective: maximize primary category (scaled)
    sf_primary = scale_factors.get(primary_category, 1.0)
    if sf_primary <= 0:
        sf_primary = 1.0
    primary_vals = _extract_stat(roster, primary_category)
    w_primary = weights.get(primary_category, 1.0)

    if primary_category in INVERSE_CATS:
        # For inverse primary: minimize the stat = maximize negative
        prob += lpSum(-x[i] * float(primary_vals[i]) * w_primary / sf_primary for i in range(n))
    else:
        prob += lpSum(x[i] * float(primary_vals[i]) * w_primary / sf_primary for i in range(n))

    # Epsilon constraints on other categories
    for cat, bound in epsilon_bounds.items():
        if cat == primary_category:
            continue
        sf = scale_factors.get(cat, 1.0)
        if sf <= 0:
            sf = 1.0
        vals = _extract_stat(roster, cat)

        if cat in INVERSE_CATS:
            # Keep scaled total BELOW the bound (lower is better)
            prob += (
                lpSum(x[i] * float(vals[i]) / sf for i in range(n)) <= bound,
                f"eps_{cat}",
            )
        else:
            # Keep scaled total ABOVE the bound
            prob += (
                lpSum(x[i] * float(vals[i]) / sf for i in range(n)) >= bound,
                f"eps_{cat}",
            )

    # Roster type constraints
    prob += (
        lpSum(x[i] for i in range(n) if is_hitter[i]) == max_hit,
        "hitter_count",
    )
    prob += (
        lpSum(x[i] for i in range(n) if not is_hitter[i]) == max_pit,
        "pitcher_count",
    )

    # Solve
    solver = PULP_CBC_CMD(msg=0, timeLimit=10)
    prob.solve(solver)
    status = LpStatus[prob.status]

    assignments: dict[str, int] = {}
    primary_value = 0.0
    satisfaction: dict[str, float] = {}

    if status == "Optimal":
        for i in range(n):
            val = x[i].varValue
            assignments[names[i]] = 1 if (val is not None and val > 0.5) else 0

        # Compute actual scaled totals for primary + constrained categories
        started = [i for i in range(n) if assignments[names[i]] == 1]

        # Primary
        raw_primary = sum(float(primary_vals[i]) for i in started)
        primary_value = raw_primary / sf_primary
        if primary_category in INVERSE_CATS:
            primary_value = -primary_value  # present in "higher is better" frame

        # Constraint satisfaction
        for cat in epsilon_bounds:
            sf = scale_factors.get(cat, 1.0)
            if sf <= 0:
                sf = 1.0
            vals = _extract_stat(roster, cat)
            raw = sum(float(vals[i]) for i in started)
            satisfaction[cat] = raw / sf
    else:
        for i in range(n):
            assignments[names[i]] = 0

    obj_val = prob.objective.value() if status == "Optimal" else 0.0

    return {
        "status": status,
        "primary_value": primary_value,
        "constraint_satisfaction": satisfaction,
        "assignments": assignments,
        "objective_value": float(obj_val) if obj_val is not None else 0.0,
    }


# ── Formulation 3: Stochastic MIP ───────────────────────────────────


def stochastic_mip(
    roster: pd.DataFrame,
    scenarios: np.ndarray,
    category_weights: dict[str, float],
    scale_factors: dict[str, float] | None = None,
    mean_weight: float = 0.7,
    cvar_weight: float = 0.3,
    cvar_alpha: float = 0.05,
) -> dict:
    """Unified mean + CVaR tail-risk optimization over scenarios.

    Builds a single MIP that trades off expected lineup value against
    worst-case tail risk.  Implements Rockafellar-Uryasev (2000) CVaR
    linearization.

    LP formulation::

        maximize  mean_weight * (1/N) * sum_s V_s
                  + cvar_weight * [eta - (1/(N*alpha)) * sum_s z_s]

        subject to:
            V_s  =  sum_i x_i * value_is         for all s
            z_s  >=  0                            for all s
            z_s  >=  eta - V_s                    for all s
            roster-type constraints
            x_i in {0, 1}

    ``value_is`` is the weighted, scale-normalised stat total for
    player *i* in scenario *s*.

    Args:
        roster: Player pool DataFrame.
        scenarios: Shape ``(n_scenarios, n_players, 10)`` from
            ``scenario_generator.generate_stat_scenarios()``.
        category_weights: Per-category importance weights.
        scale_factors: Optional per-category normalizers (SGP denoms).
            When ``None``, raw stat values are used.
        mean_weight: Weight on the expected-value component (default 0.7).
        cvar_weight: Weight on the CVaR component (default 0.3).
        cvar_alpha: Tail probability for CVaR (default 0.05 = worst 5%).

    Returns:
        Dict with keys: ``status``, ``objective_value``,
        ``mean_component``, ``cvar_component``, ``assignments``
        (player_name -> 0/1), ``eta_value``.
    """
    if not PULP_AVAILABLE:
        result = _unavailable_result()
        result.update(
            mean_component=0.0,
            cvar_component=0.0,
            eta_value=0.0,
        )
        return result

    if roster.empty or scenarios.size == 0:
        return {
            "status": "empty_roster",
            "objective_value": 0.0,
            "mean_component": 0.0,
            "cvar_component": 0.0,
            "assignments": {},
            "eta_value": 0.0,
        }

    n_scen = scenarios.shape[0]
    n_players = min(len(roster), scenarios.shape[1])
    n_cats = len(ALL_CATEGORIES)
    max_hit, max_pit = _starter_limits(roster)

    is_hitter = (
        roster["is_hitter"].astype(bool).values[:n_players]
        if "is_hitter" in roster.columns
        else np.ones(n_players, dtype=bool)
    )
    names = [_get_player_name(roster.iloc[i]) for i in range(n_players)]

    # Build weight vector (with scale and sign)
    w_vec = np.zeros(n_cats, dtype=float)
    for j, cat in enumerate(ALL_CATEGORIES):
        w = category_weights.get(cat, 0.0)
        sf = 1.0
        if scale_factors:
            sf = scale_factors.get(cat, 1.0)
            if sf <= 0:
                sf = 1.0
        if cat in INVERSE_CATS:
            w_vec[j] = -w / sf  # lower is better
        else:
            w_vec[j] = w / sf

    # Precompute per-player-per-scenario values: val[s][i]
    val = np.zeros((n_scen, n_players), dtype=float)
    for s in range(n_scen):
        for i in range(n_players):
            val[s, i] = float(np.dot(scenarios[s, i, :n_cats], w_vec))

    # Build LP
    prob = LpProblem("stochastic_mip", LpMaximize)

    x = [LpVariable(f"x_{i}", cat=LpBinary) for i in range(n_players)]
    eta = LpVariable("eta", cat=LpContinuous)
    z = [LpVariable(f"z_{s}", lowBound=0, cat=LpContinuous) for s in range(n_scen)]

    # Scenario value expressions: V_s = sum_i x_i * val[s][i]
    # We do not create explicit V_s variables — inline them.

    # Objective: mean_weight * (1/N) * sum_s sum_i x_i * val[s][i]
    #          + cvar_weight * [eta - (1/(N*alpha)) * sum_s z_s]
    mean_terms = []
    for s in range(n_scen):
        for i in range(n_players):
            coeff = mean_weight * val[s, i] / n_scen
            if abs(coeff) > 1e-12:
                mean_terms.append(x[i] * coeff)

    cvar_coeff = 1.0 / (n_scen * max(cvar_alpha, 1e-9))
    cvar_terms = [eta * cvar_weight]
    for s in range(n_scen):
        cvar_terms.append(z[s] * (-cvar_weight * cvar_coeff))

    prob += lpSum(mean_terms) + lpSum(cvar_terms)

    # CVaR constraints: z_s >= eta - V_s   for all s
    for s in range(n_scen):
        v_s_expr = lpSum(x[i] * float(val[s, i]) for i in range(n_players))
        prob += (
            z[s] >= eta - v_s_expr,
            f"cvar_{s}",
        )

    # Roster type constraints
    prob += (
        lpSum(x[i] for i in range(n_players) if is_hitter[i]) == max_hit,
        "hitter_count",
    )
    prob += (
        lpSum(x[i] for i in range(n_players) if not is_hitter[i]) == max_pit,
        "pitcher_count",
    )

    # Solve (generous time limit for larger MIPs)
    solver = PULP_CBC_CMD(msg=0, timeLimit=30)
    prob.solve(solver)
    status = LpStatus[prob.status]

    assignments: dict[str, int] = {}
    eta_val = 0.0
    mean_comp = 0.0
    cvar_comp = 0.0
    obj_val = 0.0

    if status == "Optimal":
        eta_val = float(eta.varValue) if eta.varValue is not None else 0.0
        obj_val = float(prob.objective.value()) if prob.objective.value() is not None else 0.0

        for i in range(n_players):
            v = x[i].varValue
            assignments[names[i]] = 1 if (v is not None and v > 0.5) else 0

        # Decompose objective into mean and CVaR components
        started = [i for i in range(n_players) if assignments[names[i]] == 1]
        scen_totals = np.zeros(n_scen, dtype=float)
        for s in range(n_scen):
            scen_totals[s] = sum(val[s, i] for i in started)

        mean_comp = float(np.mean(scen_totals)) * mean_weight

        # CVaR = eta - E[max(0, eta - V_s)] / alpha
        shortfalls = np.maximum(0, eta_val - scen_totals)
        cvar_comp = (eta_val - float(np.mean(shortfalls)) / max(cvar_alpha, 1e-9)) * cvar_weight
    else:
        for i in range(n_players):
            assignments[names[i]] = 0

    return {
        "status": status,
        "objective_value": obj_val,
        "mean_component": mean_comp,
        "cvar_component": cvar_comp,
        "assignments": assignments,
        "eta_value": eta_val,
    }
