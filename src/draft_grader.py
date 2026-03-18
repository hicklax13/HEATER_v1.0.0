"""Post-Draft Grader: Analyzes completed drafts with grades, steals, and reaches.

3-component grading system:
  Component 1 (40%): Team Value — total SGP vs expected at draft position
  Component 2 (35%): Pick Efficiency — per-pick surplus vs ADP-to-SGP curve
  Component 3 (25%): Category Balance — CV of category z-scores (lower = better)

Steal/Reach detection uses both SGP-based surplus AND ADP deviation with
position-dependent thresholds (scarce positions use tighter thresholds).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.valuation import LeagueConfig, SGPCalculator

# ── Constants ─────────────────────────────────────────────────────────

# Grade thresholds (composite z-score → letter grade)
GRADE_THRESHOLDS = [
    (1.5, "A+"),
    (1.0, "A"),
    (0.7, "A-"),
    (0.4, "B+"),
    (0.1, "B"),
    (-0.1, "B-"),
    (-0.4, "C+"),
    (-0.7, "C"),
    (-1.0, "C-"),
    (-1.5, "D"),
]

# Position-dependent steal/reach round thresholds
POSITION_ROUND_SCALE = {
    "C": 1.5,
    "SS": 1.5,
    "2B": 1.8,
    "3B": 2.0,
    "1B": 2.0,
    "OF": 2.5,
    "SP": 2.0,
    "RP": 2.0,
    "Util": 2.5,
}

# SGP surplus thresholds for steal/reach
SGP_STEAL_THRESHOLD = 1.0
SGP_REACH_THRESHOLD = -1.0
SGP_GREAT_STEAL_THRESHOLD = 2.0
SGP_SLIGHT_REACH_THRESHOLD = -0.5

# Component weights
WEIGHT_TEAM_VALUE = 0.40
WEIGHT_EFFICIENCY = 0.35
WEIGHT_BALANCE = 0.25


# ── Grade Mapping ─────────────────────────────────────────────────────


def composite_to_grade(composite: float) -> str:
    """Map composite z-score to letter grade."""
    for threshold, grade in GRADE_THRESHOLDS:
        if composite >= threshold:
            return grade
    return "F"


# ── Expected Value Curve ──────────────────────────────────────────────


def build_expected_sgp_curve(
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> list[float]:
    """Build expected SGP at each draft pick position.

    Sorts all players by SGP descending (approximating ADP order by value).
    Returns list where index i = expected SGP for the (i+1)-th overall pick.
    """
    if config is None:
        config = LeagueConfig()

    sgp_calc = SGPCalculator(config)
    pool = player_pool.copy()
    pool["_sgp"] = pool.apply(sgp_calc.total_sgp, axis=1)

    # Sort by ADP if available, otherwise by SGP
    if "adp" in pool.columns:
        pool["_sort"] = pd.to_numeric(pool["adp"], errors="coerce").fillna(999)
        pool = pool.sort_values("_sort")
    else:
        pool = pool.sort_values("_sgp", ascending=False)

    return pool["_sgp"].tolist()


def expected_sgp_at_pick(pick_number: int, sgp_curve: list[float]) -> float:
    """Get expected SGP for a given pick number (1-based).

    Interpolates if pick_number is beyond the curve length.
    """
    idx = pick_number - 1
    if idx < 0:
        return sgp_curve[0] if sgp_curve else 0.0
    if idx < len(sgp_curve):
        return sgp_curve[idx]
    # Beyond pool: return last value or 0
    return sgp_curve[-1] if sgp_curve else 0.0


# ── Pick Classification ──────────────────────────────────────────────


def classify_pick(
    player_sgp: float,
    expected_sgp: float,
    player_adp: float,
    pick_number: int,
    num_teams: int = 12,
    primary_position: str = "Util",
) -> tuple[str, float, float]:
    """Classify a draft pick as STEAL, FAIR, or REACH.

    Uses both SGP surplus and ADP gap with position-adjusted thresholds.

    Returns:
        (classification, sgp_surplus, adp_gap)
    """
    sgp_surplus = player_sgp - expected_sgp
    adp_gap = pick_number - player_adp if player_adp > 0 else 0

    # Position-specific round scale
    round_scale = POSITION_ROUND_SCALE.get(primary_position, 2.0) * num_teams

    if sgp_surplus > SGP_GREAT_STEAL_THRESHOLD or adp_gap < -3 * num_teams:
        classification = "GREAT STEAL"
    elif sgp_surplus > SGP_STEAL_THRESHOLD or adp_gap < -round_scale:
        classification = "STEAL"
    elif sgp_surplus < SGP_REACH_THRESHOLD or adp_gap > round_scale:
        classification = "REACH"
    elif sgp_surplus < SGP_SLIGHT_REACH_THRESHOLD or adp_gap > round_scale * 0.75:
        classification = "SLIGHT REACH"
    else:
        classification = "FAIR"

    return classification, round(sgp_surplus, 3), round(adp_gap, 1)


# ── Category Balance ─────────────────────────────────────────────────


def compute_category_projections(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> dict[str, dict]:
    """Compute projected category totals and z-scores for a drafted team.

    Returns dict: {category: {total, z_score}}
    """
    if config is None:
        config = LeagueConfig()

    from src.in_season import _roster_category_totals

    totals = _roster_category_totals(roster_ids, player_pool)

    # Compute league-average totals (assume pool average × roster scale)
    stat_map = {
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

    projections = {}
    for cat in config.all_categories:
        val = totals.get(cat, 0)
        # Use SGP denominator as rough proxy for league std
        denom = config.sgp_denominators.get(cat, 1.0)
        if abs(denom) < 1e-9:
            denom = 1.0

        # Approximate z-score using SGP as proxy
        sgp_val = val / denom
        if cat in config.inverse_stats:
            sgp_val = -sgp_val

        projections[cat] = {
            "total": round(val, 3),
            "z_score": round(sgp_val, 3),
        }

    return projections


def category_balance_score(category_projections: dict) -> float:
    """Compute category balance from coefficient of variation.

    Lower CV = more balanced team. Returns score in [0, 1] where 1 = perfectly balanced.
    """
    z_scores = [info["z_score"] for info in category_projections.values()]
    if len(z_scores) < 2:
        return 1.0

    std = np.std(z_scores)
    mean = abs(np.mean(z_scores))

    if mean < 1e-6:
        return 0.5  # Zero mean makes CV undefined

    cv = std / max(mean, 0.1)
    # Map CV to [0, 1]: CV=0 → 1.0 (perfect), CV=2.0 → 0.0 (terrible)
    balance = max(0.0, min(1.0, 1.0 - cv / 2.0))
    return round(balance, 3)


# ── Main Draft Grader ─────────────────────────────────────────────────


def grade_draft(
    draft_picks: list[dict],
    player_pool: pd.DataFrame,
    config: LeagueConfig | None = None,
) -> dict:
    """Analyze a completed draft and assign grades.

    Args:
        draft_picks: List of dicts with keys: round, pick_number, player_id, player_name.
                     Must be sorted by pick_number.
        player_pool: Full player pool DataFrame.
        config: League configuration.

    Returns:
        Dict with:
          overall_grade, overall_score,
          team_value_score, pick_efficiency_score, category_balance_score,
          picks (annotated list), steals, reaches,
          category_projections, strengths, weaknesses, recommendations
    """
    if config is None:
        config = LeagueConfig()

    if not draft_picks or player_pool.empty:
        return {
            "overall_grade": "N/A",
            "overall_score": 0.0,
            "picks": [],
            "steals": [],
            "reaches": [],
            "category_projections": {},
            "strengths": [],
            "weaknesses": [],
        }

    sgp_calc = SGPCalculator(config)
    sgp_curve = build_expected_sgp_curve(player_pool, config)
    num_teams = config.num_teams

    # ── Annotate each pick ────────────────────────────────────────────
    annotated_picks = []
    total_sgp = 0.0
    total_expected = 0.0
    surpluses = []
    roster_ids = []

    for pick in draft_picks:
        pick_num = pick.get("pick_number", 0)
        player_id = pick.get("player_id")
        player_name = pick.get("player_name", "Unknown")

        # Find player in pool
        player_row = player_pool[player_pool["player_id"] == player_id]
        if player_row.empty:
            # Try by name
            name_col = "name" if "name" in player_pool.columns else "player_name"
            player_row = player_pool[player_pool[name_col] == player_name]

        if player_row.empty:
            player_sgp = 0.0
            player_adp = 0.0
            primary_pos = "Util"
        else:
            p = player_row.iloc[0]
            player_sgp = sgp_calc.total_sgp(p)
            player_adp = float(p.get("adp", 0) or 0)
            positions = [pos.strip() for pos in str(p.get("positions", "Util")).split(",")]
            primary_pos = positions[0] if positions else "Util"

        expected = expected_sgp_at_pick(pick_num, sgp_curve)
        classification, surplus, adp_gap = classify_pick(
            player_sgp, expected, player_adp, pick_num, num_teams, primary_pos
        )

        total_sgp += player_sgp
        total_expected += expected
        surpluses.append(surplus)
        if player_id:
            roster_ids.append(player_id)

        annotated_picks.append(
            {
                "round": pick.get("round", 0),
                "pick_number": pick_num,
                "player_id": player_id,
                "player_name": player_name,
                "positions": primary_pos,
                "sgp": round(player_sgp, 3),
                "expected_sgp": round(expected, 3),
                "surplus": surplus,
                "adp": player_adp,
                "adp_gap": adp_gap,
                "classification": classification,
            }
        )

    # ── Component 1: Team Value (40%) ─────────────────────────────────
    value_delta = total_sgp - total_expected
    # Normalize: approximate std of team value across drafts
    value_std = max(np.std(surpluses) * len(draft_picks) ** 0.5, 1.0) if surpluses else 1.0
    team_value_z = value_delta / value_std

    # ── Component 2: Pick Efficiency (35%) ────────────────────────────
    if surpluses:
        efficiency_mean = np.mean(surpluses)
        efficiency_std = max(np.std(surpluses), 0.5)
        pick_efficiency_z = efficiency_mean / efficiency_std
    else:
        pick_efficiency_z = 0.0

    # ── Component 3: Category Balance (25%) ───────────────────────────
    cat_proj = compute_category_projections(roster_ids, player_pool, config)
    balance = category_balance_score(cat_proj)
    # Map balance [0, 1] to z-score-like scale: 0.5 = average (z=0)
    balance_z = (balance - 0.5) * 4.0  # 0→-2, 0.5→0, 1.0→2

    # ── Composite Grade ───────────────────────────────────────────────
    composite = WEIGHT_TEAM_VALUE * team_value_z + WEIGHT_EFFICIENCY * pick_efficiency_z + WEIGHT_BALANCE * balance_z
    overall_grade = composite_to_grade(composite)

    # ── Steals and Reaches ────────────────────────────────────────────
    steals = [p for p in annotated_picks if p["classification"] in ("STEAL", "GREAT STEAL")]
    steals.sort(key=lambda p: p["surplus"], reverse=True)

    reaches = [p for p in annotated_picks if p["classification"] in ("REACH", "SLIGHT REACH")]
    reaches.sort(key=lambda p: p["surplus"])

    # ── Category Strengths/Weaknesses ─────────────────────────────────
    cat_z_list = [(cat, info["z_score"]) for cat, info in cat_proj.items()]
    cat_z_list.sort(key=lambda x: x[1], reverse=True)
    strengths = [c[0] for c in cat_z_list[:3]]
    weaknesses = [c[0] for c in cat_z_list[-3:]]

    return {
        "overall_grade": overall_grade,
        "overall_score": round(composite, 3),
        "team_value_score": round(team_value_z, 3),
        "pick_efficiency_score": round(pick_efficiency_z, 3),
        "category_balance_score": round(balance, 3),
        "picks": annotated_picks,
        "steals": steals[:3],
        "reaches": reaches[:3],
        "category_projections": cat_proj,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "total_sgp": round(total_sgp, 3),
        "expected_sgp": round(total_expected, 3),
    }
