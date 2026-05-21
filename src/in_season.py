"""In-season analysis: trade analyzer, player comparison, free agent ranker.

Reuses SGPCalculator, compute_replacement_levels, compute_category_weights
from src/valuation.py. MC simulation pattern adapted from src/simulation.py.
"""

import numpy as np
import pandas as pd

from src.optimizer.constants_registry import CONSTANTS_REGISTRY as _CR
from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    compute_category_weights,
)


def compute_category_fit(
    player_sgp_by_cat: dict[str, float],
    team_category_profile: dict[str, str],
) -> dict:
    """How well does this player fit your team's category needs?

    Args:
        player_sgp_by_cat: Per-category SGP contributions for the player.
        team_category_profile: Maps category name to "strong", "weak", or "punt".

    Returns:
        dict with keys: helps (list of weak cats player contributes to),
        wastes (list of strong/punt cats where player over-contributes),
        fit_score (float 0-100).
    """
    helps = [c for c, s in team_category_profile.items() if s == "weak" and player_sgp_by_cat.get(c, 0) > 0.1]
    wastes = [
        c for c, s in team_category_profile.items() if s in ("strong", "punt") and player_sgp_by_cat.get(c, 0) > 0.5
    ]
    total_cats = len([c for c in team_category_profile if team_category_profile[c] != "punt"])
    fit_score = (len(helps) / max(total_cats, 1)) * 100 if helps else 0
    return {"helps": helps, "wastes": wastes, "fit_score": round(fit_score, 1)}


_IL_STATUSES = frozenset({"IL10", "IL15", "IL60", "IL-10", "IL-15", "IL-60"})

# 2026-05-20 FA-engine overhaul P1 / PR2: per-status weighting factor.
#
# OLD behavior: IL players (any status) had their projection zeroed out
# entirely. This made dropping an IL ace appear "free" in the SGP math
# because before/after roster totals were identical regardless of whether
# the IL player was on the roster — the upstream cause of the Crochet/Kirk
# bad recommendation (top-30 SP on IL15 valued as costless to drop).
#
# NEW behavior: IL players retain a fraction of their projection scaled
# by expected return window. Defaults below are research-defensible per
# the RotoWire IL Stash Guide + FantasyPros injury rankings, and are
# registered in constants_registry.py for citation + sensitivity tracking.
#
# Suspended / Restricted / NA players ARE zeroed — they're not coming
# back this season at all.
_IL_WEIGHT_DEFAULTS: dict[str, float] = {
    # Active player — full projection.
    "ACTIVE": 1.0,
    "": 1.0,
    # Day-to-day / probably-back-this-week.
    "DTD": 0.95,
    "DAY-TO-DAY": 0.95,
    # IL10 — short-term, returns inside ~2 weeks. Most of ROS still valid.
    "IL10": 0.85,
    "IL-10": 0.85,
    # IL15 — 15-day, returns inside ~3 weeks. Still a significant stash asset.
    "IL15": 0.70,
    "IL-15": 0.70,
    "IL": 0.70,  # bare "IL" — Yahoo's generic IL slot, conservatively treated as IL15
    # IL60 — long-term stash. Industry consensus: still worth ~20% of ROS
    # value if the player is top-100 ROS (per RotoWire stash strategy).
    "IL60": 0.20,
    "IL-60": 0.20,
    # Unavailable for the rest of the season.
    "NA": 0.0,
    "SUSPENDED": 0.0,
    "RESTRICTED": 0.0,
    "OUT": 0.0,
}


# 2026-05-20 FA-engine overhaul P3.5 / PR17: statuses for which a
# concrete return date (from ESPN injuries or Yahoo news) should override
# the string-based default. Short suspensions and DTD often resolve in
# days, so a known return date is more accurate than the generic 0.0/0.95.
_RETURN_DATE_OVERRIDE_STATUSES = frozenset({"SUSPENDED", "NA", "RESTRICTED", "OUT", "DTD", "DAY-TO-DAY"})


def _return_date_weight(expected_return_days: float | None) -> float | None:
    """Map days-until-return to projection weight via piecewise-linear interp.

    Returns None when expected_return_days is None or above the 60d ceiling —
    caller falls through to status-based string lookup. Returns a float
    weight in [0.0, 1.0] when a return date is known.

    Anchors:
       < 0 days (already back per ESPN): 1.0
       1 day:                            0.95
       7 days:                           0.85
      14 days:                           0.70  (matches IL10 default)
      21 days:                           0.55
      45 days:                           0.30
      60 days:                           0.20  (matches IL60 default)
      > 60 days OR None:                 None  (fall through)
    """
    if expected_return_days is None:
        return None
    try:
        d = float(expected_return_days)
    except (TypeError, ValueError):
        return None
    # NaN guard — pandas NaN floats trip arithmetic
    if d != d:  # NaN
        return None
    if d < 0:
        return 1.0  # ESPN says already back
    # Piecewise-linear between anchors. Each segment uses standard linear interp.
    anchors = [(0, 1.0), (1, 0.95), (7, 0.85), (14, 0.70), (21, 0.55), (45, 0.30), (60, 0.20)]
    if d >= anchors[-1][0]:
        # Beyond the 60-day anchor — return None so caller falls through to
        # status-based lookup (e.g. "Suspended" → 0.0 indefinite).
        return None
    for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
        if x0 <= d <= x1:
            if x1 == x0:
                return y0
            t = (d - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None  # unreachable defensive guard


def _il_weight_from_status(
    status: str,
    expected_return_days: float | None = None,
) -> float:
    """Return projection weight for a roster status string.

    Lower bound 0.0 (fully unavailable — Suspended / Restricted), upper
    bound 1.0 (active). IL statuses scale by expected return window. See
    _IL_WEIGHT_DEFAULTS for citations.

    When ``expected_return_days`` is provided (from ESPN injury feed or
    Yahoo news), the return-date curve from ``_return_date_weight`` takes
    precedence over the string-based default for "potentially short"
    statuses (Suspended / NA / Restricted / OUT / DTD). For IL10/IL15/IL60
    specifically, the return-date curve is used when it would UPGRADE the
    weight (more specific info → more accurate), e.g. an IL10 player with
    a 1-day ETA should weight higher than the generic IL10 default of 0.85.

    Unknown statuses default to 1.0 (active assumption — fail open,
    not closed; the previous zeroing was the upstream bug).
    """
    if not status:
        return 1.0
    key = str(status).upper().strip()
    # Try return-date curve first for short-term statuses.
    if expected_return_days is not None:
        rd_weight = _return_date_weight(expected_return_days)
        if rd_weight is not None:
            if key in _RETURN_DATE_OVERRIDE_STATUSES:
                # Short-suspension class — return date is authoritative.
                return rd_weight
            # IL10/IL15/IL60: if return date implies a HIGHER weight than
            # the string default, use it (more specific info wins).
            string_default = _IL_WEIGHT_DEFAULTS.get(key)
            if string_default is not None and rd_weight > string_default:
                return rd_weight
    return _IL_WEIGHT_DEFAULTS.get(key, 1.0)


def _roster_category_totals(roster_ids: list, player_pool: pd.DataFrame) -> dict:
    """Compute aggregate category totals for a set of player IDs.

    Players on the Injured List contribute a WEIGHTED fraction of their
    projection (per ``_il_weight_from_status``) rather than zero. This
    prevents the Crochet/Kirk bad-recommendation class — an IL ace whose
    full ROS projection is worth +30 SGP cannot evaluate as a "free drop"
    because his weighted contribution to the roster total is nonzero.

    Suspended / Restricted / NA players keep the previous zero-weight
    behavior — they're not coming back.
    """
    roster = player_pool[player_pool["player_id"].isin(roster_ids)]
    totals = {
        "R": 0.0,
        "HR": 0.0,
        "RBI": 0.0,
        "SB": 0.0,
        "W": 0.0,
        "L": 0.0,
        "SV": 0.0,
        "K": 0.0,
        "ab": 0.0,
        "h": 0.0,
        "bb": 0.0,
        "hbp": 0.0,
        "sf": 0.0,
        "ip": 0.0,
        "er": 0.0,
        "bb_allowed": 0.0,
        "h_allowed": 0.0,
    }
    for _, p in roster.iterrows():
        status = str(p.get("status", "") or "").upper().strip()
        # PR17: optional per-player return-date override (column may be
        # absent on legacy DataFrames; use defensive .get()).
        expected_return_days = p.get("expected_return_days") if hasattr(p, "get") else None
        weight = _il_weight_from_status(status, expected_return_days=expected_return_days)
        if weight == 0.0:
            continue  # Suspended / Restricted / NA — keep legacy zero
        totals["R"] += int(p.get("r", 0) or 0) * weight
        totals["HR"] += int(p.get("hr", 0) or 0) * weight
        totals["RBI"] += int(p.get("rbi", 0) or 0) * weight
        totals["SB"] += int(p.get("sb", 0) or 0) * weight
        totals["W"] += int(p.get("w", 0) or 0) * weight
        totals["L"] += int(p.get("l", 0) or 0) * weight
        totals["SV"] += int(p.get("sv", 0) or 0) * weight
        totals["K"] += int(p.get("k", 0) or 0) * weight
        totals["ab"] += int(p.get("ab", 0) or 0) * weight
        totals["h"] += int(p.get("h", 0) or 0) * weight
        totals["bb"] += int(p.get("bb", 0) or 0) * weight
        totals["hbp"] += int(p.get("hbp", 0) or 0) * weight
        totals["sf"] += int(p.get("sf", 0) or 0) * weight
        totals["ip"] += float(p.get("ip", 0) or 0) * weight
        totals["er"] += float(p.get("er", 0) or 0) * weight
        totals["bb_allowed"] += int(p.get("bb_allowed", 0) or 0) * weight
        totals["h_allowed"] += int(p.get("h_allowed", 0) or 0) * weight

    if totals["ab"] > 0:
        totals["AVG"] = totals["h"] / totals["ab"]
    else:
        totals["AVG"] = 0.250  # League-average neutral sentinel
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
    obp_denom = totals["ab"] + totals["bb"] + totals["hbp"] + totals["sf"]
    if obp_denom > 0:
        totals["OBP"] = (totals["h"] + totals["bb"] + totals["hbp"]) / obp_denom
    else:
        # 2026-05-17 Section 3 D8: read from registry (was 0.320 inline).
        totals["OBP"] = _CR["league_avg_woba"].value
    if totals["ip"] > 0:
        totals["ERA"] = totals["er"] * 9 / totals["ip"]
        totals["WHIP"] = (totals["bb_allowed"] + totals["h_allowed"]) / totals["ip"]
    else:
        # 2026-05-17 Section 3 D1: read from registry (was 4.50/1.30 inline).
        totals["ERA"] = _CR["league_avg_era"].value
        totals["WHIP"] = _CR["league_avg_whip"].value

    return totals


def analyze_trade(
    giving_ids: list,
    receiving_ids: list,
    user_roster_ids: list,
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    n_sims: int = 200,
) -> dict:
    """Analyze a trade proposal using Live SGP + Monte Carlo simulation.

    Returns dict with verdict, confidence_pct, category_impact,
    total_sgp_change, mc_mean, mc_std, risk_flags, before/after totals.
    """
    sgp_calc = SGPCalculator(config)

    before_ids = list(user_roster_ids)
    after_ids = [pid for pid in before_ids if pid not in giving_ids] + list(receiving_ids)

    before_totals = _roster_category_totals(before_ids, player_pool)
    after_totals = _roster_category_totals(after_ids, player_pool)

    # SF-25: per-category and total SGP changes via SGPCalculator.totals_sgp
    # (single-source-of-truth replacement for the prior inline raw_change/denom math).
    category_impact = {}
    total_sgp_change = 0.0
    for cat in config.all_categories:
        before_val = before_totals.get(cat, 0)
        after_val = after_totals.get(cat, 0)
        raw_change = after_val - before_val
        sgp_change = sgp_calc.totals_sgp({cat: raw_change})

        category_impact[cat] = round(sgp_change, 3)
        total_sgp_change += sgp_change

    # Monte Carlo: add noise to projected category changes
    mc_results = []
    for _ in range(n_sims):
        noise = np.random.normal(1.0, 0.08, size=len(config.all_categories))
        sim_change = sum(category_impact[cat] * noise[i] for i, cat in enumerate(config.all_categories))
        mc_results.append(sim_change)

    mc_results = np.array(mc_results)
    pct_positive = (mc_results > 0).mean() * 100

    # Risk flags
    risk_flags = []
    giving_players = player_pool[player_pool["player_id"].isin(giving_ids)]
    receiving_players = player_pool[player_pool["player_id"].isin(receiving_ids)]

    for _, p in receiving_players.iterrows():
        if p.get("is_injured", 0):
            risk_flags.append(f"{p.get('player_name', p.get('name', '?'))} is injured")

    for _, p in giving_players.iterrows():
        sgp = sgp_calc.total_sgp(p)
        if sgp > 3.0:
            risk_flags.append(f"Trading away elite player: {p.get('player_name', p.get('name', '?'))}")

    verdict = "ACCEPT" if pct_positive >= 55 else "DECLINE"

    return {
        "verdict": verdict,
        "confidence_pct": round(pct_positive, 1),
        "category_impact": category_impact,
        "total_sgp_change": round(total_sgp_change, 3),
        "mc_mean": round(float(mc_results.mean()), 3),
        "mc_std": round(float(mc_results.std()), 3),
        "risk_flags": risk_flags,
        "before_totals": before_totals,
        "after_totals": after_totals,
    }


def compare_players(
    player_id_a: int,
    player_id_b: int,
    player_pool: pd.DataFrame,
    config: LeagueConfig,
) -> dict:
    """Compare two players using z-score normalization across all categories."""
    player_a = player_pool[player_pool["player_id"] == player_id_a]
    player_b = player_pool[player_pool["player_id"] == player_id_b]

    if player_a.empty or player_b.empty:
        return {"error": "Player not found"}

    pa = player_a.iloc[0]
    pb = player_b.iloc[0]

    stat_map = dict(config.STAT_MAP)

    is_hitter_a = bool(pa.get("is_hitter", 1))
    is_hitter_b = bool(pb.get("is_hitter", 1))

    z_a, z_b = {}, {}
    for cat in config.all_categories:
        col = stat_map[cat]
        is_hitting_cat = cat in config.hitting_categories

        # Filter pool by player type so z-scores reflect the correct peer group
        if is_hitting_cat:
            peer_pool = player_pool[player_pool["is_hitter"] == 1]
        else:
            peer_pool = player_pool[player_pool["is_hitter"] == 0]
            # Filter out 0-IP pitchers for ERA/WHIP to avoid inflated z-scores
            if cat in config.inverse_stats and "ip" in peer_pool.columns:
                peer_pool = peer_pool[peer_pool["ip"].fillna(0) > 0]

        vals = peer_pool[col].dropna()
        mean = vals.mean()
        std = vals.std()
        if std == 0 or pd.isna(std):
            std = 1.0

        val_a = float(pa.get(col, 0) or 0)
        val_b = float(pb.get(col, 0) or 0)

        # Skip categories that don't apply to the player's type
        # (hitters don't have meaningful pitching stats, pitchers don't have meaningful hitting stats)
        a_applicable = is_hitting_cat == is_hitter_a
        b_applicable = is_hitting_cat == is_hitter_b

        if cat in config.inverse_stats:
            z_a[cat] = -(val_a - mean) / std if a_applicable else 0.0
            z_b[cat] = -(val_b - mean) / std if b_applicable else 0.0
        else:
            z_a[cat] = (val_a - mean) / std if a_applicable else 0.0
            z_b[cat] = (val_b - mean) / std if b_applicable else 0.0

    composite_a = sum(z_a.values())
    composite_b = sum(z_b.values())

    advantages = {}
    for cat in config.all_categories:
        if z_a[cat] > z_b[cat]:
            advantages[cat] = "A"
        elif z_b[cat] > z_a[cat]:
            advantages[cat] = "B"
        else:
            advantages[cat] = "TIE"

    return {
        "player_a": pa.get("player_name", pa.get("name", "?")),
        "player_b": pb.get("player_name", pb.get("name", "?")),
        "z_scores_a": z_a,
        "z_scores_b": z_b,
        "composite_a": round(composite_a, 3),
        "composite_b": round(composite_b, 3),
        "advantages": advantages,
    }


def rank_free_agents(
    user_roster_ids: list,
    fa_pool: pd.DataFrame,
    full_pool: pd.DataFrame,
    config: LeagueConfig,
    max_candidates: int = 200,
) -> pd.DataFrame:
    """Rank free agents by marginal value relative to user's roster.

    Pre-filters to top ``max_candidates`` by ADP before computing marginal
    SGP to avoid iterating over 9,000+ players row-by-row.
    """
    sgp_calc = SGPCalculator(config)
    roster_totals = _roster_category_totals(user_roster_ids, full_pool)
    weights = compute_category_weights(roster_totals, config)

    # Pre-filter: only rank FAs plausibly worth adding (top N by ADP)
    if len(fa_pool) > max_candidates and "adp" in fa_pool.columns:
        fa_pool = fa_pool.nsmallest(max_candidates, "adp")

    records = []
    for _, fa in fa_pool.iterrows():
        marginal = sgp_calc.marginal_sgp(fa, roster_totals, weights)
        total_marginal = sum(marginal.values())

        best_cat = max(marginal, key=marginal.get) if marginal else ""
        best_cat_val = marginal.get(best_cat, 0)

        records.append(
            {
                "player_id": fa["player_id"],
                "player_name": fa.get("player_name", fa.get("name", "?")),
                "positions": fa.get("positions", ""),
                "marginal_value": round(total_marginal, 3),
                "best_category": best_cat,
                "best_cat_impact": round(best_cat_val, 3),
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("marginal_value", ascending=False).reset_index(drop=True)
        # Preserve enriched columns from the full FA pool (Statcast, regression flags, health)
        _enrich_cols = [
            "regression_flag",
            "babip_regression_flag",
            "stuff_regression_flag",
            "velo_regression_flag",
            "xwoba",
            "barrel_pct",
            "hard_hit_pct",
            "stuff_plus",
            "health_score",
            "is_hitter",
            "team",
            "ytd_avg",
            "ytd_hr",
            "ytd_rbi",
            "ytd_sb",
            "ytd_era",
            "ytd_whip",
            "ytd_k",
            "ytd_sv",
            "ytd_pa",
            "consensus_rank",
            "adp",
        ]
        _available = [c for c in _enrich_cols if c in fa_pool.columns and c not in result.columns]
        if _available and "player_id" in result.columns and "player_id" in fa_pool.columns:
            _merge_src = fa_pool[["player_id"] + _available].drop_duplicates("player_id")
            result = result.merge(_merge_src, on="player_id", how="left")
    return result
