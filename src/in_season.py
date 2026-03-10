"""In-season analysis: trade analyzer, player comparison, free agent ranker.

Reuses SGPCalculator, compute_replacement_levels, compute_category_weights
from src/valuation.py. MC simulation pattern adapted from src/simulation.py.
"""

import numpy as np
import pandas as pd

from src.valuation import (
    LeagueConfig,
    SGPCalculator,
    compute_category_weights,
)


def _roster_category_totals(roster_ids: list, player_pool: pd.DataFrame) -> dict:
    """Compute aggregate category totals for a set of player IDs."""
    roster = player_pool[player_pool["player_id"].isin(roster_ids)]
    totals = {
        "R": 0, "HR": 0, "RBI": 0, "SB": 0,
        "W": 0, "SV": 0, "K": 0,
        "ab": 0, "h": 0, "ip": 0, "er": 0,
        "bb_allowed": 0, "h_allowed": 0,
    }
    for _, p in roster.iterrows():
        totals["R"] += int(p.get("r", 0) or 0)
        totals["HR"] += int(p.get("hr", 0) or 0)
        totals["RBI"] += int(p.get("rbi", 0) or 0)
        totals["SB"] += int(p.get("sb", 0) or 0)
        totals["W"] += int(p.get("w", 0) or 0)
        totals["SV"] += int(p.get("sv", 0) or 0)
        totals["K"] += int(p.get("k", 0) or 0)
        totals["ab"] += int(p.get("ab", 0) or 0)
        totals["h"] += int(p.get("h", 0) or 0)
        totals["ip"] += float(p.get("ip", 0) or 0)
        totals["er"] += int(p.get("er", 0) or 0)
        totals["bb_allowed"] += int(p.get("bb_allowed", 0) or 0)
        totals["h_allowed"] += int(p.get("h_allowed", 0) or 0)

    if totals["ab"] > 0:
        totals["AVG"] = totals["h"] / totals["ab"]
    else:
        totals["AVG"] = 0
    if totals["ip"] > 0:
        totals["ERA"] = totals["er"] * 9 / totals["ip"]
        totals["WHIP"] = (totals["bb_allowed"] + totals["h_allowed"]) / totals["ip"]
    else:
        totals["ERA"] = 0
        totals["WHIP"] = 0

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
    after_ids = [pid for pid in before_ids if pid not in giving_ids] + list(
        receiving_ids
    )

    before_totals = _roster_category_totals(before_ids, player_pool)
    after_totals = _roster_category_totals(after_ids, player_pool)

    category_impact = {}
    total_sgp_change = 0.0
    for cat in config.all_categories:
        denom = config.sgp_denominators[cat]
        if denom == 0:
            denom = 1.0

        before_val = before_totals.get(cat, 0)
        after_val = after_totals.get(cat, 0)
        raw_change = after_val - before_val

        if cat in config.inverse_stats:
            sgp_change = -raw_change / denom
        else:
            sgp_change = raw_change / denom

        category_impact[cat] = round(sgp_change, 3)
        total_sgp_change += sgp_change

    # Monte Carlo: add noise to projected category changes
    mc_results = []
    for _ in range(n_sims):
        noise = np.random.normal(1.0, 0.08, size=len(config.all_categories))
        sim_change = sum(
            category_impact[cat] * noise[i]
            for i, cat in enumerate(config.all_categories)
        )
        mc_results.append(sim_change)

    mc_results = np.array(mc_results)
    pct_positive = (mc_results > 0).mean() * 100

    # Risk flags
    risk_flags = []
    giving_players = player_pool[player_pool["player_id"].isin(giving_ids)]
    receiving_players = player_pool[player_pool["player_id"].isin(receiving_ids)]

    for _, p in receiving_players.iterrows():
        if p.get("is_injured", 0):
            risk_flags.append(
                f"{p.get('player_name', p.get('name', '?'))} is injured"
            )

    for _, p in giving_players.iterrows():
        sgp = sgp_calc.total_sgp(p)
        if sgp > 3.0:
            risk_flags.append(
                f"Trading away elite player: "
                f"{p.get('player_name', p.get('name', '?'))}"
            )

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

    stat_map = {
        "R": "r", "HR": "hr", "RBI": "rbi", "SB": "sb", "AVG": "avg",
        "W": "w", "SV": "sv", "K": "k", "ERA": "era", "WHIP": "whip",
    }

    z_a, z_b = {}, {}
    for cat in config.all_categories:
        col = stat_map[cat]
        vals = player_pool[col].dropna()
        mean = vals.mean()
        std = vals.std()
        if std == 0:
            std = 1.0

        val_a = float(pa.get(col, 0) or 0)
        val_b = float(pb.get(col, 0) or 0)

        if cat in config.inverse_stats:
            z_a[cat] = -(val_a - mean) / std
            z_b[cat] = -(val_b - mean) / std
        else:
            z_a[cat] = (val_a - mean) / std
            z_b[cat] = (val_b - mean) / std

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
) -> pd.DataFrame:
    """Rank free agents by marginal value relative to user's roster."""
    sgp_calc = SGPCalculator(config)
    roster_totals = _roster_category_totals(user_roster_ids, full_pool)
    weights = compute_category_weights(roster_totals, config)

    records = []
    for _, fa in fa_pool.iterrows():
        marginal = sgp_calc.marginal_sgp(fa, roster_totals, weights)
        total_marginal = sum(marginal.values())

        best_cat = max(marginal, key=marginal.get) if marginal else ""
        best_cat_val = marginal.get(best_cat, 0)

        records.append({
            "player_id": fa["player_id"],
            "player_name": fa.get("player_name", fa.get("name", "?")),
            "positions": fa.get("positions", ""),
            "marginal_value": round(total_marginal, 3),
            "best_category": best_cat,
            "best_cat_impact": round(best_cat_val, 3),
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(
            "marginal_value", ascending=False
        ).reset_index(drop=True)
    return result
