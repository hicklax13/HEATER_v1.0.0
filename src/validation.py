"""Validation and backtesting framework for the draft tool."""

import copy

import numpy as np
import pandas as pd

from src.draft_state import DraftState
from src.simulation import DraftSimulator
from src.valuation import LeagueConfig, SGPCalculator, value_all_players


def simulate_full_draft(
    player_pool: pd.DataFrame,
    config: LeagueConfig,
    user_team_index: int = 0,
    sigma: float = 10.0,
    use_tool: bool = True,
) -> dict:
    """Simulate a complete 12-team, 23-round draft.

    Args:
        player_pool: Full player pool with projections and ADP.
        config: League configuration.
        user_team_index: Index of the user's team.
        sigma: ADP noise for opponent modeling.
        use_tool: If True, user picks by tool's recommendation. If False, picks by ADP.

    Returns:
        dict with 'user_roster' (list of player dicts), 'user_sgp', 'all_sgp' (list per team),
        'user_rank', 'pick_log'.
    """
    ds = DraftState(num_teams=config.num_teams, num_rounds=23, user_team_index=user_team_index)
    sgp_calc = SGPCalculator(config)
    simulator = DraftSimulator(config, sigma=sigma)
    rng = np.random.default_rng()

    pool = player_pool.copy()
    pool["_adp_sort"] = pool["adp"].fillna(999)

    for pick_num in range(ds.total_picks):
        available = ds.available_players(pool)
        if available.empty:
            break

        team_idx = ds.picking_team_index()

        if team_idx == user_team_index and use_tool:
            # User picks by tool recommendation (greedy marginal SGP)
            roster_totals = ds.get_user_roster_totals(pool)
            valued = value_all_players(available, config, roster_totals)
            if valued.empty:
                break
            best = valued.iloc[0]
        elif team_idx == user_team_index:
            # User picks by ADP (baseline comparison)
            best = available.sort_values("_adp_sort").iloc[0]
        else:
            # Opponent picks by ADP + noise
            adp_vals = available["_adp_sort"].values
            weights = np.exp(-np.abs(adp_vals - pick_num) / sigma)
            weight_sum = weights.sum()
            if weight_sum <= 0:
                weights = np.ones(len(available))
                weight_sum = len(available)
            probs = weights / weight_sum
            chosen_idx = rng.choice(len(available), p=probs)
            best = available.iloc[chosen_idx]

        ds.make_pick(
            int(best["player_id"]),
            best["name"],
            str(best["positions"]),
        )

    # Score all teams
    all_team_sgp = []
    for team in ds.teams:
        team_sgp = 0
        for _, pid, _ in team.picks:
            player = pool[pool["player_id"] == pid]
            if not player.empty:
                team_sgp += sgp_calc.total_sgp(player.iloc[0])
        all_team_sgp.append(team_sgp)

    user_sgp = all_team_sgp[user_team_index]
    user_rank = sum(1 for s in all_team_sgp if s > user_sgp) + 1

    user_roster = []
    for _, pid, pname in ds.user_team.picks:
        player = pool[pool["player_id"] == pid]
        if not player.empty:
            p = player.iloc[0]
            user_roster.append(
                {
                    "name": p["name"],
                    "positions": p["positions"],
                    "total_sgp": sgp_calc.total_sgp(p),
                }
            )

    return {
        "user_sgp": user_sgp,
        "all_sgp": all_team_sgp,
        "user_rank": user_rank,
        "user_roster": user_roster,
        "pick_log": ds.pick_log,
    }


def run_benchmark(
    player_pool: pd.DataFrame, config: LeagueConfig, n_simulations: int = 100, sigma: float = 10.0
) -> dict:
    """Run a full benchmark: compare tool-assisted drafting vs ADP-based drafting.

    Returns summary statistics for both approaches.
    """
    tool_ranks = []
    tool_sgps = []
    adp_ranks = []
    adp_sgps = []

    for i in range(n_simulations):
        # Randomize user draft position
        user_pos = i % config.num_teams

        # Tool-assisted draft
        tool_result = simulate_full_draft(player_pool, config, user_team_index=user_pos, sigma=sigma, use_tool=True)
        tool_ranks.append(tool_result["user_rank"])
        tool_sgps.append(tool_result["user_sgp"])

        # ADP-based draft (baseline)
        adp_result = simulate_full_draft(player_pool, config, user_team_index=user_pos, sigma=sigma, use_tool=False)
        adp_ranks.append(adp_result["user_rank"])
        adp_sgps.append(adp_result["user_sgp"])

    return {
        "n_simulations": n_simulations,
        "tool": {
            "avg_rank": np.mean(tool_ranks),
            "median_rank": np.median(tool_ranks),
            "top3_pct": sum(1 for r in tool_ranks if r <= 3) / n_simulations * 100,
            "top6_pct": sum(1 for r in tool_ranks if r <= 6) / n_simulations * 100,
            "avg_sgp": np.mean(tool_sgps),
            "std_sgp": np.std(tool_sgps),
        },
        "adp_baseline": {
            "avg_rank": np.mean(adp_ranks),
            "median_rank": np.median(adp_ranks),
            "top3_pct": sum(1 for r in adp_ranks if r <= 3) / n_simulations * 100,
            "top6_pct": sum(1 for r in adp_ranks if r <= 6) / n_simulations * 100,
            "avg_sgp": np.mean(adp_sgps),
            "std_sgp": np.std(adp_sgps),
        },
    }


def sensitivity_analysis(player_pool: pd.DataFrame, config: LeagueConfig, n_per_test: int = 30) -> dict:
    """Test sensitivity of results to key parameters."""
    results = {}

    # Test different SGP denominator scales
    for scale in [0.8, 1.0, 1.2]:
        test_config = copy.deepcopy(config)
        test_config.sgp_denominators = {k: v * scale for k, v in config.sgp_denominators.items()}
        benchmark = run_benchmark(player_pool, test_config, n_per_test)
        results[f"sgp_scale_{scale}"] = benchmark["tool"]["avg_rank"]

    # Test different sigma values (opponent modeling tightness)
    for sigma in [6, 10, 15, 20]:
        benchmark = run_benchmark(player_pool, config, n_per_test, sigma=sigma)
        results[f"sigma_{sigma}"] = benchmark["tool"]["avg_rank"]

    return results


def _format_key_stats(player, is_hitter: bool) -> str:
    if is_hitter:
        parts = []
        hr = int(player.get("hr", 0) or 0)
        sb = int(player.get("sb", 0) or 0)
        r = int(player.get("r", 0) or 0)
        avg = player.get("avg", 0) or 0
        if hr > 0:
            parts.append(f"{hr}HR")
        if sb > 0:
            parts.append(f"{sb}SB")
        if r > 0:
            parts.append(f"{r}R")
        if avg > 0:
            parts.append(f".{round(avg * 1000):03d}")
        return " | ".join(parts[:4])
    else:
        parts = []
        w = int(player.get("w", 0) or 0)
        k = int(player.get("k", 0) or 0)
        sv = int(player.get("sv", 0) or 0)
        era = player.get("era", 0) or 0
        if w > 0:
            parts.append(f"{w}W")
        if sv > 0:
            parts.append(f"{sv}SV")
        if k > 0:
            parts.append(f"{k}K")
        if era > 0:
            parts.append(f"{era:.2f}ERA")
        return " | ".join(parts[:4])
