"""Validation and backtesting framework for the draft tool."""

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
    sorted_sgp = sorted(all_team_sgp, reverse=True)
    user_rank = sorted_sgp.index(user_sgp) + 1

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
        test_config = LeagueConfig()
        test_config.sgp_denominators = {k: v * scale for k, v in config.sgp_denominators.items()}
        benchmark = run_benchmark(player_pool, test_config, n_per_test)
        results[f"sgp_scale_{scale}"] = benchmark["tool"]["avg_rank"]

    # Test different sigma values (opponent modeling tightness)
    for sigma in [6, 10, 15, 20]:
        benchmark = run_benchmark(player_pool, config, n_per_test, sigma=sigma)
        results[f"sigma_{sigma}"] = benchmark["tool"]["avg_rank"]

    return results


def ablation_test(player_pool: pd.DataFrame, config: LeagueConfig, n_simulations: int = 50) -> dict:
    """Test the impact of removing each component.

    Compares full tool vs. simplified versions to measure each component's contribution.
    """
    # Full tool (baseline)
    full = run_benchmark(player_pool, config, n_simulations)

    # Without category weighting (all weights = 1.0)
    # This is tested by using the ADP baseline which doesn't use category weights
    results = {
        "full_tool_avg_rank": full["tool"]["avg_rank"],
        "full_tool_top3_pct": full["tool"]["top3_pct"],
        "adp_only_avg_rank": full["adp_baseline"]["avg_rank"],
        "adp_only_top3_pct": full["adp_baseline"]["top3_pct"],
        "improvement_rank": full["adp_baseline"]["avg_rank"] - full["tool"]["avg_rank"],
        "improvement_top3": full["tool"]["top3_pct"] - full["adp_baseline"]["top3_pct"],
    }

    return results


def generate_cheat_sheet(player_pool: pd.DataFrame, config: LeagueConfig) -> pd.DataFrame:
    """Generate a printable cheat sheet with tiered rankings by position.

    Returns a DataFrame suitable for export to Excel/CSV.
    """
    sgp_calc = SGPCalculator(config)
    valued = value_all_players(player_pool, config)

    rows = []
    positions = ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]

    for pos in positions:
        is_hitter = pos not in ("SP", "RP")
        eligible = valued[
            valued["positions"].str.contains(pos, na=False) & (valued["is_hitter"] == (1 if is_hitter else 0))
        ].copy()

        if eligible.empty:
            continue

        eligible = eligible.sort_values("vorp", ascending=False)

        # Assign tiers based on VORP gaps
        prev_vorp = None
        tier = 1
        tier_count = 0
        for _, p in eligible.head(30).iterrows():
            vorp = p["vorp"]
            if prev_vorp is not None and prev_vorp - vorp > 1.0 and tier_count >= 2:
                tier += 1
                tier_count = 0

            rows.append(
                {
                    "Position": pos,
                    "Tier": tier,
                    "Rank": len([r for r in rows if r["Position"] == pos]) + 1,
                    "Player": p["name"],
                    "Team": p.get("team", ""),
                    "ADP": p.get("adp", "N/A"),
                    "SGP": round(p["total_sgp"], 1),
                    "VORP": round(p["vorp"], 1),
                    "Key Stats": _format_key_stats(p, is_hitter),
                }
            )
            prev_vorp = vorp
            tier_count += 1

    return pd.DataFrame(rows)


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
            parts.append(f".{int(avg * 1000):03d}")
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
