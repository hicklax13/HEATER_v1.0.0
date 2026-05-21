"""Diagnostic: Why do unknown minor leaguers score composite=20.6
while Brandon Marsh (.33 AVG / .44 L14) doesn't make the top 10?

Inspect the _compute_base_value internals for a sample of FAs to
identify the scoring fallback that's inflating unknowns.
"""

from __future__ import annotations

import io
import logging
import sys

import pandas as pd

from src.database import init_db
from src.league_manager import get_team_roster
from src.optimizer.fa_recommender import (
    _blend_fa_row,
    _compute_base_value,
    _positional_scarcity_factor,
)
from src.optimizer.shared_data_layer import build_optimizer_context
from src.validation.dynamic_context import compute_weeks_remaining
from src.valuation import LeagueConfig
from src.yahoo_data_service import get_yahoo_data_service

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.WARNING)


def main() -> int:
    init_db()
    config = LeagueConfig()
    yds = get_yahoo_data_service()

    rosters = yds.get_rosters()
    user_teams = rosters[rosters["is_user_team"] == 1]
    user_team_name = user_teams.iloc[0]["team_name"]
    if isinstance(user_team_name, bytes):
        user_team_name = user_team_name.decode("utf-8", errors="replace")

    user_roster = get_team_roster(user_team_name)
    ctx = build_optimizer_context(
        scope="rest_of_season",
        yds=yds,
        config=config,
        weeks_remaining=compute_weeks_remaining(),
        user_team_name=user_team_name,
        roster=user_roster,
        level_filter="MLB only",
    )

    print(f"ctx.player_pool shape: {ctx.player_pool.shape}")
    print(f"ctx.free_agents shape: {ctx.free_agents.shape}")
    print(f"category_weights: {ctx.category_weights}")
    print(f"SGP denoms: {ctx.config.sgp_denominators}")
    print()

    # Check specific players
    targets = [
        "Brandon Marsh",  # should score well — actual MLB stats
        "Michael De La Cruz",  # minor leaguer
        "Cristian Hernandez",  # minor leaguer
        "Alejandro Kirk",  # backup MLB catcher
        "Jordan Westburg",  # MLB 2B/3B
        "Bryson Stott",  # MLB 2B
        "Cade Horton",  # MLB SP
        "Kyle Teel",  # MLB catcher (prospect)
    ]

    pool = ctx.player_pool
    name_col = "player_name" if "player_name" in pool.columns else "name"

    print("=" * 90)
    print(f"{'Name':25} {'Pos':18} {'is_h':5} {'PA':6} {'IP':6} {'base':8} {'scarcity':8} {'marg_val':8} {'lvl':6}")
    print("=" * 90)
    for name in targets:
        match = pool[pool[name_col].str.lower() == name.lower()]
        if match.empty:
            print(f"{name:25} NOT IN POOL")
            continue
        fa_data = match.iloc[0]
        pid = fa_data.get("player_id", "?")
        pos = str(fa_data.get("positions", "?"))[:18]
        ih = bool(int(fa_data.get("is_hitter", 1)))
        pa = float(fa_data.get("pa", 0) or 0)
        ip = float(fa_data.get("ip", 0) or 0)
        marg = fa_data.get("marginal_value")
        marg_str = f"{marg:.2f}" if pd.notna(marg) else "NaN"
        level = str(fa_data.get("level", "?"))[:6]

        # Manually compute base_value to see what's happening
        try:
            base = _compute_base_value(fa_data, ctx)
            base_str = f"{base:.2f}"
        except Exception as e:
            base_str = f"ERR:{e}"

        try:
            scarcity = _positional_scarcity_factor(str(fa_data.get("positions", "")), {})
            scar_str = f"{scarcity:.2f}"
        except Exception:
            scar_str = "ERR"

        print(f"{name:25} {pos:18} {str(ih):5} {pa:6.0f} {ip:6.1f} {base_str:8} {scar_str:8} {marg_str:8} {level:6}")

    # Now show what's in the blended row for an unknown minor leaguer
    print()
    print("=" * 90)
    print("BLENDED ROW for Michael De La Cruz (suspected to score 20.6)")
    print("=" * 90)
    match = pool[pool[name_col].str.lower() == "michael de la cruz"]
    if not match.empty:
        fa_data = match.iloc[0]
        blended = _blend_fa_row(fa_data)
        # Show relevant columns
        relevant_cols = list(ctx.config.STAT_MAP.values()) + ["pa", "ip", "ytd_gp", "marginal_value", "level"]
        for col in relevant_cols:
            if col in blended.index:
                val = blended[col]
                print(f"  {col:25} = {val!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
