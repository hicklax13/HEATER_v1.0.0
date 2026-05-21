"""What does the engine actually 'see' for Max Muncy?
He's having a top-75 season (.263 / 12 HR / 36 R in 47 GP) but the engine
ranks him as one of the cheapest hitters to drop. Check the pool data."""

from __future__ import annotations

import io
import logging
import sys

from src.database import init_db
from src.league_manager import get_team_roster
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

    pool = ctx.player_pool
    name_col = "player_name" if "player_name" in pool.columns else "name"

    # Compare Muncy (rostered, recommended to drop) vs Westburg (FA, recommended to add)
    for name in ["Max Muncy", "Jordan Westburg", "Bryson Stott"]:
        match = pool[pool[name_col].str.lower() == name.lower()]
        if match.empty:
            print(f"{name}: NOT IN POOL")
            continue
        p = match.iloc[0]
        print(f"=== {name} ===")
        print(f"  positions: {p.get('positions')}, is_hitter: {p.get('is_hitter')}, ECR: {p.get('consensus_rank')}")
        print(f"  ROS-projection columns (used by _roster_category_totals):")
        print(f"    r={p.get('r')}, hr={p.get('hr')}, rbi={p.get('rbi')}, sb={p.get('sb')}")
        print(f"    avg={p.get('avg')}, obp={p.get('obp')}, ab={p.get('ab')}, h={p.get('h')}")
        print(f"  YTD columns (used by _blend_fa_row in fa_recommender):")
        print(f"    ytd_r={p.get('ytd_r')}, ytd_hr={p.get('ytd_hr')}, ytd_rbi={p.get('ytd_rbi')}, ytd_sb={p.get('ytd_sb')}")
        print(f"    ytd_avg={p.get('ytd_avg')}, ytd_obp={p.get('ytd_obp')}, ytd_gp={p.get('ytd_gp')}")
        print(f"    ytd_pa={p.get('ytd_pa')}, ytd_h={p.get('ytd_h')}, ytd_ab={p.get('ytd_ab')}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
