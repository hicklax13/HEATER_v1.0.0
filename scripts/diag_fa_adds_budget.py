"""Quick check: is ctx.adds_remaining_this_week=0 causing the engine to
return [] early in recommend_fa_moves?"""

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

    print(f"User team: {user_team_name}")
    print(f"ctx.adds_remaining_this_week = {ctx.adds_remaining_this_week}")
    print()

    # Inspect transaction data
    txns = yds.get_transactions()
    print(f"Total transactions in DB: {len(txns)}")
    if not txns.empty:
        print(f"Columns: {list(txns.columns)}")
        adds = txns[txns["type"].str.lower() == "add"] if "type" in txns.columns else txns
        print(f"Total 'add' transactions across ALL teams: {len(adds)}")
        if "team_name" in adds.columns:
            user_adds = adds[adds["team_name"].astype(str).str.contains(user_team_name.replace("🏆 ", ""), na=False, regex=False)]
            print(f"User's add transactions: {len(user_adds)}")
        else:
            print(f"team_name column not present — can't filter to user")
            print(f"Sample rows: {adds.head(3).to_dict('records')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
