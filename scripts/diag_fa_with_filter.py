"""Diagnostic v2: verify the fix theory by manually filtering rosters
to the user's team before building the context. If the engine then
produces sensible recs (no cross-type bypass), the fix is confirmed.
"""

from __future__ import annotations

import io
import logging
import sys

from src.database import init_db
from src.optimizer.fa_recommender import (
    _HITTER_SLOTS,
    recommend_fa_moves,
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

    # FIX APPLIED: filter rosters to user's team only before passing in.
    user_rosters_only = rosters[rosters["team_name"] == user_team_name].copy()
    print(f"User team: {user_team_name}")
    print(f"Original rosters shape: {rosters.shape} (all 12 teams)")
    print(f"Filtered rosters shape: {user_rosters_only.shape} (user team only)")

    ctx = build_optimizer_context(
        scope="rest_of_season",
        yds=yds,
        config=config,
        weeks_remaining=compute_weeks_remaining(),
        user_team_name=user_team_name,
        roster=user_rosters_only,  # ← FIX: pre-filtered
    )

    print()
    print(f"ctx.user_roster_ids length: {len(ctx.user_roster_ids)}  (expected ~27)")

    # Count hitters/pitchers per engine logic
    pool = ctx.player_pool
    hitter_count = 0
    pitcher_count = 0
    for pid in ctx.user_roster_ids:
        match = pool[pool["player_id"] == pid]
        if match.empty:
            continue
        if bool(int(match.iloc[0].get("is_hitter", 1))):
            hitter_count += 1
        else:
            pitcher_count += 1
    print(f"Engine hitter_count: {hitter_count}")
    print(f"Engine pitcher_count: {pitcher_count}")
    print(f"_HITTER_SLOTS: {_HITTER_SLOTS}")
    print(f"Cross-type guard: hitter_count >= _HITTER_SLOTS = {hitter_count >= _HITTER_SLOTS}")

    # Now run recommend_fa_moves and see what it produces.
    print()
    print("=" * 70)
    print("RECOMMENDATIONS with fix applied:")
    print("=" * 70)
    moves = recommend_fa_moves(ctx, max_moves=5)
    if not moves:
        print("(no recommendations produced — possibly all blocked or no positive swaps)")
    else:
        for i, m in enumerate(moves, 1):
            print(
                f"  {i}. ADD {m.get('add_name')} ({m.get('add_positions')}) ← DROP {m.get('drop_name')} ({m.get('drop_positions')})"
            )
            print(f"     net_sgp_delta = {m.get('net_sgp_delta'):.2f}")
            cat_impact = m.get("category_impact", {})
            top_cats = sorted(cat_impact.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]
            for cat, val in top_cats:
                sign = "+" if val >= 0 else ""
                print(f"     {cat}: {sign}{val:.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
