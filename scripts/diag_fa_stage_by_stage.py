"""Stage-by-stage trace of recommend_fa_moves: where do the 5 viable
swaps from _evaluate_swaps die before reaching the final output?"""

from __future__ import annotations

import io
import logging
import sys

from src.database import init_db
from src.league_manager import get_team_roster
from src.optimizer.fa_recommender import (
    _evaluate_swaps,
    _score_drop_candidates,
    _score_fa_candidates,
    _deduplicate_and_limit,
    recommend_fa_moves,
)
from src.optimizer.shared_data_layer import build_optimizer_context
from src.validation.dynamic_context import compute_weeks_remaining
from src.valuation import LeagueConfig, SGPCalculator, compute_replacement_levels
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

    print(f"ctx.adds_remaining_this_week = {ctx.adds_remaining_this_week}")
    print(f"ctx.user_roster_ids = {len(ctx.user_roster_ids)} players")
    print(f"ctx.player_pool = {ctx.player_pool.shape}")
    print()

    # Stage 1: replacement levels
    repl = compute_replacement_levels(ctx.player_pool, ctx.config, SGPCalculator(ctx.config))
    print(f"Stage 1 — replacement_levels: {len(repl)} positions")

    # Stage 2: drop candidates
    drops = _score_drop_candidates(ctx, repl)
    print(f"Stage 2 — _score_drop_candidates: {len(drops)} drops")

    # Stage 3: fa candidates
    fas = _score_fa_candidates(ctx, repl)
    print(f"Stage 3 — _score_fa_candidates: {len(fas)} FAs")

    # Stage 4: evaluate swaps
    swaps = _evaluate_swaps(ctx, drops, fas)
    print(f"Stage 4 — _evaluate_swaps: {len(swaps)} viable swaps")
    for s in swaps[:5]:
        print(f"    +{s['net_sgp_delta']:.2f} {s['add_name']} ← {s['drop_name']}")

    # Stage 5: dedupe
    final = _deduplicate_and_limit(swaps, 5)
    print(f"Stage 5 — _deduplicate_and_limit (max_moves=5): {len(final)} recs")
    for f in final:
        print(f"    +{f['net_sgp_delta']:.2f} {f['add_name']} ← {f['drop_name']}")

    print()
    # Stage 6: full public API call
    moves = recommend_fa_moves(ctx, max_moves=5)
    print(f"Stage 6 — recommend_fa_moves (public): {len(moves)} recs")
    for m in moves:
        print(f"    +{m['net_sgp_delta']:.2f} {m['add_name']} ← {m['drop_name']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
