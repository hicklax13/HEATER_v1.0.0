"""Full pipeline trace: what does recommend_fa_moves see at each stage
post-PR19? Why does it produce zero recommendations?
"""

from __future__ import annotations

import io
import logging
import sys

import pandas as pd

from src.database import init_db
from src.league_manager import get_team_roster
from src.optimizer.fa_recommender import (
    _MAX_DROP_CANDIDATES,
    _MAX_FA_CANDIDATES,
    _score_drop_candidates,
    _score_fa_candidates,
    _evaluate_swaps,
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

    print(f"User roster: {len(ctx.user_roster_ids)} players")
    print(f"FA pool: {len(ctx.free_agents)} players")
    print()

    # Score candidates
    from src.valuation import SGPCalculator, compute_replacement_levels
    repl = compute_replacement_levels(ctx.player_pool, ctx.config, SGPCalculator(ctx.config))

    drops = _score_drop_candidates(ctx, repl)
    fas = _score_fa_candidates(ctx, repl)

    print("DROP CANDIDATES (top 5 lowest drop_cost):")
    for d in drops:
        print(f"  {d['name']:25} {d['positions']:20} is_hitter={d['is_hitter']} drop_cost={d['drop_cost']:.2f}")
    print()
    print("FA CANDIDATES (top 10 highest composite):")
    for f in fas:
        print(f"  {f['name']:25} {f['positions']:20} is_hitter={f['is_hitter']} composite={f['composite_score']:.2f} is_il={f['is_il']}")
    print()

    # Try evaluating swaps — show what blocks each
    print("=" * 70)
    print("SWAP EVALUATION (50 pairs)")
    print("=" * 70)
    swap_results = _evaluate_swaps(ctx, drops, fas)
    if swap_results:
        print(f"{len(swap_results)} viable swaps found.")
        for s in swap_results[:5]:
            print(f"  +{s['net_sgp_delta']:.2f} ADD {s['add_name']} ← DROP {s['drop_name']}")
    else:
        print("ZERO viable swaps. Checking why...")
        # Manually iterate to find blocking reasons
        from src.optimizer.fa_recommender import (
            _allow_cross_type,
            _passes_roster_construction_guard,
        )
        from src.waiver_wire import compute_net_swap_value
        losing = []  # placeholder, doesn't affect for diagnostic
        for fa in fas[:5]:
            for drop in drops[:5]:
                # IL match
                if bool(drop.get("is_il", False)) != bool(fa.get("is_il", False)):
                    continue
                # Cross-type
                cross_blocked = ""
                if fa["is_hitter"] != drop["is_hitter"]:
                    if not _allow_cross_type(ctx, fa, drop, set(losing)):
                        cross_blocked = "CROSS-TYPE-BLOCKED"
                if cross_blocked:
                    print(f"  {fa['name']:20} ← {drop['name']:20}: {cross_blocked}")
                    continue
                # Net SGP
                swap = compute_net_swap_value(fa["player_id"], drop["player_id"], ctx.user_roster_ids, ctx.player_pool, ctx.config)
                net_sgp = swap["net_sgp"]
                if net_sgp <= 0:
                    print(f"  {fa['name']:20} ← {drop['name']:20}: NEG net_sgp ({net_sgp:.2f})")
                    continue
                # Construction guard
                pool_match_fa = ctx.player_pool[ctx.player_pool["player_id"] == fa["player_id"]]
                pool_match_drop = ctx.player_pool[ctx.player_pool["player_id"] == drop["player_id"]]
                if pool_match_fa.empty or pool_match_drop.empty:
                    continue
                passes, reason = _passes_roster_construction_guard(pool_match_fa.iloc[0], pool_match_drop.iloc[0], ctx)
                if not passes:
                    print(f"  {fa['name']:20} ← {drop['name']:20}: CONSTRUCTION-GUARD-BLOCKED ({reason})")
                else:
                    print(f"  {fa['name']:20} ← {drop['name']:20}: PASSES ALL GATES (+{net_sgp:.2f} SGP)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
