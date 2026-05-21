"""Diagnostic: trace why _allow_cross_type doesn't block the 3 bad FA recs.

Reproduces the page's call chain to build the same OptimizerDataContext,
then inspects:
  1. ctx.user_roster_ids — is it the user's 27 or the league's 156?
  2. is_hitter for each of the 4 drop candidates and 4 FA candidates
  3. What _allow_cross_type returns for each bad pair

Run: python scripts/diag_fa_cross_type.py
"""

from __future__ import annotations

import io
import logging
import sys

from src.database import init_db
from src.league_manager import get_team_roster
from src.optimizer.fa_recommender import (
    _HITTER_SLOTS,
    _MAX_DROP_CANDIDATES,
    _MAX_FA_CANDIDATES,
    _allow_cross_type,
    _score_drop_candidates,
    _score_fa_candidates,
)
from src.optimizer.shared_data_layer import build_optimizer_context
from src.validation.dynamic_context import compute_weeks_remaining
from src.valuation import LeagueConfig
from src.yahoo_data_service import get_yahoo_data_service

# Force UTF-8 stdout (Windows cp1252 default chokes on emoji in team names).
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Reduce log noise
logging.basicConfig(level=logging.WARNING)


def main() -> int:
    init_db()
    config = LeagueConfig()
    yds = get_yahoo_data_service()

    print("=" * 70)
    print("DIAGNOSTIC: FA cross-type guard trace")
    print("=" * 70)

    # Build context the way pages/14_Free_Agents.py does.
    rosters = yds.get_rosters()
    if rosters.empty:
        print("ERROR: rosters empty — Yahoo not connected?")
        return 1

    user_teams = rosters[rosters["is_user_team"] == 1]
    user_team_name = user_teams.iloc[0]["team_name"]
    if isinstance(user_team_name, bytes):
        user_team_name = user_team_name.decode("utf-8", errors="replace")
    print(f"User team: {user_team_name}")
    print(f"Rosters DataFrame shape (passed to ctx): {rosters.shape}")
    print(f"Rosters unique team_name count: {rosters['team_name'].nunique()}")

    ctx = build_optimizer_context(
        scope="rest_of_season",
        yds=yds,
        config=config,
        weeks_remaining=compute_weeks_remaining(),
        user_team_name=user_team_name,
        roster=rosters,  # ← same as the page does
    )

    print()
    print("=" * 70)
    print("CTX STATE")
    print("=" * 70)
    print(f"ctx.roster shape: {ctx.roster.shape}")
    print(
        f"ctx.roster team_name unique: {ctx.roster['team_name'].nunique() if 'team_name' in ctx.roster.columns else 'no team_name col'}"
    )
    print(f"ctx.user_roster_ids length: {len(ctx.user_roster_ids)}")
    print(f"ctx.user_roster_ids first 5: {ctx.user_roster_ids[:5]}")
    print(f"ctx.player_pool shape: {ctx.player_pool.shape}")
    print(f"ctx.free_agents shape: {ctx.free_agents.shape}")
    print(f"ctx.il_stash_ids size: {len(ctx.il_stash_ids)}")
    print(f"ctx.league_rostered_ids size: {len(ctx.league_rostered_ids)}")

    # Count hitters and pitchers in ctx.user_roster_ids per the engine's logic
    pool = ctx.player_pool
    hitter_count = 0
    pitcher_count = 0
    is_hitter_missing = 0
    for pid in ctx.user_roster_ids:
        match = pool[pool["player_id"] == pid]
        if match.empty:
            continue
        row = match.iloc[0]
        ih = row.get("is_hitter")
        if ih is None or (hasattr(ih, "__class__") and str(ih) == "nan"):
            is_hitter_missing += 1
        if not match.empty and bool(int(match.iloc[0].get("is_hitter", 1))):
            hitter_count += 1
        else:
            pitcher_count += 1

    print()
    print("Engine's count via ctx.user_roster_ids:")
    print(f"  hitter_count: {hitter_count}")
    print(f"  pitcher_count: {pitcher_count}")
    print(f"  is_hitter missing/null: {is_hitter_missing}")
    print(f"  _HITTER_SLOTS constant: {_HITTER_SLOTS}")
    print(f"  hitter_count >= _HITTER_SLOTS? {hitter_count >= _HITTER_SLOTS}")

    # Now check the user's ACTUAL team
    user_roster = get_team_roster(user_team_name)
    print()
    print(f"User's actual roster from get_team_roster: {len(user_roster)} players")

    # Check is_hitter for specific players from the bad recs
    bad_drops = ["Jack Leiter", "Framber Valdez", "Eduardo Rodriguez", "Seth Lugo"]
    bad_adds = ["Alejandro Kirk", "Jordan Westburg", "Bryson Stott", "Cade Horton"]

    print()
    print("=" * 70)
    print("is_hitter SANITY CHECK on bad pair players")
    print("=" * 70)
    for name in bad_drops + bad_adds:
        match = (
            pool[pool["player_name"].str.lower() == name.lower()]
            if "player_name" in pool.columns
            else pool[pool["name"].str.lower() == name.lower()]
        )
        if match.empty:
            print(f"  {name}: NOT IN POOL")
            continue
        row = match.iloc[0]
        ih = row.get("is_hitter")
        pos = row.get("positions", "?")
        pid = row.get("player_id", "?")
        print(f"  {name} (pid={pid}, pos={pos}): is_hitter={ih!r} → bool={bool(int(ih) if ih is not None else 1)}")

    # Score the drop and fa candidates to see what _evaluate_swaps would see
    print()
    print("=" * 70)
    print("DROP CANDIDATES (top _MAX_DROP_CANDIDATES = " + str(_MAX_DROP_CANDIDATES) + ")")
    print("=" * 70)
    drop_candidates = _score_drop_candidates(ctx)
    for d in drop_candidates:
        print(
            f"  {d['name']} (pid={d['player_id']}, pos={d['positions']}): is_hitter={d['is_hitter']}, drop_cost={d['drop_cost']:.3f}, is_il={d['is_il']}"
        )

    print()
    print("=" * 70)
    print("FA CANDIDATES (top _MAX_FA_CANDIDATES = " + str(_MAX_FA_CANDIDATES) + ")")
    print("=" * 70)
    fa_candidates = _score_fa_candidates(ctx)
    for f in fa_candidates:
        print(
            f"  {f['name']} (pid={f['player_id']}, pos={f['positions']}): is_hitter={f['is_hitter']}, composite={f['composite_score']:.3f}, is_il={f['is_il']}"
        )

    # Test _allow_cross_type for the 3 known bad pairs
    print()
    print("=" * 70)
    print("CROSS-TYPE CHECK on the 3 bad pairs")
    print("=" * 70)
    bad_pairs = [
        ("Alejandro Kirk", "Jack Leiter"),
        ("Jordan Westburg", "Framber Valdez"),
        ("Bryson Stott", "Eduardo Rodriguez"),
    ]
    target_cats: set[str] = set()  # empty (engine's value depends on losing/tied)
    for add_name, drop_name in bad_pairs:
        fa = next((f for f in fa_candidates if f["name"] == add_name), None)
        drop = next((d for d in drop_candidates if d["name"] == drop_name), None)
        if fa is None:
            print(f"  {add_name} ← {drop_name}: FA not in top {_MAX_FA_CANDIDATES} candidates")
            continue
        if drop is None:
            print(f"  {add_name} ← {drop_name}: drop not in top {_MAX_DROP_CANDIDATES} candidates")
            continue
        is_cross = fa["is_hitter"] != drop["is_hitter"]
        if is_cross:
            allowed = _allow_cross_type(ctx, fa, drop, target_cats)
            print(
                f"  {add_name} ← {drop_name}: CROSS-TYPE (fa_hitter={fa['is_hitter']}, drop_hitter={drop['is_hitter']}) → _allow_cross_type={allowed}"
            )
        else:
            print(
                f"  {add_name} ← {drop_name}: SAME-TYPE (both is_hitter={fa['is_hitter']}) — would NOT call _allow_cross_type"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
