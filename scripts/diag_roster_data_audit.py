"""Comprehensive audit: for every rostered player + top FA candidate,
flag DNA collisions (Yahoo says one team, DB has another) and stale-data
red flags (YTD shows much less than expected mid-season).

Output is structured so the user can see at a glance which players the
engine is reasoning about correctly and which are corrupted.
"""

from __future__ import annotations

import io
import logging
import sys

from src.database import get_connection, init_db
from src.league_manager import get_team_roster
from src.optimizer.fa_recommender import (
    _score_drop_candidates,
    _score_fa_candidates,
)
from src.optimizer.shared_data_layer import build_optimizer_context
from src.validation.dynamic_context import compute_weeks_remaining
from src.valuation import LeagueConfig, SGPCalculator, compute_replacement_levels
from src.yahoo_data_service import get_yahoo_data_service

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.WARNING)


def audit_player(p_row, lr_row=None, ytd_gp=None, expected_gp=None, hitter=True):
    """Return list of red-flag strings for a player."""
    flags = []
    # DNA collision: editorial_team_abbr vs players.team
    if lr_row is not None:
        editorial = str(lr_row.get("editorial_team_abbr", "")).strip().upper()
        db_team = str(p_row.get("team", "")).strip().upper()
        if editorial and db_team and editorial != db_team:
            flags.append(f"DNA: Yahoo→{editorial} but DB→{db_team}")

    # Stale YTD for hitters
    if hitter and ytd_gp is not None and expected_gp is not None:
        if expected_gp >= 30 and ytd_gp == 0:
            flags.append(f"0 GP (expected ~{int(expected_gp)})")
        elif expected_gp >= 30 and ytd_gp < 0.3 * expected_gp:
            flags.append(f"LOW GP {ytd_gp}/~{int(expected_gp)}")

    return flags


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

    # Season progress for GP expectations
    days_elapsed = max(1, (26 - ctx.weeks_remaining) * 7)
    expected_hitter_gp = days_elapsed * 0.85
    print(f"Day {days_elapsed} of season. Expected GP for full-time starter: ~{int(expected_hitter_gp)}")
    print()

    pool = ctx.player_pool
    name_col = "player_name" if "player_name" in pool.columns else "name"

    # Pull league_rosters for user's team to compare editorial_team_abbr
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT player_id, editorial_team_abbr, selected_position, status, yahoo_player_key "
            "FROM league_rosters WHERE team_name = ?",
            (user_team_name,),
        )
        lr_by_pid = {r["player_id"]: dict(r) for r in cur.fetchall()}
    finally:
        conn.close()

    print("=" * 90)
    print("USER ROSTER AUDIT")
    print("=" * 90)
    print(f"{'Player':25} {'Pos':18} {'DB team':6} {'Yahoo':6} {'YTD GP':7} {'YTD HR':7} {'Flags'}")
    print("-" * 90)
    for pid in ctx.user_roster_ids:
        match = pool[pool["player_id"] == pid]
        if match.empty:
            print(f"  (pid={pid} not in pool)")
            continue
        p = match.iloc[0]
        name = str(p.get(name_col, "?"))
        pos = str(p.get("positions", "?"))[:18]
        db_team = str(p.get("team", "?"))[:6]
        is_hitter = bool(int(p.get("is_hitter", 1)))
        ytd_gp = int(p.get("ytd_gp", 0) or 0)
        ytd_hr = int(p.get("ytd_hr", 0) or 0) if is_hitter else 0
        ytd_ip_v = float(p.get("ytd_ip", 0) or 0) if not is_hitter else 0
        ytd_k_v = int(p.get("ytd_k", 0) or 0) if not is_hitter else 0
        lr = lr_by_pid.get(pid)
        editorial = str(lr.get("editorial_team_abbr", "?")) if lr else "?"
        flags = audit_player(p, lr, ytd_gp, expected_hitter_gp, is_hitter)
        if not is_hitter:
            # For pitchers, check IP signal
            if ytd_ip_v < 5 and days_elapsed >= 30:
                flags.append(f"LOW IP {ytd_ip_v:.1f}")
        flag_str = " | ".join(flags) if flags else "OK"
        if is_hitter:
            print(f"  {name:25} {pos:18} {db_team:6} {editorial:6} {ytd_gp:7} {ytd_hr:7} {flag_str}")
        else:
            print(f"  {name:25} {pos:18} {db_team:6} {editorial:6}  IP:{ytd_ip_v:5.1f} K:{ytd_k_v:5} {flag_str}")

    # Now audit top FA candidates the engine sees
    print()
    print("=" * 90)
    print("TOP FA CANDIDATES AUDIT (what engine is considering for adds)")
    print("=" * 90)
    repl = compute_replacement_levels(pool, ctx.config, SGPCalculator(ctx.config))
    fas = _score_fa_candidates(ctx, repl)
    drops = _score_drop_candidates(ctx, repl)

    print(f"{'Player':25} {'Pos':18} {'DB team':8} {'YTD GP':7} {'YTD IP':8} {'composite':10}")
    print("-" * 90)
    for f in fas:
        match = pool[pool["player_id"] == f["player_id"]]
        if match.empty:
            continue
        p = match.iloc[0]
        ytd_gp = int(p.get("ytd_gp", 0) or 0)
        ytd_ip_v = float(p.get("ytd_ip", 0) or 0)
        db_team = str(p.get("team", "?"))[:8]
        print(
            f"  {f['name']:25} {f['positions']:18} {db_team:8} {ytd_gp:7} {ytd_ip_v:8.1f} {f['composite_score']:10.2f}"
        )

    # And the drop candidates
    print()
    print("=" * 90)
    print("TOP DROP CANDIDATES AUDIT (what engine wants you to drop)")
    print("=" * 90)
    print(f"{'Player':25} {'Pos':18} {'DB team':8} {'YTD GP':7} {'YTD HR/IP':10} {'drop_cost':10}")
    print("-" * 90)
    for d in drops:
        match = pool[pool["player_id"] == d["player_id"]]
        if match.empty:
            continue
        p = match.iloc[0]
        is_hitter = bool(int(p.get("is_hitter", 1)))
        ytd_gp = int(p.get("ytd_gp", 0) or 0)
        signal = f"HR={int(p.get('ytd_hr', 0) or 0)}" if is_hitter else f"IP={float(p.get('ytd_ip', 0) or 0):.1f}"
        db_team = str(p.get("team", "?"))[:8]
        lr = lr_by_pid.get(d["player_id"])
        editorial = str(lr.get("editorial_team_abbr", "?")) if lr else "?"
        dna_flag = (
            f" ⚠ DNA: Yahoo={editorial}" if lr and editorial != "?" and editorial.upper() != db_team.upper() else ""
        )
        print(
            f"  {d['name']:25} {d['positions']:18} {db_team:8} {ytd_gp:7} {signal:10} {d['drop_cost']:10.2f}{dna_flag}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
