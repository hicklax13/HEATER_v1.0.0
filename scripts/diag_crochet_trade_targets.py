"""Crochet trade-target finder (run locally, against the live DB).

Answers: "Which league teams need a top-ranked SP, and what NON-pitcher,
can't-drop (draft rounds 1-3), top-100-ECR players could I target by
trading Garrett Crochet?"

Usage:
    python scripts/diag_crochet_trade_targets.py
    python scripts/diag_crochet_trade_targets.py --give "Garrett Crochet" --top-n 10 --ecr-cap 100

This is read-only. It uses the same engine the Trade Finder page uses, so
the output matches what HEATER shows in-app.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.database import get_connection, load_player_pool
from src.league_rules import weeks_remaining
from src.opponent_trade_analysis import compute_opponent_needs
from src.valuation import LeagueConfig

# Pitching categories an ace starter actually moves (SV is closer-driven).
SP_CATS = ["W", "K", "ERA", "WHIP"]
# A team "needs a top SP" if it is bottom-tier (rank >= WEAK_RANK) in this
# many or more of the SP-driven categories.
WEAK_RANK = 8
MIN_WEAK_SP_CATS = 2


def _pool_col(pool: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in pool.columns:
            return c
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--give", default="Garrett Crochet", help="Player you are trading away")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--ecr-cap", type=int, default=100, help="Only target players with ECR rank <= this")
    args = ap.parse_args()

    cfg = LeagueConfig()
    wr = weeks_remaining()
    pool = load_player_pool()

    ecr_col = _pool_col(pool, "ecr_rank", "rank", "ecr_consensus")
    pos_col = _pool_col(pool, "positions", "position")
    name_col = _pool_col(pool, "player_name", "name")
    if not all([ecr_col, pos_col, name_col]):
        raise SystemExit(f"Pool missing required columns. Have: {list(pool.columns)}")

    # ── Load league rosters (team -> player_ids) + undroppable flags ──
    conn = get_connection()
    try:
        rosters = pd.read_sql_query(
            "SELECT team_name, player_id, is_undroppable, status FROM league_rosters",
            conn,
        )
    finally:
        conn.close()

    if rosters.empty:
        raise SystemExit(
            "league_rosters is empty. Run the app once (Yahoo-connected) or "
            "bootstrap with force=True to populate draft results + undroppable flags."
        )

    league_rosters = {t: g["player_id"].tolist() for t, g in rosters.groupby("team_name")}
    all_team_totals = _team_totals(league_rosters, pool, cfg)

    # ── Step 1: which teams need a top-ranked SP? ──────────────────────
    print(f"\n{'=' * 70}\nTEAMS THAT NEED A TOP SP (weak in >= {MIN_WEAK_SP_CATS} of {SP_CATS})\n{'=' * 70}")
    needy_teams: list[tuple[str, list[str]]] = []
    for team in league_rosters:
        needs = compute_opponent_needs(team, all_team_totals, weeks_remaining=wr, config=cfg)
        weak = [c for c in SP_CATS if needs.get(c, {}).get("rank", 0) >= WEAK_RANK]
        if len(weak) >= MIN_WEAK_SP_CATS:
            needy_teams.append((team, weak))
    for team, weak in sorted(needy_teams, key=lambda x: -len(x[1])):
        print(f"  {team:28s} weak SP cats: {', '.join(weak)}")
    if not needy_teams:
        print("  (no team is broadly weak in SP categories right now)")

    # ── Step 2: undroppable, non-pitcher, top-ECR targets on those teams ─
    upid = set(rosters.loc[rosters["is_undroppable"] == 1, "player_id"])
    pid_to_team = dict(zip(rosters["player_id"], rosters["team_name"]))

    is_pitcher = pool[pos_col].fillna("").str.contains(r"\b(?:SP|RP|P)\b", case=False, regex=True)
    pool = pool.assign(_is_pitcher=is_pitcher)

    candidates = []
    needy_names = {t for t, _ in needy_teams}
    for _, p in pool.iterrows():
        pid = p["player_id"]
        if pid not in upid or p["_is_pitcher"]:
            continue
        ecr = pd.to_numeric(p[ecr_col], errors="coerce")
        if pd.isna(ecr) or ecr > args.ecr_cap:
            continue
        team = pid_to_team.get(pid, "?")
        candidates.append(
            {
                "player": p[name_col],
                "ecr": int(ecr),
                "pos": p[pos_col],
                "team": team,
                "team_needs_sp": team in needy_names,
            }
        )

    cand_df = pd.DataFrame(candidates)
    if cand_df.empty:
        raise SystemExit("No undroppable, non-pitcher, top-100 players found on any roster.")

    # Prioritize teams that need SP, then by ECR rank.
    cand_df = cand_df.sort_values(["team_needs_sp", "ecr"], ascending=[False, True])

    print(f"\n{'=' * 70}\nTOP {args.top_n} TARGETS for '{args.give}'")
    print(f"(can't-drop, non-pitcher, ECR <= {args.ecr_cap}; ★ = team needs SP)\n{'=' * 70}")
    print(f"{'#':>2}  {'Player':24s} {'ECR':>4} {'Pos':10s} {'Owner':24s}")
    for i, (_, r) in enumerate(cand_df.head(args.top_n).iterrows(), 1):
        star = "★" if r["team_needs_sp"] else " "
        print(f"{i:>2}{star} {r['player']:24s} {r['ecr']:>4} {r['pos']:10s} {r['team']:24s}")


def _team_totals(league_rosters, pool, cfg):
    try:
        from src.standings_utils import get_all_team_totals

        totals = get_all_team_totals(league_rosters=league_rosters, player_pool=pool)
        if totals:
            return totals
    except Exception:
        pass
    # Projection fallback if standings_utils caching path is unavailable.
    from src.in_season import _roster_category_totals

    return {t: _roster_category_totals(ids, pool) for t, ids in league_rosters.items()}


if __name__ == "__main__":
    main()
