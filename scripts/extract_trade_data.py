"""Extract authoritative HEATER data for every rostered player, partitioned by team.

Output: data/trade_analysis/<team>.json with all authoritative stats per player.
Fields NOT in HEATER (L7/L14/L30, N7P/N14P, LIVE, current rank) must be web-fetched.
"""

from __future__ import annotations

import json
import os
import sqlite3
import statistics
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

conn = sqlite3.connect("data/draft_tool.db")
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Step 1: rostered players per team
rosters: dict[str, list[int]] = {}
player_meta: dict[int, dict] = {}
for r in cur.execute(
    """
    SELECT lr.team_name, lr.player_id, lr.selected_position, lr.status,
           p.name, p.team AS mlb_team, p.positions, p.is_hitter
    FROM league_rosters lr LEFT JOIN players p ON lr.player_id = p.player_id
    """
).fetchall():
    pid = r["player_id"]
    rosters.setdefault(r["team_name"], []).append(pid)
    player_meta[pid] = {
        "pid": pid,
        "name": r["name"],
        "mlb_team": r["mlb_team"],
        "positions": r["positions"],
        "is_hitter": r["is_hitter"],
        "yahoo_slot": r["selected_position"],
        "roster_status": r["status"],
        "owner": r["team_name"],
    }

print(f"Teams: {len(rosters)} | Players: {len(player_meta)}")

all_pids = list(player_meta.keys())
ph = ",".join(["?"] * len(all_pids))

# Step 2: season_stats for 2023-2026
season_data: dict[int, dict[int, dict]] = defaultdict(dict)
for r in cur.execute(
    f"""
    SELECT player_id, season, pa, ab, h, r, hr, rbi, sb, avg, obp,
           ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed, games_played
    FROM season_stats WHERE player_id IN ({ph}) AND season IN (2023, 2024, 2025, 2026)
    """,
    all_pids,
).fetchall():
    season_data[r["player_id"]][r["season"]] = dict(r)

# Step 3: ros_projections
# Prefer ros_blended; fall back to ros_bayesian if no blended row
ros_data: dict[int, dict] = {}
for r in cur.execute(
    f"""
    SELECT player_id, system, pa, ab, r, hr, rbi, sb, avg, obp, ip, w, l, sv, k, era, whip
    FROM ros_projections WHERE player_id IN ({ph})
    """,
    all_pids,
).fetchall():
    pid = r["player_id"]
    # blended wins over bayesian
    existing = ros_data.get(pid)
    if existing is None or (r["system"] == "ros_blended" and existing.get("system") != "ros_blended"):
        ros_data[pid] = dict(r)

# Step 4: pre-season rank + ADP
ecr_data = {
    r["player_id"]: dict(r)
    for r in cur.execute(
        f"""
        SELECT player_id, consensus_rank, consensus_avg, rank_stddev, n_sources,
               espn_rank, yahoo_adp AS ecr_yahoo_adp, cbs_rank, fp_ecr
        FROM ecr_consensus WHERE player_id IN ({ph})
        """,
        all_pids,
    ).fetchall()
}

adp_data = {
    r["player_id"]: dict(r)
    for r in cur.execute(
        f"""
        SELECT player_id, yahoo_adp, fantasypros_adp, adp, nfbc_adp
        FROM adp WHERE player_id IN ({ph})
        """,
        all_pids,
    ).fetchall()
}

# Step 5: ownership trend (latest per player)
own_data = {
    r["player_id"]: dict(r)
    for r in cur.execute(
        f"""
        SELECT ot.player_id, ot.date AS latest, ot.percent_owned, ot.delta_7d
        FROM ownership_trends ot
        WHERE ot.player_id IN ({ph}) AND ot.date = (
            SELECT MAX(date) FROM ownership_trends WHERE player_id = ot.player_id
        )
        """,
        all_pids,
    ).fetchall()
}

# Step 6: latest injury news
inj_data = {
    r["player_id"]: dict(r)
    for r in cur.execute(
        f"""
        SELECT n1.player_id, n1.headline, n1.detail, n1.il_status,
               n1.injury_body_part, n1.published_at
        FROM player_news n1
        WHERE n1.player_id IN ({ph}) AND n1.news_type='injury'
        AND NOT EXISTS (
            SELECT 1 FROM player_news n2 WHERE n2.player_id=n1.player_id
            AND n2.news_type='injury' AND n2.published_at > n1.published_at
        )
        """,
        all_pids,
    ).fetchall()
}

# Step 7: Statcast 2026
sc_data = {
    r["player_id"]: dict(r)
    for r in cur.execute(
        f"""
        SELECT player_id, xwoba, barrel_pct, hard_hit_pct, sprint_speed,
               stuff_plus, location_plus, pitching_plus, whiff_pct, chase_rate,
               iso, babip, ld_pct
        FROM statcast_archive WHERE player_id IN ({ph}) AND season=2026
        """,
        all_pids,
    ).fetchall()
}

conn.close()


def compute_profile(pid: int) -> dict:
    meta = player_meta[pid]
    prof: dict = dict(meta)

    # Season totals (and per-game averages derived)
    for yr in (2023, 2024, 2025, 2026):
        raw = season_data.get(pid, {}).get(yr)
        if not raw:
            continue
        g = raw.get("games_played") or 0
        block = {
            "g": g,
            "pa": raw["pa"],
            "ab": raw["ab"],
            "r": raw["r"],
            "hr": raw["hr"],
            "rbi": raw["rbi"],
            "sb": raw["sb"],
            "avg": raw["avg"],
            "obp": raw["obp"],
            "ip": raw["ip"],
            "w": raw["w"],
            "l": raw["l"],
            "sv": raw["sv"],
            "k": raw["k"],
            "era": raw["era"],
            "whip": raw["whip"],
        }
        # Per-game averages for counting stats (hitters)
        if meta["is_hitter"] == 1 and g > 0:
            block["per_game"] = {s: round((block[s] or 0) / g, 3) for s in ("r", "hr", "rbi", "sb")}
        elif meta["is_hitter"] == 0 and (block["ip"] or 0) > 0:
            # Per-IP for pitchers (K/9, BB/9 approximation)
            ip = block["ip"]
            block["per_9"] = {
                "k_per_9": round((block["k"] or 0) * 9 / ip, 2),
            }
        prof[f"s{yr}"] = block

    # ROS projections
    if pid in ros_data:
        prof["ros"] = {
            "source": ros_data[pid].get("system"),
            **{
                k: ros_data[pid][k]
                for k in ("pa", "r", "hr", "rbi", "sb", "avg", "obp", "ip", "w", "l", "sv", "k", "era", "whip")
            },
        }

    # Pre-season rank
    if pid in ecr_data:
        e = ecr_data[pid]
        prof["rank_pre"] = {
            "consensus": e["consensus_rank"],
            "avg": e["consensus_avg"],
            "stddev": e["rank_stddev"],
            "n_sources": e["n_sources"],
            "espn": e["espn_rank"],
            "cbs": e["cbs_rank"],
            "fp_ecr": e["fp_ecr"],
        }

    # ADP
    if pid in adp_data:
        a = adp_data[pid]
        prof["adp"] = {
            "yahoo": a["yahoo_adp"],
            "fp": a["fantasypros_adp"],
            "nfbc": a["nfbc_adp"],
            "overall": a["adp"],
        }

    # Ownership
    if pid in own_data:
        o = own_data[pid]
        prof["own"] = {"date": o["latest"], "pct": o["percent_owned"], "d7": o["delta_7d"]}

    # Injury
    if pid in inj_data:
        i = inj_data[pid]
        prof["injury"] = {
            "il_status": i["il_status"],
            "body_part": i["injury_body_part"],
            "headline": i["headline"],
            "date": i["published_at"],
        }

    # Statcast
    if pid in sc_data:
        s = sc_data[pid]
        prof["statcast"] = {k: v for k, v in s.items() if v is not None and k != "player_id"}

    # 3-year std dev (year-to-year consistency)
    stds: dict = {}
    if meta["is_hitter"] == 1:
        stat_list = ("r", "hr", "rbi", "sb")
    else:
        stat_list = ("w", "k", "era", "whip")
    for stat in stat_list:
        vals = [
            season_data[pid][y][stat]
            for y in (2023, 2024, 2025)
            if y in season_data.get(pid, {}) and season_data[pid][y].get(stat) is not None
        ]
        if len(vals) >= 2:
            stds[stat] = round(statistics.stdev(vals), 2)
    if stds:
        prof["std_3y"] = stds

    return prof


all_profiles = {pid: compute_profile(pid) for pid in all_pids}

os.makedirs("data/trade_analysis", exist_ok=True)
for team, pids in rosters.items():
    safe = team.replace("🏆 ", "").replace(" ", "_")
    out = {"team_name": team, "n_players": len(pids), "players": [all_profiles[p] for p in pids]}
    with open(f"data/trade_analysis/{safe}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str, ensure_ascii=False)
    print(f"  {safe}.json: {len(pids)} players")

print("\nDone.")
