"""Find if ANY roster combination from all players ever owned could beat opponents."""

import io
import sqlite3
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
conn = sqlite3.connect("data/draft_tool.db")

scoring_cats = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
inverse_cats = {"L", "ERA", "WHIP"}

# 1. Get EVERY player that was ever on Team Hickey's roster
# Original draft picks (17 still + 6 dropped = 23)
# Plus all 24 acquired players = up to 47 unique players

all_hickey_pids = set()

# Current roster
for row in conn.execute("SELECT player_id FROM league_rosters WHERE team_name LIKE '%Hickey%'"):
    all_hickey_pids.add(row[0])

# All transaction players (added or dropped)
for row in conn.execute(
    "SELECT DISTINCT player_id FROM transactions WHERE team_from LIKE '%Hickey%' OR team_to LIKE '%Hickey%'"
):
    all_hickey_pids.add(row[0])

print(f"Total unique players ever on Team Hickey: {len(all_hickey_pids)}")

# 2. Get season stats for all these players
players = []
for pid in all_hickey_pids:
    row = conn.execute(
        """SELECT p.player_id, p.name, p.positions, p.is_hitter,
                  ss.r, ss.hr, ss.rbi, ss.sb, ss.avg, ss.obp,
                  ss.w, ss.l, ss.sv, ss.k, ss.era, ss.whip, ss.ip,
                  ss.pa, ss.ab, ss.h, ss.bb, ss.er, ss.h_allowed, ss.bb_allowed
           FROM players p
           LEFT JOIN season_stats ss ON p.player_id = ss.player_id
           WHERE p.player_id = ?
           ORDER BY ss.season DESC LIMIT 1""",
        (pid,),
    ).fetchone()
    if row:
        is_sp = "SP" in str(row[2] or "").upper()
        is_rp = "RP" in str(row[2] or "").upper()
        players.append(
            {
                "pid": row[0],
                "name": row[1],
                "pos": row[2] or "",
                "is_hitter": bool(row[3]),
                "R": row[4] or 0,
                "HR": row[5] or 0,
                "RBI": row[6] or 0,
                "SB": row[7] or 0,
                "AVG": row[8] or 0,
                "OBP": row[9] or 0,
                "W": row[10] or 0,
                "L": row[11] or 0,
                "SV": row[12] or 0,
                "K": row[13] or 0,
                "ERA": row[14] or 0,
                "WHIP": row[15] or 0,
                "IP": row[16] or 0,
                "PA": row[17] or 0,
                "AB": row[18] or 0,
                "H": row[19] or 0,
                "BB": row[20] or 0,
                "ER": row[21] or 0,
                "H_allowed": row[22] or 0,
                "BB_allowed": row[23] or 0,
                "is_sp": is_sp,
                "is_rp": is_rp,
            }
        )

hitters = [p for p in players if p["is_hitter"] and p["PA"] > 0]
pitchers = [p for p in players if not p["is_hitter"] and p["IP"] > 0]

print(f"Hitters with stats: {len(hitters)}")
print(f"Pitchers with stats: {len(pitchers)}")

# Print all available players
print("\n=== ALL HITTERS EVER OWNED ===")
hitters.sort(key=lambda x: x["HR"], reverse=True)
for h in hitters:
    print(
        f"  {h['name']:<25} {h['pos']:<15} PA:{h['PA']:>3}  R:{h['R']:>3}  HR:{h['HR']:>3}  RBI:{h['RBI']:>3}  SB:{h['SB']:>3}  AVG:{h['AVG']:.3f}  OBP:{h['OBP']:.3f}"
    )

print("\n=== ALL PITCHERS EVER OWNED ===")
pitchers.sort(key=lambda x: x["IP"], reverse=True)
for p in pitchers:
    print(
        f"  {p['name']:<25} {p['pos']:<15} IP:{p['IP']:>5.1f}  W:{p['W']:>2}  L:{p['L']:>2}  SV:{p['SV']:>2}  K:{p['K']:>3}  ERA:{p['ERA']:.2f}  WHIP:{p['WHIP']:.3f}"
    )


# 3. Opponents' season totals (as proxy for weekly — early season)
opponents = [
    (
        1,
        "The Good The Vlad The Ugly",
        {
            "R": 100,
            "HR": 23,
            "RBI": 93,
            "SB": 9,
            "AVG": 0.267,
            "OBP": 0.368,
            "W": 8,
            "L": 10,
            "SV": 3,
            "K": 125,
            "ERA": 3.99,
            "WHIP": 1.24,
        },
    ),
    (
        2,
        "Baty Babies",
        {
            "R": 88,
            "HR": 28,
            "RBI": 96,
            "SB": 8,
            "AVG": 0.257,
            "OBP": 0.358,
            "W": 12,
            "L": 8,
            "SV": 13,
            "K": 150,
            "ERA": 4.97,
            "WHIP": 1.43,
        },
    ),
    (
        3,
        "Jonny Jockstrap",
        {
            "R": 105,
            "HR": 18,
            "RBI": 88,
            "SB": 11,
            "AVG": 0.268,
            "OBP": 0.365,
            "W": 9,
            "L": 11,
            "SV": 2,
            "K": 161,
            "ERA": 4.02,
            "WHIP": 1.22,
        },
    ),
]


# 4. Find OPTIMAL roster from all available players
# Yahoo roster: C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN = 23
# Starting lineup: C, 1B, 2B, 3B, SS, OF, OF, OF, Util, Util = 10 hitters
#                  SP, SP, RP, RP, P, P, P, P = 8 pitchers
# We want to maximize category wins, so try: best hitters for counting + rate,
# best pitchers for W/K/SV while minimizing ERA/WHIP


def compute_team_stats(hitter_group, pitcher_group):
    """Compute aggregate team stats from selected hitters and pitchers."""
    stats = {}
    # Counting stats: sum
    for cat in ["R", "HR", "RBI", "SB"]:
        stats[cat] = sum(h[cat] for h in hitter_group)

    # AVG = total_H / total_AB
    total_ab = sum(h["AB"] for h in hitter_group)
    total_h = sum(h["H"] for h in hitter_group)
    stats["AVG"] = total_h / total_ab if total_ab > 0 else 0.250

    # OBP = (H + BB) / PA  (simplified)
    total_pa = sum(h["PA"] for h in hitter_group)
    total_on = sum(h["OBP"] * h["PA"] for h in hitter_group)  # weighted
    stats["OBP"] = total_on / total_pa if total_pa > 0 else 0.320

    # Pitching counting
    for cat in ["W", "SV", "K"]:
        stats[cat] = sum(p[cat] for p in pitcher_group)
    stats["L"] = sum(p["L"] for p in pitcher_group)

    # ERA = ER*9/IP, WHIP = (BB+H)/IP
    total_ip = sum(p["IP"] for p in pitcher_group)
    total_er = sum(p["ER"] for p in pitcher_group)
    total_bbh = sum(p["BB_allowed"] + p["H_allowed"] for p in pitcher_group)
    # Fallback: use weighted ERA/WHIP if component data is missing
    if total_ip > 0 and total_er > 0:
        stats["ERA"] = total_er * 9 / total_ip
    elif total_ip > 0:
        stats["ERA"] = sum(p["ERA"] * p["IP"] for p in pitcher_group) / total_ip
    else:
        stats["ERA"] = 4.50

    if total_ip > 0 and total_bbh > 0:
        stats["WHIP"] = total_bbh / total_ip
    elif total_ip > 0:
        stats["WHIP"] = sum(p["WHIP"] * p["IP"] for p in pitcher_group) / total_ip
    else:
        stats["WHIP"] = 1.30

    return stats


def count_cat_wins(my_stats, opp_stats):
    """Count category wins/losses."""
    wins, losses = 0, 0
    details = {}
    for cat in scoring_cats:
        my = my_stats[cat]
        opp = opp_stats[cat]
        if cat in inverse_cats:
            if my < opp:
                wins += 1
                details[cat] = "W"
            elif my > opp:
                losses += 1
                details[cat] = "L"
            else:
                details[cat] = "T"
        else:
            if my > opp:
                wins += 1
                details[cat] = "W"
            elif my < opp:
                losses += 1
                details[cat] = "L"
            else:
                details[cat] = "T"
    return wins, losses, details


# Strategy: try different roster constructions
# Since brute force all combos of 10 hitters from ~20+ is too many,
# use a greedy approach targeting each opponent's weaknesses

print("\n" + "=" * 90)
print("OPTIMAL ROSTER ANALYSIS: Could ANY combination have won?")
print("=" * 90)

for week, opp_name, opp_stats in opponents:
    print(f"\n{'=' * 90}")
    print(f"WEEK {week}: vs {opp_name}")
    print(f"{'=' * 90}")

    best_result = {"wins": 0, "losses": 12, "hitters": [], "pitchers": [], "stats": {}}

    # Try multiple strategies
    strategies = [
        ("Max Counting", lambda h: h["R"] + h["HR"] + h["RBI"] + h["SB"], lambda p: p["W"] + p["K"] + p["SV"]),
        ("Max AVG/OBP", lambda h: h["AVG"] * 1000 + h["OBP"] * 500, lambda p: -p["ERA"] * 100 - p["WHIP"] * 100),
        ("Max HR/RBI", lambda h: h["HR"] * 3 + h["RBI"], lambda p: p["W"] * 5 + p["K"]),
        ("Max SB Focus", lambda h: h["SB"] * 10 + h["R"] + h["OBP"] * 100, lambda p: p["SV"] * 5 + p["K"]),
        (
            "Balanced",
            lambda h: h["R"] + h["HR"] * 2 + h["RBI"] + h["SB"] * 3 + h["AVG"] * 200 + h["OBP"] * 200,
            lambda p: p["W"] * 3 + p["K"] + p["SV"] * 5 - p["ERA"] * 10 - p["WHIP"] * 20,
        ),
        (
            "Min ERA/WHIP",
            lambda h: h["AVG"] * 500 + h["OBP"] * 500,
            lambda p: -(p["ERA"] + p["WHIP"]) if p["IP"] > 5 else -99,
        ),
        ("Max W+SV", lambda h: h["R"] + h["HR"] + h["RBI"], lambda p: p["W"] * 5 + p["SV"] * 5 + p["K"]),
        (
            "Elite Pitching Only",
            lambda h: h["R"] + h["HR"] + h["RBI"] + h["SB"] + h["AVG"] * 200,
            lambda p: -(p["ERA"] + p["WHIP"]) if p["IP"] > 5 else -99,
        ),
    ]

    for strat_name, h_key, p_key in strategies:
        # Pick top 10 hitters and top 8 pitchers by strategy
        sorted_h = sorted(hitters, key=h_key, reverse=True)[:10]
        sorted_p = sorted(pitchers, key=p_key, reverse=True)[:8]

        if not sorted_h or not sorted_p:
            continue

        team_stats = compute_team_stats(sorted_h, sorted_p)
        wins, losses, details = count_cat_wins(team_stats, opp_stats)

        if wins > best_result["wins"] or (wins == best_result["wins"] and losses < best_result["losses"]):
            best_result = {
                "wins": wins,
                "losses": losses,
                "strategy": strat_name,
                "hitters": sorted_h,
                "pitchers": sorted_p,
                "stats": team_stats,
                "details": details,
            }

    # Also try: for each combo of pitchers (since there are fewer), find best hitters
    # Try top 5 pitcher combos by different sorts
    for sort_name, p_sort_key in [
        ("low ERA", lambda p: p["ERA"] if p["IP"] > 5 else 99),
        ("high K", lambda p: -p["K"]),
        ("high W", lambda p: -p["W"]),
        ("high SV", lambda p: -p["SV"]),
    ]:
        sorted_p = sorted(pitchers, key=p_sort_key)[:8]
        # For each pitcher group, try multiple hitter sorts
        for h_sort_name, h_sort_key in [
            ("R+HR+RBI", lambda h: -(h["R"] + h["HR"] + h["RBI"])),
            ("AVG+OBP", lambda h: -(h["AVG"] + h["OBP"])),
            ("balanced", lambda h: -(h["R"] + h["HR"] * 2 + h["RBI"] + h["SB"] * 3 + h["AVG"] * 300)),
        ]:
            sorted_h = sorted(hitters, key=h_sort_key)[:10]
            team_stats = compute_team_stats(sorted_h, sorted_p)
            wins, losses, details = count_cat_wins(team_stats, opp_stats)
            if wins > best_result["wins"] or (wins == best_result["wins"] and losses < best_result["losses"]):
                best_result = {
                    "wins": wins,
                    "losses": losses,
                    "strategy": f"P:{sort_name} + H:{h_sort_name}",
                    "hitters": sorted_h,
                    "pitchers": sorted_p,
                    "stats": team_stats,
                    "details": details,
                }

    # Print best result
    result = (
        "WIN"
        if best_result["wins"] > best_result["losses"]
        else ("TIE" if best_result["wins"] == best_result["losses"] else "LOSS")
    )
    print(
        f"\nBEST POSSIBLE: {best_result['wins']}-{best_result['losses']}-{12 - best_result['wins'] - best_result['losses']} = {result}"
    )
    print(f"Strategy: {best_result.get('strategy', '?')}")

    print(f"\n{'Cat':<6} {'Best Team':>10} {'Opponent':>10} {'Result':>7}")
    print("-" * 40)
    for cat in scoring_cats:
        my = best_result["stats"].get(cat, 0)
        opp = opp_stats[cat]
        d = best_result["details"].get(cat, "?")
        fmt = ".3f" if cat in ("AVG", "OBP", "ERA", "WHIP") else ".1f"
        print(f"{cat:<6} {my:>10{fmt}} {opp:>10{fmt}} {d:>7}")

    print("\nOptimal hitters:")
    for h in best_result["hitters"]:
        print(f"  {h['name']:<25} R:{h['R']:>3} HR:{h['HR']:>3} RBI:{h['RBI']:>3} SB:{h['SB']:>3} AVG:{h['AVG']:.3f}")
    print("Optimal pitchers:")
    for p in best_result["pitchers"]:
        print(
            f"  {p['name']:<25} W:{p['W']:>2} L:{p['L']:>2} SV:{p['SV']:>2} K:{p['K']:>3} ERA:{p['ERA']:.2f} WHIP:{p['WHIP']:.3f}"
        )

# Summary
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

conn.close()
