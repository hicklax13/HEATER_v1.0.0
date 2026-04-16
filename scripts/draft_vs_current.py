"""Compare draft roster vs current roster matchup results."""

import io
import sqlite3
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
conn = sqlite3.connect("data/draft_tool.db")

scoring_cats = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
inverse_cats = {"L", "ERA", "WHIP"}

# Current season totals
my_totals = {}
for cat in scoring_cats:
    row = conn.execute(
        "SELECT total FROM league_standings WHERE team_name LIKE '%Hickey%' AND category = ?",
        (cat,),
    ).fetchone()
    my_totals[cat] = row[0] if row else 0

# Dropped originals season stats
dropped = {
    "Corey Seager": {
        "R": 10,
        "HR": 4,
        "RBI": 10,
        "SB": 1,
        "AB": 58,
        "H": 12,
        "PA": 70,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Marcell Ozuna": {
        "R": 3,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "AB": 43,
        "H": 3,
        "PA": 48,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Marcus Semien": {
        "R": 2,
        "HR": 1,
        "RBI": 6,
        "SB": 0,
        "AB": 61,
        "H": 12,
        "PA": 67,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Mark Vientos": {
        "R": 5,
        "HR": 1,
        "RBI": 5,
        "SB": 0,
        "AB": 41,
        "H": 10,
        "PA": 44,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Tyler Soderstrom": {
        "R": 7,
        "HR": 2,
        "RBI": 13,
        "SB": 1,
        "AB": 62,
        "H": 13,
        "PA": 68,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Ernie Clement": {
        "R": 5,
        "HR": 0,
        "RBI": 4,
        "SB": 1,
        "AB": 61,
        "H": 19,
        "PA": 62,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
}

# Acquired replacements season stats
acquired = {
    "Matt Olson": {
        "R": 14,
        "HR": 4,
        "RBI": 11,
        "SB": 0,
        "AB": 67,
        "H": 20,
        "PA": 76,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Dansby Swanson": {
        "R": 13,
        "HR": 3,
        "RBI": 9,
        "SB": 1,
        "AB": 52,
        "H": 9,
        "PA": 65,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Andres Gimenez": {
        "R": 5,
        "HR": 1,
        "RBI": 3,
        "SB": 2,
        "AB": 55,
        "H": 13,
        "PA": 60,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Matt McLain": {
        "R": 6,
        "HR": 0,
        "RBI": 3,
        "SB": 3,
        "AB": 60,
        "H": 13,
        "PA": 70,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Angel Martinez": {
        "R": 8,
        "HR": 2,
        "RBI": 7,
        "SB": 1,
        "AB": 42,
        "H": 13,
        "PA": 55,
        "IP": 0,
        "W": 0,
        "L": 0,
        "SV": 0,
        "K": 0,
        "ER": 0,
        "BB_H": 0,
    },
    "Eduardo Rodriguez": {
        "R": 0,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "AB": 0,
        "H": 0,
        "PA": 0,
        "IP": 18,
        "W": 1,
        "L": 0,
        "SV": 0,
        "K": 11,
        "ER": 1,
        "BB_H": 18,
    },
}

# Compute draft roster hypothetical totals
acq_c = {c: sum(d.get(c, 0) for d in acquired.values()) for c in ["R", "HR", "RBI", "SB", "W", "L", "SV", "K"]}
drop_c = {c: sum(d.get(c, 0) for d in dropped.values()) for c in ["R", "HR", "RBI", "SB", "W", "L", "SV", "K"]}

acq_ab = sum(d["AB"] for d in acquired.values())
acq_h = sum(d["H"] for d in acquired.values())
drop_ab = sum(d["AB"] for d in dropped.values())
drop_h = sum(d["H"] for d in dropped.values())
acq_ip = sum(d["IP"] for d in acquired.values())
acq_er = sum(d["ER"] for d in acquired.values())
acq_bbh = sum(d["BB_H"] for d in acquired.values())
drop_ip = sum(d["IP"] for d in dropped.values())
drop_er = sum(d["ER"] for d in dropped.values())
drop_bbh = sum(d["BB_H"] for d in dropped.values())

draft_totals = {}
for cat in scoring_cats:
    if cat in ["R", "HR", "RBI", "SB", "W", "L", "SV", "K"]:
        draft_totals[cat] = my_totals[cat] - acq_c.get(cat, 0) + drop_c.get(cat, 0)
    elif cat == "AVG":
        team_ab = 850
        team_h = my_totals["AVG"] * team_ab
        draft_totals["AVG"] = (team_h - acq_h + drop_h) / (team_ab - acq_ab + drop_ab)
    elif cat == "OBP":
        team_pa = 1000
        team_on = my_totals["OBP"] * team_pa
        acq_on = sum(d["PA"] * 0.34 for d in acquired.values())  # estimate
        drop_on = sum(d["PA"] * 0.27 for d in dropped.values())  # estimate from their OBPs
        draft_totals["OBP"] = (team_on - acq_on + drop_on) / (
            team_pa - sum(d["PA"] for d in acquired.values()) + sum(d["PA"] for d in dropped.values())
        )
    elif cat == "ERA":
        team_ip = 120
        team_er = my_totals["ERA"] * team_ip / 9
        draft_totals["ERA"] = (
            (team_er - acq_er + drop_er) * 9 / (team_ip - acq_ip + drop_ip) if (team_ip - acq_ip + drop_ip) > 0 else 4.0
        )
    elif cat == "WHIP":
        team_ip = 120
        team_bbh = my_totals["WHIP"] * team_ip
        draft_totals["WHIP"] = (
            (team_bbh - acq_bbh + drop_bbh) / (team_ip - acq_ip + drop_ip) if (team_ip - acq_ip + drop_ip) > 0 else 1.3
        )

# Print comparison
print("=" * 60)
print("DRAFT ROSTER vs CURRENT ROSTER — Season Totals")
print("=" * 60)
print(f"{'Cat':<6} {'Current':>10} {'If Drafted':>10} {'Delta':>10} {'Better?':>10}")
print("-" * 55)
for cat in scoring_cats:
    curr = my_totals.get(cat, 0)
    draft = draft_totals.get(cat, 0)
    delta = draft - curr
    if cat in inverse_cats:
        better = "DRAFT" if draft < curr else ("CURRENT" if curr < draft else "SAME")
    else:
        better = "DRAFT" if draft > curr else ("CURRENT" if curr > draft else "SAME")
    if cat in ("AVG", "OBP", "ERA", "WHIP"):
        print(f"{cat:<6} {curr:>10.3f} {draft:>10.3f} {delta:>+10.3f} {better:>10}")
    else:
        print(f"{cat:<6} {curr:>10.1f} {draft:>10.1f} {delta:>+10.1f} {better:>10}")

# Matchup simulations
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

print()
print("=" * 85)
print("MATCHUP-BY-MATCHUP: Would Draft Roster Have Won?")
print("=" * 85)

# Note: using season totals as proxy for weekly (we don't have per-week splits)
for week, opp_name, opp in opponents:
    print(f"\n--- Week {week}: vs {opp_name} ---")
    print(f"{'Cat':<6} {'Current':>10} {'Draft':>10} {'Opponent':>10} {'Now':>5} {'Draft':>5}")
    print("-" * 55)
    cw, cl, dw, dl = 0, 0, 0, 0
    for cat in scoring_cats:
        c = my_totals.get(cat, 0)
        d = draft_totals.get(cat, 0)
        o = opp[cat]
        if cat in inverse_cats:
            cr = "W" if c < o else ("L" if c > o else "T")
            dr = "W" if d < o else ("L" if d > o else "T")
        else:
            cr = "W" if c > o else ("L" if c < o else "T")
            dr = "W" if d > o else ("L" if d < o else "T")
        if cr == "W":
            cw += 1
        elif cr == "L":
            cl += 1
        if dr == "W":
            dw += 1
        elif dr == "L":
            dl += 1
        fmt = ".3f" if cat in ("AVG", "OBP", "ERA", "WHIP") else ".1f"
        changed = " <--" if cr != dr else ""
        print(f"{cat:<6} {c:>10{fmt}} {d:>10{fmt}} {o:>10{fmt}} {cr:>5} {dr:>5}{changed}")
    ct = 12 - cw - cl
    dt = 12 - dw - dl
    cr_str = "WIN" if cw > cl else ("LOSS" if cl > cw else "TIE")
    dr_str = "WIN" if dw > dl else ("LOSS" if dl > dw else "TIE")
    print(f"\n  Current: {cw}-{cl}-{ct} = {cr_str}")
    print(f"  Draft:   {dw}-{dl}-{dt} = {dr_str}")
    if cr_str != dr_str:
        print("  >>> DIFFERENT OUTCOME!")

conn.close()
