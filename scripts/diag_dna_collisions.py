"""DNA collision diagnostic.

Surveys the HEATER players table for "DNA collision" risks — players whose names
appear in multiple rows (true duplicates) AND user-roster players whose name
matches a different MLB player in the players table (single-row but ambiguous).

Background: "Max Muncy" maps to LAD veteran (mlb_id=571970) and ATH rookie
(mlb_id=691777). Other risks: "Luis Garcia" (3 active MLBers), "Will Smith" (2).

Usage:
    .venv\\Scripts\\python.exe scripts/diag_dna_collisions.py
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database import get_connection  # noqa: E402

# Force UTF-8 on Windows console
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "draft_tool.db"
USER_TEAM = "\U0001f3c6 Team Hickey"  # trophy + " Team Hickey"

# Names known to be ambiguous across MLB history (current + recent past).
# Even if HEATER only has one row, Yahoo might point to the OTHER player.
KNOWN_AMBIGUOUS_NAMES = {
    "Luis Garcia",  # 3 active MLBers (WSH 2B, HOU/SD RP, PHI RP)
    "Will Smith",  # LAD C + ATL RP (retired but still appears in some sources)
    "Max Muncy",  # LAD veteran + ATH rookie (already migrated PR #104)
    "Mike Yastrzemski",  # SF OF (only one currently active but name confusion risk)
    "Adrian Houser",  # multiple in minors history
    "Andrew Knapp",  # multiple in minors history
    "Tyler Wells",  # active + minors
    "Bryan Reynolds",  # PIT OF (only one but name pattern risk)
    "Brandon Lowe",  # TB 2B
    "Brandon Drury",  # active
    "Logan Allen",  # CLE LHP (multiple in minors)
    "Daniel Castano",  # multiple in system
    "Jose Ramirez",  # CLE 3B (also pitcher Jose Ramirez in minors)
    "Jose Rodriguez",  # multiple in minors
    "Jose Hernandez",  # multiple
    "Carlos Hernandez",  # KC RP + others
    "Carlos Rodriguez",  # multiple in minors
    "Cristian Gonzalez",  # multiple
    "Edward Cabrera",  # MIA RHP
    "Eduard Bazardo",
    "Francisco Alvarez",  # NYM C (others in minors)
    "Andres Munoz",  # SEA RP
    "Jorge Lopez",  # multiple
    "Jose Suarez",  # active + minors
    "Jonathan Aranda",
    "Hunter Brown",  # HOU RHP
    "Kevin Smith",  # multiple
    "Chris Martin",  # active reliever + multiple in history
    "Trevor Williams",  # WSH RHP
    "Tony Santillan",  # CIN RP
    "Wilmer Flores",  # SF UTIL
}


def main() -> None:
    if not DB_PATH.exists():
        print(f"ERROR: DB not found at {DB_PATH}")
        return

    conn = get_connection()
    cur = conn.cursor()

    print("=" * 70)
    print("STEP 1: Duplicate names in `players` table")
    print("=" * 70)

    cur.execute("""
        SELECT name, COUNT(*) AS cnt
        FROM players
        GROUP BY LOWER(TRIM(name))
        HAVING cnt > 1
        ORDER BY cnt DESC, name
    """)
    dup_groups = cur.fetchall()
    print(f"Found {len(dup_groups)} duplicate name groups.\n")

    duplicate_names_lower: set[str] = set()
    for grp in dup_groups:
        name = grp["name"]
        duplicate_names_lower.add(name.lower().strip())
        cur.execute(
            """
            SELECT player_id, name, team, mlb_id, positions, level, is_hitter
            FROM players
            WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
            ORDER BY player_id
        """,
            (name,),
        )
        rows = cur.fetchall()
        print(f"  [{grp['cnt']}x] {name}")
        for r in rows:
            level = r["level"] or "MLB?"
            hp = "H" if r["is_hitter"] == 1 else "P" if r["is_hitter"] == 0 else "?"
            print(
                f"      player_id={r['player_id']:>6}  team={r['team'] or '---':<5}  "
                f"mlb_id={r['mlb_id'] or 'NULL':<8}  pos={r['positions'] or '---':<10}  "
                f"lvl={level:<8} {hp}"
            )
        print()

    print("=" * 70)
    print(f"STEP 2: User-roster (Team: {USER_TEAM}) collision risks")
    print("=" * 70)

    cur.execute(
        """
        SELECT lr.player_id, lr.roster_slot, lr.status, lr.selected_position,
               lr.editorial_team_abbr, lr.yahoo_player_key,
               p.name, p.team AS players_team, p.mlb_id, p.positions
        FROM league_rosters lr
        LEFT JOIN players p ON p.player_id = lr.player_id
        WHERE lr.team_name = ?
        ORDER BY p.name
    """,
        (USER_TEAM,),
    )
    user_roster = cur.fetchall()
    print(f"User roster size: {len(user_roster)}\n")

    if len(user_roster) == 0:
        # Try without emoji in case of encoding issue
        cur.execute("SELECT DISTINCT team_name FROM league_rosters ORDER BY team_name")
        teams = [r["team_name"] for r in cur.fetchall()]
        print("No rows for that exact team_name. Available team_names:")
        for t in teams:
            print(f"   {t!r}")
        # Try fallback heuristic
        for t in teams:
            if "Hickey" in t:
                print(f"\nRetrying with detected user team: {t!r}\n")
                cur.execute(
                    """
                    SELECT lr.player_id, lr.roster_slot, lr.status, lr.selected_position,
                           lr.editorial_team_abbr, lr.yahoo_player_key,
                           p.name, p.team AS players_team, p.mlb_id, p.positions
                    FROM league_rosters lr
                    LEFT JOIN players p ON p.player_id = lr.player_id
                    WHERE lr.team_name = ?
                    ORDER BY p.name
                """,
                    (t,),
                )
                user_roster = cur.fetchall()
                print(f"User roster size: {len(user_roster)}\n")
                break

    collision_count = 0
    ambiguous_single_row: list[dict] = []
    mismatch_count = 0
    mismatches: list[tuple] = []

    for r in user_roster:
        name = r["name"]
        if not name:
            print(f"  WARN: player_id={r['player_id']} has no players-table row")
            continue

        # Check for additional rows with same name
        cur.execute(
            """
            SELECT player_id, name, team, mlb_id, positions, level
            FROM players
            WHERE LOWER(TRIM(name)) = LOWER(TRIM(?)) AND player_id != ?
            ORDER BY player_id
        """,
            (name, r["player_id"]),
        )
        other_rows = cur.fetchall()

        if other_rows:
            collision_count += 1
            print(f"  [COLLISION] {name}")
            print(
                f"      User's player_id={r['player_id']}  players.team={r['players_team']}  "
                f"mlb_id={r['mlb_id']}  editorial={r['editorial_team_abbr']}  "
                f"yahoo_key={r['yahoo_player_key']}"
            )
            for o in other_rows:
                print(
                    f"      ALT     player_id={o['player_id']}  team={o['team']}  "
                    f"mlb_id={o['mlb_id']}  pos={o['positions']}  level={o['level']}"
                )
            print()
        elif name in KNOWN_AMBIGUOUS_NAMES:
            ambiguous_single_row.append(
                {
                    "name": name,
                    "player_id": r["player_id"],
                    "players_team": r["players_team"],
                    "mlb_id": r["mlb_id"],
                    "editorial": r["editorial_team_abbr"],
                }
            )

        # Step 4: mlb_team (players.team) vs editorial_team_abbr mismatch
        pt = (r["players_team"] or "").upper().strip()
        ed = (r["editorial_team_abbr"] or "").upper().strip()
        # Allow MLB → "" team, IL stash with no current team, "MLB" placeholder
        if pt and ed and pt != ed:
            # Soft equivalents - handle 2-letter vs 3-letter codes
            soft_equiv = {
                ("WSH", "WSN"),
                ("WSN", "WSH"),
                ("CHW", "CWS"),
                ("CWS", "CHW"),
                ("KC", "KCR"),
                ("KCR", "KC"),
                ("TB", "TBR"),
                ("TBR", "TB"),
                ("SD", "SDP"),
                ("SDP", "SD"),
                ("SF", "SFG"),
                ("SFG", "SF"),
            }
            if (pt, ed) in soft_equiv:
                continue
            mismatch_count += 1
            mismatches.append((name, r["player_id"], pt, ed, r["mlb_id"]))

    print("=" * 70)
    print("STEP 3: Ambiguous-name single-row risk (Team Hickey only)")
    print("=" * 70)
    print(f"Flagged {len(ambiguous_single_row)} player(s) with common-name ambiguity risk:\n")
    for a in ambiguous_single_row:
        print(f"  [SINGLE-ROW-AMBIGUOUS] {a['name']}")
        print(
            f"      player_id={a['player_id']}  players.team={a['players_team']}  "
            f"mlb_id={a['mlb_id']}  editorial={a['editorial']}"
        )
    print()

    print("=" * 70)
    print("STEP 4: players.team vs league_rosters.editorial_team_abbr mismatches")
    print("=" * 70)
    print(f"Found {mismatch_count} mismatch(es).\n")
    for name, pid, pt, ed, mid in mismatches:
        print(f"  [MISMATCH] {name}  player_id={pid}  players.team={pt}  editorial_team_abbr={ed}  mlb_id={mid}")
    print()

    print("=" * 70)
    print("BONUS: Database-wide ambiguous-name scan (any row matching list)")
    print("=" * 70)
    bonus_rows: list[sqlite3.Row] = []
    for n in sorted(KNOWN_AMBIGUOUS_NAMES):
        cur.execute(
            """
            SELECT player_id, name, team, mlb_id, positions, level
            FROM players
            WHERE LOWER(TRIM(name)) = LOWER(TRIM(?))
            ORDER BY player_id
        """,
            (n,),
        )
        rows = cur.fetchall()
        if rows:
            for r in rows:
                bonus_rows.append(r)
    print(f"Players-table rows matching ambiguous-name watchlist: {len(bonus_rows)}\n")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Duplicate name groups in players:          {len(dup_groups)}")
    print(f"  User-roster collision risks (multi-row):   {collision_count}")
    print(f"  User-roster single-row ambiguity risks:    {len(ambiguous_single_row)}")
    print(f"  players.team vs editorial mismatches:      {mismatch_count}")

    conn.close()


if __name__ == "__main__":
    main()
