import sys

sys.path.insert(0, ".")

from src.database import get_connection  # noqa: E402

conn = get_connection()
cur = conn.cursor()

cur.execute("SELECT player_id, name, team, mlb_id FROM players WHERE name LIKE '%Muncy%'")
for r in cur.fetchall():
    print(f"players: id={r[0]}, name={r[1]}, team={r[2]}, mlb_id={r[3]}")

cur.execute("""
    SELECT lr.player_id, lr.yahoo_player_key, p.name, p.team
    FROM league_rosters lr JOIN players p ON lr.player_id = p.player_id
    WHERE p.name LIKE '%Muncy%'
""")
for r in cur.fetchall():
    print(f"league_rosters: player_id={r[0]}, yahoo_key={r[1]}, name={r[2]}, mlb_team={r[3]}")

cur.execute("""
    SELECT player_id, season, r, hr, rbi, sb, avg, obp, games_played
    FROM season_stats WHERE player_id IN (71, 9864)
    ORDER BY player_id, season DESC
""")
for r in cur.fetchall():
    avg = float(r[6]) if r[6] is not None else 0
    obp = float(r[7]) if r[7] is not None else 0
    print(
        f"season_stats: player_id={r[0]}, season={r[1]}, R={r[2]}, HR={r[3]}, RBI={r[4]}, SB={r[5]}, AVG={avg:.3f}, OBP={obp:.3f}, GP={r[8]}"
    )

conn.close()
