"""Diagnostic script: test Yahoo Fantasy API end-to-end sync.

Run: python debug_yahoo.py
Shows exactly what each layer returns.
"""

import json
import logging
import os
import sys

# Force UTF-8 output for Windows console
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(stream=open("debug_yahoo_log.txt", "w", encoding="utf-8"))],
)
# Also print to console at INFO level
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
logging.getLogger().addHandler(console)

logger = logging.getLogger("debug_yahoo")
sys.path.insert(0, os.path.dirname(__file__))

from src.yahoo_api import _AUTH_DIR, YahooFantasyClient, _rate_limit  # noqa: E402

# Load stored credentials
token_file = _AUTH_DIR / "yahoo_token.json"
if not token_file.exists():
    print(f"ERROR: No token file found at {token_file}")
    sys.exit(1)

token_data = json.loads(token_file.read_text())
consumer_key = token_data.get("consumer_key", os.environ.get("YAHOO_CLIENT_ID", ""))
consumer_secret = token_data.get("consumer_secret", os.environ.get("YAHOO_CLIENT_SECRET", ""))

print("=" * 60)
print("Yahoo Fantasy API End-to-End Diagnostic")
print("=" * 60)

# Create client and authenticate
client = YahooFantasyClient(league_id="109662", game_code="mlb", season=2026)
print("\n--- Step 1: Authenticating ---")
result = client.authenticate(consumer_key, consumer_secret, token_data=token_data)
print(f"Auth result: {result}")

if not result:
    print("FAILED: Authentication failed")
    sys.exit(1)

q = client._query
print(f"league_key: {q.league_key}")
print(f"game_id: {q.game_id}")

# Test OUR get_league_standings wrapper
print("\n--- Step 2: get_league_standings() (our wrapper) ---")
standings_df = client.get_league_standings()
print(f"DataFrame shape: {standings_df.shape}")
print(f"DataFrame empty: {standings_df.empty}")
if not standings_df.empty:
    print(f"Columns: {list(standings_df.columns)}")
    print(f"First 3 rows:\n{standings_df.head(3).to_string()}")
else:
    print("EMPTY DATAFRAME - investigating raw API call...")
    # Call yfpy directly to see what we get
    try:
        _rate_limit()
        raw_standings = q.get_league_standings()
        print(f"  raw type: {type(raw_standings)}")
        # Check what attributes exist
        print(f"  dir: {[a for a in dir(raw_standings) if not a.startswith('_')]}")
        teams_attr = getattr(raw_standings, "teams", None)
        print(f"  .teams type: {type(teams_attr)}")
        print(f"  .teams value: {teams_attr}")
        if teams_attr:
            print(f"  .teams len: {len(teams_attr)}")
            if teams_attr:
                first = teams_attr[0]
                print(f"  first team type: {type(first)}")
                print(f"  first team dir: {[a for a in dir(first) if not a.startswith('_')]}")
                team = getattr(first, "team", first)
                print(f"  team.name: {getattr(team, 'name', 'MISSING')}")
                print(f"  team.team_key: {getattr(team, 'team_key', 'MISSING')}")
                print(f"  team.rank: {getattr(team, 'rank', 'MISSING')}")
                print(f"  team.team_standings: {getattr(team, 'team_standings', 'MISSING')}")
                print(f"  team.team_stats: {getattr(team, 'team_stats', 'MISSING')}")
    except Exception as e:
        print(f"  Raw call error: {e}")
        import traceback

        traceback.print_exc()

# Test OUR get_all_rosters wrapper
print("\n--- Step 3: get_all_rosters() (our wrapper) ---")
rosters_df = client.get_all_rosters()
print(f"DataFrame shape: {rosters_df.shape}")
print(f"DataFrame empty: {rosters_df.empty}")
if not rosters_df.empty:
    print(f"Columns: {list(rosters_df.columns)}")
    print(f"First 5 rows:\n{rosters_df.head(5).to_string()}")
else:
    print("EMPTY DATAFRAME - investigating raw API call...")
    try:
        _rate_limit()
        teams = q.get_league_teams()
        print(f"  teams count: {len(teams) if teams else 0}")
        if teams:
            first_entry = teams[0]
            team = getattr(first_entry, "team", first_entry)
            tk = getattr(team, "team_key", "???")
            print(f"  first team_key: {tk}")
            team_id = str(tk).rsplit(".t.", 1)[-1] if ".t." in str(tk) else tk
            print(f"  team_id: {team_id}")
            _rate_limit()
            roster = q.get_team_roster_by_week(team_id)
            print(f"  roster type: {type(roster)}")
            print(f"  roster dir: {[a for a in dir(roster) if not a.startswith('_')]}")
            # Check if it's iterable
            print(f"  roster is list: {isinstance(roster, list)}")
            players_attr = getattr(roster, "players", None)
            print(f"  roster.players type: {type(players_attr)}")
            if players_attr:
                print(f"  roster.players len: {len(players_attr)}")
                first_p = players_attr[0]
                print(f"  first player entry type: {type(first_p)}")
                player = getattr(first_p, "player", first_p)
                name_obj = getattr(player, "name", None)
                print(f"  player.name type: {type(name_obj)}")
                full = getattr(name_obj, "full", name_obj) if name_obj else "???"
                print(f"  player name.full: {full}")
    except Exception as e:
        print(f"  Raw call error: {e}")
        import traceback

        traceback.print_exc()

# Test sync_to_db if we got data
print("\n--- Step 4: sync_to_db() ---")
sync_result = client.sync_to_db()
print(f"Sync result: {sync_result}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE - check debug_yahoo_log.txt for details")
print("=" * 60)
