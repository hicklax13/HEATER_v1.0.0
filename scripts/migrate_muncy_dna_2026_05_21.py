"""One-time migration (2026-05-21): insert LAD Max Muncy as a NEW row in
`players`, update `league_rosters` to point Team Hickey at the new player_id,
and trigger a fresh season-stats fetch for the new player.

Background:
  The DB had exactly one "Max Muncy" row (player_id=71, team='ATH',
  mlb_id=691777 — the Oakland Athletics rookie). The user's Yahoo
  roster shows LAD Max Muncy (mlb_id=571970 — the veteran 3B). The
  legacy roster-sync code matched Yahoo's "Max Muncy LAD" to the
  ONLY "Max Muncy" in `players` (the ATH rookie) because the
  (name + team) precise match found nothing and the name-only
  fallback silently picked the wrong player.

  Result: every downstream FA recommender / lineup optimizer / war
  room / matchup planner query reasoned about the ATH rookie's
  struggling line (26 GP / 2 HR / .239 AVG) instead of the LAD
  veteran's strong line (47 GP / 12 HR / .263 AVG). PR22 (companion)
  prevents new instances of this bug. This script fixes the existing
  corruption for the local DB.

Idempotent: if LAD Muncy (mlb_id=571970) already exists in players,
the script is a no-op for the player insert and only updates
league_rosters if needed.

USAGE:
    PYTHONPATH=. python scripts/migrate_muncy_dna_2026_05_21.py

The script does NOT take any arguments. It writes only to the
local `data/draft_tool.db` via `get_connection()`.
"""

from __future__ import annotations

import io
import logging
import sys
from datetime import UTC, datetime

# Force UTF-8 on stdout/stderr so the team-name emoji renders cleanly.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("migrate_muncy_dna")

# Canonical LAD Max Muncy (veteran 3B) data — verified 2026-05-21 via MLB
# Stats API (statsapi.get('sports_players', {'sportId':1, 'season':2026})).
_LAD_MUNCY = {
    "name": "Max Muncy",
    "team": "LAD",
    "positions": "3B,2B,1B",  # LAD Muncy's primary eligibilities; Yahoo
    # carries 3B as selected_position, 2B/1B as backup
    "is_hitter": 1,
    "is_injured": 0,
    "mlb_id": 571970,
    "bats": "L",
    "throws": "R",
    "birth_date": "1990-08-25",
    "level": "MLB",
}

# The user's team_name as stored in league_rosters (with emoji).
_USER_TEAM = "\U0001f3c6 Team Hickey"  # 🏆 Team Hickey

# The wrong (ATH-Muncy) player_id currently rostered on Team Hickey.
_WRONG_PLAYER_ID = 71


def _find_existing_lad_muncy(conn) -> int | None:
    """Return player_id if a row matching LAD Muncy by mlb_id already
    exists, else None."""
    cur = conn.cursor()
    cur.execute(
        "SELECT player_id FROM players WHERE mlb_id = ?",
        (_LAD_MUNCY["mlb_id"],),
    )
    row = cur.fetchone()
    return row[0] if row else None


def _insert_lad_muncy(conn) -> int:
    """Insert LAD Muncy and return the new player_id."""
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO players
           (name, team, positions, is_hitter, is_injured, mlb_id,
            bats, throws, birth_date, level)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            _LAD_MUNCY["name"],
            _LAD_MUNCY["team"],
            _LAD_MUNCY["positions"],
            _LAD_MUNCY["is_hitter"],
            _LAD_MUNCY["is_injured"],
            _LAD_MUNCY["mlb_id"],
            _LAD_MUNCY["bats"],
            _LAD_MUNCY["throws"],
            _LAD_MUNCY["birth_date"],
            _LAD_MUNCY["level"],
        ),
    )
    new_pid = cur.lastrowid
    assert new_pid is not None, "INSERT did not return a lastrowid — DB error?"
    return int(new_pid)


def _update_user_roster(conn, old_pid: int, new_pid: int) -> int:
    """Point the user's "Max Muncy" roster slot at the new (LAD) player_id.

    Returns the number of rows updated.
    """
    cur = conn.cursor()
    cur.execute(
        "UPDATE league_rosters SET player_id = ? WHERE team_name = ? AND player_id = ?",
        (new_pid, _USER_TEAM, old_pid),
    )
    return cur.rowcount


def _audit_other_rosters_for_old_pid(conn, old_pid: int) -> list[dict]:
    """Return any non-user roster rows that point at the wrong (ATH) player_id.

    The user audited his own roster and found only Muncy has this DNA
    collision. But it's possible another team in the league has a player
    with a name that collides with the ATH-Muncy row too. Report it loudly
    so the operator can decide whether to follow up.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT team_name, player_id, editorial_team_abbr, selected_position, yahoo_player_key "
        "FROM league_rosters WHERE player_id = ? AND team_name != ?",
        (old_pid, _USER_TEAM),
    )
    return [
        {
            "team_name": r[0],
            "player_id": r[1],
            "editorial_team_abbr": r[2],
            "selected_position": r[3],
            "yahoo_player_key": r[4],
        }
        for r in cur.fetchall()
    ]


def _refresh_season_stats_for_player(new_pid: int, mlb_id: int) -> int:
    """Fetch 2026 season stats for the new player_id and write to season_stats.

    Returns the number of rows written (0 or 1).
    """
    try:
        import statsapi
    except ImportError:
        logger.warning(
            "MLB-StatsAPI not installed — skipping season_stats refresh. Next bootstrap cycle will populate the row."
        )
        return 0

    try:
        from src.live_stats import _parse_hitting_stat, save_season_stats_to_db
    except ImportError as e:
        logger.warning("Could not import live_stats helpers: %s", e)
        return 0

    season = 2026
    hydrate = f"stats(type=season,season={season},gameType=R)"
    try:
        data = statsapi.get(
            "people",
            {"personIds": mlb_id, "hydrate": hydrate},
            request_kwargs={"timeout": 30},
        )
    except Exception as e:
        logger.warning("MLB Stats API fetch failed for mlb_id=%d: %s", mlb_id, e)
        return 0

    people = data.get("people", [])
    if not people:
        logger.warning("MLB Stats API returned no person for mlb_id=%d", mlb_id)
        return 0

    person = people[0]
    full_name = person.get("fullName", _LAD_MUNCY["name"])
    stats_list = person.get("stats", [])

    # Find hitting splits.
    stat_dict = {}
    for stat_group in stats_list:
        if stat_group.get("group", {}).get("displayName") == "hitting":
            splits = stat_group.get("splits", [])
            if splits:
                stat_dict = splits[0].get("stat", {})
                break

    if not stat_dict:
        logger.warning(
            "MLB Stats API returned no hitting stats for %s (mlb_id=%d). Bootstrap will fill on next cycle.",
            full_name,
            mlb_id,
        )
        return 0

    import pandas as pd

    row = _parse_hitting_stat(
        {"fullName": full_name, "team_abbr": _LAD_MUNCY["team"], "mlb_id": mlb_id},
        stat_dict,
    )
    stats_df = pd.DataFrame([row])
    saved = save_season_stats_to_db(stats_df, season=season)
    return int(saved)


def main() -> int:
    from src.database import get_connection, init_db

    init_db()
    print("=" * 78)
    print("Muncy DNA Migration (2026-05-21)")
    print("=" * 78)
    print(f"Run time: {datetime.now(UTC).isoformat()}")
    print()

    conn = get_connection()
    try:
        # 1. Guard against double-execution.
        existing_pid = _find_existing_lad_muncy(conn)
        if existing_pid is not None:
            print(
                f"  LAD Max Muncy (mlb_id={_LAD_MUNCY['mlb_id']}) already exists in "
                f"players at player_id={existing_pid}."
            )
            print("  Skipping insert; will only patch league_rosters if needed.")
            new_pid = existing_pid
            inserted = False
        else:
            new_pid = _insert_lad_muncy(conn)
            inserted = True
            print(f"  Inserted LAD Max Muncy as player_id={new_pid} (mlb_id={_LAD_MUNCY['mlb_id']}).")

        # 2. Audit other-team rosters before mutating Team Hickey.
        collisions = _audit_other_rosters_for_old_pid(conn, _WRONG_PLAYER_ID)
        if collisions:
            print()
            print(
                f"  WARNING: {len(collisions)} other roster row(s) point at the "
                f"(presumed-ATH) player_id={_WRONG_PLAYER_ID}:"
            )
            for c in collisions:
                print(f"    - {c}")
            print(
                "  Each of these MAY also be a DNA collision (Yahoo says LAD but "
                "stored as ATH-rookie). Investigate per-team; this migration "
                "only fixes Team Hickey."
            )

        # 3. Patch Team Hickey's roster row.
        if new_pid == _WRONG_PLAYER_ID:
            print(
                f"  ERROR: existing player_id={new_pid} for LAD Muncy already "
                f"matches the wrong (ATH) player_id={_WRONG_PLAYER_ID}. "
                "Manual cleanup required."
            )
            return 2

        rowcount = _update_user_roster(conn, _WRONG_PLAYER_ID, new_pid)
        print()
        if rowcount > 0:
            print(
                f"  Updated league_rosters: {rowcount} row(s) on team '{_USER_TEAM}' "
                f"now point at player_id={new_pid} (was {_WRONG_PLAYER_ID})."
            )
        else:
            print(
                f"  No league_rosters row found on team '{_USER_TEAM}' with the "
                f"old player_id={_WRONG_PLAYER_ID}. Likely already migrated."
            )

        conn.commit()
        print()
        print(f"  Old (ATH-rookie) player_id: {_WRONG_PLAYER_ID}  →  New (LAD-veteran) player_id: {new_pid}")
        if inserted:
            print("  DB write: 1 INSERT into players, 1 UPDATE on league_rosters.")
        else:
            print(f"  DB write: 0 INSERT (idempotent), {rowcount} UPDATE on league_rosters.")
    finally:
        conn.close()

    # 4. Trigger a season_stats refresh for the new player_id.
    print()
    print("  Fetching 2026 season stats for the new player_id ...")
    saved = _refresh_season_stats_for_player(new_pid, _LAD_MUNCY["mlb_id"])
    if saved > 0:
        print(f"  Wrote {saved} season_stats row(s) for player_id={new_pid}.")
    else:
        print(
            "  Season-stats fetch failed or returned 0 rows. The next "
            "bootstrap cycle will fill this in via the bulk season_stats phase."
        )

    print()
    print("=" * 78)
    print("Migration complete.")
    print("=" * 78)
    print()
    print("Verify with:")
    print("  PYTHONPATH=. python scripts/diag_roster_data_audit.py")
    print()
    print("Expected after migration:")
    print("  Muncy row shows DB team=LAD (was ATH), no DNA flag.")
    print("  YTD GP should be near 47 (LAD veteran's actual), not 26 (ATH rookie).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
