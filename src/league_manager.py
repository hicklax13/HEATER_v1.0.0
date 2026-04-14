"""League management: roster imports, standings, free agents."""

import logging
import unicodedata

import pandas as pd

from src.database import (
    clear_league_rosters,
    get_connection,
    load_league_rosters,
    upsert_league_roster_entry,
    upsert_league_standing,
)

logger = logging.getLogger(__name__)


def import_league_rosters_csv(csv_path: str, user_team_name: str) -> int:
    """Import all teams' rosters from CSV.

    Expected columns: team_name, player_name, position, roster_slot
    Returns number of roster entries imported.
    """
    clear_league_rosters()
    conn = get_connection()
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        imported = 0

        teams = sorted(df["team_name"].unique())
        team_index_map = {}
        idx = 1
        user_found = False
        for t in teams:
            if t.lower() == user_team_name.lower():
                team_index_map[t] = 0
                user_found = True
            else:
                team_index_map[t] = idx
                idx += 1
        if not user_found:
            logger.warning("User team '%s' not found in CSV teams", user_team_name)

        for _, row in df.iterrows():
            team = str(row["team_name"]).strip()
            player_name = str(row["player_name"]).strip()
            roster_slot = str(row.get("roster_slot", "")).strip() if "roster_slot" in row.index else None

            cursor = conn.cursor()
            cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
            result = cursor.fetchone()
            if not result:
                parts = player_name.split()
                if len(parts) >= 2:
                    cursor.execute(
                        "SELECT player_id FROM players WHERE name LIKE ? AND name LIKE ?",
                        (f"%{parts[0]}%", f"%{parts[-1]}%"),
                    )
                    result = cursor.fetchone()
                elif len(parts) == 1:
                    # Single-name player (e.g., "Ohtani"): match on last name
                    cursor.execute(
                        "SELECT player_id FROM players WHERE name LIKE ?",
                        (f"% {parts[0]}",),
                    )
                    results = cursor.fetchall()
                    if len(results) == 1:
                        result = results[0]
            if not result:
                continue

            player_id = result[0]
            is_user = team.lower() == user_team_name.lower()
            team_idx = team_index_map.get(team, 0)

            # Inline insert instead of calling upsert_league_roster_entry to avoid double connection
            conn.execute(
                """INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team)
                   VALUES (?, ?, ?, ?, ?)""",
                (team, team_idx, player_id, roster_slot, 1 if is_user else 0),
            )
            imported += 1

        conn.commit()
    finally:
        conn.close()
    return imported


def import_standings_csv(csv_path: str) -> int:
    """Import current standings from CSV.

    Expected columns: team_name, R, HR, RBI, SB, AVG, W, SV, K, ERA, WHIP
    Returns number of teams imported.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    categories = ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "L", "SV", "K", "ERA", "WHIP"]
    imported = 0

    for _, row in df.iterrows():
        team = str(row["team_name"]).strip()
        for cat in categories:
            if cat in row.index:
                val = float(str(row[cat]).replace(",", ""))
                upsert_league_standing(team, cat, val)
        imported += 1

    return imported


def get_team_roster(team_name: str) -> pd.DataFrame:
    """Return roster for a specific team with player info and stats.

    JOINs season_stats (actual in-season performance) when available,
    falling back to blended projections for stat columns.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """SELECT lr.*, p.name, p.team, p.positions, p.is_hitter, p.mlb_id,
                  COALESCE(NULLIF(ss.pa, 0), pr.pa, 0) AS pa,
                  COALESCE(NULLIF(ss.ab, 0), pr.ab, 0) AS ab,
                  COALESCE(NULLIF(ss.h, 0), pr.h, 0) AS h,
                  COALESCE(NULLIF(ss.r, 0), pr.r, 0) AS r,
                  COALESCE(NULLIF(ss.hr, 0), pr.hr, 0) AS hr,
                  COALESCE(NULLIF(ss.rbi, 0), pr.rbi, 0) AS rbi,
                  COALESCE(NULLIF(ss.sb, 0), pr.sb, 0) AS sb,
                  COALESCE(ss.avg, pr.avg, 0) AS avg,
                  COALESCE(ss.obp, pr.obp, 0) AS obp,
                  COALESCE(NULLIF(ss.bb, 0), pr.bb, 0) AS bb,
                  COALESCE(NULLIF(ss.hbp, 0), pr.hbp, 0) AS hbp,
                  COALESCE(NULLIF(ss.sf, 0), pr.sf, 0) AS sf,
                  COALESCE(ss.ip, pr.ip, 0) AS ip,
                  COALESCE(NULLIF(ss.w, 0), pr.w, 0) AS w,
                  COALESCE(NULLIF(ss.l, 0), pr.l, 0) AS l,
                  COALESCE(NULLIF(ss.sv, 0), pr.sv, 0) AS sv,
                  COALESCE(NULLIF(ss.k, 0), pr.k, 0) AS k,
                  COALESCE(ss.era, pr.era, 0) AS era,
                  COALESCE(ss.whip, pr.whip, 0) AS whip,
                  COALESCE(NULLIF(ss.er, 0), pr.er, 0) AS er,
                  COALESCE(NULLIF(ss.bb_allowed, 0), pr.bb_allowed, 0) AS bb_allowed,
                  COALESCE(NULLIF(ss.h_allowed, 0), pr.h_allowed, 0) AS h_allowed
               FROM league_rosters lr
               JOIN players p ON lr.player_id = p.player_id
               LEFT JOIN (
                   SELECT player_id, pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf,
                          ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed
                   FROM season_stats ss_inner
                   WHERE ss_inner.season = (SELECT MAX(s2.season) FROM season_stats s2 WHERE s2.player_id = ss_inner.player_id)
               ) ss ON lr.player_id = ss.player_id
               LEFT JOIN projections pr ON lr.player_id = pr.player_id
                   AND pr.system = 'blended'
               WHERE lr.team_name = ?
               GROUP BY lr.player_id""",
            conn,
            params=(team_name,),
        )
    finally:
        conn.close()

    # Coerce numeric columns (Python 3.13+ SQLite may return bytes)
    int_cols = [
        "pa",
        "ab",
        "h",
        "r",
        "hr",
        "rbi",
        "sb",
        "bb",
        "hbp",
        "sf",
        "w",
        "l",
        "sv",
        "k",
        "er",
        "bb_allowed",
        "h_allowed",
    ]
    float_cols = ["avg", "obp", "ip", "era", "whip"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def _normalize_name(name: str) -> str:
    """Normalize player name for matching: strip accents, lowercase, strip whitespace."""
    if not name or not isinstance(name, str):
        return ""
    # Strip accents (NFD decompose then encode to ASCII)
    normalized = unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode("ascii")
    # Lowercase, strip extra whitespace, remove Jr./Sr./III/II suffixes
    normalized = normalized.lower().strip()
    for suffix in [" jr.", " sr.", " jr", " sr", " iii", " ii", " iv"]:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)].strip()
    return normalized


def get_free_agents(player_pool: pd.DataFrame) -> pd.DataFrame:
    """Return all players NOT on any league_rosters team.

    Uses both player_id matching AND name matching to handle cases where
    league_rosters has different IDs than player_pool (Yahoo vs MLB Stats API).
    """
    rostered = load_league_rosters()
    if rostered.empty:
        return player_pool

    # Primary: filter by player_id
    rostered_ids = set(rostered["player_id"].values)

    # Secondary: also collect rostered player names for defense-in-depth
    # This handles the case where league_rosters uses Yahoo IDs but
    # player_pool uses MLB Stats API IDs (before deduplication runs)
    rostered_names = set()
    if "player_id" in rostered.columns:
        conn = get_connection()
        try:
            for pid in rostered_ids:
                row = conn.execute("SELECT name FROM players WHERE player_id = ?", (int(pid),)).fetchone()
                if row and row[0]:
                    rostered_names.add(row[0])
        finally:
            conn.close()

    # Build normalized name set for fuzzy matching (accents, Jr./Sr. suffixes)
    rostered_names_normalized = {_normalize_name(n) for n in rostered_names if n}

    # Exclude players matched by EITHER id or name (exact or normalized)
    name_col = (
        "name" if "name" in player_pool.columns else ("player_name" if "player_name" in player_pool.columns else None)
    )
    if name_col and rostered_names:
        pool_names_normalized = player_pool[name_col].apply(_normalize_name)
        mask = (
            player_pool["player_id"].isin(rostered_ids)
            | player_pool[name_col].isin(rostered_names)
            | pool_names_normalized.isin(rostered_names_normalized)
        )
    else:
        mask = player_pool["player_id"].isin(rostered_ids)

    return player_pool[~mask].copy()


def add_player_to_roster(
    team_name: str,
    team_index: int,
    player_id: int,
    roster_slot: str = None,
    is_user_team: bool = False,
):
    """Add a player to a team roster."""
    upsert_league_roster_entry(team_name, team_index, player_id, roster_slot, is_user_team)


def remove_player_from_roster(team_name: str, player_id: int):
    """Remove a player from a team roster."""
    conn = get_connection()
    try:
        conn.execute(
            "DELETE FROM league_rosters WHERE team_name = ? AND player_id = ?",
            (team_name, player_id),
        )
        conn.commit()
    finally:
        conn.close()
