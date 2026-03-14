"""Live stats data pipeline: MLB Stats API + pybaseball.

Fetches current season stats, ROS projections, and park factors
from free public data sources. No Yahoo dependency.
"""

import logging
from datetime import UTC, datetime

import pandas as pd

from src.database import (
    get_connection,
    get_refresh_status,
    update_refresh_log,
    upsert_season_stats,
)

logger = logging.getLogger(__name__)

try:
    import statsapi
except ImportError:
    statsapi = None


def match_player_id(player_name: str, team_abbr: str) -> int | None:
    """Match an external player name to our players table."""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
        result = cursor.fetchone()
        if result:
            return result[0]

        parts = player_name.replace(".", "").split()
        if len(parts) >= 2:
            last_name = parts[-1]
            cursor.execute(
                "SELECT player_id FROM players WHERE name LIKE ? AND team = ?",
                (f"%{last_name}%", team_abbr),
            )
            result = cursor.fetchone()
            if result:
                return result[0]

        if len(parts) >= 1:
            last_name = parts[-1]
            cursor.execute(
                "SELECT player_id FROM players WHERE name LIKE ?",
                (f"% {last_name}",),
            )
            results = cursor.fetchall()
            if len(results) == 1:
                return results[0][0]

        return None
    finally:
        conn.close()


def _parse_hitting_stat(player_info: dict, stat: dict) -> dict:
    """Parse a hitting stat split into a row dict."""
    return {
        "player_name": player_info.get("fullName", ""),
        "team": player_info.get("team_abbr", ""),
        "is_hitter": True,
        "pa": int(stat.get("plateAppearances", 0)),
        "ab": int(stat.get("atBats", 0)),
        "h": int(stat.get("hits", 0)),
        "r": int(stat.get("runs", 0)),
        "hr": int(stat.get("homeRuns", 0)),
        "rbi": int(stat.get("rbi", 0)),
        "sb": int(stat.get("stolenBases", 0)),
        "avg": float(stat.get("avg", "0") or 0),
        "games_played": int(stat.get("gamesPlayed", 0)),
        "ip": 0,
        "w": 0,
        "sv": 0,
        "k": 0,
        "era": 0,
        "whip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    }


def _parse_pitching_stat(player_info: dict, stat: dict) -> dict:
    """Parse a pitching stat split into a row dict."""
    return {
        "player_name": player_info.get("fullName", ""),
        "team": player_info.get("team_abbr", ""),
        "is_hitter": False,
        "pa": 0,
        "ab": 0,
        "h": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "avg": 0,
        "ip": float(stat.get("inningsPitched", "0") or 0),
        "w": int(stat.get("wins", 0)),
        "sv": int(stat.get("saves", 0)),
        "k": int(stat.get("strikeOuts", 0)),
        "era": float(stat.get("era", "0") or 0),
        "whip": float(stat.get("whip", "0") or 0),
        "er": int(stat.get("earnedRuns", 0)),
        "bb_allowed": int(stat.get("baseOnBalls", 0)),
        "h_allowed": int(stat.get("hits", 0)),
        "games_played": int(stat.get("gamesPlayed", 0)),
    }


def fetch_season_stats(season: int = 2026) -> pd.DataFrame:
    """Pull season stats for all MLB players via MLB Stats API.

    Uses team rosters with hydrated stats, which is the reliable way to
    get bulk player statistics from the MLB Stats API.
    """
    if statsapi is None:
        raise ImportError("MLB-StatsAPI is required. Install with: pip install MLB-StatsAPI")

    rows = []
    seen_players: set[str] = set()  # Avoid duplicates across teams

    try:
        # Get all MLB team IDs for the season
        teams_data = statsapi.get("teams", {"sportId": 1, "season": season})
        team_ids = [t["id"] for t in teams_data.get("teams", [])]
        logger.info("Fetching %d season stats from %d teams...", season, len(team_ids))
    except Exception as e:
        logger.warning("Failed to get MLB teams for %d: %s", season, e)
        return pd.DataFrame()

    hydrate = f"person(stats(type=season,season={season},gameType=R),currentTeam)"

    for team_id in team_ids:
        try:
            roster = statsapi.get(
                "team_roster",
                {
                    "teamId": team_id,
                    "season": season,
                    "rosterType": "fullSeason",
                    "hydrate": hydrate,
                },
            )
            for entry in roster.get("roster", []):
                person = entry.get("person", {})
                full_name = person.get("fullName", "")
                if not full_name or full_name in seen_players:
                    continue
                seen_players.add(full_name)

                current_team = person.get("currentTeam", {})
                team_abbr = current_team.get("abbreviation", "")
                player_info = {"fullName": full_name, "team_abbr": team_abbr}

                # Determine position type from roster entry
                position = entry.get("position", {})
                pos_type = position.get("type", "")
                is_pitcher = pos_type == "Pitcher"

                stats_list = person.get("stats", [])
                if not stats_list:
                    continue

                for stat_group in stats_list:
                    splits = stat_group.get("splits", [])
                    if not splits:
                        continue
                    s = splits[0].get("stat", {})
                    if not s:
                        continue

                    group_name = stat_group.get("group", {}).get("displayName", "")

                    if group_name == "hitting" and not is_pitcher:
                        rows.append(_parse_hitting_stat(player_info, s))
                    elif group_name == "pitching" and is_pitcher:
                        rows.append(_parse_pitching_stat(player_info, s))
                    elif group_name == "hitting" and is_pitcher:
                        pass  # Skip pitcher hitting stats
                    elif group_name == "pitching" and not is_pitcher:
                        pass  # Skip position player pitching stats
                    elif not rows or full_name not in {r["player_name"] for r in rows}:
                        # Fallback: use position type
                        if is_pitcher:
                            rows.append(_parse_pitching_stat(player_info, s))
                        else:
                            rows.append(_parse_hitting_stat(player_info, s))

        except Exception as e:
            logger.debug("Error fetching team %d roster for %d: %s", team_id, season, e)
            continue

    logger.info("Fetched %d season stats for %d players", season, len(rows))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def save_season_stats_to_db(stats_df: pd.DataFrame) -> int:
    """Match fetched stats to players table and save to season_stats."""
    saved = 0
    for _, row in stats_df.iterrows():
        player_id = match_player_id(row["player_name"], row.get("team", ""))
        if player_id is None:
            continue
        upsert_season_stats(player_id, row.to_dict())
        saved += 1
    return saved


def _get_refresh_age_hours(source: str) -> float:
    """How many hours since the last successful refresh."""
    status = get_refresh_status(source)
    if status is None or status["last_refresh"] is None:
        return 999.0
    try:
        last = datetime.fromisoformat(status["last_refresh"])
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        return (datetime.now(UTC) - last).total_seconds() / 3600
    except (ValueError, TypeError):
        return 999.0


def refresh_all_stats(force: bool = False) -> dict:
    """Orchestrator: refresh season stats if stale (>24h)."""
    results = {}

    age = _get_refresh_age_hours("season_stats")
    if force or age > 24:
        try:
            df = fetch_season_stats()
            if not df.empty:
                saved = save_season_stats_to_db(df)
                update_refresh_log("season_stats", "success")
                results["season_stats"] = f"Saved {saved} player stats"
            else:
                update_refresh_log("season_stats", "empty")
                results["season_stats"] = "No data returned"
        except Exception as e:
            update_refresh_log("season_stats", f"error: {e}")
            results["season_stats"] = f"Error: {e}"
    else:
        results["season_stats"] = f"Fresh (updated {age:.1f}h ago)"

    return results


def fetch_all_mlb_players(season: int = 2026) -> pd.DataFrame:
    """Fetch all active MLB players (750+) from MLB Stats API."""
    if statsapi is None:
        raise ImportError("MLB-StatsAPI required. pip install MLB-StatsAPI")

    try:
        data = statsapi.get(
            "sports_players",
            {"season": season, "sportId": 1, "gameType": "R"},
        )
    except Exception as e:
        logger.warning("Failed to fetch player roster: %s", e)
        return pd.DataFrame()

    rows = []
    for p in data.get("people", []):
        if not p.get("active", False):
            continue
        pos_info = p.get("primaryPosition", {})
        pos_abbr = pos_info.get("abbreviation", "Util")
        pos_type = pos_info.get("type", "")
        is_hitter = pos_type != "Pitcher" and pos_abbr != "P"
        rows.append(
            {
                "mlb_id": p.get("id"),
                "name": p.get("fullName", ""),
                "team": p.get("currentTeam", {}).get("abbreviation", ""),
                "positions": pos_abbr if pos_abbr not in ("0", "-", "") else "Util",
                "is_hitter": is_hitter,
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_historical_stats(
    seasons: list[int] | None = None,
) -> dict[int, pd.DataFrame]:
    """Fetch season stats for multiple years. Returns {year: DataFrame}."""
    if seasons is None:
        seasons = [2023, 2024, 2025]
    results = {}
    for year in seasons:
        try:
            df = fetch_season_stats(season=year)
            if not df.empty:
                results[year] = df
                logger.info("Fetched %d player stats for %d", len(df), year)
        except Exception as e:
            logger.warning("Failed to fetch %d stats: %s", year, e)
    return results


def fetch_injury_data_bulk(
    historical_stats: dict[int, pd.DataFrame],
) -> list[dict]:
    """Convert historical stats DataFrames into injury_history records.

    Returns list of dicts: {player_name, team, season, games_played, games_available}.
    Caller must resolve player_name → player_id before DB insert.
    """
    records = []
    for season, df in historical_stats.items():
        if df.empty:
            continue
        games_available = 162
        for _, row in df.iterrows():
            gp = int(row.get("games_played", 0))
            if gp > 0:
                records.append(
                    {
                        "player_name": row.get("player_name", ""),
                        "team": row.get("team", ""),
                        "season": season,
                        "games_played": gp,
                        "games_available": games_available,
                    }
                )
    return records
