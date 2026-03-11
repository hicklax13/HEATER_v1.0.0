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


def fetch_season_stats(season: int = 2026) -> pd.DataFrame:
    """Pull current season stats for all MLB players via MLB Stats API."""
    if statsapi is None:
        raise ImportError("MLB-StatsAPI is required. Install with: pip install MLB-StatsAPI")

    rows = []

    try:
        hitting = statsapi.get(
            "sports_players",
            {
                "season": season,
                "gameType": "R",
                "group": "hitting",
                "stats": "season",
                "limit": 1000,
                "sportId": 1,
            },
        )
        for player in hitting.get("people", []):
            stats_list = player.get("stats", [])
            if not stats_list or not stats_list[0].get("splits"):
                continue
            s = stats_list[0]["splits"][0]["stat"]
            team_info = player.get("currentTeam", {})
            rows.append(
                {
                    "player_name": player.get("fullName", ""),
                    "team": team_info.get("abbreviation", ""),
                    "is_hitter": True,
                    "pa": int(s.get("plateAppearances", 0)),
                    "ab": int(s.get("atBats", 0)),
                    "h": int(s.get("hits", 0)),
                    "r": int(s.get("runs", 0)),
                    "hr": int(s.get("homeRuns", 0)),
                    "rbi": int(s.get("rbi", 0)),
                    "sb": int(s.get("stolenBases", 0)),
                    "avg": float(s.get("avg", "0") or 0),
                    "games_played": int(s.get("gamesPlayed", 0)),
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
            )
    except Exception as e:
        logger.warning("Failed to fetch hitting stats from MLB API: %s", e)

    try:
        pitching = statsapi.get(
            "sports_players",
            {
                "season": season,
                "gameType": "R",
                "group": "pitching",
                "stats": "season",
                "limit": 1000,
                "sportId": 1,
            },
        )
        for player in pitching.get("people", []):
            stats_list = player.get("stats", [])
            if not stats_list or not stats_list[0].get("splits"):
                continue
            s = stats_list[0]["splits"][0]["stat"]
            team_info = player.get("currentTeam", {})
            rows.append(
                {
                    "player_name": player.get("fullName", ""),
                    "team": team_info.get("abbreviation", ""),
                    "is_hitter": False,
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "ip": float(s.get("inningsPitched", "0") or 0),
                    "w": int(s.get("wins", 0)),
                    "sv": int(s.get("saves", 0)),
                    "k": int(s.get("strikeOuts", 0)),
                    "era": float(s.get("era", "0") or 0),
                    "whip": float(s.get("whip", "0") or 0),
                    "er": int(s.get("earnedRuns", 0)),
                    "bb_allowed": int(s.get("baseOnBalls", 0)),
                    "h_allowed": int(s.get("hits", 0)),
                    "games_played": int(s.get("gamesPlayed", 0)),
                }
            )
    except Exception as e:
        logger.warning("Failed to fetch pitching stats from MLB API: %s", e)

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
