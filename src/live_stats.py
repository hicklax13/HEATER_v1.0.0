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
    """Match an external player name to our players table.

    When multiple entries exist for the same name (duplicates from different
    data sources), prefers the entry that has projection data attached — this
    is the canonical ID used by load_player_pool() and the trade engine.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Primary: exact name match, prefer entry with projections
        cursor.execute("SELECT player_id FROM players WHERE name = ?", (player_name,))
        results = cursor.fetchall()
        if results:
            if len(results) == 1:
                return results[0][0]
            # Multiple matches — prefer the one with projections (canonical)
            return _pick_canonical_id(cursor, [r[0] for r in results])

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


def _pick_canonical_id(cursor, player_ids: list[int]) -> int:
    """Given multiple player_ids for the same name, pick the canonical one.

    Preference order:
    1. Has blended projections (used by player_pool)
    2. Has any projections
    3. Lowest player_id (oldest entry)
    """
    for pid in player_ids:
        cursor.execute(
            "SELECT COUNT(*) FROM projections WHERE player_id = ? AND system = 'blended'",
            (pid,),
        )
        if cursor.fetchone()[0] > 0:
            return pid

    for pid in player_ids:
        cursor.execute(
            "SELECT COUNT(*) FROM projections WHERE player_id = ?",
            (pid,),
        )
        if cursor.fetchone()[0] > 0:
            return pid

    return min(player_ids)


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
        "obp": float(stat.get("obp", "0") or 0),
        "bb": int(stat.get("baseOnBalls", 0)),
        "hbp": int(stat.get("hitByPitch", 0)),
        "sf": int(stat.get("sacFlies", 0)),
        "games_played": int(stat.get("gamesPlayed", 0)),
        "ip": 0,
        "w": 0,
        "l": 0,
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
        "obp": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "ip": float(stat.get("inningsPitched", "0") or 0),
        "w": int(stat.get("wins", 0)),
        "l": int(stat.get("losses", 0)),
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


def save_season_stats_to_db(stats_df: pd.DataFrame, season: int = 2026) -> int:
    """Match fetched stats to players table and save to season_stats."""
    saved = 0
    for _, row in stats_df.iterrows():
        player_id = match_player_id(row["player_name"], row.get("team", ""))
        if player_id is None:
            continue
        upsert_season_stats(player_id, row.to_dict(), season=season)
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
                "bats": p.get("batSide", {}).get("code", ""),
                "throws": p.get("pitchHand", {}).get("code", ""),
                "birth_date": p.get("birthDate", ""),
            }
        )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_extended_roster(season: int = 2026) -> pd.DataFrame:
    """Fetch extended player pool: active roster + spring training.

    Queries MLB Stats API with multiple gameTypes to capture players
    beyond the active roster (~1,200+ vs ~750 active-only).
    Deduplicates by mlb_id, preferring active roster entries.
    Falls back to fetch_all_mlb_players() on failure.
    """
    if statsapi is None:
        return fetch_all_mlb_players(season)

    all_players = []
    roster_configs = [
        ("R", "active"),
        ("S", "spring"),
    ]

    for game_type, roster_label in roster_configs:
        try:
            data = statsapi.get(
                "sports_players",
                {"season": season, "sportId": 1, "gameType": game_type},
            )
            for p in data.get("people", []):
                if not p.get("active", False):
                    continue
                pos_info = p.get("primaryPosition", {})
                pos_abbr = pos_info.get("abbreviation", "Util")
                pos_type = pos_info.get("type", "")
                is_hitter = pos_type != "Pitcher" and pos_abbr != "P"
                all_players.append(
                    {
                        "mlb_id": p.get("id"),
                        "name": p.get("fullName", ""),
                        "team": p.get("currentTeam", {}).get("abbreviation", ""),
                        "positions": pos_abbr if pos_abbr not in ("0", "-", "") else "Util",
                        "is_hitter": is_hitter,
                        "bats": p.get("batSide", {}).get("code", ""),
                        "throws": p.get("pitchHand", {}).get("code", ""),
                        "birth_date": p.get("birthDate", ""),
                        "roster_type": roster_label,
                    }
                )
        except Exception:
            logger.warning("Failed to fetch %s roster for season %d", game_type, season)

    if not all_players:
        return fetch_all_mlb_players(season)

    df = pd.DataFrame(all_players)
    # Deduplicate — keep first occurrence (active > spring)
    df = df.drop_duplicates(subset="mlb_id", keep="first")
    return df.reset_index(drop=True)


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


# ── Team Batting Stats (shared by Features 2, 4, 6) ──────────────────

# Module-level cache for team batting stats
_team_batting_cache: dict | None = None
_team_batting_cache_time: datetime | None = None
_TEAM_BATTING_CACHE_HOURS = 24


def fetch_team_batting_stats(season: int = 2026) -> dict[str, dict[str, float]]:
    """Fetch team-level batting stats via MLB Stats API.

    Returns dict mapping team abbreviation to offensive metrics:
        {team_abbr: {wrc_plus, k_pct, bb_pct, iso, woba, ops, avg}}

    Cached in-memory with 24h staleness. Falls back to league-average
    defaults if API is unavailable.
    """
    global _team_batting_cache, _team_batting_cache_time

    # Check cache
    if _team_batting_cache is not None and _team_batting_cache_time is not None:
        age = (datetime.now(UTC) - _team_batting_cache_time).total_seconds() / 3600
        if age < _TEAM_BATTING_CACHE_HOURS:
            return _team_batting_cache

    # League-average defaults (used as fallback)
    league_defaults = {
        "wrc_plus": 100.0,
        "k_pct": 22.3,
        "bb_pct": 8.5,
        "iso": 0.150,
        "woba": 0.315,
        "ops": 0.720,
        "avg": 0.248,
    }

    if statsapi is None:
        logger.warning("MLB-StatsAPI not available, using league average defaults")
        return _fallback_team_stats(league_defaults)

    try:
        # Fetch team stats from MLB Stats API
        teams_data = statsapi.get(
            "teams",
            {"sportId": 1, "season": season, "fields": "teams,id,abbreviation,name"},
        )
        teams_list = teams_data.get("teams", [])

        result: dict[str, dict[str, float]] = {}
        for team in teams_list:
            abbr = team.get("abbreviation", "")
            team_id = team.get("id")
            if not abbr or not team_id:
                continue

            try:
                stats_data = statsapi.get(
                    "team_stats",
                    {"teamId": team_id, "season": season, "stats": "season", "group": "hitting"},
                )
                splits = stats_data.get("stats", [{}])[0].get("splits", [])
                if splits:
                    stat = splits[0].get("stat", {})
                    result[abbr] = {
                        "wrc_plus": 100.0,  # Not directly in API, use 100 as proxy
                        "k_pct": _pct(stat.get("strikeOuts", 0), stat.get("plateAppearances", 1)),
                        "bb_pct": _pct(stat.get("baseOnBalls", 0), stat.get("plateAppearances", 1)),
                        "iso": float(stat.get("slg", 0.400)) - float(stat.get("avg", 0.250)),
                        "woba": 0.315,  # Approximate, not directly in API
                        "ops": float(stat.get("ops", 0.720)),
                        "avg": float(stat.get("avg", 0.248)),
                    }
                else:
                    result[abbr] = dict(league_defaults)
            except Exception:
                result[abbr] = dict(league_defaults)

        if result:
            _team_batting_cache = result
            _team_batting_cache_time = datetime.now(UTC)
            return result

    except Exception:
        logger.warning("Failed to fetch team batting stats, using defaults")

    return _fallback_team_stats(league_defaults)


def _pct(numerator: float, denominator: float) -> float:
    """Compute percentage, avoiding division by zero."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 1)


def _fallback_team_stats(defaults: dict) -> dict[str, dict[str, float]]:
    """Return league-average defaults for all 30 teams."""
    teams = [
        "ARI",
        "ATL",
        "BAL",
        "BOS",
        "CHC",
        "CWS",
        "CIN",
        "CLE",
        "COL",
        "DET",
        "HOU",
        "KC",
        "LAA",
        "LAD",
        "MIA",
        "MIL",
        "MIN",
        "NYM",
        "NYY",
        "OAK",
        "PHI",
        "PIT",
        "SD",
        "SF",
        "SEA",
        "STL",
        "TB",
        "TEX",
        "TOR",
        "WSH",
    ]
    return {t: dict(defaults) for t in teams}
