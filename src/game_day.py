"""Game-day intelligence module.

Fetches weather, opposing pitchers, team strength, lineup confirmations,
and recent player form to power daily lineup and start/sit decisions.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]

try:
    from pybaseball import team_batting, team_pitching

    PYBASEBALL_AVAILABLE = True
except ImportError:  # pragma: no cover
    team_batting = None  # type: ignore[assignment]
    team_pitching = None  # type: ignore[assignment]
    PYBASEBALL_AVAILABLE = False

try:
    import statsapi as _statsapi
except ImportError:  # pragma: no cover
    _statsapi = None  # type: ignore[assignment]

from src.database import load_team_strength, upsert_opp_pitcher, upsert_team_strength  # noqa: E402

# ── Stadium coordinates for weather lookups (lat, lon) ───────────────
STADIUM_COORDS: dict[str, tuple[float, float]] = {
    "ARI": (33.4455, -112.0667),
    "ATL": (33.8907, -84.4677),
    "BAL": (39.2838, -76.6218),
    "BOS": (42.3467, -71.0972),
    "CHC": (41.9484, -87.6553),
    "CWS": (41.8299, -87.6338),
    "CIN": (39.0975, -84.5064),
    "CLE": (41.4962, -81.6852),
    "COL": (39.7559, -104.9942),
    "DET": (42.3390, -83.0485),
    "HOU": (29.7573, -95.3555),
    "KC": (39.0517, -94.4803),
    "LAA": (33.8003, -117.8827),
    "LAD": (34.0739, -118.2400),
    "MIA": (25.7781, -80.2197),
    "MIL": (43.0280, -87.9712),
    "MIN": (44.9817, -93.2776),
    "NYM": (40.7571, -73.8458),
    "NYY": (40.8296, -73.9262),
    "ATH": (38.5816, -121.5015),
    "PHI": (39.9061, -75.1665),
    "PIT": (40.4469, -80.0057),
    "SD": (32.7076, -117.1570),
    "SF": (37.7786, -122.3893),
    "SEA": (47.5914, -122.3325),
    "STL": (38.6226, -90.1928),
    "TB": (27.7682, -82.6534),
    "TEX": (32.7512, -97.0832),
    "TOR": (43.6414, -79.3894),
    "WSH": (38.8730, -77.0074),
}

# Teams with retractable or enclosed roofs (weather less relevant)
DOME_TEAMS: set[str] = {"ARI", "HOU", "MIA", "MIL", "SEA", "TB", "TEX", "TOR"}

# T11: Outfield bearing angles (degrees from north, center field direction).
# Used by E6 for wind direction adjustment. Compiled from satellite imagery.
# 0° = North, 90° = East, 180° = South, 270° = West.
OUTFIELD_BEARING: dict[str, float] = {
    "ARI": 0,  # Chase Field (retractable, indoor mostly)
    "ATL": 175,  # Truist Park — CF faces roughly south
    "BAL": 180,  # Camden Yards — CF faces south
    "BOS": 200,  # Fenway Park — CF faces SSW
    "CHC": 185,  # Wrigley Field — CF faces south (wind blowing out = from S)
    "CWS": 200,  # Guaranteed Rate — CF faces SSW
    "CIN": 180,  # Great American — CF faces south
    "CLE": 175,  # Progressive Field — CF faces roughly south
    "COL": 175,  # Coors Field — CF faces south
    "DET": 180,  # Comerica Park — CF faces south
    "HOU": 0,  # Minute Maid (retractable)
    "KC": 180,  # Kauffman Stadium — CF faces south
    "LAA": 200,  # Angel Stadium — CF faces SSW
    "LAD": 180,  # Dodger Stadium — CF faces south
    "MIA": 0,  # loanDepot Park (retractable)
    "MIL": 0,  # American Family Field (retractable)
    "MIN": 205,  # Target Field — CF faces SSW
    "NYM": 135,  # Citi Field — CF faces SE
    "NYY": 180,  # Yankee Stadium — CF faces south
    "ATH": 170,  # Sutter Health Park (Sacramento) — CF faces roughly south
    "PHI": 190,  # Citizens Bank Park — CF faces south
    "PIT": 45,  # PNC Park — CF faces NE (unique)
    "SD": 180,  # Petco Park — CF faces south
    "SF": 180,  # Oracle Park — CF faces south
    "SEA": 0,  # T-Mobile Park (retractable)
    "STL": 180,  # Busch Stadium — CF faces south
    "TB": 0,  # Tropicana Field (dome)
    "TEX": 0,  # Globe Life Field (retractable)
    "TOR": 0,  # Rogers Centre (retractable)
    "WSH": 180,  # Nationals Park — CF faces south
}

# ── FanGraphs team name -> standard abbreviation mapping ───────────────
FG_TEAM_TO_ABBR: dict[str, str] = {
    "Angels": "LAA",
    "Astros": "HOU",
    "Athletics": "ATH",
    "Blue Jays": "TOR",
    "Braves": "ATL",
    "Brewers": "MIL",
    "Cardinals": "STL",
    "Cubs": "CHC",
    "Diamondbacks": "ARI",
    "Dodgers": "LAD",
    "Giants": "SF",
    "Guardians": "CLE",
    "Mariners": "SEA",
    "Marlins": "MIA",
    "Mets": "NYM",
    "Nationals": "WSH",
    "Orioles": "BAL",
    "Padres": "SD",
    "Phillies": "PHI",
    "Pirates": "PIT",
    "Rangers": "TEX",
    "Rays": "TB",
    "Red Sox": "BOS",
    "Reds": "CIN",
    "Rockies": "COL",
    "Royals": "KC",
    "Tigers": "DET",
    "Twins": "MIN",
    "White Sox": "CWS",
    "Yankees": "NYY",
    # FanGraphs abbreviation form (used by pybaseball team_batting/team_pitching)
    "ARI": "ARI",
    "ATH": "ATH",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CHW": "CWS",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KCR": "KC",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "PHI": "PHI",
    "PIT": "PIT",
    "SDP": "SD",
    "SEA": "SEA",
    "SFG": "SF",
    "STL": "STL",
    "TBR": "TB",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSN": "WSH",
}

# Neutral defaults when team data is unavailable
_NEUTRAL_DEFAULTS: dict[str, float] = {
    "wrc_plus": 100.0,
    "fip": 4.00,
    "team_ops": 0.720,
    "team_era": 4.00,
    "team_whip": 1.25,
    "k_pct": 22.0,
    "bb_pct": 8.0,
}


def _safe_float(val: object) -> float | None:
    """Convert a value to float, returning None on failure."""
    if val is None:
        return None
    try:
        result = float(val)
        if pd.isna(result):
            return None
        return result
    except (ValueError, TypeError):
        return None


def _parse_pct(val: object) -> float | None:
    """Parse a percentage value that may be a string like '22.5 %' or a float."""
    if val is None:
        return None
    if isinstance(val, str):
        val = val.replace("%", "").replace(" ", "").strip()
    try:
        result = float(val)
        if pd.isna(result):
            return None
        # pybaseball sometimes returns K% as 0.225 (ratio) vs 22.5 (pct)
        # Normalize to percentage form (0-100)
        if 0 < result < 1:
            result = result * 100.0
        return result
    except (ValueError, TypeError):
        return None


def fetch_game_day_intelligence() -> dict:
    """Orchestrate all game-day data fetches and return a combined intel dict.

    Returns a dict with keys: weather, pitchers, lineups, team_strength,
    fetched_at. Each value contains the relevant data for today's games.

    This is the main entry point for game-day data; it coordinates calls
    to fetch_game_day_weather(), fetch_opposing_pitchers(),
    get_todays_lineups(), and fetch_team_strength().
    """
    if _statsapi is None:
        logger.warning("statsapi not available; returning empty intelligence")
        return {
            "weather": [],
            "pitchers": [],
            "lineups": {},
            "team_strength": pd.DataFrame(),
            "fetched_at": datetime.now(UTC).isoformat(),
        }

    # MLB schedule dates are in US Eastern time, not UTC
    _ET = timezone(timedelta(hours=-4))  # EDT
    today_str = datetime.now(_ET).strftime("%Y-%m-%d")
    try:
        schedule = _statsapi.schedule(date=today_str)
    except Exception:
        logger.exception("Failed to fetch today's schedule")
        schedule = []

    weather = fetch_game_day_weather(schedule)
    pitchers = fetch_opposing_pitchers(schedule)
    lineups = get_todays_lineups(schedule)

    season = datetime.now(UTC).year
    team_str = fetch_team_strength(season)

    return {
        "games": len(schedule),
        "weather": len(weather),
        "weather_data": weather,
        "pitchers": pitchers,
        "lineups": lineups,
        "team_strength": team_str,
        "games_count": len(schedule),
        "weather_count": len(weather),
        "pitcher_count": len(pitchers),
        "fetched_at": datetime.now(UTC).isoformat(),
    }


def fetch_game_day_weather(schedule: list[dict]) -> list[dict]:
    """Fetch Open-Meteo weather forecasts for today's game venues.

    For each game, looks up the home team's stadium coordinates and fetches
    hourly weather from the Open-Meteo API (free, no API key). Dome/retractable
    roof stadiums get neutral defaults instead of a live fetch.

    Args:
        schedule: List of game dicts from ``statsapi.schedule()`` containing
            ``home_name``, ``game_pk``, ``game_datetime``, etc.

    Returns:
        List of weather dicts stored in DB, one per game.
    """
    from src.database import upsert_game_day_weather

    # Map full team names to abbreviations for STADIUM_COORDS lookup
    _NAME_TO_ABBR: dict[str, str] = {
        "Arizona Diamondbacks": "ARI",
        "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC",
        "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN",
        "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL",
        "Detroit Tigers": "DET",
        "Houston Astros": "HOU",
        "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA",
        "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN",
        "New York Mets": "NYM",
        "New York Yankees": "NYY",
        "Oakland Athletics": "ATH",
        "Athletics": "ATH",
        "Philadelphia Phillies": "PHI",
        "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD",
        "San Francisco Giants": "SF",
        "Seattle Mariners": "SEA",
        "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB",
        "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR",
        "Washington Nationals": "WSH",
    }

    _NEUTRAL = {
        "temp_f": 72.0,
        "wind_mph": 0.0,
        "wind_dir": "None",
        "precip_pct": 0.0,
        "humidity_pct": 50.0,
    }

    results: list[dict] = []
    today_str = datetime.now(UTC).strftime("%Y-%m-%d")

    for game in schedule:
        game_pk = game.get("game_id") or game.get("game_pk") or game.get("gamePk", 0)
        home_name = game.get("home_name", "")
        venue_abbr = _NAME_TO_ABBR.get(home_name, "")

        if not venue_abbr or venue_abbr not in STADIUM_COORDS:
            logger.debug("Unknown venue team: %s", home_name)
            continue

        weather = dict(_NEUTRAL)
        weather["game_pk"] = game_pk
        weather["game_date"] = today_str
        weather["venue_team"] = venue_abbr

        # Dome teams get neutral defaults — no API call needed
        if venue_abbr in DOME_TEAMS:
            try:
                upsert_game_day_weather(**weather)
            except Exception:
                logger.debug("Failed to store dome weather for %s", venue_abbr)
            results.append(weather)
            continue

        # Fetch from Open-Meteo (free, no API key)
        lat, lon = STADIUM_COORDS[venue_abbr]
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&hourly=temperature_2m,windspeed_10m,winddirection_10m,"
                f"precipitation_probability,relativehumidity_2m"
                f"&temperature_unit=fahrenheit&windspeed_unit=mph"
                f"&forecast_days=1"
            )
            resp = _requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            hourly = data.get("hourly", {})
            temps = hourly.get("temperature_2m", [])
            winds = hourly.get("windspeed_10m", [])
            wind_dirs = hourly.get("winddirection_10m", [])
            precips = hourly.get("precipitation_probability", [])
            humids = hourly.get("relativehumidity_2m", [])

            # Pick game-time hour (default 19:00 local = ~7pm first pitch)
            game_dt = game.get("game_datetime", "")
            try:
                hour = int(game_dt[11:13]) if game_dt and len(game_dt) > 13 else 19
            except (ValueError, IndexError):
                hour = 19
            # Clamp to available data range
            idx = min(hour, len(temps) - 1) if temps else 0

            if temps:
                weather["temp_f"] = round(temps[idx], 1)
            if winds:
                weather["wind_mph"] = round(winds[idx], 1)
            if wind_dirs:
                # Convert degrees to cardinal direction
                deg = wind_dirs[idx]
                directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
                weather["wind_dir"] = directions[int((deg + 22.5) / 45) % 8]
            if precips:
                weather["precip_pct"] = round(precips[idx], 1)
            if humids:
                weather["humidity_pct"] = round(humids[idx], 1)

        except Exception:
            logger.warning(
                "Open-Meteo fetch failed for %s; using neutral defaults",
                venue_abbr,
                exc_info=True,
            )

        try:
            upsert_game_day_weather(**weather)
        except Exception:
            logger.debug("Failed to store weather for %s", venue_abbr)

        results.append(weather)
        time.sleep(0.3)  # Respectful rate limiting

    logger.info(
        "Fetched weather for %d games (%d dome)", len(results), sum(1 for r in results if r["venue_team"] in DOME_TEAMS)
    )
    return results


def fetch_opposing_pitchers(schedule: list[dict]) -> list[dict]:
    """Fetch season stats + platoon splits for each probable starter today.

    For each game in the schedule, resolves both the home and away probable
    pitchers to MLB IDs, fetches their season pitching stats and vs-LHB /
    vs-RHB platoon splits, persists the data via ``upsert_opp_pitcher()``,
    and returns the collected stat dicts.

    Args:
        schedule: List of game dicts from ``statsapi.schedule()`` containing
            ``home_probable_pitcher``, ``away_probable_pitcher``, etc.

    Returns:
        List of pitcher stat dicts stored in DB.  Each dict has keys:
        pitcher_id, name, team, era, fip, xfip, whip, k_per_9, bb_per_9,
        vs_lhb_avg, vs_rhb_avg, ip, hand.
    """
    if _statsapi is None:
        logger.warning("statsapi not available; skipping opposing pitchers")
        return []

    season = datetime.now(UTC).year
    results: list[dict] = []
    seen_ids: set[int] = set()

    for game in schedule:
        for side in ("home_probable_pitcher", "away_probable_pitcher"):
            pitcher_name = game.get(side) or ""
            if not pitcher_name or pitcher_name.upper() in ("TBD", ""):
                continue

            # Determine team context from the game dict
            if side == "home_probable_pitcher":
                team_name = game.get("home_name", "")
            else:
                team_name = game.get("away_name", "")

            try:
                record = _fetch_single_pitcher(pitcher_name, team_name, season, seen_ids)
                if record is not None:
                    results.append(record)
            except Exception:
                logger.exception("Failed to fetch pitcher data for %s", pitcher_name)

            # Rate-limit between API calls
            time.sleep(0.5)

    logger.info(
        "Fetched opposing pitcher data for %d pitchers from %d games",
        len(results),
        len(schedule),
    )
    return results


def _fetch_single_pitcher(
    name: str,
    team_name: str,
    season: int,
    seen_ids: set[int],
) -> dict | None:
    """Resolve one pitcher by name and fetch season stats + platoon splits.

    Returns a stat dict on success or ``None`` if the pitcher could not be
    resolved or has already been processed.
    """
    if _statsapi is None:
        return None

    # --- Resolve name to MLB ID ---
    matches = _statsapi.lookup_player(name)
    if not matches:
        logger.warning("Could not resolve pitcher name: %s", name)
        return None

    mlb_id: int = matches[0]["id"]
    if mlb_id in seen_ids:
        return None
    seen_ids.add(mlb_id)

    # --- Season stats ---
    stat_data = _statsapi.player_stat_data(mlb_id, group="pitching", type="season", sportId=1)
    hand: str | None = stat_data.get("pitch_hand")
    team: str = stat_data.get("current_team", team_name) or team_name

    era: float | None = None
    whip: float | None = None
    ip: float | None = None
    k_per_9: float | None = None
    bb_per_9: float | None = None

    stats_list = stat_data.get("stats") or []
    if stats_list:
        s = stats_list[0].get("stats", {})
        era = _safe_float(s.get("era"))
        whip = _safe_float(s.get("whip"))
        ip = _safe_float(s.get("inningsPitched"))
        k_per_9 = _safe_float(s.get("strikeoutsPer9Inn"))
        bb_per_9 = _safe_float(s.get("walksPer9Inn"))

    # FIP / xFIP are not in the MLB Stats API season endpoint — leave None
    fip: float | None = None
    xfip: float | None = None

    # --- Platoon splits (vs LHB / vs RHB) ---
    vs_lhb_avg: float | None = None
    vs_rhb_avg: float | None = None
    try:
        time.sleep(0.5)  # rate-limit
        splits_data = _statsapi.get(
            "person",
            {
                "personId": mlb_id,
                "hydrate": ("stats(group=[pitching],type=[statSplits],sitCodes=[vl,vr])"),
            },
        )
        people = splits_data.get("people") or []
        if people:
            for stat_block in people[0].get("stats", []):
                for split in stat_block.get("splits", []):
                    code = split.get("split", {}).get("code", "")
                    avg_val = split.get("stat", {}).get("avg")
                    if code == "vl":
                        vs_lhb_avg = _safe_float(avg_val)
                    elif code == "vr":
                        vs_rhb_avg = _safe_float(avg_val)
    except Exception:
        logger.warning("Could not fetch platoon splits for %s (ID %d)", name, mlb_id)

    # --- Persist and return ---
    upsert_opp_pitcher(
        pitcher_id=mlb_id,
        season=season,
        name=name,
        team=team,
        era=era,
        fip=fip,
        xfip=xfip,
        whip=whip,
        k_per_9=k_per_9,
        bb_per_9=bb_per_9,
        vs_lhb_avg=vs_lhb_avg,
        vs_rhb_avg=vs_rhb_avg,
        ip=ip,
        hand=hand,
    )

    record = {
        "pitcher_id": mlb_id,
        "name": name,
        "team": team,
        "era": era,
        "fip": fip,
        "xfip": xfip,
        "whip": whip,
        "k_per_9": k_per_9,
        "bb_per_9": bb_per_9,
        "vs_lhb_avg": vs_lhb_avg,
        "vs_rhb_avg": vs_rhb_avg,
        "ip": ip,
        "hand": hand,
    }
    logger.debug("Fetched pitcher: %s (ID %d) — ERA %s, WHIP %s", name, mlb_id, era, whip)
    return record


def _fetch_team_strength_statsapi(season: int) -> pd.DataFrame:
    """Fallback: fetch team strength via MLB Stats API when pybaseball fails.

    Uses the same endpoints as ``fetch_team_batting_stats()`` in live_stats.py
    to get team-level batting and pitching stats. Approximates wRC+ from OPS
    and computes FIP from component stats.

    Args:
        season: MLB season year.

    Returns:
        DataFrame matching the ``fetch_team_strength()`` schema, or cached
        data from SQLite if the API call also fails.
    """
    if _statsapi is None:
        logger.warning("statsapi not available; loading cached team strength")
        return load_team_strength(season)

    try:
        teams_data = _statsapi.get(
            "teams",
            {"sportId": 1, "season": season, "fields": "teams,id,abbreviation,name"},
        )
        teams_list = teams_data.get("teams", [])
    except Exception:
        logger.exception("MLB Stats API teams fetch failed")
        return load_team_strength(season)

    if not teams_list:
        return load_team_strength(season)

    rows: list[dict] = []
    for team in teams_list:
        abbr = team.get("abbreviation", "")
        team_id = team.get("id")
        if not abbr or not team_id:
            continue

        row: dict[str, object] = {"team_abbr": abbr, "season": season}

        # Fetch batting stats
        try:
            bat_data = _statsapi.get(
                "team_stats",
                {"teamId": team_id, "season": season, "stats": "season", "group": "hitting"},
            )
            bat_splits = bat_data.get("stats", [{}])[0].get("splits", [])
            if bat_splits:
                stat = bat_splits[0].get("stat", {})
                ops = float(stat.get("ops", "0.720") or 0.720)
                # Approximate wRC+ from OPS: league average OPS ~0.720 maps to wRC+ 100
                row["wrc_plus"] = round((ops / 0.720) * 100.0, 1)
                row["team_ops"] = round(ops, 3)
                pa = int(stat.get("plateAppearances", 0) or 0)
                if pa > 0:
                    row["k_pct"] = round(int(stat.get("strikeOuts", 0) or 0) / pa * 100, 1)
                    row["bb_pct"] = round(int(stat.get("baseOnBalls", 0) or 0) / pa * 100, 1)
                else:
                    row["k_pct"] = _NEUTRAL_DEFAULTS["k_pct"]
                    row["bb_pct"] = _NEUTRAL_DEFAULTS["bb_pct"]
            else:
                row["wrc_plus"] = _NEUTRAL_DEFAULTS["wrc_plus"]
                row["team_ops"] = _NEUTRAL_DEFAULTS["team_ops"]
                row["k_pct"] = _NEUTRAL_DEFAULTS["k_pct"]
                row["bb_pct"] = _NEUTRAL_DEFAULTS["bb_pct"]
        except Exception:
            row["wrc_plus"] = _NEUTRAL_DEFAULTS["wrc_plus"]
            row["team_ops"] = _NEUTRAL_DEFAULTS["team_ops"]
            row["k_pct"] = _NEUTRAL_DEFAULTS["k_pct"]
            row["bb_pct"] = _NEUTRAL_DEFAULTS["bb_pct"]

        # Fetch pitching stats
        try:
            pit_data = _statsapi.get(
                "team_stats",
                {"teamId": team_id, "season": season, "stats": "season", "group": "pitching"},
            )
            pit_splits = pit_data.get("stats", [{}])[0].get("splits", [])
            if pit_splits:
                stat = pit_splits[0].get("stat", {})
                row["team_era"] = _safe_float(stat.get("era")) or _NEUTRAL_DEFAULTS["team_era"]
                row["team_whip"] = _safe_float(stat.get("whip")) or _NEUTRAL_DEFAULTS["team_whip"]
                # Approximate FIP from component stats: FIP ~= ERA for this fallback
                # True FIP requires HR, BB, HBP, K, IP which are available but
                # the FIP constant changes yearly. Use ERA as proxy.
                row["fip"] = row["team_era"]
            else:
                row["team_era"] = _NEUTRAL_DEFAULTS["team_era"]
                row["team_whip"] = _NEUTRAL_DEFAULTS["team_whip"]
                row["fip"] = _NEUTRAL_DEFAULTS["fip"]
        except Exception:
            row["team_era"] = _NEUTRAL_DEFAULTS["team_era"]
            row["team_whip"] = _NEUTRAL_DEFAULTS["team_whip"]
            row["fip"] = _NEUTRAL_DEFAULTS["fip"]

        rows.append(row)

        # Persist to DB
        upsert_team_strength(
            team_abbr=str(abbr),
            season=season,
            wrc_plus=row.get("wrc_plus"),  # type: ignore[arg-type]
            fip=row.get("fip"),  # type: ignore[arg-type]
            team_ops=row.get("team_ops"),  # type: ignore[arg-type]
            team_era=row.get("team_era"),  # type: ignore[arg-type]
            team_whip=row.get("team_whip"),  # type: ignore[arg-type]
            k_pct=row.get("k_pct"),  # type: ignore[arg-type]
            bb_pct=row.get("bb_pct"),  # type: ignore[arg-type]
        )

    if rows:
        logger.info(
            "Stored team strength via MLB Stats API for %d teams (season %d)",
            len(rows),
            season,
        )
        return pd.DataFrame(rows)

    return load_team_strength(season)


def fetch_team_strength(season: int) -> pd.DataFrame:
    """Fetch team-level wRC+/FIP/OPS/ERA from pybaseball (FanGraphs).

    Calls pybaseball.team_batting() and team_pitching() to get
    FanGraphs team-level advanced stats. Stores to team_strength table.

    Args:
        season: MLB season year (e.g. 2026).

    Returns:
        DataFrame with columns: team_abbr, season, wrc_plus, fip,
        team_ops, team_era, team_whip, k_pct, bb_pct, fetched_at.
    """
    if not PYBASEBALL_AVAILABLE:
        logger.warning("pybaseball not available; trying MLB Stats API fallback")
        return _fetch_team_strength_statsapi(season)

    try:
        logger.info("Fetching team batting stats from FanGraphs for %d", season)
        bat_df = team_batting(season)
        logger.info("Fetching team pitching stats from FanGraphs for %d", season)
        pit_df = team_pitching(season)
    except Exception:
        logger.exception("Failed to fetch team stats from FanGraphs; trying MLB Stats API")
        return _fetch_team_strength_statsapi(season)

    if bat_df is None or bat_df.empty or pit_df is None or pit_df.empty:
        logger.warning("Empty team stats from FanGraphs; trying MLB Stats API")
        return _fetch_team_strength_statsapi(season)

    # Map FanGraphs team names to abbreviations
    bat_df = bat_df.copy()
    pit_df = pit_df.copy()
    bat_df["team_abbr"] = bat_df["Team"].map(FG_TEAM_TO_ABBR)
    pit_df["team_abbr"] = pit_df["Team"].map(FG_TEAM_TO_ABBR)

    # Drop rows that didn't map (league averages, etc.)
    bat_df = bat_df.dropna(subset=["team_abbr"])
    pit_df = pit_df.dropna(subset=["team_abbr"])

    # Extract relevant columns — pybaseball column names vary slightly
    # Batting: wRC+, OPS, K%, BB%
    bat_cols = {}
    for col in bat_df.columns:
        col_lower = col.lower().replace(" ", "")
        if col_lower == "wrc+":
            bat_cols["wrc_plus"] = col
        elif col_lower == "ops":
            bat_cols["team_ops"] = col
        elif col_lower == "k%":
            bat_cols["k_pct"] = col
        elif col_lower == "bb%":
            bat_cols["bb_pct"] = col

    # Pitching: FIP, ERA, WHIP
    pit_cols = {}
    for col in pit_df.columns:
        col_lower = col.lower().replace(" ", "")
        if col_lower == "fip":
            pit_cols["fip"] = col
        elif col_lower == "era":
            pit_cols["team_era"] = col
        elif col_lower == "whip":
            pit_cols["team_whip"] = col

    # Build merged result
    rows: list[dict] = []
    all_abbrs = set(bat_df["team_abbr"].tolist()) | set(pit_df["team_abbr"].tolist())

    for abbr in sorted(all_abbrs):
        bat_row = bat_df[bat_df["team_abbr"] == abbr]
        pit_row = pit_df[pit_df["team_abbr"] == abbr]

        row: dict[str, object] = {"team_abbr": abbr, "season": season}

        # Extract batting metrics
        if not bat_row.empty:
            br = bat_row.iloc[0]
            row["wrc_plus"] = _safe_float(br.get(bat_cols.get("wrc_plus", ""), None))
            row["team_ops"] = _safe_float(br.get(bat_cols.get("team_ops", ""), None))
            # K% and BB% may come as strings like "22.5 %" or floats
            row["k_pct"] = _parse_pct(br.get(bat_cols.get("k_pct", ""), None))
            row["bb_pct"] = _parse_pct(br.get(bat_cols.get("bb_pct", ""), None))
        else:
            row["wrc_plus"] = None
            row["team_ops"] = None
            row["k_pct"] = None
            row["bb_pct"] = None

        # Extract pitching metrics
        if not pit_row.empty:
            pr = pit_row.iloc[0]
            row["fip"] = _safe_float(pr.get(pit_cols.get("fip", ""), None))
            row["team_era"] = _safe_float(pr.get(pit_cols.get("team_era", ""), None))
            row["team_whip"] = _safe_float(pr.get(pit_cols.get("team_whip", ""), None))
        else:
            row["fip"] = None
            row["team_era"] = None
            row["team_whip"] = None

        rows.append(row)

        # Persist to DB
        upsert_team_strength(
            team_abbr=str(abbr),
            season=season,
            wrc_plus=row.get("wrc_plus"),  # type: ignore[arg-type]
            fip=row.get("fip"),  # type: ignore[arg-type]
            team_ops=row.get("team_ops"),  # type: ignore[arg-type]
            team_era=row.get("team_era"),  # type: ignore[arg-type]
            team_whip=row.get("team_whip"),  # type: ignore[arg-type]
            k_pct=row.get("k_pct"),  # type: ignore[arg-type]
            bb_pct=row.get("bb_pct"),  # type: ignore[arg-type]
        )

    logger.info("Stored team strength for %d teams (season %d)", len(rows), season)
    return pd.DataFrame(rows)


def get_team_strength(team_abbr: str) -> dict:
    """Read team strength from DB (cached).

    Args:
        team_abbr: Standard team abbreviation (e.g. 'NYY', 'LAD').

    Returns:
        Dict with: wrc_plus, fip, team_ops, team_era, team_whip,
        k_pct, bb_pct. Returns neutral defaults if team not found.
    """
    season = datetime.now(UTC).year
    df = load_team_strength(season)

    if not df.empty:
        team_row = df[df["team_abbr"] == team_abbr]
        if not team_row.empty:
            row = team_row.iloc[0]
            return {
                "wrc_plus": _safe_float(row.get("wrc_plus")) or 100.0,
                "fip": _safe_float(row.get("fip")) or 4.00,
                "team_ops": _safe_float(row.get("team_ops")) or 0.720,
                "team_era": _safe_float(row.get("team_era")) or 4.00,
                "team_whip": _safe_float(row.get("team_whip")) or 1.25,
                "k_pct": _safe_float(row.get("k_pct")) or 22.0,
                "bb_pct": _safe_float(row.get("bb_pct")) or 8.0,
            }

    logger.debug("No team strength data for %s; returning neutral defaults", team_abbr)
    return dict(_NEUTRAL_DEFAULTS)


def _aggregate_hitting_games(games: list[dict]) -> dict:
    """Aggregate per-game hitting stats into a summary dict.

    Args:
        games: List of per-game stat dicts from statsapi gameLog.

    Returns:
        Dict with counting and rate stats. Empty dict if no games.
    """
    if not games:
        return {}

    h = sum(g.get("hits", 0) for g in games)
    ab = sum(g.get("atBats", 0) for g in games)
    pa = sum(g.get("plateAppearances", 0) for g in games)
    hr = sum(g.get("homeRuns", 0) for g in games)
    rbi = sum(g.get("rbi", 0) for g in games)
    sb = sum(g.get("stolenBases", 0) for g in games)
    r = sum(g.get("runs", 0) for g in games)
    bb = sum(g.get("baseOnBalls", 0) for g in games)
    hbp = sum(g.get("hitByPitch", 0) for g in games)
    sf = sum(g.get("sacFlies", 0) for g in games)

    avg = h / ab if ab > 0 else 0.0
    obp_denom = ab + bb + hbp + sf
    obp = (h + bb + hbp) / obp_denom if obp_denom > 0 else 0.0

    return {
        "avg": round(avg, 3),
        "obp": round(obp, 3),
        "hr": hr,
        "rbi": rbi,
        "sb": sb,
        "r": r,
        "h": h,
        "ab": ab,
        "pa": pa,
        "games": len(games),
    }


def _aggregate_pitching_games(games: list[dict]) -> dict:
    """Aggregate per-game pitching stats into a summary dict.

    Args:
        games: List of per-game stat dicts from statsapi gameLog.

    Returns:
        Dict with counting and rate stats. Empty dict if no games.
    """
    if not games:
        return {}

    total_ip = 0.0
    for g in games:
        ip_str = g.get("inningsPitched", "0")
        try:
            # MLB IP format: "6.1" means 6 and 1/3 innings
            parts = str(ip_str).split(".")
            full = int(parts[0])
            thirds = int(parts[1]) if len(parts) > 1 else 0
            total_ip += full + thirds / 3.0
        except (ValueError, IndexError):
            pass

    k = sum(g.get("strikeOuts", 0) for g in games)
    w = sum(g.get("wins", 0) for g in games)
    er = sum(g.get("earnedRuns", 0) for g in games)
    bb_allowed = sum(g.get("baseOnBalls", 0) for g in games)
    h_allowed = sum(g.get("hits", 0) for g in games)

    era = (er * 9 / total_ip) if total_ip > 0 else 0.0
    whip = (bb_allowed + h_allowed) / total_ip if total_ip > 0 else 0.0

    return {
        "era": round(era, 2),
        "whip": round(whip, 2),
        "k": k,
        "w": w,
        "ip": round(total_ip, 1),
        "h_allowed": h_allowed,
        "bb_allowed": bb_allowed,
        "er": er,
        "games": len(games),
    }


def get_player_recent_form(mlb_id: int) -> dict:
    """On-demand last 7/14/30 game stats from MLB Stats API gameLog.

    Fetches recent game stats for a player using the MLB Stats API.
    Returns stats for last 7, 14, and 30 games.

    Tries hitting stats first; if empty, falls back to pitching.

    Args:
        mlb_id: MLB player ID (integer).

    Returns:
        Dict with keys 'l7', 'l14', 'l30', each containing stat dicts.
        For hitters: avg, obp, hr, rbi, sb, r, h, ab, pa, games.
        For pitchers: era, whip, k, w, ip, h_allowed, bb_allowed, er, games.
        Also includes 'player_type' ('hitter' or 'pitcher') and 'mlb_id'.
        Returns empty dicts on failure.
    """
    empty = {
        "l7": {},
        "l14": {},
        "l30": {},
        "player_type": "unknown",
        "mlb_id": mlb_id,
    }

    try:
        import statsapi
    except ImportError:
        logger.warning("statsapi not installed, cannot fetch recent form")
        return empty

    # Try hitting first
    try:
        result = statsapi.player_stat_data(mlb_id, group="hitting", type="gameLog", sportId=1)
        game_stats = [entry["stats"] for entry in result.get("stats", [])]
        if game_stats:
            return {
                "l7": _aggregate_hitting_games(game_stats[:7]),
                "l14": _aggregate_hitting_games(game_stats[:14]),
                "l30": _aggregate_hitting_games(game_stats[:30]),
                "player_type": "hitter",
                "mlb_id": mlb_id,
            }
    except Exception:
        logger.debug("No hitting gameLog for mlb_id=%d", mlb_id)

    # Brief pause before pitching request
    time.sleep(0.5)

    # Try pitching
    try:
        result = statsapi.player_stat_data(mlb_id, group="pitching", type="gameLog", sportId=1)
        game_stats = [entry["stats"] for entry in result.get("stats", [])]
        if game_stats:
            return {
                "l7": _aggregate_pitching_games(game_stats[:7]),
                "l14": _aggregate_pitching_games(game_stats[:14]),
                "l30": _aggregate_pitching_games(game_stats[:30]),
                "player_type": "pitcher",
                "mlb_id": mlb_id,
            }
    except Exception:
        logger.debug("No pitching gameLog for mlb_id=%d", mlb_id)

    return empty


def get_player_recent_form_cached(mlb_id: int) -> dict:
    """Cached wrapper around get_player_recent_form() with 2-hour TTL.

    Uses st.session_state for caching to avoid repeated API calls.
    Imports streamlit lazily to avoid import errors in non-Streamlit
    contexts (tests, CLI scripts). Falls back to uncached fetch if
    streamlit is unavailable.

    Args:
        mlb_id: The player's MLB Stats API ID.

    Returns:
        Same structure as get_player_recent_form().
    """
    try:
        import streamlit as st

        cache_key = f"_recent_form_{mlb_id}"
        cached = st.session_state.get(cache_key)
        if cached and (time.time() - cached["ts"]) < 7200:  # 2-hour TTL
            return cached["data"]
        data = get_player_recent_form(mlb_id)
        st.session_state[cache_key] = {"data": data, "ts": time.time()}
        return data
    except ImportError:
        return get_player_recent_form(mlb_id)


def fetch_pitcher_pitch_mix(mlb_id: int, season: int = 2026) -> dict[str, float]:
    """Fetch pitch mix percentages for a pitcher from Baseball Savant.

    Uses pybaseball's statcast_pitcher endpoint to retrieve the last 30 days
    of pitch data, then computes percentage breakdown by pitch type.

    Args:
        mlb_id: MLB player ID (integer).
        season: MLB season year (default 2026).

    Returns:
        Dict mapping pitch type codes to usage percentages, e.g.
        ``{"FF": 45.2, "SL": 22.1, "CH": 18.5, "CU": 14.2}``.
        Returns empty dict if pybaseball is unavailable or data cannot
        be fetched.
    """
    # Check session_state cache first (24h TTL)
    cache_key = f"_pitch_mix_{mlb_id}_{season}"
    try:
        import streamlit as st

        cached = st.session_state.get(cache_key)
        if cached and (time.time() - cached["ts"]) < 86400:  # 24-hour TTL
            return cached["data"]
    except ImportError:
        st = None  # type: ignore[assignment]

    try:
        from pybaseball import statcast_pitcher
    except ImportError:
        logger.debug("pybaseball not available; cannot fetch pitch mix")
        return {}

    from datetime import timedelta

    today = datetime.now(UTC).date()
    start_date = today - timedelta(days=30)

    try:
        data = statcast_pitcher(
            str(start_date),
            str(today),
            mlb_id,
        )
    except Exception:
        logger.debug("statcast_pitcher fetch failed for mlb_id=%d", mlb_id)
        return {}

    if data is None or data.empty:
        logger.debug("No statcast pitch data for mlb_id=%d", mlb_id)
        return {}

    # Filter to valid pitch types
    if "pitch_type" not in data.columns:
        return {}

    pitch_counts = data["pitch_type"].dropna().value_counts()
    total = pitch_counts.sum()
    if total == 0:
        return {}

    result: dict[str, float] = {}
    for pitch_type, count in pitch_counts.items():
        pct = round((count / total) * 100, 1)
        if pct > 0:
            result[str(pitch_type)] = pct

    # Cache result
    try:
        if st is not None:
            st.session_state[cache_key] = {"data": result, "ts": time.time()}
    except Exception:
        pass

    logger.debug("Pitch mix for mlb_id=%d: %s", mlb_id, result)
    return result


def get_todays_lineups(schedule: list[dict]) -> dict:
    """Fetch confirmed starting lineups for today's games.

    Uses the MLB Stats API boxscore endpoint to retrieve batting orders
    for each game. Lineups are typically posted 1-2 hours before game time;
    if a lineup has not been posted yet the game entry will have an empty list.

    Args:
        schedule: List of game dicts from ``statsapi.schedule()`` with
            ``game_pk`` keys.

    Returns:
        Dict mapping team_abbr -> list of player name strings in batting
        order. Teams without confirmed lineups are omitted.
    """
    if _statsapi is None:
        logger.warning("statsapi not available; cannot fetch lineups")
        return {}

    # Full-name to abbreviation mapping (reused from weather)
    _NAME_TO_ABBR: dict[str, str] = {
        "Arizona Diamondbacks": "ARI",
        "Atlanta Braves": "ATL",
        "Baltimore Orioles": "BAL",
        "Boston Red Sox": "BOS",
        "Chicago Cubs": "CHC",
        "Chicago White Sox": "CWS",
        "Cincinnati Reds": "CIN",
        "Cleveland Guardians": "CLE",
        "Colorado Rockies": "COL",
        "Detroit Tigers": "DET",
        "Houston Astros": "HOU",
        "Kansas City Royals": "KC",
        "Los Angeles Angels": "LAA",
        "Los Angeles Dodgers": "LAD",
        "Miami Marlins": "MIA",
        "Milwaukee Brewers": "MIL",
        "Minnesota Twins": "MIN",
        "New York Mets": "NYM",
        "New York Yankees": "NYY",
        "Oakland Athletics": "ATH",
        "Athletics": "ATH",
        "Philadelphia Phillies": "PHI",
        "Pittsburgh Pirates": "PIT",
        "San Diego Padres": "SD",
        "San Francisco Giants": "SF",
        "Seattle Mariners": "SEA",
        "St. Louis Cardinals": "STL",
        "Tampa Bay Rays": "TB",
        "Texas Rangers": "TEX",
        "Toronto Blue Jays": "TOR",
        "Washington Nationals": "WSH",
    }

    result: dict[str, list[str]] = {}

    for game in schedule:
        game_pk = game.get("game_id") or game.get("game_pk") or game.get("gamePk", 0)
        if not game_pk:
            continue

        try:
            box = _statsapi.boxscore_data(game_pk)
        except Exception:
            logger.debug("Boxscore fetch failed for game_pk=%s", game_pk)
            continue

        for side in ("home", "away"):
            side_data = box.get(side, {}) if isinstance(box, dict) else {}
            team_name = game.get(f"{side}_name", "")
            abbr = _NAME_TO_ABBR.get(team_name, "")

            # Extract batting order — boxscore_data returns battingOrder as list
            # of player ID integers, and playerInfo with player details
            batting_order = side_data.get("battingOrder", [])
            if not batting_order:
                continue

            player_info = side_data.get("players", {})
            names: list[str] = []
            for pid in batting_order:
                # boxscore_data uses "ID{number}" keys in players dict
                pkey = f"ID{pid}"
                pdata = player_info.get(pkey, {})
                full_name = pdata.get("fullName", "")
                if not full_name:
                    # Try person sub-dict
                    person = pdata.get("person", {})
                    full_name = person.get("fullName", f"Unknown ({pid})")
                names.append(full_name)

            if names and abbr:
                result[abbr] = names

        time.sleep(0.3)

    logger.info("Fetched lineups for %d teams across %d games", len(result), len(schedule))
    return result
