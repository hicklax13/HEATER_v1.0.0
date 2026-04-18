"""FanGraphs auto-fetch data pipeline.

Fetches Steamer, ZiPS, and Depth Charts projections from FanGraphs'
internal JSON API on app startup. Normalizes JSON to DB schema, upserts
players, stores projections, creates blended projections, and extracts ADP.

CSV upload remains available as a manual fallback.
"""

import logging
import time

import pandas as pd
import requests

from src.database import (
    check_staleness,
    create_blended_projections,
    get_connection,
    init_db,
    update_refresh_log,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

_BASE_URL = "https://www.fangraphs.com/api/projections"
_TIMEOUT = 15  # seconds per request
_RATE_LIMIT = 1.5  # seconds between requests (more conservative to avoid 403)

# Persistent session with browser-like headers to avoid FanGraphs bot detection.
# FanGraphs checks User-Agent, Referer, and Accept headers; missing any of
# these triggers a 403 Forbidden.  A Session also persists cookies across
# requests, which some CDN/WAF layers require.
_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.fangraphs.com/projections",
        "Origin": "https://www.fangraphs.com",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
)

# FG API type parameter → DB system column value
SYSTEM_MAP = {
    "steamer": "steamer",
    "zips": "zips",
    "fangraphsdc": "depthcharts",
    "atc": "atc",
    "thebat": "thebat",
    "thebatx": "thebatx",
}

SYSTEMS = list(SYSTEM_MAP.keys())

# Rest-of-season variants — updated daily by FanGraphs during the season.
# These provide ROS projections that account for current-season performance.
ROS_SYSTEM_MAP = {
    "steamerr": "steamer_ros",
    "rzips": "zips_ros",
    "rfangraphsdc": "depthcharts_ros",
}
ROS_SYSTEMS = list(ROS_SYSTEM_MAP.keys())


class FetchError(Exception):
    """Raised when an API fetch fails."""


# ── Normalization ──────────────────────────────────────────────────


def normalize_hitter_json(raw: list[dict]) -> pd.DataFrame:
    """Map FanGraphs JSON hitter fields to DB schema.

    Sets is_hitter=True. Extracts primary position from 'minpos' field.
    NOTE: FG 'minpos' provides primary position only (e.g., "SS"), not
    multi-position eligibility. This is a known limitation vs CSV import.
    """
    records = []
    for player in raw:
        raw_pos = (player.get("minpos") or "").strip()
        pos = raw_pos if raw_pos and raw_pos not in ("0", "-") else "Util"
        records.append(
            {
                "name": player.get("PlayerName", ""),
                "team": player.get("Team", ""),
                "positions": pos,
                "is_hitter": True,
                "pa": int(player.get("PA", 0) or 0),
                "ab": int(player.get("AB", 0) or 0),
                "h": int(player.get("H", 0) or 0),
                "r": int(player.get("R", 0) or 0),
                "hr": int(player.get("HR", 0) or 0),
                "rbi": int(player.get("RBI", 0) or 0),
                "sb": int(player.get("SB", 0) or 0),
                "avg": float(player.get("AVG", 0) or 0),
                "obp": float(player.get("OBP", 0) or 0),
                "bb": int(player.get("BB", 0) or 0),
                "hbp": int(player.get("HBP", 0) or 0),
                "sf": int(player.get("SF", 0) or 0),
            }
        )
    return pd.DataFrame(records)


def normalize_pitcher_json(raw: list[dict]) -> pd.DataFrame:
    """Map FanGraphs JSON pitcher fields to DB schema.

    Sets is_hitter=False. Classifies SP/RP using existing logic from
    import_pitcher_csv(): GS >= 5 or IP >= 80 → "SP", SV >= 3 → "RP",
    GS >= 1 → "SP,RP", else → "RP".
    """
    records = []
    for player in raw:
        ip = float(player.get("IP", 0) or 0)
        gs = int(player.get("GS", 0) or 0)
        sv = int(player.get("SV", 0) or 0)

        # SP/RP classification (mirrors database.py import_pitcher_csv)
        if gs >= 5 or ip >= 80:
            positions = "SP"
        elif sv >= 3:
            positions = "RP"
        elif gs >= 1:
            positions = "SP,RP"
        else:
            positions = "RP"

        er = int(player.get("ER", 0) or 0)
        era = float(player.get("ERA", 0) or 0)
        bb_allowed = int(player.get("BB", 0) or 0)
        h_allowed = int(player.get("H", 0) or 0)
        whip = float(player.get("WHIP", 0) or 0)

        # Compute ER from ERA if not available (mirrors database.py)
        if er == 0 and era > 0 and ip > 0:
            er = int(round(era * ip / 9))
        # Compute H/BB from WHIP if both missing
        if h_allowed == 0 and bb_allowed == 0 and whip > 0 and ip > 0:
            total = whip * ip
            bb_allowed = int(total * 0.3)
            h_allowed = int(total * 0.7)

        records.append(
            {
                "name": player.get("PlayerName", ""),
                "team": player.get("Team", ""),
                "positions": positions,
                "is_hitter": False,
                "ip": ip,
                "w": int(player.get("W", 0) or 0),
                "l": int(player.get("L", 0) or 0),
                "sv": sv,
                "k": int(player.get("SO", 0) or 0),
                "era": era,
                "whip": whip,
                "er": er,
                "bb_allowed": bb_allowed,
                "h_allowed": h_allowed,
                "fip": float(player.get("FIP", 0) or 0),
                "xfip": float(player.get("xFIP", 0) or 0),
                "siera": float(player.get("SIERA", 0) or 0),
            }
        )
    return pd.DataFrame(records)


# ── Fetching ──────────────────────────────────────────────────────


def fetch_projections(system: str, stats: str) -> tuple[pd.DataFrame, list[dict]]:
    """Single API call to FanGraphs projections endpoint.

    Args:
        system: FG API type — "steamer", "zips", or "fangraphsdc"
        stats: "bat" or "pit"

    Returns:
        Tuple of (normalized DataFrame, raw JSON list).
        The raw JSON is preserved for ADP extraction.

    Raises:
        FetchError: On network or parse failure.
    """
    params = {
        "type": system,
        "stats": stats,
        "pos": "all",
        "team": "0",
        "lg": "all",
        "players": "0",
    }

    # Warm the session with a page visit on first request (gets cookies)
    if not getattr(_SESSION, "_fg_warmed", False):
        try:
            _SESSION.get(
                "https://www.fangraphs.com/projections",
                timeout=_TIMEOUT,
                allow_redirects=True,
            )
        except requests.exceptions.RequestException:
            pass  # Non-fatal — API call may still work
        _SESSION._fg_warmed = True  # type: ignore[attr-defined]

    # Retry up to 2 times with increasing delay
    last_exc = None
    for attempt in range(3):
        try:
            resp = _SESSION.get(_BASE_URL, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            raw = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(_RATE_LIMIT * (attempt + 1))
        except ValueError as exc:
            raise FetchError(f"Invalid JSON from {system}/{stats}: {exc}") from exc
    else:
        # JSON API failed (likely 403) — try HTML __NEXT_DATA__ extraction
        try:
            logger.info("JSON API failed for %s/%s, trying HTML extraction...", system, stats)
            time.sleep(_RATE_LIMIT)
            return _fetch_fg_html_projections(system, stats)
        except FetchError:
            pass
        raise FetchError(f"Failed to fetch {system}/{stats}: {last_exc}") from last_exc

    if stats == "bat":
        return normalize_hitter_json(raw), raw
    else:
        return normalize_pitcher_json(raw), raw


def _fetch_fg_html_projections(system: str, stats: str) -> tuple[pd.DataFrame, list[dict]]:
    """Fallback: extract projections from FanGraphs HTML __NEXT_DATA__ tag.

    The JSON API (/api/projections) returns 403 due to Cloudflare WAF,
    but HTML pages return 200 with full projection data embedded in
    Next.js server-side props as ``<script id="__NEXT_DATA__">``.

    Returns same format as :func:`fetch_projections`.
    """
    import json as _json
    import re as _re

    url = f"https://www.fangraphs.com/projections?type={system}&stats={stats}&pos=all&team=0&lg=all&players=0"
    try:
        resp = _SESSION.get(url, timeout=20)
        resp.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise FetchError(f"HTML fetch failed for {system}/{stats}: {exc}") from exc

    match = _re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text)
    if not match:
        raise FetchError(f"No __NEXT_DATA__ found for {system}/{stats}")

    try:
        nd = _json.loads(match.group(1))
        raw = nd["props"]["pageProps"]["dehydratedState"]["queries"][0]["state"]["data"]
    except (KeyError, IndexError, _json.JSONDecodeError) as exc:
        raise FetchError(f"Failed to parse __NEXT_DATA__ for {system}/{stats}: {exc}") from exc

    if not raw:
        raise FetchError(f"Empty data in __NEXT_DATA__ for {system}/{stats}")

    # Normalize column names to match the JSON API format expected by
    # normalize_hitter_json / normalize_pitcher_json
    for player in raw:
        if "ShortName" in player and "PlayerName" not in player:
            player["PlayerName"] = player["ShortName"]
        if "SO" in player and "K" not in player:
            player["K"] = player["SO"]
        # Ensure minpos exists for hitters (position detection)
        if stats == "bat" and "minpos" not in player:
            player["minpos"] = player.get("Pos", "Util")

    logger.info("HTML extraction: %d %s from %s", len(raw), stats, system)

    if stats == "bat":
        return normalize_hitter_json(raw), raw
    else:
        return normalize_pitcher_json(raw), raw


def extract_adp(hitters_raw: list[dict], pitchers_raw: list[dict]) -> pd.DataFrame:
    """Pull ADP from raw FanGraphs JSON (Steamer responses are most complete).

    Filters out ADP >= 999 and null values.
    Returns DataFrame with columns: name, adp.
    The caller (_store_adp) resolves name → player_id before DB insert.
    """
    records = []
    for player in hitters_raw + pitchers_raw:
        name = player.get("PlayerName", "")
        adp = player.get("ADP")
        if adp is None or adp >= 999:
            continue
        try:
            adp_val = float(adp)
        except (ValueError, TypeError):
            continue
        records.append({"name": name, "adp": adp_val})

    return pd.DataFrame(records) if records else pd.DataFrame(columns=["name", "adp"])


# ── Storage ────────────────────────────────────────────────────────


def _upsert_player(cursor, name: str, team: str, positions: str, is_hitter: bool) -> int:
    """Insert or find a player. Returns player_id.

    Mirrors database.py's _upsert_player() — tries name+team first,
    then falls back to name-only to prevent duplicates from team mismatches
    between MLB Stats API and FanGraphs.
    """
    # Try exact match: name + team
    cursor.execute(
        "SELECT player_id, positions FROM players WHERE name = ? AND team = ?",
        (name, team),
    )
    result = cursor.fetchone()

    # Fallback: name-only match (prevents duplicates from team mismatches)
    if result is None and name:
        cursor.execute(
            "SELECT player_id, positions FROM players WHERE name = ?",
            (name,),
        )
        result = cursor.fetchone()

    if result:
        existing = set(result[1].split(","))
        new = set(positions.split(","))
        merged = ",".join(sorted(existing | new))
        if merged != result[1]:
            cursor.execute(
                "UPDATE players SET positions = ? WHERE player_id = ?",
                (merged, result[0]),
            )
        # Update team if canonical entry has empty team
        if team:
            cursor.execute(
                "UPDATE players SET team = ? WHERE player_id = ? AND (team IS NULL OR team = '')",
                (team, result[0]),
            )
        return result[0]
    else:
        cursor.execute(
            "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
            (name, team, positions, 1 if is_hitter else 0),
        )
        return cursor.lastrowid


def _store_projections(projections: dict[str, pd.DataFrame]) -> int:
    """Upsert players then store projections in DB.

    Keys are "{db_system}_{bat|pit}" e.g. "steamer_bat", "depthcharts_pit".
    For each system: DELETE existing rows → upsert players → INSERT projections.
    Returns total row count inserted.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        total = 0

        # Collect all unique systems being stored
        systems_stored = set()
        for key in projections:
            db_system = key.rsplit("_", 1)[0]  # "steamer_bat" → "steamer"
            systems_stored.add(db_system)

        # Delete old projections for each system (idempotency)
        for system in systems_stored:
            cursor.execute("DELETE FROM projections WHERE system = ?", (system,))

        # Insert new projections
        for key, df in projections.items():
            db_system = key.rsplit("_", 1)[0]
            for _, row in df.iterrows():
                is_hitter = bool(row.get("is_hitter", True))
                player_id = _upsert_player(
                    cursor,
                    str(row["name"]),
                    str(row.get("team", "")),
                    str(row.get("positions", "Util")),
                    is_hitter,
                )

                # Check if a row already exists for this (player_id, system)
                # — handles two-way players (e.g. Ohtani) appearing in both bat and pit
                cursor.execute(
                    "SELECT id FROM projections WHERE player_id = ? AND system = ?",
                    (player_id, db_system),
                )
                existing_row = cursor.fetchone()

                if is_hitter:
                    if existing_row:
                        # Merge hitting stats into existing pitching row
                        cursor.execute(
                            """UPDATE projections
                               SET pa = ?, ab = ?, h = ?, r = ?, hr = ?, rbi = ?,
                                   sb = ?, avg = ?, obp = ?, bb = ?, hbp = ?, sf = ?
                               WHERE id = ?""",
                            (
                                int(row.get("pa", 0)),
                                int(row.get("ab", 0)),
                                int(row.get("h", 0)),
                                int(row.get("r", 0)),
                                int(row.get("hr", 0)),
                                int(row.get("rbi", 0)),
                                int(row.get("sb", 0)),
                                float(row.get("avg", 0)),
                                float(row.get("obp", 0)),
                                int(row.get("bb", 0)),
                                int(row.get("hbp", 0)),
                                int(row.get("sf", 0)),
                                existing_row[0],
                            ),
                        )
                    else:
                        cursor.execute(
                            """INSERT INTO projections
                               (player_id, system, pa, ab, h, r, hr, rbi, sb, avg,
                                obp, bb, hbp, sf)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                player_id,
                                db_system,
                                int(row.get("pa", 0)),
                                int(row.get("ab", 0)),
                                int(row.get("h", 0)),
                                int(row.get("r", 0)),
                                int(row.get("hr", 0)),
                                int(row.get("rbi", 0)),
                                int(row.get("sb", 0)),
                                float(row.get("avg", 0)),
                                float(row.get("obp", 0)),
                                int(row.get("bb", 0)),
                                int(row.get("hbp", 0)),
                                int(row.get("sf", 0)),
                            ),
                        )
                else:
                    if existing_row:
                        # Merge pitching stats into existing hitting row
                        cursor.execute(
                            """UPDATE projections
                               SET ip = ?, w = ?, l = ?, sv = ?, k = ?, era = ?,
                                   whip = ?, er = ?, bb_allowed = ?, h_allowed = ?,
                                   fip = ?, xfip = ?, siera = ?
                               WHERE id = ?""",
                            (
                                float(row.get("ip", 0)),
                                int(row.get("w", 0)),
                                int(row.get("l", 0)),
                                int(row.get("sv", 0)),
                                int(row.get("k", 0)),
                                float(row.get("era", 0)),
                                float(row.get("whip", 0)),
                                int(row.get("er", 0)),
                                int(row.get("bb_allowed", 0)),
                                int(row.get("h_allowed", 0)),
                                float(row.get("fip", 0)),
                                float(row.get("xfip", 0)),
                                float(row.get("siera", 0)),
                                existing_row[0],
                            ),
                        )
                    else:
                        cursor.execute(
                            """INSERT INTO projections
                               (player_id, system, ip, w, l, sv, k, era, whip, er,
                                bb_allowed, h_allowed, fip, xfip, siera)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                player_id,
                                db_system,
                                float(row.get("ip", 0)),
                                int(row.get("w", 0)),
                                int(row.get("l", 0)),
                                int(row.get("sv", 0)),
                                int(row.get("k", 0)),
                                float(row.get("era", 0)),
                                float(row.get("whip", 0)),
                                int(row.get("er", 0)),
                                int(row.get("bb_allowed", 0)),
                                int(row.get("h_allowed", 0)),
                                float(row.get("fip", 0)),
                                float(row.get("xfip", 0)),
                                float(row.get("siera", 0)),
                            ),
                        )
                total += 1

        conn.commit()
        return total
    except Exception:
        logger.exception("Failed to store projections")
        raise
    finally:
        conn.close()


def _update_fangraphs_ids(raw_data: dict[str, list[dict]]) -> int:
    """Update fangraphs_id on players table from raw FanGraphs JSON.

    The FanGraphs JSON includes a 'playerid' field (numeric) for each player.
    Match by name to the players table and set fangraphs_id = playerid.

    Args:
        raw_data: Dict keyed by "{db_system}_{stats}" containing raw JSON lists.

    Returns:
        Number of players updated.
    """
    # Collect unique (name, fg_id) pairs from all raw data
    fg_ids: dict[str, str] = {}
    for _key, records in raw_data.items():
        for player in records:
            name = player.get("PlayerName", "")
            fg_id = player.get("playerid")
            if name and fg_id is not None:
                fg_ids[name] = str(fg_id)

    if not fg_ids:
        return 0

    conn = get_connection()
    try:
        cursor = conn.cursor()
        updated = 0
        for name, fg_id in fg_ids.items():
            # Exact name match first
            cursor.execute(
                "SELECT player_id FROM players WHERE name = ?",
                (name,),
            )
            result = cursor.fetchone()
            if result:
                cursor.execute(
                    "UPDATE players SET fangraphs_id = ? WHERE player_id = ?",
                    (fg_id, result[0]),
                )
                updated += 1
        conn.commit()
        logger.info("Updated fangraphs_id for %d players", updated)
        return updated
    except Exception:
        logger.exception("Failed to update fangraphs_ids")
        return 0
    finally:
        conn.close()


def _store_adp(adp_df: pd.DataFrame) -> int:
    """Resolve player names to player_ids and store ADP.

    Uses exact-match against players table (name field).
    Falls back to fuzzy match (first + last name LIKE) for unresolved names.
    Deletes only FanGraphs-sourced ADP entries (by player_id) before INSERT.
    Preserves Yahoo ADP and other sources.
    Returns row count inserted.
    """
    if adp_df.empty:
        return 0

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Resolve all player_ids first so we only delete entries we're replacing
        records = []
        for _, row in adp_df.iterrows():
            name = str(row["name"])
            adp_val = float(row["adp"])

            cursor.execute("SELECT player_id FROM players WHERE name = ?", (name,))
            result = cursor.fetchone()
            if result is None:
                parts = name.split()
                if len(parts) >= 2:
                    cursor.execute(
                        "SELECT player_id FROM players WHERE name LIKE ? AND name LIKE ?",
                        (f"%{parts[0]}%", f"%{parts[-1]}%"),
                    )
                    matches = cursor.fetchall()
                    if len(matches) == 1:
                        result = matches[0]
                    elif len(matches) > 1:
                        logger.warning(
                            "ADP: fuzzy match for '%s' returned %d players, skipping",
                            name,
                            len(matches),
                        )
            if result is None:
                logger.warning("ADP: no player_id found for '%s', skipping", name)
                continue
            records.append({"player_id": result[0], "adp": adp_val})

        # Delete only entries for players in the new FanGraphs batch
        new_ids = [rec["player_id"] for rec in records]
        if new_ids:
            placeholders = ",".join("?" * len(new_ids))
            cursor.execute(f"DELETE FROM adp WHERE player_id IN ({placeholders})", new_ids)

        count = 0
        for rec in records:
            cursor.execute(
                """INSERT INTO adp (player_id, adp) VALUES (?, ?)
                   ON CONFLICT(player_id) DO UPDATE SET adp=excluded.adp""",
                (rec["player_id"], rec["adp"]),
            )
            count += 1

        conn.commit()
        return count
    finally:
        conn.close()


# ── Orchestration ──────────────────────────────────────────────────


def fetch_all_projections() -> tuple[dict[str, pd.DataFrame], dict[str, list[dict]]]:
    """Fetch all 6 endpoints (3 systems × bat/pit).

    Returns:
        Tuple of (projections_dict, raw_json_dict).
        projections_dict: keyed by "{db_system}_{stats}" e.g. "steamer_bat"
        raw_json_dict: same keys, contains raw JSON for ADP extraction
    """
    projections: dict[str, pd.DataFrame] = {}
    raw_data: dict[str, list[dict]] = {}
    first = True

    for fg_system in SYSTEMS:
        db_system = SYSTEM_MAP[fg_system]
        for stats in ("bat", "pit"):
            if not first:
                time.sleep(_RATE_LIMIT)
            first = False

            try:
                df, raw = fetch_projections(fg_system, stats)
                key = f"{db_system}_{stats}"
                projections[key] = df
                raw_data[key] = raw
                logger.info("Fetched %s/%s: %d players", fg_system, stats, len(df))
            except FetchError as exc:
                logger.warning("Failed to fetch %s/%s: %s", fg_system, stats, exc)

    # Fetch ROS projection variants (in-season, updated daily by FanGraphs)
    for fg_system in ROS_SYSTEMS:
        db_system = ROS_SYSTEM_MAP[fg_system]
        for stats in ("bat", "pit"):
            time.sleep(_RATE_LIMIT)
            try:
                df, raw = fetch_projections(fg_system, stats)
                key = f"{db_system}_{stats}"
                projections[key] = df
                raw_data[key] = raw
                logger.info("Fetched ROS %s/%s: %d rows", fg_system, stats, len(df))
            except FetchError:
                logger.warning(
                    "ROS fetch failed for %s/%s (may not be available yet)",
                    fg_system,
                    stats,
                )
            except Exception:
                logger.exception("Unexpected error fetching ROS %s/%s", fg_system, stats)

    return projections, raw_data


def refresh_if_stale(force: bool = False) -> bool:
    """Fetch projections if stale or missing.

    Calls init_db() to ensure tables exist.
    When force=False, skips fetch if projections were refreshed within 7 days.
    Orchestrates: fetch → normalize → upsert players → store projections
                  → blend → ADP → refresh log.
    Returns True if data was refreshed or already fresh,
    False if all fetches failed.
    """
    init_db()

    # Skip if data is fresh and not forcing refresh
    if not force:
        try:
            is_stale = check_staleness("projections", max_age_hours=24)
        except Exception:
            logger.exception("Staleness check failed, proceeding with fetch")
            is_stale = True
        if not is_stale:
            return True  # Data is fresh, skip fetch

    # Fetch from FanGraphs
    projections, raw_data = fetch_all_projections()
    if not projections:
        logger.error("All FanGraphs fetches failed")
        update_refresh_log("projections", status="error", message="all FanGraphs fetches failed")
        return False

    # Store projections (upserts players automatically)
    total = _store_projections(projections)
    logger.info("Stored %d projection rows", total)

    # Cross-reference FanGraphs IDs from raw JSON
    try:
        fg_count = _update_fangraphs_ids(raw_data)
        logger.info("Updated %d FanGraphs IDs", fg_count)
    except Exception as exc:
        logger.warning("FanGraphs ID update failed (non-fatal): %s", exc)

    # Create blended projections from all available systems
    try:
        create_blended_projections()
        logger.info("Blended projections created")
    except Exception as exc:
        logger.warning("Blended projection creation failed: %s", exc)

    # Extract and store ADP (prefer Steamer data)
    hitters_raw = raw_data.get("steamer_bat", [])
    pitchers_raw = raw_data.get("steamer_pit", [])
    if hitters_raw or pitchers_raw:
        adp_df = extract_adp(hitters_raw, pitchers_raw)
        adp_count = _store_adp(adp_df)
        logger.info("Stored %d ADP records", adp_count)

    # Log success
    update_refresh_log("projections", status="success")
    return True
