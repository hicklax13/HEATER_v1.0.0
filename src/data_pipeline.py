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
_TIMEOUT = 10  # seconds per request
_RATE_LIMIT = 0.5  # seconds between requests

# FG API type parameter → DB system column value
SYSTEM_MAP = {
    "steamer": "steamer",
    "zips": "zips",
    "fangraphsdc": "depthcharts",
}

SYSTEMS = list(SYSTEM_MAP.keys())


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
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }

    try:
        resp = requests.get(_BASE_URL, params=params, headers=headers, timeout=_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()
    except requests.exceptions.RequestException as exc:
        raise FetchError(f"Failed to fetch {system}/{stats}: {exc}") from exc
    except ValueError as exc:
        raise FetchError(f"Invalid JSON from {system}/{stats}: {exc}") from exc

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

                if is_hitter:
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
                    cursor.execute(
                        """INSERT INTO projections
                           (player_id, system, ip, w, l, sv, k, era, whip, er,
                            bb_allowed, h_allowed)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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


def _store_adp(adp_df: pd.DataFrame) -> int:
    """Resolve player names to player_ids and store ADP.

    Uses exact-match against players table (name field).
    Falls back to fuzzy match (first + last name LIKE) for unresolved names.
    DELETE FROM adp before INSERT (idempotency).
    Returns row count inserted.
    """
    if adp_df.empty:
        return 0

    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Clear existing ADP (idempotency)
        cursor.execute("DELETE FROM adp")

        count = 0
        for _, row in adp_df.iterrows():
            name = str(row["name"])
            adp_val = float(row["adp"])

            # Resolve name → player_id (exact match first, fuzzy fallback)
            cursor.execute("SELECT player_id FROM players WHERE name = ?", (name,))
            result = cursor.fetchone()
            if result is None:
                # Fuzzy fallback: first + last name LIKE match
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

            player_id = result[0]
            cursor.execute(
                """INSERT INTO adp (player_id, adp) VALUES (?, ?)
                   ON CONFLICT(player_id) DO UPDATE SET adp=excluded.adp""",
                (player_id, adp_val),
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
            is_stale = check_staleness("fangraphs_projections", max_age_hours=168)
        except Exception:
            logger.exception("Staleness check failed, proceeding with fetch")
            is_stale = True
        if not is_stale:
            return True  # Data is fresh, skip fetch

    # Fetch from FanGraphs
    projections, raw_data = fetch_all_projections()
    if not projections:
        logger.error("All FanGraphs fetches failed")
        update_refresh_log("fangraphs_projections", status="failed")
        return False

    # Store projections (upserts players automatically)
    total = _store_projections(projections)
    logger.info("Stored %d projection rows", total)

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
    update_refresh_log("fangraphs_projections", status="success")
    return True
