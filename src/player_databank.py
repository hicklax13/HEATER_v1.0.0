"""Player Databank: game log fetching, loading, and stat view configuration.

Provides per-game stat storage, retrieval, and the STAT_VIEW_OPTIONS /
STAT_VIEW_PARAMS dictionaries that drive the Player Databank UI dropdowns.
"""

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.database import get_connection

logger = logging.getLogger(__name__)

try:
    import statsapi
except ImportError:
    statsapi = None  # type: ignore[assignment]

# ── Fantasy category column groups ───────────────────────────────────────────

HITTING_CATS = ["R", "HR", "RBI", "SB", "AVG", "OBP"]
PITCHING_CATS = ["W", "L", "SV", "K", "ERA", "WHIP"]

HITTING_COLS_TOTAL = ["h", "ab", "r", "hr", "rbi", "sb", "pa", "bb", "hbp", "sf"]
PITCHING_COLS_TOTAL = ["ip", "w", "l", "sv", "k", "er", "bb_allowed", "h_allowed"]

# ── Stat view options (28 keys) ───────────────────────────────────────────────

STAT_VIEW_OPTIONS: dict[str, str] = {
    "S_PS7": "Next 7 Days (proj)",
    "S_PS14": "Next 14 Days (proj)",
    "S_PSR": "Remaining Games (proj)",
    "S_L": "Today (live)",
    "S_L7": "Last 7 Days (total)",
    "S_L14": "Last 14 Days (total)",
    "S_L30": "Last 30 Days (total)",
    "S_S_2026": "Season (total)",
    "S_S_2025": "2025 Season (total)",
    "S_S_2024": "2024 Season (total)",
    "ADVST_ADVST_2026": "2026 Advanced",
    "ADVST_ADVST_2025": "2025 Advanced",
    "S_AL7": "Last 7 Days (avg)",
    "S_AL14": "Last 14 Days (avg)",
    "S_AL30": "Last 30 Days (avg)",
    "S_AS_2026": "Season (avg)",
    "S_AS_2025": "2025 Season (avg)",
    "S_AS_2024": "2024 Season (avg)",
    "S_SDL7": "Last 7 Days (std dev)",
    "S_SDL14": "Last 14 Days (std dev)",
    "S_SDL30": "Last 30 Days (std dev)",
    "S_SD_2026": "Season (std dev)",
    "S_SD_2025": "2025 Season (std dev)",
    "S_SD_2024": "2024 Season (std dev)",
    "K_K": "Ranks",
    "R_O": "Research",
    "M_W": "Fantasy Matchups",
    "O_O": "Opponents",
}

# ── Stat view computation parameters ─────────────────────────────────────────
# Each entry maps a STAT_VIEW_OPTIONS key to its computation parameters:
#   type  : "proj" | "live" | "total" | "avg" | "stddev" | "special"
#   days  : number of days back (None = entire season or special handling)
#   season: integer season year (None = current / multi-season)

STAT_VIEW_PARAMS: dict[str, dict] = {
    "S_PS7": {"type": "proj", "days": 7, "season": None},
    "S_PS14": {"type": "proj", "days": 14, "season": None},
    "S_PSR": {"type": "proj", "days": None, "season": None},
    "S_L": {"type": "live", "days": 0, "season": None},
    "S_L7": {"type": "total", "days": 7, "season": None},
    "S_L14": {"type": "total", "days": 14, "season": None},
    "S_L30": {"type": "total", "days": 30, "season": None},
    "S_S_2026": {"type": "total", "days": None, "season": 2026},
    "S_S_2025": {"type": "total", "days": None, "season": 2025},
    "S_S_2024": {"type": "total", "days": None, "season": 2024},
    "ADVST_ADVST_2026": {"type": "advanced", "days": None, "season": 2026},
    "ADVST_ADVST_2025": {"type": "advanced", "days": None, "season": 2025},
    "S_AL7": {"type": "avg", "days": 7, "season": None},
    "S_AL14": {"type": "avg", "days": 14, "season": None},
    "S_AL30": {"type": "avg", "days": 30, "season": None},
    "S_AS_2026": {"type": "avg", "days": None, "season": 2026},
    "S_AS_2025": {"type": "avg", "days": None, "season": 2025},
    "S_AS_2024": {"type": "avg", "days": None, "season": 2024},
    "S_SDL7": {"type": "stddev", "days": 7, "season": None},
    "S_SDL14": {"type": "stddev", "days": 14, "season": None},
    "S_SDL30": {"type": "stddev", "days": 30, "season": None},
    "S_SD_2026": {"type": "stddev", "days": None, "season": 2026},
    "S_SD_2025": {"type": "stddev", "days": None, "season": 2025},
    "S_SD_2024": {"type": "stddev", "days": None, "season": 2024},
    "K_K": {"type": "special", "days": None, "season": None},
    "R_O": {"type": "special", "days": None, "season": None},
    "M_W": {"type": "special", "days": None, "season": None},
    "O_O": {"type": "special", "days": None, "season": None},
}


# ── Game log loading ──────────────────────────────────────────────────────────


def load_game_logs(
    player_ids: list[int],
    days: int | None = None,
    season: int | None = None,
) -> pd.DataFrame:
    """Load game logs from the local SQLite database.

    Args:
        player_ids: List of player IDs to query.
        days: If provided, restrict to games within the last N days from today.
              If None, no date-range filter is applied (use ``season`` to
              restrict by year instead).
        season: If provided, restrict to games for this season year.
                If None, no season filter is applied.

    Returns:
        DataFrame of matching game log rows. Empty DataFrame if no matches or
        if ``player_ids`` is empty.
    """
    if not player_ids:
        return pd.DataFrame()

    conn = get_connection()
    try:
        placeholders = ",".join("?" * len(player_ids))
        params: list = list(player_ids)
        where_clauses = [f"player_id IN ({placeholders})"]

        if season is not None:
            where_clauses.append("season = ?")
            params.append(season)

        if days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
            where_clauses.append("game_date >= ?")
            params.append(cutoff)

        where_sql = " AND ".join(where_clauses)
        sql = f"SELECT * FROM game_logs WHERE {where_sql} ORDER BY game_date DESC"

        df = pd.read_sql_query(sql, conn, params=params)
        return df
    except Exception:
        logger.exception("Failed to load game logs for player_ids=%s", player_ids)
        return pd.DataFrame()
    finally:
        conn.close()


# ── Game log fetching from MLB Stats API ──────────────────────────────────────


def fetch_game_logs_from_api(
    season: int = 2026,
    force: bool = False,
) -> int:
    """Fetch per-game stats from the MLB Stats API and persist to SQLite.

    Iterates through all players in the ``players`` table that have a non-null
    ``mlb_id``, calls ``statsapi.player_stat_data()`` for each, and stores the
    results into the ``game_logs`` table via INSERT OR REPLACE.

    Args:
        season: The MLB season year to fetch game logs for.
        force: If True, re-fetch even if records already exist. (Currently
               INSERT OR REPLACE always overwrites; this flag is reserved for
               future staleness checks.)

    Returns:
        Total number of game-log rows inserted / replaced across all players.
    """
    if statsapi is None:
        logger.warning("statsapi not installed — skipping fetch_game_logs_from_api")
        return 0

    # Load all players with a known MLB ID from the DB
    conn = get_connection()
    try:
        players_df = pd.read_sql_query(
            "SELECT player_id, mlb_id, is_hitter FROM players WHERE mlb_id IS NOT NULL",
            conn,
        )
    except Exception:
        logger.exception("Failed to read players table for game log fetch")
        return 0
    finally:
        conn.close()

    if players_df.empty:
        logger.info("No players with mlb_id found; skipping game log fetch")
        return 0

    total_inserted = 0

    for _, row in players_df.iterrows():
        player_id = int(row["player_id"])
        mlb_id = int(row["mlb_id"])
        is_hitter = int(row.get("is_hitter", 1))

        try:
            inserted = _fetch_and_store_player_logs(
                player_id=player_id,
                mlb_id=mlb_id,
                is_hitter=is_hitter,
                season=season,
            )
            total_inserted += inserted
        except Exception:
            logger.warning(
                "Error fetching game logs for player_id=%d mlb_id=%d",
                player_id,
                mlb_id,
                exc_info=True,
            )

    logger.info(
        "fetch_game_logs_from_api: inserted/replaced %d rows for season %d",
        total_inserted,
        season,
    )
    return total_inserted


def _fetch_and_store_player_logs(
    player_id: int,
    mlb_id: int,
    is_hitter: int,
    season: int,
) -> int:
    """Fetch game logs for a single player and write to game_logs table.

    Tries the appropriate stat group first (hitting for hitters, pitching for
    pitchers). Falls back to the other group if the primary returns no data.

    Args:
        player_id: Internal DB player ID (FK into ``players`` table).
        mlb_id: MLB Stats API player ID.
        is_hitter: 1 if the player is primarily a hitter, 0 for pitcher.
        season: Season year.

    Returns:
        Number of rows inserted.
    """
    groups = ["hitting", "pitching"] if is_hitter else ["pitching", "hitting"]
    rows: list[dict] = []

    for group in groups:
        try:
            result = statsapi.player_stat_data(
                mlb_id,
                group=group,
                type="gameLog",
                sportId=1,
            )
            entries = result.get("stats", [])
            if not entries:
                continue

            for entry in entries:
                game_date = entry.get("date", "")
                if not game_date:
                    continue

                raw = entry.get("stats", entry)
                row = _parse_game_log_row(player_id, game_date, season, group, raw)
                rows.append(row)

            # If we found data in this group, don't try the other
            break
        except Exception:
            logger.debug("No %s gameLog for mlb_id=%d", group, mlb_id, exc_info=False)

    if not rows:
        return 0

    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT OR REPLACE INTO game_logs
                (player_id, game_date, season,
                 pa, ab, h, r, hr, rbi, sb, bb, hbp, sf,
                 ip, w, l, sv, k, er, bb_allowed, h_allowed)
            VALUES
                (:player_id, :game_date, :season,
                 :pa, :ab, :h, :r, :hr, :rbi, :sb, :bb, :hbp, :sf,
                 :ip, :w, :l, :sv, :k, :er, :bb_allowed, :h_allowed)
            """,
            rows,
        )
        conn.commit()
        return len(rows)
    except Exception:
        logger.exception("Failed to store game logs for player_id=%d", player_id)
        return 0
    finally:
        conn.close()


def _parse_game_log_row(
    player_id: int,
    game_date: str,
    season: int,
    group: str,
    raw: dict,
) -> dict:
    """Convert a raw statsapi game-log entry into a game_logs table row dict.

    Args:
        player_id: Internal DB player ID.
        game_date: ISO date string (YYYY-MM-DD).
        season: Season year integer.
        group: "hitting" or "pitching".
        raw: The ``stats`` sub-dict from the statsapi response entry.

    Returns:
        Dict with all game_logs column keys populated (zero-defaults for
        columns from the other stat group).
    """

    def _int(key: str) -> int:
        try:
            return int(raw.get(key, 0) or 0)
        except (ValueError, TypeError):
            return 0

    def _float(key: str) -> float:
        try:
            val = raw.get(key, 0) or 0
            # IP is sometimes stored as "2.1" (outs notation) but statsapi
            # returns decimal innings directly (e.g. 6.0 for 6 full innings).
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    base: dict = {
        "player_id": player_id,
        "game_date": game_date,
        "season": season,
        # Hitting
        "pa": 0,
        "ab": 0,
        "h": 0,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        # Pitching
        "ip": 0.0,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
    }

    if group == "hitting":
        base.update(
            {
                "pa": _int("plateAppearances"),
                "ab": _int("atBats"),
                "h": _int("hits"),
                "r": _int("runs"),
                "hr": _int("homeRuns"),
                "rbi": _int("rbi"),
                "sb": _int("stolenBases"),
                "bb": _int("baseOnBalls"),
                "hbp": _int("hitByPitch"),
                "sf": _int("sacFlies"),
            }
        )
    else:  # pitching
        base.update(
            {
                "ip": _float("inningsPitched"),
                "w": _int("wins"),
                "l": _int("losses"),
                "sv": _int("saves"),
                "k": _int("strikeOuts"),
                "er": _int("earnedRuns"),
                "bb_allowed": _int("baseOnBalls"),
                "h_allowed": _int("hits"),
            }
        )

    return base
