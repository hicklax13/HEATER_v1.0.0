"""Player Databank: game log fetching, loading, and stat view configuration.

Provides per-game stat storage, retrieval, and the STAT_VIEW_OPTIONS /
STAT_VIEW_PARAMS dictionaries that drive the Player Databank UI dropdowns.
"""

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd

from src.database import get_connection, load_player_pool

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


# ── Rolling stat computation ──────────────────────────────────────────────────


def compute_rolling_stats(
    player_ids: list[int],
    days: int | None = None,
    stat_type: str = "total",
    season: int = 2026,
) -> pd.DataFrame:
    """Compute rolling window stats from game logs.

    Args:
        player_ids: Player IDs to compute for.
        days: Number of days in the rolling window. None = entire season.
        stat_type: "total", "avg", or "stddev".
        season: Season year.

    Returns:
        DataFrame with one row per player, stat columns computed per stat_type.
        For "total" and "avg", computed rate stat columns are appended:
        ``avg_calc``, ``obp_calc``, ``era_calc``, ``whip_calc``.
    """
    logs = load_game_logs(player_ids, days=days, season=season)
    if logs.empty:
        return pd.DataFrame()

    # Determine numeric columns present in the result
    all_count_cols = HITTING_COLS_TOTAL + PITCHING_COLS_TOTAL
    num_cols = [c for c in all_count_cols if c in logs.columns]

    if stat_type == "stddev":
        result = logs.groupby("player_id")[num_cols].std(ddof=1).fillna(0).reset_index()
        return result

    # "total" and "avg" both start from sums
    sums = logs.groupby("player_id")[num_cols].sum()
    game_counts = logs.groupby("player_id").size().rename("_game_count")
    result = sums.join(game_counts).reset_index()

    if stat_type == "avg":
        for col in num_cols:
            result[col] = result[col] / result["_game_count"]

    result = result.drop(columns=["_game_count"])

    # Append computed rate stats (using weighted formulas per CLAUDE.md)
    def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
        """Divide num by den, returning NaN where denominator is zero."""
        return num.where(den != 0).div(den.where(den != 0))

    # AVG = sum(h) / sum(ab)
    if "h" in result.columns and "ab" in result.columns:
        result["avg_calc"] = _safe_div(result["h"], result["ab"])

    # OBP = sum(h + bb + hbp) / sum(ab + bb + hbp + sf)
    if all(c in result.columns for c in ["h", "bb", "hbp", "ab", "sf"]):
        obp_num = result["h"] + result["bb"] + result["hbp"]
        obp_den = result["ab"] + result["bb"] + result["hbp"] + result["sf"]
        result["obp_calc"] = _safe_div(obp_num, obp_den)

    # ERA = sum(er) * 9 / sum(ip)
    if "er" in result.columns and "ip" in result.columns:
        result["era_calc"] = _safe_div(result["er"] * 9, result["ip"])

    # WHIP = sum(bb_allowed + h_allowed) / sum(ip)
    if all(c in result.columns for c in ["bb_allowed", "h_allowed", "ip"]):
        whip_num = result["bb_allowed"] + result["h_allowed"]
        result["whip_calc"] = _safe_div(whip_num, result["ip"])

    return result


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


# ── Databank assembly ─────────────────────────────────────────────────────────


def load_databank(
    stat_view: str,
    season: int = 2026,
) -> pd.DataFrame:
    """Load a player pool DataFrame for the given stat view key.

    Uses ``STAT_VIEW_PARAMS`` to determine the computation type and dispatches
    to the appropriate data source.  The returned DataFrame always has a
    ``player_name`` column (renamed from ``name`` if necessary).

    Args:
        stat_view: One of the keys in ``STAT_VIEW_OPTIONS`` / ``STAT_VIEW_PARAMS``
                   (e.g. ``"S_S_2026"``, ``"S_L7"``).
        season: Fallback season year when not encoded in ``stat_view``.

    Returns:
        DataFrame with player metadata and stats columns for the requested view.
        Returns an empty DataFrame if the player pool is unavailable.
    """
    params = STAT_VIEW_PARAMS.get(stat_view, {"type": "total", "days": None, "season": None})
    view_type = params.get("type", "total")
    days = params.get("days")
    view_season = params.get("season") or season

    # Load base player pool (includes metadata + blended/ROS projections)
    try:
        pool = load_player_pool()
    except Exception:
        logger.exception("load_databank: failed to load player pool")
        return pd.DataFrame()

    if pool is None or pool.empty:
        return pd.DataFrame()

    # Normalise "name" → "player_name" (database returns "name"; pool may
    # already have "player_name" if enriched by _enrich_pool).
    if "player_name" not in pool.columns and "name" in pool.columns:
        pool = pool.rename(columns={"name": "player_name"})

    # For views backed by season-level projections, return pool as-is.
    # The pool already holds the best available season/ROS projection stats.
    if view_type in ("total", "proj", "special", "ranks", "research", "matchups", "opponents", "live"):
        return pool.copy()

    # For rolling avg / stddev views, compute from game logs and merge metadata.
    if view_type in ("avg", "stddev"):
        player_ids: list[int] = pool["player_id"].dropna().astype(int).tolist()
        if not player_ids:
            return pool.copy()

        rolled = compute_rolling_stats(
            player_ids,
            days=days,
            stat_type=view_type,
            season=view_season,
        )
        if rolled.empty:
            # No game log data — return pool metadata without stat columns
            return pool.copy()

        # Merge rolled stats onto pool metadata
        meta_cols = [c for c in pool.columns if c not in rolled.columns or c == "player_id"]
        merged = pool[meta_cols].merge(rolled, on="player_id", how="left")
        return merged

    # For advanced (Statcast) views, return pool which already contains
    # xwoba, barrel_pct, ev_mean, stuff_plus, etc. columns.
    if view_type == "advanced":
        return pool.copy()

    # Fallback: return pool unchanged
    return pool.copy()


# ── Databank filtering ────────────────────────────────────────────────────────


def filter_databank(
    df: pd.DataFrame,
    position: str = "B",
    status: str = "A",
    mlb_team: str = "ALL",
    fantasy_team: str = "NONE",
    search: str = "",
    show_my_team: bool = False,
) -> pd.DataFrame:
    """Filter a databank DataFrame by common UI criteria.

    Args:
        df: DataFrame returned by ``load_databank()``.
        position: ``"B"`` = batters (``is_hitter==1``), ``"P"`` = pitchers
                  (``is_hitter==0``), or a specific position string (e.g.
                  ``"SS"``) matched via case-insensitive substring in the
                  ``positions`` column.  Anything else is treated as ``"ALL"``
                  (no filter).
        status: ``"A"`` = available (not on a fantasy roster), ``"FA"`` = free
                agents, ``"T"`` = taken (on a roster), ``"ALL"`` = no filter.
                Requires an ``is_available`` or ``roster_team`` column; skipped
                gracefully when absent.
        mlb_team: Filter to players on this MLB team.  ``"ALL"`` disables the
                  filter.  Matched against the ``team`` column.
        fantasy_team: Filter to players on this fantasy roster.  ``"NONE"``
                      disables the filter.  Matched against ``roster_team``.
        search: Case-insensitive substring search applied to ``player_name``
                (or ``name``).  Empty string disables the filter.
        show_my_team: When ``True`` and an ``is_user_team`` column is present,
                      restrict to rows where ``is_user_team`` is truthy.

    Returns:
        Filtered copy of ``df``.  Never modifies the input DataFrame.
    """
    result = df.copy()

    # ── Position filter ──────────────────────────────────────────────────────
    if position == "B":
        if "is_hitter" in result.columns:
            result = result[result["is_hitter"] == 1]
    elif position == "P":
        if "is_hitter" in result.columns:
            result = result[result["is_hitter"] == 0]
    elif position not in ("ALL", "", None):
        # Specific positional filter (e.g. "SS", "2B", "SP", "RP")
        if "positions" in result.columns:
            pos_upper = str(position).upper()
            result = result[result["positions"].fillna("").str.upper().str.contains(pos_upper, regex=False)]

    # ── Status filter ────────────────────────────────────────────────────────
    if status == "A":
        # Available = not on any fantasy roster
        if "is_available" in result.columns:
            result = result[result["is_available"].fillna(True).astype(bool)]
        elif "roster_team" in result.columns:
            result = result[result["roster_team"].isna() | (result["roster_team"] == "")]
    elif status == "FA":
        if "roster_team" in result.columns:
            result = result[result["roster_team"].isna() | (result["roster_team"] == "")]
    elif status == "T":
        if "roster_team" in result.columns:
            result = result[result["roster_team"].notna() & (result["roster_team"] != "")]
    # "ALL" — no filter

    # ── MLB team filter ──────────────────────────────────────────────────────
    if mlb_team and mlb_team != "ALL":
        if "team" in result.columns:
            result = result[result["team"] == mlb_team]

    # ── Fantasy team filter ──────────────────────────────────────────────────
    if fantasy_team and fantasy_team != "NONE":
        if "roster_team" in result.columns:
            result = result[result["roster_team"] == fantasy_team]

    # ── Search filter ────────────────────────────────────────────────────────
    if search:
        name_col = "player_name" if "player_name" in result.columns else "name"
        if name_col in result.columns:
            result = result[result[name_col].fillna("").str.contains(search, case=False, regex=False)]
        else:
            # No name column — return empty
            result = result.iloc[0:0]

    # ── My team filter ───────────────────────────────────────────────────────
    if show_my_team and "is_user_team" in result.columns:
        result = result[result["is_user_team"].fillna(False).astype(bool)]

    return result
