"""Database module: SQLite schema, data loading, and projection blending."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "draft_tool.db"


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS players (
            player_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            team TEXT,
            positions TEXT NOT NULL,  -- comma-separated: "SS,OF"
            is_hitter INTEGER NOT NULL DEFAULT 1,  -- 1=hitter, 0=pitcher
            is_injured INTEGER NOT NULL DEFAULT 0,
            injury_note TEXT
        );

        CREATE TABLE IF NOT EXISTS projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            system TEXT NOT NULL,  -- 'steamer', 'zips', 'depthcharts', 'blended'
            -- Hitting stats
            pa INTEGER DEFAULT 0,
            ab INTEGER DEFAULT 0,
            h INTEGER DEFAULT 0,
            r INTEGER DEFAULT 0,
            hr INTEGER DEFAULT 0,
            rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0,
            avg REAL DEFAULT 0,
            -- Pitching stats
            ip REAL DEFAULT 0,
            w INTEGER DEFAULT 0,
            sv INTEGER DEFAULT 0,
            k INTEGER DEFAULT 0,
            era REAL DEFAULT 0,
            whip REAL DEFAULT 0,
            er INTEGER DEFAULT 0,
            bb_allowed INTEGER DEFAULT 0,
            h_allowed INTEGER DEFAULT 0,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS adp (
            player_id INTEGER PRIMARY KEY,
            yahoo_adp REAL,
            fantasypros_adp REAL,
            adp REAL NOT NULL,  -- best available ADP
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE INDEX IF NOT EXISTS idx_projections_player ON projections(player_id);
        CREATE INDEX IF NOT EXISTS idx_projections_system ON projections(system);
        CREATE INDEX IF NOT EXISTS idx_players_name ON players(name);
    """)
    conn.commit()
    conn.close()

    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS season_stats (
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL DEFAULT 2026,
            pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
            r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0, avg REAL DEFAULT 0,
            ip REAL DEFAULT 0, w INTEGER DEFAULT 0, sv INTEGER DEFAULT 0,
            k INTEGER DEFAULT 0, era REAL DEFAULT 0, whip REAL DEFAULT 0,
            er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0, h_allowed INTEGER DEFAULT 0,
            games_played INTEGER DEFAULT 0,
            last_updated TEXT,
            PRIMARY KEY (player_id, season),
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS ros_projections (
            player_id INTEGER NOT NULL,
            system TEXT NOT NULL,
            pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
            r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0, avg REAL DEFAULT 0,
            ip REAL DEFAULT 0, w INTEGER DEFAULT 0, sv INTEGER DEFAULT 0,
            k INTEGER DEFAULT 0, era REAL DEFAULT 0, whip REAL DEFAULT 0,
            er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0, h_allowed INTEGER DEFAULT 0,
            updated_at TEXT,
            PRIMARY KEY (player_id, system),
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS league_rosters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            team_index INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            roster_slot TEXT,
            is_user_team INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS league_standings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            category TEXT NOT NULL,
            total REAL DEFAULT 0,
            rank INTEGER,
            points REAL
        );

        CREATE TABLE IF NOT EXISTS park_factors (
            team_code TEXT PRIMARY KEY,
            factor_hitting REAL DEFAULT 1.0,
            factor_pitching REAL DEFAULT 1.0
        );

        CREATE TABLE IF NOT EXISTS refresh_log (
            source TEXT PRIMARY KEY,
            last_refresh TEXT,
            status TEXT DEFAULT 'success'
        );

        CREATE INDEX IF NOT EXISTS idx_season_stats_player ON season_stats(player_id);
        CREATE INDEX IF NOT EXISTS idx_ros_proj_player ON ros_projections(player_id);
        CREATE INDEX IF NOT EXISTS idx_league_rosters_team ON league_rosters(team_name);
    """)
    conn.commit()
    conn.close()


def import_hitter_csv(csv_path: str, system: str):
    """Import a FanGraphs hitter projection CSV.

    Expected columns (flexible matching):
    Name, Team, PA, AB, H, R, HR, RBI, SB, AVG, plus position info.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    col_map = _build_column_map(df, is_hitter=True)
    conn = get_connection()
    cursor = conn.cursor()
    imported = 0

    for _, row in df.iterrows():
        name = str(row[col_map["name"]]).strip()
        team = str(row.get(col_map.get("team", "Team"), "")).strip() if col_map.get("team") else ""
        pa = int(_safe_num(row, col_map.get("pa", "PA")))
        if pa < 50:
            continue  # Skip trivial playing time

        # Determine positions
        positions = _extract_positions(row, col_map)
        if not positions:
            positions = "Util"

        # Upsert player
        player_id = _upsert_player(cursor, name, team, positions, is_hitter=True)

        ab = int(_safe_num(row, col_map.get("ab", "AB")))
        h = int(_safe_num(row, col_map.get("h", "H")))
        r = int(_safe_num(row, col_map.get("r", "R")))
        hr = int(_safe_num(row, col_map.get("hr", "HR")))
        rbi = int(_safe_num(row, col_map.get("rbi", "RBI")))
        sb = int(_safe_num(row, col_map.get("sb", "SB")))
        avg = float(_safe_num(row, col_map.get("avg", "AVG")))

        cursor.execute(
            """
            INSERT INTO projections (player_id, system, pa, ab, h, r, hr, rbi, sb, avg)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (player_id, system, pa, ab, h, r, hr, rbi, sb, avg),
        )
        imported += 1

    conn.commit()
    conn.close()
    return imported


def import_pitcher_csv(csv_path: str, system: str):
    """Import a FanGraphs pitcher projection CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    col_map = _build_column_map(df, is_hitter=False)
    conn = get_connection()
    cursor = conn.cursor()
    imported = 0

    for _, row in df.iterrows():
        name = str(row[col_map["name"]]).strip()
        team = str(row.get(col_map.get("team", "Team"), "")).strip() if col_map.get("team") else ""
        ip = float(_safe_num(row, col_map.get("ip", "IP")))
        if ip < 10:
            continue

        # Determine SP vs RP
        sv = int(_safe_num(row, col_map.get("sv", "SV")))
        gs = int(_safe_num(row, col_map.get("gs", "GS"))) if col_map.get("gs") else 0
        if gs >= 5 or ip >= 80:
            positions = "SP"
        elif sv >= 3:
            positions = "RP"
        else:
            positions = "SP,RP" if gs >= 1 else "RP"

        player_id = _upsert_player(cursor, name, team, positions, is_hitter=False)

        w = int(_safe_num(row, col_map.get("w", "W")))
        k = int(_safe_num(row, col_map.get("k", "K")))
        if not k and col_map.get("so"):
            k = int(_safe_num(row, col_map["so"]))
        era = float(_safe_num(row, col_map.get("era", "ERA")))
        whip = float(_safe_num(row, col_map.get("whip", "WHIP")))
        er = int(_safe_num(row, col_map.get("er", "ER")))
        bb_allowed = int(_safe_num(row, col_map.get("bb", "BB")))
        h_allowed = int(_safe_num(row, col_map.get("h_allowed", "H")))

        # Compute ER from ERA if not directly available
        if er == 0 and era > 0 and ip > 0:
            er = int(round(era * ip / 9))
        # Compute H allowed and BB from WHIP if needed
        if h_allowed == 0 and bb_allowed == 0 and whip > 0 and ip > 0:
            total = whip * ip
            bb_allowed = int(total * 0.3)
            h_allowed = int(total * 0.7)

        cursor.execute(
            """
            INSERT INTO projections (player_id, system, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (player_id, system, ip, w, sv, k, era, whip, er, bb_allowed, h_allowed),
        )
        imported += 1

    conn.commit()
    conn.close()
    return imported


def create_blended_projections():
    """Create blended projections by averaging all imported systems."""
    conn = get_connection()
    cursor = conn.cursor()

    # Remove old blended projections
    cursor.execute("DELETE FROM projections WHERE system = 'blended'")

    # Get all players with projections
    cursor.execute("SELECT DISTINCT player_id FROM projections WHERE system != 'blended'")
    player_ids = [row[0] for row in cursor.fetchall()]

    for pid in player_ids:
        cursor.execute(
            """
            SELECT pa, ab, h, r, hr, rbi, sb, avg,
                   ip, w, sv, k, era, whip, er, bb_allowed, h_allowed
            FROM projections WHERE player_id = ? AND system != 'blended'
        """,
            (pid,),
        )
        rows = cursor.fetchall()
        if not rows:
            continue

        n = len(rows)
        # Average all stats
        avg_stats = {}
        stat_names = [
            "pa",
            "ab",
            "h",
            "r",
            "hr",
            "rbi",
            "sb",
            "avg",
            "ip",
            "w",
            "sv",
            "k",
            "era",
            "whip",
            "er",
            "bb_allowed",
            "h_allowed",
        ]
        for i, name in enumerate(stat_names):
            vals = [row[i] or 0 for row in rows]
            avg_stats[name] = sum(vals) / n

        # For rate stats, recompute from components rather than averaging rates
        if avg_stats["ab"] > 0:
            avg_stats["avg"] = avg_stats["h"] / avg_stats["ab"]
        if avg_stats["ip"] > 0:
            avg_stats["era"] = avg_stats["er"] * 9 / avg_stats["ip"]
            avg_stats["whip"] = (avg_stats["bb_allowed"] + avg_stats["h_allowed"]) / avg_stats["ip"]

        cursor.execute(
            """
            INSERT INTO projections (player_id, system, pa, ab, h, r, hr, rbi, sb, avg,
                                     ip, w, sv, k, era, whip, er, bb_allowed, h_allowed)
            VALUES (?, 'blended', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                pid,
                int(avg_stats["pa"]),
                int(avg_stats["ab"]),
                int(avg_stats["h"]),
                int(avg_stats["r"]),
                int(avg_stats["hr"]),
                int(avg_stats["rbi"]),
                int(avg_stats["sb"]),
                round(avg_stats["avg"], 3),
                round(avg_stats["ip"], 1),
                int(avg_stats["w"]),
                int(avg_stats["sv"]),
                int(avg_stats["k"]),
                round(avg_stats["era"], 2),
                round(avg_stats["whip"], 2),
                int(avg_stats["er"]),
                int(avg_stats["bb_allowed"]),
                int(avg_stats["h_allowed"]),
            ),
        )

    conn.commit()
    conn.close()
    return len(player_ids)


def import_adp_csv(csv_path: str, source: str = "fantasypros"):
    """Import ADP data from a CSV file.

    Expected columns: Name (or Player), ADP (or AVG or Rank).
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    # Find name column
    name_col = None
    for candidate in ["Player", "Name", "PLAYER", "player", "name"]:
        if candidate in df.columns:
            name_col = candidate
            break
    if name_col is None:
        raise ValueError(f"Cannot find player name column. Columns: {list(df.columns)}")

    # Find ADP column
    adp_col = None
    for candidate in ["ADP", "AVG", "Rank", "Overall", "adp", "avg"]:
        if candidate in df.columns:
            adp_col = candidate
            break
    if adp_col is None:
        raise ValueError(f"Cannot find ADP column. Columns: {list(df.columns)}")

    conn = get_connection()
    cursor = conn.cursor()
    imported = 0

    for _, row in df.iterrows():
        name = str(row[name_col]).strip()
        adp_val = _safe_num(row, adp_col)
        if adp_val <= 0:
            continue

        # Find matching player
        cursor.execute("SELECT player_id FROM players WHERE name = ?", (name,))
        result = cursor.fetchone()
        if not result:
            # Try fuzzy match (first + last name)
            parts = name.split()
            if len(parts) >= 2:
                cursor.execute(
                    "SELECT player_id FROM players WHERE name LIKE ? AND name LIKE ?",
                    (f"%{parts[0]}%", f"%{parts[-1]}%"),
                )
                result = cursor.fetchone()
        if not result:
            continue

        player_id = result[0]
        if source == "yahoo":
            cursor.execute(
                """
                INSERT INTO adp (player_id, yahoo_adp, adp) VALUES (?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET yahoo_adp = ?, adp = min(adp, ?)
            """,
                (player_id, adp_val, adp_val, adp_val, adp_val),
            )
        else:
            cursor.execute(
                """
                INSERT INTO adp (player_id, fantasypros_adp, adp) VALUES (?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET fantasypros_adp = ?, adp = min(adp, ?)
            """,
                (player_id, adp_val, adp_val, adp_val, adp_val),
            )
        imported += 1

    conn.commit()
    conn.close()
    return imported


def load_player_pool() -> pd.DataFrame:
    """Load the full player pool with blended projections and ADP for the valuation engine."""
    conn = get_connection()

    # Prefer blended, fall back to any available system
    # Note: Do NOT use CAST on numeric columns — Python 3.14 SQLite returns raw bytes
    # for NumPy integers, and CAST corrupts them. Fix bytes in Python after loading.
    df = pd.read_sql_query(
        """
        SELECT
            p.player_id, p.name, p.team, p.positions, p.is_hitter, p.is_injured,
            proj.pa, proj.ab, proj.h, proj.r, proj.hr, proj.rbi, proj.sb, proj.avg,
            proj.ip, proj.w, proj.sv, proj.k, proj.era, proj.whip,
            proj.er, proj.bb_allowed, proj.h_allowed,
            COALESCE(a.adp, 999) as adp
        FROM players p
        JOIN projections proj ON p.player_id = proj.player_id
        LEFT JOIN adp a ON p.player_id = a.player_id
        WHERE proj.system = 'blended'
          AND p.is_injured = 0
        ORDER BY COALESCE(a.adp, 999)
    """,
        conn,
    )

    # If no blended projections exist, try any system
    if df.empty:
        df = pd.read_sql_query(
            """
            SELECT
                p.player_id, p.name, p.team, p.positions, p.is_hitter, p.is_injured,
                proj.pa, proj.ab, proj.h, proj.r, proj.hr, proj.rbi, proj.sb, proj.avg,
                proj.ip, proj.w, proj.sv, proj.k, proj.era, proj.whip,
                proj.er, proj.bb_allowed, proj.h_allowed,
                COALESCE(a.adp, 999) as adp
            FROM players p
            JOIN projections proj ON p.player_id = proj.player_id
            LEFT JOIN adp a ON p.player_id = a.player_id
            WHERE p.is_injured = 0
            ORDER BY COALESCE(a.adp, 999)
        """,
            conn,
        )

    conn.close()

    # Fix Python 3.14 SQLite bytes issue: NumPy ints may be stored as raw bytes
    import struct

    def _fix_bytes_col(series, as_float=False):
        def _decode(val):
            if isinstance(val, bytes):
                if len(val) == 8:
                    return struct.unpack("<q", val)[0]
                elif len(val) == 4:
                    return struct.unpack("<i", val)[0]
                try:
                    return float(val) if as_float else int(val)
                except (ValueError, TypeError):
                    return 0.0 if as_float else 0
            return val

        return series.map(_decode)

    # Ensure numeric columns are properly typed
    int_cols = ["pa", "ab", "h", "r", "hr", "rbi", "sb", "w", "sv", "k", "er", "bb_allowed", "h_allowed"]
    float_cols = ["avg", "ip", "era", "whip", "adp"]
    for col in int_cols:
        if col in df.columns:
            df[col] = _fix_bytes_col(df[col])
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols:
        if col in df.columns:
            df[col] = _fix_bytes_col(df[col], as_float=True)
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


# ── In-Season Data Functions ──────────────────────────────────────────


def upsert_season_stats(player_id: int, stats: dict, season: int = 2026):
    """Insert or update a player's season stats."""
    conn = get_connection()
    cols = [
        "pa",
        "ab",
        "h",
        "r",
        "hr",
        "rbi",
        "sb",
        "avg",
        "ip",
        "w",
        "sv",
        "k",
        "era",
        "whip",
        "er",
        "bb_allowed",
        "h_allowed",
        "games_played",
    ]
    values = {c: stats.get(c, 0) for c in cols}
    values["last_updated"] = datetime.now(UTC).isoformat()

    conn.execute(
        f"""INSERT INTO season_stats (player_id, season, {", ".join(cols)}, last_updated)
            VALUES (?, ?, {", ".join("?" for _ in cols)}, ?)
            ON CONFLICT(player_id, season) DO UPDATE SET
            {", ".join(f"{c}=excluded.{c}" for c in cols)},
            last_updated=excluded.last_updated""",
        (player_id, season, *[values[c] for c in cols], values["last_updated"]),
    )
    conn.commit()
    conn.close()


def load_season_stats(season: int = 2026) -> pd.DataFrame:
    """Load all season stats as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM season_stats WHERE season = ?", conn, params=(season,))
    conn.close()
    return df


def upsert_ros_projection(player_id: int, system: str, stats: dict):
    """Insert or update a ROS projection for a player."""
    conn = get_connection()
    cols = [
        "pa",
        "ab",
        "h",
        "r",
        "hr",
        "rbi",
        "sb",
        "avg",
        "ip",
        "w",
        "sv",
        "k",
        "era",
        "whip",
        "er",
        "bb_allowed",
        "h_allowed",
    ]
    values = {c: stats.get(c, 0) for c in cols}
    values["updated_at"] = datetime.now(UTC).isoformat()

    conn.execute(
        f"""INSERT INTO ros_projections (player_id, system, {", ".join(cols)}, updated_at)
            VALUES (?, ?, {", ".join("?" for _ in cols)}, ?)
            ON CONFLICT(player_id, system) DO UPDATE SET
            {", ".join(f"{c}=excluded.{c}" for c in cols)},
            updated_at=excluded.updated_at""",
        (player_id, system, *[values[c] for c in cols], values["updated_at"]),
    )
    conn.commit()
    conn.close()


def upsert_league_roster_entry(
    team_name: str,
    team_index: int,
    player_id: int,
    roster_slot: str = None,
    is_user_team: bool = False,
):
    """Add a player to a league roster."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team)
           VALUES (?, ?, ?, ?, ?)""",
        (team_name, team_index, player_id, roster_slot, 1 if is_user_team else 0),
    )
    conn.commit()
    conn.close()


def load_league_rosters() -> pd.DataFrame:
    """Load all league rosters as a DataFrame."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM league_rosters", conn)
    conn.close()
    return df


def clear_league_rosters():
    """Remove all league roster entries (for re-import)."""
    conn = get_connection()
    conn.execute("DELETE FROM league_rosters")
    conn.commit()
    conn.close()


def upsert_league_standing(team_name: str, category: str, total: float, rank: int = None, points: float = None):
    """Insert or update a league standing entry."""
    conn = get_connection()
    conn.execute(
        "DELETE FROM league_standings WHERE team_name = ? AND category = ?",
        (team_name, category),
    )
    conn.execute(
        "INSERT INTO league_standings (team_name, category, total, rank, points) VALUES (?, ?, ?, ?, ?)",
        (team_name, category, total, rank, points),
    )
    conn.commit()
    conn.close()


def update_refresh_log(source: str, status: str = "success"):
    """Update the refresh log for a data source."""
    conn = get_connection()
    conn.execute(
        """INSERT INTO refresh_log (source, last_refresh, status)
           VALUES (?, ?, ?)
           ON CONFLICT(source) DO UPDATE SET
           last_refresh=excluded.last_refresh, status=excluded.status""",
        (source, datetime.now(UTC).isoformat(), status),
    )
    conn.commit()
    conn.close()


def get_refresh_status(source: str) -> dict | None:
    """Get the last refresh status for a data source."""
    conn = get_connection()
    cursor = conn.execute(
        "SELECT source, last_refresh, status FROM refresh_log WHERE source = ?",
        (source,),
    )
    row = cursor.fetchone()
    conn.close()
    if row is None:
        return None
    return {"source": row[0], "last_refresh": row[1], "status": row[2]}


# ── Helpers ──────────────────────────────────────────────────────────


def _build_column_map(df: pd.DataFrame, is_hitter: bool) -> dict:
    """Flexibly map CSV columns to expected names."""
    cols = {c.lower().strip(): c for c in df.columns}
    m = {}

    # Name
    for k in ["name", "player", "playername"]:
        if k in cols:
            m["name"] = cols[k]
            break
    if "name" not in m:
        m["name"] = df.columns[0]

    # Team
    for k in ["team", "tm"]:
        if k in cols:
            m["team"] = cols[k]
            break

    if is_hitter:
        for stat, candidates in {
            "pa": ["pa"],
            "ab": ["ab"],
            "h": ["h", "hits"],
            "r": ["r", "runs"],
            "hr": ["hr", "homerun", "homeruns"],
            "rbi": ["rbi"],
            "sb": ["sb", "stolenbases"],
            "avg": ["avg", "ba", "battingaverage"],
        }.items():
            for k in candidates:
                if k in cols:
                    m[stat] = cols[k]
                    break
    else:
        for stat, candidates in {
            "ip": ["ip", "innings"],
            "w": ["w", "wins"],
            "sv": ["sv", "saves"],
            "k": ["k", "so", "strikeouts"],
            "so": ["so"],
            "era": ["era"],
            "whip": ["whip"],
            "er": ["er", "earnedruns"],
            "bb": ["bb", "walks", "bb_allowed"],
            "h_allowed": ["h", "hits", "h_allowed"],
            "gs": ["gs", "gamesstarted"],
        }.items():
            for k in candidates:
                if k in cols:
                    m[stat] = cols[k]
                    break

    return m


def _extract_positions(row, col_map) -> str:
    """Try to extract position from various CSV column formats."""
    # Try common position column names
    for col_name in ["Pos", "Position", "POS", "pos", "Positions"]:
        if col_name in row.index:
            pos_val = str(row[col_name]).strip()
            if pos_val and pos_val != "nan":
                return pos_val.replace("/", ",").replace("-", ",")
    return ""


def _upsert_player(cursor, name: str, team: str, positions: str, is_hitter: bool) -> int:
    """Insert or find a player. Returns player_id."""
    cursor.execute("SELECT player_id, positions FROM players WHERE name = ? AND team = ?", (name, team))
    result = cursor.fetchone()
    if result:
        # Merge positions
        existing = set(result[1].split(","))
        new = set(positions.split(","))
        merged = ",".join(sorted(existing | new))
        if merged != result[1]:
            cursor.execute("UPDATE players SET positions = ? WHERE player_id = ?", (merged, result[0]))
        return result[0]
    else:
        cursor.execute(
            "INSERT INTO players (name, team, positions, is_hitter) VALUES (?, ?, ?, ?)",
            (name, team, positions, 1 if is_hitter else 0),
        )
        return cursor.lastrowid


def _safe_num(row, col_name, default=0):
    """Safely extract a numeric value from a row."""
    if col_name is None or col_name not in row.index:
        return default
    val = row[col_name]
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
