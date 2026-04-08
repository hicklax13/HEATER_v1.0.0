"""Database module: SQLite schema, data loading, and projection blending."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "data" / "draft_tool.db"

# ── Stat column lists for bytes coercion ─────────────────────────
# Python 3.13+ SQLite returns bytes for some integer columns.
# These lists define all numeric stat columns used across tables.
_INT_STAT_COLS = [
    "pa",
    "ab",
    "h",
    "r",
    "hr",
    "rbi",
    "sb",
    "w",
    "l",
    "sv",
    "k",
    "er",
    "bb",
    "bb_allowed",
    "h_allowed",
    "hbp",
    "sf",
    "player_id",
    "is_hitter",
    "is_injured",
    "games_played",
    "games_available",
    "il_stints",
    "il_days",
    "season",
    "team_index",
    "is_user_team",
    "rank",
    "consensus_rank",
    "ecr_sources",
    "ytd_pa",
    "ytd_hr",
    "ytd_rbi",
    "ytd_sb",
    "ytd_sv",
    "ytd_k",
]
_FLOAT_STAT_COLS = [
    "avg",
    "obp",
    "ip",
    "era",
    "whip",
    "adp",
    "total",
    "points",
    "fip",
    "xfip",
    "siera",
    "ecr_avg",
    "ytd_avg",
    "ytd_era",
    "ytd_whip",
    "health_score",
    "scarcity_mult",
    "xwoba",
    "xba",
    "barrel_pct",
    "hard_hit_pct",
    "ev_mean",
    "woba_approx",
    "xwoba_delta",
]


def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce known numeric columns from bytes to proper types.

    Python 3.13+ with SQLite may return raw bytes for integer columns.
    This function applies pd.to_numeric() to all known stat columns,
    safely ignoring columns that don't exist in the DataFrame.
    """
    for col in _INT_STAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in _FLOAT_STAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return df


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
        CREATE INDEX IF NOT EXISTS idx_projections_player_system ON projections(player_id, system);
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
            FOREIGN KEY (player_id) REFERENCES players(player_id),
            UNIQUE(team_name, player_id)
        );

        CREATE TABLE IF NOT EXISTS league_teams (
            team_key TEXT PRIMARY KEY,
            team_name TEXT NOT NULL,
            team_index INTEGER,
            logo_url TEXT,
            manager_name TEXT,
            is_user_team INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS league_standings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            category TEXT NOT NULL,
            total REAL DEFAULT 0,
            rank INTEGER,
            points REAL,
            UNIQUE(team_name, category)
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

        CREATE TABLE IF NOT EXISTS injury_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            games_played INTEGER DEFAULT 0,
            games_available INTEGER DEFAULT 162,
            il_stints INTEGER DEFAULT 0,
            il_days INTEGER DEFAULT 0,
            created_at TEXT,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            league_id TEXT,
            player_id INTEGER NOT NULL,
            type TEXT NOT NULL,  -- 'add', 'drop', 'trade'
            team_from TEXT,
            team_to TEXT,
            timestamp TEXT,
            created_at TEXT,
            FOREIGN KEY (player_id) REFERENCES players(player_id)
        );

        CREATE TABLE IF NOT EXISTS statcast_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            ev_mean REAL, ev_p90 REAL, barrel_pct REAL, hard_hit_pct REAL,
            xba REAL, xslg REAL, xwoba REAL, whiff_pct REAL, chase_rate REAL,
            sprint_speed REAL, ff_avg_speed REAL, ff_spin_rate REAL,
            k_pct REAL, bb_pct REAL, gb_pct REAL,
            stuff_plus REAL, location_plus REAL, pitching_plus REAL,
            babip REAL, iso REAL,
            hitter_k_pct REAL, hitter_bb_pct REAL,
            ld_pct REAL, hitter_fb_pct REAL, hitter_gb_pct REAL,
            bat_speed REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_id, season)
        );

        CREATE INDEX IF NOT EXISTS idx_season_stats_player ON season_stats(player_id);
        CREATE INDEX IF NOT EXISTS idx_ros_proj_player ON ros_projections(player_id);
        CREATE INDEX IF NOT EXISTS idx_league_rosters_team ON league_rosters(team_name);
        CREATE INDEX IF NOT EXISTS idx_injury_history_player ON injury_history(player_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_injury_history_player_season ON injury_history(player_id, season);
        CREATE INDEX IF NOT EXISTS idx_transactions_player ON transactions(player_id);
        CREATE INDEX IF NOT EXISTS idx_statcast_archive_player ON statcast_archive(player_id);

        CREATE TABLE IF NOT EXISTS player_tags (
            player_id INTEGER NOT NULL,
            tag TEXT NOT NULL CHECK(tag IN ('Sleeper','Target','Avoid','Breakout','Bust')),
            note TEXT DEFAULT '',
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (player_id, tag)
        );

        CREATE TABLE IF NOT EXISTS leagues (
            league_id TEXT PRIMARY KEY,
            platform TEXT DEFAULT 'manual',
            league_name TEXT NOT NULL,
            num_teams INTEGER DEFAULT 12,
            scoring_format TEXT DEFAULT 'h2h_categories',
            yahoo_league_id TEXT,
            created_at TEXT NOT NULL,
            is_active INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS prospect_rankings (
            prospect_id INTEGER PRIMARY KEY AUTOINCREMENT,
            mlb_id INTEGER,
            name TEXT NOT NULL,
            team TEXT,
            position TEXT,
            fg_rank INTEGER,
            fg_fv INTEGER,
            fg_eta TEXT,
            fg_risk TEXT,
            age INTEGER,
            hit_present INTEGER, hit_future INTEGER,
            game_present INTEGER, game_future INTEGER,
            raw_present INTEGER, raw_future INTEGER,
            speed INTEGER, field INTEGER,
            ctrl_present INTEGER, ctrl_future INTEGER,
            scouting_report TEXT,
            tldr TEXT,
            milb_level TEXT,
            milb_avg REAL, milb_obp REAL, milb_slg REAL,
            milb_k_pct REAL, milb_bb_pct REAL, milb_hr INTEGER, milb_sb INTEGER,
            milb_ip REAL, milb_era REAL, milb_whip REAL, milb_k9 REAL, milb_bb9 REAL,
            readiness_score REAL,
            fetched_at TEXT
        );

        CREATE TABLE IF NOT EXISTS ecr_consensus (
            player_id INTEGER PRIMARY KEY,
            espn_rank INTEGER,
            yahoo_adp REAL,
            cbs_rank INTEGER,
            nfbc_adp REAL,
            fg_adp REAL,
            fp_ecr INTEGER,
            heater_sgp_rank INTEGER,
            consensus_rank INTEGER,
            consensus_avg REAL,
            rank_min INTEGER,
            rank_max INTEGER,
            rank_stddev REAL,
            n_sources INTEGER,
            fetched_at TEXT
        );

        CREATE TABLE IF NOT EXISTS player_id_map (
            player_id INTEGER PRIMARY KEY,
            espn_id INTEGER,
            yahoo_key TEXT,
            fg_id INTEGER,
            mlb_id INTEGER,
            cbs_id INTEGER,
            nfbc_id INTEGER,
            name TEXT,
            team TEXT,
            updated_at TEXT
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_id_map_espn ON player_id_map(espn_id) WHERE espn_id IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_id_map_yahoo ON player_id_map(yahoo_key) WHERE yahoo_key IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_id_map_fg ON player_id_map(fg_id) WHERE fg_id IS NOT NULL;
        CREATE UNIQUE INDEX IF NOT EXISTS idx_id_map_mlb ON player_id_map(mlb_id) WHERE mlb_id IS NOT NULL;

        CREATE TABLE IF NOT EXISTS player_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            source TEXT NOT NULL,
            headline TEXT NOT NULL,
            detail TEXT,
            news_type TEXT,
            injury_body_part TEXT,
            il_status TEXT,
            sentiment_score REAL,
            published_at TEXT,
            fetched_at TEXT,
            UNIQUE(player_id, source, headline, published_at)
        );
        CREATE INDEX IF NOT EXISTS idx_player_news_player ON player_news(player_id);
        CREATE INDEX IF NOT EXISTS idx_player_news_type ON player_news(news_type);

        CREATE TABLE IF NOT EXISTS ownership_trends (
            player_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            percent_owned REAL,
            delta_7d REAL,
            PRIMARY KEY (player_id, date)
        );

        CREATE TABLE IF NOT EXISTS opponent_profiles (
            team_name TEXT PRIMARY KEY,
            tier INTEGER NOT NULL DEFAULT 3,
            threat_level TEXT,
            strengths TEXT,
            weaknesses TEXT,
            manager TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS league_schedule (
            week INTEGER NOT NULL,
            team_a TEXT NOT NULL,
            team_b TEXT NOT NULL,
            PRIMARY KEY (week, team_a)
        );

        CREATE TABLE IF NOT EXISTS league_schedule_full (
            week INTEGER NOT NULL,
            team_a TEXT NOT NULL,
            team_b TEXT NOT NULL,
            PRIMARY KEY (week, team_a, team_b)
        );

        CREATE TABLE IF NOT EXISTS league_records (
            team_name TEXT PRIMARY KEY,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            ties INTEGER DEFAULT 0,
            win_pct REAL DEFAULT 0.0,
            points_for REAL DEFAULT 0.0,
            points_against REAL DEFAULT 0.0,
            streak TEXT DEFAULT '',
            rank INTEGER DEFAULT 0,
            updated_at TEXT
        );

        CREATE TABLE IF NOT EXISTS yahoo_free_agents (
            player_key TEXT PRIMARY KEY,
            player_name TEXT NOT NULL,
            positions TEXT,
            team TEXT,
            percent_owned REAL,
            fetched_at TEXT
        );

        CREATE TABLE IF NOT EXISTS roster_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            team_name TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            roster_slot TEXT,
            UNIQUE(snapshot_date, team_name, player_id)
        );

        CREATE TABLE IF NOT EXISTS game_day_weather (
            game_pk INTEGER NOT NULL,
            game_date TEXT NOT NULL,
            venue_team TEXT NOT NULL,
            temp_f REAL,
            wind_mph REAL,
            wind_dir TEXT,
            precip_pct REAL,
            humidity_pct REAL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (game_pk)
        );

        CREATE TABLE IF NOT EXISTS team_strength (
            team_abbr TEXT NOT NULL,
            season INTEGER NOT NULL,
            wrc_plus REAL,
            fip REAL,
            team_ops REAL,
            team_era REAL,
            team_whip REAL,
            k_pct REAL,
            bb_pct REAL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (team_abbr, season)
        );

        CREATE TABLE IF NOT EXISTS opp_pitcher_stats (
            pitcher_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            name TEXT,
            team TEXT,
            era REAL,
            fip REAL,
            xfip REAL,
            whip REAL,
            k_per_9 REAL,
            bb_per_9 REAL,
            vs_lhb_avg REAL,
            vs_rhb_avg REAL,
            ip REAL,
            hand TEXT,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (pitcher_id, season)
        );
    """)
    conn.commit()

    # Add new columns to existing tables (safe ALTER — ignores if already exists)
    _safe_add_column(conn, "players", "birth_date", "TEXT")
    _safe_add_column(conn, "players", "mlb_id", "INTEGER")
    _safe_add_column(conn, "projections", "birth_date", "TEXT")
    _safe_add_column(conn, "projections", "mlb_id", "INTEGER")

    # Phase 1 league format migration: OBP/L/BB/HBP/SF columns
    for table in ("projections", "season_stats", "ros_projections"):
        _safe_add_column(conn, table, "obp", "REAL DEFAULT 0")
        _safe_add_column(conn, table, "l", "INTEGER DEFAULT 0")
        _safe_add_column(conn, table, "bb", "INTEGER DEFAULT 0")
        _safe_add_column(conn, table, "hbp", "INTEGER DEFAULT 0")
        _safe_add_column(conn, table, "sf", "INTEGER DEFAULT 0")

    # Phase 2 data foundation: player bio fields
    _safe_add_column(conn, "players", "bats", "TEXT")
    _safe_add_column(conn, "players", "throws", "TEXT")

    # Phase 2 data foundation: advanced pitcher metrics
    for table in ("projections", "season_stats", "ros_projections"):
        _safe_add_column(conn, table, "fip", "REAL DEFAULT 0")
        _safe_add_column(conn, table, "xfip", "REAL DEFAULT 0")
        _safe_add_column(conn, table, "siera", "REAL DEFAULT 0")

    # Contract year / arbitration tracking
    _safe_add_column(conn, "players", "arbitration_eligible", "INTEGER DEFAULT 0")

    # Multi-source ADP: NFBC ADP column
    _safe_add_column(conn, "adp", "nfbc_adp", "REAL")

    # Gap closure: cross-reference IDs
    _safe_add_column(conn, "players", "fangraphs_id", "TEXT")
    _safe_add_column(conn, "players", "yahoo_id", "TEXT")

    # Gap closure: roster type, role, contract details
    _safe_add_column(conn, "players", "roster_type", "TEXT DEFAULT 'active'")
    _safe_add_column(conn, "players", "role_status", "TEXT")
    _safe_add_column(conn, "players", "contract_details", "TEXT")
    _safe_add_column(conn, "players", "spring_training_stats", "TEXT")

    # Gap closure: Statcast Stuff+/Location+/Pitching+ + T5: gmLI
    for table in ("projections", "season_stats", "ros_projections"):
        _safe_add_column(conn, table, "stuff_plus", "REAL")
        _safe_add_column(conn, table, "location_plus", "REAL")
        _safe_add_column(conn, table, "pitching_plus", "REAL")
    _safe_add_column(conn, "season_stats", "gmli", "REAL")

    # Gap 6: Persist computed fields on players table
    _safe_add_column(conn, "players", "depth_chart_role", "TEXT")
    _safe_add_column(conn, "players", "contract_year", "INTEGER DEFAULT 0")
    _safe_add_column(conn, "players", "news_sentiment", "REAL")
    _safe_add_column(conn, "players", "lineup_slot", "INTEGER")
    _safe_add_column(conn, "players", "spring_training_era", "REAL")

    # Gap 7: Persist Yahoo IL/DTD/NA player status on league_rosters
    _safe_add_column(conn, "league_rosters", "status", "TEXT DEFAULT 'active'")
    _safe_add_column(conn, "league_rosters", "selected_position", "TEXT DEFAULT ''")
    _safe_add_column(conn, "league_rosters", "editorial_team_abbr", "TEXT DEFAULT ''")

    # Gap 8a: League draft picks table (stores your league's actual draft, not generic ADP)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS league_draft_picks (
                pick_number INTEGER NOT NULL,
                round INTEGER NOT NULL,
                team_name TEXT NOT NULL,
                player_id INTEGER,
                player_name TEXT NOT NULL,
                PRIMARY KEY (pick_number)
            )
        """)
        conn.commit()
    except Exception:
        pass

    # Gap 8: Team details (FAAB, waiver priority, activity counts)
    _safe_add_column(conn, "league_teams", "faab_balance", "REAL")
    _safe_add_column(conn, "league_teams", "waiver_priority", "INTEGER")
    _safe_add_column(conn, "league_teams", "number_of_moves", "INTEGER")
    _safe_add_column(conn, "league_teams", "number_of_trades", "INTEGER")

    # T2: Advanced batting stats in statcast_archive
    _safe_add_column(conn, "statcast_archive", "babip", "REAL")
    _safe_add_column(conn, "statcast_archive", "iso", "REAL")
    _safe_add_column(conn, "statcast_archive", "hitter_k_pct", "REAL")
    _safe_add_column(conn, "statcast_archive", "hitter_bb_pct", "REAL")
    _safe_add_column(conn, "statcast_archive", "ld_pct", "REAL")
    _safe_add_column(conn, "statcast_archive", "hitter_fb_pct", "REAL")
    _safe_add_column(conn, "statcast_archive", "hitter_gb_pct", "REAL")
    _safe_add_column(conn, "statcast_archive", "bat_speed", "REAL")

    conn.close()


_VALID_TABLE_NAMES = frozenset(
    {
        "players",
        "projections",
        "adp",
        "league_config",
        "draft_picks",
        "league_draft_picks",
        "blended_projections",
        "player_pool",
        "season_stats",
        "ros_projections",
        "league_rosters",
        "league_standings",
        "league_teams",
        "park_factors",
        "refresh_log",
        "injury_history",
        "transactions",
        "ecr_consensus",
        "prospect_rankings",
        "player_news",
        "news_player_map",
        "roster_snapshots",
        "game_day_weather",
        "team_strength",
        "opp_pitcher_stats",
        "statcast_archive",
    }
)
_VALID_COL_RE = __import__("re").compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _safe_add_column(conn: sqlite3.Connection, table: str, column: str, col_type: str):
    """Add a column to a table if it doesn't already exist.

    Validates table/column names against allowlists to prevent SQL injection.
    """
    if table not in _VALID_TABLE_NAMES:
        raise ValueError(f"Invalid table name for ALTER TABLE: {table!r}")
    if not _VALID_COL_RE.match(column):
        raise ValueError(f"Invalid column name for ALTER TABLE: {column!r}")
    if not _VALID_COL_RE.match(col_type.split()[0]):
        raise ValueError(f"Invalid column type for ALTER TABLE: {col_type!r}")
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column" not in str(e).lower():
            raise


def import_hitter_csv(csv_path: str, system: str):
    """Import a FanGraphs hitter projection CSV.

    Expected columns (flexible matching):
    Name, Team, PA, AB, H, R, HR, RBI, SB, AVG, plus position info.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    col_map = _build_column_map(df, is_hitter=True)
    conn = get_connection()
    try:
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
            obp = float(_safe_num(row, col_map.get("obp", "OBP")))
            bb = int(_safe_num(row, col_map.get("bb", "BB")))
            hbp = int(_safe_num(row, col_map.get("hbp", "HBP")))
            sf = int(_safe_num(row, col_map.get("sf", "SF")))

            cursor.execute(
                """
                INSERT INTO projections (player_id, system, pa, ab, h, r, hr, rbi, sb, avg,
                                         obp, bb, hbp, sf)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (player_id, system, pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf),
            )
            imported += 1

        conn.commit()
    finally:
        conn.close()
    return imported


def import_pitcher_csv(csv_path: str, system: str):
    """Import a FanGraphs pitcher projection CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    col_map = _build_column_map(df, is_hitter=False)
    conn = get_connection()
    try:
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
            l = int(_safe_num(row, col_map.get("l", "L")))
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
                INSERT INTO projections (player_id, system, ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (player_id, system, ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed),
            )
            imported += 1

        conn.commit()
    finally:
        conn.close()
    return imported


def create_blended_projections():
    """Create blended projections by averaging all imported systems."""
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Remove old blended projections
        cursor.execute("DELETE FROM projections WHERE system = 'blended'")

        # Get all players with projections
        cursor.execute("SELECT DISTINCT player_id FROM projections WHERE system != 'blended'")
        player_ids = [row[0] for row in cursor.fetchall()]

        for pid in player_ids:
            cursor.execute(
                """
                SELECT pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf,
                       ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed,
                       COALESCE(fip, 0) as fip, COALESCE(xfip, 0) as xfip,
                       COALESCE(siera, 0) as siera
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
                "obp",
                "bb",
                "hbp",
                "sf",
                "ip",
                "w",
                "l",
                "sv",
                "k",
                "era",
                "whip",
                "er",
                "bb_allowed",
                "h_allowed",
                "fip",
                "xfip",
                "siera",
            ]
            for i, name in enumerate(stat_names):
                vals = [row[i] or 0 for row in rows]
                avg_stats[name] = sum(vals) / n

            # For rate stats, recompute from components rather than averaging rates
            if avg_stats["ab"] > 0:
                avg_stats["avg"] = avg_stats["h"] / avg_stats["ab"]
            # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
            obp_denom = avg_stats["ab"] + avg_stats["bb"] + avg_stats["hbp"] + avg_stats["sf"]
            if obp_denom > 0:
                avg_stats["obp"] = (avg_stats["h"] + avg_stats["bb"] + avg_stats["hbp"]) / obp_denom
            if avg_stats["ip"] > 0:
                avg_stats["era"] = avg_stats["er"] * 9 / avg_stats["ip"]
                avg_stats["whip"] = (avg_stats["bb_allowed"] + avg_stats["h_allowed"]) / avg_stats["ip"]

            cursor.execute(
                """
                INSERT INTO projections (player_id, system, pa, ab, h, r, hr, rbi, sb, avg,
                                         obp, bb, hbp, sf,
                                         ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed,
                                         fip, xfip, siera)
                VALUES (?, 'blended', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    round(avg_stats["obp"], 3),
                    int(avg_stats["bb"]),
                    int(avg_stats["hbp"]),
                    int(avg_stats["sf"]),
                    round(avg_stats["ip"], 1),
                    int(avg_stats["w"]),
                    int(avg_stats["l"]),
                    int(avg_stats["sv"]),
                    int(avg_stats["k"]),
                    round(avg_stats["era"], 2),
                    round(avg_stats["whip"], 2),
                    int(avg_stats["er"]),
                    int(avg_stats["bb_allowed"]),
                    int(avg_stats["h_allowed"]),
                    round(avg_stats["fip"], 2),
                    round(avg_stats["xfip"], 2),
                    round(avg_stats["siera"], 2),
                ),
            )

        conn.commit()
    finally:
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
    try:
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
    finally:
        conn.close()
    return imported


def load_player_pool() -> pd.DataFrame:
    """Load the full player pool with blended projections and ADP for the valuation engine.

    Results are cached for 5 minutes when running inside Streamlit to avoid
    re-executing the ~6s query on every page interaction.
    """
    try:
        import streamlit as st
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        # Only use caching when running inside a real Streamlit app,
        # not in pytest or other non-Streamlit contexts.
        if get_script_run_ctx() is not None:

            @st.cache_data(ttl=300, show_spinner=False)
            def _cached_load(_db_path: str):
                return _load_player_pool_impl()

            return _cached_load(str(DB_PATH))
    except (ImportError, Exception):
        pass
    return _load_player_pool_impl()


# ── Enrichment helpers for player pool ────────────────────────────────


def _load_health_scores_for_pool() -> dict[int, float]:
    """Load per-player health scores from injury_history with B2/B3 adjustments."""
    try:
        from src.injury_model import (
            age_risk_adjustment,
            compute_health_score,
            injury_type_adjustment,
        )

        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT player_id, games_played, games_available "
                "FROM injury_history WHERE season >= 2024 "
                "ORDER BY player_id, season DESC",
                conn,
            )
            # B2+B3: Load player age, position, and injury notes
            player_info = pd.read_sql_query(
                "SELECT player_id, positions, is_hitter, "
                "CASE WHEN birth_date IS NOT NULL AND birth_date != '' "
                "  THEN CAST((julianday('now') - julianday(birth_date)) / 365.25 AS INTEGER) "
                "  ELSE NULL END AS age, "
                "injury_note "
                "FROM players",
                conn,
            )
        finally:
            conn.close()
        if df.empty:
            return {}

        # Build lookup dicts for age/position/injury
        _info = {}
        for _, row in player_info.iterrows():
            _info[int(row["player_id"])] = {
                "age": row.get("age"),
                "positions": str(row.get("positions", "")),
                "is_hitter": int(row.get("is_hitter", 1)),
                "injury_note": str(row.get("injury_note", "") or ""),
            }

        scores: dict[int, float] = {}
        for pid, group in df.groupby("player_id"):
            pid_int = int(pid)
            gp = group["games_played"].tolist()[:3]
            ga = group["games_available"].tolist()[:3]
            base_score = compute_health_score(gp, ga)

            info = _info.get(pid_int, {})
            age = info.get("age")
            pos = info.get("positions", "")
            is_pitcher = info.get("is_hitter", 1) == 0

            # B2: Position-specific age adjustment
            age_mult = age_risk_adjustment(age, is_pitcher, pos)
            adjusted = base_score * age_mult

            # B3: Injury-type floor (TJ, hamstring, etc.)
            inj_floor = injury_type_adjustment(info.get("injury_note", ""))
            if inj_floor < 1.0:
                adjusted = min(adjusted, inj_floor)

            scores[pid_int] = adjusted
        return scores
    except Exception:
        return {}


def _load_roster_statuses_for_pool() -> dict[int, str]:
    """Load per-player roster status from league_rosters (no trade_intelligence import)."""
    try:
        conn = get_connection()
        try:
            df = pd.read_sql_query(
                "SELECT player_id, status FROM league_rosters WHERE status IS NOT NULL",
                conn,
            )
        finally:
            conn.close()
        if df.empty:
            return {}
        return dict(
            zip(
                pd.to_numeric(df["player_id"], errors="coerce").fillna(0).astype(int),
                df["status"].astype(str),
            )
        )
    except Exception:
        return {}


# H3: Graduated positional scarcity (C most scarce → OF/1B/DH least)
_SV_SCARCITY_MULT = 1.3
_GRADUATED_SCARCITY = {"C": 1.20, "2B": 1.15, "SS": 1.10, "3B": 1.05}


def _enrich_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Add health_score, status, is_closer, scarcity_mult to player pool.

    Display-only enrichment — does NOT reduce counting stats.
    Trade engines that need stat adjustment still call get_health_adjusted_pool().
    """
    if df.empty:
        return df

    # Health scores from injury_history
    health_scores = _load_health_scores_for_pool()
    df["health_score"] = df["player_id"].map(health_scores).fillna(0.85)

    # Roster status from league_rosters
    roster_statuses = _load_roster_statuses_for_pool()
    df["status"] = df["player_id"].map(roster_statuses).fillna("active")

    # Adjust health_score based on IL/DTD status (display caps only, no stat reduction)
    status_lower = df["status"].str.lower()
    mask_il = status_lower.isin(["il10", "il15", "dl"])
    df.loc[mask_il & (df["health_score"] >= 0.80), "health_score"] = 0.65
    mask_il60 = status_lower.isin(["il60", "out"])
    df.loc[mask_il60 & (df["health_score"] >= 0.60), "health_score"] = 0.40
    mask_dtd = status_lower == "dtd"
    df.loc[mask_dtd & (df["health_score"] >= 0.80), "health_score"] = 0.75

    # Scarcity flags
    sv = pd.to_numeric(df.get("sv", pd.Series(dtype=float)), errors="coerce").fillna(0)
    df["is_closer"] = sv >= 5

    def _scarcity(row):
        if row.get("is_closer", False):
            return _SV_SCARCITY_MULT
        positions = set(p.strip() for p in str(row.get("positions", "")).split(","))
        # H3: graduated scarcity — highest multiplier among eligible positions
        best_mult = 1.0
        for pos, mult in _GRADUATED_SCARCITY.items():
            if pos in positions and mult > best_mult:
                best_mult = mult
        return best_mult

    df["scarcity_mult"] = df.apply(_scarcity, axis=1)

    # xwOBA regression flags
    if "xwoba" in df.columns and "obp" in df.columns:
        # Approximate wOBA from OBP (wOBA ~ OBP * 1.15 for league-average hitters)
        df["woba_approx"] = (
            pd.to_numeric(df.get("obp", 0), errors="coerce").fillna(0) * 1.15
        )
        xwoba = pd.to_numeric(df.get("xwoba", 0), errors="coerce").fillna(0)
        df["xwoba_delta"] = xwoba - df["woba_approx"]
        df["regression_flag"] = ""
        mask_buy = (df["xwoba_delta"] >= 0.030) & (xwoba > 0)
        mask_sell = (df["xwoba_delta"] <= -0.030) & (xwoba > 0)
        df.loc[mask_buy, "regression_flag"] = "BUY_LOW"
        df.loc[mask_sell, "regression_flag"] = "SELL_HIGH"

    # G3: BABIP regression flags (best early-season signal weeks 3-8)
    # Career BABIP ~.300 for league average; deviation >0.030 suggests regression.
    # Hitters only. Requires minimum 30 PA (ytd_pa) to avoid noise.
    _BABIP_LEAGUE_AVG = 0.300
    _BABIP_THRESHOLD = 0.030
    _BABIP_MIN_PA = 30
    df["babip_regression_flag"] = ""
    if "babip" in df.columns:
        babip_val = pd.to_numeric(df.get("babip", 0), errors="coerce").fillna(0)
        ytd_pa = pd.to_numeric(df.get("ytd_pa", 0), errors="coerce").fillna(0)
        is_hitter = df.get("is_hitter", 0) == 1
        has_data = (babip_val > 0) & (ytd_pa >= _BABIP_MIN_PA) & is_hitter
        babip_delta = babip_val - _BABIP_LEAGUE_AVG
        df["babip_delta"] = babip_delta
        # High BABIP = lucky, will regress down = SELL_HIGH
        # Low BABIP = unlucky, will regress up = BUY_LOW
        mask_sell_babip = has_data & (babip_delta >= _BABIP_THRESHOLD)
        mask_buy_babip = has_data & (babip_delta <= -_BABIP_THRESHOLD)
        df.loc[mask_sell_babip, "babip_regression_flag"] = "SELL_HIGH"
        df.loc[mask_buy_babip, "babip_regression_flag"] = "BUY_LOW"

    # G5: Velocity trend signal for pitchers
    # 1.0 mph decline ~ 0.5-0.8 ERA increase (research consensus).
    # Compare current year ff_avg_speed to prior year.
    df["velo_regression_flag"] = ""
    _VELO_DECLINE_THRESHOLD = 1.0  # mph
    try:
        _conn_velo = get_connection()
        try:
            _year = datetime.now(UTC).year
            _prior_year = _year - 1
            _velo_df = pd.read_sql_query(
                f"""
                SELECT sa_curr.player_id,
                       sa_curr.ff_avg_speed AS curr_velo,
                       sa_prev.ff_avg_speed AS prev_velo
                FROM statcast_archive sa_curr
                LEFT JOIN statcast_archive sa_prev
                    ON sa_curr.player_id = sa_prev.player_id AND sa_prev.season = {_prior_year}
                WHERE sa_curr.season = {_year}
                    AND sa_curr.ff_avg_speed IS NOT NULL AND sa_curr.ff_avg_speed > 0
                    AND sa_prev.ff_avg_speed IS NOT NULL AND sa_prev.ff_avg_speed > 0
                """,
                _conn_velo,
            )
            if not _velo_df.empty:
                _velo_map = {}
                for _, _vrow in _velo_df.iterrows():
                    _delta = float(_vrow["curr_velo"]) - float(_vrow["prev_velo"])
                    _velo_map[int(_vrow["player_id"])] = _delta
                df["velo_delta"] = df["player_id"].map(_velo_map).fillna(0.0)
                is_pitcher = df.get("is_hitter", 1) == 0
                # Decline >= 1.0 mph = SELL_HIGH (early warning of ERA spike)
                mask_decline = is_pitcher & (df["velo_delta"] <= -_VELO_DECLINE_THRESHOLD)
                # Gain >= 1.0 mph = BUY_LOW (stuff improvement not yet in ERA)
                mask_gain = is_pitcher & (df["velo_delta"] >= _VELO_DECLINE_THRESHOLD)
                df.loc[mask_decline, "velo_regression_flag"] = "SELL_HIGH"
                df.loc[mask_gain, "velo_regression_flag"] = "BUY_LOW"
        finally:
            _conn_velo.close()
    except Exception:
        pass

    # G2: Stuff+ pitcher regression flags
    # Stuff+ >100 = above-average stuff. If actual ERA >> expected (high Stuff+ but bad ERA),
    # pitcher is likely unlucky and will regress to better performance = BUY_LOW.
    # Conversely, low Stuff+ with great ERA = SELL_HIGH (outperforming stuff).
    _STUFF_THRESHOLD = 15  # Stuff+ deviation from 100 (league avg)
    df["stuff_regression_flag"] = ""
    if "stuff_plus" in df.columns and "era" in df.columns:
        stuff = pd.to_numeric(df.get("stuff_plus", 0), errors="coerce").fillna(0)
        era = pd.to_numeric(df.get("era", 0), errors="coerce").fillna(0)
        ytd_era = pd.to_numeric(df.get("ytd_era", 0), errors="coerce").fillna(0)
        is_pitcher = df.get("is_hitter", 1) == 0
        has_stuff = (stuff > 0) & is_pitcher & (era > 0)
        # Stuff+ > 115 but YTD ERA > proj ERA + 0.50 → bad luck, BUY_LOW
        # Stuff+ < 85 but YTD ERA < proj ERA - 0.50 → good luck, SELL_HIGH
        stuff_buy = has_stuff & (stuff >= 100 + _STUFF_THRESHOLD) & (ytd_era > era + 0.50) & (ytd_era > 0)
        stuff_sell = has_stuff & (stuff <= 100 - _STUFF_THRESHOLD) & (ytd_era < era - 0.50) & (ytd_era > 0)
        df.loc[stuff_buy, "stuff_regression_flag"] = "BUY_LOW"
        df.loc[stuff_sell, "stuff_regression_flag"] = "SELL_HIGH"

    return df


def _load_player_pool_impl() -> pd.DataFrame:
    """Internal implementation — loads player pool from SQLite.

    Projection priority: ros_projections (Bayesian ROS, updated with live
    2026 stats) > projections.blended > any projection system average.
    This ensures all consumers (Trade Finder, Trade Analyzer, Lineup
    Optimizer, Free Agents) use the most current player-level projections.
    """
    conn = get_connection()
    try:
        # Check if ros_projections has data (Bayesian ROS with live stats)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM ros_projections")
        ros_count = cursor.fetchone()[0]

        if ros_count > 0:
            # Prefer Bayesian ROS projections — blended with actual 2026 performance
            df = pd.read_sql_query(
                """
                SELECT
                    p.player_id, p.name, p.team, p.positions, p.is_hitter, p.is_injured,
                    p.mlb_id,
                    CASE WHEN p.birth_date IS NOT NULL AND p.birth_date != ''
                         THEN CAST((julianday('now') - julianday(p.birth_date)) / 365.25 AS INTEGER)
                         ELSE NULL END AS age,
                    COALESCE(ros.pa, 0) as pa, COALESCE(ros.ab, 0) as ab,
                    COALESCE(ros.h, 0) as h, COALESCE(ros.r, 0) as r,
                    COALESCE(ros.hr, 0) as hr, COALESCE(ros.rbi, 0) as rbi,
                    COALESCE(ros.sb, 0) as sb, ros.avg, ros.obp,
                    COALESCE(ros.bb, 0) as bb, COALESCE(ros.hbp, 0) as hbp,
                    COALESCE(ros.sf, 0) as sf,
                    COALESCE(ros.ip, 0) as ip, COALESCE(ros.w, 0) as w,
                    COALESCE(ros.l, 0) as l, COALESCE(ros.sv, 0) as sv,
                    COALESCE(ros.k, 0) as k, ros.era, ros.whip,
                    COALESCE(ros.er, 0) as er, COALESCE(ros.bb_allowed, 0) as bb_allowed,
                    COALESCE(ros.h_allowed, 0) as h_allowed,
                    ros.fip, ros.xfip, ros.siera,
                    COALESCE(a.adp, 999) as adp,
                    ecr.consensus_rank,
                    ecr.consensus_avg AS ecr_avg,
                    ecr.n_sources AS ecr_sources,
                    COALESCE(ss.pa, 0) AS ytd_pa,
                    COALESCE(ss.avg, 0) AS ytd_avg,
                    COALESCE(ss.hr, 0) AS ytd_hr,
                    COALESCE(ss.rbi, 0) AS ytd_rbi,
                    COALESCE(ss.sb, 0) AS ytd_sb,
                    COALESCE(ss.era, 0) AS ytd_era,
                    COALESCE(ss.whip, 0) AS ytd_whip,
                    COALESCE(ss.sv, 0) AS ytd_sv,
                    COALESCE(ss.k, 0) AS ytd_k,
                    sa.xwoba AS xwoba,
                    sa.xba AS xba,
                    sa.barrel_pct AS barrel_pct,
                    sa.hard_hit_pct AS hard_hit_pct,
                    sa.ev_mean AS ev_mean,
                    sa.stuff_plus AS stuff_plus,
                    sa.babip AS babip
                FROM players p
                LEFT JOIN ros_projections ros ON p.player_id = ros.player_id
                LEFT JOIN adp a ON p.player_id = a.player_id
                LEFT JOIN ecr_consensus ecr ON p.player_id = ecr.player_id
                LEFT JOIN season_stats ss ON p.player_id = ss.player_id AND ss.season = 2026
                LEFT JOIN statcast_archive sa ON p.player_id = sa.player_id AND sa.season = 2026
                ORDER BY COALESCE(a.adp, 999)
            """,
                conn,
            )
            if not df.empty:
                return _enrich_pool(coerce_numeric_df(df))

        # Fallback: static blended projections (pre-season Steamer/ZiPS/DepthCharts)
        # Note: Do NOT use CAST on numeric columns — Python 3.14 SQLite returns raw bytes
        # for NumPy integers, and CAST corrupts them. Fix bytes in Python after loading.
        df = pd.read_sql_query(
            """
            SELECT
                p.player_id, p.name, p.team, p.positions, p.is_hitter, p.is_injured,
                p.mlb_id,
                CASE WHEN p.birth_date IS NOT NULL AND p.birth_date != ''
                     THEN CAST((julianday('now') - julianday(p.birth_date)) / 365.25 AS INTEGER)
                     ELSE NULL END AS age,
                COALESCE(proj.pa, 0) as pa, COALESCE(proj.ab, 0) as ab,
                COALESCE(proj.h, 0) as h, COALESCE(proj.r, 0) as r,
                COALESCE(proj.hr, 0) as hr, COALESCE(proj.rbi, 0) as rbi,
                COALESCE(proj.sb, 0) as sb, proj.avg, proj.obp,
                COALESCE(proj.bb, 0) as bb, COALESCE(proj.hbp, 0) as hbp,
                COALESCE(proj.sf, 0) as sf,
                COALESCE(proj.ip, 0) as ip, COALESCE(proj.w, 0) as w,
                COALESCE(proj.l, 0) as l, COALESCE(proj.sv, 0) as sv,
                COALESCE(proj.k, 0) as k, proj.era, proj.whip,
                COALESCE(proj.er, 0) as er, COALESCE(proj.bb_allowed, 0) as bb_allowed,
                COALESCE(proj.h_allowed, 0) as h_allowed,
                proj.fip, proj.xfip, proj.siera,
                COALESCE(a.adp, 999) as adp,
                ecr.consensus_rank,
                ecr.consensus_avg AS ecr_avg,
                ecr.n_sources AS ecr_sources,
                COALESCE(ss.pa, 0) AS ytd_pa,
                COALESCE(ss.avg, 0) AS ytd_avg,
                COALESCE(ss.hr, 0) AS ytd_hr,
                COALESCE(ss.rbi, 0) AS ytd_rbi,
                COALESCE(ss.sb, 0) AS ytd_sb,
                COALESCE(ss.era, 0) AS ytd_era,
                COALESCE(ss.whip, 0) AS ytd_whip,
                COALESCE(ss.sv, 0) AS ytd_sv,
                COALESCE(ss.k, 0) AS ytd_k,
                sa.xwoba AS xwoba,
                sa.xba AS xba,
                sa.barrel_pct AS barrel_pct,
                sa.hard_hit_pct AS hard_hit_pct,
                sa.ev_mean AS ev_mean,
                sa.stuff_plus AS stuff_plus,
                sa.babip AS babip
            FROM players p
            LEFT JOIN projections proj ON p.player_id = proj.player_id
                AND proj.system = 'blended'
            LEFT JOIN adp a ON p.player_id = a.player_id
            LEFT JOIN ecr_consensus ecr ON p.player_id = ecr.player_id
            LEFT JOIN season_stats ss ON p.player_id = ss.player_id AND ss.season = 2026
            LEFT JOIN statcast_archive sa ON p.player_id = sa.player_id AND sa.season = 2026
            ORDER BY COALESCE(a.adp, 999)
        """,
            conn,
        )

        # If no blended projections exist, try averaging any available system
        if df.empty:
            df = pd.read_sql_query(
                """
                SELECT
                    p.player_id, p.name, p.team, p.positions, p.is_hitter, p.is_injured,
                    p.mlb_id,
                    CASE WHEN p.birth_date IS NOT NULL AND p.birth_date != ''
                         THEN CAST((julianday('now') - julianday(p.birth_date)) / 365.25 AS INTEGER)
                         ELSE NULL END AS age,
                    COALESCE(AVG(proj.pa), 0) as pa, COALESCE(AVG(proj.ab), 0) as ab,
                    COALESCE(AVG(proj.h), 0) as h, COALESCE(AVG(proj.r), 0) as r,
                    COALESCE(AVG(proj.hr), 0) as hr, COALESCE(AVG(proj.rbi), 0) as rbi,
                    COALESCE(AVG(proj.sb), 0) as sb, AVG(proj.avg) as avg,
                    AVG(proj.obp) as obp,
                    COALESCE(AVG(proj.bb), 0) as bb, COALESCE(AVG(proj.hbp), 0) as hbp,
                    COALESCE(AVG(proj.sf), 0) as sf,
                    COALESCE(AVG(proj.ip), 0) as ip, COALESCE(AVG(proj.w), 0) as w,
                    COALESCE(AVG(proj.l), 0) as l, COALESCE(AVG(proj.sv), 0) as sv,
                    COALESCE(AVG(proj.k), 0) as k, AVG(proj.era) as era,
                    AVG(proj.whip) as whip, COALESCE(AVG(proj.er), 0) as er,
                    COALESCE(AVG(proj.bb_allowed), 0) as bb_allowed,
                    COALESCE(AVG(proj.h_allowed), 0) as h_allowed,
                    AVG(proj.fip) as fip, AVG(proj.xfip) as xfip,
                    AVG(proj.siera) as siera,
                    COALESCE(a.adp, 999) as adp,
                    ecr.consensus_rank,
                    ecr.consensus_avg AS ecr_avg,
                    ecr.n_sources AS ecr_sources,
                    COALESCE(ss.pa, 0) AS ytd_pa,
                    COALESCE(ss.avg, 0) AS ytd_avg,
                    COALESCE(ss.hr, 0) AS ytd_hr,
                    COALESCE(ss.rbi, 0) AS ytd_rbi,
                    COALESCE(ss.sb, 0) AS ytd_sb,
                    COALESCE(ss.era, 0) AS ytd_era,
                    COALESCE(ss.whip, 0) AS ytd_whip,
                    COALESCE(ss.sv, 0) AS ytd_sv,
                    COALESCE(ss.k, 0) AS ytd_k,
                    sa.xwoba AS xwoba,
                    sa.xba AS xba,
                    sa.barrel_pct AS barrel_pct,
                    sa.hard_hit_pct AS hard_hit_pct,
                    sa.ev_mean AS ev_mean
                FROM players p
                LEFT JOIN projections proj ON p.player_id = proj.player_id
                LEFT JOIN adp a ON p.player_id = a.player_id
                LEFT JOIN ecr_consensus ecr ON p.player_id = ecr.player_id
                LEFT JOIN season_stats ss ON p.player_id = ss.player_id AND ss.season = 2026
                LEFT JOIN statcast_archive sa ON p.player_id = sa.player_id AND sa.season = 2026
                GROUP BY p.player_id
                ORDER BY COALESCE(a.adp, 999)
            """,
                conn,
            )
    finally:
        conn.close()

    # Fix Python 3.13+ SQLite bytes issue + enrichment (health, scarcity)
    df = _enrich_pool(coerce_numeric_df(df))

    return df


# ── In-Season Data Functions ──────────────────────────────────────────

# Whitelist of valid column names for SQL interpolation safety
_VALID_STAT_COLUMNS = frozenset(
    [
        "pa",
        "ab",
        "h",
        "r",
        "hr",
        "rbi",
        "sb",
        "avg",
        "obp",
        "ip",
        "w",
        "l",
        "sv",
        "k",
        "era",
        "whip",
        "er",
        "bb",
        "bb_allowed",
        "h_allowed",
        "hbp",
        "sf",
        "fip",
        "xfip",
        "siera",
        "games_played",
    ]
)


def _validate_columns(cols: list) -> list:
    """Validate column names against whitelist to prevent SQL injection."""
    for c in cols:
        if c not in _VALID_STAT_COLUMNS:
            raise ValueError(f"Invalid column name: {c!r}")
    return cols


def upsert_season_stats(player_id: int, stats: dict, season: int = 2026):
    """Insert or update a player's season stats."""
    conn = get_connection()
    try:
        cols = [
            "pa",
            "ab",
            "h",
            "r",
            "hr",
            "rbi",
            "sb",
            "avg",
            "obp",
            "bb",
            "hbp",
            "sf",
            "ip",
            "w",
            "l",
            "sv",
            "k",
            "era",
            "whip",
            "er",
            "bb_allowed",
            "h_allowed",
            "fip",
            "xfip",
            "siera",
            "games_played",
        ]
        _validate_columns(cols)
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
    finally:
        conn.close()


def load_season_stats(season: int = 2026) -> pd.DataFrame:
    """Load all season stats as a DataFrame."""
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM season_stats WHERE season = ?", conn, params=(season,))
    finally:
        conn.close()
    return coerce_numeric_df(df)


def upsert_ros_projection(player_id: int, system: str, stats: dict):
    """Insert or update a ROS projection for a player."""
    conn = get_connection()
    try:
        cols = [
            "pa",
            "ab",
            "h",
            "r",
            "hr",
            "rbi",
            "sb",
            "avg",
            "obp",
            "bb",
            "hbp",
            "sf",
            "ip",
            "w",
            "l",
            "sv",
            "k",
            "era",
            "whip",
            "er",
            "bb_allowed",
            "h_allowed",
            "fip",
            "xfip",
            "siera",
        ]
        _validate_columns(cols)
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
    finally:
        conn.close()


def upsert_league_roster_entry(
    team_name: str,
    team_index: int,
    player_id: int,
    roster_slot: str = None,
    is_user_team: bool = False,
    status: str = "active",
    selected_position: str = "",
    editorial_team_abbr: str = "",
):
    """Add a player to a league roster.

    Args:
        status: Yahoo roster status — ``active``, ``IL10``, ``IL15``,
            ``IL60``, ``DTD``, ``NA``, or other Yahoo status strings.
        selected_position: The lineup slot the manager assigned (e.g. ``C``,
            ``1B``, ``BN``, ``IL``).
        editorial_team_abbr: MLB team abbreviation from Yahoo (e.g. ``NYY``).
    """
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO league_rosters
               (team_name, team_index, player_id, roster_slot, is_user_team, status,
                selected_position, editorial_team_abbr)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(team_name, player_id) DO UPDATE SET
               team_index=excluded.team_index, roster_slot=excluded.roster_slot,
               is_user_team=excluded.is_user_team, status=excluded.status,
               selected_position=excluded.selected_position,
               editorial_team_abbr=excluded.editorial_team_abbr""",
            (
                team_name,
                team_index,
                player_id,
                roster_slot,
                1 if is_user_team else 0,
                status or "active",
                selected_position or "",
                editorial_team_abbr or "",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_league_rosters() -> pd.DataFrame:
    """Load all league rosters as a DataFrame."""
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM league_rosters", conn)
    finally:
        conn.close()
    return coerce_numeric_df(df)


def get_all_rostered_player_ids(league_rosters_df: pd.DataFrame | None = None) -> set:
    """Return set of all player IDs rostered by any team in the league.

    Use this to filter player pools down to true free agents. Covers all
    12 teams, not just the user's team.

    Args:
        league_rosters_df: Optional pre-loaded league_rosters DataFrame.
            If None, loads from database via ``load_league_rosters()``.

    Returns:
        Set of player_id values on any team roster.
    """
    if league_rosters_df is None:
        league_rosters_df = load_league_rosters()
    if league_rosters_df.empty:
        return set()
    return set(league_rosters_df["player_id"].dropna().values)


def load_league_standings() -> pd.DataFrame:
    """Load all league standings as a DataFrame."""
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM league_standings", conn)
    finally:
        conn.close()
    return coerce_numeric_df(df)


def clear_league_rosters():
    """Remove all league roster entries (for re-import)."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM league_rosters")
        conn.commit()
    finally:
        conn.close()


def upsert_league_standing(team_name: str, category: str, total: float, rank: int = None, points: float = None):
    """Insert or update a league standing entry."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO league_standings (team_name, category, total, rank, points)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(team_name, category) DO UPDATE SET
               total=excluded.total, rank=excluded.rank, points=excluded.points""",
            (team_name, category, total, rank, points),
        )
        conn.commit()
    finally:
        conn.close()


def update_refresh_log(source: str, status: str = "success"):
    """Update the refresh log for a data source."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO refresh_log (source, last_refresh, status)
               VALUES (?, ?, ?)
               ON CONFLICT(source) DO UPDATE SET
               last_refresh=excluded.last_refresh, status=excluded.status""",
            (source, datetime.now(UTC).isoformat(), status),
        )
        conn.commit()
    finally:
        conn.close()


def get_refresh_status(source: str) -> dict | None:
    """Get the last refresh status for a data source."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT source, last_refresh, status FROM refresh_log WHERE source = ?",
            (source,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"source": row[0], "last_refresh": row[1], "status": row[2]}


def check_staleness(source: str, max_age_hours: float) -> bool:
    """Return True if source needs refresh (no record or older than max_age_hours)."""
    if max_age_hours <= 0:
        return True
    status = get_refresh_status(source)
    if status is None or status["last_refresh"] is None:
        return True
    try:
        last = datetime.fromisoformat(status["last_refresh"])
        if last.tzinfo is None:
            last = last.replace(tzinfo=UTC)
        age_hours = (datetime.now(UTC) - last).total_seconds() / 3600
        return age_hours > max_age_hours
    except (ValueError, TypeError):
        return True


def upsert_league_schedule(week: int, team_name: str, opponent_name: str) -> None:
    """Insert or update a single week's matchup in the league_schedule table.

    Args:
        week: Week number (1-24).
        team_name: The team whose schedule entry this is.
        opponent_name: The opponent for that week.
    """
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO league_schedule (week, team_a, team_b) VALUES (?, ?, ?)",
            (week, team_name, opponent_name),
        )
        conn.commit()
    finally:
        conn.close()


def load_league_schedule() -> dict[int, str]:
    """Load the league schedule from the DB.

    Returns:
        Dict mapping week number to opponent name, e.g. ``{1: "Team Foo", 2: "Team Bar"}``.
        Returns from the user team's perspective (team_a = user team).
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        # Get the user team name to filter schedule
        cursor.execute("SELECT team_name FROM league_teams WHERE is_user_team = 1")
        user_row = cursor.fetchone()
        if not user_row:
            # No user team identified — return empty to let callers use fallback
            return {}

        user_team = user_row[0]
        cursor.execute(
            "SELECT week, team_b FROM league_schedule WHERE team_a = ? ORDER BY week",
            (user_team,),
        )
        rows = cursor.fetchall()
        return {int(row[0]): row[1] for row in rows}
    finally:
        conn.close()


def upsert_league_schedule_full(week: int, team_a: str, team_b: str) -> None:
    """Insert or replace a full-league matchup entry."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO league_schedule_full (week, team_a, team_b) VALUES (?, ?, ?)",
            (week, team_a, team_b),
        )
        conn.commit()
    finally:
        conn.close()


def load_league_schedule_full() -> dict[int, list[tuple[str, str]]]:
    """Load full league schedule: {week: [(team_a, team_b), ...]}."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT week, team_a, team_b FROM league_schedule_full ORDER BY week").fetchall()
    finally:
        conn.close()
    result: dict[int, list[tuple[str, str]]] = {}
    for week, team_a, team_b in rows:
        week = int(week)
        result.setdefault(week, []).append((str(team_a), str(team_b)))
    return result


def upsert_league_record(
    team_name: str,
    wins: int = 0,
    losses: int = 0,
    ties: int = 0,
    win_pct: float = 0.0,
    points_for: float = 0.0,
    points_against: float = 0.0,
    streak: str = "",
    rank: int = 0,
) -> None:
    """Insert or replace a team's W-L-T record."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO league_records
               (team_name, wins, losses, ties, win_pct, points_for,
                points_against, streak, rank, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                team_name,
                wins,
                losses,
                ties,
                win_pct,
                points_for,
                points_against,
                streak,
                rank,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_league_records() -> pd.DataFrame:
    """Load all team W-L-T records."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM league_records ORDER BY rank",
            conn,
        )
    finally:
        conn.close()
    if df.empty:
        return pd.DataFrame(
            columns=[
                "team_name",
                "wins",
                "losses",
                "ties",
                "win_pct",
                "points_for",
                "points_against",
                "streak",
                "rank",
                "updated_at",
            ]
        )
    for col in ("wins", "losses", "ties", "rank"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["win_pct"] = pd.to_numeric(df["win_pct"], errors="coerce").fillna(0.0)
    return df


def save_league_draft_picks(draft_df: "pd.DataFrame") -> int:
    """Store league-specific draft results.

    Args:
        draft_df: DataFrame from yahoo_api.get_draft_results() with columns
            pick_number, round, team_name, player_name, player_id.

    Returns:
        Number of picks stored.
    """
    if draft_df is None or draft_df.empty:
        return 0
    conn = get_connection()
    count = 0
    try:
        for _, row in draft_df.iterrows():
            pick = int(row.get("pick_number", 0))
            rd = int(row.get("round", 0))
            team = str(row.get("team_name", ""))
            name = str(row.get("player_name", ""))
            if not pick or not name:
                continue

            # Resolve to local player_id
            pid = None
            try:
                from src.live_stats import match_player_id

                pid = match_player_id(name, "")
            except Exception:
                pass

            try:
                conn.execute(
                    "INSERT OR REPLACE INTO league_draft_picks "
                    "(pick_number, round, team_name, player_id, player_name) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (pick, rd, team, pid, name),
                )
                count += 1
            except Exception:
                pass
        conn.commit()
    finally:
        conn.close()
    return count


def get_player_draft_round(player_id: int) -> int | None:
    """Get the round a player was drafted in your league.

    Returns:
        Round number (1-23), or None if undrafted/not found.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT round FROM league_draft_picks WHERE player_id = ?",
            (player_id,),
        )
        row = cursor.fetchone()
        return int(row[0]) if row else None
    finally:
        conn.close()


def compute_ownership_deltas(lookback_days: int = 7) -> int:
    """Compute ownership % change over the last N days.

    Updates the delta_7d column in ownership_trends for the most recent date.

    Args:
        lookback_days: Number of days to look back for delta calculation.

    Returns:
        Number of rows updated.
    """
    from datetime import UTC, datetime, timedelta

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    past = (datetime.now(UTC) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """UPDATE ownership_trends SET delta_7d = (
                SELECT ot.percent_owned - COALESCE(
                    (SELECT o2.percent_owned FROM ownership_trends o2
                     WHERE o2.player_id = ot.player_id AND o2.date <= ?
                     ORDER BY o2.date DESC LIMIT 1), ot.percent_owned
                )
                FROM ownership_trends ot
                WHERE ot.player_id = ownership_trends.player_id
                  AND ot.date = ownership_trends.date
            ) WHERE date = ?""",
            (past, today),
        )
        count = cursor.rowcount
        conn.commit()
        return count
    finally:
        conn.close()


def save_mlb_transactions(df: pd.DataFrame) -> int:
    """Store MLB transactions in the transactions table.

    Args:
        df: DataFrame from fetch_mlb_transactions().

    Returns:
        Number of rows inserted.
    """
    if df is None or df.empty:
        return 0

    conn = get_connection()
    count = 0
    try:
        from datetime import UTC, datetime

        now = datetime.now(UTC).isoformat()
        for _, row in df.iterrows():
            player_name = row.get("player_name", "")
            if not player_name:
                continue

            # Try to resolve player_id
            from src.live_stats import match_player_id

            pid = match_player_id(player_name, row.get("to_team", ""))
            if pid is None:
                pid = match_player_id(player_name, row.get("from_team", ""))
            if pid is None:
                continue

            try:
                conn.execute(
                    """INSERT OR IGNORE INTO transactions
                       (player_id, type, team_from, team_to, timestamp, created_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        pid,
                        row.get("type_desc", ""),
                        row.get("from_team", ""),
                        row.get("to_team", ""),
                        row.get("date", ""),
                        now,
                    ),
                )
                count += 1
            except Exception:
                pass
        conn.commit()
    finally:
        conn.close()
    return count


def snapshot_league_rosters() -> int:
    """Take a snapshot of current league_rosters for change tracking."""
    from datetime import UTC, datetime

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT team_name, player_id, roster_slot FROM league_rosters")
        rows = cursor.fetchall()
        count = 0
        for team_name, player_id, slot in rows:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO roster_snapshots "
                    "(snapshot_date, team_name, player_id, roster_slot) "
                    "VALUES (?, ?, ?, ?)",
                    (today, team_name, int(player_id), slot),
                )
                count += 1
            except Exception:
                pass
        conn.commit()
        return count
    finally:
        conn.close()


def get_roster_changes(team_name: str, days: int = 7) -> list[dict]:
    """Detect roster adds/drops for a team over the last N days."""
    from datetime import UTC, datetime, timedelta

    cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT MIN(snapshot_date), MAX(snapshot_date) FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date >= ?",
            (team_name, cutoff),
        )
        row = cursor.fetchone()
        if not row or row[0] is None or row[1] is None or row[0] == row[1]:
            return []

        earliest, latest = row

        cursor.execute(
            "SELECT player_id FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date = ? "
            "EXCEPT "
            "SELECT player_id FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date = ?",
            (team_name, latest, team_name, earliest),
        )
        adds = [
            {
                "change_type": "add",
                "player_id": int(r[0]),
                "team_name": team_name,
                "detected_date": latest,
            }
            for r in cursor.fetchall()
        ]

        cursor.execute(
            "SELECT player_id FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date = ? "
            "EXCEPT "
            "SELECT player_id FROM roster_snapshots "
            "WHERE team_name = ? AND snapshot_date = ?",
            (team_name, earliest, team_name, latest),
        )
        drops = [
            {
                "change_type": "drop",
                "player_id": int(r[0]),
                "team_name": team_name,
                "detected_date": latest,
            }
            for r in cursor.fetchall()
        ]

        return adds + drops
    finally:
        conn.close()


def upsert_player_bulk(players: list[dict]) -> int:
    """Bulk upsert players. Each dict needs: name, team, positions, is_hitter.

    Optional: mlb_id, bats, throws, birth_date, roster_type.

    Uses SELECT-first approach since players table has no UNIQUE constraint on name.
    """
    conn = get_connection()
    try:
        saved = 0
        for p in players:
            existing = conn.execute("SELECT player_id FROM players WHERE name = ?", (p["name"],)).fetchone()
            mlb_id = p.get("mlb_id")
            bats = p.get("bats")
            throws = p.get("throws")
            birth_date = p.get("birth_date")
            roster_type = p.get("roster_type", "active")
            if existing:
                conn.execute(
                    """UPDATE players SET team = ?, positions = ?, is_hitter = ?,
                       mlb_id = COALESCE(?, mlb_id),
                       bats = COALESCE(?, bats),
                       throws = COALESCE(?, throws),
                       birth_date = COALESCE(?, birth_date),
                       roster_type = COALESCE(?, roster_type)
                       WHERE player_id = ?""",
                    (
                        p["team"],
                        p["positions"],
                        int(p["is_hitter"]),
                        mlb_id,
                        bats,
                        throws,
                        birth_date,
                        roster_type,
                        existing[0],
                    ),
                )
            else:
                conn.execute(
                    """INSERT INTO players (name, team, positions, is_hitter, is_injured,
                       mlb_id, bats, throws, birth_date, roster_type)
                       VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?)""",
                    (
                        p["name"],
                        p["team"],
                        p["positions"],
                        int(p["is_hitter"]),
                        mlb_id,
                        bats,
                        throws,
                        birth_date,
                        roster_type,
                    ),
                )
            saved += 1
        conn.commit()
        return saved
    finally:
        conn.close()


def deduplicate_players() -> dict[str, int]:
    """Find and merge duplicate player entries in the players table.

    When the same player has multiple rows (from different insertion paths like
    MLB Stats API vs FanGraphs), this function:
    1. Identifies duplicates by name (case-insensitive)
    2. Picks the canonical entry (the one with projections, or lowest player_id)
    3. Remaps all FK references in dependent tables to the canonical ID
    4. Deletes the duplicate rows

    Returns:
        Dict with keys: duplicates_found, players_merged, fk_remapped
    """
    import logging

    logger = logging.getLogger(__name__)
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Step 1: Find all names with more than one player_id
        cursor.execute("""
            SELECT LOWER(TRIM(name)) as norm_name, COUNT(*) as cnt
            FROM players
            GROUP BY norm_name
            HAVING cnt > 1
        """)
        duplicates = cursor.fetchall()
        if not duplicates:
            logger.info("No duplicate players found.")
            return {"duplicates_found": 0, "players_merged": 0, "fk_remapped": 0}

        dup_names = [row[0] for row in duplicates]
        logger.info("Found %d duplicate player names: %s", len(dup_names), dup_names[:10])

        total_merged = 0
        total_remapped = 0

        for norm_name in dup_names:
            # Get all player_ids for this name
            cursor.execute(
                "SELECT player_id FROM players WHERE LOWER(TRIM(name)) = ? ORDER BY player_id",
                (norm_name,),
            )
            ids = [row[0] for row in cursor.fetchall()]
            if len(ids) < 2:
                continue

            # Step 2: Pick canonical — prefer the one with projections (blended first)
            canonical_id = ids[0]  # default to lowest
            for pid in ids:
                cursor.execute(
                    "SELECT COUNT(*) FROM projections WHERE player_id = ? AND system = 'blended'",
                    (pid,),
                )
                if cursor.fetchone()[0] > 0:
                    canonical_id = pid
                    break
            else:
                # No blended — try any projection system
                for pid in ids:
                    cursor.execute(
                        "SELECT COUNT(*) FROM projections WHERE player_id = ?",
                        (pid,),
                    )
                    if cursor.fetchone()[0] > 0:
                        canonical_id = pid
                        break

            duplicate_ids = [pid for pid in ids if pid != canonical_id]
            if not duplicate_ids:
                continue

            logger.info(
                "Merging player '%s': canonical=%d, duplicates=%s",
                norm_name,
                canonical_id,
                duplicate_ids,
            )

            # Step 3: Remap FK references in all dependent tables
            # Tables with simple player_id FK (safe UPDATE)
            simple_fk_tables = [
                "league_rosters",
                "transactions",
                "player_news",
            ]
            for table in simple_fk_tables:
                for dup_id in duplicate_ids:
                    cursor.execute(
                        f"UPDATE {table} SET player_id = ? WHERE player_id = ?",
                        (canonical_id, dup_id),
                    )
                    total_remapped += cursor.rowcount

            # Handle adp table specially (has PRIMARY KEY on player_id)
            for dup_id in duplicate_ids:
                cursor.execute("SELECT 1 FROM adp WHERE player_id = ?", (canonical_id,))
                if cursor.fetchone():
                    cursor.execute("DELETE FROM adp WHERE player_id = ?", (dup_id,))
                else:
                    cursor.execute(
                        "UPDATE adp SET player_id = ? WHERE player_id = ?",
                        (canonical_id, dup_id),
                    )
                total_remapped += cursor.rowcount

            # Handle ecr_consensus and player_id_map (PRIMARY KEY on player_id)
            for pk_table in ["ecr_consensus", "player_id_map"]:
                for dup_id in duplicate_ids:
                    cursor.execute(
                        f"SELECT 1 FROM {pk_table} WHERE player_id = ?",
                        (canonical_id,),
                    )
                    if cursor.fetchone():
                        cursor.execute(
                            f"DELETE FROM {pk_table} WHERE player_id = ?",
                            (dup_id,),
                        )
                    else:
                        cursor.execute(
                            f"UPDATE {pk_table} SET player_id = ? WHERE player_id = ?",
                            (canonical_id, dup_id),
                        )
                    total_remapped += cursor.rowcount

            # player_tags: PRIMARY KEY (player_id, tag) — may conflict
            for dup_id in duplicate_ids:
                cursor.execute(
                    "SELECT player_id, tag FROM player_tags WHERE player_id = ?",
                    (dup_id,),
                )
                dup_rows = cursor.fetchall()
                for _, tag in dup_rows:
                    cursor.execute(
                        "SELECT 1 FROM player_tags WHERE player_id = ? AND tag = ?",
                        (canonical_id, tag),
                    )
                    if cursor.fetchone() is None:
                        cursor.execute(
                            "UPDATE player_tags SET player_id = ? WHERE player_id = ? AND tag = ?",
                            (canonical_id, dup_id, tag),
                        )
                        total_remapped += 1
                    else:
                        cursor.execute(
                            "DELETE FROM player_tags WHERE player_id = ? AND tag = ?",
                            (dup_id, tag),
                        )

            # ownership_trends: PRIMARY KEY (player_id, date) — may conflict
            for dup_id in duplicate_ids:
                cursor.execute(
                    "SELECT player_id, date FROM ownership_trends WHERE player_id = ?",
                    (dup_id,),
                )
                dup_rows = cursor.fetchall()
                for _, date_val in dup_rows:
                    cursor.execute(
                        "SELECT 1 FROM ownership_trends WHERE player_id = ? AND date = ?",
                        (canonical_id, date_val),
                    )
                    if cursor.fetchone() is None:
                        cursor.execute(
                            "UPDATE ownership_trends SET player_id = ? WHERE player_id = ? AND date = ?",
                            (canonical_id, dup_id, date_val),
                        )
                        total_remapped += 1
                    else:
                        cursor.execute(
                            "DELETE FROM ownership_trends WHERE player_id = ? AND date = ?",
                            (dup_id, date_val),
                        )

            # statcast_archive: UNIQUE(player_id, season) — may conflict
            for dup_id in duplicate_ids:
                cursor.execute(
                    "SELECT player_id, season FROM statcast_archive WHERE player_id = ?",
                    (dup_id,),
                )
                dup_rows = cursor.fetchall()
                for _, season in dup_rows:
                    cursor.execute(
                        "SELECT 1 FROM statcast_archive WHERE player_id = ? AND season = ?",
                        (canonical_id, season),
                    )
                    if cursor.fetchone() is None:
                        cursor.execute(
                            "UPDATE statcast_archive SET player_id = ? WHERE player_id = ? AND season = ?",
                            (canonical_id, dup_id, season),
                        )
                        total_remapped += 1
                    else:
                        cursor.execute(
                            "DELETE FROM statcast_archive WHERE player_id = ? AND season = ?",
                            (dup_id, season),
                        )

            # Tables with composite PK including player_id — need conflict handling
            # projections: (id PK autoincrement, player_id FK) — safe UPDATE
            for dup_id in duplicate_ids:
                cursor.execute(
                    "UPDATE projections SET player_id = ? WHERE player_id = ?",
                    (canonical_id, dup_id),
                )
                total_remapped += cursor.rowcount

            # season_stats: PRIMARY KEY (player_id, season) — may conflict
            for dup_id in duplicate_ids:
                cursor.execute(
                    "SELECT player_id, season FROM season_stats WHERE player_id = ?",
                    (dup_id,),
                )
                dup_rows = cursor.fetchall()
                for _, season in dup_rows:
                    # Check if canonical already has this season
                    cursor.execute(
                        "SELECT 1 FROM season_stats WHERE player_id = ? AND season = ?",
                        (canonical_id, season),
                    )
                    if cursor.fetchone() is None:
                        # Safe to remap
                        cursor.execute(
                            "UPDATE season_stats SET player_id = ? WHERE player_id = ? AND season = ?",
                            (canonical_id, dup_id, season),
                        )
                        total_remapped += 1
                    else:
                        # Canonical already has data for this season — delete duplicate
                        cursor.execute(
                            "DELETE FROM season_stats WHERE player_id = ? AND season = ?",
                            (dup_id, season),
                        )

            # ros_projections: PRIMARY KEY (player_id, system) — may conflict
            for dup_id in duplicate_ids:
                cursor.execute(
                    "SELECT player_id, system FROM ros_projections WHERE player_id = ?",
                    (dup_id,),
                )
                dup_rows = cursor.fetchall()
                for _, system in dup_rows:
                    cursor.execute(
                        "SELECT 1 FROM ros_projections WHERE player_id = ? AND system = ?",
                        (canonical_id, system),
                    )
                    if cursor.fetchone() is None:
                        cursor.execute(
                            "UPDATE ros_projections SET player_id = ? WHERE player_id = ? AND system = ?",
                            (canonical_id, dup_id, system),
                        )
                        total_remapped += 1
                    else:
                        cursor.execute(
                            "DELETE FROM ros_projections WHERE player_id = ? AND system = ?",
                            (dup_id, system),
                        )

            # injury_history: UNIQUE(player_id, season) — may conflict
            for dup_id in duplicate_ids:
                cursor.execute(
                    "SELECT player_id, season FROM injury_history WHERE player_id = ?",
                    (dup_id,),
                )
                dup_rows = cursor.fetchall()
                for _, season in dup_rows:
                    cursor.execute(
                        "SELECT 1 FROM injury_history WHERE player_id = ? AND season = ?",
                        (canonical_id, season),
                    )
                    if cursor.fetchone() is None:
                        cursor.execute(
                            "UPDATE injury_history SET player_id = ? WHERE player_id = ? AND season = ?",
                            (canonical_id, dup_id, season),
                        )
                        total_remapped += 1
                    else:
                        cursor.execute(
                            "DELETE FROM injury_history WHERE player_id = ? AND season = ?",
                            (dup_id, season),
                        )

            # Step 4: Merge any useful data from duplicates into canonical
            # (e.g., mlb_id, team info, positions)
            for dup_id in duplicate_ids:
                cursor.execute(
                    "SELECT team, positions, mlb_id FROM players WHERE player_id = ?",
                    (dup_id,),
                )
                dup_data = cursor.fetchone()
                if dup_data:
                    dup_team, dup_positions, dup_mlb_id = dup_data
                    # Merge positions
                    cursor.execute(
                        "SELECT positions, mlb_id, team FROM players WHERE player_id = ?",
                        (canonical_id,),
                    )
                    canon_data = cursor.fetchone()
                    if canon_data:
                        canon_pos = set(canon_data[0].split(",")) if canon_data[0] else set()
                        dup_pos = set(dup_positions.split(",")) if dup_positions else set()
                        merged_pos = ",".join(sorted(canon_pos | dup_pos))
                        # Take mlb_id if canonical doesn't have one
                        new_mlb_id = canon_data[1] or dup_mlb_id
                        # Prefer non-empty team
                        new_team = canon_data[2] if canon_data[2] else dup_team
                        cursor.execute(
                            "UPDATE players SET positions = ?, mlb_id = ?, team = ? WHERE player_id = ?",
                            (merged_pos, new_mlb_id, new_team, canonical_id),
                        )

            # Step 5: Delete duplicate player rows
            for dup_id in duplicate_ids:
                cursor.execute("DELETE FROM players WHERE player_id = ?", (dup_id,))

            total_merged += len(duplicate_ids)

        conn.commit()
        result = {
            "duplicates_found": len(dup_names),
            "players_merged": total_merged,
            "fk_remapped": total_remapped,
        }
        logger.info("Deduplication complete: %s", result)
        return result
    finally:
        conn.close()


def upsert_injury_history_bulk(records: list[dict]) -> int:
    """Bulk upsert injury history. Each dict: player_id, season, games_played, games_available.

    Uses ON CONFLICT with the UNIQUE index on (player_id, season).
    """
    conn = get_connection()
    try:
        saved = 0
        for r in records:
            conn.execute(
                """INSERT INTO injury_history (player_id, season, games_played, games_available)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(player_id, season) DO UPDATE SET
                     games_played = excluded.games_played,
                     games_available = excluded.games_available""",
                (r["player_id"], r["season"], r["games_played"], r["games_available"]),
            )
            saved += 1
        conn.commit()
        return saved
    finally:
        conn.close()


def upsert_park_factors(factors: list[dict]) -> int:
    """Bulk upsert park factors. Each dict: team_code, factor_hitting, factor_pitching.

    park_factors table uses team_code as PRIMARY KEY, so ON CONFLICT works.
    """
    conn = get_connection()
    try:
        saved = 0
        for f in factors:
            conn.execute(
                """INSERT INTO park_factors (team_code, factor_hitting, factor_pitching)
                   VALUES (?, ?, ?)
                   ON CONFLICT(team_code) DO UPDATE SET
                     factor_hitting = excluded.factor_hitting,
                     factor_pitching = excluded.factor_pitching""",
                (
                    f["team_code"],
                    f.get("factor_hitting", f.get("park_factor", 1.0)),
                    f.get("factor_pitching", f.get("park_factor", 1.0)),
                ),
            )
            saved += 1
        conn.commit()
        return saved
    finally:
        conn.close()


def upsert_statcast_bulk(records: list[dict]) -> int:
    """Bulk upsert statcast archive records.

    Each dict needs: player_id, season.
    Optional: ev_mean, ev_p90, barrel_pct, hard_hit_pct, xba, xslg, xwoba,
              whiff_pct, chase_rate, sprint_speed, ff_avg_speed, ff_spin_rate,
              k_pct, bb_pct, gb_pct, stuff_plus, location_plus, pitching_plus.

    Uses INSERT OR REPLACE with the UNIQUE(player_id, season) constraint.
    """
    conn = get_connection()
    try:
        saved = 0
        for r in records:
            conn.execute(
                """INSERT OR REPLACE INTO statcast_archive
                   (player_id, season, ev_mean, ev_p90, barrel_pct, hard_hit_pct,
                    xba, xslg, xwoba, whiff_pct, chase_rate, sprint_speed,
                    ff_avg_speed, ff_spin_rate, k_pct, bb_pct, gb_pct,
                    stuff_plus, location_plus, pitching_plus, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           CURRENT_TIMESTAMP)""",
                (
                    r["player_id"],
                    r["season"],
                    r.get("ev_mean"),
                    r.get("ev_p90"),
                    r.get("barrel_pct"),
                    r.get("hard_hit_pct"),
                    r.get("xba"),
                    r.get("xslg"),
                    r.get("xwoba"),
                    r.get("whiff_pct"),
                    r.get("chase_rate"),
                    r.get("sprint_speed"),
                    r.get("ff_avg_speed"),
                    r.get("ff_spin_rate"),
                    r.get("k_pct"),
                    r.get("bb_pct"),
                    r.get("gb_pct"),
                    r.get("stuff_plus"),
                    r.get("location_plus"),
                    r.get("pitching_plus"),
                ),
            )
            saved += 1
        conn.commit()
        return saved
    finally:
        conn.close()


# ── Game-Day Intelligence DB Helpers ─────────────────────────────────


def upsert_game_day_weather(
    game_pk: int,
    game_date: str,
    venue_team: str,
    temp_f: float | None,
    wind_mph: float | None,
    wind_dir: str | None,
    precip_pct: float | None,
    humidity_pct: float | None,
) -> None:
    """Insert or update weather data for a specific game."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO game_day_weather
               (game_pk, game_date, venue_team, temp_f, wind_mph,
                wind_dir, precip_pct, humidity_pct, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_pk,
                game_date,
                venue_team,
                temp_f,
                wind_mph,
                wind_dir,
                precip_pct,
                humidity_pct,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_game_day_weather(game_date: str) -> pd.DataFrame:
    """Load all weather records for a given game date."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM game_day_weather WHERE game_date = ?",
            conn,
            params=(game_date,),
        )
    finally:
        conn.close()
    return coerce_numeric_df(df) if not df.empty else df


def upsert_team_strength(
    team_abbr: str,
    season: int,
    wrc_plus: float | None,
    fip: float | None,
    team_ops: float | None,
    team_era: float | None,
    team_whip: float | None,
    k_pct: float | None,
    bb_pct: float | None,
) -> None:
    """Insert or update team-level strength metrics for a season."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO team_strength
               (team_abbr, season, wrc_plus, fip, team_ops, team_era,
                team_whip, k_pct, bb_pct, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                team_abbr,
                season,
                wrc_plus,
                fip,
                team_ops,
                team_era,
                team_whip,
                k_pct,
                bb_pct,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_team_strength(season: int) -> pd.DataFrame:
    """Load all team strength records for a given season."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM team_strength WHERE season = ?",
            conn,
            params=(season,),
        )
    finally:
        conn.close()
    return coerce_numeric_df(df) if not df.empty else df


def upsert_opp_pitcher(
    pitcher_id: int,
    season: int,
    name: str | None,
    team: str | None,
    era: float | None,
    fip: float | None,
    xfip: float | None,
    whip: float | None,
    k_per_9: float | None,
    bb_per_9: float | None,
    vs_lhb_avg: float | None,
    vs_rhb_avg: float | None,
    ip: float | None,
    hand: str | None,
) -> None:
    """Insert or update opposing pitcher stats for a season."""
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO opp_pitcher_stats
               (pitcher_id, season, name, team, era, fip, xfip, whip,
                k_per_9, bb_per_9, vs_lhb_avg, vs_rhb_avg, ip, hand, fetched_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                pitcher_id,
                season,
                name,
                team,
                era,
                fip,
                xfip,
                whip,
                k_per_9,
                bb_per_9,
                vs_lhb_avg,
                vs_rhb_avg,
                ip,
                hand,
                datetime.now(UTC).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_opp_pitchers(season: int) -> pd.DataFrame:
    """Load all opposing pitcher stats for a given season."""
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM opp_pitcher_stats WHERE season = ?",
            conn,
            params=(season,),
        )
    finally:
        conn.close()
    return coerce_numeric_df(df) if not df.empty else df


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
            "obp": ["obp", "on_base_pct"],
            "bb": ["bb", "walks"],
            "hbp": ["hbp", "hit_by_pitch"],
            "sf": ["sf", "sac_flies"],
        }.items():
            for k in candidates:
                if k in cols:
                    m[stat] = cols[k]
                    break
    else:
        for stat, candidates in {
            "ip": ["ip", "innings"],
            "w": ["w", "wins"],
            "l": ["l", "losses"],
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
    """Insert or find a player. Returns player_id.

    Tries name+team first for precision, then falls back to name-only
    to prevent creating duplicate entries when the same player appears
    with different team values from different data sources.
    """
    # Try exact match: name + team
    cursor.execute("SELECT player_id, positions FROM players WHERE name = ? AND team = ?", (name, team))
    result = cursor.fetchone()

    # Fallback: name-only match (prevents duplicates from team mismatches)
    if result is None and name:
        cursor.execute("SELECT player_id, positions FROM players WHERE name = ?", (name,))
        result = cursor.fetchone()

    if result:
        # Merge positions
        existing = set(result[1].split(","))
        new = set(positions.split(","))
        merged = ",".join(sorted(existing | new))
        if merged != result[1]:
            cursor.execute("UPDATE players SET positions = ? WHERE player_id = ?", (merged, result[0]))
        # Update team if it was empty and we have a real value
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
