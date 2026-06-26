"""TDD tests for the Marcel projection fallback (M-1).

On Railway, FanGraphs projection fetches 403 (Cloudflare blocks the datacenter
IP) so the ``projections`` table never populates → the player pool has no
projection stats → free-agent "value" renders 0 for every player. Marcel is
pure compute over MLB ``season_stats`` (no FanGraphs / no network), so it can
populate the projections blend on the server.

These tests are DB-free: they build a temp-file SQLite DB seeded with synthetic
``players`` + ``season_stats`` and monkeypatch ``get_connection`` (patched at the
name BOUND in each consumer module) to point at it.
"""

from __future__ import annotations

import sqlite3

import pytest

# --- Schema mirrors the post-init_db() shape that the production code relies on.
# Only the columns the Marcel fallback + the blender touch are included.
_PLAYERS_DDL = """
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    team TEXT,
    positions TEXT NOT NULL DEFAULT 'Util',
    is_hitter INTEGER NOT NULL DEFAULT 1,
    is_injured INTEGER NOT NULL DEFAULT 0,
    injury_note TEXT,
    birth_date TEXT,
    mlb_id INTEGER,
    level TEXT
);
"""

_SEASON_STATS_DDL = """
CREATE TABLE season_stats (
    player_id INTEGER NOT NULL,
    season INTEGER NOT NULL DEFAULT 2026,
    pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
    r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
    sb INTEGER DEFAULT 0, avg REAL DEFAULT 0, obp REAL DEFAULT 0,
    bb INTEGER DEFAULT 0, hbp INTEGER DEFAULT 0, sf INTEGER DEFAULT 0,
    ip REAL DEFAULT 0, w INTEGER DEFAULT 0, l INTEGER DEFAULT 0,
    sv INTEGER DEFAULT 0, k INTEGER DEFAULT 0, era REAL DEFAULT 0,
    whip REAL DEFAULT 0, er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0,
    h_allowed INTEGER DEFAULT 0, games_played INTEGER DEFAULT 0,
    last_updated TEXT,
    PRIMARY KEY (player_id, season)
);
"""

# The projections table as it exists AFTER init_db()'s _safe_add_column calls
# (base CREATE + obp/l/bb/hbp/sf + fip/xfip/siera + forecast_season).
_PROJECTIONS_DDL = """
CREATE TABLE projections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    system TEXT NOT NULL,
    forecast_season INTEGER,
    pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
    r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
    sb INTEGER DEFAULT 0, avg REAL DEFAULT 0, obp REAL DEFAULT 0,
    bb INTEGER DEFAULT 0, hbp INTEGER DEFAULT 0, sf INTEGER DEFAULT 0,
    ip REAL DEFAULT 0, w INTEGER DEFAULT 0, l INTEGER DEFAULT 0,
    sv INTEGER DEFAULT 0, k INTEGER DEFAULT 0, era REAL DEFAULT 0,
    whip REAL DEFAULT 0, er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0,
    h_allowed INTEGER DEFAULT 0, fip REAL DEFAULT 0, xfip REAL DEFAULT 0,
    siera REAL DEFAULT 0
);
"""

_REFRESH_LOG_DDL = """
CREATE TABLE refresh_log (
    source TEXT PRIMARY KEY,
    last_refresh TEXT,
    status TEXT DEFAULT 'unknown',
    rows_written INTEGER,
    rows_expected_min INTEGER,
    message TEXT,
    tier TEXT DEFAULT 'primary'
);
"""


def _seed_db(db_path: str) -> None:
    """Create + seed a temp-file DB with a few hitters + pitchers."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_PLAYERS_DDL + _SEASON_STATS_DDL + _PROJECTIONS_DDL + _REFRESH_LOG_DDL)

        # Two hitters (with HR history), two pitchers (with K history),
        # plus one rookie hitter with NO history.
        conn.executescript(
            """
            INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES
                (1, 'Slugger One', 'NYY', 'OF', 1),
                (2, 'Contact Two', 'LAD', '2B', 1),
                (3, 'Ace Three', 'ATL', 'SP', 0),
                (4, 'Closer Four', 'HOU', 'RP', 0),
                (5, 'Rookie Five', 'SEA', 'SS', 1);

            -- Hitter 1: two seasons, lots of HR.
            INSERT INTO season_stats
                (player_id, season, pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf)
            VALUES
                (1, 2025, 650, 580, 160, 95, 38, 100, 5, 0.276, 0.350, 60, 5, 5),
                (1, 2024, 640, 575, 155, 90, 34, 95, 6, 0.270, 0.345, 58, 4, 5);

            -- Hitter 2: one season, contact bat.
            INSERT INTO season_stats
                (player_id, season, pa, ab, h, r, hr, rbi, sb, avg, obp, bb, hbp, sf)
            VALUES
                (2, 2025, 600, 540, 162, 80, 12, 60, 25, 0.300, 0.365, 50, 4, 6);

            -- Pitcher 3: two seasons, strikeout starter.
            INSERT INTO season_stats
                (player_id, season, ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed)
            VALUES
                (3, 2025, 190.0, 14, 8, 0, 210, 3.10, 1.05, 65, 45, 155),
                (3, 2024, 180.0, 12, 9, 0, 195, 3.40, 1.10, 68, 48, 150);

            -- Pitcher 4: one season, closer.
            INSERT INTO season_stats
                (player_id, season, ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed)
            VALUES
                (4, 2025, 65.0, 4, 3, 35, 90, 2.50, 1.00, 18, 20, 45);

            -- Player 5: NO season_stats rows at all (rookie, no history).
            """
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def marcel_db(tmp_path, monkeypatch):
    """Temp-file DB pointed at by both ``DB_PATH`` and the BOUND ``get_connection``
    in every consumer module, so the whole projections chain (the new Marcel
    fallback + the blender + the bootstrap phase) runs against ONE isolated DB.

    A temp FILE (not :memory:) is used so independent ``get_connection()`` calls
    see the same data and ``conn.close()`` is harmless. ``DB_PATH`` is redirected
    (via ``monkeypatch``, so it auto-reverts) so that any code reaching the real
    ``src.database.get_connection`` (e.g. ``data_pipeline`` imports it by value)
    still lands on the temp DB and cannot touch — or leak — the worktree DB.
    """
    import pathlib

    import src.data_pipeline as data_pipeline_mod
    import src.database as database_mod
    import src.marcel_bootstrap as marcel_bootstrap_mod

    db_path = str(tmp_path / "marcel_test.db")
    _seed_db(db_path)

    def _fake_get_connection():
        return sqlite3.connect(db_path)

    # Redirect DB_PATH so the REAL get_connection (used by modules that imported
    # it by value, e.g. data_pipeline) lands on the temp DB. Auto-reverted.
    monkeypatch.setattr(database_mod, "DB_PATH", pathlib.Path(db_path))
    # Patch the BOUND name in each module that imported get_connection directly
    # (bound-name monkeypatch lesson): src.database (used by create_blended) +
    # src.marcel_bootstrap + src.data_pipeline.
    monkeypatch.setattr(database_mod, "get_connection", _fake_get_connection)
    monkeypatch.setattr(marcel_bootstrap_mod, "get_connection", _fake_get_connection)
    monkeypatch.setattr(data_pipeline_mod, "get_connection", _fake_get_connection)
    return db_path


def _rows(db_path: str, sql: str, params=()):
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# ── (a) generate_marcel_projections writes system='marcel' rows ───────────────
def test_generate_marcel_writes_marcel_rows(marcel_db):
    from src.marcel_bootstrap import generate_marcel_projections

    count = generate_marcel_projections()

    marcel_rows = _rows(marcel_db, "SELECT player_id FROM projections WHERE system = 'marcel'")
    written_ids = {r["player_id"] for r in marcel_rows}

    # One row per player (all 5, including the history-less rookie).
    assert count == 5
    assert written_ids == {1, 2, 3, 4, 5}


# ── (b) projected stats are sane / non-zero ──────────────────────────────────
def test_marcel_hitter_projects_positive_hr(marcel_db):
    from src.marcel_bootstrap import generate_marcel_projections

    generate_marcel_projections()

    row = _rows(
        marcel_db,
        "SELECT hr, pa FROM projections WHERE system = 'marcel' AND player_id = 1",
    )[0]
    # A hitter with ~36 HR history must project HR > 0.
    assert row["hr"] > 0
    assert row["pa"] > 0


def test_marcel_pitcher_projects_positive_k(marcel_db):
    from src.marcel_bootstrap import generate_marcel_projections

    generate_marcel_projections()

    row = _rows(
        marcel_db,
        "SELECT k, ip FROM projections WHERE system = 'marcel' AND player_id = 3",
    )[0]
    # A pitcher with ~200 K history must project K > 0.
    assert row["k"] > 0
    assert row["ip"] > 0


# ── (c) after blending, a 'blended' row exists with non-zero stats ───────────
def test_blended_row_nonzero_after_marcel(marcel_db):
    from src.database import create_blended_projections
    from src.marcel_bootstrap import generate_marcel_projections

    generate_marcel_projections()
    create_blended_projections()

    # Hitter blended row has non-zero HR and AVG (Marcel is the only system →
    # blended ≈ Marcel; the blender must not zero out the rate stats).
    hitter = _rows(
        marcel_db,
        "SELECT hr, avg, pa FROM projections WHERE system = 'blended' AND player_id = 1",
    )
    assert len(hitter) == 1
    assert hitter[0]["hr"] > 0
    assert hitter[0]["avg"] > 0.0
    assert hitter[0]["pa"] > 0

    # Pitcher blended row has non-zero K and ERA (era must survive the
    # blender's era = er*9/ip recomputation — components derived to preserve it).
    pitcher = _rows(
        marcel_db,
        "SELECT k, era, ip FROM projections WHERE system = 'blended' AND player_id = 3",
    )
    assert len(pitcher) == 1
    assert pitcher[0]["k"] > 0
    assert pitcher[0]["era"] > 0.0
    assert pitcher[0]["ip"] > 0


# ── (d) a player with NO history gets a regressed projection, not a crash ─────
def test_no_history_player_regresses_to_league_mean(marcel_db):
    from src.marcel import LEAGUE_AVG_HITTING
    from src.marcel_bootstrap import generate_marcel_projections

    # Must not raise on the history-less rookie.
    generate_marcel_projections()

    row = _rows(
        marcel_db,
        "SELECT hr, avg FROM projections WHERE system = 'marcel' AND player_id = 5",
    )
    assert len(row) == 1
    # With no history, Marcel returns the league-mean line (regression to mean).
    # HR is a counting stat → league mean HR > 0; AVG → league mean AVG > 0.
    assert row[0]["hr"] > 0
    assert row[0]["avg"] > 0.0
    # Sanity: the projected HR should be in the league-average neighborhood,
    # not the slugger's ~36.
    assert row[0]["hr"] < LEAGUE_AVG_HITTING["hr"] + 10


# ── (e) the bootstrap threshold triggers Marcel when rows are insufficient ────
def test_bootstrap_falls_back_to_marcel_when_few_rows(marcel_db, monkeypatch):
    """When FanGraphs yields < threshold non-blended rows, the bootstrap phase
    runs the Marcel fallback + the blender, and the projections table fills.

    Simulates the Railway 403 condition: FanGraphs returns nothing, so the
    projections table is empty after the fetch (< threshold) and the Marcel
    fallback fires. ``fetch_all_projections`` is patched (NOT ``refresh_if_stale``)
    so the REAL ``refresh_if_stale`` still runs ``init_db()`` against the temp DB.
    """
    from src.data_bootstrap import BootstrapProgress, _bootstrap_projections

    # FanGraphs "succeeds" but writes 0 rows (Railway 403 condition) → the
    # row-count gate trips and the Marcel fallback runs. Patching refresh_if_stale
    # (bound INSIDE _bootstrap_projections via a local import) keeps the test
    # focused on the fallback trigger, not the FanGraphs fetch internals.
    monkeypatch.setattr("src.data_pipeline.refresh_if_stale", lambda force=False: True, raising=True)

    result = _bootstrap_projections(BootstrapProgress())

    # After the phase, the projections table must have a non-zero number of
    # marcel + blended rows (the fallback fired).
    marcel_rows = _rows(marcel_db, "SELECT 1 FROM projections WHERE system = 'marcel'")
    blended_rows = _rows(marcel_db, "SELECT 1 FROM projections WHERE system = 'blended'")
    assert len(marcel_rows) == 5
    assert len(blended_rows) >= 1
    assert "marcel" in result.lower()
