"""Tests for the forecast_season foundation of the ridge-stacking blend.

Covers PV-C1 / PV-E1 / PV-C3 (2026-06-07 fix campaign):

1. Schema: ``projections`` carries a ``forecast_season`` column after
   ``init_db()`` and after a legacy-DB migration (table created without it).
2. ``_store_projections`` tags inserted rows with the current season and only
   replaces same-``(player_id, system, forecast_season)`` rows — prior-season
   forecasts are RETAINED for future training.
3. ``_load_stacking_weights`` returns UNIFORM 1/n weights when no matched
   (forecast_season=Y AND season=Y) pair exists (the current reality), and
   NEVER regresses current-year forecasts on prior-year actuals. When a
   matched pair IS seeded, it learns from THAT pair.
4. PV-C3: the volume columns (pa/ab/ip) blend with a single shared
   playing-time weight so recomputed rate stats use a coherent volume basis.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd
import pytest

import src.database as db
from src.database import (
    _load_stacking_weights,
    create_blended_projections,
    get_connection,
    init_db,
)

CURRENT_SEASON = datetime.now(UTC).year


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Point src.database at an isolated, freshly-init'd DB."""
    dbfile = tmp_path / "forecast_season.db"
    monkeypatch.setattr(db, "DB_PATH", dbfile)
    init_db()
    return dbfile


def _columns(conn, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


# ── 1. Schema ──────────────────────────────────────────────────────────


def test_init_db_creates_forecast_season_column(fresh_db):
    conn = get_connection()
    try:
        assert "forecast_season" in _columns(conn, "projections")
    finally:
        conn.close()


def test_legacy_db_migration_adds_forecast_season(tmp_path, monkeypatch):
    """A pre-existing projections table WITHOUT the column gets it via init_db()."""
    dbfile = tmp_path / "legacy.db"
    # Build a legacy projections table that predates forecast_season.
    raw = sqlite3.connect(str(dbfile))
    raw.executescript(
        """
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, team TEXT, positions TEXT, is_hitter INTEGER
        );
        CREATE TABLE projections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            system TEXT NOT NULL,
            pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
            hr INTEGER DEFAULT 0, avg REAL DEFAULT 0
        );
        """
    )
    raw.execute("INSERT INTO projections (player_id, system, hr) VALUES (1, 'steamer', 25)")
    raw.commit()
    raw.close()

    assert "forecast_season" not in _columns(sqlite3.connect(str(dbfile)), "projections")

    monkeypatch.setattr(db, "DB_PATH", dbfile)
    init_db()

    conn = get_connection()
    try:
        cols = _columns(conn, "projections")
        assert "forecast_season" in cols
        # Legacy row preserved.
        n = conn.execute("SELECT COUNT(*) FROM projections").fetchone()[0]
        assert n == 1
    finally:
        conn.close()


# ── 2. _store_projections retains prior-season forecasts ───────────────


def _store_one(system: str, name: str, hr: int, forecast_season: int | None = None):
    """Drive _store_projections for a single hitter of one system.

    When forecast_season is None, _store_projections tags with the current
    season. When set, we seed directly so we can simulate a PRIOR season's
    forecast already living in the table.
    """
    from src.data_pipeline import _store_projections

    df = pd.DataFrame(
        [
            {
                "name": name,
                "team": "NYY",
                "positions": "OF",
                "is_hitter": True,
                "pa": 600,
                "ab": 540,
                "h": 150,
                "r": 90,
                "hr": hr,
                "rbi": 80,
                "sb": 10,
                "avg": 0.278,
                "obp": 0.350,
                "bb": 50,
                "hbp": 5,
                "sf": 5,
            }
        ]
    )
    _store_projections({f"{system}_bat": df})


def test_store_projections_tags_current_season(fresh_db):
    _store_one("steamer", "Aaron Judge", 50)
    conn = get_connection()
    try:
        rows = conn.execute("SELECT forecast_season FROM projections WHERE system = 'steamer'").fetchall()
    finally:
        conn.close()
    assert rows
    assert all(r["forecast_season"] == CURRENT_SEASON for r in rows)


def test_store_projections_retains_prior_forecast_season(fresh_db):
    """A new current-season write must NOT delete a prior season's forecasts."""
    prior = CURRENT_SEASON - 1
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO players (player_id, name, positions, is_hitter) VALUES (1, 'Aaron Judge', 'OF', 1)")
        cur.execute(
            "INSERT INTO projections (player_id, system, hr, forecast_season) VALUES (1, 'steamer', 44, ?)",
            (prior,),
        )
        conn.commit()
    finally:
        conn.close()

    # New current-season run for the same system.
    _store_one("steamer", "Aaron Judge", 50)

    conn = get_connection()
    try:
        prior_rows = conn.execute(
            "SELECT hr FROM projections WHERE system = 'steamer' AND forecast_season = ?",
            (prior,),
        ).fetchall()
        cur_rows = conn.execute(
            "SELECT hr FROM projections WHERE system = 'steamer' AND forecast_season = ?",
            (CURRENT_SEASON,),
        ).fetchall()
    finally:
        conn.close()

    assert len(prior_rows) == 1 and prior_rows[0]["hr"] == 44, (
        "prior-season forecast was wiped by the current-season write"
    )
    assert len(cur_rows) == 1 and cur_rows[0]["hr"] == 50


def test_store_projections_replaces_same_season_only(fresh_db):
    """Re-running the same system for the current season replaces (not duplicates)."""
    _store_one("steamer", "Aaron Judge", 50)
    _store_one("steamer", "Aaron Judge", 47)  # corrected fetch
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT hr FROM projections WHERE system = 'steamer' AND forecast_season = ?",
            (CURRENT_SEASON,),
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1, "same-season re-run should replace, not duplicate"
    assert rows[0]["hr"] == 47


# ── 3. _load_stacking_weights: no cross-year leakage ───────────────────


def _seed_systems_only(conn, season_actuals: int | None):
    """Seed two systems of CURRENT-season forecasts + optional prior actuals.

    Mirrors the live DB shape: forecasts have no prior-season counterpart, so
    stacking has NO matched (forecast_season=Y AND season=Y) pair to learn from.
    """
    cur = conn.cursor()
    for pid in range(1, 41):
        cur.execute(
            "INSERT INTO players (player_id, name, positions, is_hitter) VALUES (?, ?, 'OF', 1)",
            (pid, f"P{pid}"),
        )
    # Two CURRENT-season forecast systems.
    for sysname, base in (("steamer", 20), ("zips", 22)):
        for pid in range(1, 41):
            cur.execute(
                "INSERT INTO projections (player_id, system, hr, forecast_season) VALUES (?, ?, ?, ?)",
                (pid, sysname, base + (pid % 7), CURRENT_SEASON),
            )
    if season_actuals is not None:
        for pid in range(1, 41):
            cur.execute(
                "INSERT INTO season_stats (player_id, season, hr) VALUES (?, ?, ?)",
                (pid, season_actuals, 18 + (pid % 5)),
            )
    conn.commit()


def test_uniform_when_no_matched_pair(fresh_db):
    """The CURRENT reality: 2026 forecasts + 2025 actuals but NO 2025 forecasts.

    Must return UNIFORM weights — NOT a cross-year regression.
    """
    conn = get_connection()
    try:
        _seed_systems_only(conn, season_actuals=CURRENT_SEASON - 1)
        weights = _load_stacking_weights(conn)
    finally:
        conn.close()

    # Either empty (→ caller uses uniform) or explicit uniform per stat.
    if weights:
        for _stat, sysw in weights.items():
            if not sysw:
                continue
            vals = list(sysw.values())
            assert all(abs(v - vals[0]) < 1e-9 for v in vals), (
                f"non-uniform weights for {_stat}: {sysw} — cross-year regression leaked"
            )


def test_no_regression_of_current_forecasts_on_prior_actuals(fresh_db, monkeypatch):
    """compute_all_stat_weights must NOT be called on a current-forecast/prior-actual pair."""
    calls: list = []

    import src.database as dbmod

    real = None
    try:
        from src import projection_stacking as ps

        real = ps.compute_all_stat_weights

        def _spy(systems, actuals, *a, **k):
            calls.append((systems, actuals))
            return real(systems, actuals, *a, **k)

        monkeypatch.setattr(ps, "compute_all_stat_weights", _spy)
        monkeypatch.setattr(dbmod, "compute_all_stat_weights", _spy, raising=False)
    except ImportError:
        pytest.skip("projection_stacking unavailable")

    conn = get_connection()
    try:
        _seed_systems_only(conn, season_actuals=CURRENT_SEASON - 1)
        _load_stacking_weights(conn)
    finally:
        conn.close()

    # With no matched pair, the regression path must not run at all.
    assert calls == [], (
        "compute_all_stat_weights was invoked despite no matched "
        "(forecast_season=Y AND season=Y) pair — cross-year leakage"
    )


def test_learns_from_matched_pair_when_seeded(fresh_db):
    """Seed forecast_season=Y-1 forecasts + season=Y-1 actuals → learns from THAT."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        prior = CURRENT_SEASON - 1
        for pid in range(1, 41):
            cur.execute(
                "INSERT INTO players (player_id, name, positions, is_hitter) VALUES (?, ?, 'OF', 1)",
                (pid, f"P{pid}"),
            )
        # 'good' system tracks actuals closely; 'bad' is noisy/biased.
        for pid in range(1, 41):
            actual = 10 + (pid % 13)
            cur.execute(
                "INSERT INTO season_stats (player_id, season, hr) VALUES (?, ?, ?)",
                (pid, prior, actual),
            )
            cur.execute(
                "INSERT INTO projections (player_id, system, hr, forecast_season) VALUES (?, 'good', ?, ?)",
                (pid, actual, prior),
            )
            cur.execute(
                "INSERT INTO projections (player_id, system, hr, forecast_season) VALUES (?, 'bad', ?, ?)",
                (pid, 25 - (pid % 13), prior),
            )
        # Also seed CURRENT-season forecasts (what we'd actually blend) — these
        # must NOT pollute the learning, which keys off matched prior years.
        for pid in range(1, 41):
            for sysname in ("good", "bad"):
                cur.execute(
                    "INSERT INTO projections (player_id, system, hr, forecast_season) VALUES (?, ?, ?, ?)",
                    (pid, sysname, 20, CURRENT_SEASON),
                )
        conn.commit()
        weights = _load_stacking_weights(conn)
    finally:
        conn.close()

    assert weights, "expected learned weights from the matched prior-year pair"
    hr_w = weights.get("hr", {})
    assert hr_w, "expected hr weights"
    assert hr_w.get("good", 0) > hr_w.get("bad", 0), f"the system that matched actuals should win: {hr_w}"


# ── 4. PV-C3: shared playing-time weight for volume cols ────────────────


def test_volume_cols_share_single_weight(fresh_db, monkeypatch):
    """pa/ab/ip blend with ONE shared playing-time weight, not per-stat weights.

    Construct two systems where per-stat ridge weights would diverge across
    pa vs ab; assert the blended ab/pa ratio matches a single coherent
    playing-time basis rather than independently-weighted columns.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        # Two hitters' worth of systems with DIFFERENT pa/ab so a per-stat
        # weight scheme would blend pa and ab with different mixes.
        cur.execute("INSERT INTO players (player_id, name, positions, is_hitter) VALUES (1, 'H', 'OF', 1)")
        # system A: pa=700 ab=630 h=189 (avg .300); system B: pa=500 ab=450 h=99 (.220)
        cur.execute(
            "INSERT INTO projections (player_id, system, pa, ab, h, hr, avg, "
            "forecast_season) VALUES (1, 'sysA', 700, 630, 189, 30, 0.300, ?)",
            (CURRENT_SEASON,),
        )
        cur.execute(
            "INSERT INTO projections (player_id, system, pa, ab, h, hr, avg, "
            "forecast_season) VALUES (1, 'sysB', 500, 450, 99, 10, 0.220, ?)",
            (CURRENT_SEASON,),
        )
        conn.commit()

        # Force a stacking-weight map that diverges between pa and ab. If the
        # implementation honored per-stat weights for volume cols, blended pa
        # and ab would use different system mixes → incoherent ab/pa ratio.
        diverging = {
            "pa": {"sysA": 0.9, "sysB": 0.1},
            "ab": {"sysA": 0.1, "sysB": 0.9},
            "h": {"sysA": 0.5, "sysB": 0.5},
            "hr": {"sysA": 0.5, "sysB": 0.5},
        }
        monkeypatch.setattr(db, "_load_stacking_weights", lambda _conn: diverging)

        create_blended_projections()

        row = conn.execute("SELECT pa, ab, h FROM projections WHERE player_id = 1 AND system = 'blended'").fetchone()
    finally:
        conn.close()

    pa, ab, _h = row["pa"], row["ab"], row["h"]
    # A coherent shared playing-time weight means pa and ab use the SAME mix,
    # so the blended ab/pa ratio stays inside the two source ratios
    # (630/700 = 0.900 and 450/500 = 0.900 here — both 0.90). With divergent
    # per-stat weights, pa≈680 (mostly A) and ab≈468 (mostly B) → ratio ~0.69,
    # which is impossible for any single weight.
    ratio = ab / pa
    assert 0.88 <= ratio <= 0.92, (
        f"ab/pa ratio {ratio:.3f} indicates incoherent per-stat volume weights "
        f"(pa={pa}, ab={ab}); expected a single shared playing-time weight"
    )
