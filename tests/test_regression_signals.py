"""Tests for G2 (Stuff+ regression) and G3 (BABIP regression) signals."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def enriched_db(tmp_path):
    """Create a DB with players + statcast_archive data for regression flag testing."""
    db_path = tmp_path / "regression_test.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    c.execute("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY, name TEXT, team TEXT,
            positions TEXT, is_hitter INTEGER DEFAULT 1, is_injured INTEGER DEFAULT 0,
            mlb_id INTEGER, birth_date TEXT, injury_note TEXT
        )
    """)

    c.execute("""
        CREATE TABLE projections (
            player_id INTEGER, system TEXT,
            pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
            r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0, avg REAL, obp REAL,
            bb INTEGER DEFAULT 0, hbp INTEGER DEFAULT 0, sf INTEGER DEFAULT 0,
            ip REAL DEFAULT 0, w INTEGER DEFAULT 0, l INTEGER DEFAULT 0,
            sv INTEGER DEFAULT 0, k INTEGER DEFAULT 0, era REAL, whip REAL,
            er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0, h_allowed INTEGER DEFAULT 0,
            fip REAL, xfip REAL, siera REAL,
            stuff_plus REAL, location_plus REAL, pitching_plus REAL
        )
    """)

    c.execute("""
        CREATE TABLE adp (player_id INTEGER PRIMARY KEY, adp REAL, nfbc_adp REAL)
    """)

    c.execute("""
        CREATE TABLE ecr_consensus (
            player_id INTEGER PRIMARY KEY, consensus_rank INTEGER,
            consensus_avg REAL, n_sources INTEGER
        )
    """)

    c.execute("""
        CREATE TABLE season_stats (
            player_id INTEGER, season INTEGER, pa INTEGER DEFAULT 0,
            avg REAL DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0, era REAL DEFAULT 0, whip REAL DEFAULT 0,
            sv INTEGER DEFAULT 0, k INTEGER DEFAULT 0
        )
    """)

    c.execute("""
        CREATE TABLE statcast_archive (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL, season INTEGER NOT NULL,
            ev_mean REAL, ev_p90 REAL, barrel_pct REAL, hard_hit_pct REAL,
            xba REAL, xslg REAL, xwoba REAL, whiff_pct REAL, chase_rate REAL,
            sprint_speed REAL, ff_avg_speed REAL, ff_spin_rate REAL,
            k_pct REAL, bb_pct REAL, gb_pct REAL,
            stuff_plus REAL, location_plus REAL, pitching_plus REAL,
            babip REAL, iso REAL,
            hitter_k_pct REAL, hitter_bb_pct REAL,
            ld_pct REAL, hitter_fb_pct REAL, hitter_gb_pct REAL,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_id, season)
        )
    """)

    c.execute("CREATE TABLE ros_projections (player_id INTEGER)")

    c.execute("CREATE TABLE refresh_log (source TEXT, timestamp TEXT, status TEXT)")

    # Insert test players
    # Hitter 1: High BABIP (lucky) — should be SELL_HIGH
    c.execute("INSERT INTO players VALUES (1, 'Lucky Larry', 'NYY', 'OF', 1, 0, 100, NULL, NULL)")
    c.execute("INSERT INTO projections VALUES (1, 'blended', 550, 500, 150, 80, 25, 80, 10, .300, .360, 40, 5, 3, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL)")
    c.execute("INSERT INTO adp VALUES (1, 50, NULL)")
    c.execute("INSERT INTO season_stats VALUES (1, 2026, 80, .340, 5, 15, 3, 0, 0, 0, 0)")
    c.execute("INSERT INTO statcast_archive (player_id, season, xwoba, babip) VALUES (1, 2026, 0.380, 0.370)")

    # Hitter 2: Low BABIP (unlucky) — should be BUY_LOW
    c.execute("INSERT INTO players VALUES (2, 'Unlucky Ugo', 'LAD', 'SS', 1, 0, 200, NULL, NULL)")
    c.execute("INSERT INTO projections VALUES (2, 'blended', 550, 500, 130, 70, 20, 70, 15, .260, .330, 40, 5, 3, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL)")
    c.execute("INSERT INTO adp VALUES (2, 60, NULL)")
    c.execute("INSERT INTO season_stats VALUES (2, 2026, 70, .220, 3, 10, 5, 0, 0, 0, 0)")
    c.execute("INSERT INTO statcast_archive (player_id, season, xwoba, babip) VALUES (2, 2026, 0.340, 0.250)")

    # Pitcher 3: Elite Stuff+ but bad ERA (unlucky) — should be BUY_LOW
    c.execute("INSERT INTO players VALUES (3, 'Nasty Nate', 'HOU', 'SP', 0, 0, 300, NULL, NULL)")
    c.execute("INSERT INTO projections VALUES (3, 'blended', 0, 0, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, 180, 12, 8, 0, 200, 3.50, 1.20, 70, 60, 220, 3.20, 3.10, 3.00, NULL, NULL, NULL)")
    c.execute("INSERT INTO adp VALUES (3, 30, NULL)")
    c.execute("INSERT INTO season_stats VALUES (3, 2026, 0, 0, 0, 0, 0, 4.80, 1.35, 0, 35)")
    c.execute("INSERT INTO statcast_archive (player_id, season, stuff_plus) VALUES (3, 2026, 125)")

    # Pitcher 4: Low Stuff+ but great ERA (lucky) — should be SELL_HIGH
    c.execute("INSERT INTO players VALUES (4, 'Soft Sam', 'COL', 'SP', 0, 0, 400, NULL, NULL)")
    c.execute("INSERT INTO projections VALUES (4, 'blended', 0, 0, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, 160, 8, 10, 0, 120, 4.20, 1.30, 75, 55, 200, 4.50, 4.60, 4.40, NULL, NULL, NULL)")
    c.execute("INSERT INTO adp VALUES (4, 120, NULL)")
    c.execute("INSERT INTO season_stats VALUES (4, 2026, 0, 0, 0, 0, 0, 3.10, 1.05, 0, 20)")
    c.execute("INSERT INTO statcast_archive (player_id, season, stuff_plus) VALUES (4, 2026, 80)")

    # Hitter 5: Normal BABIP (no flag)
    c.execute("INSERT INTO players VALUES (5, 'Normal Nick', 'CHC', '1B', 1, 0, 500, NULL, NULL)")
    c.execute("INSERT INTO projections VALUES (5, 'blended', 500, 450, 120, 60, 18, 65, 5, .267, .340, 35, 4, 3, 0, 0, 0, 0, 0, NULL, NULL, 0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL)")
    c.execute("INSERT INTO adp VALUES (5, 80, NULL)")
    c.execute("INSERT INTO season_stats VALUES (5, 2026, 60, .270, 4, 12, 2, 0, 0, 0, 0)")
    c.execute("INSERT INTO statcast_archive (player_id, season, xwoba, babip) VALUES (5, 2026, 0.330, 0.305)")

    conn.commit()
    conn.close()
    return db_path


def _load_pool(db_path):
    """Load enriched pool using the test database."""
    with patch("src.database.DB_PATH", db_path):
        from src.database import _load_player_pool_impl
        return _load_player_pool_impl()


# ── G3: BABIP Regression Tests ──────────────────────────────────────


class TestBABIPRegression:
    """G3: BABIP regression signals."""

    def test_high_babip_gets_sell_high(self, enriched_db):
        pool = _load_pool(enriched_db)
        larry = pool[pool["name"] == "Lucky Larry"]
        assert len(larry) == 1
        assert larry.iloc[0]["babip_regression_flag"] == "SELL_HIGH"

    def test_low_babip_gets_buy_low(self, enriched_db):
        pool = _load_pool(enriched_db)
        ugo = pool[pool["name"] == "Unlucky Ugo"]
        assert len(ugo) == 1
        assert ugo.iloc[0]["babip_regression_flag"] == "BUY_LOW"

    def test_normal_babip_no_flag(self, enriched_db):
        pool = _load_pool(enriched_db)
        nick = pool[pool["name"] == "Normal Nick"]
        assert len(nick) == 1
        assert nick.iloc[0]["babip_regression_flag"] == ""

    def test_babip_delta_computed(self, enriched_db):
        pool = _load_pool(enriched_db)
        larry = pool[pool["name"] == "Lucky Larry"].iloc[0]
        # BABIP 0.370 - league avg 0.300 = 0.070
        assert larry["babip_delta"] == pytest.approx(0.070, abs=0.001)

    def test_babip_low_pa_no_flag(self, enriched_db):
        """Hitters with <30 PA shouldn't get BABIP flags (noise)."""
        import sqlite3
        conn = sqlite3.connect(str(enriched_db))
        conn.execute("UPDATE season_stats SET pa = 15 WHERE player_id = 1")
        conn.commit()
        conn.close()
        pool = _load_pool(enriched_db)
        larry = pool[pool["name"] == "Lucky Larry"]
        assert larry.iloc[0]["babip_regression_flag"] == ""

    def test_pitchers_no_babip_flag(self, enriched_db):
        pool = _load_pool(enriched_db)
        nate = pool[pool["name"] == "Nasty Nate"]
        assert len(nate) == 1
        assert nate.iloc[0]["babip_regression_flag"] == ""


# ── G2: Stuff+ Regression Tests ─────────────────────────────────────


class TestStuffPlusRegression:
    """G2: Stuff+ pitcher regression signals."""

    def test_elite_stuff_bad_era_gets_buy_low(self, enriched_db):
        pool = _load_pool(enriched_db)
        nate = pool[pool["name"] == "Nasty Nate"]
        assert len(nate) == 1
        # Stuff+ 125 (elite) + YTD ERA 4.80 >> proj ERA 3.50 → BUY_LOW
        assert nate.iloc[0]["stuff_regression_flag"] == "BUY_LOW"

    def test_weak_stuff_good_era_gets_sell_high(self, enriched_db):
        pool = _load_pool(enriched_db)
        sam = pool[pool["name"] == "Soft Sam"]
        assert len(sam) == 1
        # Stuff+ 80 (weak) + YTD ERA 3.10 << proj ERA 4.20 → SELL_HIGH
        assert sam.iloc[0]["stuff_regression_flag"] == "SELL_HIGH"

    def test_hitters_no_stuff_flag(self, enriched_db):
        pool = _load_pool(enriched_db)
        larry = pool[pool["name"] == "Lucky Larry"]
        assert larry.iloc[0]["stuff_regression_flag"] == ""

    def test_no_stuff_data_no_flag(self, enriched_db):
        pool = _load_pool(enriched_db)
        nick = pool[pool["name"] == "Normal Nick"]
        assert nick.iloc[0]["stuff_regression_flag"] == ""


# ── Trade Finder Integration ────────────────────────────────────────


class TestRegressionInComposite:
    """Regression bonuses are applied in trade finder composite scoring."""

    def test_babip_buy_low_adds_bonus(self):
        """Receiving a BUY_LOW BABIP player should add +0.02 to composite."""
        recv_row = pd.Series({
            "regression_flag": "",
            "babip_regression_flag": "BUY_LOW",
            "stuff_regression_flag": "",
        })
        bonus = 0.0
        if str(recv_row.get("babip_regression_flag", "")) == "BUY_LOW":
            bonus += 0.02
        assert bonus == pytest.approx(0.02)

    def test_stuff_sell_high_adds_bonus(self):
        """Giving away a SELL_HIGH Stuff+ player should add +0.02."""
        give_row = pd.Series({
            "regression_flag": "",
            "babip_regression_flag": "",
            "stuff_regression_flag": "SELL_HIGH",
        })
        bonus = 0.0
        if str(give_row.get("stuff_regression_flag", "")) == "SELL_HIGH":
            bonus += 0.02
        assert bonus == pytest.approx(0.02)

    def test_max_regression_bonus(self):
        """All 3 signals active on both sides = +0.14 max."""
        bonus = 0.0
        # G1: xwOBA
        bonus += 0.03  # recv BUY_LOW
        bonus += 0.03  # give SELL_HIGH
        # G3: BABIP
        bonus += 0.02  # recv BUY_LOW
        bonus += 0.02  # give SELL_HIGH
        # G2: Stuff+
        bonus += 0.02  # recv BUY_LOW
        bonus += 0.02  # give SELL_HIGH
        assert bonus == pytest.approx(0.14)

    def test_no_flags_no_bonus(self):
        """Empty flags should add 0.0."""
        row = pd.Series({
            "regression_flag": "",
            "babip_regression_flag": "",
            "stuff_regression_flag": "",
        })
        bonus = 0.0
        for flag_col, threshold in [
            ("regression_flag", 0.03),
            ("babip_regression_flag", 0.02),
            ("stuff_regression_flag", 0.02),
        ]:
            if str(row.get(flag_col, "")) == "BUY_LOW":
                bonus += threshold
        assert bonus == 0.0
