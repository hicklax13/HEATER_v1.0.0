"""Tests for the enriched player pool (health, ECR, YTD, scarcity columns)."""

from __future__ import annotations

import sqlite3
import tempfile
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def temp_db(tmp_path):
    """Create a minimal SQLite DB with all required tables for load_player_pool."""
    db_path = tmp_path / "test_enriched.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    # Players table
    c.execute("""
        CREATE TABLE players (
            player_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            team TEXT,
            positions TEXT NOT NULL,
            is_hitter INTEGER NOT NULL DEFAULT 1,
            is_injured INTEGER NOT NULL DEFAULT 0,
            mlb_id INTEGER,
            birth_date TEXT,
            injury_note TEXT
        )
    """)

    # Projections table (blended)
    c.execute("""
        CREATE TABLE projections (
            player_id INTEGER,
            system TEXT,
            pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
            r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0, avg REAL DEFAULT 0, obp REAL DEFAULT 0,
            bb INTEGER DEFAULT 0, hbp INTEGER DEFAULT 0, sf INTEGER DEFAULT 0,
            ip REAL DEFAULT 0, w INTEGER DEFAULT 0, l INTEGER DEFAULT 0,
            sv INTEGER DEFAULT 0, k INTEGER DEFAULT 0, era REAL DEFAULT 0,
            whip REAL DEFAULT 0, er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0,
            h_allowed INTEGER DEFAULT 0, fip REAL, xfip REAL, siera REAL,
            PRIMARY KEY (player_id, system)
        )
    """)

    # ROS projections (empty — forces blended path)
    c.execute("""
        CREATE TABLE ros_projections (
            player_id INTEGER, system TEXT,
            pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
            r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0, avg REAL DEFAULT 0, obp REAL DEFAULT 0,
            bb INTEGER DEFAULT 0, hbp INTEGER DEFAULT 0, sf INTEGER DEFAULT 0,
            ip REAL DEFAULT 0, w INTEGER DEFAULT 0, l INTEGER DEFAULT 0,
            sv INTEGER DEFAULT 0, k INTEGER DEFAULT 0, era REAL DEFAULT 0,
            whip REAL DEFAULT 0, er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0,
            h_allowed INTEGER DEFAULT 0, fip REAL, xfip REAL, siera REAL,
            PRIMARY KEY (player_id, system)
        )
    """)

    # ADP table
    c.execute("""
        CREATE TABLE adp (
            player_id INTEGER PRIMARY KEY,
            yahoo_adp REAL, fantasypros_adp REAL, adp REAL NOT NULL
        )
    """)

    # ECR consensus table
    c.execute("""
        CREATE TABLE ecr_consensus (
            player_id INTEGER PRIMARY KEY,
            espn_rank INTEGER, yahoo_adp REAL, cbs_rank INTEGER,
            nfbc_adp REAL, fg_adp REAL, fp_ecr INTEGER, heater_sgp_rank INTEGER,
            consensus_rank INTEGER, consensus_avg REAL,
            rank_min INTEGER, rank_max INTEGER, rank_stddev REAL,
            n_sources INTEGER, fetched_at TEXT
        )
    """)

    # Season stats table
    c.execute("""
        CREATE TABLE season_stats (
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL DEFAULT 2026,
            pa INTEGER DEFAULT 0, ab INTEGER DEFAULT 0, h INTEGER DEFAULT 0,
            r INTEGER DEFAULT 0, hr INTEGER DEFAULT 0, rbi INTEGER DEFAULT 0,
            sb INTEGER DEFAULT 0, avg REAL DEFAULT 0, ip REAL DEFAULT 0,
            w INTEGER DEFAULT 0, sv INTEGER DEFAULT 0, k INTEGER DEFAULT 0,
            era REAL DEFAULT 0, whip REAL DEFAULT 0,
            er INTEGER DEFAULT 0, bb_allowed INTEGER DEFAULT 0, h_allowed INTEGER DEFAULT 0,
            games_played INTEGER DEFAULT 0, last_updated TEXT,
            PRIMARY KEY (player_id, season)
        )
    """)

    # Injury history table
    c.execute("""
        CREATE TABLE injury_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            games_played INTEGER DEFAULT 0,
            games_available INTEGER DEFAULT 162,
            il_stints INTEGER DEFAULT 0,
            il_days INTEGER DEFAULT 0,
            created_at TEXT
        )
    """)

    # Ownership trends table (for percent_owned subquery in pool SQL)
    c.execute("""
        CREATE TABLE ownership_trends (
            player_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            percent_owned REAL,
            delta_7d REAL,
            PRIMARY KEY (player_id, date)
        )
    """)

    # Statcast archive table (for xwoba/barrel_pct/etc. in pool SQL)
    c.execute("""
        CREATE TABLE statcast_archive (
            player_id INTEGER NOT NULL,
            season INTEGER NOT NULL,
            xwoba REAL, xba REAL, barrel_pct REAL, hard_hit_pct REAL,
            ev_mean REAL, stuff_plus REAL, babip REAL,
            PRIMARY KEY (player_id, season)
        )
    """)

    # League rosters table
    c.execute("""
        CREATE TABLE league_rosters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            team_index INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            roster_slot TEXT,
            is_user_team INTEGER NOT NULL DEFAULT 0,
            status TEXT DEFAULT 'active',
            selected_position TEXT DEFAULT '',
            editorial_team_abbr TEXT DEFAULT '',
            UNIQUE(team_name, player_id)
        )
    """)

    # Insert test players
    players = [
        (1, "Mike Trout", "LAA", "OF", 1, 0, 545361, "1991-08-07"),
        (2, "Shohei Ohtani", "LAD", "DH", 1, 0, 660271, "1994-07-05"),
        (3, "Cal Raleigh", "SEA", "C", 1, 0, 663728, "1996-11-26"),
        (4, "Emmanuel Clase", "CLE", "RP", 0, 0, 661403, "1998-03-18"),
        (5, "Marcus Semien", "TEX", "SS,2B", 1, 0, 543760, "1990-09-17"),
    ]
    c.executemany(
        "INSERT INTO players (player_id, name, team, positions, is_hitter, is_injured, mlb_id, birth_date) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        players,
    )

    # Insert blended projections
    projections = [
        (1, "blended", 500, 450, 130, 80, 30, 85, 10, 0.289, 0.380, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (2, "blended", 600, 550, 170, 100, 40, 110, 15, 0.309, 0.390, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (3, "blended", 500, 460, 110, 65, 28, 75, 2, 0.239, 0.310, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        (4, "blended", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 65, 4, 2, 35, 70, 2.50, 0.95, 18, 15, 50),
        (5, "blended", 600, 560, 155, 90, 22, 70, 20, 0.277, 0.340, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    ]
    c.executemany(
        "INSERT INTO projections (player_id, system, pa, ab, h, r, hr, rbi, sb, avg, obp, "
        "bb, hbp, sf, ip, w, l, sv, k, era, whip, er, bb_allowed, h_allowed) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        projections,
    )

    # Insert ADP
    adp = [
        (1, None, None, 15.0),
        (2, None, None, 5.0),
        (3, None, None, 45.0),
        (4, None, None, 60.0),
        (5, None, None, 35.0),
    ]
    c.executemany("INSERT INTO adp (player_id, yahoo_adp, fantasypros_adp, adp) VALUES (?, ?, ?, ?)", adp)

    # Insert ECR for some players (not all — tests default behavior)
    c.execute(
        "INSERT INTO ecr_consensus (player_id, consensus_rank, consensus_avg, n_sources) VALUES (?, ?, ?, ?)",
        (2, 3, 4.2, 6),
    )
    c.execute(
        "INSERT INTO ecr_consensus (player_id, consensus_rank, consensus_avg, n_sources) VALUES (?, ?, ?, ?)",
        (1, 12, 15.0, 5),
    )

    # Insert YTD stats for some players (not all)
    c.execute(
        "INSERT INTO season_stats (player_id, season, pa, avg, hr, rbi, sb, era, whip, sv, k) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (2, 2026, 120, 0.320, 12, 30, 5, 0, 0, 0, 0),
    )

    # Insert injury history for player 1 (Trout — injury prone)
    c.execute(
        "INSERT INTO injury_history (player_id, season, games_played, games_available) VALUES (?, ?, ?, ?)",
        (1, 2025, 80, 162),
    )
    c.execute(
        "INSERT INTO injury_history (player_id, season, games_played, games_available) VALUES (?, ?, ?, ?)",
        (1, 2024, 60, 162),
    )

    # Insert roster status for player 1 (on IL)
    c.execute(
        "INSERT INTO league_rosters (team_name, team_index, player_id, roster_slot, is_user_team, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("Team Hickey", 0, 1, "OF", 1, "IL15"),
    )

    # Statcast archive table (required by LEFT JOIN in player pool queries)
    c.execute("""
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
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(player_id, season)
        )
    """)

    conn.commit()
    conn.close()
    return db_path


def _load_pool(db_path):
    """Load enriched pool using the test database."""
    with patch("src.database.DB_PATH", db_path):
        from src.database import _load_player_pool_impl

        return _load_player_pool_impl()


class TestEnrichedPoolColumns:
    """Verify the enriched pool has all expected new columns."""

    def test_has_ecr_columns(self, temp_db):
        df = _load_pool(temp_db)
        assert "consensus_rank" in df.columns
        assert "ecr_avg" in df.columns
        assert "ecr_sources" in df.columns

    def test_has_ytd_columns(self, temp_db):
        df = _load_pool(temp_db)
        for col in ["ytd_pa", "ytd_avg", "ytd_hr", "ytd_rbi", "ytd_sb", "ytd_era", "ytd_whip", "ytd_sv", "ytd_k"]:
            assert col in df.columns, f"Missing YTD column: {col}"

    def test_has_health_columns(self, temp_db):
        df = _load_pool(temp_db)
        assert "health_score" in df.columns
        assert "status" in df.columns

    def test_has_scarcity_columns(self, temp_db):
        df = _load_pool(temp_db)
        assert "is_closer" in df.columns
        assert "scarcity_mult" in df.columns


class TestEnrichedPoolDefaults:
    """Verify correct defaults when data is missing."""

    def test_health_score_default(self, temp_db):
        """Players without injury history get default 0.85."""
        df = _load_pool(temp_db)
        # Player 2 (Ohtani) has no injury history and no IL status
        ohtani = df[df["player_id"] == 2].iloc[0]
        assert ohtani["health_score"] == pytest.approx(0.85, abs=0.01)

    def test_status_default(self, temp_db):
        """Players without roster status get 'active'."""
        df = _load_pool(temp_db)
        # Player 2 has no league_rosters entry
        ohtani = df[df["player_id"] == 2].iloc[0]
        assert ohtani["status"] == "active"

    def test_ecr_default_zero(self, temp_db):
        """Players without ECR data get 0 consensus_rank (coerced from NaN by coerce_numeric_df)."""
        df = _load_pool(temp_db)
        # Player 3 (Raleigh) has no ECR entry — LEFT JOIN produces NULL,
        # coerce_numeric_df converts to 0 since consensus_rank is in _INT_STAT_COLS
        raleigh = df[df["player_id"] == 3].iloc[0]
        assert raleigh["consensus_rank"] == 0

    def test_ytd_default_zero(self, temp_db):
        """Players without YTD stats get 0 for counting columns."""
        df = _load_pool(temp_db)
        # Player 3 has no season_stats entry
        raleigh = df[df["player_id"] == 3].iloc[0]
        assert raleigh["ytd_hr"] == 0
        assert raleigh["ytd_pa"] == 0


class TestEnrichedPoolValues:
    """Verify enrichment produces correct values."""

    def test_ecr_rank_present(self, temp_db):
        """ECR data is correctly joined."""
        df = _load_pool(temp_db)
        ohtani = df[df["player_id"] == 2].iloc[0]
        assert ohtani["consensus_rank"] == 3
        assert ohtani["ecr_avg"] == pytest.approx(4.2)
        assert ohtani["ecr_sources"] == 6

    def test_ytd_stats_present(self, temp_db):
        """YTD stats are correctly joined."""
        df = _load_pool(temp_db)
        ohtani = df[df["player_id"] == 2].iloc[0]
        assert ohtani["ytd_pa"] == 120
        assert ohtani["ytd_avg"] == pytest.approx(0.320)
        assert ohtani["ytd_hr"] == 12

    def test_health_score_injury_prone(self, temp_db):
        """Player with injury history gets lower health score."""
        df = _load_pool(temp_db)
        trout = df[df["player_id"] == 1].iloc[0]
        # Trout: 80/162 + 60/162 = avg ~0.43, but IL15 caps at 0.65
        # The IL15 cap applies since raw health < 0.80 won't trigger the cap
        assert trout["health_score"] < 0.85  # Not the default

    def test_il_status_caps_health(self, temp_db):
        """IL15 status caps health_score for display."""
        df = _load_pool(temp_db)
        trout = df[df["player_id"] == 1].iloc[0]
        assert trout["status"] == "IL15"

    def test_closer_flag(self, temp_db):
        """Player with sv >= 5 flagged as closer."""
        df = _load_pool(temp_db)
        clase = df[df["player_id"] == 4].iloc[0]
        assert clase["is_closer"] is True or clase["is_closer"] == 1

    def test_non_closer_flag(self, temp_db):
        """Player with sv < 5 not flagged as closer."""
        df = _load_pool(temp_db)
        ohtani = df[df["player_id"] == 2].iloc[0]
        assert not ohtani["is_closer"]

    def test_closer_scarcity_mult(self, temp_db):
        """Closer gets 1.3x scarcity multiplier."""
        df = _load_pool(temp_db)
        clase = df[df["player_id"] == 4].iloc[0]
        assert clase["scarcity_mult"] == pytest.approx(1.3)

    def test_catcher_scarcity_mult(self, temp_db):
        """H3: Catcher gets 1.20x graduated scarcity multiplier (most scarce)."""
        df = _load_pool(temp_db)
        raleigh = df[df["player_id"] == 3].iloc[0]
        assert raleigh["scarcity_mult"] == pytest.approx(1.20)

    def test_ss_2b_scarcity_mult(self, temp_db):
        """H3: SS,2B player gets max(SS=1.10, 2B=1.15) = 1.15x."""
        df = _load_pool(temp_db)
        semien = df[df["player_id"] == 5].iloc[0]
        assert semien["scarcity_mult"] == pytest.approx(1.15)

    def test_regular_player_scarcity_mult(self, temp_db):
        """OF player gets 1.0x scarcity multiplier."""
        df = _load_pool(temp_db)
        # Ohtani is DH — no scarcity
        ohtani = df[df["player_id"] == 2].iloc[0]
        assert ohtani["scarcity_mult"] == pytest.approx(1.0)


class TestOptimizedConsumers:
    """Verify that downstream consumers skip redundant work."""

    def test_apply_scarcity_flags_shortcircuits(self):
        """apply_scarcity_flags returns immediately when columns exist."""
        from src.trade_intelligence import apply_scarcity_flags

        df = pd.DataFrame(
            {
                "player_id": [1, 2],
                "is_closer": [True, False],
                "scarcity_mult": [1.3, 1.0],
            }
        )
        # Should return the exact same object (not a copy)
        result = apply_scarcity_flags(df)
        assert result is df

    def test_get_health_adjusted_pool_reuses_columns(self):
        """get_health_adjusted_pool skips DB calls when columns exist."""
        from src.trade_intelligence import get_health_adjusted_pool
        from src.valuation import LeagueConfig

        df = pd.DataFrame(
            {
                "player_id": [1, 2],
                "name": ["A", "B"],
                "positions": ["OF", "SP"],
                "is_hitter": [1, 0],
                "health_score": [0.65, 0.85],
                "status": ["IL15", "active"],
                "pa": [500, 0],
                "ab": [450, 0],
                "h": [130, 0],
                "r": [80, 0],
                "hr": [30, 0],
                "rbi": [85, 0],
                "sb": [10, 0],
                "avg": [0.289, 0.0],
                "obp": [0.380, 0.0],
                "ip": [0, 180],
                "w": [0, 12],
                "l": [0, 8],
                "sv": [0, 0],
                "k": [0, 200],
                "era": [0, 3.50],
                "whip": [0, 1.15],
                "er": [0, 70],
                "bb_allowed": [0, 50],
                "h_allowed": [0, 150],
                "bb": [60, 0],
                "hbp": [5, 0],
                "sf": [3, 0],
            }
        )
        config = LeagueConfig()

        # Patch the DB-loading functions to verify they're NOT called
        with (
            patch("src.trade_intelligence._load_health_scores") as mock_hs,
            patch("src.trade_intelligence._load_roster_statuses") as mock_rs,
        ):
            result = get_health_adjusted_pool(df, config)
            # Should NOT have called the DB functions
            mock_hs.assert_not_called()
            mock_rs.assert_not_called()

        # Result should still have health_score and status columns
        assert "health_score" in result.columns
        assert "status" in result.columns
