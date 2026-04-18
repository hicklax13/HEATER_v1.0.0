"""Regression tests for the 2026-04-17 data-fetch audit.

Locks in fixes for:
  1. refresh_log validation (update_refresh_log_auto, status-aware staleness)
  2. ros_projections updated_at timestamp population
  3. season_stats save — is_hitter guard, no fuzzy fallback when mlb_id present
  4. game_logs parser — raw MLB API splits parsing
  5. FanGraphs skip-vs-error classification
  6. Extended-roster ordering in bootstrap_all_data
"""

from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Fixture ──────────────────────────────────────────────────────────────


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    """Isolate each test against a fresh temp SQLite DB."""
    import src.database as dbmod

    db_path = tmp_path / "test.db"
    monkeypatch.setattr(dbmod, "DB_PATH", db_path)
    dbmod.init_db()
    yield db_path


# ── Group 1: refresh_log validation foundation ──────────────────────────


class TestRefreshLogValidation:
    def test_auto_partial_below_expected_min(self, isolated_db):
        from src.database import get_refresh_status, update_refresh_log_auto

        update_refresh_log_auto("t", 5, expected_min=100)
        s = get_refresh_status("t")
        assert s["status"] == "partial"
        assert s["rows_written"] == 5
        assert s["rows_expected_min"] == 100

    def test_auto_no_data_on_zero(self, isolated_db):
        from src.database import get_refresh_status, update_refresh_log_auto

        update_refresh_log_auto("t", 0, expected_min=100)
        assert get_refresh_status("t")["status"] == "no_data"

    def test_auto_success_at_threshold(self, isolated_db):
        from src.database import get_refresh_status, update_refresh_log_auto

        update_refresh_log_auto("t", 100, expected_min=100)
        assert get_refresh_status("t")["status"] == "success"

    def test_auto_error_flag_overrides(self, isolated_db):
        from src.database import get_refresh_status, update_refresh_log_auto

        update_refresh_log_auto("t", 500, expected_min=100, error=True)
        assert get_refresh_status("t")["status"] == "error"

    def test_check_staleness_treats_non_success_as_stale(self, isolated_db):
        from src.database import check_staleness, update_refresh_log

        for stale_status in ("no_data", "partial", "error", "unknown"):
            update_refresh_log("t", stale_status)
            assert check_staleness("t", max_age_hours=24) is True, f"status={stale_status} should be treated as stale"

    def test_check_staleness_fresh_success_not_stale(self, isolated_db):
        from src.database import check_staleness, update_refresh_log

        update_refresh_log("t", "success")
        assert check_staleness("t", max_age_hours=24) is False

    def test_invalid_status_downgraded_to_unknown(self, isolated_db):
        from src.database import get_refresh_status, update_refresh_log

        update_refresh_log("t", "completely_bogus_status")
        assert get_refresh_status("t")["status"] == "unknown"

    def test_message_persisted(self, isolated_db):
        from src.database import get_refresh_status, update_refresh_log

        update_refresh_log("t", "success", message="fetched 42 rows")
        assert get_refresh_status("t")["message"] == "fetched 42 rows"


# ── Group 2: ros_projections updated_at ────────────────────────────────


class TestRosProjectionsUpdatedAt:
    def test_migration_backfills_null_updated_at(self, isolated_db):
        """Legacy rows with NULL updated_at should be backfilled on init_db."""
        from src.database import get_connection, init_db

        conn = get_connection()
        # Simulate legacy data
        conn.execute("INSERT INTO ros_projections (player_id, system, updated_at) VALUES (1, 'ros_bayesian', NULL)")
        conn.commit()
        conn.close()

        init_db()  # run migrations again

        conn = get_connection()
        row = conn.execute("SELECT updated_at FROM ros_projections WHERE player_id = 1").fetchone()
        conn.close()
        assert row[0] is not None, "updated_at should be backfilled by T4 migration"


# ── Group 3: save_season_stats_to_db hardening ─────────────────────────


class TestSeasonStatsSave:
    def test_is_hitter_mismatch_rejected(self, isolated_db):
        """Pitcher stat row aimed at a hitter player_id must be rejected.

        Prevents the 2026-04-17 Bellinger-with-7.2-IP corruption class.
        """
        from src.database import get_connection
        from src.live_stats import save_season_stats_to_db

        conn = get_connection()
        conn.execute(
            "INSERT INTO players (player_id, name, team, positions, is_hitter, mlb_id) "
            "VALUES (5000, 'Cody Bellinger', 'CHC', 'OF', 1, 660111)"
        )
        conn.commit()
        conn.close()

        # Build a PITCHER stat row with matching mlb_id
        df = pd.DataFrame(
            [
                {
                    "player_name": "Cody Bellinger",
                    "team": "CHC",
                    "mlb_id": 660111,
                    "is_hitter": False,  # row claims pitcher
                    "pa": 0,
                    "ab": 0,
                    "h": 0,
                    "r": 0,
                    "hr": 0,
                    "rbi": 0,
                    "sb": 0,
                    "avg": 0,
                    "obp": 0,
                    "bb": 0,
                    "hbp": 0,
                    "sf": 0,
                    "ip": 7.2,
                    "w": 0,
                    "l": 1,
                    "sv": 0,
                    "k": 6,
                    "era": 5.87,
                    "whip": 1.50,
                    "er": 5,
                    "bb_allowed": 3,
                    "h_allowed": 9,
                    "games_played": 2,
                }
            ]
        )
        saved = save_season_stats_to_db(df, season=2026)
        assert saved == 0, "pitcher row on hitter player_id must be rejected"

        # Verify no IP was written
        conn = get_connection()
        row = conn.execute("SELECT ip, k, era FROM season_stats WHERE player_id = 5000 AND season = 2026").fetchone()
        conn.close()
        assert row is None, "no row should have been inserted"

    def test_hitter_row_succeeds_on_hitter_player_id(self, isolated_db):
        from src.database import get_connection
        from src.live_stats import save_season_stats_to_db

        conn = get_connection()
        conn.execute(
            "INSERT INTO players (player_id, name, team, positions, is_hitter, mlb_id) "
            "VALUES (5001, 'Yordan Alvarez', 'HOU', 'DH,OF', 1, 670541)"
        )
        conn.commit()
        conn.close()

        df = pd.DataFrame(
            [
                {
                    "player_name": "Yordan Alvarez",
                    "team": "HOU",
                    "mlb_id": 670541,
                    "is_hitter": True,
                    "pa": 70,
                    "ab": 62,
                    "h": 20,
                    "r": 11,
                    "hr": 4,
                    "rbi": 14,
                    "sb": 1,
                    "avg": 0.323,
                    "obp": 0.400,
                    "bb": 8,
                    "hbp": 0,
                    "sf": 0,
                    "ip": 0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "games_played": 16,
                }
            ]
        )
        saved = save_season_stats_to_db(df, season=2026)
        assert saved == 1
        conn = get_connection()
        row = conn.execute("SELECT hr, rbi FROM season_stats WHERE player_id = 5001 AND season = 2026").fetchone()
        conn.close()
        assert row[0] == 4
        assert row[1] == 14

    def test_no_fuzzy_fallback_when_mlb_id_present_but_unmatched(self, isolated_db):
        """With mlb_id present but no DB match, the row must be skipped (not fuzzy-matched)."""
        from src.database import get_connection
        from src.live_stats import save_season_stats_to_db

        conn = get_connection()
        # A different "Bellinger" (e.g. a pitcher) exists but with different mlb_id
        conn.execute(
            "INSERT INTO players (player_id, name, team, positions, is_hitter, mlb_id) "
            "VALUES (5002, 'Eric Bellinger', 'LAD', 'OF', 1, 999999)"
        )
        conn.commit()
        conn.close()

        # Row uses a DIFFERENT mlb_id (no DB match)
        df = pd.DataFrame(
            [
                {
                    "player_name": "Cody Bellinger",
                    "team": "CHC",
                    "mlb_id": 660111,  # NOT in DB
                    "is_hitter": True,
                    "pa": 50,
                    "ab": 45,
                    "h": 15,
                    "r": 5,
                    "hr": 2,
                    "rbi": 7,
                    "sb": 0,
                    "avg": 0.333,
                    "obp": 0.400,
                    "bb": 5,
                    "hbp": 0,
                    "sf": 0,
                    "ip": 0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "games_played": 15,
                }
            ]
        )
        saved = save_season_stats_to_db(df, season=2026)
        # mlb_id mismatch AND name+team mismatch → skip
        assert saved == 0

    def test_exact_name_team_match_with_mlb_id_backfill(self, isolated_db):
        """When players.mlb_id is NULL, match by exact name+team and backfill mlb_id."""
        from src.database import get_connection
        from src.live_stats import save_season_stats_to_db

        conn = get_connection()
        conn.execute(
            "INSERT INTO players (player_id, name, team, positions, is_hitter, mlb_id) "
            "VALUES (5003, 'Alex Bregman', 'BOS', '3B', 1, NULL)"
        )
        conn.commit()
        conn.close()

        df = pd.DataFrame(
            [
                {
                    "player_name": "Alex Bregman",
                    "team": "BOS",
                    "mlb_id": 608324,
                    "is_hitter": True,
                    "pa": 60,
                    "ab": 55,
                    "h": 11,
                    "r": 4,
                    "hr": 2,
                    "rbi": 7,
                    "sb": 0,
                    "avg": 0.200,
                    "obp": 0.267,
                    "bb": 4,
                    "hbp": 0,
                    "sf": 0,
                    "ip": 0,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "games_played": 15,
                }
            ]
        )
        saved = save_season_stats_to_db(df, season=2026)
        assert saved == 1

        conn = get_connection()
        row = conn.execute("SELECT mlb_id FROM players WHERE player_id = 5003").fetchone()
        conn.close()
        assert row[0] == 608324, "mlb_id should have been backfilled"


# ── Group 4: game_logs parser ──────────────────────────────────────────


class TestGameLogsParser:
    def test_parser_reads_splits_structure(self, isolated_db, monkeypatch):
        """The parser must read stats[0].splits[] with per-split date, not
        the flattened statsapi.player_stat_data() shape that drops dates.
        """
        from src.database import get_connection
        from src.player_databank import _fetch_and_store_player_logs

        conn = get_connection()
        conn.execute(
            "INSERT INTO players (player_id, name, team, positions, is_hitter, mlb_id) "
            "VALUES (5100, 'Test Player', 'TEST', 'OF', 1, 123456)"
        )
        conn.commit()
        conn.close()

        # Raw MLB API response shape (what statsapi.get returns)
        fake_response = {
            "stats": [
                {
                    "type": {"displayName": "gameLog"},
                    "group": {"displayName": "hitting"},
                    "splits": [
                        {
                            "date": "2026-04-01",
                            "stat": {
                                "plateAppearances": 4,
                                "atBats": 4,
                                "hits": 2,
                                "homeRuns": 1,
                                "rbi": 2,
                                "runs": 1,
                                "stolenBases": 0,
                                "baseOnBalls": 0,
                                "hitByPitch": 0,
                                "sacFlies": 0,
                            },
                        },
                        {
                            "date": "2026-04-02",
                            "stat": {
                                "plateAppearances": 5,
                                "atBats": 5,
                                "hits": 0,
                                "homeRuns": 0,
                                "rbi": 0,
                                "runs": 0,
                                "stolenBases": 1,
                                "baseOnBalls": 0,
                                "hitByPitch": 0,
                                "sacFlies": 0,
                            },
                        },
                    ],
                }
            ]
        }

        mock_statsapi = MagicMock()
        mock_statsapi.get = MagicMock(return_value=fake_response)
        monkeypatch.setattr("src.player_databank.statsapi", mock_statsapi)

        count = _fetch_and_store_player_logs(player_id=5100, mlb_id=123456, is_hitter=1, season=2026)
        assert count == 2, "parser should produce 2 rows from 2 splits"

        conn = get_connection()
        rows = conn.execute(
            "SELECT game_date, h, hr, sb FROM game_logs WHERE player_id = 5100 ORDER BY game_date"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        assert rows[0][0] == "2026-04-01" and rows[0][1] == 2 and rows[0][2] == 1 and rows[0][3] == 0
        assert rows[1][0] == "2026-04-02" and rows[1][1] == 0 and rows[1][2] == 0 and rows[1][3] == 1

    def test_parser_handles_empty_splits(self, isolated_db, monkeypatch):
        from src.player_databank import _fetch_and_store_player_logs

        mock_statsapi = MagicMock()
        mock_statsapi.get = MagicMock(return_value={"stats": []})
        monkeypatch.setattr("src.player_databank.statsapi", mock_statsapi)

        count = _fetch_and_store_player_logs(player_id=5101, mlb_id=123457, is_hitter=1, season=2026)
        assert count == 0

    def test_parser_skips_splits_missing_date(self, isolated_db, monkeypatch):
        from src.player_databank import _fetch_and_store_player_logs

        fake_response = {
            "stats": [
                {
                    "splits": [
                        {"date": "", "stat": {"hits": 1}},
                        {"stat": {"hits": 1}},  # no date key at all
                    ]
                }
            ]
        }
        mock_statsapi = MagicMock()
        mock_statsapi.get = MagicMock(return_value=fake_response)
        monkeypatch.setattr("src.player_databank.statsapi", mock_statsapi)

        count = _fetch_and_store_player_logs(player_id=5102, mlb_id=123458, is_hitter=1, season=2026)
        assert count == 0


# ── Group 5: FanGraphs error classification ───────────────────────────


class TestFanGraphsClassification:
    def test_403_classified_as_skipped(self):
        from src.data_bootstrap import _classify_fetch_error

        exc = Exception("Error accessing 'https://www.fangraphs.com/leaders-legacy.aspx'. Received status code 403")
        assert _classify_fetch_error(exc) == "skipped"

    def test_429_classified_as_skipped(self):
        from src.data_bootstrap import _classify_fetch_error

        assert _classify_fetch_error(Exception("rate limited: 429")) == "skipped"

    def test_timeout_classified_as_skipped(self):
        from src.data_bootstrap import _classify_fetch_error

        assert _classify_fetch_error(Exception("connection timeout")) == "skipped"
        assert _classify_fetch_error(Exception("Request timed out after 30s")) == "skipped"

    def test_generic_error_classified_as_error(self):
        from src.data_bootstrap import _classify_fetch_error

        assert _classify_fetch_error(Exception("ValueError: bad data")) == "error"
        assert _classify_fetch_error(Exception("column x missing")) == "error"


# ── Group 6: rosterType fix (compile-time sanity) ─────────────────────


class TestRosterTypeFix:
    def test_live_stats_uses_fullRoster_not_fullSeason(self):
        """Lock in the fullSeason→fullRoster fix so a future regression surfaces."""
        import inspect

        import src.live_stats as ls_mod

        source = inspect.getsource(ls_mod.fetch_season_stats)
        assert '"rosterType": "fullRoster"' in source or "'rosterType': 'fullRoster'" in source, (
            "fetch_season_stats must request rosterType=fullRoster "
            "(2026-04-17 fix: fullSeason dropped every player who hadn't "
            "logged a game yet, silently losing 80%+ of stars)."
        )
        assert 'rosterType": "fullSeason"' not in source, "fetch_season_stats must NOT use rosterType=fullSeason"
