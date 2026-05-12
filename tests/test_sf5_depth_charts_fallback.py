"""SF-5 regression: depth charts MLB Stats API fallback when Roster Resource fails.

Per CLAUDE.md SF-5: `players.depth_chart_role` is 100% NULL when the Roster
Resource scrape returns empty (HTTP 403 / JS-gated). This test verifies:

1. ``fetch_depth_charts_via_statsapi()`` returns the expected shape (matching
   ``fetch_depth_charts()``) so downstream ``_persist_depth_chart_roles()``
   keeps working unchanged.
2. ``_bootstrap_depth_charts()`` falls through to the statsapi tier when the
   primary scrape returns empty, persists roles + lineup slots, and writes
   ``tier='fallback'`` to ``refresh_log``.
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database with a small players table."""
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import get_connection, init_db

        init_db()

        # Seed a few players we'll exercise the persistence path with
        conn = get_connection()
        try:
            conn.executemany(
                "INSERT INTO players (player_id, name, team, positions, is_hitter) VALUES (?, ?, ?, ?, ?)",
                [
                    (1001, "Aaron Judge", "NYY", "OF", 1),
                    (1002, "Juan Soto", "NYY", "OF", 1),
                    (1003, "Gerrit Cole", "NYY", "P", 0),
                    (1004, "Devin Williams", "NYY", "P", 0),  # closer
                    (1005, "Jonathan Loaisiga", "NYY", "P", 0),  # reliever
                ],
            )
            conn.commit()
        finally:
            conn.close()

        yield db_path


# ── 1. Shape contract ────────────────────────────────────────────────────


class TestStatsapiFallbackShape:
    """Output of fetch_depth_charts_via_statsapi must match fetch_depth_charts()."""

    def test_returns_dict_keyed_by_team_code(self):
        from src.depth_charts import fetch_depth_charts_via_statsapi

        with patch("src.depth_charts._STATSAPI_TEAM_IDS", {"NYY": 147}):
            with patch("src.depth_charts._fetch_team_via_statsapi") as mock_fetch:
                mock_fetch.return_value = {
                    "lineup": ["A", "B", "C"],
                    "rotation": ["X", "Y"],
                    "bullpen": {"CL": "Z"},
                }
                out = fetch_depth_charts_via_statsapi()

        assert isinstance(out, dict)
        assert "NYY" in out
        assert set(out["NYY"].keys()) == {"lineup", "rotation", "bullpen"}

    def test_classifies_position_players_into_lineup(self):
        """Non-pitcher non-DH players go into lineup as ordered list."""
        from src.depth_charts import _classify_team_roster_response

        roster_response = {
            "roster": [
                {
                    "person": {"fullName": "Aaron Judge", "stats": []},
                    "position": {"abbreviation": "RF"},
                },
                {
                    "person": {"fullName": "Juan Soto", "stats": []},
                    "position": {"abbreviation": "LF"},
                },
            ]
        }
        out = _classify_team_roster_response(roster_response)
        assert "Aaron Judge" in out["lineup"]
        assert "Juan Soto" in out["lineup"]

    def test_classifies_starter_pitcher_into_rotation(self):
        """Pitcher with gamesStarted >= 5 → rotation."""
        from src.depth_charts import _classify_team_roster_response

        roster_response = {
            "roster": [
                {
                    "person": {
                        "fullName": "Gerrit Cole",
                        "stats": [
                            {
                                "group": {"displayName": "pitching"},
                                "splits": [{"stat": {"gamesStarted": 8, "saves": 0, "gamesPlayed": 8}}],
                            }
                        ],
                    },
                    "position": {"abbreviation": "P"},
                }
            ]
        }
        out = _classify_team_roster_response(roster_response)
        assert "Gerrit Cole" in out["rotation"]

    def test_classifies_closer_into_bullpen_cl(self):
        """Pitcher with saves >= 5 → bullpen.CL (closer)."""
        from src.depth_charts import _classify_team_roster_response

        roster_response = {
            "roster": [
                {
                    "person": {
                        "fullName": "Devin Williams",
                        "stats": [
                            {
                                "group": {"displayName": "pitching"},
                                "splits": [{"stat": {"gamesStarted": 0, "saves": 8, "gamesPlayed": 16}}],
                            }
                        ],
                    },
                    "position": {"abbreviation": "P"},
                }
            ]
        }
        out = _classify_team_roster_response(roster_response)
        # CL slot must be present and contain Williams (string or list)
        bullpen_cl = out["bullpen"].get("CL")
        assert bullpen_cl is not None
        if isinstance(bullpen_cl, list):
            assert "Devin Williams" in bullpen_cl
        else:
            assert "Devin Williams" in str(bullpen_cl)

    def test_classifies_other_pitcher_into_bullpen(self):
        """Pitcher with neither GS>=5 nor SV>=5 → bullpen.MR (reliever)."""
        from src.depth_charts import _classify_team_roster_response

        roster_response = {
            "roster": [
                {
                    "person": {
                        "fullName": "Jonathan Loaisiga",
                        "stats": [
                            {
                                "group": {"displayName": "pitching"},
                                "splits": [{"stat": {"gamesStarted": 0, "saves": 1, "gamesPlayed": 12}}],
                            }
                        ],
                    },
                    "position": {"abbreviation": "P"},
                }
            ]
        }
        out = _classify_team_roster_response(roster_response)
        # Should land somewhere in the bullpen, not lineup/rotation
        assert "Jonathan Loaisiga" not in out["lineup"]
        assert "Jonathan Loaisiga" not in out["rotation"]
        all_bp_names = []
        for v in out["bullpen"].values():
            if isinstance(v, list):
                all_bp_names.extend(v)
            else:
                all_bp_names.append(v)
        assert "Jonathan Loaisiga" in all_bp_names


# ── 2. Bootstrap fallthrough ─────────────────────────────────────────────


class TestBootstrapDepthChartsFallback:
    """_bootstrap_depth_charts uses statsapi fallback when scrape returns empty."""

    def test_fallback_persists_and_records_tier(self, temp_db):
        """When primary returns empty and fallback returns data:
        - _persist_depth_chart_roles is called
        - refresh_log gets tier='fallback'
        - returned message mentions fallback tier
        """
        with patch("src.database.DB_PATH", temp_db):
            from src.data_bootstrap import BootstrapProgress, _bootstrap_depth_charts

            mock_depth_data = {
                "NYY": {
                    "lineup": ["Aaron Judge", "Juan Soto"],
                    "rotation": ["Gerrit Cole"],
                    "bullpen": {"CL": "Devin Williams", "MR": ["Jonathan Loaisiga"]},
                }
            }

            with patch("src.depth_charts.fetch_depth_charts", return_value={}):
                with patch(
                    "src.depth_charts.fetch_depth_charts_via_statsapi",
                    return_value=mock_depth_data,
                ):
                    msg = _bootstrap_depth_charts(BootstrapProgress())

            assert "fallback" in msg.lower()

            # refresh_log should record tier='fallback'
            from src.database import get_connection

            conn = get_connection()
            try:
                row = conn.execute("SELECT status, tier FROM refresh_log WHERE source = 'depth_charts'").fetchone()
            finally:
                conn.close()
            assert row is not None
            # Wave 7 INFRA-F6: row-count gate downgrades to "partial" when
            # < expected_min rows; the mock fixture has only 5 players, so
            # accept either "success" or "partial" (both mean data exists).
            assert row[0] in ("success", "partial"), f"Expected success or partial, got {row[0]!r}"
            assert row[1] == "fallback"

            # Players got their depth_chart_role / lineup_slot populated
            conn = get_connection()
            try:
                judge = conn.execute(
                    "SELECT depth_chart_role, lineup_slot FROM players WHERE player_id = 1001"
                ).fetchone()
                cole = conn.execute("SELECT depth_chart_role FROM players WHERE player_id = 1003").fetchone()
                williams = conn.execute("SELECT depth_chart_role FROM players WHERE player_id = 1004").fetchone()
            finally:
                conn.close()

            assert judge[0] == "starter"  # batting order slot 1
            assert judge[1] == 1
            assert cole[0] == "starter"  # rotation slot 1
            assert williams[0] == "closer"

    def test_primary_success_does_not_use_fallback(self, temp_db):
        """When primary returns data, fallback is NOT called and tier='primary'."""
        with patch("src.database.DB_PATH", temp_db):
            from src.data_bootstrap import BootstrapProgress, _bootstrap_depth_charts

            mock_depth_data = {
                "NYY": {
                    "lineup": ["Aaron Judge"],
                    "rotation": [],
                    "bullpen": {},
                }
            }

            with patch("src.depth_charts.fetch_depth_charts", return_value=mock_depth_data):
                with patch("src.depth_charts.fetch_depth_charts_via_statsapi") as mock_fallback:
                    msg = _bootstrap_depth_charts(BootstrapProgress())

            mock_fallback.assert_not_called()
            assert "primary" in msg.lower() or "fallback" not in msg.lower()

            from src.database import get_connection

            conn = get_connection()
            try:
                row = conn.execute("SELECT tier FROM refresh_log WHERE source = 'depth_charts'").fetchone()
            finally:
                conn.close()
            assert row[0] == "primary"

    def test_both_empty_records_no_data(self, temp_db):
        """When both primary and fallback return empty, refresh_log = no_data."""
        with patch("src.database.DB_PATH", temp_db):
            from src.data_bootstrap import BootstrapProgress, _bootstrap_depth_charts

            with patch("src.depth_charts.fetch_depth_charts", return_value={}):
                with patch("src.depth_charts.fetch_depth_charts_via_statsapi", return_value={}):
                    msg = _bootstrap_depth_charts(BootstrapProgress())

            assert "skip" in msg.lower() or "no data" in msg.lower()

            from src.database import get_connection

            conn = get_connection()
            try:
                row = conn.execute("SELECT status FROM refresh_log WHERE source = 'depth_charts'").fetchone()
            finally:
                conn.close()
            assert row[0] == "no_data"
