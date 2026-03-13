"""Tests for the FanGraphs auto-fetch data pipeline."""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db, update_refresh_log


@pytest.fixture(autouse=True)
def temp_db():
    """Use a temp database for every test."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


# ── Sample FanGraphs JSON records ──────────────────────────────────

SAMPLE_HITTER_JSON = [
    {
        "PlayerName": "Aaron Judge",
        "Team": "NYY",
        "minpos": "OF",
        "PA": 650,
        "AB": 550,
        "H": 160,
        "R": 110,
        "HR": 45,
        "RBI": 120,
        "SB": 5,
        "AVG": 0.291,
        "ADP": 3.5,
    },
    {
        "PlayerName": "Mookie Betts",
        "Team": "LAD",
        "minpos": "SS",
        "PA": 620,
        "AB": 540,
        "H": 170,
        "R": 105,
        "HR": 25,
        "RBI": 85,
        "SB": 15,
        "AVG": 0.315,
        "ADP": 5.0,
    },
]

SAMPLE_PITCHER_JSON = [
    {
        "PlayerName": "Gerrit Cole",
        "Team": "NYY",
        "IP": 200.0,
        "W": 16,
        "SV": 0,
        "SO": 250,
        "ERA": 2.95,
        "WHIP": 1.05,
        "ER": 66,
        "BB": 45,
        "H": 165,
        "GS": 33,
        "G": 33,
        "ADP": 15.0,
    },
    {
        "PlayerName": "Josh Hader",
        "Team": "HOU",
        "IP": 65.0,
        "W": 4,
        "SV": 38,
        "SO": 85,
        "ERA": 2.50,
        "WHIP": 0.95,
        "ER": 18,
        "BB": 20,
        "H": 42,
        "GS": 0,
        "G": 65,
        "ADP": 55.0,
    },
    {
        "PlayerName": "Dual Starter",
        "Team": "SEA",
        "IP": 70.0,
        "W": 5,
        "SV": 1,
        "SO": 70,
        "ERA": 3.80,
        "WHIP": 1.20,
        "ER": 30,
        "BB": 25,
        "H": 59,
        "GS": 3,
        "G": 40,
        "ADP": 250.0,
    },
]


# ── Tests ──────────────────────────────────────────────────────────


class TestNormalizeHitterJson:
    def test_column_mapping(self):
        """FG JSON fields map to correct DB column names."""
        from src.data_pipeline import normalize_hitter_json

        df = normalize_hitter_json(SAMPLE_HITTER_JSON)
        expected_cols = {"name", "team", "positions", "is_hitter", "pa", "ab", "h", "r", "hr", "rbi", "sb", "avg"}
        assert expected_cols.issubset(set(df.columns))

    def test_values(self):
        """Numeric values are preserved correctly."""
        from src.data_pipeline import normalize_hitter_json

        df = normalize_hitter_json(SAMPLE_HITTER_JSON)
        judge = df[df["name"] == "Aaron Judge"].iloc[0]
        assert judge["pa"] == 650
        assert judge["hr"] == 45
        assert judge["avg"] == pytest.approx(0.291)
        assert judge["is_hitter"] is True or judge["is_hitter"] == 1

    def test_position_from_minpos(self):
        """minpos field correctly maps to positions column."""
        from src.data_pipeline import normalize_hitter_json

        df = normalize_hitter_json(SAMPLE_HITTER_JSON)
        judge = df[df["name"] == "Aaron Judge"].iloc[0]
        assert judge["positions"] == "OF"
        betts = df[df["name"] == "Mookie Betts"].iloc[0]
        assert betts["positions"] == "SS"


class TestNormalizePitcherJson:
    def test_column_mapping(self):
        """FG JSON pitcher fields map to correct DB column names."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        expected_cols = {
            "name",
            "team",
            "positions",
            "is_hitter",
            "ip",
            "w",
            "sv",
            "k",
            "era",
            "whip",
            "er",
            "bb_allowed",
            "h_allowed",
        }
        assert expected_cols.issubset(set(df.columns))

    def test_so_to_k_mapping(self):
        """FG 'SO' field maps to DB 'k' column."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["k"] == 250

    def test_sp_classification(self):
        """GS >= 5 → 'SP'."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        cole = df[df["name"] == "Gerrit Cole"].iloc[0]
        assert cole["positions"] == "SP"

    def test_rp_classification(self):
        """SV >= 3 → 'RP'."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        hader = df[df["name"] == "Josh Hader"].iloc[0]
        assert hader["positions"] == "RP"

    def test_dual_classification(self):
        """GS >= 1 but < 5, SV < 3 → 'SP,RP'."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        dual = df[df["name"] == "Dual Starter"].iloc[0]
        assert dual["positions"] == "SP,RP"

    def test_is_hitter_false(self):
        """All pitchers have is_hitter=False."""
        from src.data_pipeline import normalize_pitcher_json

        df = normalize_pitcher_json(SAMPLE_PITCHER_JSON)
        assert all(row == False or row == 0 for row in df["is_hitter"])  # noqa: E712


class TestSystemMapping:
    def test_fangraphsdc_maps_to_depthcharts(self):
        """FG API 'fangraphsdc' must map to DB 'depthcharts'."""
        from src.data_pipeline import SYSTEM_MAP

        assert SYSTEM_MAP["fangraphsdc"] == "depthcharts"

    def test_all_systems_present(self):
        """All 3 projection systems are in SYSTEM_MAP."""
        from src.data_pipeline import SYSTEM_MAP

        assert set(SYSTEM_MAP.keys()) == {"steamer", "zips", "fangraphsdc"}

    def test_steamer_identity(self):
        """Steamer maps to itself."""
        from src.data_pipeline import SYSTEM_MAP

        assert SYSTEM_MAP["steamer"] == "steamer"


class TestFetchProjections:
    @patch("src.data_pipeline.requests.get")
    def test_success_hitters(self, mock_get):
        """Successful fetch returns normalized hitter DataFrame + raw JSON."""
        from src.data_pipeline import fetch_projections

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_HITTER_JSON
        mock_get.return_value = mock_response

        df, raw = fetch_projections("steamer", "bat")
        assert len(df) == 2
        assert "name" in df.columns
        assert df.iloc[0]["is_hitter"] is True or df.iloc[0]["is_hitter"] == 1
        assert raw == SAMPLE_HITTER_JSON

    @patch("src.data_pipeline.requests.get")
    def test_success_pitchers(self, mock_get):
        """Successful fetch returns normalized pitcher DataFrame + raw JSON."""
        from src.data_pipeline import fetch_projections

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_PITCHER_JSON
        mock_get.return_value = mock_response

        df, raw = fetch_projections("steamer", "pit")
        assert len(df) == 3
        assert "k" in df.columns

    @patch("src.data_pipeline.requests.get")
    def test_network_error_raises(self, mock_get):
        """Network error raises FetchError."""
        from src.data_pipeline import FetchError, fetch_projections

        mock_get.side_effect = requests.exceptions.ConnectionError("No network")
        with pytest.raises(FetchError):
            fetch_projections("steamer", "bat")

    @patch("src.data_pipeline.requests.get")
    def test_correct_url_params(self, mock_get):
        """Verifies the correct URL and query params are sent."""
        from src.data_pipeline import fetch_projections

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response

        fetch_projections("zips", "bat")
        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs[1]["params"]["type"] == "zips"
        assert call_kwargs[1]["params"]["stats"] == "bat"


class TestExtractAdp:
    def test_filters_high_adp(self):
        """ADP >= 999 is excluded."""
        from src.data_pipeline import extract_adp

        raw_with_high_adp = SAMPLE_HITTER_JSON + [
            {
                "PlayerName": "Nobody",
                "Team": "FA",
                "minpos": "Util",
                "PA": 100,
                "AB": 90,
                "H": 20,
                "R": 10,
                "HR": 2,
                "RBI": 8,
                "SB": 0,
                "AVG": 0.222,
                "ADP": 999,
            },
        ]
        adp_df = extract_adp(raw_with_high_adp, [])
        assert "Nobody" not in adp_df["name"].values

    def test_filters_null_adp(self):
        """Null ADP values are excluded."""
        from src.data_pipeline import extract_adp

        raw_with_null = SAMPLE_HITTER_JSON + [
            {
                "PlayerName": "Ghost",
                "Team": "FA",
                "minpos": "Util",
                "PA": 100,
                "AB": 90,
                "H": 20,
                "R": 10,
                "HR": 2,
                "RBI": 8,
                "SB": 0,
                "AVG": 0.222,
                "ADP": None,
            },
        ]
        adp_df = extract_adp(raw_with_null, [])
        assert "Ghost" not in adp_df["name"].values

    def test_valid_adp_preserved(self):
        """Valid ADP values are kept."""
        from src.data_pipeline import extract_adp

        adp_df = extract_adp(SAMPLE_HITTER_JSON, SAMPLE_PITCHER_JSON)
        assert len(adp_df) > 0
        judge_adp = adp_df[adp_df["name"] == "Aaron Judge"]["adp"].iloc[0]
        assert judge_adp == pytest.approx(3.5)

    def test_combines_hitters_and_pitchers(self):
        """ADP is extracted from both hitter and pitcher records."""
        from src.data_pipeline import extract_adp

        adp_df = extract_adp(SAMPLE_HITTER_JSON, SAMPLE_PITCHER_JSON)
        names = set(adp_df["name"].values)
        assert "Aaron Judge" in names
        assert "Gerrit Cole" in names


class TestStoreProjections:
    def test_stores_and_resolves_player_ids(self):
        """Players are upserted and projections get valid player_ids."""
        from src.data_pipeline import _store_projections, normalize_hitter_json

        init_db()
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        projections = {"steamer_bat": hitters}
        count = _store_projections(projections)
        assert count == 2

        conn = db_mod.get_connection()
        rows = conn.execute(
            "SELECT p.name, pr.system, pr.hr "
            "FROM projections pr JOIN players p ON pr.player_id = p.player_id "
            "WHERE pr.system = 'steamer'"
        ).fetchall()
        conn.close()
        assert len(rows) == 2
        names = {r[0] for r in rows}
        assert "Aaron Judge" in names

    def test_idempotent(self):
        """Storing the same system twice doesn't create duplicates."""
        from src.data_pipeline import _store_projections, normalize_hitter_json

        init_db()
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        projections = {"steamer_bat": hitters}
        _store_projections(projections)
        _store_projections(projections)  # second store

        conn = db_mod.get_connection()
        count = conn.execute("SELECT COUNT(*) FROM projections WHERE system = 'steamer'").fetchone()[0]
        conn.close()
        assert count == 2  # Not 4


class TestStoreAdp:
    def test_stores_with_player_id(self):
        """ADP records are stored with resolved player_ids."""
        from src.data_pipeline import (
            _store_adp,
            _store_projections,
            normalize_hitter_json,
        )

        init_db()
        # Must store players first so ADP can resolve names → player_ids
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        _store_projections({"steamer_bat": hitters})

        adp_df = pd.DataFrame(
            [
                {"name": "Aaron Judge", "adp": 3.5},
                {"name": "Mookie Betts", "adp": 5.0},
            ]
        )
        count = _store_adp(adp_df)
        assert count == 2

        conn = db_mod.get_connection()
        rows = conn.execute("SELECT player_id, adp FROM adp").fetchall()
        conn.close()
        assert len(rows) == 2
        assert all(r[0] is not None and r[0] > 0 for r in rows)


class TestFetchAllProjections:
    @patch("src.data_pipeline.fetch_projections")
    def test_partial_failure(self, mock_fetch):
        """1 system fails, 2 succeed → returns dicts with 4 entries."""
        from src.data_pipeline import FetchError, fetch_all_projections

        def side_effect(system, stats):
            if system == "zips":
                raise FetchError("ZiPS unavailable")
            if stats == "bat":
                df = pd.DataFrame(
                    {
                        "name": ["Test"],
                        "team": ["TST"],
                        "positions": ["OF"],
                        "is_hitter": [True],
                        "pa": [500],
                        "ab": [450],
                        "h": [120],
                        "r": [70],
                        "hr": [20],
                        "rbi": [60],
                        "sb": [5],
                        "avg": [0.267],
                    }
                )
                return df, [{"PlayerName": "Test", "ADP": 100}]
            else:
                df = pd.DataFrame(
                    {
                        "name": ["Ace"],
                        "team": ["TST"],
                        "positions": ["SP"],
                        "is_hitter": [False],
                        "ip": [180.0],
                        "w": [12],
                        "sv": [0],
                        "k": [200],
                        "era": [3.20],
                        "whip": [1.10],
                        "er": [64],
                        "bb_allowed": [50],
                        "h_allowed": [148],
                    }
                )
                return df, [{"PlayerName": "Ace", "ADP": 50}]

        mock_fetch.side_effect = side_effect
        projections, raw_data = fetch_all_projections()
        assert len(projections) == 4
        assert "steamer_bat" in projections
        assert "depthcharts_pit" in projections
        assert "zips_bat" not in projections

    @patch("src.data_pipeline.fetch_projections")
    def test_total_failure(self, mock_fetch):
        """All systems fail → returns empty dicts."""
        from src.data_pipeline import FetchError, fetch_all_projections

        mock_fetch.side_effect = FetchError("down")
        projections, raw_data = fetch_all_projections()
        assert projections == {}
        assert raw_data == {}


class TestRefreshIfStale:
    @patch("src.data_pipeline.fetch_all_projections")
    def test_success(self, mock_fetch_all):
        """Successful refresh returns True and populates DB."""
        from src.data_pipeline import normalize_hitter_json, refresh_if_stale

        init_db()
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        mock_fetch_all.return_value = (
            {"steamer_bat": hitters},
            {"steamer_bat": SAMPLE_HITTER_JSON},
        )

        result = refresh_if_stale(force=True)
        assert result is True

        # Verify data in DB
        conn = db_mod.get_connection()
        count = conn.execute("SELECT COUNT(*) FROM projections").fetchone()[0]
        conn.close()
        assert count > 0

    @patch("src.data_pipeline.fetch_all_projections")
    def test_total_failure_returns_false(self, mock_fetch_all):
        """Total failure returns False."""
        from src.data_pipeline import refresh_if_stale

        init_db()
        mock_fetch_all.return_value = ({}, {})
        result = refresh_if_stale(force=True)
        assert result is False

    @patch("src.data_pipeline.fetch_all_projections")
    def test_skips_when_data_exists(self, mock_fetch_all):
        """When force=False and data exists, skip fetch and return True."""
        from src.data_pipeline import _store_projections, normalize_hitter_json, refresh_if_stale

        init_db()
        # Pre-populate DB with some projections
        hitters = normalize_hitter_json(SAMPLE_HITTER_JSON)
        _store_projections({"steamer_bat": hitters})
        # Mark projections as recently refreshed so check_staleness sees fresh data
        update_refresh_log("fangraphs_projections", "success")

        result = refresh_if_stale(force=False)
        assert result is True
        # fetch_all_projections should NOT have been called
        mock_fetch_all.assert_not_called()
