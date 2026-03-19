# tests/test_prospect_engine.py
"""Tests for prospect rankings engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# -- Readiness score tests ----------------------------------------------------


def test_fv_normalized_bounds():
    from src.prospect_engine import _fv_normalized

    assert _fv_normalized(80) == 100.0
    assert _fv_normalized(20) == 0.0
    assert 0 <= _fv_normalized(50) <= 100


def test_fv_normalized_clamps():
    from src.prospect_engine import _fv_normalized

    assert _fv_normalized(10) == 0.0  # below min
    assert _fv_normalized(90) == 100.0  # above max


def test_eta_proximity_current_year():
    from src.prospect_engine import _eta_proximity

    assert _eta_proximity("2026") == 100.0


def test_eta_proximity_future():
    from src.prospect_engine import _eta_proximity

    assert _eta_proximity("2027") == 75.0
    assert _eta_proximity("2028") == 50.0
    assert _eta_proximity("2029") == 25.0
    assert _eta_proximity("2030") == 0.0


def test_eta_proximity_past():
    from src.prospect_engine import _eta_proximity

    assert _eta_proximity("2025") == 100.0  # already MLB-ready


def test_eta_proximity_empty():
    from src.prospect_engine import _eta_proximity

    assert _eta_proximity("") == 50.0  # unknown defaults to mid


def test_risk_factor():
    from src.prospect_engine import _risk_factor

    assert _risk_factor("Low") == 1.0
    assert _risk_factor("Medium") == 0.8
    assert _risk_factor("High") == 0.6
    assert _risk_factor("Extreme") == 0.4
    assert _risk_factor("Unknown") == 0.7  # default


def test_readiness_score_bounds():
    from src.prospect_engine import compute_mlb_readiness_score

    row = {
        "fg_fv": 60,
        "fg_eta": "2026",
        "fg_risk": "Medium",
        "milb_obp": 0.350,
        "milb_slg": 0.500,
        "milb_level": "AAA",
        "age": 23,
    }
    score = compute_mlb_readiness_score(row)
    assert 0 <= score <= 100


def test_readiness_score_high_fv_near_eta():
    from src.prospect_engine import compute_mlb_readiness_score

    elite = {
        "fg_fv": 70,
        "fg_eta": "2026",
        "fg_risk": "Low",
        "milb_obp": 0.400,
        "milb_slg": 0.550,
        "milb_level": "AAA",
        "age": 22,
    }
    mediocre = {
        "fg_fv": 40,
        "fg_eta": "2029",
        "fg_risk": "High",
        "milb_obp": 0.300,
        "milb_slg": 0.380,
        "milb_level": "A",
        "age": 20,
    }
    assert compute_mlb_readiness_score(elite) > compute_mlb_readiness_score(mediocre)


def test_readiness_score_missing_milb():
    from src.prospect_engine import compute_mlb_readiness_score

    row = {
        "fg_fv": 55,
        "fg_eta": "2027",
        "fg_risk": "Medium",
        "milb_obp": None,
        "milb_slg": None,
        "milb_level": None,
        "age": None,
    }
    score = compute_mlb_readiness_score(row)
    assert 0 <= score <= 100  # should still compute without MiLB data


# -- FanGraphs API parsing tests ----------------------------------------------

MOCK_FG_RESPONSE = [
    {
        "PlayerName": "Prospect Alpha",
        "Team": "NYY",
        "Position": "SS",
        "FV": "60",
        "ETA": "2026",
        "Risk": "Medium",
        "minorMasterId": 12345,
        "pHit": 55,
        "fHit": 60,
        "pGame": 50,
        "fGame": 55,
        "pRaw": 45,
        "fRaw": 50,
        "pSpd": 55,
        "Fld": 50,
        "Age": 21,
    },
    {
        "PlayerName": "Prospect Beta",
        "Team": "LAD",
        "Position": "SP",
        "FV": "55",
        "ETA": "2027",
        "Risk": "High",
        "minorMasterId": 67890,
        "pCtl": 45,
        "fCtl": 55,
        "pHit": None,
        "fHit": None,
        "pGame": None,
        "fGame": None,
        "Age": 19,
    },
]


def test_parse_fg_response():
    from src.prospect_engine import _parse_fg_prospects

    df = _parse_fg_prospects(MOCK_FG_RESPONSE)
    assert len(df) == 2
    assert df.iloc[0]["name"] == "Prospect Alpha"
    assert df.iloc[0]["fg_fv"] == 60
    assert df.iloc[0]["hit_present"] == 55
    assert df.iloc[0]["hit_future"] == 60


def test_parse_fg_pitcher_fields():
    from src.prospect_engine import _parse_fg_prospects

    df = _parse_fg_prospects(MOCK_FG_RESPONSE)
    pitcher = df[df["name"] == "Prospect Beta"].iloc[0]
    assert pitcher["ctrl_present"] == 45
    assert pitcher["ctrl_future"] == 55


# -- Filtering tests ----------------------------------------------------------


def test_get_prospect_rankings_top_n():
    from src.prospect_engine import get_prospect_rankings

    # Uses fallback static list when DB empty and FG unavailable
    with patch("src.prospect_engine._fetch_from_db") as mock_db:
        mock_db.return_value = pd.DataFrame()
        with patch("src.prospect_engine.refresh_prospect_rankings") as mock_refresh:
            mock_refresh.return_value = pd.DataFrame()
            df = get_prospect_rankings(top_n=5)
            # Should fall back to static list
            assert len(df) <= 20  # static list has 20


def test_filter_by_position():
    from src.prospect_engine import get_prospect_rankings

    with patch("src.prospect_engine._fetch_from_db") as mock_db:
        mock_db.return_value = pd.DataFrame(
            [
                {"name": "A", "position": "SS", "readiness_score": 80, "fg_rank": 1},
                {"name": "B", "position": "SP", "readiness_score": 70, "fg_rank": 2},
                {"name": "C", "position": "OF", "readiness_score": 60, "fg_rank": 3},
            ]
        )
        df = get_prospect_rankings(position="SP")
        assert len(df) == 1
        assert df.iloc[0]["name"] == "B"


def test_filter_by_org():
    from src.prospect_engine import get_prospect_rankings

    with patch("src.prospect_engine._fetch_from_db") as mock_db:
        mock_db.return_value = pd.DataFrame(
            [
                {"name": "A", "team": "NYY", "position": "SS", "readiness_score": 80, "fg_rank": 1},
                {"name": "B", "team": "LAD", "position": "SP", "readiness_score": 70, "fg_rank": 2},
            ]
        )
        df = get_prospect_rankings(org="NYY")
        assert len(df) == 1


# -- DB round-trip: get_prospect_detail() -------------------------------------


def test_get_prospect_detail_round_trip(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        from src.prospect_engine import _store_prospects, get_prospect_detail

        df = pd.DataFrame(
            [
                {
                    "mlb_id": 12345,
                    "name": "Test Prospect",
                    "team": "NYY",
                    "position": "SS",
                    "fg_rank": 1,
                    "fg_fv": 60,
                    "fg_eta": "2026",
                    "fg_risk": "Medium",
                    "age": 21,
                    "readiness_score": 72.5,
                    "hit_present": 55,
                    "hit_future": 60,
                }
            ]
        )
        _store_prospects(df)
        detail = get_prospect_detail(1)  # prospect_id=1 (first inserted)
        assert detail is not None
        assert detail["name"] == "Test Prospect"
        assert detail["fg_fv"] == 60


def test_get_prospect_detail_missing(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        from src.prospect_engine import get_prospect_detail

        assert get_prospect_detail(999) is None
