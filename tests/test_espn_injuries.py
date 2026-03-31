"""Tests for ESPN injuries module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.espn_injuries import fetch_espn_injuries, save_espn_injuries_to_db

MOCK_ESPN_RESPONSE = {
    "items": [
        {
            "team": {"displayName": "New York Yankees", "abbreviation": "NYY"},
            "injuries": [
                {
                    "athlete": {"displayName": "Gerrit Cole"},
                    "status": {
                        "type": {"abbreviation": "IL15"},
                        "detail": "Right elbow inflammation",
                    },
                    "type": {"description": "Elbow"},
                    "returnDate": "2026-05-01",
                },
                {
                    "athlete": {"displayName": "Anthony Rizzo"},
                    "status": {
                        "type": {"abbreviation": "DTD"},
                        "detail": "Back tightness",
                    },
                    "type": {"description": "Back"},
                },
            ],
        }
    ]
}


def test_fetch_parses_response():
    with patch("src.espn_injuries.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = MOCK_ESPN_RESPONSE
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        injuries = fetch_espn_injuries()
        assert len(injuries) == 2
        assert injuries[0]["player_name"] == "Gerrit Cole"
        assert injuries[0]["status"] == "IL15"
        assert injuries[0]["team"] == "NYY"
        assert injuries[1]["player_name"] == "Anthony Rizzo"
        assert injuries[1]["status"] == "DTD"


def test_fetch_handles_error():
    with patch("src.espn_injuries.requests.get", side_effect=Exception("network error")):
        injuries = fetch_espn_injuries()
        assert injuries == []


def test_fetch_handles_empty_response():
    with patch("src.espn_injuries.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"items": []}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        injuries = fetch_espn_injuries()
        assert injuries == []
