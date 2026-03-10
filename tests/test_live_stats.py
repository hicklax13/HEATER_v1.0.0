"""Test live stats module — MLB Stats API + pybaseball integration."""

import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.database as db_mod
from src.database import init_db


@pytest.fixture(autouse=True)
def temp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    original = db_mod.DB_PATH
    db_mod.DB_PATH = Path(tmp.name)
    init_db()
    conn = sqlite3.connect(tmp.name)
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (1, 'Aaron Judge', 'NYY', 'OF', 1)"
    )
    conn.execute(
        "INSERT INTO players (player_id, name, team, positions, is_hitter) "
        "VALUES (2, 'Gerrit Cole', 'NYY', 'SP', 0)"
    )
    conn.commit()
    conn.close()
    yield tmp.name
    db_mod.DB_PATH = original
    try:
        os.unlink(tmp.name)
    except PermissionError:
        pass


def test_match_player_id(temp_db):
    from src.live_stats import match_player_id

    pid = match_player_id("Aaron Judge", "NYY")
    assert pid == 1


def test_match_player_id_not_found(temp_db):
    from src.live_stats import match_player_id

    pid = match_player_id("Nonexistent Player", "XXX")
    assert pid is None


@patch("src.live_stats.statsapi")
def test_fetch_season_stats_structure(mock_statsapi, temp_db):
    from src.live_stats import fetch_season_stats

    mock_statsapi.get.return_value = {
        "people": [
            {
                "id": 592450,
                "fullName": "Aaron Judge",
                "currentTeam": {"abbreviation": "NYY"},
                "stats": [
                    {
                        "splits": [
                            {
                                "stat": {
                                    "plateAppearances": 500,
                                    "atBats": 450,
                                    "hits": 130,
                                    "runs": 80,
                                    "homeRuns": 35,
                                    "rbi": 90,
                                    "stolenBases": 5,
                                    "avg": ".289",
                                    "gamesPlayed": 120,
                                }
                            }
                        ]
                    }
                ],
            }
        ]
    }

    df = fetch_season_stats(season=2026)
    assert isinstance(df, pd.DataFrame)
    assert "player_name" in df.columns
    assert "hr" in df.columns


def test_get_refresh_age():
    from src.live_stats import _get_refresh_age_hours

    age = _get_refresh_age_hours("nonexistent_source")
    assert age > 24


@patch("src.live_stats.statsapi")
def test_refresh_all_stats(mock_statsapi, temp_db):
    from src.live_stats import refresh_all_stats

    mock_statsapi.get.return_value = {"people": []}

    result = refresh_all_stats(force=True)
    assert isinstance(result, dict)
    assert "season_stats" in result
