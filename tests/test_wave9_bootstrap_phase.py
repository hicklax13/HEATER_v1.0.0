"""Wave 9 / Task 3: _bootstrap_minor_league_rosters phase."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def temp_db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    from src.database import init_db

    init_db()
    return db_path


def test_bootstrap_minor_league_inserts_players_with_level(temp_db):
    """The phase writes rows with level='AAA'/'AA' and updates refresh_log."""
    from src.data_bootstrap import BootstrapProgress, _bootstrap_minor_league_rosters
    from src.database import get_connection

    fake_df = pd.DataFrame(
        [
            {
                "mlb_id": 700001,
                "name": "AAA Prospect 1",
                "team": "SCR",
                "positions": "OF",
                "is_hitter": True,
                "bats": "R",
                "throws": "R",
                "birth_date": "2002-05-01",
                "level": "AAA",
            },
            {
                "mlb_id": 700002,
                "name": "AA Prospect 1",
                "team": "SOM",
                "positions": "SP",
                "is_hitter": False,
                "bats": "R",
                "throws": "R",
                "birth_date": "2003-08-15",
                "level": "AA",
            },
        ]
    )

    with patch("src.live_stats.fetch_minor_league_players", return_value=fake_df):
        progress = MagicMock()
        progress.phase = ""
        progress.detail = ""
        result = _bootstrap_minor_league_rosters(progress)

    conn = get_connection()
    try:
        rows = conn.execute("SELECT name, team, positions, level FROM players WHERE level IN ('AAA', 'AA')").fetchall()
    finally:
        conn.close()
    assert len(rows) == 2
    assert {r[3] for r in rows} == {"AAA", "AA"}
    # Result message describes what happened
    assert "2" in result or "minor" in result.lower()


def test_bootstrap_minor_league_handles_empty_response(temp_db):
    """Empty DataFrame from fetch → refresh_log logs 'no_data', no crash."""
    from src.data_bootstrap import BootstrapProgress, _bootstrap_minor_league_rosters
    from src.database import get_connection

    with patch("src.live_stats.fetch_minor_league_players", return_value=pd.DataFrame()):
        progress = MagicMock()
        progress.phase = ""
        progress.detail = ""
        result = _bootstrap_minor_league_rosters(progress)

    conn = get_connection()
    try:
        row = conn.execute("SELECT status FROM refresh_log WHERE source = 'minor_league_rosters'").fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] in ("no_data", "error")
