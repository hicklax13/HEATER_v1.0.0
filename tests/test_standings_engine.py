"""Tests for standings engine — database layer."""

from __future__ import annotations

import pytest
import pandas as pd

from src.database import (
    init_db,
    get_connection,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    """Ensure fresh DB for each test, clearing new tables."""
    init_db()
    conn = get_connection()
    try:
        conn.execute("DELETE FROM league_schedule_full")
        conn.execute("DELETE FROM league_records")
        conn.commit()
    finally:
        conn.close()


class TestLeagueScheduleFullTable:
    """Tests for league_schedule_full table CRUD."""

    def test_upsert_and_load_full_schedule(self):
        from src.database import upsert_league_schedule_full, load_league_schedule_full

        upsert_league_schedule_full(1, "Team A", "Team B")
        upsert_league_schedule_full(1, "Team C", "Team D")
        upsert_league_schedule_full(2, "Team A", "Team C")

        result = load_league_schedule_full()
        assert isinstance(result, dict)
        assert 1 in result
        assert 2 in result
        assert len(result[1]) == 2  # 2 matchups in week 1
        assert ("Team A", "Team B") in result[1]

    def test_upsert_full_schedule_idempotent(self):
        from src.database import upsert_league_schedule_full, load_league_schedule_full

        upsert_league_schedule_full(1, "Team A", "Team B")
        upsert_league_schedule_full(1, "Team A", "Team B")  # duplicate

        result = load_league_schedule_full()
        assert len(result[1]) == 1  # no duplicate

    def test_load_empty_full_schedule(self):
        from src.database import load_league_schedule_full

        result = load_league_schedule_full()
        assert result == {}


class TestLeagueRecordsTable:
    """Tests for league_records table CRUD."""

    def test_upsert_and_load_records(self):
        from src.database import upsert_league_record, load_league_records

        upsert_league_record("Team Hickey", wins=42, losses=32, ties=6,
                             win_pct=0.563, streak="L1", rank=3)
        upsert_league_record("Jonny Jockstrap", wins=48, losses=26, ties=6,
                             win_pct=0.638, streak="W3", rank=1)

        df = load_league_records()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.loc[df["team_name"] == "Team Hickey", "wins"].iloc[0] == 42

    def test_upsert_record_overwrites(self):
        from src.database import upsert_league_record, load_league_records

        upsert_league_record("Team A", wins=10, losses=5, ties=1,
                             win_pct=0.656, streak="W1", rank=1)
        upsert_league_record("Team A", wins=11, losses=5, ties=1,
                             win_pct=0.688, streak="W2", rank=1)

        df = load_league_records()
        assert len(df) == 1
        assert df.iloc[0]["wins"] == 11

    def test_load_empty_records(self):
        from src.database import load_league_records

        df = load_league_records()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
