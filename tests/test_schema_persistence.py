"""Tests for database persistence of statcast archive, cross-reference IDs, and computed fields.

Covers Gap 3 (statcast_archive table), Gap 5 (fangraphs_id cross-reference),
and Gap 6 (depth_chart_role, contract_year, news_sentiment, lineup_slot,
spring_training_era columns on players table).
"""

import sqlite3
from unittest.mock import patch

import pytest

from src.database import get_connection, init_db, upsert_player_bulk, upsert_statcast_bulk


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Create a fresh in-memory-like temp DB for each test."""
    db_path = tmp_path / "test_draft_tool.db"
    monkeypatch.setattr("src.database.DB_PATH", db_path)
    init_db()
    return db_path


# ── Gap 3: statcast_archive table ──────────────────────────────────


def test_statcast_archive_table_exists():
    """statcast_archive table should exist after init_db()."""
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='statcast_archive'")
        result = cursor.fetchone()
        assert result is not None, "statcast_archive table should exist"
        assert result[0] == "statcast_archive"
    finally:
        conn.close()


def test_upsert_statcast_bulk_inserts():
    """upsert_statcast_bulk should insert new records."""
    # Create a player first
    upsert_player_bulk([{"name": "Aaron Judge", "team": "NYY", "positions": "OF", "is_hitter": True}])
    conn = get_connection()
    try:
        pid = conn.execute("SELECT player_id FROM players WHERE name = 'Aaron Judge'").fetchone()[0]
    finally:
        conn.close()

    records = [
        {
            "player_id": pid,
            "season": 2025,
            "ev_mean": 92.5,
            "ev_p90": 108.3,
            "barrel_pct": 0.18,
            "hard_hit_pct": 0.52,
            "xba": 0.285,
            "xslg": 0.560,
            "xwoba": 0.400,
            "whiff_pct": 0.28,
            "chase_rate": 0.25,
            "sprint_speed": 27.1,
        }
    ]
    count = upsert_statcast_bulk(records)
    assert count == 1

    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT ev_mean, barrel_pct, xwoba FROM statcast_archive WHERE player_id = ? AND season = ?",
            (pid, 2025),
        ).fetchone()
        assert row is not None
        assert abs(row[0] - 92.5) < 0.01
        assert abs(row[1] - 0.18) < 0.01
        assert abs(row[2] - 0.400) < 0.01
    finally:
        conn.close()


def test_upsert_statcast_bulk_updates():
    """upsert_statcast_bulk should update existing records on conflict."""
    upsert_player_bulk([{"name": "Shohei Ohtani", "team": "LAD", "positions": "Util", "is_hitter": True}])
    conn = get_connection()
    try:
        pid = conn.execute("SELECT player_id FROM players WHERE name = 'Shohei Ohtani'").fetchone()[0]
    finally:
        conn.close()

    # Insert initial record
    upsert_statcast_bulk([{"player_id": pid, "season": 2025, "ev_mean": 90.0, "xwoba": 0.380}])
    # Update same player+season
    upsert_statcast_bulk([{"player_id": pid, "season": 2025, "ev_mean": 93.0, "xwoba": 0.410}])

    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT ev_mean, xwoba FROM statcast_archive WHERE player_id = ? AND season = ?",
            (pid, 2025),
        ).fetchall()
        # Should be exactly 1 row (updated, not duplicated)
        assert len(rows) == 1
        assert abs(rows[0][0] - 93.0) < 0.01
        assert abs(rows[0][1] - 0.410) < 0.01
    finally:
        conn.close()


# ── Gap 5: fangraphs_id cross-reference ────────────────────────────


def test_fangraphs_id_column_exists():
    """players table should have a fangraphs_id column after init_db()."""
    conn = get_connection()
    try:
        cursor = conn.execute("PRAGMA table_info(players)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "fangraphs_id" in columns, "fangraphs_id column should exist on players"
    finally:
        conn.close()


def test_fangraphs_id_updated_via_pipeline():
    """_update_fangraphs_ids should set fangraphs_id on matched players."""
    upsert_player_bulk(
        [
            {"name": "Mike Trout", "team": "LAA", "positions": "OF", "is_hitter": True},
            {"name": "Gerrit Cole", "team": "NYY", "positions": "SP", "is_hitter": False},
        ]
    )

    from src.data_pipeline import _update_fangraphs_ids

    raw_data = {
        "steamer_bat": [
            {"PlayerName": "Mike Trout", "playerid": 10155, "PA": 500},
        ],
        "steamer_pit": [
            {"PlayerName": "Gerrit Cole", "playerid": 13125, "IP": 180},
        ],
    }

    count = _update_fangraphs_ids(raw_data)
    assert count == 2

    conn = get_connection()
    try:
        trout_fg = conn.execute("SELECT fangraphs_id FROM players WHERE name = 'Mike Trout'").fetchone()
        cole_fg = conn.execute("SELECT fangraphs_id FROM players WHERE name = 'Gerrit Cole'").fetchone()
        assert trout_fg[0] == "10155"
        assert cole_fg[0] == "13125"
    finally:
        conn.close()


# ── Gap 6: Computed field columns ──────────────────────────────────


def test_players_schema_has_new_columns():
    """players table should have all 5 new computed field columns."""
    conn = get_connection()
    try:
        cursor = conn.execute("PRAGMA table_info(players)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "depth_chart_role",
            "contract_year",
            "news_sentiment",
            "lineup_slot",
            "spring_training_era",
        }
        missing = expected - columns
        assert not missing, f"Missing columns on players table: {missing}"
    finally:
        conn.close()


def test_depth_chart_role_persisted():
    """_persist_depth_chart_roles should set depth_chart_role on matching players."""
    upsert_player_bulk(
        [
            {"name": "Aaron Judge", "team": "NYY", "positions": "OF", "is_hitter": True},
            {"name": "Gerrit Cole", "team": "NYY", "positions": "SP", "is_hitter": False},
        ]
    )

    from src.data_bootstrap import _persist_depth_chart_roles

    depth_data = {
        "NYY": {
            "lineup": ["Aaron Judge", "Juan Soto", "Giancarlo Stanton"],
            "rotation": ["Gerrit Cole", "Carlos Rodon"],
            "bullpen": {},
        }
    }

    count = _persist_depth_chart_roles(depth_data)
    assert count >= 2  # Judge and Cole should both be updated

    conn = get_connection()
    try:
        judge = conn.execute("SELECT depth_chart_role, lineup_slot FROM players WHERE name = 'Aaron Judge'").fetchone()
        cole = conn.execute("SELECT depth_chart_role, lineup_slot FROM players WHERE name = 'Gerrit Cole'").fetchone()
        assert judge[0] == "starter"
        assert judge[1] == 1  # First in lineup
        assert cole[0] == "starter"  # In rotation -> starter
    finally:
        conn.close()


def test_contract_year_persisted():
    """_persist_contract_years should set contract_year=1 on matching players."""
    upsert_player_bulk(
        [
            {"name": "Mike Trout", "team": "LAA", "positions": "OF", "is_hitter": True},
            {"name": "Mookie Betts", "team": "LAD", "positions": "SS,OF", "is_hitter": True},
        ]
    )

    from src.data_bootstrap import _persist_contract_years

    contract_names = {"mike trout"}  # lowercase as returned by fetch_contract_year_players
    count = _persist_contract_years(contract_names)
    assert count >= 1

    conn = get_connection()
    try:
        trout = conn.execute("SELECT contract_year FROM players WHERE name = 'Mike Trout'").fetchone()
        betts = conn.execute("SELECT contract_year FROM players WHERE name = 'Mookie Betts'").fetchone()
        assert trout[0] == 1, "Trout should be marked as contract year"
        assert betts[0] == 0, "Betts should not be marked as contract year"
    finally:
        conn.close()


def test_news_sentiment_persisted():
    """_persist_news_sentiment should store sentiment scores on players table."""
    upsert_player_bulk(
        [
            {"name": "Juan Soto", "team": "NYY", "positions": "OF", "is_hitter": True},
        ]
    )

    from src.data_bootstrap import _persist_news_sentiment

    # Mock transactions with positive news for Juan Soto
    transactions = [
        {
            "player_name": "Juan Soto",
            "description": "Juan Soto signed a record contract extension.",
            "date": "2026-03-15",
            "transaction_type": "Signing",
        },
    ]

    count = _persist_news_sentiment(transactions)
    assert count >= 1

    conn = get_connection()
    try:
        row = conn.execute("SELECT news_sentiment FROM players WHERE name = 'Juan Soto'").fetchone()
        assert row is not None
        assert row[0] is not None, "news_sentiment should be set"
        # The keyword 'signed' and 'contract' are positive sentiment words
        assert isinstance(row[0], float)
    finally:
        conn.close()


def test_statcast_archive_schema_columns():
    """statcast_archive table should have all expected columns."""
    conn = get_connection()
    try:
        cursor = conn.execute("PRAGMA table_info(statcast_archive)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id",
            "player_id",
            "season",
            "ev_mean",
            "ev_p90",
            "barrel_pct",
            "hard_hit_pct",
            "xba",
            "xslg",
            "xwoba",
            "whiff_pct",
            "chase_rate",
            "sprint_speed",
            "ff_avg_speed",
            "ff_spin_rate",
            "k_pct",
            "bb_pct",
            "gb_pct",
            "stuff_plus",
            "location_plus",
            "pitching_plus",
            "updated_at",
        }
        missing = expected - columns
        assert not missing, f"Missing columns on statcast_archive: {missing}"
    finally:
        conn.close()
