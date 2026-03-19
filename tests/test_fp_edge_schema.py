# tests/test_fp_edge_schema.py
"""Tests for FP Edge feature database tables."""

import sqlite3
from unittest.mock import patch

import pytest


@pytest.fixture
def _temp_db(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        yield db_path


def _table_exists(db_path, table_name):
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def _get_columns(db_path, table_name):
    conn = sqlite3.connect(str(db_path))
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    cols = [row[1] for row in cur.fetchall()]
    conn.close()
    return cols


def test_prospect_rankings_table_exists(_temp_db):
    assert _table_exists(_temp_db, "prospect_rankings")


def test_prospect_rankings_columns(_temp_db):
    cols = _get_columns(_temp_db, "prospect_rankings")
    for expected in [
        "prospect_id",
        "mlb_id",
        "name",
        "fg_rank",
        "fg_fv",
        "readiness_score",
        "milb_avg",
        "milb_era",
        "age",
    ]:
        assert expected in cols, f"Missing column: {expected}"


def test_ecr_consensus_table_exists(_temp_db):
    assert _table_exists(_temp_db, "ecr_consensus")


def test_ecr_consensus_columns(_temp_db):
    cols = _get_columns(_temp_db, "ecr_consensus")
    for expected in [
        "player_id",
        "espn_rank",
        "yahoo_adp",
        "consensus_rank",
        "consensus_avg",
        "rank_stddev",
        "n_sources",
    ]:
        assert expected in cols, f"Missing column: {expected}"


def test_player_id_map_table_exists(_temp_db):
    assert _table_exists(_temp_db, "player_id_map")


def test_player_id_map_columns(_temp_db):
    cols = _get_columns(_temp_db, "player_id_map")
    for expected in ["player_id", "espn_id", "yahoo_key", "fg_id", "mlb_id"]:
        assert expected in cols, f"Missing column: {expected}"


def test_player_news_table_exists(_temp_db):
    assert _table_exists(_temp_db, "player_news")


def test_player_news_columns(_temp_db):
    cols = _get_columns(_temp_db, "player_news")
    for expected in [
        "player_id",
        "source",
        "headline",
        "news_type",
        "il_status",
        "sentiment_score",
    ]:
        assert expected in cols, f"Missing column: {expected}"


def test_ownership_trends_table_exists(_temp_db):
    assert _table_exists(_temp_db, "ownership_trends")


def test_ownership_trends_columns(_temp_db):
    cols = _get_columns(_temp_db, "ownership_trends")
    for expected in ["player_id", "date", "percent_owned", "delta_7d"]:
        assert expected in cols, f"Missing column: {expected}"


def test_player_id_map_unique_indexes(_temp_db):
    """Verify filtered unique indexes exist on player_id_map."""
    conn = sqlite3.connect(str(_temp_db))
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='player_id_map'")
    indexes = [row[0] for row in cur.fetchall()]
    conn.close()
    assert "idx_id_map_espn" in indexes
    assert "idx_id_map_yahoo" in indexes
    assert "idx_id_map_fg" in indexes
    assert "idx_id_map_mlb" in indexes


def test_player_news_indexes(_temp_db):
    conn = sqlite3.connect(str(_temp_db))
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='player_news'")
    indexes = [row[0] for row in cur.fetchall()]
    conn.close()
    assert "idx_player_news_player" in indexes
    assert "idx_player_news_type" in indexes
