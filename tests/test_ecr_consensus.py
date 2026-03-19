# tests/test_ecr_consensus.py
"""Tests for multi-platform ECR consensus."""

from __future__ import annotations

import json
import statistics
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ── Trimmed Borda Count tests ────────────────────────────────────────


def test_consensus_single_source():
    from src.ecr import _compute_player_consensus

    result = _compute_player_consensus({"fg_adp": 25})
    assert result["consensus_avg"] == 25.0
    assert result["n_sources"] == 1


def test_consensus_three_sources_no_trim():
    from src.ecr import _compute_player_consensus

    result = _compute_player_consensus({"a": 10, "b": 20, "c": 30})
    assert result["consensus_avg"] == 20.0
    assert result["n_sources"] == 3
    assert result["rank_min"] == 10
    assert result["rank_max"] == 30


def test_consensus_four_sources_trims_outliers():
    from src.ecr import _compute_player_consensus

    result = _compute_player_consensus({"a": 5, "b": 20, "c": 25, "d": 100})
    # Trim highest (100) and lowest (5), avg of [20, 25] = 22.5
    assert result["consensus_avg"] == 22.5
    assert result["n_sources"] == 4
    # min/max from ALL sources, not trimmed
    assert result["rank_min"] == 5
    assert result["rank_max"] == 100


def test_consensus_five_sources_trims():
    from src.ecr import _compute_player_consensus

    result = _compute_player_consensus({"a": 1, "b": 10, "c": 20, "d": 30, "e": 200})
    # Trim 1 and 200, avg of [10, 20, 30] = 20.0
    assert result["consensus_avg"] == 20.0


def test_consensus_stddev():
    from src.ecr import _compute_player_consensus

    result = _compute_player_consensus({"a": 10, "b": 20, "c": 30})
    expected = round(statistics.stdev([10, 20, 30]), 1)
    assert result["rank_stddev"] == expected


def test_consensus_stddev_single_source():
    from src.ecr import _compute_player_consensus

    result = _compute_player_consensus({"a": 10})
    assert result["rank_stddev"] == 0.0


def test_consensus_empty():
    from src.ecr import _compute_player_consensus

    result = _compute_player_consensus({})
    assert result["consensus_avg"] is None


def test_consensus_none_values_filtered():
    from src.ecr import _compute_player_consensus

    result = _compute_player_consensus({"a": 10, "b": None, "c": 30})
    assert result["n_sources"] == 2
    assert result["consensus_avg"] == 20.0


def test_consensus_seven_sources():
    from src.ecr import _compute_player_consensus

    result = _compute_player_consensus(
        {
            "espn": 15,
            "yahoo": 20,
            "cbs": 18,
            "nfbc": 22,
            "fg": 16,
            "fp": 19,
            "heater": 17,
        }
    )
    # Trim highest (22) and lowest (15), avg of [16, 17, 18, 19, 20] = 18.0
    assert result["consensus_avg"] == 18.0
    assert result["n_sources"] == 7


# ── Consensus rank assignment ────────────────────────────────────────


def test_assign_consensus_ranks_sequential():
    from src.ecr import assign_consensus_ranks

    players = [
        {"name": "A", "consensus_avg": 30.0},
        {"name": "B", "consensus_avg": 10.0},
        {"name": "C", "consensus_avg": 20.0},
    ]
    result = assign_consensus_ranks(players)
    ranked = {p["name"]: p["consensus_rank"] for p in result}
    assert ranked["B"] == 1  # lowest avg = rank 1
    assert ranked["C"] == 2
    assert ranked["A"] == 3


def test_assign_consensus_ranks_none_unranked():
    from src.ecr import assign_consensus_ranks

    players = [
        {"name": "A", "consensus_avg": 10.0},
        {"name": "B", "consensus_avg": None},
    ]
    result = assign_consensus_ranks(players)
    a = next(p for p in result if p["name"] == "A")
    b = next(p for p in result if p["name"] == "B")
    assert a["consensus_rank"] == 1
    assert b.get("consensus_rank") is None


# ── Disagreement detection ───────────────────────────────────────────


def test_disagreement_high():
    from src.ecr import compute_ecr_disagreement

    row = {
        "rank_stddev": 35.0,
        "n_sources": 4,
        "rank_min": 10,
        "rank_max": 90,
        "consensus_avg": 40.0,
        "espn_rank": 10,
        "yahoo_adp": 90,
    }
    badge = compute_ecr_disagreement(row)
    assert badge is not None
    assert "High" in badge


def test_disagreement_moderate():
    from src.ecr import compute_ecr_disagreement

    row = {"rank_stddev": 20.0, "n_sources": 3, "rank_min": 20, "rank_max": 60, "consensus_avg": 35.0}
    badge = compute_ecr_disagreement(row)
    assert badge is not None
    assert "Moderate" in badge


def test_disagreement_none_low_stddev():
    from src.ecr import compute_ecr_disagreement

    row = {"rank_stddev": 5.0, "n_sources": 5}
    badge = compute_ecr_disagreement(row)
    assert badge is None


def test_disagreement_none_too_few_sources():
    from src.ecr import compute_ecr_disagreement

    row = {"rank_stddev": 50.0, "n_sources": 2}
    badge = compute_ecr_disagreement(row)
    assert badge is None


# ── ESPN API parsing ─────────────────────────────────────────────────

MOCK_ESPN_PLAYER = {
    "id": 12345,
    "fullName": "Mike Trout",
    "defaultPositionId": 8,
    "draftRanksByRankType": {"STANDARD": {"rank": 42}},
}


def test_parse_espn_player():
    from src.ecr import _parse_espn_player

    result = _parse_espn_player(MOCK_ESPN_PLAYER)
    assert result["espn_id"] == 12345
    assert result["name"] == "Mike Trout"
    assert result["espn_rank"] == 42


def test_parse_espn_player_missing_rank():
    from src.ecr import _parse_espn_player

    player = {"id": 1, "fullName": "Nobody", "draftRanksByRankType": {}}
    result = _parse_espn_player(player)
    assert result["espn_rank"] is None


# ── Backward compatibility ───────────────────────────────────────────


def test_blend_ecr_with_projections_signature():
    from src.ecr import blend_ecr_with_projections

    pool = pd.DataFrame(
        [
            {"name": "Player A", "pick_score": 10.0},
            {"name": "Player B", "pick_score": 8.0},
        ]
    )
    # Should accept both old ecr_df and new consensus_df forms
    result = blend_ecr_with_projections(pool, pd.DataFrame(), ecr_weight=0.15)
    assert "blended_rank" in result.columns
    assert "ecr_badge" in result.columns


def test_blend_preserves_columns():
    from src.ecr import blend_ecr_with_projections

    pool = pd.DataFrame([{"name": "A", "pick_score": 10.0}])
    result = blend_ecr_with_projections(pool, pd.DataFrame())
    assert "pick_score" in result.columns


# ── CBS parsing (best-effort) ────────────────────────────────────────


def test_parse_cbs_empty_html():
    from src.ecr import _parse_cbs_rankings

    result = _parse_cbs_rankings("")
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# ── DB round-trip ────────────────────────────────────────────────────


def test_store_and_load_consensus(tmp_path):
    from unittest.mock import patch

    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        from src.ecr import _store_consensus, load_ecr_consensus

        consensus_df = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "espn_rank": 10,
                    "yahoo_adp": 15.0,
                    "cbs_rank": None,
                    "nfbc_adp": 12.0,
                    "fg_adp": 11.0,
                    "fp_ecr": 13,
                    "heater_sgp_rank": 8,
                    "consensus_rank": 1,
                    "consensus_avg": 11.5,
                    "rank_min": 8,
                    "rank_max": 15,
                    "rank_stddev": 2.3,
                    "n_sources": 6,
                }
            ]
        )
        count = _store_consensus(consensus_df)
        assert count == 1
        loaded = load_ecr_consensus()
        assert len(loaded) == 1
        assert loaded.iloc[0]["consensus_rank"] == 1


# ── ESPN pagination ─────────────────────────────────────────────────


def test_espn_pagination_fetches_multiple_pages():
    from src.ecr import fetch_espn_rankings

    page1 = [
        {"id": i, "fullName": f"Player {i}", "defaultPositionId": 8, "draftRanksByRankType": {"STANDARD": {"rank": i}}}
        for i in range(1, 51)
    ]
    page2 = [
        {"id": i, "fullName": f"Player {i}", "defaultPositionId": 8, "draftRanksByRankType": {"STANDARD": {"rank": i}}}
        for i in range(51, 60)
    ]
    with patch("src.ecr._espn_api_request") as mock_req:
        mock_req.side_effect = [page1, page2, []]  # 3 pages, last empty
        df = fetch_espn_rankings()
        assert len(df) == 59


# ── Yahoo ADP extraction ────────────────────────────────────────────


def test_fetch_yahoo_adp_from_client():
    from src.ecr import fetch_yahoo_adp

    mock_client = MagicMock()
    mock_client.get_league_draft_results.return_value = pd.DataFrame(
        [
            {"player_name": "Player A", "pick": 5},
            {"player_name": "Player B", "pick": 12},
        ]
    )
    with patch("src.ecr._get_yahoo_client", return_value=mock_client):
        df = fetch_yahoo_adp()
        assert "yahoo_adp" in df.columns or "pick" in df.columns


def test_fetch_yahoo_adp_no_client():
    from src.ecr import fetch_yahoo_adp

    with patch("src.ecr._get_yahoo_client", return_value=None):
        df = fetch_yahoo_adp()
        assert df.empty


# ── Player ID map DB round-trip ─────────────────────────────────────


def test_store_and_load_player_id_map(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        from src.ecr import _lookup_player_id_by_espn, _upsert_player_id_map

        _upsert_player_id_map(player_id=42, espn_id=30836, name="Mike Trout")
        result = _lookup_player_id_by_espn(30836)
        assert result == 42


def test_player_id_map_upsert_updates(tmp_path):
    db_path = tmp_path / "test.db"
    with patch("src.database.DB_PATH", db_path):
        from src.database import init_db

        init_db()
        from src.ecr import _lookup_player_id_by_espn, _upsert_player_id_map

        _upsert_player_id_map(player_id=42, espn_id=30836, name="Mike Trout")
        _upsert_player_id_map(player_id=42, espn_id=30836, yahoo_key="mlb.p.545361")
        # Should update, not duplicate
        from src.database import get_connection

        conn = get_connection()
        try:
            count = conn.execute("SELECT COUNT(*) FROM player_id_map WHERE player_id = 42").fetchone()[0]
            assert count == 1
        finally:
            conn.close()


# ── ID dedup/merge conflict ─────────────────────────────────────────


def test_resolve_player_ids_dedup():
    from src.ecr import _resolve_name_to_player_id

    # When two sources have different names for the same player,
    # fuzzy matching should still resolve to same player_id
    pool = pd.DataFrame(
        [
            {"player_id": 1, "name": "Ronald Acuna Jr."},
            {"player_id": 2, "name": "Vladimir Guerrero Jr."},
        ]
    )
    assert _resolve_name_to_player_id("Ronald Acuna", pool) == 1
    assert _resolve_name_to_player_id("Ronald Acuña Jr", pool) == 1
