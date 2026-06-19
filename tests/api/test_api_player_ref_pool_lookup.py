"""Unit tests for player_ref_from_pool (M0 slice 2)."""

from __future__ import annotations

import pandas as pd

from api.services.player_ref import player_ref_from_pool


def _pool() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"player_id": 1, "name": "Aaron Judge", "positions": "OF", "mlb_id": 592450, "team": "NYY"},
            {"player_id": 2, "name": "NoIds", "positions": "SP", "mlb_id": None, "team": None},
        ]
    )


def test_found_enriches_and_prefers_provided_name_positions():
    ref = player_ref_from_pool(1, _pool(), name="Judge (engine)", positions="RF")
    assert ref.id == 1
    assert ref.mlb_id == 592450
    assert ref.team_abbr == "NYY"
    assert ref.team_id == 147
    assert ref.name == "Judge (engine)"  # provided name preferred over pool's
    assert ref.positions == "RF"  # provided positions preferred


def test_found_falls_back_to_pool_name_positions_when_not_provided():
    ref = player_ref_from_pool(1, _pool())
    assert ref.name == "Aaron Judge"
    assert ref.positions == "OF"
    assert ref.mlb_id == 592450


def test_found_row_with_null_ids_degrades_cleanly():
    ref = player_ref_from_pool(2, _pool(), name="NoIds", positions="SP")
    assert ref.mlb_id is None
    assert ref.team_abbr is None
    assert ref.team_id is None


def test_not_found_uses_provided_then_placeholder():
    ref = player_ref_from_pool(999, _pool(), name="Ghost", positions="OF")
    assert ref.id == 999
    assert ref.name == "Ghost"
    assert ref.mlb_id is None
    ref2 = player_ref_from_pool(999, _pool())
    assert ref2.name == "Player 999"  # placeholder when nothing provided


def test_none_or_empty_pool_does_not_raise():
    assert player_ref_from_pool(1, None, name="X", positions="OF").mlb_id is None
    assert player_ref_from_pool(1, pd.DataFrame(), name="X", positions="OF").team_abbr is None
