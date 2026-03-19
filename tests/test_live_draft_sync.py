"""Tests for live draft sync."""

from __future__ import annotations

import pandas as pd

from src.live_draft_sync import DraftPick, LiveDraftSyncer, SyncResult


def _make_pool():
    return pd.DataFrame(
        [
            {"player_id": 1, "name": "Mike Trout", "team": "LAA", "positions": "OF"},
            {"player_id": 2, "name": "Shohei Ohtani", "team": "LAD", "positions": "DH,SP"},
            {"player_id": 3, "name": "Aaron Judge", "team": "NYY", "positions": "OF"},
        ]
    )


def _make_syncer(**kwargs):
    return LiveDraftSyncer(player_pool=_make_pool(), **kwargs)


def test_detect_new_picks():
    s = _make_syncer()
    picks = [{"player_id": 1, "player_name": "Mike Trout", "team_index": 0, "positions": "OF"}]
    new = s.detect_new_picks(picks)
    assert len(new) == 1
    assert new[0].player_name == "Mike Trout"


def test_incremental_sync():
    s = _make_syncer()
    picks1 = [{"player_id": 1, "player_name": "Trout", "team_index": 0}]
    s.detect_new_picks(picks1)
    assert s.known_pick_count == 1
    picks2 = picks1 + [{"player_id": 2, "player_name": "Ohtani", "team_index": 1}]
    new = s.detect_new_picks(picks2)
    assert len(new) == 1
    assert s.known_pick_count == 2


def test_no_change_poll():
    s = _make_syncer()
    picks = [{"player_id": 1, "player_name": "Trout", "team_index": 0}]
    s.detect_new_picks(picks)
    new = s.detect_new_picks(picks)
    assert len(new) == 0


def test_player_resolution():
    s = _make_syncer()
    assert s.resolve_player("Mike Trout") == 1
    assert s.resolve_player("Shohei Ohtani") == 2


def test_player_resolution_missing():
    s = _make_syncer()
    assert s.resolve_player("Nobody Here") is None


def test_user_turn_detection():
    s = _make_syncer(num_teams=12, user_team_index=0)
    assert s.is_user_turn(0) is True  # Pick 1
    assert s.is_user_turn(1) is False  # Pick 2


def test_user_turn_snake_reverse():
    s = _make_syncer(num_teams=12, user_team_index=0)
    # Round 2 (picks 12-23): reversed, so team 0 picks last = pick 23
    assert s.is_user_turn(23) is True


def test_draft_complete():
    s = _make_syncer(num_teams=2, num_rounds=2)
    picks = [{"player_id": i, "player_name": f"P{i}", "team_index": i % 2} for i in range(4)]
    result = s.poll_and_sync(picks)
    assert result.draft_complete is True


def test_full_reconciliation():
    s = _make_syncer()
    s.add_known_pick(DraftPick(1, 0, 99, "Old"))
    picks = [{"player_id": 1, "player_name": "Trout", "team_index": 0}]
    result = s.full_reconciliation(picks)
    assert s.known_pick_count == 1


def test_last_n_picks():
    s = _make_syncer()
    for i in range(5):
        s.add_known_pick(DraftPick(i + 1, 0, i + 1, f"Player {i + 1}"))
    last3 = s.get_last_n_picks(3)
    assert len(last3) == 3
    assert last3[-1].player_name == "Player 5"


def test_sync_result_structure():
    result = SyncResult()
    assert result.new_picks == []
    assert result.is_user_turn is False
    assert result.error is None


def test_error_counting():
    s = _make_syncer()
    assert not s.is_syncing_disabled
    s._consecutive_errors = 3
    assert s.is_syncing_disabled
