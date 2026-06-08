"""#10 (2026-06-07): Trade Finder draft-round N+1 fix.

scan_1_for_1 / compute_adp_fairness called get_player_draft_round (one SQLite
connect + query each) per give×recv pair — ~3,500 queries across 11 partners,
~10s of a ~26s local scan (worse on the small Railway replica). Batch all needed
rounds in ONE query (get_player_draft_rounds) and resolve per-pair from the dict.
Draft rounds are immutable post-draft, so the result is identical.
"""

import pandas as pd

from src import trade_finder
from src.database import get_connection, get_player_draft_rounds


def _seed_draft_picks(rows):
    """rows = [(player_id, round)]."""
    conn = get_connection()
    try:
        conn.execute("DELETE FROM league_draft_picks")
        for i, (pid, rnd) in enumerate(rows):
            conn.execute(
                "INSERT INTO league_draft_picks (pick_number, round, team_name, player_id, player_name) "
                "VALUES (?, ?, ?, ?, ?)",
                (i + 1, rnd, "Team X", pid, f"Player {pid}"),
            )
        conn.commit()
    finally:
        conn.close()


def test_get_player_draft_rounds_batch_returns_dict():
    _seed_draft_picks([(101, 2), (102, 15)])
    out = get_player_draft_rounds([101, 102, 999])
    assert out[101] == 2
    assert out[102] == 15
    assert 999 not in out  # undrafted omitted


def test_get_player_draft_rounds_empty_input_no_query():
    assert get_player_draft_rounds([]) == {}


def test_compute_adp_fairness_uses_passed_draft_rounds(monkeypatch):
    """With a draft_rounds dict provided, compute_adp_fairness must NOT call the
    per-player DB function — that's the N+1 elimination."""
    calls = []
    monkeypatch.setattr("src.database.get_player_draft_round", lambda pid: calls.append(pid))
    pool = pd.DataFrame([{"player_id": 1, "adp": 10}, {"player_id": 2, "adp": 12}])
    fairness = trade_finder.compute_adp_fairness(1, 2, pool, draft_rounds={1: 2, 2: 5})
    assert calls == [], "must use the passed dict, not query per-player"
    assert abs(fairness - (1.0 - 3 / 23)) < 1e-9  # gap 3 over 23 max rounds


def test_compute_adp_fairness_backward_compat_queries_when_no_dict(monkeypatch):
    """No dict passed -> falls back to the per-player query (existing behavior)."""
    monkeypatch.setattr("src.database.get_player_draft_round", lambda pid: {1: 2, 2: 5}.get(pid))
    pool = pd.DataFrame([{"player_id": 1, "adp": 10}, {"player_id": 2, "adp": 12}])
    fairness = trade_finder.compute_adp_fairness(1, 2, pool)
    assert abs(fairness - (1.0 - 3 / 23)) < 1e-9
