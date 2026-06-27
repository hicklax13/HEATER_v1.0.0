"""Start/Sit compare: contract shapes + DB-free service/endpoint tests.

The service imports src engines lazily inside methods, so unit tests monkeypatch
at the SOURCE module (the worktree/CI DB is empty — see reference_worktree_empty_db)."""

from __future__ import annotations


def test_contracts_import_and_shape():
    from api.contracts.common import PlayerRef
    from api.contracts.start_sit import (
        StartSitCandidate,
        StartSitCompareRequest,
        StartSitCompareResponse,
        StartSitOptimizeRequest,
        StartSitVerdict,
    )

    req = StartSitCompareRequest(team_name="Team Hickey", scope="today", player_ids=[1, 2])
    assert req.scope == "today" and req.player_ids == [1, 2]

    cand = StartSitCandidate(
        player=PlayerRef(id=1, name="X", positions="OF"),
        start_score=72.0,
        rank=1,
        eligible_slots=["OF", "Util"],
        projected=[],
        category_impact=[],
        matchup="vs SF",
        reason="favorable park",
        playable=True,
    )
    resp = StartSitCompareResponse(
        scope="today",
        candidates=[cand],
        verdict=StartSitVerdict(start_ids=[1], sit_ids=[2], reasoning="r"),
        open_slots={"OF": 1},
        confidence=0.6,
        confidence_label="Lean",
    )
    assert resp.candidates[0].rank == 1 and resp.open_slots["OF"] == 1
    assert resp.verdict.start_ids == [1]

    opt = StartSitOptimizeRequest(team_name=None, scope="rest_of_week", player_ids=[1, 2, 3])
    assert opt.team_name is None and opt.scope == "rest_of_week"
