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


def _svc():
    from api.services.start_sit_service import StartSitService

    return StartSitService()


def test_league_starting_slots_template():
    from api.services.start_sit_service import STARTING_SLOTS

    # FourzynBurn starting template (no BN/IL — those are not "open lineup slots").
    assert STARTING_SLOTS == [
        "C",
        "1B",
        "2B",
        "3B",
        "SS",
        "OF",
        "OF",
        "OF",
        "Util",
        "Util",
        "SP",
        "SP",
        "RP",
        "RP",
        "P",
        "P",
        "P",
        "P",
    ]


def test_eligible_slots_maps_positions_to_template():
    svc = _svc()
    # A 2B/SS hitter is eligible at 2B, SS, and Util (any hitter fills Util).
    assert set(svc._eligible_slots("2B,SS", is_hitter=True)) == {"2B", "SS", "Util"}
    # A pure SP is eligible at SP and the generic P slot (not RP).
    assert set(svc._eligible_slots("SP", is_hitter=False)) == {"SP", "P"}
    # A SP/RP swingman fills SP, RP, and P.
    assert set(svc._eligible_slots("SP,RP", is_hitter=False)) == {"SP", "RP", "P"}
    # An OF fills OF and Util.
    assert set(svc._eligible_slots("OF", is_hitter=True)) == {"OF", "Util"}


def test_open_slots_subtracts_current_lineup():
    import pandas as pd

    svc = _svc()
    # Roster: C + 1B already started (selected_position set); a benched 2B; an IL SS.
    roster = pd.DataFrame(
        [
            {"player_id": 1, "positions": "C", "selected_position": "C", "is_hitter": 1},
            {"player_id": 2, "positions": "1B", "selected_position": "1B", "is_hitter": 1},
            {"player_id": 3, "positions": "2B", "selected_position": "BN", "is_hitter": 1},
            {"player_id": 4, "positions": "SS", "selected_position": "IL", "is_hitter": 1},
        ]
    )
    open_slots = svc._open_slots(roster)
    # C and 1B are taken; everything else in the template is open.
    assert open_slots.get("C", 0) == 0
    assert open_slots.get("1B", 0) == 0
    assert open_slots.get("2B", 0) == 1
    assert open_slots.get("OF", 0) == 3  # all three OF slots open
    assert open_slots.get("P", 0) == 4


def test_open_slots_empty_roster_returns_full_template():
    import pandas as pd

    svc = _svc()
    open_slots = svc._open_slots(pd.DataFrame())
    assert open_slots["OF"] == 3 and open_slots["P"] == 4 and open_slots["C"] == 1
