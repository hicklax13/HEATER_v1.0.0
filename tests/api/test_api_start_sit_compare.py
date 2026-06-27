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


def _fake_ctx(pool, roster_ids):
    import types

    ctx = types.SimpleNamespace()
    ctx.player_pool = pool
    ctx.user_roster_ids = roster_ids
    ctx.category_weights = {"hr": 1.4, "sb": 0.6}
    ctx.roster = pool[pool["player_id"].isin(roster_ids)].copy()
    ctx.adds_remaining_this_week = 10
    ctx.scope = "today"
    return ctx


def test_compare_scores_ranks_and_builds_verdict(monkeypatch):
    import pandas as pd

    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "OF One",
                "positions": "OF",
                "is_hitter": 1,
                "team": "NYY",
                "selected_position": "BN",
                "hr": 30,
                "r": 90,
                "rbi": 88,
                "sb": 5,
                "avg": 0.290,
                "obp": 0.370,
                "ab": 550,
                "h": 160,
                "bb": 60,
                "hbp": 5,
                "sf": 4,
                "pa": 620,
                "mlb_id": 592450,
            },
            {
                "player_id": 2,
                "name": "OF Two",
                "positions": "OF",
                "is_hitter": 1,
                "team": "BOS",
                "selected_position": "BN",
                "hr": 12,
                "r": 60,
                "rbi": 50,
                "sb": 20,
                "avg": 0.255,
                "obp": 0.320,
                "ab": 500,
                "h": 128,
                "bb": 45,
                "hbp": 3,
                "sf": 4,
                "pa": 560,
                "mlb_id": 519222,
            },
            {
                "player_id": 3,
                "name": "MI Three",
                "positions": "2B,SS",
                "is_hitter": 1,
                "team": "KC",
                "selected_position": "BN",
                "hr": 18,
                "r": 75,
                "rbi": 70,
                "sb": 15,
                "avg": 0.275,
                "obp": 0.340,
                "ab": 540,
                "h": 149,
                "bb": 50,
                "hbp": 4,
                "sf": 5,
                "pa": 600,
                "mlb_id": 677951,
            },
        ]
    )

    monkeypatch.setattr(
        "src.optimizer.shared_data_layer.build_optimizer_context",
        lambda **k: _fake_ctx(pool, [1, 2, 3]),
    )
    # start_sit_recommendation returns higher score for player 1.
    monkeypatch.setattr(
        "src.start_sit.start_sit_recommendation",
        lambda *a, **k: {
            "recommendation": 1,
            "confidence": 0.42,
            "confidence_label": "Clear Start",
            "players": [
                {
                    "player_id": 1,
                    "name": "OF One",
                    "start_score": 9.0,
                    "matchup_factors": {},
                    "floor": 8.0,
                    "ceiling": 10.0,
                    "category_impact": {"HR": 1.2, "SB": 0.1},
                    "reasoning": ["Favorable park"],
                },
                {
                    "player_id": 3,
                    "name": "MI Three",
                    "start_score": 6.0,
                    "matchup_factors": {},
                    "floor": 5.0,
                    "ceiling": 7.0,
                    "category_impact": {"HR": 0.7, "SB": 0.5},
                    "reasoning": ["Average matchup"],
                },
                {
                    "player_id": 2,
                    "name": "OF Two",
                    "start_score": 4.0,
                    "matchup_factors": {},
                    "floor": 3.0,
                    "ceiling": 5.0,
                    "category_impact": {"HR": 0.4, "SB": 0.9},
                    "reasoning": ["SB upside"],
                },
            ],
        },
    )
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: object())

    from api.contracts.start_sit import StartSitCompareRequest

    resp = _svc().compare(StartSitCompareRequest(team_name="Team Hickey", scope="today", player_ids=[1, 2, 3]))
    assert resp.scope == "today"
    # ranked by start_score desc -> 1, 3, 2
    assert [c.player.id for c in resp.candidates] == [1, 3, 2]
    assert resp.candidates[0].rank == 1 and resp.candidates[0].start_score == 100.0  # normalized top = 100
    # 3 OF/Util players, but only 3 OF + 2 Util open -> all 3 can be started here.
    assert set(resp.verdict.start_ids).issubset({1, 2, 3})
    assert resp.candidates[0].eligible_slots  # non-empty (OF/Util)
    assert resp.candidates[0].category_impact  # StatItem list
    assert resp.confidence_label in ("Clear", "Lean", "Toss-up")


def test_compare_cold_env_returns_empty(monkeypatch):
    # build_optimizer_context raises -> graceful empty (never 500).
    def _boom(**k):
        raise RuntimeError("no data")

    monkeypatch.setattr("src.optimizer.shared_data_layer.build_optimizer_context", _boom)
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: object())

    from api.contracts.start_sit import StartSitCompareRequest

    resp = _svc().compare(StartSitCompareRequest(team_name="Team Hickey", scope="today", player_ids=[1, 2]))
    assert resp.candidates == [] and resp.scope == "today"


def test_compare_clamps_to_six(monkeypatch):
    import pandas as pd

    pool = pd.DataFrame(
        [
            {
                "player_id": i,
                "name": f"P{i}",
                "positions": "OF",
                "is_hitter": 1,
                "team": "NYY",
                "selected_position": "BN",
                "hr": 10,
                "r": 50,
                "rbi": 40,
                "sb": 5,
                "avg": 0.26,
                "obp": 0.33,
                "ab": 400,
                "h": 104,
                "bb": 30,
                "hbp": 2,
                "sf": 3,
                "pa": 440,
                "mlb_id": 1000 + i,
            }
            for i in range(1, 9)
        ]
    )
    monkeypatch.setattr(
        "src.optimizer.shared_data_layer.build_optimizer_context", lambda **k: _fake_ctx(pool, list(range(1, 9)))
    )
    captured = {}

    def _rec(player_ids, *a, **k):
        captured["n"] = len(player_ids)
        return {"recommendation": None, "confidence": 0.0, "confidence_label": "Toss-up", "players": []}

    monkeypatch.setattr("src.start_sit.start_sit_recommendation", _rec)
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: object())

    from api.contracts.start_sit import StartSitCompareRequest

    _svc().compare(StartSitCompareRequest(team_name="T", scope="rest_of_season", player_ids=list(range(1, 9))))
    assert captured["n"] == 6  # clamped 8 -> 6


def test_compare_endpoint_contract():
    from starlette.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.start_sit import StartSitCandidate, StartSitCompareResponse, StartSitVerdict
    from api.deps import get_start_sit_service
    from api.main import create_app

    class _Fake:
        def compare(self, req):
            return StartSitCompareResponse(
                scope=req.scope,
                candidates=[
                    StartSitCandidate(
                        player=PlayerRef(id=1, name="X", positions="OF"),
                        start_score=100.0,
                        rank=1,
                        eligible_slots=["OF", "Util"],
                        matchup="vs SF",
                        reason="park",
                    )
                ],
                verdict=StartSitVerdict(start_ids=[1], sit_ids=[2], reasoning="r"),
                open_slots={"OF": 2},
                confidence=0.4,
                confidence_label="Clear",
            )

    app = create_app()
    app.dependency_overrides[get_start_sit_service] = lambda: _Fake()
    try:
        body = (
            TestClient(app)
            .post("/api/start-sit/compare", json={"team_name": "Team Hickey", "scope": "today", "player_ids": [1, 2]})
            .json()
        )
        assert body["scope"] == "today"
        assert body["candidates"][0]["rank"] == 1 and body["candidates"][0]["start_score"] == 100.0
        assert body["verdict"]["start_ids"] == [1]
        assert body["open_slots"]["OF"] == 2
        assert body["confidence_label"] == "Clear"
    finally:
        app.dependency_overrides.clear()


def test_compare_all_negative_scores_clamped(monkeypatch):
    """MUST-FIX 1: an all-negative raw-score slate (e.g. every player a bad
    matchup) must still produce start_scores within [0, 100] — never negative,
    never >100 — so the frontend heat bar stays valid."""
    import pandas as pd

    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Neg One",
                "positions": "OF",
                "is_hitter": 1,
                "team": "NYY",
                "selected_position": "BN",
                "hr": 5,
                "r": 20,
                "rbi": 18,
                "sb": 1,
                "avg": 0.21,
                "obp": 0.28,
                "ab": 300,
                "h": 63,
                "bb": 20,
                "hbp": 1,
                "sf": 2,
                "pa": 330,
                "mlb_id": 1,
            },
            {
                "player_id": 2,
                "name": "Neg Two",
                "positions": "OF",
                "is_hitter": 1,
                "team": "BOS",
                "selected_position": "BN",
                "hr": 4,
                "r": 18,
                "rbi": 15,
                "sb": 0,
                "avg": 0.20,
                "obp": 0.27,
                "ab": 290,
                "h": 58,
                "bb": 18,
                "hbp": 1,
                "sf": 2,
                "pa": 318,
                "mlb_id": 2,
            },
            {
                "player_id": 3,
                "name": "Neg Three",
                "positions": "OF",
                "is_hitter": 1,
                "team": "KC",
                "selected_position": "BN",
                "hr": 3,
                "r": 15,
                "rbi": 12,
                "sb": 0,
                "avg": 0.19,
                "obp": 0.26,
                "ab": 280,
                "h": 53,
                "bb": 16,
                "hbp": 1,
                "sf": 2,
                "pa": 305,
                "mlb_id": 3,
            },
        ]
    )
    monkeypatch.setattr(
        "src.optimizer.shared_data_layer.build_optimizer_context", lambda **k: _fake_ctx(pool, [1, 2, 3])
    )
    # Every player scores NEGATIVE (engine sorts desc -> -3 best, -8 worst).
    monkeypatch.setattr(
        "src.start_sit.start_sit_recommendation",
        lambda *a, **k: {
            "recommendation": 3,
            "confidence": 0.05,
            "confidence_label": "Toss-up",
            "players": [
                {"player_id": 3, "name": "Neg Three", "start_score": -3.0, "category_impact": {}, "reasoning": []},
                {"player_id": 1, "name": "Neg One", "start_score": -5.0, "category_impact": {}, "reasoning": []},
                {"player_id": 2, "name": "Neg Two", "start_score": -8.0, "category_impact": {}, "reasoning": []},
            ],
        },
    )
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: object())

    from api.contracts.start_sit import StartSitCompareRequest

    resp = _svc().compare(StartSitCompareRequest(team_name="T", scope="rest_of_season", player_ids=[1, 2, 3]))
    assert resp.candidates  # non-empty
    for c in resp.candidates:
        assert 0.0 <= c.start_score <= 100.0, f"start_score {c.start_score} out of [0,100]"
    # all-negative slate -> the real max is <= 0 -> every normalized score is 0.0.
    assert all(c.start_score == 0.0 for c in resp.candidates)


def test_compare_passes_computed_weeks_remaining(monkeypatch):
    """MUST-FIX 2: build_optimizer_context must receive a COMPUTED weeks_remaining
    (not the default 16), or late-season projections over-count. Assert the value
    from compute_weeks_remaining reaches the context builder."""
    import pandas as pd

    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "OF One",
                "positions": "OF",
                "is_hitter": 1,
                "team": "NYY",
                "selected_position": "BN",
                "hr": 20,
                "r": 70,
                "rbi": 60,
                "sb": 8,
                "avg": 0.27,
                "obp": 0.34,
                "ab": 500,
                "h": 135,
                "bb": 40,
                "hbp": 3,
                "sf": 4,
                "pa": 547,
                "mlb_id": 1,
            },
            {
                "player_id": 2,
                "name": "OF Two",
                "positions": "OF",
                "is_hitter": 1,
                "team": "BOS",
                "selected_position": "BN",
                "hr": 15,
                "r": 60,
                "rbi": 50,
                "sb": 12,
                "avg": 0.26,
                "obp": 0.33,
                "ab": 490,
                "h": 127,
                "bb": 38,
                "hbp": 2,
                "sf": 3,
                "pa": 533,
                "mlb_id": 2,
            },
        ]
    )
    captured = {}

    def _ctx(**k):
        captured["weeks_remaining"] = k.get("weeks_remaining")
        return _fake_ctx(pool, [1, 2])

    monkeypatch.setattr("src.optimizer.shared_data_layer.build_optimizer_context", _ctx)
    # Pin compute_weeks_remaining to a recognizable sentinel (NOT 16).
    monkeypatch.setattr("src.validation.dynamic_context.compute_weeks_remaining", lambda *a, **k: 7)
    monkeypatch.setattr(
        "src.start_sit.start_sit_recommendation",
        lambda *a, **k: {"recommendation": 1, "confidence": 0.2, "confidence_label": "Lean", "players": []},
    )
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: object())

    from api.contracts.start_sit import StartSitCompareRequest

    _svc().compare(StartSitCompareRequest(team_name="T", scope="rest_of_season", player_ids=[1, 2]))
    assert captured["weeks_remaining"] == 7  # computed value reached the context builder (not the 16 default)
