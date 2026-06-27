"""Start/Sit optimize: DB-free service + endpoint test.

WS3 uses PATH-B: WS1's LineupService.optimize() does NOT accept an extra_ids hook
(verified against the live signature), so optimize() composes the matchup-aware
roster + the LP pipeline DIRECTLY, injecting the selected candidates into the
roster the LP fills, and reuses LineupService._to_slots / _daily_slots for the
shape mapping. The LP/daily machinery is faked at its source module (worktree DB
is empty)."""

from __future__ import annotations


def _svc():
    from api.services.start_sit_service import StartSitService

    return StartSitService()


def test_optimize_injects_candidates_and_forwards_scope(monkeypatch):
    import pandas as pd

    captured = {}

    class _FakeYds:
        def get_team_roster(self, team_name):
            captured["team_name"] = team_name
            # User's enriched roster: one already-rostered OF.
            return pd.DataFrame(
                [{"player_id": 9, "name": "Rostered OF", "positions": "OF", "selected_position": "OF", "is_hitter": 1}]
            )

        def get_matchup(self):
            return {"some": "matchup"}

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _FakeYds())
    # Pool carries the two SELECTED candidates (ids 1, 2) the LP should be able to slot.
    pool = pd.DataFrame(
        [
            {"player_id": 1, "name": "Cand One", "positions": "OF", "is_hitter": 1, "team": "NYY", "mlb_id": 111},
            {"player_id": 2, "name": "Cand Two", "positions": "OF", "is_hitter": 1, "team": "BOS", "mlb_id": 222},
            {"player_id": 9, "name": "Rostered OF", "positions": "OF", "is_hitter": 1, "team": "KC", "mlb_id": 999},
        ]
    )
    monkeypatch.setattr("src.database.load_player_pool", lambda: pool)
    monkeypatch.setattr("src.game_day.get_target_game_date", lambda: "2026-06-27")

    class _FakePipeline:
        def __init__(self, roster, mode="standard", config=None, **k):
            # Assert the SELECTED candidate ids reached the roster the LP fills.
            captured["mode"] = mode
            captured["roster_ids"] = sorted(int(x) for x in roster["player_id"].tolist())

        def optimize(self, **k):
            captured["matchup"] = k.get("matchup")
            return {
                "lineup": {
                    "assignments": [
                        {"player_id": 1, "slot": "OF", "player_name": "Cand One", "positions": "OF"},
                    ],
                    "bench": [],
                    "projected_stats": {"HR": 30},
                    "status": "optimal",
                }
            }

    monkeypatch.setattr("src.optimizer.pipeline.LineupOptimizerPipeline", _FakePipeline)

    from api.contracts.start_sit import StartSitOptimizeRequest

    resp = _svc().optimize(StartSitOptimizeRequest(team_name="Team Hickey", scope="rest_of_week", player_ids=[1, 2]))
    assert resp.scope == "rest_of_week"
    assert captured["team_name"] == "Team Hickey"
    assert captured["mode"] == "standard"  # rest_of_week -> standard LP
    # the user's rostered player (9) + the two selected candidates (1, 2) all reach the LP.
    assert captured["roster_ids"] == [1, 2, 9]
    assert captured["matchup"] == {"some": "matchup"}  # matchup-aware
    # the LP started candidate 1 -> it surfaces as a filled starter slot.
    assert [s.player.id for s in resp.slots] == [1]


def test_optimize_today_uses_daily_mode(monkeypatch):
    import pandas as pd

    captured = {}

    class _FakeYds:
        def get_team_roster(self, team_name):
            return pd.DataFrame()

        def get_matchup(self):
            return None

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: _FakeYds())
    monkeypatch.setattr("src.database.load_player_pool", lambda: pd.DataFrame())
    monkeypatch.setattr("src.game_day.get_target_game_date", lambda: "2026-06-27")
    monkeypatch.setattr("statsapi.schedule", lambda *a, **k: [])

    class _FakePipeline:
        def __init__(self, roster, mode="standard", config=None, **k):
            captured["mode"] = mode

        def optimize(self, **k):
            return {"daily_dcv": pd.DataFrame(), "daily_lineup": {}}

    monkeypatch.setattr("src.optimizer.pipeline.LineupOptimizerPipeline", _FakePipeline)

    from api.contracts.start_sit import StartSitOptimizeRequest

    resp = _svc().optimize(StartSitOptimizeRequest(team_name="T", scope="today", player_ids=[1]))
    assert resp.scope == "today"
    assert captured["mode"] == "daily"  # today -> daily DCV path


def test_optimize_cold_env_returns_empty(monkeypatch):
    def _boom():
        raise RuntimeError("no data")

    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", _boom)

    from api.contracts.start_sit import StartSitOptimizeRequest

    resp = _svc().optimize(StartSitOptimizeRequest(team_name="T", scope="today", player_ids=[1]))
    assert resp.slots == [] and resp.scope == "today"


def test_optimize_endpoint_contract():
    from starlette.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.lineup import LineupSlot
    from api.contracts.start_sit import StartSitOptimizeResponse
    from api.deps import get_start_sit_service
    from api.main import create_app

    class _Fake:
        def optimize(self, req):
            return StartSitOptimizeResponse(
                scope=req.scope,
                slots=[
                    LineupSlot(
                        slot="OF", player=PlayerRef(id=1, name="X", positions="OF"), action="START", status="start"
                    )
                ],
                bench=[],
                summary="1 starters set.",
            )

    app = create_app()
    app.dependency_overrides[get_start_sit_service] = lambda: _Fake()
    try:
        body = (
            TestClient(app)
            .post(
                "/api/start-sit/optimize", json={"team_name": "Team Hickey", "scope": "rest_of_week", "player_ids": [1]}
            )
            .json()
        )
        assert body["scope"] == "rest_of_week"
        assert body["slots"][0]["player"]["id"] == 1
    finally:
        app.dependency_overrides.clear()
