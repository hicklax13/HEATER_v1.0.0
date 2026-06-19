import pandas as pd
from starlette.testclient import TestClient

from api.contracts.draft import (
    DraftClock,
    DraftConfig,
    DraftPick,
    DraftSimulatePicksRequest,
    DraftSimulatePicksResponse,
)
from api.deps import get_draft_service
from api.main import create_app
from api.services.draft_service import DraftService


def test_simulate_request_defaults():
    req = DraftSimulatePicksRequest()
    assert req.config.num_teams == 12
    assert req.config.user_team_index == 0
    assert req.pick_log == []
    assert req.seed is None


def test_simulate_response_shape():
    resp = DraftSimulatePicksResponse(
        clock=DraftClock(current_pick=2, round=1, pick_in_round=3, picking_team_index=2, is_user_turn=True),
        picks=[DraftPick(pick=0, team_index=0, player_id=101, player_name="A", positions="SS")],
        summary="1 opponent pick simulated.",
    )
    dumped = resp.model_dump()
    assert dumped["clock"]["is_user_turn"] is True
    assert dumped["picks"][0]["player_id"] == 101
    assert dumped["summary"].startswith("1 opponent")


def _sim_pool() -> pd.DataFrame:
    positions = ["SS", "OF", "2B", "3B", "1B", "SP", "RP", "C"]
    rows = [
        {
            "player_id": 100 + i,
            "name": f"Player{i}",
            "player_name": f"Player{i}",
            "positions": positions[i % len(positions)],
            "adp": float(i + 1),
        }
        for i in range(20)
    ]
    return pd.DataFrame(rows)


def test_simulate_picks_advances_to_user_turn_with_seed():
    # user at seat 2, fresh draft → exactly seats 0 and 1 auto-pick, then user's turn.
    req = DraftSimulatePicksRequest(config=DraftConfig(num_teams=12, user_team_index=2), pick_log=[], seed=42)
    resp = DraftService().simulate_picks(req, pool=_sim_pool())
    assert resp.clock.is_user_turn is True
    assert resp.clock.picking_team_index == 2
    assert [p.team_index for p in resp.picks] == [0, 1]
    assert "2 opponent picks" in resp.summary


def test_simulate_picks_seed_is_reproducible():
    req = DraftSimulatePicksRequest(config=DraftConfig(user_team_index=2), seed=7)
    pool = _sim_pool()
    a = [p.player_id for p in DraftService().simulate_picks(req, pool=pool).picks]
    b = [p.player_id for p in DraftService().simulate_picks(req, pool=pool).picks]
    assert a == b


def test_simulate_picks_no_picks_when_user_on_clock():
    req = DraftSimulatePicksRequest(config=DraftConfig(user_team_index=0), seed=1)
    resp = DraftService().simulate_picks(req, pool=_sim_pool())
    assert resp.picks == []
    assert resp.clock.is_user_turn is True


def test_simulate_picks_graceful_when_pool_load_fails():
    # pool=None forces the real load_player_pool(); in a DB-less env it raises and
    # the service must still return a valid clock with no picks (never 500).
    req = DraftSimulatePicksRequest(config=DraftConfig(user_team_index=3), seed=1)
    resp = DraftService().simulate_picks(req, pool=None)
    assert isinstance(resp, DraftSimulatePicksResponse)
    assert resp.clock.round >= 1  # clock always computed from the rebuilt state


class _FakeDraftService:
    def simulate_picks(self, req) -> DraftSimulatePicksResponse:
        return DraftSimulatePicksResponse(
            clock=DraftClock(current_pick=2, round=1, pick_in_round=3, picking_team_index=2, is_user_turn=True),
            picks=[
                DraftPick(pick=0, team_index=0, player_id=101, player_name="A", positions="SS"),
                DraftPick(pick=1, team_index=1, player_id=102, player_name="B", positions="OF"),
            ],
            summary="2 opponent picks simulated.",
        )


def test_post_draft_simulate_returns_contract():
    app = create_app()
    app.dependency_overrides[get_draft_service] = lambda: _FakeDraftService()
    client = TestClient(app)
    resp = client.post(
        "/api/draft/simulate-picks",
        json={"config": {"num_teams": 12, "user_team_index": 2}, "pick_log": [], "seed": 7},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["clock"]["is_user_turn"] is True
    assert len(body["picks"]) == 2
    assert body["picks"][0]["player_id"] == 101


def test_post_draft_simulate_accepts_empty_body_defaults():
    app = create_app()
    app.dependency_overrides[get_draft_service] = lambda: _FakeDraftService()
    client = TestClient(app)
    resp = client.post("/api/draft/simulate-picks", json={})  # all fields default
    assert resp.status_code == 200
    assert resp.json()["picks"][0]["team_index"] == 0
