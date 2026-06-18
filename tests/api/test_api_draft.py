import pandas as pd
from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.draft import (
    DraftClock,
    DraftConfig,
    DraftPick,
    DraftRecommendation,
    DraftRecommendRequest,
    DraftRecommendResponse,
)
from api.deps import get_draft_service
from api.main import create_app
from api.services.draft_service import DraftService


def test_request_defaults():
    req = DraftRecommendRequest()
    assert req.config.num_teams == 12
    assert req.config.num_rounds == 23
    assert req.config.user_team_index == 0
    assert req.config.roster_config is None
    assert req.pick_log == []
    assert req.top_n == 8
    assert req.n_simulations == 300


def test_request_with_picks():
    req = DraftRecommendRequest(
        config=DraftConfig(num_teams=12, user_team_index=3),
        pick_log=[DraftPick(pick=0, team_index=0, player_id=1, player_name="A", positions="SS")],
        top_n=5,
    )
    dumped = req.model_dump()
    assert dumped["config"]["user_team_index"] == 3
    assert dumped["pick_log"][0]["player_id"] == 1
    assert dumped["top_n"] == 5


def test_response_shape():
    resp = DraftRecommendResponse(
        clock=DraftClock(current_pick=0, round=1, pick_in_round=1, picking_team_index=0, is_user_turn=True),
        recommendations=[
            DraftRecommendation(
                player=PlayerRef(id=1, name="A. Player", positions="SS"),
                rank=1,
                score=87.5,
                projected_sgp=4.2,
                confidence=0.8,
                tag="BUY",
            )
        ],
        summary="1 recommendation",
    )
    dumped = resp.model_dump()
    assert dumped["clock"]["is_user_turn"] is True
    assert dumped["recommendations"][0]["player"]["name"] == "A. Player"
    assert dumped["recommendations"][0]["reason"] == ""


def _req(picks=None, top_n_req=8):
    return DraftRecommendRequest(pick_log=picks or [], top_n=top_n_req)


class _FakeEngine:
    """Stand-in for DraftRecommendationEngine — returns a canned result frame."""

    def __init__(self, frame):
        self._frame = frame
        self.called_with = None

    def recommend(self, player_pool, draft_state, top_n=8, n_simulations=300):
        self.called_with = {"top_n": top_n, "n_simulations": n_simulations}
        return self._frame


def test_rebuild_and_clock_empty_draft():
    # 12 teams, user seat 0, no picks yet → pick 0, round 1, user on the clock.
    svc = DraftService()
    ds = svc._rebuild_state(_req())
    clock = svc._clock(ds)
    assert clock.current_pick == 0
    assert clock.round == 1
    assert clock.pick_in_round == 1
    assert clock.picking_team_index == 0
    assert clock.is_user_turn is True


def test_clock_snake_order_after_replays():
    # After 1 pick → team 1 on the clock (forward round 1).
    svc = DraftService()
    one = [DraftPick(pick=0, team_index=0, player_id=1, player_name="A", positions="SS")]
    clock1 = svc._clock(svc._rebuild_state(_req(picks=one)))
    assert clock1.current_pick == 1 and clock1.picking_team_index == 1 and clock1.is_user_turn is False
    # After a full round of 12 picks → round 2, snake reverses → team 11 on the clock.
    twelve = [
        DraftPick(pick=i, team_index=i, player_id=100 + i, player_name=f"P{i}", positions="OF") for i in range(12)
    ]
    clock2 = svc._clock(svc._rebuild_state(_req(picks=twelve)))
    assert clock2.current_pick == 12 and clock2.round == 2 and clock2.picking_team_index == 11


def test_to_recs_maps_and_is_nan_safe():
    frame = pd.DataFrame(
        [
            {
                "player_id": 7,
                "player_name": "Star Hitter",
                "positions": "OF",
                "overall_rank": 1,
                "composite_value": 92.3,
                "mean_sgp": 5.1,
                "confidence": 0.77,
                "buy_fair_avoid": "BUY",
            },
            {
                "player_id": 9,
                "player_name": "Other Guy",
                "positions": "2B",
                "overall_rank": 2,
                "composite_value": float("nan"),  # missing → must degrade to 0.0, not NaN
                "mean_sgp": 3.0,
                "confidence": float("nan"),  # missing → None
                "buy_fair_avoid": None,
            },
        ]
    )
    recs = DraftService._to_recs(frame)
    assert len(recs) == 2
    assert recs[0].player.id == 7 and recs[0].rank == 1 and recs[0].score == 92.3
    assert recs[0].tag == "BUY" and recs[0].confidence == 0.77
    assert recs[1].score == 0.0  # NaN composite_value degraded
    assert recs[1].confidence is None  # NaN confidence → None
    assert recs[1].tag is None


def test_to_recs_empty_frame():
    assert DraftService._to_recs(pd.DataFrame()) == []
    assert DraftService._to_recs(None) == []


def test_recommend_full_path_with_injected_engine():
    frame = pd.DataFrame(
        [
            {
                "player_id": 7,
                "player_name": "Star",
                "positions": "OF",
                "overall_rank": 1,
                "composite_value": 90.0,
                "mean_sgp": 5.0,
            }
        ]
    )
    fake = _FakeEngine(frame)
    resp = DraftService().recommend(_req(top_n_req=5), engine=fake, pool=object())
    assert resp.clock.is_user_turn is True
    assert len(resp.recommendations) == 1 and resp.recommendations[0].player.id == 7
    assert "1 recommendation" in resp.summary
    assert fake.called_with["top_n"] == 5  # request top_n forwarded (within cap)


def test_recommend_is_graceful_when_engine_raises():
    class _BoomEngine:
        def recommend(self, player_pool, draft_state, top_n=8, n_simulations=300):
            raise RuntimeError("engine exploded")

    resp = DraftService().recommend(_req(), engine=_BoomEngine(), pool=object())
    assert resp.recommendations == []
    assert resp.clock.round == 1 and resp.clock.is_user_turn is True  # clock still computed
    assert "unavailable" in resp.summary.lower()


class _FakeDraftService:
    def recommend(self, req) -> DraftRecommendResponse:
        return DraftRecommendResponse(
            clock=DraftClock(current_pick=0, round=1, pick_in_round=1, picking_team_index=0, is_user_turn=True),
            recommendations=[
                DraftRecommendation(
                    player=PlayerRef(id=1, name="A. Player", positions="SS"),
                    rank=1,
                    score=88.0,
                    projected_sgp=4.0,
                )
            ],
            summary="1 recommendation for pick 1.",
        )


def test_post_draft_recommend_returns_contract():
    app = create_app()
    app.dependency_overrides[get_draft_service] = lambda: _FakeDraftService()
    client = TestClient(app)
    resp = client.post("/api/draft/recommend", json={"config": {"num_teams": 12, "user_team_index": 0}, "pick_log": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["clock"]["is_user_turn"] is True
    assert body["recommendations"][0]["player"]["name"] == "A. Player"


def test_post_draft_recommend_accepts_empty_body_defaults():
    app = create_app()
    app.dependency_overrides[get_draft_service] = lambda: _FakeDraftService()
    client = TestClient(app)
    resp = client.post("/api/draft/recommend", json={})  # all fields default
    assert resp.status_code == 200
    assert resp.json()["recommendations"][0]["rank"] == 1
