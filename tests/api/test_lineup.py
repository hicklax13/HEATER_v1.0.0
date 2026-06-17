from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.lineup import LineupOptimizeResponse, LineupSlot
from api.deps import get_lineup_service
from api.main import create_app


def test_lineup_contract_shape():
    resp = LineupOptimizeResponse(
        team_name="Team Hickey",
        date="2027-04-05",
        slots=[
            LineupSlot(
                slot="OF",
                player=PlayerRef(id=1, name="A. Player", positions="OF"),
                action="START",
                projected=4.2,
                forced_start=False,
                reason=None,
            )
        ],
        summary="9 starters set; 0 forced.",
    )
    dumped = resp.model_dump()
    assert dumped["slots"][0]["action"] == "START"
    assert dumped["slots"][0]["player"]["name"] == "A. Player"


class _FakeLineupService:
    def optimize(self, team_name: str, date=None, scope: str = "rest_of_season") -> LineupOptimizeResponse:
        return LineupOptimizeResponse(
            team_name=team_name,
            date=date or "2027-04-05",
            slots=[
                LineupSlot(
                    slot="OF",
                    player=PlayerRef(id=1, name="A. Player", positions="OF"),
                    action="START",
                    projected=4.2,
                )
            ],
            summary="1 starter",
        )


def test_post_lineup_optimize_returns_contract():
    app = create_app()
    app.dependency_overrides[get_lineup_service] = lambda: _FakeLineupService()
    client = TestClient(app)
    resp = client.post("/api/lineup/optimize", json={"team_name": "Team Hickey", "date": "2027-04-05"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["slots"][0]["action"] == "START"
