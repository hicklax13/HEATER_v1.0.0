from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.free_agents import FreeAgentRec, FreeAgentsResponse
from api.deps import get_fa_service
from api.main import create_app


def test_free_agents_contract_shape():
    resp = FreeAgentsResponse(
        team_name="Team Hickey",
        recommendations=[
            FreeAgentRec(
                add=PlayerRef(id=1, name="A. Player", positions="OF"),
                drop=PlayerRef(id=2, name="B. Bench", positions="OF"),
                marginal_value=2.31,
                categories_helped=["SB", "R"],
                ownership_pct=44.0,
                rationale="Adds steals you're behind in.",
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["recommendations"][0]["add"]["name"] == "A. Player"
    assert dumped["recommendations"][0]["categories_helped"] == ["SB", "R"]
    # drop is optional (roster-grow adds have none)
    assert FreeAgentRec(add=PlayerRef(id=3, name="C", positions="SP"), marginal_value=1.0).drop is None


class _FakeFAService:
    def get_free_agents(self, team_name: str, limit: int = 5) -> FreeAgentsResponse:
        return FreeAgentsResponse(
            team_name=team_name,
            recommendations=[
                FreeAgentRec(
                    add=PlayerRef(id=1, name="A. Player", positions="OF"),
                    drop=PlayerRef(id=2, name="B. Bench", positions="OF"),
                    marginal_value=2.31,
                    categories_helped=["SB"],
                    ownership_pct=44.0,
                    rationale="steals",
                )
            ],
        )


def test_get_free_agents_returns_contract():
    app = create_app()
    app.dependency_overrides[get_fa_service] = lambda: _FakeFAService()
    client = TestClient(app)
    resp = client.get("/api/free-agents", params={"team_name": "Team Hickey", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["recommendations"][0]["add"]["name"] == "A. Player"
