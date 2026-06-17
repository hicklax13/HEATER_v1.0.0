"""Tests for GET /api/free-agents endpoint (Slice 2, Task 1)."""

from starlette.testclient import TestClient

from api.contracts.free_agents import FreeAgentRec, FreeAgentsResponse
from api.deps import get_fa_service
from api.main import create_app


class _FakeFaService:
    def get_free_agents(self, team_name: str, max_moves: int) -> FreeAgentsResponse:
        return FreeAgentsResponse(
            team_name=team_name,
            adds_remaining=7,
            recommendations=[
                FreeAgentRec(
                    add_name="Luis Garcia",
                    add_positions="2B/SS",
                    drop_name="Joey Wiemer",
                    drop_positions="OF",
                    net_sgp_delta=0.82,
                    urgency_categories=["SB"],
                    reasoning="Upgrades SB; clean drop.",
                )
            ],
        )


def test_get_free_agents_returns_contract():
    app = create_app()
    app.dependency_overrides[get_fa_service] = lambda: _FakeFaService()
    client = TestClient(app)
    resp = client.get("/api/free-agents", params={"team_name": "Team Hickey"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["adds_remaining"] == 7
    assert len(body["recommendations"]) == 1
    rec = body["recommendations"][0]
    assert rec["add_name"] == "Luis Garcia"
    assert rec["drop_name"] == "Joey Wiemer"
    assert rec["net_sgp_delta"] == 0.82
    assert rec["urgency_categories"] == ["SB"]


def test_free_agents_contract_shape():
    resp = FreeAgentsResponse(
        team_name="Team Hickey",
        adds_remaining=7,
        recommendations=[
            FreeAgentRec(
                add_name="Luis Garcia",
                add_positions="2B/SS",
                drop_name="Joey Wiemer",
                drop_positions="OF",
                net_sgp_delta=0.82,
                urgency_categories=["SB"],
                reasoning="Upgrades SB.",
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["team_name"] == "Team Hickey"
    assert dumped["recommendations"][0]["add_name"] == "Luis Garcia"
    assert dumped["recommendations"][0]["net_sgp_delta"] == 0.82


def test_free_agents_empty_recommendations():
    """Engine returns empty list when no good moves are found."""
    resp = FreeAgentsResponse(
        team_name="Team Hickey",
        adds_remaining=0,
        recommendations=[],
    )
    assert resp.recommendations == []
    assert resp.adds_remaining == 0
