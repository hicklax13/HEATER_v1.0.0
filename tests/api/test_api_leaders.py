from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.leaders import LeaderRow, LeadersResponse
from api.deps import get_leaders_service
from api.main import create_app


def test_leaders_contract_shape():
    resp = LeadersResponse(
        category="HR",
        rows=[
            LeaderRow(
                rank=1,
                player=PlayerRef(id=1, name="Aaron Judge", positions="OF"),
                value=35.0,
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["category"] == "HR"
    assert dumped["rows"][0]["player"]["name"] == "Aaron Judge"
    assert dumped["rows"][0]["rank"] == 1
    assert dumped["rows"][0]["value"] == 35.0
    # empty rows
    assert LeadersResponse(category="SB", rows=[]).rows == []


class _FakeLeadersService:
    def get_leaders(self, category: str, limit: int = 25) -> LeadersResponse:
        return LeadersResponse(
            category=category,
            rows=[
                LeaderRow(
                    rank=1,
                    player=PlayerRef(id=1, name="Aaron Judge", positions="OF"),
                    value=35.0,
                )
            ],
        )


def test_get_leaders_returns_contract():
    app = create_app()
    app.dependency_overrides[get_leaders_service] = lambda: _FakeLeadersService()
    client = TestClient(app)
    resp = client.get("/api/leaders", params={"category": "HR", "limit": 25})
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "HR"
    assert body["rows"][0]["player"]["name"] == "Aaron Judge"
