from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.databank import DatabankResponse, SeasonStat
from api.deps import get_databank_service
from api.main import create_app


def test_databank_contract_shape():
    resp = DatabankResponse(
        player=PlayerRef(id=42, name="Mike Trout", positions="OF"),
        seasons=[
            SeasonStat(year=2025, stats={"HR": 28.0, "AVG": 0.310}),
            SeasonStat(year=2024, stats={"HR": 22.0, "AVG": 0.285}),
        ],
    )
    dumped = resp.model_dump()
    assert dumped["player"]["name"] == "Mike Trout"
    assert dumped["seasons"][0]["year"] == 2025
    assert dumped["seasons"][1]["stats"]["HR"] == 22.0
    # defaults
    assert SeasonStat(year=2026).stats == {}
    assert DatabankResponse(player=PlayerRef(id=1, name="X", positions="")).seasons == []


class _FakeDatabankService:
    def get_player(self, player_id: int) -> DatabankResponse:
        return DatabankResponse(
            player=PlayerRef(id=player_id, name="Test Player", positions="SP"),
            seasons=[
                SeasonStat(year=2025, stats={"W": 15.0, "ERA": 3.10}),
            ],
        )


def test_get_databank_returns_contract():
    app = create_app()
    app.dependency_overrides[get_databank_service] = lambda: _FakeDatabankService()
    client = TestClient(app)
    resp = client.get("/api/databank?player_id=99")
    assert resp.status_code == 200
    body = resp.json()
    assert body["player"]["id"] == 99
    assert body["player"]["name"] == "Test Player"
    assert body["seasons"][0]["year"] == 2025
    assert body["seasons"][0]["stats"]["ERA"] == 3.10
