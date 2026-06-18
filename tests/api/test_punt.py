from starlette.testclient import TestClient

from api.contracts.punt import PuntCategory, PuntResponse
from api.deps import get_punt_service
from api.main import create_app


def test_punt_contract_shape():
    resp = PuntResponse(
        team_name="Team Hickey",
        punt_candidates=["ERA", "WHIP"],
        categories=[
            PuntCategory(cat="ERA", current_rank=11, gainable=False, recommendation="Punt"),
            PuntCategory(cat="HR", current_rank=3, gainable=True, recommendation="Contend"),
        ],
    )
    dumped = resp.model_dump()
    assert dumped["team_name"] == "Team Hickey"
    assert "ERA" in dumped["punt_candidates"]
    assert dumped["categories"][0]["gainable"] is False
    assert dumped["categories"][1]["recommendation"] == "Contend"
    # defaults
    assert PuntResponse(team_name="X").punt_candidates == []
    assert PuntResponse(team_name="X").categories == []
    assert PuntCategory(cat="SB", current_rank=5, gainable=True).recommendation == ""


class _FakePuntService:
    def get_punt(self, team_name: str) -> PuntResponse:
        return PuntResponse(
            team_name=team_name,
            punt_candidates=["ERA"],
            categories=[
                PuntCategory(cat="ERA", current_rank=11, gainable=False, recommendation="Punt"),
            ],
        )


def test_get_punt_returns_contract():
    app = create_app()
    app.dependency_overrides[get_punt_service] = lambda: _FakePuntService()
    client = TestClient(app)
    resp = client.get("/api/punt?team_name=Team+Hickey")
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["punt_candidates"] == ["ERA"]
    assert body["categories"][0]["cat"] == "ERA"
