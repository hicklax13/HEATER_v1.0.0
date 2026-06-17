from starlette.testclient import TestClient

from api.contracts.standings import StandingsResponse, TeamStanding
from api.deps import get_standings_service
from api.main import create_app


def test_standings_contract_shape():
    resp = StandingsResponse(
        teams=[
            TeamStanding(
                rank=1,
                team_name="Team Hickey",
                wins=8,
                losses=4,
                ties=0,
                points=62.5,
                category_ranks={"HR": 1, "SB": 3},
            )
        ]
    )
    dumped = resp.model_dump()
    assert dumped["teams"][0]["team_name"] == "Team Hickey"
    assert dumped["teams"][0]["category_ranks"]["HR"] == 1
    # defaults
    assert TeamStanding(rank=2, team_name="Other").wins == 0
    assert TeamStanding(rank=2, team_name="Other").category_ranks == {}


class _FakeStandingsService:
    def get_standings(self) -> StandingsResponse:
        return StandingsResponse(
            teams=[
                TeamStanding(
                    rank=1,
                    team_name="Team Hickey",
                    wins=8,
                    losses=4,
                    ties=0,
                    points=62.5,
                    category_ranks={"HR": 1},
                )
            ]
        )


def test_get_standings_returns_contract():
    app = create_app()
    app.dependency_overrides[get_standings_service] = lambda: _FakeStandingsService()
    client = TestClient(app)
    resp = client.get("/api/standings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["teams"][0]["team_name"] == "Team Hickey"
    assert body["teams"][0]["rank"] == 1
