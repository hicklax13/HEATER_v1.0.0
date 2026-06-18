from starlette.testclient import TestClient

from api.contracts.matchup import MatchupCategory, MatchupResponse
from api.deps import get_matchup_service
from api.main import create_app


def test_matchup_contract_shape():
    resp = MatchupResponse(
        team_name="Team Hickey",
        opponent="Team Two",
        week=12,
        projected_cat_wins=7.5,
        win_prob=0.62,
        categories=[
            MatchupCategory(cat="HR", you=45.0, opp=38.0, win_prob=0.72, inverse=False),
            MatchupCategory(cat="ERA", you=3.25, opp=3.80, win_prob=0.65, inverse=True),
        ],
    )
    dumped = resp.model_dump()
    assert dumped["team_name"] == "Team Hickey"
    assert dumped["categories"][0]["cat"] == "HR"
    assert dumped["categories"][1]["inverse"] is True
    # defaults
    assert MatchupResponse(team_name="X").week == 0
    assert MatchupResponse(team_name="X").categories == []


class _FakeMatchupService:
    def get_matchup(self, team_name: str) -> MatchupResponse:
        return MatchupResponse(
            team_name=team_name,
            opponent="Opponent",
            week=5,
            projected_cat_wins=6.0,
            win_prob=0.55,
            categories=[
                MatchupCategory(cat="HR", you=40.0, opp=35.0, win_prob=0.65, inverse=False),
            ],
        )


def test_get_matchup_returns_contract():
    app = create_app()
    app.dependency_overrides[get_matchup_service] = lambda: _FakeMatchupService()
    client = TestClient(app)
    resp = client.get("/api/matchup?team_name=Team+Hickey")
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["categories"][0]["cat"] == "HR"
    assert body["week"] == 5
