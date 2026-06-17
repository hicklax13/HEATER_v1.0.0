from starlette.testclient import TestClient

from api.contracts.my_team import CategoryLine, MatchupHero, MyTeamResponse
from api.deps import get_team_service
from api.main import create_app


class _FakeTeamService:
    def get_my_team(self, team_name: str) -> MyTeamResponse:
        return MyTeamResponse(
            team_name=team_name,
            record="4-7-1",
            rank=10,
            matchup=MatchupHero(opponent="Baty Babies", week=13, win_prob=0.46, tie_prob=0.19, loss_prob=0.35),
            categories=[CategoryLine(cat="SB", you=42.0, opp=55.0, edge=-13.0, win_prob=0.18)],
        )


def test_get_me_team_returns_contract():
    app = create_app()
    app.dependency_overrides[get_team_service] = lambda: _FakeTeamService()
    client = TestClient(app)
    resp = client.get("/api/me/team", params={"team_name": "Team Hickey"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["rank"] == 10
    assert body["matchup"]["opponent"] == "Baty Babies"
    assert body["categories"][0]["cat"] == "SB"


def test_my_team_contract_shape():
    resp = MyTeamResponse(
        team_name="Team Hickey",
        record="4-7-1",
        rank=10,
        matchup=MatchupHero(opponent="Baty Babies", week=13, win_prob=0.46, tie_prob=0.19, loss_prob=0.35),
        categories=[CategoryLine(cat="SB", you=42.0, opp=55.0, edge=-13.0, win_prob=0.18, inverse=False)],
    )
    # win/tie/loss must sum to ~1
    m = resp.matchup
    assert abs((m.win_prob + m.tie_prob + m.loss_prob) - 1.0) < 1e-6
    # round-trips to the JSON shape the frontend consumes
    dumped = resp.model_dump()
    assert dumped["matchup"]["win_prob"] == 0.46
    assert dumped["categories"][0]["cat"] == "SB"
