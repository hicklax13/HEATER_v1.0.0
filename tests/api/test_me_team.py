from api.contracts.my_team import CategoryLine, MatchupHero, MyTeamResponse


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
