from api.contracts.common import PlayerRef
from api.contracts.draft import (
    DraftClock,
    DraftConfig,
    DraftPick,
    DraftRecommendation,
    DraftRecommendRequest,
    DraftRecommendResponse,
)


def test_request_defaults():
    req = DraftRecommendRequest()
    assert req.config.num_teams == 12
    assert req.config.num_rounds == 23
    assert req.config.user_team_index == 0
    assert req.config.roster_config is None
    assert req.pick_log == []
    assert req.top_n == 8
    assert req.n_simulations == 300


def test_request_with_picks():
    req = DraftRecommendRequest(
        config=DraftConfig(num_teams=12, user_team_index=3),
        pick_log=[DraftPick(pick=0, team_index=0, player_id=1, player_name="A", positions="SS")],
        top_n=5,
    )
    dumped = req.model_dump()
    assert dumped["config"]["user_team_index"] == 3
    assert dumped["pick_log"][0]["player_id"] == 1
    assert dumped["top_n"] == 5


def test_response_shape():
    resp = DraftRecommendResponse(
        clock=DraftClock(current_pick=0, round=1, pick_in_round=1, picking_team_index=0, is_user_turn=True),
        recommendations=[
            DraftRecommendation(
                player=PlayerRef(id=1, name="A. Player", positions="SS"),
                rank=1,
                score=87.5,
                projected_sgp=4.2,
                confidence=0.8,
                tag="BUY",
            )
        ],
        summary="1 recommendation",
    )
    dumped = resp.model_dump()
    assert dumped["clock"]["is_user_turn"] is True
    assert dumped["recommendations"][0]["player"]["name"] == "A. Player"
    assert dumped["recommendations"][0]["reason"] == ""
