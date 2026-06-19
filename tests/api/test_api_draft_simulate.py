from starlette.testclient import TestClient

from api.contracts.draft import (
    DraftClock,
    DraftConfig,
    DraftPick,
    DraftSimulatePicksRequest,
    DraftSimulatePicksResponse,
)
from api.deps import get_draft_service
from api.main import create_app
from api.services.draft_service import DraftService


def test_simulate_request_defaults():
    req = DraftSimulatePicksRequest()
    assert req.config.num_teams == 12
    assert req.config.user_team_index == 0
    assert req.pick_log == []
    assert req.seed is None


def test_simulate_response_shape():
    resp = DraftSimulatePicksResponse(
        clock=DraftClock(current_pick=2, round=1, pick_in_round=3, picking_team_index=2, is_user_turn=True),
        picks=[DraftPick(pick=0, team_index=0, player_id=101, player_name="A", positions="SS")],
        summary="1 opponent pick simulated.",
    )
    dumped = resp.model_dump()
    assert dumped["clock"]["is_user_turn"] is True
    assert dumped["picks"][0]["player_id"] == 101
    assert dumped["summary"].startswith("1 opponent")
