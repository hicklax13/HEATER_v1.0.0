from api.contracts.common import PlayerRef
from api.contracts.roster_write import (
    AddDropRequest,
    LineupAssignment,
    LineupSetRequest,
    MutationResult,
)


def test_player_ref_yahoo_key_is_optional_and_defaults_none():
    p = PlayerRef(id=1, name="A. Player", positions="OF")
    assert p.yahoo_player_key is None
    p2 = PlayerRef(id=1, name="A. Player", positions="OF", yahoo_player_key="469.p.1")
    assert p2.model_dump()["yahoo_player_key"] == "469.p.1"


def test_lineup_set_request_shape():
    req = LineupSetRequest(
        team_name="Team Hickey",
        date="2027-04-05",
        assignments=[
            LineupAssignment(yahoo_player_key="469.p.1", slot="SS", player_id=1),
            LineupAssignment(yahoo_player_key="469.p.2", slot="BN"),
        ],
    )
    dumped = req.model_dump()
    assert dumped["assignments"][0]["yahoo_player_key"] == "469.p.1"
    assert dumped["assignments"][1]["player_id"] is None


def test_add_drop_request_allows_partial():
    assert AddDropRequest(add_player_key="469.p.9").drop_player_key is None
    assert AddDropRequest(drop_player_key="469.p.3").add_player_key is None
    assert AddDropRequest().add_player_key is None  # both-null is a valid shape; service returns ok=False


def test_mutation_result_shape():
    ok = MutationResult(ok=True, applied=2)
    assert ok.model_dump() == {"ok": True, "applied": 2, "error": None, "status": None}
    fail = MutationResult(ok=False, error="denied", status=403)
    assert fail.ok is False and fail.status == 403
