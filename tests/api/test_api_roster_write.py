from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.roster_write import (
    AddDropRequest,
    LineupAssignment,
    LineupSetRequest,
    MutationResult,
)
from api.deps import get_roster_write_service
from api.main import create_app
from api.services.roster_write_service import RosterWriteService


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


_UNSET = object()


class _FakeClient:
    """Stand-in for YahooFantasyClient — records calls, returns canned dicts."""

    def __init__(self, lineup_ret=_UNSET, addrop_ret=_UNSET):
        self._lineup_ret = lineup_ret if lineup_ret is not _UNSET else {"ok": True, "applied": 2}
        self._addrop_ret = addrop_ret if addrop_ret is not _UNSET else {"ok": True}
        self.last_assignments = None
        self.last_date = None
        self.last_add = None
        self.last_drop = None

    def set_lineup(self, assignments, coverage_date):
        self.last_assignments = assignments
        self.last_date = coverage_date
        return self._lineup_ret

    def add_drop(self, add_player_key, drop_player_key):
        self.last_add = add_player_key
        self.last_drop = drop_player_key
        return self._addrop_ret


def _lineup_req():
    return LineupSetRequest(
        team_name="Team Hickey",
        date="2027-04-05",
        assignments=[
            LineupAssignment(yahoo_player_key="469.p.1", slot="SS"),
            LineupAssignment(yahoo_player_key="469.p.2", slot="BN"),
        ],
    )


def test_service_set_lineup_maps_and_passes_through():
    fake = _FakeClient()
    result = RosterWriteService().set_lineup(_lineup_req(), client=fake)
    assert result.ok is True and result.applied == 2
    # the service translates {yahoo_player_key, slot} -> {player_key, position}
    assert fake.last_assignments == [
        {"player_key": "469.p.1", "position": "SS"},
        {"player_key": "469.p.2", "position": "BN"},
    ]
    assert fake.last_date == "2027-04-05"


def test_service_add_drop_passes_keys_through():
    fake = _FakeClient(addrop_ret={"ok": True})
    result = RosterWriteService().add_drop(
        AddDropRequest(add_player_key="469.p.9", drop_player_key="469.p.3"), client=fake
    )
    assert result.ok is True
    assert fake.last_add == "469.p.9" and fake.last_drop == "469.p.3"


def test_service_passes_through_write_scope_denial():
    fake = _FakeClient(lineup_ret={"ok": False, "error": "Yahoo write access denied — re-authorize.", "status": 403})
    result = RosterWriteService().set_lineup(_lineup_req(), client=fake)
    assert result.ok is False and result.status == 403
    assert "re-authorize" in (result.error or "")


def test_service_no_client_is_graceful(monkeypatch):
    svc = RosterWriteService()
    monkeypatch.setattr(svc, "_client", lambda: None)
    result = svc.set_lineup(_lineup_req())
    assert result.ok is False and result.status is None
    assert "Not connected" in (result.error or "")


def test_service_non_dict_response_is_graceful():
    fake = _FakeClient(lineup_ret=None)  # client returned something unexpected
    result = RosterWriteService().set_lineup(_lineup_req(), client=fake)
    assert result.ok is False and "Unexpected" in (result.error or "")


class _FakeWriteService:
    def set_lineup(self, req) -> MutationResult:
        return MutationResult(ok=True, applied=len(req.assignments))

    def add_drop(self, req) -> MutationResult:
        if not req.add_player_key and not req.drop_player_key:
            return MutationResult(
                ok=False,
                error="Must provide at least one of add_player_key or drop_player_key.",
                status=None,
            )
        return MutationResult(ok=True)


def _client_with_fake():
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _FakeWriteService()
    return TestClient(app)


def test_post_lineup_set_returns_contract():
    client = _client_with_fake()
    resp = client.post(
        "/api/lineup/set",
        json={
            "team_name": "Team Hickey",
            "date": "2027-04-05",
            "assignments": [
                {"yahoo_player_key": "469.p.1", "slot": "SS"},
                {"yahoo_player_key": "469.p.2", "slot": "BN"},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True and body["applied"] == 2


def test_post_add_drop_returns_contract():
    client = _client_with_fake()
    resp = client.post(
        "/api/transactions/add-drop",
        json={"add_player_key": "469.p.9", "drop_player_key": "469.p.3"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_post_add_drop_both_null_is_ok_false_not_http_error():
    client = _client_with_fake()
    resp = client.post("/api/transactions/add-drop", json={})
    assert resp.status_code == 200  # graceful: failure is in the body, not the HTTP status
    assert resp.json()["ok"] is False
