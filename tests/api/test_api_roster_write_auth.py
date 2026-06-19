import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from api.auth import EnvTokenVerifier, Principal, _bearer, get_auth_verifier
from api.contracts.roster_write import MutationResult
from api.deps import get_roster_write_service
from api.main import create_app

_ENV = "HEATER_API_WRITE_TOKEN"


# ---------------------------------------------------------------------------
# Verifier unit tests (Task 1)
# ---------------------------------------------------------------------------


def test_bearer_parsing():
    assert _bearer(None) is None
    assert _bearer("") is None
    assert _bearer("Bearer abc") == "abc"
    assert _bearer("bearer abc") == "abc"  # scheme is case-insensitive
    assert _bearer("Bearer   ") is None  # empty token after scheme
    assert _bearer("Token abc") is None  # wrong scheme
    assert _bearer("abc") is None  # no scheme


def test_verify_denies_by_default_when_secret_unset(monkeypatch):
    monkeypatch.delenv(_ENV, raising=False)
    with pytest.raises(HTTPException) as ei:
        EnvTokenVerifier().verify("Bearer anything")
    assert ei.value.status_code == 401
    assert "not configured" in ei.value.detail.lower()


def test_verify_rejects_missing_and_bad_token(monkeypatch):
    monkeypatch.setenv(_ENV, "s3cret")
    for header in (None, "", "Bearer", "Bearer wrong", "Token s3cret"):
        with pytest.raises(HTTPException) as ei:
            EnvTokenVerifier().verify(header)
        assert ei.value.status_code == 401


def test_verify_accepts_matching_token(monkeypatch):
    monkeypatch.setenv(_ENV, "s3cret")
    principal = EnvTokenVerifier().verify("Bearer s3cret")
    assert isinstance(principal, Principal)
    assert principal.subject == "api-token"


def test_verify_non_ascii_token_denies_not_crashes(monkeypatch):
    # A hostile non-ASCII bearer token must yield a clean 401, NOT a TypeError
    # (hmac.compare_digest rejects non-ASCII on the str/str path → would 500).
    monkeypatch.setenv(_ENV, "s3cret")
    with pytest.raises(HTTPException) as ei:
        EnvTokenVerifier().verify("Bearer s3crét")  # "s3crét"
    assert ei.value.status_code == 401


def test_verify_non_ascii_secret_does_not_crash(monkeypatch):
    # Defense for both operands: a non-ASCII configured secret must still
    # 401 a wrong token and accept the exact match — never raise TypeError.
    monkeypatch.setenv(_ENV, "sécret")  # non-ASCII secret
    with pytest.raises(HTTPException) as ei:
        EnvTokenVerifier().verify("Bearer wrong")
    assert ei.value.status_code == 401
    principal = EnvTokenVerifier().verify("Bearer sécret")
    assert principal.subject == "api-token"


# ---------------------------------------------------------------------------
# Endpoint auth tests (Task 2)
# ---------------------------------------------------------------------------


class _RecordingWriteService:
    def __init__(self):
        self.called = False

    def set_lineup(self, req) -> MutationResult:
        self.called = True
        return MutationResult(ok=True, applied=len(req.assignments))

    def add_drop(self, req) -> MutationResult:
        self.called = True
        return MutationResult(ok=True)


class _AlwaysVerifier:
    def verify(self, authorization):
        return Principal(subject="test")


_LINEUP_BODY = {
    "team_name": "Team Hickey",
    "date": "2027-04-05",
    "assignments": [{"yahoo_player_key": "469.p.1", "slot": "SS"}],
}


def test_write_denied_without_token_and_service_not_called(monkeypatch):
    monkeypatch.delenv("HEATER_API_WRITE_TOKEN", raising=False)
    app = create_app()
    fake = _RecordingWriteService()
    app.dependency_overrides[get_roster_write_service] = lambda: fake
    client = TestClient(app)
    resp = client.post("/api/lineup/set", json=_LINEUP_BODY)
    assert resp.status_code == 401
    assert fake.called is False  # the gate runs before the handler


def test_write_denied_with_bad_token(monkeypatch):
    monkeypatch.setenv("HEATER_API_WRITE_TOKEN", "s3cret")
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _RecordingWriteService()
    client = TestClient(app)
    resp = client.post("/api/lineup/set", json=_LINEUP_BODY, headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 401


def test_write_allowed_with_env_token(monkeypatch):
    monkeypatch.setenv("HEATER_API_WRITE_TOKEN", "s3cret")
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _RecordingWriteService()
    client = TestClient(app)
    resp = client.post("/api/lineup/set", json=_LINEUP_BODY, headers={"Authorization": "Bearer s3cret"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_write_allowed_via_verifier_override(monkeypatch):
    # The Clerk-at-B4 seam: overriding the verifier authenticates without env.
    monkeypatch.delenv("HEATER_API_WRITE_TOKEN", raising=False)
    app = create_app()
    app.dependency_overrides[get_roster_write_service] = lambda: _RecordingWriteService()
    app.dependency_overrides[get_auth_verifier] = lambda: _AlwaysVerifier()
    client = TestClient(app)
    resp = client.post("/api/transactions/add-drop", json={"add_player_key": "469.p.9"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_reads_are_not_auth_gated(monkeypatch):
    # The gate is writes-only: a read endpoint answers without a token
    # (any non-401 status proves it is not behind require_principal).
    monkeypatch.delenv("HEATER_API_WRITE_TOKEN", raising=False)
    client = TestClient(create_app())
    resp = client.get("/api/standings")
    assert resp.status_code != 401
