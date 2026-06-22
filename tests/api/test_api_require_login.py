"""require_login — the auth-only gate for league-wide PRIVATE reads.

Dormant when Clerk is off (CLERK_ISSUER unset): the read stays open, byte-for-byte
today's behavior. Once Clerk is live, an open read with no/invalid login is rejected
(401). These endpoints expose the league's private data (full rosters + manager
names, standings), so they must require a logged-in user."""

from __future__ import annotations

import pytest
from fastapi import Depends, FastAPI, HTTPException, status
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier, require_login


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/private", dependencies=[Depends(require_login)])
    def private():
        return {"ok": True}

    return app


class _FakeVerifier:
    """Mimics ClerkVerifier.verify: 401 on a missing/blank token, else a Principal."""

    def verify(self, authorization):
        if not authorization:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="no token")
        return Principal(subject="u1", clerk_user_id="u1")


def test_dormant_open_when_clerk_off(monkeypatch):
    # Clerk unset → the gate is a no-op → the read is open (no token needed).
    monkeypatch.setattr("api.auth.clerk_configured", lambda: False)
    r = TestClient(_app()).get("/private")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_401_when_clerk_on_and_no_token(monkeypatch):
    monkeypatch.setattr("api.auth.clerk_configured", lambda: True)
    app = _app()
    app.dependency_overrides[get_auth_verifier] = lambda: _FakeVerifier()
    r = TestClient(app).get("/private")  # no Authorization header
    assert r.status_code == 401


def test_401_when_clerk_on_and_invalid_token(monkeypatch):
    monkeypatch.setattr("api.auth.clerk_configured", lambda: True)
    app = _app()

    class _Reject:
        def verify(self, authorization):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="bad")

    app.dependency_overrides[get_auth_verifier] = lambda: _Reject()
    r = TestClient(app).get("/private", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_200_when_clerk_on_and_valid_token(monkeypatch):
    monkeypatch.setattr("api.auth.clerk_configured", lambda: True)
    app = _app()
    app.dependency_overrides[get_auth_verifier] = lambda: _FakeVerifier()
    r = TestClient(app).get("/private", headers={"Authorization": "Bearer good"})
    assert r.status_code == 200


def test_require_login_adds_no_openapi_parameters(monkeypatch):
    """The gate uses Request (not a declared Header param), so it must add NO
    Authorization parameter to the route schema — keeps api/openapi.json untouched."""
    monkeypatch.setattr("api.auth.clerk_configured", lambda: False)
    schema = _app().openapi()
    params = schema["paths"]["/private"]["get"].get("parameters", [])
    assert not any(p.get("name", "").lower() == "authorization" for p in params)


# ── all 7 private-league read routers must carry the gate ────────────────────
def test_all_seven_private_routers_carry_require_login():
    """Structural guard: the league-wide private read routers gate at the router
    level, so a NEW route added to any of them is auto-gated. Catches an
    accidentally-ungated new private endpoint (the leak this fix closes)."""
    from api.routers import closers, compare, databank, leaders, players, standings, streaming

    routers = {
        "standings": standings.router,
        "closers": closers.router,
        "compare": compare.router,
        "databank": databank.router,
        "leaders": leaders.router,
        "players": players.router,
        "streaming": streaming.router,
    }
    for name, r in routers.items():
        deps = [d.dependency for d in (r.dependencies or [])]
        assert require_login in deps, f"{name} router is missing the require_login gate"


def test_private_reads_401_when_clerk_on(monkeypatch):
    """Live activation: with Clerk configured + no token, every private read 401s
    (the gate short-circuits before the service runs — DB-free)."""
    from api.main import create_app

    monkeypatch.setattr("api.auth.clerk_configured", lambda: True)
    app = create_app()
    app.dependency_overrides[get_auth_verifier] = lambda: _FakeVerifier()
    client = TestClient(app)
    gets = [
        "/api/standings",
        "/api/closers",
        "/api/compare?ids=1",
        "/api/databank?player_id=1",
        "/api/leaders",
        "/api/leaders/overall",
        "/api/players/search?q=trout",
        "/api/league/rosters",
        "/api/streaming",
    ]
    for path in gets:
        assert client.get(path).status_code == 401, f"GET {path} is not gated"
    assert client.post("/api/streaming/analyze", json={"pitcher_id": 1}).status_code == 401
