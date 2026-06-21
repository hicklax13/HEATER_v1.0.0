"""Provisioning tests — provision_app_user (pure, DB-free via in-memory store) +
require_app_user wired into a throwaway FastAPI route (proves the dependency
resolves auth → store → AppUser, and is dormant on the env-token path)."""

from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.deps import get_user_store
from api.identity import provision_app_user, require_app_user
from api.stores.user_store import InMemoryUserStore


def test_provision_returns_none_on_env_token_path():
    # A Principal without clerk_user_id (env-token/server path) provisions nothing.
    assert provision_app_user(Principal(subject="api-token"), InMemoryUserStore()) is None


def test_provision_get_or_creates_for_clerk_principal():
    store = InMemoryUserStore()
    p = Principal(subject="user_9", clerk_user_id="user_9")
    u1 = provision_app_user(p, store)
    u2 = provision_app_user(p, store)
    assert u1 is not None
    assert u1.id == u2.id  # idempotent
    assert u1.clerk_user_id == "user_9"


class _ClerkPrincipalVerifier:
    def verify(self, authorization):
        return Principal(subject="user_42", clerk_user_id="user_42")


class _EnvStyleVerifier:
    def verify(self, authorization):
        return Principal(subject="api-token")


def _app_with_protected_route():
    app = FastAPI()

    @app.get("/whoami")
    def whoami(user=Depends(require_app_user)):
        return {"id": user.id, "clerk_user_id": user.clerk_user_id} if user else {"id": None}

    return app


def test_require_app_user_provisions_via_dependency():
    app = _app_with_protected_route()
    store = InMemoryUserStore()
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkPrincipalVerifier()
    app.dependency_overrides[get_user_store] = lambda: store
    body = TestClient(app).get("/whoami", headers={"Authorization": "Bearer anything"}).json()
    assert body["clerk_user_id"] == "user_42"
    assert body["id"] == 1


def test_require_app_user_none_on_env_path():
    app = _app_with_protected_route()
    app.dependency_overrides[get_auth_verifier] = lambda: _EnvStyleVerifier()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    body = TestClient(app).get("/whoami", headers={"Authorization": "Bearer x"}).json()
    assert body["id"] is None
