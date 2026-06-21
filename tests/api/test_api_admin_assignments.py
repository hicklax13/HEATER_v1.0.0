"""Admin assignment endpoint — gate (env allowlist, deny-by-default) + assign/list,
all DB-free via in-memory stores + a faked team-names provider."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.admin import get_team_names_provider
from api.auth import Principal, get_auth_verifier
from api.deps import get_league_store, get_membership_store, get_user_store
from api.routers.admin import router as admin_router
from api.stores.league_store import InMemoryLeagueStore
from api.stores.membership_store import InMemoryMembershipStore
from api.stores.user_store import InMemoryUserStore


class _Verifier:
    def __init__(self, clerk_id):
        self._id = clerk_id

    def verify(self, authorization):
        return Principal(subject=self._id, clerk_user_id=self._id)


def _client(clerk_id, monkeypatch, names=("🏆 Team Hickey", "Bronx Bombers")):
    monkeypatch.setenv("HEATER_ADMIN_CLERK_IDS", "admin_1,admin_2")
    # Shared store instances so a POST and a later GET in the same test see one state.
    users, leagues, members = InMemoryUserStore(), InMemoryLeagueStore(), InMemoryMembershipStore()
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_auth_verifier] = lambda: _Verifier(clerk_id)
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: leagues
    app.dependency_overrides[get_membership_store] = lambda: members
    app.dependency_overrides[get_team_names_provider] = lambda: lambda: list(names)
    return TestClient(app)


def test_non_admin_clerk_user_forbidden(monkeypatch):
    c = _client("not_admin", monkeypatch)
    r = c.post(
        "/api/admin/assignments",
        json={"clerk_user_id": "u1", "team_name": "Team Hickey"},
        headers={"Authorization": "Bearer x"},
    )
    assert r.status_code == 403


def test_env_token_path_forbidden(monkeypatch):
    # A caller with no clerk_user_id (env-token/server path) is never an admin.
    monkeypatch.setenv("HEATER_ADMIN_CLERK_IDS", "admin_1")
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_auth_verifier] = lambda: type(
        "V", (), {"verify": lambda self, a: Principal(subject="api-token")}
    )()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    app.dependency_overrides[get_team_names_provider] = lambda: lambda: []
    r = TestClient(app).post(
        "/api/admin/assignments",
        json={"clerk_user_id": "u1", "team_name": "X"},
        headers={"Authorization": "Bearer x"},
    )
    assert r.status_code == 403


def test_admin_assigns_and_canonicalizes_team_name(monkeypatch):
    c = _client("admin_1", monkeypatch)
    r = c.post(
        "/api/admin/assignments",
        json={"clerk_user_id": "member_a", "team_name": "Team Hickey"},
        headers={"Authorization": "Bearer x"},
    )
    assert r.status_code == 200
    assert r.json()["team_name"] == "🏆 Team Hickey"  # reconciled to the exact roster name
    assert r.json()["validated"] is True  # matched against a non-empty roster list


def test_admin_assign_cold_db_persists_unvalidated(monkeypatch):
    # Roster source cold/empty → name accepted as-is (cold-start seeding), but the
    # response flags validated=False so the unvalidated write is observable.
    c = _client("admin_1", monkeypatch, names=())
    r = c.post(
        "/api/admin/assignments",
        json={"clerk_user_id": "member_a", "team_name": "Team Hickee"},  # typo, can't be caught cold
        headers={"Authorization": "Bearer x"},
    )
    assert r.status_code == 200
    assert r.json()["team_name"] == "Team Hickee"
    assert r.json()["validated"] is False


def test_admin_assign_unknown_team_is_422(monkeypatch):
    c = _client("admin_1", monkeypatch)
    r = c.post(
        "/api/admin/assignments",
        json={"clerk_user_id": "member_a", "team_name": "Ghost Team"},
        headers={"Authorization": "Bearer x"},
    )
    assert r.status_code == 422
    assert "Bronx Bombers" in r.json()["detail"]  # candidates surfaced


def test_admin_list_assignments(monkeypatch):
    c = _client("admin_1", monkeypatch)
    c.post(
        "/api/admin/assignments",
        json={"clerk_user_id": "member_a", "team_name": "Bronx Bombers"},
        headers={"Authorization": "Bearer x"},
    )
    r = c.get("/api/admin/assignments", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    body = r.json()
    assert any(a["team_name"] == "Bronx Bombers" for a in body["assignments"])
    assert "Bronx Bombers" in body["available_teams"]


def test_empty_allowlist_denies_everyone(monkeypatch):
    monkeypatch.delenv("HEATER_ADMIN_CLERK_IDS", raising=False)
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[get_auth_verifier] = lambda: _Verifier("anyone")
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    app.dependency_overrides[get_team_names_provider] = lambda: lambda: []
    r = TestClient(app).get("/api/admin/assignments", headers={"Authorization": "Bearer x"})
    assert r.status_code == 403
