"""A team-required router returns 409 team_not_linked for an authed-unassigned
viewer (HIGH-1), and 200 for an assigned one. Uses a tiny app whose route mirrors
the real routers' `resolve_required_team(ctx, team_name)` call."""

from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.deps import get_league_store, get_membership_store, get_user_store
from api.stores.league_store import InMemoryLeagueStore
from api.stores.membership_store import InMemoryMembershipStore
from api.stores.user_store import InMemoryUserStore
from api.tenancy import ViewerContext, require_viewer_context, resolve_required_team


class _Clerk:
    def verify(self, authorization):
        return Principal(subject="user_42", clerk_user_id="user_42")


def _app():
    app = FastAPI()

    @app.get("/needs-team")
    def needs_team(team_name: str = "", ctx: ViewerContext = Depends(require_viewer_context)):
        return {"team": resolve_required_team(ctx, team_name)}

    return app


def test_authed_unassigned_gets_409():
    app = _app()
    app.dependency_overrides[get_auth_verifier] = lambda: _Clerk()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    r = TestClient(app).get("/needs-team?team_name=Team%20Hickey", headers={"Authorization": "Bearer x"})
    assert r.status_code == 409
    assert r.json()["detail"] == "team_not_linked"


def test_authed_assigned_gets_their_team():
    app = _app()
    users, leagues, members = InMemoryUserStore(), InMemoryLeagueStore(), InMemoryMembershipStore()
    u = users.get_or_create("user_42")
    lg = leagues.get_or_create_default()
    members.assign(u.id, lg.id, "Bronx Bombers", team_key=None, assigned_by=u.id)
    app.dependency_overrides[get_auth_verifier] = lambda: _Clerk()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: leagues
    app.dependency_overrides[get_membership_store] = lambda: members
    r = TestClient(app).get("/needs-team?team_name=Team%20Hickey", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert r.json()["team"] == "Bronx Bombers"


def test_dormant_passes_param_through():
    app = _app()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    r = TestClient(app).get("/needs-team?team_name=Team%20Hickey")
    assert r.status_code == 200
    assert r.json()["team"] == "Team Hickey"
