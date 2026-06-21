"""The personalized endpoints resolve the viewer's team from identity, falling back
to the query param when dormant. Targets the /free-agents/pool route with a spy
service (FreeAgentPoolResponse is no-arg constructible) to assert which team name
the router forwarded — DB-free. (The resolver itself is proven in
test_api_tenancy_resolver.py; this proves the real routers wire it.)"""

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.contracts.free_agents import FreeAgentPoolResponse
from api.deps import get_fa_pool_service, get_league_store, get_membership_store, get_user_store
from api.routers.free_agents import router as fa_router
from api.stores.league_store import InMemoryLeagueStore
from api.stores.membership_store import InMemoryMembershipStore
from api.stores.user_store import InMemoryUserStore


class _SpyPoolService:
    def __init__(self):
        self.seen = None

    def get_free_agents_pool(self, team_name, limit=100):
        self.seen = team_name  # capture what the router forwarded
        return FreeAgentPoolResponse()


class _Clerk:
    def verify(self, authorization):
        return Principal(subject="user_42", clerk_user_id="user_42")


def _app(spy):
    app = FastAPI()
    app.include_router(fa_router)
    app.dependency_overrides[get_fa_pool_service] = lambda: spy
    return app


def test_pool_endpoint_dormant_uses_query_param():
    spy = _SpyPoolService()
    app = _app(spy)
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    r = TestClient(app).get("/api/free-agents/pool?team_name=Team%20Hickey")
    assert r.status_code == 200
    assert spy.seen == "Team Hickey"  # unchanged behavior when off


def test_pool_endpoint_resolves_assigned_team_for_clerk_user():
    spy = _SpyPoolService()
    app = _app(spy)
    users, leagues, members = InMemoryUserStore(), InMemoryLeagueStore(), InMemoryMembershipStore()
    u = users.get_or_create("user_42")
    lg = leagues.get_or_create_default()
    members.assign(u.id, lg.id, "Bronx Bombers", team_key=None, assigned_by=u.id)
    app.dependency_overrides[get_auth_verifier] = lambda: _Clerk()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: leagues
    app.dependency_overrides[get_membership_store] = lambda: members
    r = TestClient(app).get("/api/free-agents/pool?team_name=Team%20Hickey", headers={"Authorization": "Bearer x"})
    assert r.status_code == 200
    assert spy.seen == "Bronx Bombers"  # resolved overrides the param
