"""Tenancy helpers + resolver tests. This file covers the pure helpers and the
ViewerContext resolver (incl. the dormant open-read fallback)."""

from fastapi import Depends, FastAPI
from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.deps import get_league_store, get_membership_store, get_user_store
from api.stores.league_store import InMemoryLeagueStore
from api.stores.membership_store import InMemoryMembershipStore
from api.stores.user_store import InMemoryUserStore
from api.tenancy import (
    ViewerContext,
    normalize_team_name,
    reconcile_team_name,
    require_viewer_context,
)


def test_normalize_strips_emoji_whitespace_punctuation():
    # Same semantics as src.auth._normalize_team_name (replicated, not imported).
    assert normalize_team_name("🏆 Team Hickey") == normalize_team_name("Team Hickey")
    assert normalize_team_name("Team Hickey") == "teamhickey"
    assert normalize_team_name("  A.B-C  ") == "abc"


def test_reconcile_exact_match_returns_canonical_roster_name():
    names = ["🏆 Team Hickey", "Bronx Bombers"]
    assert reconcile_team_name("Team Hickey", names) == "🏆 Team Hickey"


def test_reconcile_exact_string_short_circuits():
    names = ["Team Hickey", "Other"]
    assert reconcile_team_name("Team Hickey", names) == "Team Hickey"


def test_reconcile_no_match_with_known_names_returns_none():
    assert reconcile_team_name("Nonexistent", ["A", "B"]) is None


def test_reconcile_empty_names_returns_assigned_as_is():
    # Cold/empty roster source must NOT block assignment (graceful).
    assert reconcile_team_name("Team Hickey", []) == "Team Hickey"


def test_viewer_context_effective_team_prefers_resolved():
    assert ViewerContext(user_id=1, league_id=1, team_name="Mine").effective_team("fallback") == "Mine"


def test_viewer_context_effective_team_falls_back_when_unresolved():
    assert ViewerContext(user_id=None, league_id=None, team_name=None).effective_team("Team Hickey") == "Team Hickey"
    assert ViewerContext(user_id=1, league_id=1, team_name=None).effective_team("Team Hickey") == "Team Hickey"


class _ClerkVerifier:
    def verify(self, authorization):
        return Principal(subject="user_42", clerk_user_id="user_42")


def _resolver_app():
    app = FastAPI()

    @app.get("/probe")
    def probe(team_name: str = "", ctx: ViewerContext = Depends(require_viewer_context)):
        return {"team": ctx.effective_team(team_name), "resolved": ctx.team_name}

    return app


def test_resolver_dormant_when_no_token_uses_query_param():
    # Clerk off / no Authorization header → reads stay OPEN, fall back to the param.
    app = _resolver_app()
    app.dependency_overrides[get_user_store] = lambda: InMemoryUserStore()
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    body = TestClient(app).get("/probe?team_name=Team%20Hickey").json()
    assert body == {"team": "Team Hickey", "resolved": None}


def test_resolver_overrides_with_assigned_team_for_clerk_user():
    app = _resolver_app()
    users, leagues, members = InMemoryUserStore(), InMemoryLeagueStore(), InMemoryMembershipStore()
    u = users.get_or_create("user_42")
    lg = leagues.get_or_create_default()
    members.assign(user_id=u.id, league_id=lg.id, team_name="Bronx Bombers", team_key=None, assigned_by=u.id)
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkVerifier()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: leagues
    app.dependency_overrides[get_membership_store] = lambda: members
    body = TestClient(app).get("/probe?team_name=Team%20Hickey", headers={"Authorization": "Bearer x"}).json()
    assert body["resolved"] == "Bronx Bombers"
    assert body["team"] == "Bronx Bombers"  # resolved wins over the query param


def test_resolver_clerk_user_without_assignment_resolves_none():
    # Logged-in but unassigned → team_name None (never another user's team); the
    # endpoint falls back to the param (open read) in M3-1.
    app = _resolver_app()
    users = InMemoryUserStore()
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkVerifier()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: InMemoryLeagueStore()
    app.dependency_overrides[get_membership_store] = lambda: InMemoryMembershipStore()
    body = TestClient(app).get("/probe?team_name=Fallback", headers={"Authorization": "Bearer x"}).json()
    assert body["resolved"] is None
    assert body["team"] == "Fallback"
