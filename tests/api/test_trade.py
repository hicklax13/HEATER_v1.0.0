from starlette.testclient import TestClient

from api.auth import Principal, get_auth_verifier
from api.contracts.common import PlayerRef
from api.contracts.trade import (
    CategoryImpact,
    GradeRange,
    TradeEvaluateRequest,
    TradeEvaluationResponse,
)
from api.deps import get_league_store, get_membership_store, get_trade_service, get_user_store
from api.main import create_app
from api.stores.league_store import InMemoryLeagueStore
from api.stores.membership_store import InMemoryMembershipStore
from api.stores.user_store import InMemoryUserStore


def test_trade_contract_shape():
    resp = TradeEvaluationResponse(
        grade="B+",
        verdict="Accept",
        confidence_pct=72.5,
        surplus_sgp=0.85,
        grade_range=GradeRange(grade="B+", low=0.5, center=0.85, high=1.3),
        giving=[PlayerRef(id=1, name="Logan Webb", positions="SP")],
        receiving=[PlayerRef(id=2, name="Yordan Alvarez", positions="OF,1B")],
        category_impacts=[CategoryImpact(cat="HR", delta=0.4)],
        delta_playoff_prob=0.03,
        delta_champ_prob=0.01,
        mc_enabled=False,
        summary="Favorable trade.",
        warnings=["Slight IP-floor risk."],
    )
    dumped = resp.model_dump()
    assert dumped["grade"] == "B+"
    assert dumped["verdict"] == "Accept"
    assert dumped["surplus_sgp"] == 0.85
    assert dumped["grade_range"]["grade"] == "B+"
    assert dumped["grade_range"]["low"] == 0.5
    assert dumped["grade_range"]["center"] == 0.85
    assert dumped["grade_range"]["high"] == 1.3
    assert dumped["giving"][0]["name"] == "Logan Webb"
    assert dumped["receiving"][0]["positions"] == "OF,1B"
    assert dumped["category_impacts"][0]["cat"] == "HR"
    assert dumped["category_impacts"][0]["delta"] == 0.4
    assert dumped["mc_enabled"] is False
    assert dumped["warnings"] == ["Slight IP-floor risk."]


def test_trade_evaluate_request_defaults():
    req = TradeEvaluateRequest(team_name="Team Hickey", giving_ids=[1], receiving_ids=[2])
    assert req.enable_mc is False


class _FakeTradeService:
    def evaluate(self, team_name, giving_ids, receiving_ids, enable_mc=False):
        return TradeEvaluationResponse(
            grade="A",
            verdict="Accept",
            confidence_pct=85.0,
            surplus_sgp=1.5,
            grade_range=GradeRange(grade="A", low=1.0, center=1.5, high=2.0),
            giving=[PlayerRef(id=g, name=f"Player {g}", positions="OF") for g in giving_ids],
            receiving=[PlayerRef(id=r, name=f"Player {r}", positions="1B") for r in receiving_ids],
            mc_enabled=enable_mc,
            summary="Strong accept.",
        )


def test_post_trade_evaluate_returns_contract():
    app = create_app()
    app.dependency_overrides[get_trade_service] = lambda: _FakeTradeService()
    client = TestClient(app)
    resp = client.post(
        "/api/trade/evaluate",
        json={"team_name": "Team Hickey", "giving_ids": [1], "receiving_ids": [2]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["grade"] == "A"
    assert body["verdict"] == "Accept"
    assert body["surplus_sgp"] == 1.5
    assert body["mc_enabled"] is False


def test_post_trade_evaluate_with_mc_opt_in():
    app = create_app()
    app.dependency_overrides[get_trade_service] = lambda: _FakeTradeService()
    client = TestClient(app)
    resp = client.post(
        "/api/trade/evaluate",
        json={"team_name": "Team Hickey", "giving_ids": [1], "receiving_ids": [2], "enable_mc": True},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["mc_enabled"] is True


# ── M3-1 viewer-team resolution (the trade EVALUATOR was missed in the first wiring) ──


class _SpyTradeService:
    """Captures the team_name the router forwarded to the engine."""

    def __init__(self):
        self.seen = None

    def evaluate(self, team_name, giving_ids, receiving_ids, enable_mc=False):
        self.seen = team_name
        return TradeEvaluationResponse(
            grade="A",
            verdict="Accept",
            confidence_pct=85.0,
            surplus_sgp=1.5,
            grade_range=GradeRange(grade="A", low=1.0, center=1.5, high=2.0),
            giving=[],
            receiving=[],
            mc_enabled=enable_mc,
            summary="ok",
        )


class _ClerkVerifier:
    def verify(self, authorization):
        return Principal(subject="user_42", clerk_user_id="user_42")


def test_trade_evaluate_resolves_assigned_team_over_body():
    # A logged-in user's assigned team OVERRIDES the body team_name (closes the gap
    # where the evaluator trusted a client-supplied team).
    app = create_app()
    spy = _SpyTradeService()
    users, leagues, members = InMemoryUserStore(), InMemoryLeagueStore(), InMemoryMembershipStore()
    u = users.get_or_create("user_42")
    lg = leagues.get_or_create_default()
    members.assign(u.id, lg.id, "Bronx Bombers", team_key=None, assigned_by=u.id)
    app.dependency_overrides[get_trade_service] = lambda: spy
    app.dependency_overrides[get_auth_verifier] = lambda: _ClerkVerifier()
    app.dependency_overrides[get_user_store] = lambda: users
    app.dependency_overrides[get_league_store] = lambda: leagues
    app.dependency_overrides[get_membership_store] = lambda: members
    resp = TestClient(app).post(
        "/api/trade/evaluate",
        json={"team_name": "Team Hickey", "giving_ids": [], "receiving_ids": []},
        headers={"Authorization": "Bearer x"},
    )
    assert resp.status_code == 200
    assert spy.seen == "Bronx Bombers"  # resolved viewer team, NOT the body's team_name


def test_trade_evaluate_dormant_uses_body_team():
    # Clerk off / no token → body team_name honored (byte-for-byte today's behavior).
    app = create_app()
    spy = _SpyTradeService()
    app.dependency_overrides[get_trade_service] = lambda: spy
    resp = TestClient(app).post(
        "/api/trade/evaluate",
        json={"team_name": "Team Hickey", "giving_ids": [], "receiving_ids": []},
    )
    assert resp.status_code == 200
    assert spy.seen == "Team Hickey"


def test_build_response_nan_safe():
    """A NaN headline value (weighted_gain) must serialize as 0.0, not NaN, in the
    trade response — `float(v or 0.0)` doesn't guard NaN. The honest headline is now
    assembled by _build_response; surplus/confidence derive from weighted_gain."""
    from api.services.trade_service import TradeService

    nan = float("nan")
    resp = TradeService._build_response(
        cat_net={"HR": nan},
        weighted_gain=nan,
        pool=None,
        giving_ids=[],
        receiving_ids=[],
        enable_mc=False,
        engine_extra={},
    )
    assert resp.surplus_sgp == 0.0  # NaN gain → 0.0 surplus (Even / fair value)
    assert resp.verdict == "Even / fair value"
    # A NaN category delta is dropped (not serialized as NaN).
    assert all(ci.delta == ci.delta for ci in resp.category_impacts)  # no NaN
