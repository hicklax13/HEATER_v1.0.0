from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.trade import (
    CategoryImpact,
    GradeRange,
    TradeEvaluateRequest,
    TradeEvaluationResponse,
)
from api.deps import get_trade_service
from api.main import create_app


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
