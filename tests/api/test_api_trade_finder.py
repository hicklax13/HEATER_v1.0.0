from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.trade_finder import TradeFinderResponse, TradeSuggestion
from api.deps import get_trade_finder_service
from api.main import create_app


def test_trade_finder_contract_shape():
    resp = TradeFinderResponse(
        team_name="Team Hickey",
        suggestions=[
            TradeSuggestion(
                partner_team="Team Smith",
                giving=[PlayerRef(id=1, name="Player A", positions="OF")],
                receiving=[PlayerRef(id=2, name="Player B", positions="1B")],
                net_sgp=1.5,
                rationale="Improves HR and SB.",
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["team_name"] == "Team Hickey"
    assert dumped["suggestions"][0]["partner_team"] == "Team Smith"
    assert dumped["suggestions"][0]["giving"][0]["name"] == "Player A"
    assert dumped["suggestions"][0]["net_sgp"] == 1.5
    # defaults
    assert TradeSuggestion(partner_team="X").giving == []
    assert TradeSuggestion(partner_team="X").net_sgp == 0.0
    assert TradeFinderResponse(team_name="T").suggestions == []


class _FakeTradeFinderService:
    def get_suggestions(self, team_name: str, limit: int = 10) -> TradeFinderResponse:
        return TradeFinderResponse(
            team_name=team_name,
            suggestions=[
                TradeSuggestion(
                    partner_team="Rival Team",
                    giving=[PlayerRef(id=10, name="Give Guy", positions="SS")],
                    receiving=[PlayerRef(id=20, name="Recv Guy", positions="2B")],
                    net_sgp=0.8,
                    rationale="Good fit.",
                )
            ],
        )


def test_get_trade_finder_returns_contract():
    app = create_app()
    app.dependency_overrides[get_trade_finder_service] = lambda: _FakeTradeFinderService()
    client = TestClient(app)
    resp = client.get("/api/trade-finder?team_name=Team+Hickey&limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["suggestions"][0]["partner_team"] == "Rival Team"
    assert body["suggestions"][0]["giving"][0]["name"] == "Give Guy"
    assert body["suggestions"][0]["net_sgp"] == 0.8


def test_trade_finder_method_discipline_get_is_supported_post_405():
    """M-8: the live `POST /api/trade-finder` returned 405. The supported method is
    **GET** (the router exposes only @router.get), and the frontend correctly calls it
    with apiGet (`web/src/lib/trades-data.ts`). So the 405 on POST is CORRECT, not a
    server bug — the page's "No trade ideas yet" empty state is the documented
    button-gated / roster-relative empty (suggestions are empty until live Yahoo data).
    This locks the contract: GET → 200, POST → 405."""
    app = create_app()
    app.dependency_overrides[get_trade_finder_service] = lambda: _FakeTradeFinderService()
    client = TestClient(app)

    ok = client.get("/api/trade-finder?team_name=Team+Hickey&limit=5")
    assert ok.status_code == 200
    assert "suggestions" in ok.json()  # the trade-finder contract shape

    # POST is not a defined method on this route → 405 Method Not Allowed (expected).
    assert client.post("/api/trade-finder", json={"team_name": "Team Hickey"}).status_code == 405
