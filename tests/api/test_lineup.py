"""Tests for POST /api/lineup/optimize endpoint (Slice 2, Task 3)."""

from starlette.testclient import TestClient

from api.contracts.lineup import LineupOptimizeRequest, LineupOptimizeResponse, LineupSlot
from api.deps import get_lineup_service
from api.main import create_app


class _FakeLineupService:
    def optimize(self, req: LineupOptimizeRequest) -> LineupOptimizeResponse:
        return LineupOptimizeResponse(
            team_name=req.team_name,
            mode=req.mode,
            slots=[
                LineupSlot(slot="C", player_name="Cal Raleigh", player_id=101),
                LineupSlot(slot="1B", player_name="Freddie Freeman", player_id=102),
            ],
            recommendations=["Start Freeman — strong platoon advantage."],
        )


def test_optimize_lineup_returns_contract():
    app = create_app()
    app.dependency_overrides[get_lineup_service] = lambda: _FakeLineupService()
    client = TestClient(app)
    payload = {
        "team_name": "Team Hickey",
        "roster_ids": [101, 102, 103],
        "mode": "quick",
        "weeks_remaining": 14,
    }
    resp = client.post("/api/lineup/optimize", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["team_name"] == "Team Hickey"
    assert body["mode"] == "quick"
    assert len(body["slots"]) == 2
    assert body["slots"][0]["slot"] == "C"
    assert body["slots"][0]["player_name"] == "Cal Raleigh"
    assert body["recommendations"] == ["Start Freeman — strong platoon advantage."]


def test_lineup_contract_shape():
    req = LineupOptimizeRequest(
        team_name="Team Hickey",
        roster_ids=[1, 2, 3],
        mode="standard",
        weeks_remaining=12,
    )
    resp = LineupOptimizeResponse(
        team_name=req.team_name,
        mode=req.mode,
        slots=[LineupSlot(slot="SP", player_name="Paul Skenes", player_id=200)],
        recommendations=["Skenes streams well."],
    )
    dumped = resp.model_dump()
    assert dumped["team_name"] == "Team Hickey"
    assert dumped["slots"][0]["slot"] == "SP"
    assert dumped["recommendations"][0] == "Skenes streams well."


def test_lineup_empty_roster():
    """Empty roster_ids still returns a valid (empty) response via fake."""

    class _EmptyFake:
        def optimize(self, req: LineupOptimizeRequest) -> LineupOptimizeResponse:
            return LineupOptimizeResponse(
                team_name=req.team_name,
                mode=req.mode,
                slots=[],
                recommendations=["No roster data available."],
            )

    app = create_app()
    app.dependency_overrides[get_lineup_service] = lambda: _EmptyFake()
    client = TestClient(app)
    resp = client.post(
        "/api/lineup/optimize",
        json={"team_name": "Team Hickey", "roster_ids": [], "mode": "quick"},
    )
    assert resp.status_code == 200
    assert resp.json()["slots"] == []
