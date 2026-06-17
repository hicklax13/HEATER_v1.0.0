from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.streaming import StreamCandidate, StreamingResponse
from api.deps import get_streaming_service
from api.main import create_app


def test_streaming_contract_shape():
    resp = StreamingResponse(
        date="2026-06-17",
        candidates=[
            StreamCandidate(
                player=PlayerRef(id=1001, name="Gerrit Cole", positions="SP"),
                team="NYY",
                opponent="BOS",
                score=82.5,
                actionable=True,
                status="PROBABLE",
                reason="Elite matchup vs weak lineup",
            )
        ],
    )
    dumped = resp.model_dump()
    assert dumped["date"] == "2026-06-17"
    assert dumped["candidates"][0]["player"]["name"] == "Gerrit Cole"
    assert dumped["candidates"][0]["score"] == 82.5
    # defaults
    assert StreamCandidate(player=PlayerRef(id=1, name="X", positions="SP")).actionable is True
    assert StreamCandidate(player=PlayerRef(id=1, name="X", positions="SP")).score == 0.0


class _FakeStreamingService:
    def get_streaming(self, date: str | None = None, limit: int = 25) -> StreamingResponse:
        target = date or "2026-06-17"
        return StreamingResponse(
            date=target,
            candidates=[
                StreamCandidate(
                    player=PlayerRef(id=555, name="Shohei Ohtani", positions="SP"),
                    team="LAD",
                    opponent="SF",
                    score=91.0,
                    actionable=True,
                    status="PROBABLE",
                    reason="",
                )
            ],
        )


def test_get_streaming_returns_contract():
    app = create_app()
    app.dependency_overrides[get_streaming_service] = lambda: _FakeStreamingService()
    client = TestClient(app)
    resp = client.get("/api/streaming")
    assert resp.status_code == 200
    body = resp.json()
    assert "date" in body
    assert body["candidates"][0]["player"]["name"] == "Shohei Ohtani"
    assert body["candidates"][0]["score"] == 91.0
