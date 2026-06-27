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


def test_streaming_response_urgency_field_defaults_empty():
    """StreamingResponse carries an additive `urgency` map (per-category 0-1
    this-week need), defaulting to {} so the field is backward-compatible."""
    resp = StreamingResponse(date="2026-06-27")
    assert resp.urgency == {}
    resp2 = StreamingResponse(date="2026-06-27", urgency={"K": 0.82, "ERA": 0.18})
    dumped = resp2.model_dump()
    assert dumped["urgency"] == {"K": 0.82, "ERA": 0.18}


def test_get_streaming_attaches_urgency_from_ctx(monkeypatch):
    """get_streaming surfaces ctx.urgency_weights['urgency'] (the matchup-derived
    per-category need) on StreamingResponse.urgency, finite-coerced. DB-free:
    the engine + Yahoo singletons are faked at their BOUND import sites."""
    import api.services.streaming_service as svc

    class _Ctx:
        # what build_optimizer_context would return; the service only reads
        # urgency_weights/category_gaps/adds_remaining_this_week off it.
        urgency_weights = {"urgency": {"K": 0.82, "ERA": 0.18, "BAD": float("nan")}}
        category_gaps = {}
        adds_remaining_this_week = 7
        live_matchup = {"categories": []}

    def _fake_ctx(*a, **k):
        return _Ctx()

    def _fake_board(ctx, date, include_rostered=False):
        import pandas as pd

        return pd.DataFrame()  # empty board → no candidates, urgency still attaches

    # Patch the names the service imports INSIDE get_streaming (function-local imports).
    monkeypatch.setattr("src.optimizer.shared_data_layer.build_optimizer_context", _fake_ctx, raising=False)
    monkeypatch.setattr("src.optimizer.stream_analyzer.build_stream_board", _fake_board, raising=False)
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", lambda: object(), raising=False)
    # LeagueConfig import inside the try is real + cheap; leave it.

    out = svc.StreamingService().get_streaming(date="2026-06-27")
    assert out.date == "2026-06-27"
    # NaN dropped, finite kept:
    assert out.urgency == {"K": 0.82, "ERA": 0.18}
