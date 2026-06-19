"""Streaming contract widen: mapping + budget/top_pick unit tests + contract test."""

from __future__ import annotations

from api.services.streaming_service import StreamingService, _build_budget, _pick_top


def _board_row(**over):
    row = {
        "player_id": 660271,
        "mlb_id": 660271,
        "player_name": "Tarik Skubal",
        "team": "DET",
        "opponent": "CWS",
        "is_home": True,
        "stream_score": 82.5,
        "status": "PROBABLE",
        "confidence": "HIGH",
        "num_starts": 2,
        "actionable": True,
        "net_sgp": 1.8,
        "opp_wrc_plus": 88.0,
        "opp_k_pct": 24.5,
        "park_factor": 0.96,
        "expected_ip": 6.0,
        "expected_k": 7.0,
        "expected_er": 2.0,
        "win_probability": 0.62,
        "percent_owned": 99.0,
        "risk_flags": ["LOW_CONFIDENCE"],
        "components": {"matchup": 0.4, "env": 0.1, "form": 0.2, "lineup": -0.1, "sgp": 0.5, "winprob": 0.3},
    }
    row.update(over)
    return row


def test_to_candidate_maps_all_widened_fields():
    c = StreamingService._to_candidate(_board_row(), rank=1)
    assert c.rank == 1
    assert c.player.mlb_id == 660271
    assert c.player.team_id == 116  # DET
    assert c.is_home is True
    assert c.score == 82.5
    assert c.status == "PROBABLE"  # raw engine value (frontend adapter lowercases)
    assert c.confidence == "HIGH"
    assert c.num_starts == 2
    assert c.net_sgp == 1.8
    assert c.opp_wrc_plus == 88.0
    assert c.opp_k_pct == 24.5
    assert c.park == 0.96
    assert c.expected_ip == 6.0
    assert c.win_pct == 0.62
    assert c.own_pct == 99.0
    assert c.risk_flags == ["LOW_CONFIDENCE"]
    assert c.components.matchup == 0.4
    assert c.components.winprob == 0.3
    assert c.expected_line == "6.0 IP · 7 K · 2 ER"


def test_to_candidate_is_nan_safe():
    c = StreamingService._to_candidate(
        _board_row(opp_wrc_plus=float("nan"), percent_owned=float("nan"), components={"matchup": float("nan")}),
        rank=3,
    )
    assert c.opp_wrc_plus == 0.0  # NaN → 0.0, never reaches JSON as NaN
    assert c.own_pct == 0.0
    assert c.components.matchup == 0.0


def test_pick_top_returns_first_actionable():
    a = StreamingService._to_candidate(_board_row(actionable=False, stream_score=99.0), rank=1)
    b = StreamingService._to_candidate(_board_row(actionable=True, stream_score=80.0), rank=2)
    assert _pick_top([a, b]) is b  # skips the locked higher-scorer
    assert _pick_top([]) is None


class _FakeCtx:
    adds_remaining_this_week = 7
    category_gaps = {"K": -2.0, "ERA": 1.5, "SV": -0.5, "HR": -3.0}  # HR is a hitting cat


def test_build_budget_from_ctx():
    from src.ip_tracker import WEEKLY_TARGET

    b = _build_budget(_FakeCtx())
    assert b.adds_total == 10
    assert b.adds_left == 7
    assert b.ip_target == float(WEEKLY_TARGET)  # real constant (~53.85); not the 54.0 fallback
    assert b.ip_pace == 0.0  # deferred
    # only contested (gap<=0) PITCHING cats; HR (hitting) excluded, ERA (gap>0) excluded
    assert set(b.cats_in_play) == {"K", "SV"}


def test_streaming_endpoint_includes_top_pick_and_budget():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.streaming import BudgetStrip, StreamCandidate, StreamingResponse
    from api.deps import get_streaming_service
    from api.main import create_app

    class _FakeStreaming:
        def get_streaming(self, date=None, limit=25):
            cand = StreamCandidate(
                player=PlayerRef(id=1, mlb_id=660271, name="X", positions="SP", team_abbr="DET", team_id=116),
                score=82.5,
                rank=1,
                expected_line="6.0 IP · 7 K · 2 ER",
            )
            return StreamingResponse(
                date="2026-06-19",
                candidates=[cand],
                top_pick=cand,
                budget=BudgetStrip(adds_left=7, cats_in_play=["K"]),
            )

    app = create_app()
    app.dependency_overrides[get_streaming_service] = lambda: _FakeStreaming()
    try:
        body = TestClient(app).get("/api/streaming").json()
        assert body["top_pick"]["expected_line"] == "6.0 IP · 7 K · 2 ER"
        assert body["budget"]["adds_left"] == 7
        assert body["budget"]["ip_target"] == 54.0
        assert body["candidates"][0]["components"]["matchup"] == 0.0
    finally:
        app.dependency_overrides.clear()
