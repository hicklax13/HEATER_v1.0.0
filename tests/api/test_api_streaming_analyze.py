"""Streaming analyze + probables: helper unit tests + contract tests."""

from __future__ import annotations

from api.services.streaming_service import _factors, _likelihood_from, _to_probable


def _row(**over):
    row = {
        "player_id": 660271,
        "mlb_id": 660271,
        "player_name": "Tarik Skubal",
        "team": "DET",
        "opponent": "CWS",
        "is_home": True,
        "confidence": "HIGH",
        "opp_wrc_plus": 88.0,
        "opp_k_pct": 24.5,
        "park_factor": 0.96,
        "net_sgp": 1.8,
        "win_probability": 0.62,
        "components": {"matchup": 0.4, "env": 0.1, "form": 0.2, "lineup": -0.1, "sgp": 0.5, "winprob": 0.3},
    }
    row.update(over)
    return row


def test_likelihood_from_confidence():
    assert _likelihood_from("HIGH") == "confirmed"
    assert _likelihood_from("MEDIUM") == "likely"
    assert _likelihood_from("LOW") == "projected"
    assert _likelihood_from("") == "projected"  # unknown → projected


def test_to_probable_maps_row():
    p = _to_probable(_row())
    assert p.player.mlb_id == 660271
    assert p.player.team_id == 116  # DET
    assert p.team == "DET"
    assert p.opponent == "CWS"
    assert p.is_home is True
    assert p.pos_group == "SP"
    assert p.start_likelihood == "confirmed"


def test_factors_six_components_weights_and_details():
    from src.optimizer.constants_registry import CONSTANTS_REGISTRY as _CR

    factors = _factors(_row())
    assert [f.key for f in factors] == ["matchup", "sgp", "form", "lineup", "env", "winprob"]
    by_key = {f.key: f for f in factors}
    assert by_key["matchup"].value == 0.4
    # NOTE: f"{24.5:.0f}" = "24" via Python banker's rounding (rounds to nearest even)
    assert by_key["matchup"].detail == "vs CWS: 88 wRC+, 24% K"
    assert by_key["sgp"].detail == "+1.80 SGP"
    assert by_key["env"].detail == "Park factor 0.96"
    assert by_key["winprob"].detail == "62% team win prob"
    # weights come from the registry (live read)
    assert by_key["matchup"].weight == float(_CR["stream_score_w_matchup"].value)


def test_factors_is_nan_safe():
    factors = _factors(_row(opp_wrc_plus=float("nan"), components={"matchup": float("nan")}))
    by_key = {f.key: f for f in factors}
    assert by_key["matchup"].value == 0.0  # NaN component → 0.0
    assert "wRC+" in by_key["matchup"].detail  # composes without crashing (0 wRC+)


def test_streaming_get_includes_probables():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.streaming import ProbableStarter, StreamingResponse
    from api.deps import get_streaming_service
    from api.main import create_app

    class _Fake:
        def get_streaming(self, date=None, limit=25):
            return StreamingResponse(
                date="2026-06-19",
                probables=[
                    ProbableStarter(
                        player=PlayerRef(
                            id=1,
                            mlb_id=660271,
                            name="X",
                            positions="SP",
                            team_abbr="DET",
                            team_id=116,
                        ),
                        team="DET",
                        opponent="CWS",
                        is_home=True,
                        start_likelihood="confirmed",
                    )
                ],
            )

    app = create_app()
    app.dependency_overrides[get_streaming_service] = lambda: _Fake()
    try:
        body = TestClient(app).get("/api/streaming").json()
        assert body["probables"][0]["start_likelihood"] == "confirmed"
        assert body["probables"][0]["player"]["mlb_id"] == 660271
    finally:
        app.dependency_overrides.clear()


def test_analyze_endpoint_returns_scorecard():
    from fastapi.testclient import TestClient

    from api.contracts.common import PlayerRef
    from api.contracts.streaming import FactorDetail, PitcherScorecard, StreamAnalyzeResponse
    from api.deps import get_streaming_service
    from api.main import create_app

    class _Fake:
        def analyze_pitcher(self, req):
            return StreamAnalyzeResponse(
                found=True,
                scorecard=PitcherScorecard(
                    player=PlayerRef(
                        id=1,
                        mlb_id=660271,
                        name="Skubal",
                        positions="SP",
                        team_abbr="DET",
                        team_id=116,
                    ),
                    score=82.5,
                    rank=1,
                    expected_line="6.0 IP · 7 K · 2 ER",
                    factors=[
                        FactorDetail(
                            key="matchup",
                            label="Matchup",
                            value=0.4,
                            weight=0.35,
                            detail="vs CWS: 88 wRC+, 24% K",
                        )
                    ],
                ),
            )

    app = create_app()
    app.dependency_overrides[get_streaming_service] = lambda: _Fake()
    try:
        resp = TestClient(app).post("/api/streaming/analyze", json={"pitcher_id": 660271, "date": "2026-06-19"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["found"] is True
        assert body["scorecard"]["score"] == 82.5
        assert body["scorecard"]["factors"][0]["key"] == "matchup"
        assert body["scorecard"]["factors"][0]["weight"] == 0.35
    finally:
        app.dependency_overrides.clear()
