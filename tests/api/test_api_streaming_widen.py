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
