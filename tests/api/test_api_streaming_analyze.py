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
