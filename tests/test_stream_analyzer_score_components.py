"""Stream Score guards: component breakdown, registry-driven weights, risk flags.

Phase 1 of the Pitcher Streaming Analyzer plan. The composite must be fully
decomposable (six components, each in [-1, 1]), read its weights from
CONSTANTS_REGISTRY at call time (calibration-compatible), degrade neutrally
when optional data is missing, and fire each risk flag on the documented
threshold from the registry.
"""

from __future__ import annotations

from unittest import mock

import pytest

from src.game_day import OUTFIELD_BEARING
from src.optimizer.constants_registry import CONSTANTS_REGISTRY, ConstantEntry
from src.optimizer.stream_analyzer import score_stream_candidate
from src.valuation import LeagueConfig

_COMPONENT_KEYS = {"matchup", "sgp", "form", "lineup", "env", "winprob"}


def _pitcher(**overrides):
    row = {
        "player_id": 1,
        "player_name": "Streamer A",
        "team": "SEA",
        "throws": "R",
        "era": 3.40,
        "whip": 1.08,
        "k": 175.0,
        "w": 11.0,
        "ip": 170.0,
        "ip_per_start": 5.8,
        "k_bb_pct": 0.17,
        "xfip": 3.70,
        "csw_pct": 0.30,
        "percent_owned": 12.0,
    }
    row.update(overrides)
    return row


def _start(**overrides):
    info = {
        "game_date": "2026-06-12",
        "opponent": "CHW",
        "is_home": True,
        "venue": "SEA",
        "park_factor": 1.00,
        "weather": {"wind_mph": 4.0, "wind_dir": 0.0},
        "confidence": "HIGH",
        "num_starts": 1,
    }
    info.update(overrides)
    return info


def _opp(wrc_plus=95.0, k_pct=23.0):
    return {
        "wrc_plus": wrc_plus,
        "k_pct": k_pct,
        "bb_pct": 8.0,
        "iso": None,
        "l14_wrc_plus": None,
        "split_source": "overall",
    }


def test_result_shape_and_ranges():
    res = score_stream_candidate(_pitcher(), _start(), _opp(), LeagueConfig())
    assert 0.0 <= res["stream_score"] <= 100.0
    assert set(res["components"]) == _COMPONENT_KEYS
    for name, val in res["components"].items():
        assert -1.0 <= val <= 1.0, f"component {name}={val} outside [-1, 1]"
    assert isinstance(res["risk_flags"], list)
    assert {"ip", "k", "er", "win_prob"} <= set(res["expected_line"])
    assert isinstance(res["net_sgp"], float)


def test_weights_read_from_registry_at_call_time():
    """Patching the registry must change the score (sigmoid-calibrator pattern)."""
    pitcher = _pitcher(era=5.60, whip=1.45, k_bb_pct=0.20, xfip=3.30, csw_pct=0.33)
    start = _start()
    opp = _opp(wrc_plus=80.0, k_pct=27.0)
    cfg = LeagueConfig()

    def _weights_patch(**w):
        return {
            f"stream_score_w_{name}": ConstantEntry(
                value=val,
                lower_bound=0.0,
                upper_bound=1.0,
                citation="test",
                module="optimizer/stream_analyzer.py",
                sensitivity="HIGH",
                description="test",
            )
            for name, val in w.items()
        }

    all_matchup = _weights_patch(matchup=1.0, sgp=0.0, form=0.0, lineup=0.0, env=0.0, winprob=0.0)
    all_sgp = _weights_patch(matchup=0.0, sgp=1.0, form=0.0, lineup=0.0, env=0.0, winprob=0.0)

    with mock.patch.dict(CONSTANTS_REGISTRY, all_matchup):
        score_matchup = score_stream_candidate(pitcher, start, opp, cfg)["stream_score"]
    with mock.patch.dict(CONSTANTS_REGISTRY, all_sgp):
        score_sgp = score_stream_candidate(pitcher, start, opp, cfg)["stream_score"]

    assert score_matchup != pytest.approx(score_sgp), (
        "score must respond to registry weight changes at call time — weights are being cached at import"
    )


def test_matchup_monotonicity():
    """Weak offense in a pitcher park must outscore elite offense in a hitter park."""
    pitcher = _pitcher()
    cfg = LeagueConfig()
    soft = score_stream_candidate(
        pitcher,
        _start(park_factor=0.92),
        _opp(wrc_plus=80.0, k_pct=26.5),
        cfg,
    )
    brutal = score_stream_candidate(
        pitcher,
        _start(park_factor=1.12),
        _opp(wrc_plus=115.0, k_pct=18.0),
        cfg,
    )
    assert soft["stream_score"] > brutal["stream_score"]


def test_missing_lineup_data_is_neutral():
    res = score_stream_candidate(_pitcher(), _start(), _opp(), LeagueConfig(), lineup_exposure=None)
    assert res["components"]["lineup"] == 0.0


def test_missing_form_data_is_neutral():
    res = score_stream_candidate(_pitcher(), _start(), _opp(), LeagueConfig(), recent_form=None)
    assert res["components"]["form"] == 0.0


def test_clean_fixture_has_no_risk_flags():
    res = score_stream_candidate(_pitcher(), _start(), _opp(), LeagueConfig())
    assert res["risk_flags"] == []


@pytest.mark.parametrize(
    ("pitcher_kw", "start_kw", "opp_kw", "expected_flag"),
    [
        ({"whip": 1.55}, {}, {}, "HIGH_WHIP"),
        ({"ip_per_start": 3.8}, {}, {}, "SHORT_LEASH"),
        ({}, {}, {"wrc_plus": 115.0}, "ELITE_OFFENSE"),
        ({}, {"park_factor": 1.12}, {}, "HITTER_PARK"),
        ({}, {"confidence": "LOW"}, {}, "LOW_CONFIDENCE"),
    ],
)
def test_risk_flags_fire(pitcher_kw, start_kw, opp_kw, expected_flag):
    res = score_stream_candidate(
        _pitcher(**pitcher_kw),
        _start(**start_kw),
        _opp(**opp_kw),
        LeagueConfig(),
    )
    assert expected_flag in res["risk_flags"]


def test_wind_out_flag_fires_in_open_park():
    bearing = OUTFIELD_BEARING["BOS"]
    # Meteorological wind_dir = where wind comes FROM; blowing out means the
    # wind blows TOWARD the outfield bearing.
    blowing_out_from = (bearing + 180.0) % 360.0
    res = score_stream_candidate(
        _pitcher(team="BOS"),
        _start(venue="BOS", weather={"wind_mph": 15.0, "wind_dir": blowing_out_from}),
        _opp(),
        LeagueConfig(),
    )
    assert "WIND_OUT" in res["risk_flags"]


def test_wind_out_flag_suppressed_in_dome():
    res = score_stream_candidate(
        _pitcher(team="MIL"),
        _start(venue="MIL", weather={"wind_mph": 20.0, "wind_dir": 0.0}),
        _opp(),
        LeagueConfig(),
    )
    assert "WIND_OUT" not in res["risk_flags"]


def test_form_component_rewards_hot_streak():
    cfg = LeagueConfig()
    hot = {"l14": {"era": 1.80, "whip": 0.90, "k": 20.0, "ip": 14.0, "games": 3}}
    cold = {"l14": {"era": 6.80, "whip": 1.70, "k": 7.0, "ip": 12.0, "games": 3}}
    res_hot = score_stream_candidate(_pitcher(), _start(), _opp(), cfg, recent_form=hot)
    res_cold = score_stream_candidate(_pitcher(), _start(), _opp(), cfg, recent_form=cold)
    assert res_hot["components"]["form"] > 0.0
    assert res_cold["components"]["form"] < 0.0
    assert res_hot["stream_score"] > res_cold["stream_score"]


def test_form_volume_gate():
    """Below 5 L14 IP the form signal is noise — component must stay neutral."""
    tiny = {"l14": {"era": 0.00, "whip": 0.50, "k": 6.0, "ip": 3.0, "games": 1}}
    res = score_stream_candidate(_pitcher(), _start(), _opp(), LeagueConfig(), recent_form=tiny)
    assert res["components"]["form"] == 0.0
