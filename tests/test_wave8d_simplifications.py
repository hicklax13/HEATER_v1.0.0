"""Wave 8d structural assertions: named constants + dead-param removal.

These tests guard against regression of the LOW-severity audit cleanups
in Wave 8d. Each test pins a specific simplification:

  - Magic-number defaults that were hoisted to named module-level
    constants must keep their original numeric values (behavior preserved).
  - The dead-param removal from ``weibull_survival`` must keep the
    function signature at 3 parameters (no accidental restoration).
  - Duplicate literals collapsed in lookup tuples must not re-grow.
"""

from __future__ import annotations

import inspect

# ── Health-score baseline (DEFAULT_HEALTH_SCORE = 0.85) ──────────────


def test_trade_intelligence_default_health_score():
    from src.trade_intelligence import DEFAULT_HEALTH_SCORE

    assert DEFAULT_HEALTH_SCORE == 0.85


def test_playing_time_model_default_health_score():
    from src.playing_time_model import DEFAULT_HEALTH_SCORE

    assert DEFAULT_HEALTH_SCORE == 0.85


def test_cheat_sheet_default_health_score():
    """``src.cheat_sheet`` imports ``weasyprint`` at top level, which
    is optional on Windows. Skip if the dependency is unavailable."""
    import pytest

    try:
        from src.cheat_sheet import DEFAULT_HEALTH_SCORE
    except OSError as exc:  # pragma: no cover - weasyprint native libs missing
        pytest.skip(f"weasyprint native libs unavailable: {exc}")
        return
    assert DEFAULT_HEALTH_SCORE == 0.85


def test_validation_dynamic_context_default_health_score():
    from src.validation.dynamic_context import DEFAULT_HEALTH_SCORE

    assert DEFAULT_HEALTH_SCORE == 0.85


def test_draft_engine_default_health_score_canonical():
    """draft_engine has had DEFAULT_HEALTH_SCORE for a while; it sets the
    canonical value that all sibling modules now mirror."""
    from src.draft_engine import DEFAULT_HEALTH_SCORE

    assert DEFAULT_HEALTH_SCORE == 0.85


# ── Health adjustment thresholds (trade_intelligence) ────────────────


def test_trade_intelligence_health_adjustment_thresholds():
    from src.trade_intelligence import (
        _HEALTH_AFTER_DTD,
        _HEALTH_AFTER_IL10_15,
        _HEALTH_AFTER_IL60_OUT,
        _HEALTH_MODERATELY_HEALTHY,
        _HEALTH_NEAR_HEALTHY,
    )

    # Thresholds
    assert _HEALTH_NEAR_HEALTHY == 0.80
    assert _HEALTH_MODERATELY_HEALTHY == 0.60
    # Adjusted values
    assert _HEALTH_AFTER_IL10_15 == 0.65
    assert _HEALTH_AFTER_IL60_OUT == 0.40
    assert _HEALTH_AFTER_DTD == 0.75


# ── League-average rate-stat baselines ───────────────────────────────


def test_streaming_league_avg_constants():
    from src.optimizer.streaming import (
        _LEAGUE_AVG_ERA,
        _LEAGUE_AVG_WHIP,
        _LEAGUE_AVG_WOBA,
    )

    assert _LEAGUE_AVG_ERA == 4.50
    assert _LEAGUE_AVG_WHIP == 1.30
    assert _LEAGUE_AVG_WOBA == 0.320


def test_war_room_hotcold_league_avg_constants():
    from src.war_room_hotcold import _LEAGUE_AVG_ERA, _LEAGUE_AVG_WHIP

    assert _LEAGUE_AVG_ERA == 4.50
    assert _LEAGUE_AVG_WHIP == 1.30


def test_waiver_wire_whip_constants():
    from src.waiver_wire import _LEAGUE_AVG_WHIP, _WHIP_SAFETY_CEILING

    assert _LEAGUE_AVG_WHIP == 1.30
    assert _WHIP_SAFETY_CEILING == 1.40


# ── Streaming-specific clip bounds ───────────────────────────────────


def test_streaming_whip_safety_constants():
    from src.optimizer.streaming import (
        _WHIP_SAFETY_CEILING,
        _WHIP_UNSAFE_PENALTY,
    )

    assert _WHIP_SAFETY_CEILING == 1.40
    assert _WHIP_UNSAFE_PENALTY == 0.5


def test_streaming_sb_constants():
    from src.optimizer.streaming import (
        _LEAGUE_AVG_PITCHER_DELIVERY,
        _LEAGUE_AVG_POP_TIME,
        _LEAGUE_AVG_SPRINT_SPEED,
        _SB_MULT_HI,
        _SB_MULT_LO,
    )

    assert _LEAGUE_AVG_POP_TIME == 1.95
    assert _LEAGUE_AVG_PITCHER_DELIVERY == 1.35
    assert _LEAGUE_AVG_SPRINT_SPEED == 27.0
    # Clip is symmetric around 1.0
    assert _SB_MULT_LO == 0.85
    assert _SB_MULT_HI == 1.15
    assert _SB_MULT_LO + _SB_MULT_HI == 2.0


# ── Daily optimizer clip bounds ──────────────────────────────────────


def test_daily_optimizer_form_clip_constants():
    from src.optimizer.daily_optimizer import (
        _FORM_CLIP_HI,
        _FORM_CLIP_LO,
        _FULL_SEASON_GAMES,
        _OFFENSE_MULT_HI,
        _OFFENSE_MULT_LO,
        _PLATOON_MULT_HI,
        _PLATOON_MULT_LO,
    )

    # All three multiplier groups use the same ±20% bound
    assert _FORM_CLIP_LO == 0.80
    assert _FORM_CLIP_HI == 1.20
    assert _OFFENSE_MULT_LO == 0.80
    assert _OFFENSE_MULT_HI == 1.20
    assert _PLATOON_MULT_LO == 0.80
    assert _PLATOON_MULT_HI == 1.20
    assert _FULL_SEASON_GAMES == 162.0


# ── Weather thresholds (start_sit) ───────────────────────────────────


def test_start_sit_weather_constants():
    from src.start_sit import (
        _COLD_HR_SUPPRESS,
        _DEFAULT_TEMP_F,
        _HEAT_HR_BOOST,
        _TEMP_COLD_F,
        _TEMP_HOT_F,
        _TREND_COLD_MULT,
        _TREND_HOT_MULT,
    )

    assert _TEMP_HOT_F == 85.0
    assert _TEMP_COLD_F == 50.0
    assert _DEFAULT_TEMP_F == 72.0
    assert _HEAT_HR_BOOST == 1.03
    assert _COLD_HR_SUPPRESS == 0.97
    assert _TREND_HOT_MULT == 1.05
    assert _TREND_COLD_MULT == 0.95


# ── Dead-param removal (weibull_survival) ────────────────────────────


def test_weibull_survival_signature_has_three_params():
    """Audit D6D-024..037: ``adp_distance`` was a dead 4th param. After
    Wave 8d/2 cleanup, the signature must remain at exactly 3 params.
    Restoring the dead param will fail this test."""
    from src.pick_predictor import weibull_survival

    sig = inspect.signature(weibull_survival)
    params = list(sig.parameters.keys())
    assert params == ["picks_remaining", "shape", "scale"], f"weibull_survival signature drifted: {params}"


def test_weibull_survival_callable_with_three_args():
    """Sanity: positional call site in ``compute_survival_curve`` should
    keep working."""
    from src.pick_predictor import weibull_survival

    # scale=0 short-circuit returns 1.0
    assert weibull_survival(10, 1.0, 0.0) == 1.0
    # Bounded to [0, 1]
    s = weibull_survival(5, 1.0, 30.0)
    assert 0.0 <= s <= 1.0


# ── Duplicate-literal cleanup (data_bootstrap gmli tuple) ────────────


def test_data_bootstrap_gmli_tuple_dedup():
    """Audit D1A-011: ``("gmli", "gmli", "leverageindex", ...)`` had a
    literal duplicate. After Wave 8d/3 the source file must have only
    one occurrence of ``"gmli"`` in the column-alias lookup tuple."""
    from pathlib import Path

    src_path = Path(__file__).parent.parent / "src" / "data_bootstrap.py"
    text = src_path.read_text(encoding="utf-8")
    # The cleaned tuple has form: cl in ("gmli", "leverageindex", "gmleverageindex")
    # The pre-cleanup form had: cl in ("gmli", "gmli", "leverageindex", "gmleverageindex")
    assert '"gmli", "gmli"' not in text, "data_bootstrap.py still has duplicate 'gmli' literal in column-alias tuple"
