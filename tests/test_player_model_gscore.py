"""Tests for the Layer-0 display G-score (Phase 1 slice 3)."""

import math

import pandas as pd
import pytest

from src.player_model.posterior import player_posteriors
from src.valuation import LeagueConfig


def test_kappa_small_sample_correction():
    from src.player_model.gscore import kappa

    # kappa = 2N/(2N-1): >1, decreasing toward 1 as N grows.
    assert kappa(1) == pytest.approx(2 / 1)  # 2.0
    assert kappa(10) == pytest.approx(20 / 19)
    assert kappa(10) < kappa(2)
    assert kappa(10_000) == pytest.approx(1.0, abs=1e-3)


def test_category_gscore_above_league_is_positive():
    from src.player_model.gscore import category_gscore

    g = category_gscore(mean=1.5, tau2=1.8, league_mean=1.0, league_sd=0.4, n_slots=10, inverse=False)
    assert g > 0  # above-league HR rate -> positive value


def test_category_gscore_below_league_is_negative():
    from src.player_model.gscore import category_gscore

    g = category_gscore(mean=0.6, tau2=1.8, league_mean=1.0, league_sd=0.4, n_slots=10, inverse=False)
    assert g < 0


def test_category_gscore_inverse_lower_is_better():
    from src.player_model.gscore import category_gscore

    # ERA: a 3.00 ERA vs a 4.00 league mean is GOOD -> positive G (numerator flipped).
    g = category_gscore(mean=3.00, tau2=0.5, league_mean=4.00, league_sd=0.7, n_slots=8, inverse=True)
    assert g > 0
    worse = category_gscore(mean=5.00, tau2=0.5, league_mean=4.00, league_sd=0.7, n_slots=8, inverse=True)
    assert worse < 0


def test_category_gscore_reduces_to_zscore_when_tau2_zero():
    from src.player_model.gscore import category_gscore

    # tau2=0 -> denominator = league_sd -> classic z-score (Rosenof: |W|=1 special case).
    g = category_gscore(mean=1.4, tau2=0.0, league_mean=1.0, league_sd=0.5, n_slots=10, inverse=False)
    assert g == pytest.approx((1.4 - 1.0) / 0.5)


def test_category_gscore_zero_denominator_is_zero_not_error():
    from src.player_model.gscore import category_gscore

    g = category_gscore(mean=1.4, tau2=0.0, league_mean=1.0, league_sd=0.0, n_slots=10, inverse=False)
    assert g == 0.0  # no spread + no noise -> no defined signal, return 0 (never divide-by-zero)


def _elite_hitter():
    return pd.Series(
        dict(
            player_id=1,
            name="Elite",
            is_hitter=1,
            r=110,
            hr=42,
            rbi=115,
            sb=20,
            avg=0.310,
            obp=0.390,
            ab=560,
            pa=640,
            ytd_pa=600,
            ytd_ip=0.0,
        )
    )


def _replacement_hitter():
    return pd.Series(
        dict(
            player_id=2,
            name="Repl",
            is_hitter=1,
            r=55,
            hr=10,
            rbi=50,
            sb=3,
            avg=0.235,
            obp=0.290,
            ab=480,
            pa=520,
            ytd_pa=500,
            ytd_ip=0.0,
        )
    )


def _league_ctx():
    from src.player_model.gscore import LeagueContext

    # Per-week league means + per-player league spreads (rough mid-pack values).
    return LeagueContext(
        means={"R": 80 / 26, "HR": 22 / 26, "RBI": 80 / 26, "SB": 10 / 26, "AVG": 0.255, "OBP": 0.320},
        sds={"R": 0.6, "HR": 0.35, "RBI": 0.6, "SB": 0.4, "AVG": 0.022, "OBP": 0.025},
    )


def test_player_gscore_elite_beats_replacement():
    from src.player_model.gscore import player_gscore

    cfg = LeagueConfig()
    ctx = _league_ctx()
    elite = player_gscore(player_posteriors(_elite_hitter(), cfg), ctx, cfg)
    repl = player_gscore(player_posteriors(_replacement_hitter(), cfg), ctx, cfg)
    assert elite > repl
    assert elite > 0  # an above-average hitter has positive aggregate value


def test_player_gscore_returns_total_and_per_cat():
    from src.player_model.gscore import player_gscore

    cfg = LeagueConfig()
    out = player_gscore(player_posteriors(_elite_hitter(), cfg), _league_ctx(), cfg, detail=True)
    assert "total" in out and "per_category" in out
    assert set(out["per_category"]) == set(cfg.hitting_categories)
    assert out["total"] == pytest.approx(sum(out["per_category"].values()))


def test_player_gscore_empty_context_no_raise():
    from src.player_model.gscore import LeagueContext, player_gscore

    cfg = LeagueConfig()
    out = player_gscore(player_posteriors(_elite_hitter(), cfg), LeagueContext(), cfg, detail=True)
    # No league spreads -> every denominator collapses -> all-zero, but no error.
    assert math.isfinite(out["total"])
    assert all(math.isfinite(v) for v in out["per_category"].values())


def test_category_gscore_nan_inputs_safe():
    from src.player_model.gscore import category_gscore

    g = category_gscore(
        mean=float("nan"),
        tau2=float("nan"),
        league_mean=float("nan"),
        league_sd=float("nan"),
        n_slots=10,
        inverse=False,
    )
    assert g == 0.0  # all-NaN -> coerced to 0 -> zero denominator -> 0.0


def test_pitcher_gscore_uses_pitcher_slots_and_inverse():
    from src.player_model.gscore import LeagueContext, player_gscore

    cfg = LeagueConfig()
    pitcher = pd.Series(
        dict(
            player_id=3,
            name="Ace",
            is_hitter=0,
            w=16,
            l=6,
            sv=0,
            k=220,
            era=2.90,
            whip=1.02,
            ip=190.0,
            ytd_ip=70.0,
            ytd_pa=0,
        )
    )
    ctx = LeagueContext(
        means={"W": 10 / 26, "L": 8 / 26, "SV": 5 / 26, "K": 180 / 26, "ERA": 3.95, "WHIP": 1.25},
        sds={"W": 0.2, "L": 0.2, "SV": 0.6, "K": 1.5, "ERA": 0.7, "WHIP": 0.12},
    )
    out = player_gscore(player_posteriors(pitcher, cfg), ctx, cfg, detail=True)
    # A 2.90-ERA ace beats a 3.95 league ERA -> positive ERA G (inverse handled).
    assert out["per_category"]["ERA"] > 0
    assert out["per_category"]["WHIP"] > 0
    assert set(out["per_category"]) == set(cfg.pitching_categories)
