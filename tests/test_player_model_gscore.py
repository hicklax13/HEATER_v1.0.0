"""Tests for the Layer-0 display G-score (Phase 1 slice 3)."""

import math

import pytest

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
