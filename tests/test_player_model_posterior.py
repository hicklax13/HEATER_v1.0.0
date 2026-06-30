"""Tests for the Layer-0 posterior variance core (Phase 1 slice 1)."""

import math

import numpy as np
import pandas as pd
import pytest

from src.valuation import LeagueConfig


def _hitter_row(**over):
    base = dict(
        player_id=1,
        name="Test Hitter",
        is_hitter=1,
        r=90,
        hr=30,
        rbi=95,
        sb=12,
        avg=0.270,
        obp=0.340,
        ab=560,
        pa=620,
        ip=0.0,
        w=0,
        l=0,
        sv=0,
        k=0,
        era=0.0,
        whip=0.0,
        ytd_pa=0,
        ytd_ip=0.0,
    )
    base.update(over)
    return pd.Series(base)


def _pitcher_row(**over):
    base = dict(
        player_id=2,
        name="Test Pitcher",
        is_hitter=0,
        r=0,
        hr=0,
        rbi=0,
        sb=0,
        avg=0.0,
        obp=0.0,
        ab=0,
        pa=0,
        ip=180.0,
        w=14,
        l=8,
        sv=0,
        k=200,
        era=3.50,
        whip=1.15,
        ytd_pa=0,
        ytd_ip=0.0,
    )
    base.update(over)
    return pd.Series(base)


def test_category_kind_classification():
    from src.player_model.posterior import classify_kind

    assert classify_kind("HR") == "counting"
    assert classify_kind("SB") == "counting"
    assert classify_kind("K") == "counting"
    assert classify_kind("L") == "counting"  # inverse but still a counting total
    assert classify_kind("AVG") == "rate_prop"
    assert classify_kind("OBP") == "rate_prop"
    assert classify_kind("ERA") == "rate_ratio"  # events/IP, not a [0,1] proportion
    assert classify_kind("WHIP") == "rate_ratio"


def test_counting_mean_is_per_week():
    from src.player_model.posterior import category_posterior

    cfg = LeagueConfig()
    p = category_posterior(_hitter_row(hr=30), "HR", cfg, weeks=26)
    assert p.mean == pytest.approx(30 / 26)  # season total spread over weeks


def test_rate_mean_is_the_rate_directly():
    from src.player_model.posterior import category_posterior

    cfg = LeagueConfig()
    p = category_posterior(_hitter_row(avg=0.270), "AVG", cfg, weeks=26)
    assert p.mean == pytest.approx(0.270)  # rate cats are not divided by weeks


def test_posterior_fields_present():
    from src.player_model.posterior import CategoryPosterior, category_posterior

    cfg = LeagueConfig()
    p = category_posterior(_hitter_row(), "HR", cfg)
    assert isinstance(p, CategoryPosterior)
    assert p.category == "HR"
    assert p.kind == "counting"
    assert p.mean > 0 and p.sigma2 > 0 and p.tau2 > 0
    assert isinstance(p.margin, dict) and p.margin["dist"] == "nb"
