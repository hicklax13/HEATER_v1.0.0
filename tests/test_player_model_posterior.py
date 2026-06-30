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


def test_sigma2_heteroscedastic_low_sample_is_more_uncertain():
    from src.player_model.posterior import between_player_sigma2

    # Same mean, different YTD sample size -> low-sample player has LARGER true-talent variance.
    lo = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=50, is_hitter=True)
    hi = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=1200, is_hitter=True)
    assert lo > hi


def test_sigma2_floor_never_vanishes():
    from src.player_model.posterior import between_player_sigma2

    # Even an "infinitely sampled" player keeps the irreducible projection-error floor (gap G3).
    huge = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=10_000_000, is_hitter=True)
    assert huge > 0.0
    # The floor equals (|mean| * _PROJ_FLOOR_CV)^2 in the large-sample limit.
    from src.player_model.posterior import _PROJ_FLOOR_CV

    assert huge == pytest.approx((1.15 * _PROJ_FLOOR_CV) ** 2, rel=1e-6)


def test_sigma2_rate_uses_absolute_std_not_cv():
    from src.player_model.posterior import _RATE_FLOOR_STD, between_player_sigma2

    # Rate cats use an ABSOLUTE std floor (a 0.40 CV on a 0.270 AVG would be absurd).
    s = between_player_sigma2(mean=0.270, kind="rate_prop", category="AVG", n=10_000_000, is_hitter=True)
    assert s == pytest.approx(_RATE_FLOOR_STD["AVG"] ** 2, rel=1e-6)


def test_sigma2_zero_sample_is_max_uncertainty():
    from src.player_model.posterior import between_player_sigma2

    # n=0 -> shrink=1 -> full talent spread + floor.
    s0 = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=0, is_hitter=True)
    s_big = between_player_sigma2(mean=1.15, kind="counting", category="HR", n=5000, is_hitter=True)
    assert s0 > s_big


def test_tau2_counting_is_phi_times_mean_and_nb_margin_consistent():
    from src.player_model.posterior import _COUNTING_OVERDISPERSION, week_to_week_tau2

    mu = 1.15
    tau2, margin = week_to_week_tau2(mean=mu, kind="counting", category="HR", weekly_vol=0.0)
    phi = _COUNTING_OVERDISPERSION["HR"]
    assert tau2 == pytest.approx(phi * mu)
    assert margin["dist"] == "nb"
    # NB variance mu + mu^2/r must equal tau2 (the margin reconstructs the same variance).
    r = margin["r"]
    assert mu + mu * mu / r == pytest.approx(tau2, rel=1e-6)


def test_tau2_counting_poisson_limit_when_phi_one(monkeypatch):
    import src.player_model.posterior as pm

    monkeypatch.setitem(pm._COUNTING_OVERDISPERSION, "HR", 1.0)
    tau2, margin = pm.week_to_week_tau2(mean=2.0, kind="counting", category="HR", weekly_vol=0.0)
    assert tau2 == pytest.approx(2.0)  # Poisson: variance == mean
    assert math.isinf(margin["r"])  # phi=1 -> r = inf (Poisson)


def test_tau2_rate_prop_decreases_with_weekly_volume():
    from src.player_model.posterior import week_to_week_tau2

    lo_vol, _ = week_to_week_tau2(mean=0.270, kind="rate_prop", category="AVG", weekly_vol=10.0)
    hi_vol, _ = week_to_week_tau2(mean=0.270, kind="rate_prop", category="AVG", weekly_vol=40.0)
    assert lo_vol > hi_vol  # more AB/week -> tighter weekly AVG
    _, margin = week_to_week_tau2(mean=0.270, kind="rate_prop", category="AVG", weekly_vol=20.0)
    assert margin["dist"] == "beta_binomial"
    assert margin["theta"] == pytest.approx(0.270)


def test_tau2_rate_ratio_margin_is_ratio_normal():
    from src.player_model.posterior import week_to_week_tau2

    tau2, margin = week_to_week_tau2(mean=3.50, kind="rate_ratio", category="ERA", weekly_vol=20.0)
    assert tau2 > 0
    assert margin["dist"] == "ratio_normal"
    assert margin["std_week"] == pytest.approx(math.sqrt(tau2), rel=1e-6)


def test_category_posterior_full_object_hitter_counting():
    from src.player_model.posterior import category_posterior

    cfg = LeagueConfig()
    p = category_posterior(_hitter_row(hr=30, ytd_pa=300), "HR", cfg, weeks=26)
    assert p.category == "HR" and p.kind == "counting"
    assert p.mean == pytest.approx(30 / 26)
    assert p.sigma2 > 0 and p.tau2 > 0
    assert p.margin["dist"] == "nb"


def test_category_posterior_rate_uses_weekly_volume_from_pool():
    from src.player_model.posterior import category_posterior

    cfg = LeagueConfig()
    # 520 AB over 26 weeks -> 20 AB/week feeds the beta-binomial n.
    p = category_posterior(_hitter_row(avg=0.300, ab=520), "AVG", cfg, weeks=26)
    assert p.kind == "rate_prop"
    assert p.margin["n"] == pytest.approx(20.0)
    assert p.margin["theta"] == pytest.approx(0.300)


def test_player_posteriors_covers_only_relevant_cats():
    from src.player_model.posterior import player_posteriors

    cfg = LeagueConfig()
    hit = player_posteriors(_hitter_row(), cfg)
    assert set(hit.keys()) == set(cfg.hitting_categories)  # hitter -> 6 hitting cats only
    pit = player_posteriors(_pitcher_row(), cfg)
    assert set(pit.keys()) == set(cfg.pitching_categories)  # pitcher -> 6 pitching cats only


def test_sigma2_wired_to_ytd_sample_through_posterior():
    from src.player_model.posterior import category_posterior

    cfg = LeagueConfig()
    rookie = category_posterior(_hitter_row(hr=30, ytd_pa=40), "HR", cfg)
    veteran = category_posterior(_hitter_row(hr=30, ytd_pa=1500), "HR", cfg)
    assert rookie.sigma2 > veteran.sigma2  # low YTD sample -> wider true-talent band


def test_never_raises_on_nan_and_missing_columns():
    from src.player_model.posterior import category_posterior, player_posteriors

    cfg = LeagueConfig()
    sparse = pd.Series({"player_id": 9, "is_hitter": 1, "hr": np.nan})  # missing most columns
    p = category_posterior(sparse, "HR", cfg)
    assert math.isfinite(p.mean) and math.isfinite(p.sigma2) and math.isfinite(p.tau2)
    assert p.sigma2 > 0  # floor still applies even with a NaN mean (degrades to 0 mean -> floor only path)
    out = player_posteriors(sparse, cfg)
    assert set(out.keys()) == set(cfg.hitting_categories)


def test_pitcher_rate_ratio_uses_ip_volume_and_ytd_ip_sample():
    from src.player_model.posterior import category_posterior

    cfg = LeagueConfig()
    p = category_posterior(_pitcher_row(era=3.20, ip=180.0, ytd_ip=60.0), "ERA", cfg, weeks=26)
    assert p.kind == "rate_ratio"
    assert p.margin["dist"] == "ratio_normal"
    assert p.mean == pytest.approx(3.20)
    assert math.isfinite(p.sigma2) and p.sigma2 > 0


def test_categories_locked_to_league_config():
    # No hardcoded category list — every produced category is a LeagueConfig category.
    from src.player_model.posterior import player_posteriors

    cfg = LeagueConfig()
    keys = set(player_posteriors(_hitter_row(), cfg)) | set(player_posteriors(_pitcher_row(), cfg))
    assert keys == set(cfg.all_categories)
