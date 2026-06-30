"""Tests for the Layer-0 validation machinery (Phase 1 slice 5). DB-free, network-free."""

import numpy as np
import pytest


def test_mean_absolute_error():
    from src.player_model.validation import mean_absolute_error

    assert mean_absolute_error([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == pytest.approx(2.0 / 3.0)
    assert mean_absolute_error([], []) == 0.0


def test_gaussian_coverage_well_calibrated_hits_nominal():
    from src.player_model.validation import gaussian_interval_coverage

    rng = np.random.default_rng(0)
    n = 20000
    mean = np.zeros(n)
    sigma = np.ones(n)
    realized = rng.normal(mean, sigma)  # noise EXACTLY matches sigma
    cov = gaussian_interval_coverage(mean, sigma, realized, level=0.80)
    assert cov == pytest.approx(0.80, abs=0.02)  # calibrated -> ~80% inside the 80% interval


def test_gaussian_coverage_overconfident_under_covers():
    from src.player_model.validation import gaussian_interval_coverage

    rng = np.random.default_rng(1)
    n = 20000
    realized = rng.normal(0, 2.0, size=n)  # true noise 2.0 ...
    sigma = np.full(n, 1.0)  # ... but model claims 1.0 (overconfident)
    cov = gaussian_interval_coverage(np.zeros(n), sigma, realized, level=0.80)
    assert cov < 0.80  # too-narrow intervals miss too often


def test_pit_calibrated_is_uniform_mean_half():
    from src.player_model.validation import pit_values

    rng = np.random.default_rng(2)
    n = 20000
    realized = rng.normal(0, 1, size=n)
    pit = pit_values(np.zeros(n), np.ones(n), realized)
    assert pit.min() >= 0.0 and pit.max() <= 1.0
    assert pit.mean() == pytest.approx(0.5, abs=0.02)  # uniform -> mean 0.5


def test_conformal_quantile_covers_at_least_level():
    from src.player_model.validation import conformal_quantile

    rng = np.random.default_rng(3)
    residuals = rng.normal(0, 1.5, size=10000)
    q = conformal_quantile(residuals, level=0.90)
    # |residual| <= q for >= 90% of a fresh sample.
    fresh = np.abs(rng.normal(0, 1.5, size=10000))
    assert np.mean(fresh <= q) >= 0.88
    assert q > 0


def test_coverage_implied_sigma_scale_flags_overconfidence():
    from src.player_model.validation import coverage_implied_sigma_scale

    # Observed coverage 0.60 at a nominal 0.80 -> intervals too narrow -> scale > 1.
    scale_up = coverage_implied_sigma_scale(observed_coverage=0.60, level=0.80)
    assert scale_up > 1.0
    # Observed coverage 0.95 at nominal 0.80 -> too wide -> scale < 1.
    scale_down = coverage_implied_sigma_scale(observed_coverage=0.95, level=0.80)
    assert scale_down < 1.0
    # Calibrated -> ~1.0
    assert coverage_implied_sigma_scale(0.80, 0.80) == pytest.approx(1.0, abs=0.05)
