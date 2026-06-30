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
