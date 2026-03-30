"""Tests for trade signal integration — Kalman + regime adjustments."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.trade_signals import compute_trend_adjustment


def test_hot_streak_positive_adjustment():
    observations = np.array([0.250, 0.270, 0.290, 0.310])
    adj = compute_trend_adjustment(observations, prior_mean=0.260)
    assert adj > 0


def test_cold_streak_negative_adjustment():
    observations = np.array([0.310, 0.290, 0.270, 0.250])
    adj = compute_trend_adjustment(observations, prior_mean=0.280)
    assert adj < 0


def test_stable_near_zero():
    observations = np.array([0.270, 0.268, 0.272, 0.269])
    adj = compute_trend_adjustment(observations, prior_mean=0.270)
    assert abs(adj) < 5.0


def test_too_few_observations_returns_zero():
    observations = np.array([0.300])
    adj = compute_trend_adjustment(observations, prior_mean=0.280)
    assert adj == 0.0


def test_adjustment_bounded():
    observations = np.array([0.200, 0.250, 0.300, 0.350, 0.400])
    adj = compute_trend_adjustment(observations, prior_mean=0.250)
    assert -15 <= adj <= 15
