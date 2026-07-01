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


import pandas as pd  # noqa: E402


def _records(n=400, model_sigma=1.0, true_noise=1.0, baseline_bias=0.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1, size=n)
    realized = truth + rng.normal(0, true_noise, size=n)
    return pd.DataFrame(
        {
            "model_mean": truth,
            "model_sigma": np.full(n, model_sigma),
            "baseline_pred": truth + baseline_bias,  # baseline = same mean unless biased
            "realized": realized,
        }
    )


def test_validate_layer0_calibrated_model_passes_gate():
    from src.player_model.validation import validate_layer0

    recs = _records(n=4000, model_sigma=1.0, true_noise=1.0, seed=1)  # sigma matches noise
    out = validate_layer0(
        recs,
        mean_col="model_mean",
        sigma_col="model_sigma",
        baseline_col="baseline_pred",
        realized_col="realized",
        level=0.80,
    )
    assert out["coverage"] == pytest.approx(0.80, abs=0.04)
    assert out["mae_model"] == pytest.approx(out["mae_baseline"], abs=1e-9)  # same mean -> same MAE
    assert out["passes_gate"] is True


def test_validate_layer0_flags_overconfident_model():
    from src.player_model.validation import validate_layer0

    recs = _records(n=4000, model_sigma=0.3, true_noise=1.0, seed=2)  # sigma too small
    out = validate_layer0(
        recs,
        mean_col="model_mean",
        sigma_col="model_sigma",
        baseline_col="baseline_pred",
        realized_col="realized",
        level=0.80,
    )
    assert out["coverage"] < 0.70  # under-covers
    assert out["sigma_scale_hint"] > 1.0  # calibration says "widen"
    assert out["passes_gate"] is False  # miscalibrated -> gate fails


def test_validate_layer0_detects_worse_baseline():
    from src.player_model.validation import validate_layer0

    # Baseline is biased -> model MAE strictly lower -> model "a_better" on the DM test.
    recs = _records(n=4000, model_sigma=1.0, true_noise=1.0, baseline_bias=1.5, seed=3)
    out = validate_layer0(
        recs,
        mean_col="model_mean",
        sigma_col="model_sigma",
        baseline_col="baseline_pred",
        realized_col="realized",
        level=0.80,
    )
    assert out["mae_model"] < out["mae_baseline"]
    assert out["model_beats_baseline_mae"] is True


def test_validate_layer0_no_baseline_still_reports_coverage():
    from src.player_model.validation import validate_layer0

    recs = _records(n=2000, model_sigma=1.0, true_noise=1.0, seed=4)
    out = validate_layer0(
        recs, mean_col="model_mean", sigma_col="model_sigma", baseline_col=None, realized_col="realized", level=0.80
    )
    assert "coverage" in out and out["mae_baseline"] is None
    assert out["model_beats_baseline_mae"] is None


def test_build_records_from_predictions_shape():
    # The script's pure record-builder must be importable + DB-free for the per-week join.
    import numpy as np

    from scripts.validate_player_model import build_records

    preds = {
        1: {"mean": 1.2, "sigma": 0.4, "baseline": 1.1},
        2: {"mean": 0.8, "sigma": 0.3, "baseline": 0.9},
    }
    actuals = {1: 1.5, 2: 0.6}
    df = build_records(preds, actuals)
    assert list(df.columns) == ["player_id", "model_mean", "model_sigma", "baseline_pred", "realized"]
    assert len(df) == 2
    assert set(df["player_id"]) == {1, 2}


def test_build_records_skips_players_without_actuals():
    from scripts.validate_player_model import build_records

    preds = {1: {"mean": 1.0, "sigma": 0.2, "baseline": 1.0}, 2: {"mean": 1.0, "sigma": 0.2, "baseline": 1.0}}
    actuals = {1: 1.3}  # player 2 has no realized outcome this week
    df = build_records(preds, actuals)
    assert len(df) == 1 and df.iloc[0]["player_id"] == 1
