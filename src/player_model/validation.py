"""Layer-0 validation machinery — the Phase-1 gate's measuring tools.

Scores a set of (predicted distribution, realized outcome) records by point accuracy (MAE),
interval coverage, PIT calibration, and a Diebold-Mariano model-vs-baseline test (reusing
src/backtest_harness.py). The Phase-1 gate: the player model's MAE is NO WORSE than the pool
baseline AND its posterior intervals are calibrated (coverage ~ nominal) — i.e. it adds honest
uncertainty without degrading the point projection.

DB-free, network-free, pure NumPy/SciPy — unit-testable on synthetic data. The real-data gate
run lives in scripts/validate_player_model.py (it hits MLB-StatsAPI, which the test network
guard blocks). Never raises on empty/degenerate input.
"""

from __future__ import annotations

import numpy as np


def _arrays(*cols):
    return tuple(np.asarray(c, dtype=float) for c in cols)


def mean_absolute_error(pred, realized) -> float:
    """Mean |pred - realized|. Empty / length-mismatch -> 0.0."""
    p, r = _arrays(pred, realized)
    if p.shape[0] == 0 or p.shape[0] != r.shape[0]:
        return 0.0
    return float(np.mean(np.abs(p - r)))


def gaussian_interval_coverage(mean, sigma, realized, level: float = 0.80) -> float:
    """Fraction of realized values inside the central `level` Gaussian interval mean +/- z*sigma.
    Well-calibrated => ~level. Empty / non-positive sigma rows are skipped. Returns 0.0 if none."""
    from scipy.stats import norm

    m, s, r = _arrays(mean, sigma, realized)
    if m.shape[0] == 0 or not (m.shape[0] == s.shape[0] == r.shape[0]):
        return 0.0
    valid = np.isfinite(m) & np.isfinite(s) & np.isfinite(r) & (s > 0)
    if not valid.any():
        return 0.0
    z = float(norm.ppf(0.5 + level / 2.0))
    lo = m[valid] - z * s[valid]
    hi = m[valid] + z * s[valid]
    inside = (r[valid] >= lo) & (r[valid] <= hi)
    return float(np.mean(inside))


def pit_values(mean, sigma, realized) -> np.ndarray:
    """Probability-integral-transform Phi((realized - mean)/sigma). Uniform[0,1] iff calibrated.
    Non-positive-sigma / non-finite rows are dropped. Empty -> empty array."""
    from scipy.stats import norm

    m, s, r = _arrays(mean, sigma, realized)
    if m.shape[0] == 0 or not (m.shape[0] == s.shape[0] == r.shape[0]):
        return np.asarray([], dtype=float)
    valid = np.isfinite(m) & np.isfinite(s) & np.isfinite(r) & (s > 0)
    return np.asarray(norm.cdf((r[valid] - m[valid]) / s[valid]), dtype=float)


def conformal_quantile(residuals, level: float = 0.80) -> float:
    """Split-conformal half-width: the (ceil((n+1)*level)/n) empirical quantile of |residuals|.
    Distribution-free — an interval mean +/- this value covers >= level out-of-sample. Empty -> 0.0."""
    r = np.abs(np.asarray(residuals, dtype=float))
    r = r[np.isfinite(r)]
    n = r.shape[0]
    if n == 0:
        return 0.0
    # finite-sample conformal correction
    rank = min(1.0, np.ceil((n + 1) * level) / n)
    return float(np.quantile(r, rank, method="higher"))


def coverage_implied_sigma_scale(observed_coverage: float, level: float = 0.80) -> float:
    """Multiplicative sigma correction implied by an observed coverage vs the nominal `level`.
    If intervals under-cover (observed < level), returns > 1 (widen sigma); over-cover -> < 1.
    Derived from the Gaussian z ratio z(observed)/z(level). The one-step bridge to the full
    variance-calibration follow-on. Clamped to [0.25, 4.0]; degenerate -> 1.0."""
    from scipy.stats import norm

    oc = float(observed_coverage)
    lv = float(level)
    if not (0.0 < oc < 1.0) or not (0.0 < lv < 1.0):
        return 1.0
    z_obs = norm.ppf(0.5 + oc / 2.0)
    z_lvl = norm.ppf(0.5 + lv / 2.0)
    if z_obs <= 0:
        return 4.0
    return float(min(4.0, max(0.25, z_lvl / z_obs)))


def validate_layer0(
    records,
    *,
    mean_col: str,
    sigma_col: str,
    realized_col: str,
    baseline_col: str | None = None,
    level: float = 0.80,
    coverage_tol: float = 0.05,
) -> dict:
    """The Phase-1 gate verdict for a records DataFrame. Returns MAE (model + baseline),
    interval coverage at `level`, PIT-mean, the coverage-implied sigma scale, and (when a
    baseline column is given) the Diebold-Mariano model-vs-baseline squared-error verdict.
    `passes_gate` = coverage within `coverage_tol` of nominal AND the model does not lose the
    MAE comparison (baseline NOT significantly better). Never raises."""
    from src.backtest_harness import diebold_mariano

    mean = records[mean_col].to_numpy(dtype=float)
    sigma = records[sigma_col].to_numpy(dtype=float)
    realized = records[realized_col].to_numpy(dtype=float)

    mae_model = mean_absolute_error(mean, realized)
    coverage = gaussian_interval_coverage(mean, sigma, realized, level=level)
    pit = pit_values(mean, sigma, realized)
    pit_mean = float(pit.mean()) if pit.size else 0.5
    sigma_scale = coverage_implied_sigma_scale(coverage, level)

    mae_baseline = None
    model_beats_baseline = None
    baseline_significantly_better = False
    if baseline_col is not None:
        base = records[baseline_col].to_numpy(dtype=float)
        mae_baseline = mean_absolute_error(base, realized)
        model_beats_baseline = bool(mae_model < mae_baseline)
        # DM on squared error: dm<0 => model (A) lower loss. baseline better == dm>0 & significant.
        err_model = (mean - realized) ** 2
        err_base = (base - realized) ** 2
        dm, p = diebold_mariano(err_model, err_base)
        baseline_significantly_better = bool(dm > 0 and p < 0.05)

    coverage_ok = abs(coverage - level) <= coverage_tol
    passes_gate = bool(coverage_ok and not baseline_significantly_better)

    return {
        "n": int(len(records)),
        "mae_model": mae_model,
        "mae_baseline": mae_baseline,
        "model_beats_baseline_mae": model_beats_baseline,
        "coverage": coverage,
        "coverage_target": level,
        "pit_mean": pit_mean,
        "sigma_scale_hint": sigma_scale,
        "passes_gate": passes_gate,
    }
