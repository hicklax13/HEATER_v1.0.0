# Phase 1 Slice 5 — Layer-0 Validation Gate (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the validation that gates Phase 1 — prove the Layer-0 player model's projection MAE is **no worse** than the current pool baseline AND its posterior intervals are **calibrated** (realized outcomes fall inside the X% interval ≈ X% of the time). Because the model reuses the pool's mean (per the Phase-1 scope decision), the honest, achievable bar is: *adds calibrated, honest uncertainty WITHOUT degrading the point projection* — measured out-of-sample on game-log backtests via the Phase-0 harness.

**Architecture:** A new DB-free, network-free validation MACHINERY module `src/player_model/validation.py` (interval coverage, PIT calibration, MAE, split-conformal width, and a `validate_layer0` facade that reuses `src/backtest_harness.py`'s Diebold-Mariano for the model-vs-baseline comparison) — fully unit-testable on synthetic data today. PLUS a runnable real-data gate script `scripts/validate_player_model.py` that assembles historical weekly game-log records (via `src/optimizer/backtest_runner.py`) over the slice-4 `build_player_models` output and prints the gate verdict. This mirrors the Phase-0 harness pattern exactly (machinery DB-free + synthetic-tested; real-data assembly in a manual script, since `backtest_runner` hits MLB-StatsAPI which the test network-guard blocks).

**Tech Stack:** Python 3.12/3.14, NumPy, SciPy (`scipy.stats.norm`), pandas, pytest. Reuses `src/backtest_harness.py` (`BacktestHarness`, `diebold_mariano`), `src/player_model/` (slice 4 facade), `src/optimizer/backtest_runner.py` (game-log data), `src/database.py` (`load_player_pool`).

**Spec:** `docs/superpowers/specs/2026-06-30-heater-advanced-value-engine-design.md` §3 (backtest harness — "Adaptive/weighted Conformal intervals; rank by proper score; reliability/coverage on realized weekly category outcomes") + §4 phase table (Phase 1 gate: "projection MAE + coverage beat the current pool") + research-validation Risk 1 / open-item #1 (calibrate variance, validate by reliability/coverage).

**Scope note:** This slice delivers the validation MACHINERY + the runnable gate. It does NOT perform the full σ/ρ calibration LOOP (adjusting the slice-1 seeds to optimize OOS coverage — that is the highest-value FOLLOW-ON that *uses* this machinery) and does NOT add adaptive/weighted conformal (a split-conformal width is provided; the adaptive variant is a follow-on). A `coverage_implied_sigma_scale` helper gives the one-step calibration bridge. The real-data gate run (network + DB) is a manual operator step, not CI.

---

## File Structure

- **Create `src/player_model/validation.py`** — `mean_absolute_error`, `gaussian_interval_coverage`, `pit_values`, `conformal_quantile`, `coverage_implied_sigma_scale`, `validate_layer0`. One responsibility: score a set of (predicted distribution, realized outcome) records.
- **Create `scripts/validate_player_model.py`** — the runnable real-data gate (assembles game-logs, runs `validate_layer0`, prints the verdict). Network/DB; manual.
- **Create `tests/test_player_model_validation.py`** — TDD tests (synthetic, DB-free, network-free).

---

### Task 1: Coverage + MAE + PIT calibration primitives

The core diagnostics: MAE (point accuracy), interval coverage (does the X% Gaussian interval contain the realized value X% of the time?), and the probability-integral-transform (PIT — uniform[0,1] iff perfectly calibrated).

**Files:**
- Create: `src/player_model/validation.py`
- Test: `tests/test_player_model_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_player_model_validation.py
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
    realized = rng.normal(mean, sigma)        # noise EXACTLY matches sigma
    cov = gaussian_interval_coverage(mean, sigma, realized, level=0.80)
    assert cov == pytest.approx(0.80, abs=0.02)   # calibrated -> ~80% inside the 80% interval


def test_gaussian_coverage_overconfident_under_covers():
    from src.player_model.validation import gaussian_interval_coverage
    rng = np.random.default_rng(1)
    n = 20000
    realized = rng.normal(0, 2.0, size=n)     # true noise 2.0 ...
    sigma = np.full(n, 1.0)                    # ... but model claims 1.0 (overconfident)
    cov = gaussian_interval_coverage(np.zeros(n), sigma, realized, level=0.80)
    assert cov < 0.80                          # too-narrow intervals miss too often


def test_pit_calibrated_is_uniform_mean_half():
    from src.player_model.validation import pit_values
    rng = np.random.default_rng(2)
    n = 20000
    realized = rng.normal(0, 1, size=n)
    pit = pit_values(np.zeros(n), np.ones(n), realized)
    assert pit.min() >= 0.0 and pit.max() <= 1.0
    assert pit.mean() == pytest.approx(0.5, abs=0.02)   # uniform -> mean 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_validation.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.player_model.validation'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/player_model/validation.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_validation.py -q`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/validation.py tests/test_player_model_validation.py
git commit -m "feat(player_model): coverage + MAE + PIT validation primitives (P1 slice5)"
```

---

### Task 2: split-conformal width + `coverage_implied_sigma_scale` calibration bridge

Two calibration tools: `conformal_quantile` (the distribution-free half-width from a residual set — a baseball-robust alternative to the Gaussian assumption) and `coverage_implied_sigma_scale` (the multiplicative σ-correction implied by observed coverage — the one-step bridge to the σ/ρ calibration follow-on).

**Files:**
- Modify: `src/player_model/validation.py`
- Test: `tests/test_player_model_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_validation.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_validation.py -k "conformal or sigma_scale" -q`
Expected: FAIL — `ImportError: cannot import name 'conformal_quantile'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/validation.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_validation.py -k "conformal or sigma_scale" -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/validation.py tests/test_player_model_validation.py
git commit -m "feat(player_model): conformal width + coverage-implied sigma-scale bridge (P1 slice5)"
```

---

### Task 3: `validate_layer0` — the gate facade (MAE-vs-baseline DM + coverage + calibration)

Tie it together: given a records DataFrame with model predicted mean/sigma, a baseline prediction, and the realized outcome, return the gate verdict — MAE (model + baseline), coverage at the level, PIT-mean calibration, and the Diebold-Mariano model-vs-baseline test (reusing `BacktestHarness`). `passes_gate` is True when the model does NOT degrade MAE (DM not significantly worse) AND coverage is within tolerance of nominal.

**Files:**
- Modify: `src/player_model/validation.py`
- Test: `tests/test_player_model_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_validation.py
import pandas as pd  # noqa: E402


def _records(n=400, model_sigma=1.0, true_noise=1.0, baseline_bias=0.0, seed=0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1, size=n)
    realized = truth + rng.normal(0, true_noise, size=n)
    return pd.DataFrame({
        "model_mean": truth,
        "model_sigma": np.full(n, model_sigma),
        "baseline_pred": truth + baseline_bias,   # baseline = same mean unless biased
        "realized": realized,
    })


def test_validate_layer0_calibrated_model_passes_gate():
    from src.player_model.validation import validate_layer0
    recs = _records(n=4000, model_sigma=1.0, true_noise=1.0, seed=1)   # sigma matches noise
    out = validate_layer0(recs, mean_col="model_mean", sigma_col="model_sigma",
                          baseline_col="baseline_pred", realized_col="realized", level=0.80)
    assert out["coverage"] == pytest.approx(0.80, abs=0.04)
    assert out["mae_model"] == pytest.approx(out["mae_baseline"], abs=1e-9)  # same mean -> same MAE
    assert out["passes_gate"] is True


def test_validate_layer0_flags_overconfident_model():
    from src.player_model.validation import validate_layer0
    recs = _records(n=4000, model_sigma=0.3, true_noise=1.0, seed=2)   # sigma too small
    out = validate_layer0(recs, mean_col="model_mean", sigma_col="model_sigma",
                          baseline_col="baseline_pred", realized_col="realized", level=0.80)
    assert out["coverage"] < 0.70                # under-covers
    assert out["sigma_scale_hint"] > 1.0         # calibration says "widen"
    assert out["passes_gate"] is False           # miscalibrated -> gate fails


def test_validate_layer0_detects_worse_baseline():
    from src.player_model.validation import validate_layer0
    # Baseline is biased -> model MAE strictly lower -> model "a_better" on the DM test.
    recs = _records(n=4000, model_sigma=1.0, true_noise=1.0, baseline_bias=1.5, seed=3)
    out = validate_layer0(recs, mean_col="model_mean", sigma_col="model_sigma",
                          baseline_col="baseline_pred", realized_col="realized", level=0.80)
    assert out["mae_model"] < out["mae_baseline"]
    assert out["model_beats_baseline_mae"] is True


def test_validate_layer0_no_baseline_still_reports_coverage():
    from src.player_model.validation import validate_layer0
    recs = _records(n=2000, model_sigma=1.0, true_noise=1.0, seed=4)
    out = validate_layer0(recs, mean_col="model_mean", sigma_col="model_sigma",
                          baseline_col=None, realized_col="realized", level=0.80)
    assert "coverage" in out and out["mae_baseline"] is None
    assert out["model_beats_baseline_mae"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_validation.py -k validate_layer0 -q`
Expected: FAIL — `ImportError: cannot import name 'validate_layer0'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/player_model/validation.py
def validate_layer0(records, *, mean_col: str, sigma_col: str, realized_col: str,
                    baseline_col: str | None = None, level: float = 0.80,
                    coverage_tol: float = 0.05) -> dict:
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_player_model_validation.py -q`
Expected: PASS (all tests)

- [ ] **Step 5: Commit**

```bash
git add src/player_model/validation.py tests/test_player_model_validation.py
git commit -m "feat(player_model): validate_layer0 gate facade (DM + coverage + calibration) (P1 slice5)"
```

---

### Task 4: `scripts/validate_player_model.py` — the runnable real-data gate

A manual operator script: load the pool, build the Layer-0 models (slice 4), assemble historical weekly game-log records via `backtest_runner`, scale each model's posterior to the week, build the records DataFrame, run `validate_layer0`, and print the verdict. Network + DB; not CI. Guarded by a `__main__` block and a tiny importable smoke test.

**Files:**
- Create: `scripts/validate_player_model.py`
- Test: `tests/test_player_model_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_player_model_validation.py
def test_build_records_from_predictions_shape():
    # The script's pure record-builder must be importable + DB-free for the per-week join.
    from scripts.validate_player_model import build_records
    import numpy as np
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
    actuals = {1: 1.3}   # player 2 has no realized outcome this week
    df = build_records(preds, actuals)
    assert len(df) == 1 and df.iloc[0]["player_id"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_player_model_validation.py -k build_records -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.validate_player_model'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/validate_player_model.py
"""Real-data Phase-1 gate run for the Layer-0 player model.

Assembles historical weekly game-log records (predicted posterior vs realized outcome) and runs
src.player_model.validation.validate_layer0 to print the gate verdict: does the model's MAE match
the pool baseline AND are its posterior intervals calibrated?

Network + DB (MLB-StatsAPI via src/optimizer/backtest_runner.py) -> run MANUALLY, not in CI:
    python -m scripts.validate_player_model --weeks 6 --category HR

`build_records` is a pure, DB-free join (unit-tested); the data assembly around it needs the live DB.
"""

from __future__ import annotations

import argparse

import pandas as pd


def build_records(predictions: dict[int, dict], actuals: dict[int, float]) -> pd.DataFrame:
    """Join per-player predicted posterior {mean, sigma, baseline} with realized outcomes into the
    validate_layer0 records frame. Players without a realized outcome are skipped. Pure / DB-free."""
    rows = []
    for pid, pred in predictions.items():
        if pid not in actuals:
            continue
        rows.append({
            "player_id": int(pid),
            "model_mean": float(pred.get("mean", 0.0)),
            "model_sigma": float(pred.get("sigma", 0.0)),
            "baseline_pred": float(pred.get("baseline", pred.get("mean", 0.0))),
            "realized": float(actuals[pid]),
        })
    return pd.DataFrame(rows, columns=["player_id", "model_mean", "model_sigma", "baseline_pred", "realized"])


def run_gate(weeks: int, category: str) -> dict:  # pragma: no cover - needs live DB + network
    """Assemble real game-log records for `category` over the last `weeks` and run the gate.
    Imports the heavy deps lazily so the module + build_records stay import-light for tests."""
    import math

    from src.database import load_player_pool
    from src.player_model import build_player_models
    from src.player_model.validation import validate_layer0
    from src.optimizer.backtest_runner import run_backtest  # noqa: F401  (assembly hook)
    from src.valuation import LeagueConfig

    cfg = LeagueConfig()
    pool = load_player_pool()
    models = build_player_models(pool, cfg)

    # Predicted per-week posterior for the category: mean + sigma=sqrt(sigma2 + tau2);
    # baseline = the pool's own per-week projection (the incumbent we must not degrade).
    predictions: dict[int, dict] = {}
    for pid, pm in models.items():
        post = pm.posteriors.get(category)
        if post is None:
            continue
        predictions[pid] = {
            "mean": post.mean,
            "sigma": math.sqrt(max(post.sigma2 + post.tau2, 0.0)),
            "baseline": post.mean,  # reuse-the-mean: baseline == model mean this phase
        }

    # NOTE: assembling `actuals` (realized weekly category totals per player) from game logs uses
    # src.optimizer.backtest_runner.run_backtest / statsapi over the chosen weeks. Operator wires the
    # specific weeks here; left as the integration point (needs the live DB + network).
    actuals: dict[int, float] = {}
    records = build_records(predictions, actuals)
    return validate_layer0(records, mean_col="model_mean", sigma_col="model_sigma",
                           baseline_col="baseline_pred", realized_col="realized")


def main() -> None:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description="Layer-0 player-model Phase-1 gate (real data).")
    parser.add_argument("--weeks", type=int, default=6, help="historical weeks to backtest")
    parser.add_argument("--category", type=str, default="HR", help="scoring category to validate")
    args = parser.parse_args()
    verdict = run_gate(args.weeks, args.category)
    print("Layer-0 gate verdict:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print("PASS" if verdict.get("passes_gate") else "FAIL (see coverage / sigma_scale_hint)")


if __name__ == "__main__":  # pragma: no cover
    main()
```

- [ ] **Step 4: Run test to verify it passes + full slice suite + lint**

Run: `python -m pytest tests/test_player_model_validation.py -q`
Expected: PASS (all tests)

Run: `python -m pytest tests/test_player_model_posterior.py tests/test_player_model_availability.py tests/test_player_model_gscore.py tests/test_player_model_facade.py tests/test_player_model_validation.py -q`
Expected: PASS (the whole Layer-0 suite together)

Run: `python -m ruff check src/player_model/ scripts/validate_player_model.py tests/test_player_model_validation.py && python -m ruff format src/player_model/validation.py scripts/validate_player_model.py tests/test_player_model_validation.py`
Expected: no lint errors; format clean.

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_player_model.py tests/test_player_model_validation.py
git commit -m "feat(player_model): runnable real-data gate script + build_records (P1 slice5)"
```

---

## Self-Review

**Spec coverage (§3 harness + §4 Phase-1 gate):**
- "projection MAE + coverage beat the current pool" → `validate_layer0` reports `mae_model` vs `mae_baseline` (DM-tested, not degraded) + `coverage` (calibrated), with an explicit `passes_gate`. The honest framing (reuse-the-mean ⇒ tie on MAE, win on calibration) is encoded: gate passes on calibrated coverage + non-degraded MAE. ✅
- "reliability/coverage on realized weekly category outcomes" → `gaussian_interval_coverage` + `pit_values` (Task 1). ✅
- "rank by proper score; conformal intervals" → `conformal_quantile` split-conformal width (Task 2); PIT calibration as the reliability diagnostic. Adaptive/weighted conformal = named follow-on. ✅
- "Calibrate every per-category variance" → `coverage_implied_sigma_scale` is the one-step bridge; the full SURE calibration LOOP is the named highest-value follow-on (uses this machinery). ✅
- Reuses the Phase-0 harness (`diebold_mariano`) rather than re-implementing — DRY. ✅
- **Deferred (named follow-ons):** the σ/ρ calibration loop; adaptive/weighted conformal; the per-category correlation-matrix calibration; full game-log `actuals` assembly in `run_gate` (the operator integration point).

**Placeholder scan:** `run_gate`/`main` are `# pragma: no cover` real-data entry points (network+DB) with a documented `actuals`-assembly integration point — NOT a code placeholder in the tested surface; `build_records` (the pure join) is fully implemented + unit-tested. Every other step has complete runnable code + exact commands.

**Type consistency:** `mean_absolute_error(pred, realized)`, `gaussian_interval_coverage(mean, sigma, realized, level)`, `pit_values(mean, sigma, realized)`, `conformal_quantile(residuals, level)`, `coverage_implied_sigma_scale(observed_coverage, level)`, `validate_layer0(records, *, mean_col, sigma_col, realized_col, baseline_col, level, coverage_tol)`, `build_records(predictions, actuals)` — used identically across tasks/tests. `validate_layer0` consumes `backtest_harness.diebold_mariano` with its `(loss_a, loss_b) -> (dm, p)` contract (verified against slice 0).

**Reuse (DRY):** Phase-0 `BacktestHarness`/`diebold_mariano`; slice-4 `build_player_models`; `backtest_runner.run_backtest` for game-logs; `LeagueConfig`. No re-implementation.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-heater-value-engine-phase1-slice5-validation-gate.md`.
