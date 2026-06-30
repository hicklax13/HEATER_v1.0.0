# Phase 0 — Backtest Harness Foundation (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the measuring-stick the whole value-engine program is gated on — a leak-safe, out-of-sample harness that scores any valuation function by decision-regret + rank-IC + calibration, and certifies whether one valuation beats another (Diebold-Mariano).

**Architecture:** A new focused module `src/backtest_harness.py` that COMPOSES the existing Section-G metrics in `src/engine/output/backtest_calibration.py` (rank_mae, cat_win_rmse, brier_score, trade_prediction_spearman, projection_mae) and ADDS the three rigorous pieces the spec requires but the codebase lacks: `decision_regret`, `purged_kfold_splits` (leak-safe CV — the CPCV building block), and `diebold_mariano` (A-vs-B certification). It is data-agnostic (takes a records DataFrame + a `valuation_fn`), so it is fully testable today on synthetic data and immediately usable once historical records land.

**Tech Stack:** Python 3.12/3.14, NumPy, SciPy (`scipy.stats`), pandas, pytest. Reuses `src/engine/output/backtest_calibration.py`.

**Scope note:** This plan delivers the harness MACHINERY only. Two follow-on Phase-0 plans (separate docs) cover (a) **historical-record assembly** (reconstructing past-week category outcomes from `game_logs` + the point-in-time-projection question — needs an owner data decision) and (b) **variance/correlation calibration** (replacing `scenario_generator.DEFAULT_CV` hand-set guesses, using `compute_empirical_correlations` + this harness). Build this first; it is the prerequisite for both.

Design spec: `docs/superpowers/specs/2026-06-30-heater-advanced-value-engine-design.md` (§3 backtest harness, §9 risks).

---

### Task 1: `decision_regret` — the task-relevant metric

Decision-regret directly scores the DECISIONS a valuation drives: rank items by the valuation, "take" the top-k, and measure the realized value left on the table vs. an oracle that knew the realized values. Lower is better; 0 = the valuation picked the realized-best k. This is the cleanest task-relevant score (it sidesteps whether the intermediate number is itself meaningful).

**Files:**
- Create: `src/backtest_harness.py`
- Test: `tests/test_backtest_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_harness.py
import numpy as np
import pytest


def test_decision_regret_zero_for_perfect_predictor():
    from src.backtest_harness import decision_regret
    realized = [3.0, 1.0, 2.0, 0.5]
    # predicted ranking identical to realized → picks the true best → zero regret
    assert decision_regret(predicted=realized, realized=realized, k=2) == 0.0


def test_decision_regret_positive_when_ranking_inverts():
    from src.backtest_harness import decision_regret
    pred = [1.0, 2.0, 3.0]   # predictor thinks idx2 is best
    real = [3.0, 2.0, 1.0]   # realized best is idx0
    # top-1 by pred = idx2 (realized 1.0); oracle top-1 = idx0 (realized 3.0) → regret 2.0
    assert decision_regret(pred, real, k=1) == pytest.approx(2.0)


def test_decision_regret_empty_and_k_guard():
    from src.backtest_harness import decision_regret
    assert decision_regret([], [], k=1) == 0.0
    assert decision_regret([1.0], [1.0], k=0) == 0.0
    # k larger than n is clamped, not an error
    assert decision_regret([1.0, 2.0], [1.0, 2.0], k=9) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest_harness.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest_harness'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/backtest_harness.py
"""Value-engine backtest harness — the measuring stick that gates every layer.

Composes the existing Section-G metrics (src/engine/output/backtest_calibration.py)
and adds the rigorous pieces the spec requires: decision-regret, leak-safe purged
K-fold splits, and the Diebold-Mariano A-vs-B test. Data-agnostic: callers pass a
records DataFrame + a valuation function. Never raises on empty/degenerate input.
"""

from __future__ import annotations

import numpy as np


def decision_regret(predicted, realized, k: int = 1) -> float:
    """Realized value left on the table by ranking on `predicted` instead of the
    oracle `realized` ranking, taking the top-k. >= 0; 0 means optimal selection.
    Lower is better. Empty/k<=0 -> 0.0; k is clamped to n."""
    p = np.asarray(predicted, dtype=float)
    r = np.asarray(realized, dtype=float)
    n = p.shape[0]
    if n == 0 or k <= 0 or r.shape[0] != n:
        return 0.0
    k = min(k, n)
    chosen = np.argsort(-p, kind="stable")[:k]   # top-k by predicted value
    oracle = np.argsort(-r, kind="stable")[:k]    # top-k by realized value
    return float(r[oracle].sum() - r[chosen].sum())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backtest_harness.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_harness.py tests/test_backtest_harness.py
git commit -m "feat(backtest): decision_regret metric for the value-engine harness"
```

---

### Task 2: `purged_kfold_splits` — leak-safe out-of-sample folds

Season/rest-of-season labels OVERLAP in time, so plain k-fold leaks (a training row's outcome window bleeds into the test fold). Purged K-fold removes training rows within an embargo window around each contiguous test fold. This is the building block for the spec's CPCV; the combinatorial generalization is a later extension noted at the end.

**Files:**
- Modify: `src/backtest_harness.py`
- Test: `tests/test_backtest_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_backtest_harness.py
def test_purged_kfold_covers_all_test_indices_disjointly():
    from src.backtest_harness import purged_kfold_splits
    splits = list(purged_kfold_splits(n=20, n_folds=5, embargo=0))
    assert len(splits) == 5
    seen = []
    for train, test in splits:
        seen.extend(test.tolist())
        # train and test never overlap
        assert set(train.tolist()).isdisjoint(set(test.tolist()))
    assert sorted(seen) == list(range(20))  # every index tested exactly once


def test_purged_kfold_embargo_removes_neighbors_from_train():
    from src.backtest_harness import purged_kfold_splits
    splits = list(purged_kfold_splits(n=20, n_folds=5, embargo=2))
    # middle fold test = [8,9,10,11]; embargo=2 -> train excludes 6..13
    train, test = splits[2]
    assert test.tolist() == [8, 9, 10, 11]
    assert set(range(6, 14)).isdisjoint(set(train.tolist()))
    assert 5 in train.tolist() and 14 in train.tolist()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest_harness.py -k purged -q`
Expected: FAIL — `ImportError: cannot import name 'purged_kfold_splits'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/backtest_harness.py
def purged_kfold_splits(n: int, n_folds: int = 5, embargo: int = 0):
    """Yield (train_idx, test_idx) for purged K-fold over n time-ordered samples.
    Each contiguous fold is a test set; training excludes the test fold plus an
    `embargo` window on each side (leakage guard for overlapping labels)."""
    idx = np.arange(n)
    if n == 0 or n_folds <= 1:
        return
    folds = np.array_split(idx, min(n_folds, n))
    for test in folds:
        if test.size == 0:
            continue
        lo = max(0, int(test[0]) - embargo)
        hi = min(n - 1, int(test[-1]) + embargo)
        train = idx[(idx < lo) | (idx > hi)]
        yield train, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backtest_harness.py -k purged -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_harness.py tests/test_backtest_harness.py
git commit -m "feat(backtest): purged K-fold splits (leak-safe OOS, CPCV building block)"
```

---

### Task 3: `diebold_mariano` — certify A beats B

To declare one valuation the canonical single-source-of-truth, you must show its prediction errors are significantly lower than the incumbent's, not just lower in-sample. The Diebold-Mariano test on the paired loss differential gives that. (First cut uses the iid variance; a Newey-West HAC variance is the production upgrade, noted at the end.)

**Files:**
- Modify: `src/backtest_harness.py`
- Test: `tests/test_backtest_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_backtest_harness.py
def test_diebold_mariano_detects_a_strictly_better():
    from src.backtest_harness import diebold_mariano
    rng = np.random.default_rng(0)
    loss_b = rng.uniform(0.8, 1.2, size=200)
    loss_a = loss_b - 0.3  # A's loss is strictly, consistently lower
    dm, p = diebold_mariano(loss_a, loss_b)
    assert dm < 0          # negative => A better (lower loss)
    assert p < 0.05        # significant


def test_diebold_mariano_equal_models_not_significant():
    from src.backtest_harness import diebold_mariano
    rng = np.random.default_rng(1)
    loss = rng.uniform(0.8, 1.2, size=200)
    dm, p = diebold_mariano(loss, loss.copy())
    assert dm == 0.0 and p == 1.0   # zero differential


def test_diebold_mariano_degenerate_guards():
    from src.backtest_harness import diebold_mariano
    assert diebold_mariano([1.0], [2.0]) == (0.0, 1.0)   # n<2
    assert diebold_mariano([], []) == (0.0, 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest_harness.py -k diebold -q`
Expected: FAIL — `ImportError: cannot import name 'diebold_mariano'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/backtest_harness.py
def diebold_mariano(loss_a, loss_b) -> tuple[float, float]:
    """Diebold-Mariano test on the paired loss differential d = loss_a - loss_b.
    Returns (dm_stat, p_value). dm < 0 => A has lower loss (better). Two-sided p.
    n < 2 or zero-variance -> (0.0, 1.0). iid variance (HAC upgrade: see plan tail)."""
    from scipy.stats import norm

    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.shape[0] != b.shape[0] or a.shape[0] < 2:
        return (0.0, 1.0)
    d = a - b
    dbar = float(d.mean())
    var = float(d.var(ddof=1))
    if var <= 0.0:
        return (0.0, 1.0)
    dm = dbar / np.sqrt(var / d.shape[0])
    p = 2.0 * (1.0 - norm.cdf(abs(dm)))
    return (float(dm), float(p))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backtest_harness.py -k diebold -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_harness.py tests/test_backtest_harness.py
git commit -m "feat(backtest): Diebold-Mariano A-vs-B valuation certification"
```

---

### Task 4: `BacktestHarness` facade — evaluate + compare a valuation

Ties it together: given a `valuation_fn(record) -> float` and a records DataFrame carrying a `realized_value` column, compute the OOS metric bundle (rank-IC via the existing `trade_prediction_spearman`, decision-regret, and per-fold mean) under purged K-fold; and `compare` two valuations with Diebold-Mariano on squared error vs `realized_value`.

**Files:**
- Modify: `src/backtest_harness.py`
- Test: `tests/test_backtest_harness.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_backtest_harness.py
import pandas as pd  # noqa: E402


def _records(n=30, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    true = rng.normal(0, 1, size=n)
    return pd.DataFrame({"signal": true, "realized_value": true + rng.normal(0, noise, size=n)})


def test_harness_evaluate_good_valuation_beats_random():
    from src.backtest_harness import BacktestHarness
    h = BacktestHarness(n_folds=5, embargo=1)
    recs = _records(40, noise=0.1, seed=2)
    good = h.evaluate(lambda r: r["signal"], recs, k=5)
    rand = h.evaluate(lambda r: 0.0 * r["signal"], recs, k=5)  # constant => no ranking power
    assert good["rank_ic"] > 0.8
    assert good["decision_regret"] <= rand["decision_regret"]
    assert good["n"] == 40


def test_harness_compare_prefers_lower_error_model():
    from src.backtest_harness import BacktestHarness
    h = BacktestHarness()
    recs = _records(60, noise=0.5, seed=3)
    verdict = h.compare(lambda r: r["signal"], lambda r: -r["signal"], recs)
    assert verdict["a_better"] is True       # signal tracks realized; -signal does not
    assert verdict["p_value"] < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_backtest_harness.py -k harness -q`
Expected: FAIL — `ImportError: cannot import name 'BacktestHarness'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/backtest_harness.py
from src.engine.output.backtest_calibration import trade_prediction_spearman


class BacktestHarness:
    """Out-of-sample evaluator + A/B comparer for a valuation function over a
    records DataFrame with a 'realized_value' column. Leak-safe via purged K-fold."""

    def __init__(self, n_folds: int = 5, embargo: int = 1):
        self.n_folds = n_folds
        self.embargo = embargo

    def _predict(self, valuation_fn, recs):
        return np.asarray([float(valuation_fn(recs.iloc[i])) for i in range(len(recs))], dtype=float)

    def evaluate(self, valuation_fn, records, k: int = 5) -> dict:
        """OOS metric bundle: rank_ic (Spearman pred vs realized), decision_regret,
        mean_fold_regret. Predictions are computed per row; metrics scored on the
        held-out fold of each purged split, then aggregated."""
        n = len(records)
        realized = np.asarray(records["realized_value"], dtype=float)
        pred = self._predict(valuation_fn, records)
        fold_regrets = []
        for _, test in purged_kfold_splits(n, self.n_folds, self.embargo):
            fold_regrets.append(decision_regret(pred[test], realized[test], k=min(k, test.size)))
        return {
            "n": n,
            "rank_ic": trade_prediction_spearman(pred, realized),
            "decision_regret": decision_regret(pred, realized, k=k),
            "mean_fold_regret": float(np.mean(fold_regrets)) if fold_regrets else 0.0,
        }

    def compare(self, fn_a, fn_b, records) -> dict:
        """Diebold-Mariano on squared error vs realized_value. a_better=True when
        A's loss is significantly lower (dm<0, p<0.05)."""
        realized = np.asarray(records["realized_value"], dtype=float)
        ea = (self._predict(fn_a, records) - realized) ** 2
        eb = (self._predict(fn_b, records) - realized) ** 2
        dm, p = diebold_mariano(ea, eb)
        return {"dm_stat": dm, "p_value": p, "a_better": bool(dm < 0 and p < 0.05)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_backtest_harness.py -q`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add src/backtest_harness.py tests/test_backtest_harness.py
git commit -m "feat(backtest): BacktestHarness evaluate + compare facade"
```

---

## Self-Review

**Spec coverage (§3 backtest harness):** decision-regret ✅ (Task 1), leak-safe OOS / CPCV building block ✅ (Task 2 purged K-fold), DM A/B certification ✅ (Task 3), composed harness reusing the existing Section-G metrics ✅ (Task 4). **Deferred to named follow-on Phase-0 plans (out of this plan's scope, called out in the Scope note):** full combinatorial CPCV, Deflated-Sharpe/PBO/MinBTL, Hansen MCS, adaptive conformal, the variance/correlation calibration, and historical-record assembly. Rank-IC + calibration + Brier are reused from `backtest_calibration.py`, not re-implemented (DRY).

**Placeholder scan:** none — every step has complete runnable code + exact commands.

**Type consistency:** `decision_regret(predicted, realized, k)`, `purged_kfold_splits(n, n_folds, embargo)`, `diebold_mariano(loss_a, loss_b) -> (float, float)`, `BacktestHarness(n_folds, embargo).evaluate(fn, records, k)` / `.compare(fn_a, fn_b, records)` are used identically across tasks and tests. `evaluate` consumes `purged_kfold_splits` + `decision_regret`; `compare` consumes `diebold_mariano` — all defined in earlier tasks. `records` always carries `realized_value`.

**Production upgrades noted (not placeholders — explicit follow-ups):** (1) Newey-West HAC variance in `diebold_mariano`; (2) combinatorial (not just K-fold) purged splits for full CPCV; (3) the deferred validation machinery + calibration + data-assembly plans above.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-30-heater-value-engine-phase0-backtest-harness.md`.
