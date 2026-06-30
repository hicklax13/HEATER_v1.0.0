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
    chosen = np.argsort(-p, kind="stable")[:k]
    oracle = np.argsort(-r, kind="stable")[:k]
    return float(r[oracle].sum() - r[chosen].sum())


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


def diebold_mariano(loss_a, loss_b) -> tuple[float, float]:
    """Diebold-Mariano test on the paired loss differential d = loss_a - loss_b.
    Returns (dm_stat, p_value). dm < 0 => A has lower loss (better). Two-sided p.
    n < 2 or zero-variance -> (0.0, 1.0). iid variance (HAC upgrade: future work)."""
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


from src.engine.output.backtest_calibration import trade_prediction_spearman  # noqa: E402


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
        mean_fold_regret."""
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
