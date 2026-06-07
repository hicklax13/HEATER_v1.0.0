"""LO-E3 / BR-6: lineup optimizer H2H win-prob on Skellam + Gaussian copula.

The lineup optimizer's ``estimate_h2h_win_probability`` used a per-category
NORMAL approximation AND treated the 12 categories as INDEPENDENT (a Normal
approx of the count of category wins). That made its overall weekly win-prob
~3.4x more extreme than the Matchup Planner's Gaussian-copula simulation for
the SAME matchup (8.5% vs 29% live). The codebase already ships both better
models — this brings the lineup engine onto them:

  * low-count COUNTING cats (SB, SV, W, L) -> the trade engine's Skellam helper
    (``src.engine.output.weekly_matrix._category_win_prob_skellam``), which
    captures the right-skew the Gaussian approx misses;
  * the OVERALL win-prob -> a CORRELATED joint draw via the shared
    ``src.engine.portfolio.copula.GaussianCopula`` / ``DEFAULT_CORRELATION``
    (the same correlation structure the Matchup Planner / trade engine use),
    counting category wins per draw and taking P(win majority) — instead of
    summing INDEPENDENT per-category Bernoullis.

These tests guard:
  (a) the overall win-prob matches an independent copula recomputation and the
      correlated estimate is materially LESS over-confident than the legacy
      independent-Bernoulli result for a slightly-behind matchup;
  (b) Skellam dispatch: a low-count cat (SB) uses Skellam (differs from the
      Normal-CV value and captures right-skew), a rate cat (AVG) uses Normal;
  (c) inverse cats (ERA/WHIP/L) still flip (lower is better).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm

from src.engine.output.weekly_matrix import _category_win_prob_skellam
from src.engine.portfolio.copula import (
    CAT_ORDER,
    DEFAULT_CORRELATION,
    GaussianCopula,
)
from src.optimizer.h2h_engine import (
    INVERSE_CATS,
    SKELLAM_CATS,
    default_category_variances,
    estimate_h2h_win_probability,
)

# ── Shared fixtures ───────────────────────────────────────────────────


def _slightly_behind() -> tuple[dict[str, float], dict[str, float]]:
    """A realistic matchup where the user trails modestly in most cats.

    This is the regime where the old independent-Bernoulli overall estimate
    was over-confident (pushed toward 0), diverging ~3.4x from the copula
    estimate (BR-6).
    """
    my = {
        "r": 78.0,
        "hr": 19.0,
        "rbi": 74.0,
        "sb": 9.0,
        "avg": 0.262,
        "obp": 0.330,
        "w": 7.0,
        "l": 6.0,
        "sv": 5.0,
        "k": 68.0,
        "era": 3.95,
        "whip": 1.24,
    }
    opp = {
        "r": 82.0,
        "hr": 21.0,
        "rbi": 78.0,
        "sb": 11.0,
        "avg": 0.268,
        "obp": 0.338,
        "w": 8.0,
        "l": 5.0,
        "sv": 6.0,
        "k": 72.0,
        "era": 3.80,
        "whip": 1.20,
    }
    return my, opp


def _independent_overall(per_category: dict[str, float]) -> float:
    """Legacy independent-Bernoulli Normal approximation of P(win majority)."""
    n = len(per_category)
    ew = sum(per_category.values())
    var = sum(p * (1.0 - p) for p in per_category.values())
    sd = math.sqrt(var) if var > 1e-12 else 1e-12
    return float(norm.cdf((ew - n / 2.0) / sd))


def _reference_copula_overall(per_category: dict[str, float], seed: int) -> float:
    """Independent re-implementation of the correlated overall win-prob.

    Mirrors the production copula path so we can assert the engine lands on the
    same number (within MC noise) without reaching into private helpers.
    """
    cats = [c for c in CAT_ORDER if c.lower() in per_category]
    idx = [CAT_ORDER.index(c) for c in cats]
    probs = np.array([per_category[c.lower()] for c in cats], dtype=float)
    sub = DEFAULT_CORRELATION[np.ix_(idx, idx)]
    cop = GaussianCopula(sub)
    rng = np.random.RandomState(seed)
    u = cop.sample(20000, rng)
    wins = (u <= probs).sum(axis=1).astype(float)
    half = len(cats) / 2.0
    return float(((wins > half).sum() + 0.5 * (wins == half).sum()) / len(wins))


# ── (a) overall win-prob: correlated, not independent ─────────────────


class TestOverallCopula:
    def test_overall_matches_independent_copula_recompute(self) -> None:
        """The engine's overall win-prob equals a copula recomputation built
        from the SAME per-category marginals (within Monte-Carlo noise)."""
        my, opp = _slightly_behind()
        seed = 12345
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(seed), n_sims=20000)
        ref = _reference_copula_overall(res["per_category"], seed=seed)
        assert res["overall_win_prob"] == pytest.approx(ref, abs=0.02), (
            f"engine overall {res['overall_win_prob']:.4f} != copula ref {ref:.4f}"
        )

    def test_correlated_overall_less_overconfident_than_independent(self) -> None:
        """For a slightly-behind matchup the correlated estimate must be pulled
        back toward 0.5 vs the over-confident independent-Bernoulli result —
        the core BR-6 symptom (lineup ~3.4x off the copula estimate)."""
        my, opp = _slightly_behind()
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(7), n_sims=20000)
        independent = _independent_overall(res["per_category"])

        # Both sides agree the user is the underdog (< 0.5)...
        assert res["overall_win_prob"] < 0.5
        assert independent < 0.5
        # ...but the correlated estimate is meaningfully HIGHER (less extreme).
        assert res["overall_win_prob"] > independent + 0.02, (
            f"correlated {res['overall_win_prob']:.4f} not pulled up from independent {independent:.4f}"
        )
        # And the gap is no longer the ~3.4x-class over-confidence: the
        # correlated estimate is well within 2x of the independent one.
        assert res["overall_win_prob"] < 2.0 * independent + 0.05

    def test_overall_deterministic_with_seed(self) -> None:
        """Same seed -> identical overall win-prob (reproducible for callers)."""
        my, opp = _slightly_behind()
        a = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(99))
        b = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(99))
        assert a["overall_win_prob"] == b["overall_win_prob"]

    def test_overall_bounded(self) -> None:
        my, opp = _slightly_behind()
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(1))
        assert 0.0 <= res["overall_win_prob"] <= 1.0

    def test_partial_categories_overall_is_coinflip_when_empty(self) -> None:
        """No overlapping categories -> 0.5 (no data)."""
        res = estimate_h2h_win_probability({"r": 80.0}, {"hr": 20.0})
        assert res["overall_win_prob"] == 0.5
        assert res["per_category"] == {}


# ── (b) Skellam dispatch by stat type ─────────────────────────────────


class TestSkellamDispatch:
    def test_skellam_cats_are_low_count_counting(self) -> None:
        """SB/SV/W/L route to Skellam; rate cats never do."""
        assert SKELLAM_CATS == frozenset({"sb", "sv", "w", "l"})
        # No rate cat sneaks in.
        rate = {"avg", "obp", "era", "whip"}
        assert SKELLAM_CATS.isdisjoint(rate)

    def test_low_count_cat_uses_skellam_value(self) -> None:
        """SB per-category prob equals the Skellam helper, NOT the Normal-CV
        value — and the two differ (right-skew captured)."""
        my = {"sb": 9.0, "avg": 0.270}
        opp = {"sb": 11.0, "avg": 0.262}
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(0))

        skellam_p = _category_win_prob_skellam(9.0, 11.0, inverse=False)
        var = default_category_variances()
        sigma = math.sqrt(2.0 * var["sb"])
        normal_p = float(norm.cdf((9.0 - 11.0) / sigma))

        assert res["per_category"]["sb"] == pytest.approx(skellam_p, abs=1e-9)
        assert res["per_category"]["sb"] != pytest.approx(normal_p, abs=1e-6)

    def test_rate_cat_uses_normal_value(self) -> None:
        """A rate cat (AVG) per-category prob equals the Normal-CDF value."""
        my = {"avg": 0.290, "sb": 9.0}
        opp = {"avg": 0.262, "sb": 9.0}
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(0))

        var = default_category_variances()
        sigma = math.sqrt(2.0 * var["avg"])
        normal_p = float(norm.cdf((0.290 - 0.262) / sigma))
        assert res["per_category"]["avg"] == pytest.approx(normal_p, abs=1e-9)

    def test_high_count_counting_cat_uses_normal(self) -> None:
        """K (high-count) is NOT a Skellam cat -> Normal model."""
        assert "k" not in SKELLAM_CATS
        my = {"k": 75.0, "avg": 0.270}
        opp = {"k": 68.0, "avg": 0.270}
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(0))
        var = default_category_variances()
        sigma = math.sqrt(2.0 * var["k"])
        normal_p = float(norm.cdf((75.0 - 68.0) / sigma))
        assert res["per_category"]["k"] == pytest.approx(normal_p, abs=1e-9)


# ── (c) inverse cats still flip ───────────────────────────────────────


class TestInverseCats:
    def test_inverse_set_unchanged(self) -> None:
        assert INVERSE_CATS == {"l", "era", "whip"}

    def test_lower_era_higher_win_prob_normal(self) -> None:
        """ERA is an inverse RATE cat (Normal path): lower ERA -> p > 0.5."""
        my = {"era": 2.80, "avg": 0.270}
        opp = {"era": 4.20, "avg": 0.270}
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(0))
        assert res["per_category"]["era"] > 0.5

    def test_lower_whip_higher_win_prob_normal(self) -> None:
        my = {"whip": 1.00, "avg": 0.270}
        opp = {"whip": 1.40, "avg": 0.270}
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(0))
        assert res["per_category"]["whip"] > 0.5

    def test_fewer_losses_higher_win_prob_skellam(self) -> None:
        """L is an inverse LOW-COUNT cat (Skellam path): fewer L -> p > 0.5."""
        assert "l" in SKELLAM_CATS
        my = {"l": 3.0, "avg": 0.270}
        opp = {"l": 7.0, "avg": 0.270}
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(0))
        assert res["per_category"]["l"] > 0.5
        # And it matches the Skellam helper with inverse=True.
        assert res["per_category"]["l"] == pytest.approx(_category_win_prob_skellam(3.0, 7.0, inverse=True), abs=1e-9)

    def test_more_losses_lower_win_prob_skellam(self) -> None:
        my = {"l": 7.0, "avg": 0.270}
        opp = {"l": 3.0, "avg": 0.270}
        res = estimate_h2h_win_probability(my, opp, rng=np.random.RandomState(0))
        assert res["per_category"]["l"] < 0.5
