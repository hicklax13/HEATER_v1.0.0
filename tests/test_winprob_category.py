"""Tests for the Layer-1 per-category win/tie/loss math (Phase 2a slice 1)."""

import math

import pytest


def test_sum_to_one_counting_normal_path():
    from src.winprob_proxy.category import category_win_tie_loss

    # large sigma -> Normal-CC path; must sum to 1 exactly.
    w, t, ell = category_win_tie_loss(
        a_mu=40, a_var=60, b_mu=34, b_var=55, kind="counting", inverse=False, rounding_unit=0.0
    )
    assert w + t + ell == pytest.approx(1.0, abs=1e-9)
    assert w > ell  # A's mean is higher -> A favored


def test_sum_to_one_rate_welch():
    from src.winprob_proxy.category import category_win_tie_loss

    w, t, ell = category_win_tie_loss(
        a_mu=0.276, a_var=0.0004, b_mu=0.271, b_var=0.0004, kind="rate", inverse=False, rounding_unit=0.001
    )
    assert w + t + ell == pytest.approx(1.0, abs=1e-9)
    assert w > ell  # higher AVG favored


def test_identical_rosters_symmetric():
    from src.winprob_proxy.category import category_win_tie_loss

    w, t, ell = category_win_tie_loss(
        a_mu=20, a_var=30, b_mu=20, b_var=30, kind="counting", inverse=False, rounding_unit=0.0
    )
    assert w == pytest.approx(ell)  # mu_D=0 -> win == loss exactly
    assert w + t + ell == pytest.approx(1.0, abs=1e-9)


def test_inverse_swaps_win_and_loss_keeps_tie():
    from src.winprob_proxy.category import category_win_tie_loss

    # ERA (rate, inverse): A 3.10 vs B 3.60 -> A (lower) should WIN.
    w_inv, t_inv, l_inv = category_win_tie_loss(3.10, 0.05, 3.60, 0.05, kind="rate", inverse=True, rounding_unit=0.01)
    w_non, t_non, l_non = category_win_tie_loss(3.10, 0.05, 3.60, 0.05, kind="rate", inverse=False, rounding_unit=0.01)
    assert (w_inv, l_inv) == (l_non, w_non)  # tails swapped
    assert t_inv == pytest.approx(t_non)  # tie invariant
    assert w_inv > l_inv  # lower ERA wins under inverse


def test_degenerate_zero_variance():
    from src.winprob_proxy.category import category_win_tie_loss

    # sigma_D = 0: hard comparison against the tie band.
    assert category_win_tie_loss(5, 0, 3, 0, "counting", False, 0.0) == (1.0, 0.0, 0.0)  # |mu_D|=2 >= 0.5 -> win
    assert category_win_tie_loss(3, 0, 3, 0, "counting", False, 0.0) == (0.0, 1.0, 0.0)  # equal -> tie
    assert category_win_tie_loss(3, 0, 5, 0, "counting", False, 0.0) == (0.0, 0.0, 1.0)  # loss
    # inverse degenerate flips
    assert category_win_tie_loss(3.0, 0, 3.6, 0, "rate", True, 0.01) == (1.0, 0.0, 0.0)  # lower ERA -> win


def test_nan_and_negative_inputs_safe():
    from src.winprob_proxy.category import category_win_tie_loss

    w, t, ell = category_win_tie_loss(float("nan"), -5.0, float("nan"), float("nan"), "counting", False, 0.0)
    assert (w, t, ell) == (0.0, 1.0, 0.0)  # NaN means -> 0, neg var -> 0 -> degenerate tie
    assert math.isfinite(w) and math.isfinite(t) and math.isfinite(ell)


def test_skellam_moment_match_reproduces_mean_and_var():
    # The chosen lambdas must reproduce the target margin mean+variance exactly.
    from scipy.stats import skellam

    mu_d, var_d = 1.0, 3.0  # low-count SV-like: sigma_D ~ 1.73 -> Skellam regime
    lam1 = 0.5 * (var_d + mu_d)
    lam2 = 0.5 * (var_d - mu_d)
    m, v = skellam.stats(lam1, lam2, moments="mv")
    assert float(m) == pytest.approx(mu_d)
    assert float(v) == pytest.approx(var_d)


def test_skellam_branch_used_in_low_count_and_sums_to_one():
    from src.winprob_proxy.category import category_win_tie_loss

    # sigma_D ~ 1.73 (< 3) and feasible -> Skellam path; must sum to 1 and give a sizeable tie.
    w, t, ell = category_win_tie_loss(
        a_mu=2.0, a_var=1.5, b_mu=1.0, b_var=1.5, kind="counting", inverse=False, rounding_unit=0.0
    )
    assert w + t + ell == pytest.approx(1.0, abs=1e-9)
    assert t > 0.10  # low-count integer totals tie often
    assert w > ell  # A's mean higher


def test_skellam_infeasible_falls_back_to_normal():
    from src.winprob_proxy.category import category_win_tie_loss

    # var_D (=1) < |mu_D| (=8): Skellam infeasible -> Normal-CC fallback, still finite + sums to 1,
    # and the trailing team keeps a small non-zero loss/win (moment-faithful, not clamped to 0).
    w, t, ell = category_win_tie_loss(
        a_mu=10, a_var=0.5, b_mu=2, b_var=0.5, kind="counting", inverse=False, rounding_unit=0.0
    )
    assert w + t + ell == pytest.approx(1.0, abs=1e-9)
    assert w > 0.99 and ell > 0.0  # A dominant but B's win prob honestly > 0


def test_skellam_zero_lambda_falls_through_no_nan():
    from src.winprob_proxy.category import category_win_tie_loss

    # A lambda of exactly 0 (feasibility boundary var_d==|mu_d|, or a saveless SV opponent with
    # zero variance) makes scipy.stats.skellam return nan. These must fall through to Normal-CC and
    # stay FINITE + sum-to-1 (regression: the raw skellam nan-triple used to escape the renorm guard).
    for a_mu, a_var, b_mu, b_var in [(1.0, 1.0, 0.0, 0.0), (4.0, 2.0, 1.0, 1.0), (3.0, 3.0, 0.0, 0.0)]:
        w, t, ell = category_win_tie_loss(a_mu, a_var, b_mu, b_var, "counting", False, 0.0)
        assert all(math.isfinite(x) for x in (w, t, ell)), (a_mu, a_var, b_mu, b_var)
        assert w + t + ell == pytest.approx(1.0, abs=1e-9)
        assert w > ell  # A leads


def test_skellam_vs_normal_agree_at_large_sigma():
    from src.winprob_proxy.category import category_win_tie_loss

    # sigma_D ~ 10.7 (>> 3) -> Normal path; independently the Skellam would agree to <1%. Here we just
    # assert the Normal branch is taken (large sigma) and behaves (P(win) high but < 1).
    w, t, ell = category_win_tie_loss(
        a_mu=40, a_var=60, b_mu=34, b_var=55, kind="counting", inverse=False, rounding_unit=0.0
    )
    assert 0.6 < w < 0.85 and t < 0.05


def test_skellam_pmf_zero_matches_bessel():
    # P(D=0) = e^{-(l1+l2)} I_0(2 sqrt(l1 l2)) — the exact Bessel form.
    from scipy.special import i0
    from scipy.stats import skellam

    lam1, lam2 = 2.0, 1.0
    expected = math.exp(-(lam1 + lam2)) * i0(2 * math.sqrt(lam1 * lam2))
    assert float(skellam.pmf(0, lam1, lam2)) == pytest.approx(expected, rel=1e-9)


def test_kind_and_rounding_helpers():
    from src.winprob_proxy.category import kind_for, rounding_unit_for

    assert kind_for("HR") == "counting"
    assert kind_for("AVG") == "rate" and kind_for("ERA") == "rate"
    assert rounding_unit_for("AVG") == 0.001 and rounding_unit_for("OBP") == 0.001
    assert rounding_unit_for("ERA") == 0.01 and rounding_unit_for("WHIP") == 0.01
    assert rounding_unit_for("HR") == 0.0  # counting -> unused


def test_near_zero_means_large_tie_no_nan():
    from src.winprob_proxy.category import category_win_tie_loss

    # SV with two saveless teams: mu ~ 0, small var -> mostly TIE (both likely 0), finite.
    w, t, ell = category_win_tie_loss(
        a_mu=0.2, a_var=0.2, b_mu=0.1, b_var=0.2, kind="counting", inverse=False, rounding_unit=0.0
    )
    assert w + t + ell == pytest.approx(1.0, abs=1e-9)
    assert t > 0.5 and all(math.isfinite(x) for x in (w, t, ell))


def test_skellam_normal_convergence_monotone_in_sigma():
    # At fixed standardized gap, the Skellam and Normal-CC triples converge as sigma_D grows.
    from scipy.stats import norm, skellam

    def normal_cc(mu_d, var_d):
        s = var_d**0.5
        w = float(norm.sf(0.5, loc=mu_d, scale=s))
        ell = float(norm.cdf(-0.5, loc=mu_d, scale=s))
        return (w, 1 - w - ell, ell)

    def skel(mu_d, var_d):
        l1, l2 = 0.5 * (var_d + mu_d), 0.5 * (var_d - mu_d)
        return (float(skellam.sf(0, l1, l2)), float(skellam.pmf(0, l1, l2)), float(skellam.cdf(-1, l1, l2)))

    diffs = []
    for scale in (1.0, 3.0, 9.0):
        mu_d, var_d = 1.0 * scale, 3.0 * scale  # keep mu_D/sigma_D shape, grow sigma_D
        d = max(abs(a - b) for a, b in zip(normal_cc(mu_d, var_d), skel(mu_d, var_d)))
        diffs.append(d)
    assert diffs[0] > diffs[1] > diffs[2]  # convergence
    assert diffs[2] < 0.01  # <1% agreement at large sigma_D
