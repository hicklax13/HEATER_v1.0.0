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
