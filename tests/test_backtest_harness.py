# tests/test_backtest_harness.py
import numpy as np
import pytest


def test_decision_regret_zero_for_perfect_predictor():
    from src.backtest_harness import decision_regret

    realized = [3.0, 1.0, 2.0, 0.5]
    assert decision_regret(predicted=realized, realized=realized, k=2) == 0.0


def test_decision_regret_positive_when_ranking_inverts():
    from src.backtest_harness import decision_regret

    pred = [1.0, 2.0, 3.0]
    real = [3.0, 2.0, 1.0]
    assert decision_regret(pred, real, k=1) == pytest.approx(2.0)


def test_decision_regret_empty_and_k_guard():
    from src.backtest_harness import decision_regret

    assert decision_regret([], [], k=1) == 0.0
    assert decision_regret([1.0], [1.0], k=0) == 0.0
    assert decision_regret([1.0, 2.0], [1.0, 2.0], k=9) == 0.0
