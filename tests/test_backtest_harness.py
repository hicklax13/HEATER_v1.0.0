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


def test_purged_kfold_covers_all_test_indices_disjointly():
    from src.backtest_harness import purged_kfold_splits

    splits = list(purged_kfold_splits(n=20, n_folds=5, embargo=0))
    assert len(splits) == 5
    seen = []
    for train, test in splits:
        seen.extend(test.tolist())
        assert set(train.tolist()).isdisjoint(set(test.tolist()))
    assert sorted(seen) == list(range(20))


def test_purged_kfold_embargo_removes_neighbors_from_train():
    from src.backtest_harness import purged_kfold_splits

    splits = list(purged_kfold_splits(n=20, n_folds=5, embargo=2))
    train, test = splits[2]
    assert test.tolist() == [8, 9, 10, 11]
    assert set(range(6, 14)).isdisjoint(set(train.tolist()))
    assert 5 in train.tolist() and 14 in train.tolist()


def test_diebold_mariano_detects_a_strictly_better():
    from src.backtest_harness import diebold_mariano

    rng = np.random.default_rng(0)
    loss_b = rng.uniform(0.8, 1.2, size=200)
    loss_a = loss_b - 0.3
    dm, p = diebold_mariano(loss_a, loss_b)
    assert dm < 0
    assert p < 0.05


def test_diebold_mariano_equal_models_not_significant():
    from src.backtest_harness import diebold_mariano

    rng = np.random.default_rng(1)
    loss = rng.uniform(0.8, 1.2, size=200)
    dm, p = diebold_mariano(loss, loss.copy())
    assert dm == 0.0 and p == 1.0


def test_diebold_mariano_degenerate_guards():
    from src.backtest_harness import diebold_mariano

    assert diebold_mariano([1.0], [2.0]) == (0.0, 1.0)
    assert diebold_mariano([], []) == (0.0, 1.0)
