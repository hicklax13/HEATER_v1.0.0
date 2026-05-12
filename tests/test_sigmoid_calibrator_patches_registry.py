"""BUG-006 fix: sigmoid_calibrator must patch CONSTANTS_REGISTRY values
(the runtime read-path), not the legacy module aliases."""

from unittest.mock import patch


def test_compute_category_urgency_responds_to_registry_value():
    """`compute_category_urgency` should produce different results when the
    registry sigmoid_k values change — proving the function reads from
    CONSTANTS_REGISTRY at call time."""
    from src.optimizer.category_urgency import compute_category_urgency, patch_sigmoid_k

    my_totals = {"HR": 30, "AVG": 0.280}
    opp_totals = {"HR": 50, "AVG": 0.300}
    with patch_sigmoid_k(counting_k=0.5, rate_k=0.5):
        urgency_low = compute_category_urgency(my_totals, opp_totals)
    with patch_sigmoid_k(counting_k=5.0, rate_k=5.0):
        urgency_high = compute_category_urgency(my_totals, opp_totals)
    diffs = [abs(urgency_low[c] - urgency_high[c]) for c in urgency_low if c in urgency_high]
    assert max(diffs) > 0.05, (
        f"BUG-006: compute_category_urgency unresponsive to registry k change. "
        f"Max urgency diff between k=0.5 and k=5.0: {max(diffs):.4f}"
    )


def test_alias_patch_does_not_affect_function():
    """Patching the legacy module-level alias should NOT affect the function —
    proving why the prior calibrator was a no-op."""
    from src.optimizer.category_urgency import compute_category_urgency

    my_totals = {"HR": 30, "AVG": 0.280}
    opp_totals = {"HR": 50, "AVG": 0.300}
    with patch("src.optimizer.category_urgency.COUNTING_STAT_K", 0.5):
        urgency_a = compute_category_urgency(my_totals, opp_totals)
    with patch("src.optimizer.category_urgency.COUNTING_STAT_K", 5.0):
        urgency_b = compute_category_urgency(my_totals, opp_totals)
    diffs = [abs(urgency_a[c] - urgency_b[c]) for c in urgency_a if c in urgency_b]
    assert max(diffs) < 0.001, (
        "Surprise: alias patching DID affect compute_category_urgency. "
        "Either BUG-006 is actually fixed at the alias level (re-check the "
        "audit assumption) or the function reads from both alias and registry."
    )


def test_patch_sigmoid_k_restores_original_on_exit():
    """The context manager must restore registry values on exit."""
    from src.optimizer.category_urgency import patch_sigmoid_k
    from src.optimizer.constants_registry import CONSTANTS_REGISTRY

    original_counting = CONSTANTS_REGISTRY["sigmoid_k_counting"].value
    original_rate = CONSTANTS_REGISTRY["sigmoid_k_rate"].value
    with patch_sigmoid_k(counting_k=99.0, rate_k=88.0):
        assert CONSTANTS_REGISTRY["sigmoid_k_counting"].value == 99.0
        assert CONSTANTS_REGISTRY["sigmoid_k_rate"].value == 88.0
    # After exit, originals restored
    assert CONSTANTS_REGISTRY["sigmoid_k_counting"].value == original_counting
    assert CONSTANTS_REGISTRY["sigmoid_k_rate"].value == original_rate
