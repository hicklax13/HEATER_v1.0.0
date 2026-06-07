"""FA-E4: move the last inline FA tunables into CONSTANTS_REGISTRY.

Two sets of inline magic numbers remained in src/optimizer/fa_recommender.py
after PR #107 centralized most FA thresholds:
  * the ECR-stddev nudge (polarizing/consensus thresholds + their multipliers), and
  * the regression-flag nudge magnitudes (BUY_LOW / SELL_HIGH).

This guard locks them as registry-backed constants (value, citation, bounds,
sensitivity), confirms fa_recommender reads them at import (no remaining inline
literals), and verifies the values are unchanged (no behavior change).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from src.optimizer import fa_recommender
from src.optimizer.constants_registry import CONSTANTS_REGISTRY

_FA_PATH = Path(fa_recommender.__file__)

_NEW_KEYS = [
    "ecr_stddev_polarizing_threshold",
    "ecr_stddev_consensus_threshold",
    "ecr_polarizing_mult",
    "ecr_consensus_mult",
    "regression_buy_low_mult",
    "regression_sell_high_mult",
]

_EXPECTED_VALUES = {
    "ecr_stddev_polarizing_threshold": 20,
    "ecr_stddev_consensus_threshold": 5,
    "ecr_polarizing_mult": 0.95,
    "ecr_consensus_mult": 1.02,
    "regression_buy_low_mult": 1.05,
    "regression_sell_high_mult": 0.95,
}


@pytest.mark.parametrize("key", _NEW_KEYS)
def test_registry_entry_complete(key):
    """Each new key is registered with full metadata."""
    assert key in CONSTANTS_REGISTRY, f"FA-E4: {key} not registered"
    entry = CONSTANTS_REGISTRY[key]
    for attr in ("value", "citation", "lower_bound", "upper_bound", "sensitivity"):
        assert getattr(entry, attr, None) is not None, f"FA-E4: {key}.{attr} missing"
    assert entry.citation, f"FA-E4: {key}.citation empty"
    # Strict bounds (mirrors test_constants_registry.py invariant).
    assert entry.lower_bound < entry.value < entry.upper_bound, (
        f"FA-E4: {key} value {entry.value} not strictly within bounds"
    )


@pytest.mark.parametrize("key,expected", list(_EXPECTED_VALUES.items()))
def test_value_preserved(key, expected):
    """Centralizing must not change the constant's value (no behavior change)."""
    assert CONSTANTS_REGISTRY[key].value == pytest.approx(expected)


def test_module_aliases_read_registry():
    """fa_recommender module-level aliases must read from the registry."""
    assert (
        fa_recommender._ECR_STDDEV_POLARIZING_THRESHOLD == CONSTANTS_REGISTRY["ecr_stddev_polarizing_threshold"].value
    )
    assert fa_recommender._ECR_STDDEV_CONSENSUS_THRESHOLD == CONSTANTS_REGISTRY["ecr_stddev_consensus_threshold"].value
    assert fa_recommender._ECR_POLARIZING_MULT == CONSTANTS_REGISTRY["ecr_polarizing_mult"].value
    assert fa_recommender._ECR_CONSENSUS_MULT == CONSTANTS_REGISTRY["ecr_consensus_mult"].value
    assert fa_recommender._REGRESSION_BUY_LOW_MULT == CONSTANTS_REGISTRY["regression_buy_low_mult"].value
    assert fa_recommender._REGRESSION_SELL_HIGH_MULT == CONSTANTS_REGISTRY["regression_sell_high_mult"].value


def test_no_inline_ecr_regression_literals():
    """The ECR-stddev + regression-flag blocks must use the named constants,
    not bare numeric literals (so calibration takes effect without code edits)."""
    src = _FA_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # Find the _score_fa_candidates function body source span.
    func = next(
        (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_score_fa_candidates"),
        None,
    )
    assert func is not None, "FA-E4: _score_fa_candidates not found"
    func_src = ast.get_source_segment(src, func)
    assert func_src is not None

    # The polarizing/consensus thresholds and the four multipliers must no
    # longer appear as bare literals in the scoring loop.
    banned_patterns = [
        r"_ecr_stddev\s*>\s*20",
        r"_ecr_stddev\s*<\s*5",
        r"composite\s*\*=\s*0\.95\b",
        r"composite\s*\*=\s*1\.02\b",
        r"composite\s*\*=\s*1\.05\b",
        r"1\.0\s*/\s*1\.05",
        r"1\.0\s*/\s*0\.95",
    ]
    for pat in banned_patterns:
        assert not re.search(pat, func_src), f"FA-E4: inline literal still present (pattern {pat!r})"
