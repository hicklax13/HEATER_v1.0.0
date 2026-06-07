"""BR-2b: My Team "Priority Targets" must rank losing categories by a
NORMALIZED (cross-comparable) closeness-to-flip measure, not raw diff.

The red callout surfaces the top-2 losing categories as "priority targets".
The old selection sorted by raw ``diff``, which mixes counting-cat gaps
(R behind by 6) with rate-cat gaps (AVG behind by 0.069) on incompatible
scales — the big-count cats almost always sorted first, so a near-flippable
rate cat was never surfaced.

``_rank_priority_losing_cats`` normalizes each losing cat's gap by that
category's weekly standard deviation (``h2h_engine.default_weekly_sigmas()``)
to a unit-free z-gap; the SMALLEST |z-gap| losing cats are the closest to
flipping = the real priority targets.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from src.optimizer.h2h_engine import default_weekly_sigmas

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "1_My_Team.py"


def _load_helper():
    """Import ``_rank_priority_losing_cats`` from the page module.

    The page imports streamlit at module load but defines the helper at
    module top level, so a normal import is sufficient (no Streamlit runtime
    needed for the pure function).
    """
    spec = importlib.util.spec_from_file_location("_my_team_page", _PAGE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._rank_priority_losing_cats


def _gap_row(cat: str, diff: float, *, is_inverse: bool = False) -> dict:
    """Build a losing-category gap row (above/tied False)."""
    return {
        "cat": cat,
        "diff": diff,
        "above": False,
        "tied": False,
        "is_inverse": is_inverse,
    }


def test_near_flip_rate_cat_beats_wide_count_cat():
    """A near-flip rate cat (small z-gap) must outrank a wide-gap counting cat
    (large z-gap) — the old raw-diff sort buried the rate cat."""
    rank = _load_helper()
    sigmas = default_weekly_sigmas()

    # R behind by 14 (~1 sigma, R sigma ~15) — a WIDE z-gap.
    # AVG behind by 0.003 (~0.2 sigma, AVG sigma ~0.015) — a NEAR-FLIP z-gap.
    rows = [
        _gap_row("R", -14.0),
        _gap_row("AVG", -0.003),
    ]
    result = rank(rows, sigmas, top_n=2)
    cats = [r["cat"] for r in result]
    # Both are losing so both appear, but AVG (closer to flip) must rank FIRST.
    assert cats[0] == "AVG", f"near-flip AVG should be the top priority target, got order {cats}"


def test_raw_diff_sort_would_have_buried_the_rate_cat():
    """Direct contrast with the OLD behavior: sorting by raw diff puts R first
    (−14 < −0.003), which is exactly the bug. The normalized sort inverts it."""
    rank = _load_helper()
    sigmas = default_weekly_sigmas()
    rows = [_gap_row("R", -14.0), _gap_row("AVG", -0.003)]

    old_order = [r["cat"] for r in sorted(rows, key=lambda x: x["diff"])]
    new_order = [r["cat"] for r in rank(rows, sigmas, top_n=2)]

    assert old_order[0] == "R"  # the bug
    assert new_order[0] == "AVG"  # the fix
    assert old_order != new_order


def test_only_losing_cats_returned():
    """Winning / tied rows are excluded even if passed in."""
    rank = _load_helper()
    sigmas = default_weekly_sigmas()
    rows = [
        {"cat": "HR", "diff": 5.0, "above": True, "tied": False, "is_inverse": False},
        {"cat": "SB", "diff": 0.0, "above": False, "tied": True, "is_inverse": False},
        _gap_row("RBI", -3.0),
    ]
    result = rank(rows, sigmas, top_n=2)
    cats = {r["cat"] for r in result}
    assert cats == {"RBI"}


def test_top_n_caps_subset():
    """Returns at most ``top_n`` losing cats."""
    rank = _load_helper()
    sigmas = default_weekly_sigmas()
    rows = [_gap_row("R", -2.0), _gap_row("HR", -2.0), _gap_row("RBI", -2.0)]
    assert len(rank(rows, sigmas, top_n=2)) == 2


def test_inverse_cat_gap_magnitude_used():
    """For inverse cats, the gap magnitude (closeness) is what matters; an
    inverse cat barely behind should rank ahead of a counting cat far behind."""
    rank = _load_helper()
    sigmas = default_weekly_sigmas()
    # ERA "diff" is already oriented so positive = winning (opp - my). A small
    # negative diff = barely losing = near flip. ERA sigma ~0.5.
    rows = [
        _gap_row("K", -10.0),  # ~0.83 sigma behind (K sigma ~12)
        _gap_row("ERA", -0.05, is_inverse=True),  # ~0.1 sigma behind = near flip
    ]
    result = rank(rows, sigmas, top_n=2)
    assert result[0]["cat"] == "ERA"


def test_missing_sigma_treated_as_large_gap():
    """A category with no sigma entry must not crash and must sort to the
    bottom (treated as far from flipping)."""
    rank = _load_helper()
    sigmas = {"AVG": 0.015}  # only AVG has a sigma
    rows = [_gap_row("AVG", -0.003), _gap_row("MADEUP", -0.001)]
    result = rank(rows, sigmas, top_n=2)
    # AVG (has sigma, small z-gap) ranks ahead of the unknown cat.
    assert result[0]["cat"] == "AVG"


def test_empty_input_returns_empty():
    rank = _load_helper()
    assert rank([], default_weekly_sigmas(), top_n=2) == []
