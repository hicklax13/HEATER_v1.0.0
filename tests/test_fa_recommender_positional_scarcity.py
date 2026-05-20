"""FA-engine P1 PR3 (2026-05-20): positional scarcity factor multiplies
the FA base_value in recommend_fa_moves.

Background:
  The recommendation engine previously evaluated all FAs at a single
  flat raw-SGP multiplier — no recognition that a top-2 catcher is
  more valuable to roster construction than the 25th-best OF with the
  same nominal SGP. This is part of the Crochet/Kirk class of bug:
  the engine ranked a backup catcher (Kirk) above a top SP (Crochet)
  partly because no positional adjustment was applied.

  Industry consensus (Razzball Player Rater FAQ, FanGraphs Auction
  Calculator) values players partially by replacement-level scarcity
  at their position. We use the dynamic per-position replacement
  level from compute_replacement_levels — scarcer positions (lower
  replacement) yield a higher scarcity multiplier.

These tests pin:
  * _positional_scarcity_factor returns 1.0 for missing data
  * _positional_scarcity_factor caps the boost at _POSITIONAL_SCARCITY_MAX_BOOST
  * Scarce positions (low replacement) yield boost > 1.0
  * Deep positions (high replacement) yield 1.0 (no penalty)
  * Multi-position players use their SCARCEST eligible position
"""

from __future__ import annotations

import pytest

from src.optimizer.fa_recommender import (
    _POSITIONAL_SCARCITY_MAX_BOOST,
    _positional_scarcity_factor,
)

# ── Edge cases ───────────────────────────────────────────────────────


def test_no_positions_returns_neutral():
    """Empty position string → 1.0 (no boost, no penalty)."""
    assert _positional_scarcity_factor("", {"C": 5.0, "OF": 8.0}) == 1.0
    assert _positional_scarcity_factor(None, {"C": 5.0, "OF": 8.0}) == 1.0


def test_no_replacement_levels_returns_neutral():
    """No replacement_levels data → 1.0 (fail-safe default)."""
    assert _positional_scarcity_factor("C,Util", {}) == 1.0


def test_position_not_in_levels_returns_neutral():
    """A position not present in the replacement levels → 1.0.
    Avoids overweighting unfamiliar position strings."""
    assert _positional_scarcity_factor("XX", {"C": 5.0, "OF": 8.0}) == 1.0


def test_all_zero_replacement_levels_returns_neutral():
    """If median replacement is 0 (no usable signal) → 1.0.
    Prevents divide-by-zero."""
    assert _positional_scarcity_factor("C,OF", {"C": 0.0, "OF": 0.0}) == 1.0


# ── Scarcity math ────────────────────────────────────────────────────


def test_scarce_position_yields_boost():
    """Catcher with replacement_level=2.0 vs median=8.0 → boost >1.0.
    Formula: 1.0 + (8 - 2) / 8 = 1.75 → capped at 1 + max_boost."""
    levels = {"C": 2.0, "1B": 8.0, "OF": 12.0, "SS": 6.0, "2B": 4.0, "3B": 8.0}
    mult = _positional_scarcity_factor("C", levels)
    # Should be at the cap since (median - 2)/median is large.
    assert mult == pytest.approx(1.0 + _POSITIONAL_SCARCITY_MAX_BOOST)


def test_deep_position_yields_no_boost():
    """OF at replacement_level above median → 1.0 (no boost).
    The formula bounds negative ratios at 0."""
    levels = {"C": 2.0, "1B": 8.0, "OF": 12.0, "SS": 6.0, "2B": 4.0, "3B": 8.0}
    mult = _positional_scarcity_factor("OF", levels)
    assert mult == 1.0


def test_median_position_yields_no_boost():
    """A position exactly at median replacement → 1.0."""
    levels = {"C": 2.0, "1B": 8.0, "OF": 12.0, "SS": 6.0, "2B": 4.0, "3B": 8.0}
    # median of [2, 4, 6, 8, 8, 12] sorted = 7 (middle index 3 = 8 for
    # 6-element odd index? let's compute)
    # For 6 elements sorted [2, 4, 6, 8, 8, 12], len//2 = 3, so median[3] = 8
    mult = _positional_scarcity_factor("1B", levels)
    assert mult == 1.0  # 1B at 8 = median, ratio = 0


def test_multi_position_uses_scarcest_eligible():
    """A 2B/SS-eligible player picks the SCARCEST eligible position.
    Use values where neither hits the cap so the comparison is meaningful.
    SS at 9.0 (ratio 0.10), 2B at 8.0 (ratio 0.20), median = 10 → 2B
    yields a higher multiplier than SS, both below the cap."""
    levels = {"C": 9.5, "1B": 10.0, "OF": 11.0, "SS": 9.0, "2B": 8.0, "3B": 10.5}
    # median of [8, 9, 9.5, 10, 10.5, 11] sorted, len=6, idx 3 = 10
    mult_2b_ss = _positional_scarcity_factor("2B,SS", levels)
    mult_only_ss = _positional_scarcity_factor("SS", levels)
    mult_only_2b = _positional_scarcity_factor("2B", levels)
    # 2B is scarcer than SS, so the multi-eligible player uses 2B.
    assert mult_2b_ss == mult_only_2b
    assert mult_2b_ss > mult_only_ss
    # Neither should be at the cap with these values.
    assert mult_2b_ss < 1.0 + _POSITIONAL_SCARCITY_MAX_BOOST


def test_scarcity_cap_is_enforced():
    """A pathologically scarce position (replacement near 0) maxes out
    at _POSITIONAL_SCARCITY_MAX_BOOST."""
    levels = {"C": 0.01, "OF": 10.0, "1B": 10.0, "SS": 10.0}
    mult = _positional_scarcity_factor("C", levels)
    assert mult <= 1.0 + _POSITIONAL_SCARCITY_MAX_BOOST
    assert mult == pytest.approx(1.0 + _POSITIONAL_SCARCITY_MAX_BOOST, abs=0.001)


def test_max_boost_is_reasonable_value():
    """Sanity: the cap should be in a reasonable range for fantasy
    valuation. Industry-published scarcity adjustments range 5-25%."""
    assert 0.05 <= _POSITIONAL_SCARCITY_MAX_BOOST <= 0.40, (
        f"_POSITIONAL_SCARCITY_MAX_BOOST={_POSITIONAL_SCARCITY_MAX_BOOST} is "
        "outside the reasonable 5-40% range for fantasy positional scarcity. "
        "If you changed this intentionally, update the test bounds too."
    )
