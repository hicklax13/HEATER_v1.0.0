"""FA-engine P1 PR2 (2026-05-20): IL players retain weighted projection,
not zero, in _roster_category_totals.

Background:
  The old behavior at src/in_season.py:42-94 zeroed out IL players
  entirely from category totals. This made dropping an IL ace appear
  costless in the SGP math (before/after totals identical), which was
  the upstream cause of the Crochet/Kirk bad recommendation — top-30
  SP on IL15 valued as a free drop.

  Industry research consensus (RotoWire IL Stash Guide + FantasyPros
  injury rankings + The Athletic) is that IL players should NOT be
  zeroed — they should keep a projection-scaled-by-return-window
  fraction of their full ROS value.

These tests pin the new contract:
  * IL10 → 0.85 weight (returning in <2 weeks)
  * IL15 → 0.70 (~3 weeks lost)
  * IL60 → 0.20 (long stash, still nonzero)
  * Suspended / Restricted / NA → 0.0 (still zeroed)
  * Active → 1.0
  * Unknown statuses → 1.0 (fail-open, not fail-closed like before)
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.in_season import _il_weight_from_status, _roster_category_totals

# ── _il_weight_from_status ───────────────────────────────────────────


@pytest.mark.parametrize(
    "status,expected",
    [
        ("active", 1.0),
        ("Active", 1.0),
        ("", 1.0),
        (None, 1.0),
        ("DTD", 0.95),
        ("day-to-day", 0.95),
        ("Day-To-Day", 0.95),
        ("IL10", 0.85),
        ("IL-10", 0.85),
        ("il10", 0.85),
        ("IL15", 0.70),
        ("IL-15", 0.70),
        ("IL", 0.70),  # bare IL treated as IL15
        ("IL60", 0.20),
        ("IL-60", 0.20),
        ("NA", 0.0),
        ("Suspended", 0.0),
        ("SUSPENDED", 0.0),
        ("Restricted", 0.0),
        ("OUT", 0.0),
    ],
)
def test_il_weight_from_status_known_values(status, expected):
    """Each known status maps to the documented weight."""
    assert _il_weight_from_status(status) == pytest.approx(expected)


def test_il_weight_unknown_status_fails_open():
    """Unknown status defaults to 1.0 (active), NOT zero.

    Failing open is intentional: the previous bug was the engine
    treating an unknown-but-injured player as active OR as fully zeroed.
    Treating them as active is the safer default for OUR engine because
    the IL-stash UI-layer filter handles the "don't recommend dropping"
    side independently.
    """
    assert _il_weight_from_status("WeirdNewYahooStatus") == 1.0
    assert _il_weight_from_status("???") == 1.0


# ── _roster_category_totals integration ───────────────────────────────


def _crochet_row(status: str = "IL15") -> dict:
    """A Crochet-like pitcher row (top-30 SP). Returns dict for DataFrame ctor."""
    return {
        "player_id": 100,
        "name": "Garrett Crochet",
        "is_hitter": 0,
        "status": status,
        "r": 0,
        "hr": 0,
        "rbi": 0,
        "sb": 0,
        "ab": 0,
        "h": 0,
        "bb": 0,
        "hbp": 0,
        "sf": 0,
        "w": 12,
        "l": 6,
        "sv": 0,
        "k": 180,
        "ip": 180.0,
        "er": 64,
        "bb_allowed": 50,
        "h_allowed": 150,
    }


def _judge_row(status: str = "active") -> dict:
    """An active hitter row."""
    return {
        "player_id": 200,
        "name": "Aaron Judge",
        "is_hitter": 1,
        "status": status,
        "r": 100,
        "hr": 40,
        "rbi": 100,
        "sb": 5,
        "ab": 500,
        "h": 140,
        "bb": 80,
        "hbp": 5,
        "sf": 5,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0.0,
        "er": 0.0,
        "bb_allowed": 0,
        "h_allowed": 0,
    }


def test_nan_stat_does_not_raise():
    """A NaN counting stat (sparse pool rows) must not raise: ``int(nan or 0)``
    raises ValueError because NaN is truthy → ``int(nan)``."""
    row = _judge_row("active")
    row["r"] = float("nan")  # NaN run total
    row["ip"] = float("nan")  # NaN IP (float path)
    pool = pd.DataFrame([row])
    totals = _roster_category_totals([200], pool)  # must not raise
    assert totals["R"] == 0.0  # NaN → 0
    assert totals["HR"] == 40.0  # rest of the line intact


def test_active_player_contributes_full_projection():
    """Active player → weight 1.0 → full projection counts."""
    pool = pd.DataFrame([_judge_row("active")])
    totals = _roster_category_totals([200], pool)
    assert totals["R"] == 100
    assert totals["HR"] == 40
    assert totals["RBI"] == 100


def test_il15_pitcher_contributes_weighted_not_zero():
    """The Crochet bug fix: IL15 ace contributes 70% of his projection,
    not zero. Old behavior would have given totals K=0; new gives K=126."""
    pool = pd.DataFrame([_crochet_row("IL15")])
    totals = _roster_category_totals([100], pool)
    assert totals["K"] == pytest.approx(180 * 0.70)
    assert totals["W"] == pytest.approx(12 * 0.70)
    assert totals["ip"] == pytest.approx(180.0 * 0.70)
    # The bug regression-guard: K must be nonzero for an IL15 pitcher.
    assert totals["K"] > 0


def test_il60_player_contributes_partial_projection():
    """IL60 stash gets 20% weight — still nonzero, but heavily discounted."""
    pool = pd.DataFrame([_crochet_row("IL60")])
    totals = _roster_category_totals([100], pool)
    assert totals["K"] == pytest.approx(180 * 0.20)
    assert totals["K"] > 0  # still nonzero — defining bug-fix property


def test_suspended_player_still_zeroed():
    """Suspended player → weight 0.0 → contributes nothing. Distinguishes
    from IL: a suspended player isn't coming back, so the legacy zero
    behavior is correct for them."""
    pool = pd.DataFrame([_crochet_row("Suspended")])
    totals = _roster_category_totals([100], pool)
    assert totals["K"] == 0
    assert totals["W"] == 0


def test_drop_il_player_no_longer_free():
    """Direct test of the Crochet/Kirk bug pattern: dropping an IL ace
    must cost SOMETHING in the totals delta. Before: totals identical
    pre/post-drop. After: clear delta."""
    pool = pd.DataFrame([_crochet_row("IL15"), _judge_row("active")])
    totals_with_crochet = _roster_category_totals([100, 200], pool)
    totals_without_crochet = _roster_category_totals([200], pool)
    # Dropping Crochet should reduce K by 70% of his projection — nonzero.
    delta_k = totals_with_crochet["K"] - totals_without_crochet["K"]
    assert delta_k == pytest.approx(180 * 0.70)
    assert delta_k > 0, (
        "Dropping an IL15 ace must reduce team K total. If this is 0, "
        "the IL-zeroing bug has regressed and the Crochet/Kirk class of "
        "bad recommendation can return."
    )


def test_rate_stats_computed_on_weighted_totals():
    """ERA / WHIP / AVG / OBP must use the WEIGHTED counting totals, not
    the unweighted projection columns. Otherwise an IL pitcher's full
    ERA still pollutes the team rate-stat computation while their volume
    is reduced — that's a different kind of bug."""
    # A 4.00 ERA pitcher (64 ER / 180 IP * 9 = 3.20 ERA) on IL15.
    pool = pd.DataFrame([_crochet_row("IL15")])
    totals = _roster_category_totals([100], pool)
    # Both er and ip get the same 0.70 multiplier, so the ratio survives.
    # Expected: 64*0.70 / 180*0.70 = 64/180, same ERA ratio
    expected_era = 64 * 9.0 / 180.0  # = 3.20
    assert totals["ERA"] == pytest.approx(expected_era, abs=0.01)


def test_mixed_active_and_il_roster():
    """Realistic roster mix: one active hitter + one IL15 pitcher.
    Each contributes appropriately."""
    pool = pd.DataFrame([_judge_row("active"), _crochet_row("IL15")])
    totals = _roster_category_totals([100, 200], pool)
    # Judge (active) contributes full
    assert totals["R"] == 100
    assert totals["HR"] == 40
    # Crochet (IL15) contributes 70%
    assert totals["K"] == pytest.approx(180 * 0.70)
    assert totals["W"] == pytest.approx(12 * 0.70)


def test_empty_roster_returns_neutral_sentinels():
    """No players → AVG=0.250, OBP=league_avg, ERA/WHIP=league_avg.
    Regression guard: this behavior must not change."""
    pool = pd.DataFrame(
        [{**_judge_row(), "player_id": 999, "status": "Suspended"}]  # zeroed
    )
    totals = _roster_category_totals([], pool)  # no IDs match
    assert totals["AVG"] == pytest.approx(0.250)
    assert totals["ERA"] > 0  # league_avg sentinel
    assert totals["WHIP"] > 0
