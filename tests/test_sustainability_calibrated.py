"""FA-engine P2 PR5 (2026-05-20): sustainability score is a calibrated
sigmoid, not a 5-bucket step function.

Background:
  The old compute_sustainability_score returned one of 5 discrete values
  via step functions, displayed as a percentage in the UI. The score
  read like a calibrated probability ("60% sustainability") but was
  actually a hand-picked bucket. Worse: pitcher logic was INVERTED — a
  pitcher with ERA 6.00 returned sustainability 0.7 ("likely to improve")
  while one with ERA 2.40 returned 0.4 ("unsustainably low"). The
  pitcher-version semantic was REGRESSION-FAVORABLE, not "will continue
  producing" — yet the UI string was "Strong underlying metrics support
  continued production".

  New design:
    * Anchor on canonical industry regression signals: xwOBA-wOBA gap
      for hitters, ERA-xFIP gap for pitchers.
    * Combine via sigmoid for a calibrated 0-1 output.
    * Secondary signals: BABIP vs .300 (hitters), Stuff+ (pitchers).
    * Raised sample-size thresholds to AB > 80 / IP > 30 (closer to
      Pizza Cutter / Russell Carleton stabilization points).
    * 0.5 returned for insufficient samples (no signal — neutral).

Pool convention (CANONICAL — see src/database.py:1503):
    df["xwoba_delta"] = xwoba - woba_approx   # i.e. xwOBA - wOBA
A POSITIVE delta therefore means xwOBA > wOBA = the hitter is
UNDERPERFORMING his quality of contact = regression UP coming = BUY_LOW.
database.py:1505 flags xwoba_delta >= 0.030 as BUY_LOW; src/alerts.py
labels positive delta "underperforming contact quality". So a positive
xwoba_delta must map to HIGH sustainability (the underlying skill supports
better results going forward).

These tests pin the calibration:
  * Underperforming hitter (positive xwoba_delta, BUY_LOW) → high
    sustainability (regression UP coming — buy low)
  * Overperforming hitter (negative xwoba_delta, SELL_HIGH) → low
    sustainability (regression DOWN coming — sell high)
  * Lucky pitcher (low ERA, high xFIP) → low sustainability
  * Unlucky pitcher (high ERA, low xFIP) → high sustainability
  * Output is continuous in (0, 1) — no more discrete buckets
  * Insufficient samples return exactly 0.5
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.waiver_wire import compute_sustainability_score


def _hitter(ab=400, h=110, hr=20, sf=4, k=80, xwoba_delta=0.0, **overrides) -> pd.Series:
    base = {
        "is_hitter": 1,
        "ab": ab,
        "h": h,
        "hr": hr,
        "sf": sf,
        "k": k,
        "xwoba_delta": xwoba_delta,
    }
    base.update(overrides)
    return pd.Series(base)


def _pitcher(ip=100, era=4.0, xfip=4.0, stuff_plus=100.0, **overrides) -> pd.Series:
    base = {
        "is_hitter": 0,
        "ip": ip,
        "era": era,
        "xfip": xfip,
        "stuff_plus": stuff_plus,
    }
    base.update(overrides)
    return pd.Series(base)


# ── Sample-size gating ───────────────────────────────────────────────


def test_hitter_below_threshold_returns_neutral():
    """AB < 80 → 0.5 (no signal)."""
    p = _hitter(ab=50, xwoba_delta=0.050)  # would be "sell high" with sample
    assert compute_sustainability_score(p) == 0.5


def test_pitcher_below_threshold_returns_neutral():
    """IP < 30 → 0.5 (no signal)."""
    p = _pitcher(ip=20, era=2.50, xfip=4.50)
    assert compute_sustainability_score(p) == 0.5


# ── Hitter signal ────────────────────────────────────────────────────


def test_hitter_underperforming_xwoba_yields_high_sustainability():
    """Pool convention: xwoba_delta = xwOBA - wOBA (src/database.py:1503).
    A POSITIVE delta means xwOBA > wOBA = hitter is UNDERPERFORMING his
    quality of contact — regression UP coming. Sustainability HIGH (buy
    low). This is the same direction database.py flags as BUY_LOW at the
    +0.030 threshold."""
    p = _hitter(ab=400, h=80, xwoba_delta=0.040)
    score = compute_sustainability_score(p)
    assert score > 0.5, (
        f"Hitter underperforming xwOBA (+0.040 = xwOBA above wOBA) should have "
        f"sustainability > 0.5, got {score:.3f}. Buy-low signal."
    )


def test_hitter_overperforming_xwoba_yields_low_sustainability():
    """A NEGATIVE xwoba_delta means xwOBA < wOBA = hitter is
    OVERPERFORMING his quality of contact — regression DOWN coming.
    Sustainability should be LOW (sell high). Same direction database.py
    flags as SELL_HIGH at the -0.030 threshold."""
    p = _hitter(ab=400, h=120, xwoba_delta=-0.040)
    score = compute_sustainability_score(p)
    assert score < 0.5, (
        f"Hitter overperforming xwOBA (-0.040 = xwOBA below wOBA) should have "
        f"sustainability < 0.5, got {score:.3f}. Sell-high signal."
    )


def test_sustainability_direction_agrees_with_regression_flag():
    """The sustainability direction must agree with how _build_player_pool
    derives regression_flag from the SAME xwoba_delta column.

    src/database.py:1503-1508:
        xwoba_delta = xwOBA - wOBA
        xwoba_delta >= 0.030  → BUY_LOW   (underperforming → buy)
        xwoba_delta <= -0.030 → SELL_HIGH (overperforming → sell)

    A BUY_LOW hitter has more upside than he's shown, so his current
    (depressed) line is SUSTAINABLE-or-better → sustainability should be
    HIGH. A SELL_HIGH hitter is riding luck → LOW sustainability.
    """
    _BUY_LOW_THRESHOLD = 0.030  # mirrors database.py:1505
    _SELL_HIGH_THRESHOLD = -0.030  # mirrors database.py:1506

    buy_low = _hitter(ab=400, h=90, xwoba_delta=_BUY_LOW_THRESHOLD)
    sell_high = _hitter(ab=400, h=130, xwoba_delta=_SELL_HIGH_THRESHOLD)

    s_buy = compute_sustainability_score(buy_low)
    s_sell = compute_sustainability_score(sell_high)

    assert s_buy > 0.5, (
        f"xwoba_delta=+0.030 is flagged BUY_LOW by _build_player_pool; "
        f"sustainability must be HIGH (>0.5), got {s_buy:.3f}."
    )
    assert s_sell < 0.5, (
        f"xwoba_delta=-0.030 is flagged SELL_HIGH by _build_player_pool; "
        f"sustainability must be LOW (<0.5), got {s_sell:.3f}."
    )
    assert s_buy > s_sell, (
        f"BUY_LOW hitter must be more sustainable than the SELL_HIGH hitter; got buy={s_buy:.3f} sell={s_sell:.3f}."
    )


def test_hitter_neutral_xwoba_yields_around_half():
    """xwoba_delta ≈ 0 with average BABIP → score near 0.5."""
    p = _hitter(ab=400, h=120, hr=15, sf=4, k=80, xwoba_delta=0.0)
    score = compute_sustainability_score(p)
    # Expect somewhere in [0.4, 0.6] — close to neutral but BABIP secondary signal may nudge
    assert 0.35 < score < 0.65, f"Expected near-neutral score, got {score:.3f}"


# ── Pitcher signal ───────────────────────────────────────────────────


def test_pitcher_lucky_low_era_yields_low_sustainability():
    """Pitcher with ERA 2.50 and xFIP 4.00 is LUCKY — his quality of
    pitches doesn't support the low ERA. Expect regression UP → low
    sustainability."""
    p = _pitcher(ip=80, era=2.50, xfip=4.00, stuff_plus=100)
    score = compute_sustainability_score(p)
    assert score < 0.5, (
        f"Lucky pitcher (ERA 2.50 vs xFIP 4.00) should have sustainability < 0.5, "
        f"got {score:.3f}. Sell-high (overperforming)."
    )


def test_pitcher_unlucky_high_era_yields_high_sustainability():
    """Pitcher with ERA 5.50 but xFIP 3.50 is UNLUCKY. Regression DOWN
    coming → buy low → high sustainability."""
    p = _pitcher(ip=80, era=5.50, xfip=3.50, stuff_plus=110)
    score = compute_sustainability_score(p)
    assert score > 0.5, (
        f"Unlucky pitcher (ERA 5.50 vs xFIP 3.50) should have sustainability > 0.5, "
        f"got {score:.3f}. Buy-low (underperforming)."
    )


def test_pitcher_stuff_plus_adds_to_sustainability():
    """Two pitchers with same ERA-xFIP gap. The one with HIGHER Stuff+
    should score slightly higher (underlying skill supports continued
    production)."""
    p_high_stuff = _pitcher(ip=80, era=4.00, xfip=4.00, stuff_plus=115)
    p_low_stuff = _pitcher(ip=80, era=4.00, xfip=4.00, stuff_plus=85)
    s_high = compute_sustainability_score(p_high_stuff)
    s_low = compute_sustainability_score(p_low_stuff)
    assert s_high > s_low, (
        f"Higher Stuff+ should yield higher sustainability. got {s_high:.3f} (Stuff+ 115) vs {s_low:.3f} (Stuff+ 85)"
    )


# ── Inversion-bug regression guards ──────────────────────────────────


def test_pitcher_logic_no_longer_inverted():
    """Regression guard: old code returned 0.4 for ERA<2.5 and 0.7 for
    ERA>5.5 (semantically: 'likely to improve' = buy low). With xFIP
    matching ERA, the new code returns neutral — and only diverges based
    on the SKILL signal (xFIP gap), not the raw ERA.

    The bug was: UI text said 'Strong underlying metrics support
    continued production' when the score was HIGH for a struggling
    pitcher. That's contradictory — high ERA + 'continued production'
    is nonsense. Now: high score = signal-favorable outcome (buy-low
    for pitchers means ERA will improve)."""
    # ERA matches xFIP → no luck signal → near-neutral.
    p = _pitcher(ip=80, era=6.00, xfip=6.00, stuff_plus=100)
    score = compute_sustainability_score(p)
    # Should be near 0.5 (no regression signal), NOT 0.7 like the old code.
    assert 0.40 < score < 0.60, (
        f"Pitcher with ERA=xFIP=6.00 should have near-neutral sustainability "
        f"(no luck-based signal), got {score:.3f}. The old code returned 0.7 here, "
        f"which was the inverted-logic bug."
    )


def test_sustainability_is_continuous_not_stepped():
    """Old code had 5 discrete output values for hitters (0.3/0.7/0.8/
    interpolated). New code should produce many distinct values across
    different inputs."""
    scores = set()
    for delta in [-0.060, -0.040, -0.020, -0.010, 0.0, 0.010, 0.020, 0.040, 0.060]:
        p = _hitter(ab=400, h=120, hr=15, sf=4, k=80, xwoba_delta=delta)
        scores.add(round(compute_sustainability_score(p), 5))
    # Each input should yield a distinct output (continuous function).
    assert len(scores) >= 8, (
        f"Expected continuous output (≥8 distinct values for 9 inputs), got {len(scores)}. "
        f"Step-function regression: {sorted(scores)}"
    )


def test_score_always_in_unit_interval():
    """Sigmoid output bounded in (0, 1)."""
    for delta in [-0.500, -0.100, 0.0, 0.100, 0.500]:
        p = _hitter(ab=400, h=120, hr=15, sf=4, k=80, xwoba_delta=delta)
        score = compute_sustainability_score(p)
        assert 0.0 < score < 1.0, f"Score {score} out of (0, 1) for delta={delta}"

    for era_diff in [-3.0, -1.0, 0.0, 1.0, 3.0]:
        p = _pitcher(ip=80, era=4.0 + era_diff, xfip=4.0, stuff_plus=100)
        score = compute_sustainability_score(p)
        assert 0.0 < score < 1.0, f"Pitcher score {score} out of (0, 1)"
