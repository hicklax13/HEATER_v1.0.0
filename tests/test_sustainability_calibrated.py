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

These tests pin the calibration:
  * Overperforming hitter (positive xwoba_delta) → low sustainability
    (regression DOWN coming — sell high)
  * Underperforming hitter (negative xwoba_delta) → high sustainability
    (regression UP coming — buy low)
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


def test_hitter_overperforming_xwoba_yields_low_sustainability():
    """Positive xwoba_delta (actual wOBA > xwOBA) means hitter is
    overperforming his quality of contact — regression DOWN coming.
    Sustainability should be LOW (sell high)."""
    p = _hitter(ab=400, h=120, xwoba_delta=0.040)
    score = compute_sustainability_score(p)
    assert score < 0.5, (
        f"Hitter overperforming xwOBA (+0.040) should have sustainability < 0.5, got {score:.3f}. Sell-high signal."
    )


def test_hitter_underperforming_xwoba_yields_high_sustainability():
    """Negative xwoba_delta (actual wOBA < xwOBA) means hitter is
    underperforming his quality of contact — regression UP coming.
    Sustainability HIGH (buy low)."""
    p = _hitter(ab=400, h=80, xwoba_delta=-0.040)
    score = compute_sustainability_score(p)
    assert score > 0.5, (
        f"Hitter underperforming xwOBA (-0.040) should have sustainability > 0.5, got {score:.3f}. Buy-low signal."
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
