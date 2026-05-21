"""P5e: compute_sustainability_score should differentiate counting vs rate
stat sustainability signals.

xwOBA-wOBA gap signals AVG/OBP regression risk well, but doesn't predict
HR sustainability. HR/FB% drives HR regression. Same for pitchers:
SIERA/xFIP - ERA gap drives ERA/WHIP, but LOB% drives strand-rate
regression which feeds back into ERA over a longer horizon.

These tests pin the new counting-stat-aware behavior:
  * Hitters with elevated HR/FB% (e.g. 30% vs league avg ~13%) get
    LOWER sustainability than hitters with normal HR/FB% but otherwise
    identical rate-stat signals.
  * Pitchers with elevated LOB% (>0.80) AND mediocre underlying skill
    get LOWER sustainability than pitchers with normal LOB%.
  * Output stays in [0, 1].
  * Backward-compat: missing counting-stat columns falls back to the
    pre-P5e rate-only sustainability path. No crash.
"""

from __future__ import annotations

import pandas as pd

from src.waiver_wire import compute_sustainability_score


def test_high_hr_fb_pct_hitter_lower_hr_sustainability():
    """A hitter with unsustainably high HR/FB rate should get lower
    sustainability than one with normal HR/FB rate."""
    # ab=400 to clear the 80-AB sample-size gate; identical xwoba_delta + babip
    # so the only differentiator is hr_per_fb.
    elevated_hr_fb = pd.Series(
        {
            "is_hitter": 1,
            "ab": 400,
            "h": 110,
            "hr": 30,
            "sf": 4,
            "k": 80,
            "xwoba": 0.350,
            "obp": 0.350,
            "babip": 0.300,
            "career_babip": 0.300,
            "ev_mean": 91.0,
            "hr_per_fb": 0.30,  # unsustainable
            "ytd_hr": 15,
            "xwoba_delta": 0.0,
        }
    )
    normal_hr_fb = pd.Series(
        {
            "is_hitter": 1,
            "ab": 400,
            "h": 110,
            "hr": 15,
            "sf": 4,
            "k": 80,
            "xwoba": 0.350,
            "obp": 0.350,
            "babip": 0.300,
            "career_babip": 0.300,
            "ev_mean": 91.0,
            "hr_per_fb": 0.13,  # normal
            "ytd_hr": 15,
            "xwoba_delta": 0.0,
        }
    )
    s_elevated = compute_sustainability_score(elevated_hr_fb)
    s_normal = compute_sustainability_score(normal_hr_fb)
    # The high-HR/FB player should have LOWER sustainability (more regression risk)
    assert s_elevated < s_normal, (
        f"30% HR/FB should be less sustainable than 13%. Got {s_elevated:.3f} vs {s_normal:.3f}"
    )


def test_pitcher_high_strand_rate_lower_sustainability():
    """Pitcher with unsustainably high LOB% (strand rate) should get
    lower sustainability than one with normal LOB%.

    The high-LOB pitcher in this fixture also has worse underlying SIERA/
    xFIP (3.80/3.90 vs 2.80/2.85), so both signals push him toward LOW
    sustainability. The combined signal should reflect that."""
    high_strand = pd.Series(
        {
            "is_hitter": 0,
            "ip": 80,
            "era": 2.50,
            "siera": 3.80,
            "xfip": 3.90,
            "lob_pct": 0.85,  # unsustainable
            "stuff_plus": 100,
        }
    )
    normal_strand = pd.Series(
        {
            "is_hitter": 0,
            "ip": 80,
            "era": 2.50,
            "siera": 2.80,
            "xfip": 2.85,
            "lob_pct": 0.74,  # league avg
            "stuff_plus": 100,
        }
    )
    s_high = compute_sustainability_score(high_strand)
    s_normal = compute_sustainability_score(normal_strand)
    # The high-LOB pitcher with worse underlying is less sustainable
    assert s_high <= s_normal, (
        f"High LOB% + bad underlying should be less sustainable. Got {s_high:.3f} vs {s_normal:.3f}"
    )


def test_sustainability_returns_in_unit_range():
    """All sustainability scores must be in [0, 1] regardless of inputs."""
    fa = pd.Series(
        {
            "is_hitter": 1,
            "ab": 400,
            "h": 110,
            "hr": 30,
            "sf": 4,
            "k": 80,
            "xwoba": 0.5,
            "obp": 0.2,
            "babip": 0.5,
            "career_babip": 0.300,
            "hr_per_fb": 0.5,
            "ytd_hr": 30,
            "xwoba_delta": 0.3,
        }
    )
    s = compute_sustainability_score(fa)
    assert 0.0 <= s <= 1.0, f"Score {s} out of [0, 1]"


def test_backward_compat_no_counting_columns():
    """When counting-stat columns are missing, the function should fall
    back to the pre-P5e rate-only sustainability (no crash, sensible
    value in [0, 1])."""
    fa = pd.Series(
        {
            "is_hitter": 1,
            "ab": 400,
            "h": 110,
            "hr": 15,
            "sf": 4,
            "k": 80,
            "xwoba": 0.340,
            "obp": 0.340,
            "babip": 0.290,
            "career_babip": 0.290,
            "xwoba_delta": 0.0,
            # NOTE: no hr_per_fb, no lob_pct
        }
    )
    s = compute_sustainability_score(fa)
    assert 0.0 <= s <= 1.0


def test_backward_compat_pitcher_no_lob_pct():
    """Pitcher with missing lob_pct should still get a sensible score
    (rate-only fallback)."""
    fa = pd.Series(
        {
            "is_hitter": 0,
            "ip": 80,
            "era": 3.50,
            "xfip": 3.40,
            "siera": 3.45,
            "stuff_plus": 105,
            # NOTE: no lob_pct
        }
    )
    s = compute_sustainability_score(fa)
    assert 0.0 <= s <= 1.0
