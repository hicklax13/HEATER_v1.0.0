"""MS-C2 fix: apply_time_decay defaults total_weeks to LeagueConfig.season_weeks
and clamps time_fraction to [0, 1].

Before the fix the default total_weeks was a hardcoded 24 (canonical season is
26), and time_fraction = weeks_remaining / total_weeks had no upper clamp, so
early-season callers (weeks_remaining up to 26) produced time_fraction > 1.0 and
AMPLIFIED counting-stat SGP instead of decaying it.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.trade_intelligence import apply_time_decay
from src.valuation import LeagueConfig


def test_time_fraction_never_exceeds_one_across_full_season():
    """For weeks_remaining in [1, 26] with the default total_weeks, the
    counting-stat decay factor must never exceed 1.0 (no amplification)."""
    sgp = {"HR": 1.0}  # HR is a counting stat → scaled by time_fraction
    for wr in range(1, 27):
        out = apply_time_decay(sgp, weeks_remaining=wr)
        assert out["HR"] <= 1.0 + 1e-9, (
            f"MS-C2: weeks_remaining={wr} amplified counting SGP (factor={out['HR']:.4f} > 1.0)"
        )
        assert out["HR"] >= 0.0


def test_default_total_weeks_is_canonical_season_length():
    """At weeks_remaining == season_weeks, a counting stat should be at full
    weight (time_fraction == 1.0), proving the default matches the canonical
    26-week season rather than a stale 24."""
    sgp = {"R": 2.0}
    full = LeagueConfig().season_weeks
    out = apply_time_decay(sgp, weeks_remaining=full)
    assert abs(out["R"] - 2.0) < 1e-9, (
        f"MS-C2: at weeks_remaining={full} the counting stat should be at full weight; got {out['R']}"
    )


def test_time_fraction_clamped_when_weeks_remaining_exceeds_total():
    """A pathological weeks_remaining > total_weeks must still clamp to 1.0."""
    sgp = {"RBI": 3.0}
    out = apply_time_decay(sgp, weeks_remaining=40, total_weeks=26)
    assert abs(out["RBI"] - 3.0) < 1e-9


def test_midseason_decay_still_reduces():
    """Sanity: half the season remaining → roughly half the counting SGP."""
    sgp = {"SB": 4.0}
    out = apply_time_decay(sgp, weeks_remaining=13, total_weeks=26)
    assert abs(out["SB"] - 2.0) < 1e-9
