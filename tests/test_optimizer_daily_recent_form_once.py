"""LO-E2: 'daily' mode must apply L14 recent form exactly ONCE.

Recent-form (L14) blending lives in TWO places:
  - Stage 1: src/optimizer/projections.py::_apply_recent_form_adjustment
    (run inside build_enhanced_projections, Step 4).
  - Stage 11: src/optimizer/daily_optimizer.py::build_daily_dcv_table
    (the DCV form_adjustments).

The 'daily' MODE_PRESET enables BOTH enable_projections (→ Stage 1 recent
form) AND enable_daily_dcv (→ DCV recent form), so a daily pipeline run
double-applies L14. In daily mode the DCV stage is the canonical recent-form
applier, so Stage 1's Step 4 must be gated OFF; non-daily modes are unchanged.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

import src.optimizer.projections as _proj
from src.optimizer.pipeline import LineupOptimizerPipeline


@pytest.fixture()
def sample_roster() -> pd.DataFrame:
    """Roster with enough players for a feasible solve."""
    return pd.DataFrame(
        {
            "player_id": list(range(1, 7)),
            "player_name": ["A", "B", "C", "D", "E", "F"],
            "positions": ["1B", "OF", "SS", "OF,DH", "SP", "RP"],
            "is_hitter": [True, True, True, True, False, False],
            "r": [80, 70, 60, 50, 0, 0],
            "hr": [30, 25, 20, 15, 0, 0],
            "rbi": [90, 80, 70, 60, 0, 0],
            "sb": [10, 5, 15, 20, 0, 0],
            "avg": [0.290, 0.270, 0.260, 0.250, 0.0, 0.0],
            "obp": [0.360, 0.340, 0.330, 0.320, 0.0, 0.0],
            "w": [0, 0, 0, 0, 12, 0],
            "l": [0, 0, 0, 0, 6, 3],
            "sv": [0, 0, 0, 0, 0, 20],
            "k": [0, 0, 0, 0, 180, 60],
            "era": [0.0, 0.0, 0.0, 0.0, 3.20, 2.80],
            "whip": [0.0, 0.0, 0.0, 0.0, 1.10, 1.05],
            "ip": [0, 0, 0, 0, 180, 60],
            "h": [145, 135, 125, 115, 0, 0],
            "ab": [500, 500, 480, 460, 0, 0],
            "pa": [560, 560, 540, 520, 0, 0],
            "er": [0, 0, 0, 0, 64, 18],
            "bb_allowed": [0, 0, 0, 0, 50, 20],
            "h_allowed": [0, 0, 0, 0, 148, 43],
        }
    )


def test_daily_mode_does_not_run_stage1_recent_form(sample_roster) -> None:
    """In 'daily' mode the DCV stage is the sole L14 applier, so Stage 1's
    _apply_recent_form_adjustment (Step 4) must NOT run — otherwise L14 is
    blended twice (once in projections, once in DCV)."""
    calls = {"n": 0}

    def _spy(roster, *args, **kwargs):
        calls["n"] += 1
        return roster

    with patch.object(_proj, "_apply_recent_form_adjustment", side_effect=_spy):
        pipe = LineupOptimizerPipeline(sample_roster, mode="daily", weeks_remaining=16)
        pipe.optimize()

    assert calls["n"] == 0, (
        f"LO-E2: 'daily' mode must not run Stage 1 recent-form blending "
        f"(DCV applies L14); _apply_recent_form_adjustment ran {calls['n']} times"
    )


def test_non_daily_mode_still_runs_stage1_recent_form(sample_roster) -> None:
    """Non-daily modes (standard) keep Stage 1 recent-form blending — the fix
    must not change behavior outside daily mode."""
    calls = {"n": 0}

    def _spy(roster, *args, **kwargs):
        calls["n"] += 1
        return roster

    with patch.object(_proj, "_apply_recent_form_adjustment", side_effect=_spy):
        pipe = LineupOptimizerPipeline(sample_roster, mode="standard", weeks_remaining=16)
        pipe.optimize()

    assert calls["n"] == 1, (
        f"LO-E2 regression: 'standard' mode must still run Stage 1 recent-form "
        f"blending exactly once; _apply_recent_form_adjustment ran {calls['n']} times"
    )


def test_build_enhanced_projections_recent_form_gate(sample_roster) -> None:
    """Unit-level: build_enhanced_projections must honor enable_recent_form.
    When False, Step 4's _apply_recent_form_adjustment is skipped entirely."""
    calls = {"n": 0}

    def _spy(roster, *args, **kwargs):
        calls["n"] += 1
        return roster

    with patch.object(_proj, "_apply_recent_form_adjustment", side_effect=_spy):
        _proj.build_enhanced_projections(
            sample_roster.copy(),
            enable_bayesian=False,
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
            enable_playing_time=False,
            enable_recent_form=False,
        )
    assert calls["n"] == 0, "enable_recent_form=False must skip Step 4 recent-form blending"

    # Default (True) still runs it — back-compat for non-daily callers.
    calls["n"] = 0
    with patch.object(_proj, "_apply_recent_form_adjustment", side_effect=_spy):
        _proj.build_enhanced_projections(
            sample_roster.copy(),
            enable_bayesian=False,
            enable_kalman=False,
            enable_statcast=False,
            enable_injury=False,
            enable_playing_time=False,
        )
    assert calls["n"] == 1, "default enable_recent_form=True must run Step 4 once"
