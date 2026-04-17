"""Tests for src/projection_blending.py — YTD-aware projection blending."""

from __future__ import annotations

import pandas as pd

from src.projection_blending import (
    apply_ytd_corrections,
    blend_projection_with_ytd,
    detect_role_change,
)


def _hitter(player_id: int, name: str, **kwargs) -> dict:
    row = {
        "player_id": player_id,
        "name": name,
        "is_hitter": 1,
        "pa": 500,
        "ab": 450,
        "avg": 0.270,
        "hr": 20,
        "rbi": 70,
        "sb": 10,
        "ytd_gp": 0,
        "ytd_pa": 0,
        "ytd_avg": 0.0,
        "ytd_hr": 0,
        "ytd_rbi": 0,
        "ytd_sb": 0,
    }
    row.update(kwargs)
    return row


def _pitcher(player_id: int, name: str, **kwargs) -> dict:
    row = {
        "player_id": player_id,
        "name": name,
        "is_hitter": 0,
        "ip": 60.0,
        "w": 3,
        "l": 3,
        "sv": 0,
        "k": 70,
        "era": 3.80,
        "whip": 1.20,
        "ytd_gp": 0,
        "ytd_era": 0.0,
        "ytd_whip": 0.0,
        "ytd_sv": 0,
        "ytd_k": 0,
    }
    row.update(kwargs)
    return row


class TestBlendProjectionWithYTD:
    def test_no_ytd_returns_projection_unchanged(self):
        pool = pd.DataFrame([_hitter(1, "A")])
        out = blend_projection_with_ytd(pool)
        assert out.loc[0, "avg"] == 0.270
        assert out.loc[0, "hr"] == 20

    def test_hot_hitter_pulls_avg_up(self):
        # Projected .270, hitting .345 over 60 PA
        pool = pd.DataFrame([_hitter(1, "A", ytd_gp=14, ytd_pa=60, ytd_avg=0.345, ytd_hr=5)])
        out = blend_projection_with_ytd(pool)
        # Blend weight at 60 PA / 250 regression ~= 0.24
        # Blended AVG should be between .270 and .345
        assert 0.270 < out.loc[0, "avg"] < 0.345
        # HR projection should increase (scaled-up rate)
        assert out.loc[0, "hr"] > 20

    def test_cold_hitter_pulls_avg_down(self):
        # Projected .270, hitting .200 over 60 PA
        pool = pd.DataFrame([_hitter(1, "A", ytd_gp=14, ytd_pa=60, ytd_avg=0.200, ytd_hr=1)])
        out = blend_projection_with_ytd(pool)
        assert 0.200 < out.loc[0, "avg"] < 0.270

    def test_hot_pitcher_pulls_era_down(self):
        # Projected 3.80 ERA, actual 1.06
        pool = pd.DataFrame([_pitcher(1, "A", ytd_gp=6, ytd_era=1.06, ytd_whip=0.90, ytd_k=40)])
        out = blend_projection_with_ytd(pool)
        assert out.loc[0, "era"] < 3.80

    def test_cold_pitcher_era_capped_at_3x_projection(self):
        # Projected 4.00 ERA, actual 14.40 (extreme early-season outlier)
        pool = pd.DataFrame([_pitcher(1, "A", ip=55.0, era=4.00, ytd_gp=6, ytd_era=14.40, ytd_whip=2.4)])
        out = blend_projection_with_ytd(pool)
        # The YTD should be capped at 12.00 (3 * 4.00), so final ERA
        # cannot exceed what a blend with 12.00 produces
        # Weight at 6 GP * 1.5 = 9 IP / 80 = ~0.11
        # Max expected: 0.89 * 4.00 + 0.11 * 12.00 = ~4.88
        assert out.loc[0, "era"] < 5.0
        # And should still reflect some upward pressure
        assert out.loc[0, "era"] > 4.00

    def test_max_weight_is_respected(self):
        # Massive YTD sample should not fully override projection
        pool = pd.DataFrame([_hitter(1, "A", ytd_gp=200, ytd_pa=900, ytd_avg=0.400, ytd_hr=50)])
        out = blend_projection_with_ytd(pool)
        # MAX_YTD_WEIGHT is 0.40; blended should be at most
        # 0.60 * 0.270 + 0.40 * 0.400 = 0.322
        assert out.loc[0, "avg"] < 0.323

    def test_zero_ytd_pa_is_safe(self):
        pool = pd.DataFrame([_hitter(1, "A", ytd_gp=1, ytd_pa=0)])
        # Should not divide by zero
        out = blend_projection_with_ytd(pool)
        assert out.loc[0, "avg"] == 0.270


class TestDetectRoleChange:
    def test_closer_with_saves_flagged_closer(self):
        # Projected 30 SV, 6 GP, 4 SV — clearly still closing
        pool = pd.DataFrame([_pitcher(1, "A", sv=30, ytd_gp=6, ytd_sv=4)])
        out = detect_role_change(pool)
        assert out.loc[0, "role_status_inferred"] == "closer"
        assert out.loc[0, "sv"] == 30

    def test_projected_closer_with_zero_saves_demoted(self):
        # Projected 18 SV, 7 GP, 0 SV — role change signal
        pool = pd.DataFrame([_pitcher(1, "Suarez", sv=18, ytd_gp=7, ytd_sv=0)])
        out = detect_role_change(pool)
        assert out.loc[0, "role_status_inferred"] == "demoted"
        # SV projection reduced to 25% * 18 = 4.5, floored at 3
        assert out.loc[0, "sv"] == 4.5

    def test_demoted_sv_floor_enforced(self):
        # Projected 11 SV, demoted: 11 * 0.25 = 2.75, should floor at 3
        pool = pd.DataFrame([_pitcher(1, "A", sv=11, ytd_gp=8, ytd_sv=0)])
        out = detect_role_change(pool)
        assert out.loc[0, "sv"] == 3.0

    def test_low_sv_projection_not_checked(self):
        # 5 SV projection doesn't meet closer threshold
        pool = pd.DataFrame([_pitcher(1, "A", sv=5, ytd_gp=8, ytd_sv=0)])
        out = detect_role_change(pool)
        assert out.loc[0, "role_status_inferred"] == "unknown"
        assert out.loc[0, "sv"] == 5

    def test_insufficient_sample_not_flagged(self):
        # 3 GP isn't enough to infer role change
        pool = pd.DataFrame([_pitcher(1, "A", sv=20, ytd_gp=3, ytd_sv=0)])
        out = detect_role_change(pool)
        assert out.loc[0, "role_status_inferred"] == "unknown"
        assert out.loc[0, "sv"] == 20

    def test_hitters_untouched(self):
        pool = pd.DataFrame([_hitter(1, "A", ytd_gp=10)])
        out = detect_role_change(pool)
        assert out.loc[0, "role_status_inferred"] == "unknown"


class TestApplyYTDCorrections:
    def test_role_change_then_blend(self):
        # Ensure role change caps SV before blend rescales it
        pool = pd.DataFrame(
            [
                _pitcher(
                    1,
                    "Suarez",
                    sv=18,
                    k=55,
                    era=3.10,
                    ytd_gp=7,
                    ytd_sv=0,
                    ytd_k=8,
                    ytd_era=1.29,
                    ytd_whip=1.14,
                )
            ]
        )
        out = apply_ytd_corrections(pool)
        # Role status captured
        assert out.loc[0, "role_status_inferred"] == "demoted"
        # SV significantly reduced from original 18
        assert out.loc[0, "sv"] < 6
        # ERA pulled down toward YTD 1.29 (even capped)
        assert out.loc[0, "era"] < 3.10

    def test_empty_pool_safe(self):
        pool = pd.DataFrame()
        out = apply_ytd_corrections(pool)
        assert out.empty
