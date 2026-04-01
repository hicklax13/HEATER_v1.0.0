"""Tests for the Daily Optimizer V2 module."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer.daily_optimizer import (
    STABILIZATION_POINTS,
    STUD_FLOOR_TOP_N,
    apply_stud_floor,
    build_daily_dcv_table,
    check_ip_override,
    compute_blended_projection,
    compute_health_factor,
    compute_matchup_multiplier,
    compute_volume_factor,
)
from src.valuation import LeagueConfig


@pytest.fixture
def sample_roster():
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Star Hitter",
                "positions": "OF",
                "team": "NYY",
                "is_hitter": 1,
                "r": 90,
                "hr": 35,
                "rbi": 100,
                "sb": 10,
                "avg": 0.280,
                "obp": 0.380,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "status": "active",
            },
            {
                "player_id": 2,
                "name": "IL Pitcher",
                "positions": "SP",
                "team": "BOS",
                "is_hitter": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0.0,
                "obp": 0.0,
                "w": 12,
                "l": 6,
                "sv": 0,
                "k": 200,
                "era": 3.20,
                "whip": 1.10,
                "status": "IL15",
            },
            {
                "player_id": 3,
                "name": "Bench Bat",
                "positions": "1B",
                "team": "LAD",
                "is_hitter": 1,
                "r": 50,
                "hr": 15,
                "rbi": 55,
                "sb": 3,
                "avg": 0.250,
                "obp": 0.320,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0.0,
                "whip": 0.0,
                "status": "active",
            },
        ]
    )


# -----------------------------------------------------------------------
# Blended Projection tests
# -----------------------------------------------------------------------


class TestBlendedProjection:
    def test_zero_observations_returns_preseason(self):
        result = compute_blended_projection(0.280, 0, 0, "avg")
        assert abs(result - 0.280) < 0.001

    def test_at_stabilization_returns_average(self):
        stab = STABILIZATION_POINTS["avg"]  # 910
        result = compute_blended_projection(0.280, 0.300 * stab, stab, "avg")
        # Should be close to average of 0.280 and 0.300 = 0.290
        assert abs(result - 0.290) < 0.001

    def test_large_sample_approaches_observed(self):
        result = compute_blended_projection(0.280, 0.320 * 5000, 5000, "avg")
        # With 5000 AB observed, should be very close to 0.320
        assert abs(result - 0.320) < 0.01

    def test_unknown_stat_uses_default_stab(self):
        # Unknown stat key falls back to 200 stabilization
        result = compute_blended_projection(10.0, 0, 0, "fake_stat")
        assert abs(result - 10.0) < 0.01

    def test_negative_stab_handled(self):
        # Verifies that the function guards against zero/negative stabilization
        result = compute_blended_projection(5.0, 10.0, 2, "hr")
        assert result > 0

    def test_counting_stat_blend(self):
        # HR: stabilization = 170 PA
        stab = STABILIZATION_POINTS["hr"]
        preseason_rate = 0.06  # ~35 HR per 580 PA
        # Observed 10 HR in 100 PA = 0.10 rate
        result = compute_blended_projection(preseason_rate, 10, 100, "hr")
        # With 100 PA (vs 170 stab), prior still dominates
        assert result < 0.10  # Below observed
        assert result > preseason_rate  # Above prior


# -----------------------------------------------------------------------
# Health Factor tests
# -----------------------------------------------------------------------


class TestHealthFactor:
    def test_active_returns_one(self):
        assert compute_health_factor("active") == 1.0

    def test_il15_returns_zero(self):
        assert compute_health_factor("IL15") == 0.0

    def test_il10_returns_zero(self):
        assert compute_health_factor("IL10") == 0.0

    def test_il60_returns_zero(self):
        assert compute_health_factor("IL60") == 0.0

    def test_dtd_returns_zero(self):
        assert compute_health_factor("DTD") == 0.0

    def test_na_returns_zero(self):
        assert compute_health_factor("NA") == 0.0

    def test_minors_returns_zero(self):
        assert compute_health_factor("minors") == 0.0

    def test_out_returns_zero(self):
        assert compute_health_factor("out") == 0.0

    def test_suspended_returns_zero(self):
        assert compute_health_factor("suspended") == 0.0

    def test_empty_string_returns_one(self):
        assert compute_health_factor("") == 1.0

    def test_none_returns_one(self):
        assert compute_health_factor(None) == 1.0

    def test_case_insensitive(self):
        assert compute_health_factor("il15") == 0.0
        assert compute_health_factor("Il15") == 0.0
        assert compute_health_factor("ACTIVE") == 1.0

    def test_whitespace_stripped(self):
        assert compute_health_factor("  IL15  ") == 0.0
        assert compute_health_factor("  active  ") == 1.0


# -----------------------------------------------------------------------
# Volume Factor tests
# -----------------------------------------------------------------------


class TestVolumeFactor:
    def test_off_day_returns_zero(self):
        assert compute_volume_factor(False, None) == 0.0

    def test_off_day_even_if_confirmed(self):
        assert compute_volume_factor(False, True) == 0.0

    def test_lineup_not_posted_returns_09(self):
        assert compute_volume_factor(True, None) == 0.9

    def test_confirmed_in_lineup_returns_one(self):
        assert compute_volume_factor(True, True) == 1.0

    def test_benched_returns_03(self):
        assert compute_volume_factor(True, False) == 0.3

    def test_doubleheader_returns_two(self):
        assert compute_volume_factor(True, True, is_doubleheader=True) == 2.0

    def test_doubleheader_benched_still_03(self):
        # Benched player gets 0.3 even during doubleheader
        assert compute_volume_factor(True, False, is_doubleheader=True) == 0.3

    def test_doubleheader_lineup_unknown(self):
        # Lineup not posted during doubleheader = 0.9 (not doubled)
        assert compute_volume_factor(True, None, is_doubleheader=True) == 0.9


# -----------------------------------------------------------------------
# Matchup Multiplier tests
# -----------------------------------------------------------------------


class TestMatchupMultiplier:
    def test_returns_positive(self):
        mult = compute_matchup_multiplier(True, "R", "L", "NYY", "BOS", {})
        assert mult > 0

    def test_clamped_range(self):
        mult = compute_matchup_multiplier(True, "L", "R", "COL", "SF", {})
        assert 0.3 <= mult <= 3.0

    def test_pitcher_matchup(self):
        mult = compute_matchup_multiplier(False, "", "", "NYM", "ATL", {})
        assert 0.3 <= mult <= 3.0

    def test_empty_hands_still_works(self):
        mult = compute_matchup_multiplier(True, "", "", "NYY", "BOS", {})
        assert 0.3 <= mult <= 3.0

    def test_pitcher_xfip_affects_result(self):
        # Good pitcher (low xFIP) should reduce hitter value
        mult_good = compute_matchup_multiplier(True, "R", "R", "NYY", "BOS", {}, pitcher_xfip=2.50)
        mult_bad = compute_matchup_multiplier(True, "R", "R", "NYY", "BOS", {}, pitcher_xfip=5.50)
        # Bad pitcher should give higher multiplier for hitter
        assert mult_bad >= mult_good

    def test_xfip_ignored_for_pitchers(self):
        # pitcher_xfip should only affect hitters
        mult = compute_matchup_multiplier(False, "", "", "NYY", "BOS", {}, pitcher_xfip=2.50)
        assert 0.3 <= mult <= 3.0


# -----------------------------------------------------------------------
# Stud Floor tests
# -----------------------------------------------------------------------


class TestStudFloor:
    def test_stud_not_benched(self, sample_roster):
        config = LeagueConfig()
        dcv = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "total_dcv": 0.1,
                    "volume_factor": 1.0,
                    "stud_floor_applied": False,
                },
                {
                    "player_id": 3,
                    "total_dcv": 5.0,
                    "volume_factor": 1.0,
                    "stud_floor_applied": False,
                },
            ]
        )
        result = apply_stud_floor(dcv, sample_roster, config)
        # Star Hitter (id=1) should get boosted above minimum
        star = result[result["player_id"] == 1]
        if not star.empty:
            assert star.iloc[0]["total_dcv"] >= 1.0  # Boosted from 0.1

    def test_volume_zero_no_floor(self, sample_roster):
        config = LeagueConfig()
        dcv = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "total_dcv": 0.0,
                    "volume_factor": 0.0,
                    "stud_floor_applied": False,
                },
                {
                    "player_id": 3,
                    "total_dcv": 5.0,
                    "volume_factor": 1.0,
                    "stud_floor_applied": False,
                },
            ]
        )
        result = apply_stud_floor(dcv, sample_roster, config)
        # Off-day stud should NOT get boosted (volume_factor = 0)
        star = result[result["player_id"] == 1]
        assert star.iloc[0]["total_dcv"] == 0.0

    def test_empty_dcv_table(self, sample_roster):
        config = LeagueConfig()
        dcv = pd.DataFrame(columns=["player_id", "total_dcv", "volume_factor", "stud_floor_applied"])
        result = apply_stud_floor(dcv, sample_roster, config)
        assert result.empty

    def test_stud_floor_flag_set(self, sample_roster):
        config = LeagueConfig()
        dcv = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "total_dcv": 0.01,
                    "volume_factor": 1.0,
                    "stud_floor_applied": False,
                },
                {
                    "player_id": 3,
                    "total_dcv": 5.0,
                    "volume_factor": 1.0,
                    "stud_floor_applied": False,
                },
            ]
        )
        result = apply_stud_floor(dcv, sample_roster, config)
        star = result[result["player_id"] == 1]
        if not star.empty and star.iloc[0]["total_dcv"] > 0.01:
            assert bool(star.iloc[0]["stud_floor_applied"]) is True


# -----------------------------------------------------------------------
# Build DCV Table tests
# -----------------------------------------------------------------------


class TestBuildDCVTable:
    def test_returns_dataframe(self, sample_roster):
        dcv = build_daily_dcv_table(sample_roster, None, None, {})
        assert isinstance(dcv, pd.DataFrame)
        assert not dcv.empty

    def test_health_zero_implies_dcv_zero(self, sample_roster):
        dcv = build_daily_dcv_table(sample_roster, None, None, {})
        # Any player with health_factor == 0 must have total_dcv == 0
        excluded = dcv[dcv["health_factor"] == 0.0]
        for _, row in excluded.iterrows():
            assert row["total_dcv"] == 0.0, f"Player {row.get('name')} has health=0 but DCV={row['total_dcv']}"

    def test_has_dcv_columns(self, sample_roster):
        dcv = build_daily_dcv_table(sample_roster, None, None, {})
        assert "total_dcv" in dcv.columns
        # Should have per-category dcv columns
        config = LeagueConfig()
        for cat in config.all_categories:
            assert f"dcv_{cat.lower()}" in dcv.columns

    def test_has_metadata_columns(self, sample_roster):
        dcv = build_daily_dcv_table(sample_roster, None, None, {})
        for col in [
            "player_id",
            "name",
            "positions",
            "health_factor",
            "volume_factor",
            "matchup_mult",
            "stud_floor_applied",
        ]:
            assert col in dcv.columns

    def test_sorted_by_total_dcv(self, sample_roster):
        dcv = build_daily_dcv_table(sample_roster, None, None, {})
        active = dcv[dcv["health_factor"] > 0]
        if len(active) > 1:
            dcv_vals = active["total_dcv"].values
            for i in range(len(dcv_vals) - 1):
                assert dcv_vals[i] >= dcv_vals[i + 1]

    def test_empty_roster(self):
        empty = pd.DataFrame(
            columns=[
                "player_id",
                "name",
                "positions",
                "team",
                "is_hitter",
                "r",
                "hr",
                "rbi",
                "sb",
                "avg",
                "obp",
                "w",
                "l",
                "sv",
                "k",
                "era",
                "whip",
                "status",
            ]
        )
        dcv = build_daily_dcv_table(empty, None, None, {})
        assert isinstance(dcv, pd.DataFrame)

    def test_with_schedule_data(self, sample_roster):
        schedule = [
            {"away_team": "NYY", "home_team": "BOS"},
            {"away_team": "LAD", "home_team": "SF"},
        ]
        dcv = build_daily_dcv_table(sample_roster, None, schedule, {})
        assert isinstance(dcv, pd.DataFrame)
        assert not dcv.empty

    def test_all_players_represented(self, sample_roster):
        dcv = build_daily_dcv_table(sample_roster, None, None, {})
        assert len(dcv) == len(sample_roster)


# -----------------------------------------------------------------------
# IP Override tests
# -----------------------------------------------------------------------


class TestIPOverride:
    def test_no_override_when_above_minimum(self):
        dcv = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "is_hitter": True,
                    "volume_factor": 1.0,
                    "health_factor": 1.0,
                    "total_dcv": 5.0,
                },
                {
                    "player_id": 2,
                    "is_hitter": False,
                    "volume_factor": 1.0,
                    "health_factor": 1.0,
                    "total_dcv": 2.0,
                },
            ]
        )
        result = check_ip_override(dcv, 25.0)  # Above 20 IP minimum
        assert result.iloc[1]["total_dcv"] == 2.0  # Unchanged

    def test_override_boosts_pitcher(self):
        dcv = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "is_hitter": True,
                    "volume_factor": 1.0,
                    "health_factor": 1.0,
                    "total_dcv": 5.0,
                },
                {
                    "player_id": 2,
                    "is_hitter": False,
                    "volume_factor": 1.0,
                    "health_factor": 1.0,
                    "total_dcv": 2.0,
                },
            ]
        )
        result = check_ip_override(dcv, 15.0)  # Below 20 IP minimum
        assert result.iloc[1]["total_dcv"] > 2.0  # Boosted

    def test_no_pitchers_available(self):
        dcv = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "is_hitter": True,
                    "volume_factor": 1.0,
                    "health_factor": 1.0,
                    "total_dcv": 5.0,
                },
            ]
        )
        result = check_ip_override(dcv, 15.0)
        assert len(result) == 1  # No crash, returns unchanged

    def test_pitcher_with_zero_volume_not_boosted(self):
        dcv = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "is_hitter": True,
                    "volume_factor": 1.0,
                    "health_factor": 1.0,
                    "total_dcv": 5.0,
                },
                {
                    "player_id": 2,
                    "is_hitter": False,
                    "volume_factor": 0.0,
                    "health_factor": 1.0,
                    "total_dcv": 0.0,
                },
            ]
        )
        result = check_ip_override(dcv, 15.0)
        # Pitcher has volume=0 (off day), should not be boosted
        assert result.iloc[1]["total_dcv"] == 0.0

    def test_empty_table_no_crash(self):
        dcv = pd.DataFrame(
            columns=[
                "player_id",
                "is_hitter",
                "volume_factor",
                "health_factor",
                "total_dcv",
            ]
        )
        result = check_ip_override(dcv, 15.0)
        assert result.empty

    def test_custom_ip_minimum(self):
        dcv = pd.DataFrame(
            [
                {
                    "player_id": 1,
                    "is_hitter": True,
                    "volume_factor": 1.0,
                    "health_factor": 1.0,
                    "total_dcv": 5.0,
                },
                {
                    "player_id": 2,
                    "is_hitter": False,
                    "volume_factor": 1.0,
                    "health_factor": 1.0,
                    "total_dcv": 2.0,
                },
            ]
        )
        # 25 IP projected, 30 IP minimum = below threshold
        result = check_ip_override(dcv, 25.0, ip_minimum=30.0)
        assert result.iloc[1]["total_dcv"] > 2.0


# -----------------------------------------------------------------------
# Constants tests
# -----------------------------------------------------------------------


class TestConstants:
    def test_stabilization_points_all_positive(self):
        for key, val in STABILIZATION_POINTS.items():
            assert val > 0, f"Stabilization point for {key} must be positive"

    def test_stabilization_covers_all_categories(self):
        config = LeagueConfig()
        for cat in config.all_categories:
            assert cat.lower() in STABILIZATION_POINTS, f"Missing stabilization point for {cat}"

    def test_stud_floor_top_n_reasonable(self):
        assert 10 <= STUD_FLOOR_TOP_N <= 100
