"""Tests for contextual draft factors: closer hierarchy, platoon risk,
lineup protection, schedule strength, and contract year boost."""

import pandas as pd
import pytest

from src.contextual_factors import (
    _TEAM_TO_DIVISION,
    LINEUP_SLOT_PA,
    MLB_DIVISIONS,
    _normalize_team,
    compute_lineup_protection,
    compute_platoon_risk,
    compute_schedule_strength,
    contract_year_boost,
    detect_closer_role,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_PARK_FACTORS: dict[str, float] = {
    "ARI": 1.06,
    "ATL": 1.01,
    "BAL": 1.03,
    "BOS": 1.04,
    "CHC": 1.02,
    "CWS": 1.01,
    "CIN": 1.08,
    "CLE": 0.97,
    "COL": 1.38,
    "DET": 0.96,
    "HOU": 1.00,
    "KC": 0.98,
    "LAA": 0.97,
    "LAD": 0.98,
    "MIA": 0.88,
    "MIL": 1.02,
    "MIN": 1.03,
    "NYM": 0.95,
    "NYY": 1.05,
    "OAK": 0.96,
    "PHI": 1.03,
    "PIT": 0.94,
    "SD": 0.93,
    "SF": 0.93,
    "SEA": 0.95,
    "STL": 0.98,
    "TB": 0.96,
    "TEX": 1.05,
    "TOR": 1.03,
    "WSH": 1.00,
}


def _make_player(**kwargs) -> pd.Series:
    """Build a minimal pd.Series representing a player."""
    defaults = {
        "name": "Test Player",
        "is_hitter": True,
        "positions": "OF",
        "sv": 0,
        "hr": 15,
        "rbi": 60,
        "sb": 10,
        "obp": 0.320,
        "team": "NYY",
    }
    defaults.update(kwargs)
    return pd.Series(defaults)


# ===========================================================================
# detect_closer_role
# ===========================================================================


class TestDetectCloserRole:
    """Tests for closer-role classification."""

    def test_elite_closer_high_sv(self):
        p = _make_player(is_hitter=False, positions="RP", sv=35)
        result = detect_closer_role(p)
        assert result["role"] == "Closer"
        assert result["confidence"] == 0.9
        assert result["draft_bonus"] == pytest.approx(2.0)

    def test_closer_threshold_exact(self):
        p = _make_player(is_hitter=False, positions="RP", sv=20)
        result = detect_closer_role(p)
        assert result["role"] == "Closer"
        assert result["confidence"] == 0.9
        assert result["draft_bonus"] == pytest.approx(1.5)

    def test_closer_bonus_scales_linearly(self):
        p = _make_player(is_hitter=False, positions="RP", sv=25)
        result = detect_closer_role(p)
        assert result["role"] == "Closer"
        expected_bonus = 1.5 + (25 - 20) * 0.05  # 1.75
        assert result["draft_bonus"] == pytest.approx(expected_bonus)

    def test_closer_bonus_capped_at_2(self):
        p = _make_player(is_hitter=False, positions="RP", sv=50)
        result = detect_closer_role(p)
        assert result["draft_bonus"] == pytest.approx(2.0)

    def test_setup_role(self):
        p = _make_player(is_hitter=False, positions="RP", sv=18)
        result = detect_closer_role(p)
        assert result["role"] == "Setup"
        assert result["confidence"] == 0.6
        assert result["draft_bonus"] == pytest.approx(0.5)

    def test_setup_threshold_exact(self):
        p = _make_player(is_hitter=False, positions="RP", sv=15)
        result = detect_closer_role(p)
        assert result["role"] == "Setup"

    def test_committee_role(self):
        p = _make_player(is_hitter=False, positions="RP", sv=10)
        result = detect_closer_role(p)
        assert result["role"] == "Committee"
        assert result["confidence"] == 0.4
        assert result["draft_bonus"] == pytest.approx(0.3)

    def test_committee_threshold_exact(self):
        p = _make_player(is_hitter=False, positions="RP", sv=5)
        result = detect_closer_role(p)
        assert result["role"] == "Committee"

    def test_middle_reliever_low_sv(self):
        p = _make_player(is_hitter=False, positions="RP", sv=2)
        result = detect_closer_role(p)
        assert result["role"] == "Middle"
        assert result["confidence"] == 0.0
        assert result["draft_bonus"] == 0.0

    def test_hitter_returns_middle(self):
        p = _make_player(is_hitter=True, positions="OF", sv=30)
        result = detect_closer_role(p)
        assert result["role"] == "Middle"
        assert result["draft_bonus"] == 0.0

    def test_non_rp_pitcher_still_classified(self):
        """SP-only pitcher with saves (rare) classified by is_hitter=False."""
        p = _make_player(is_hitter=False, positions="SP", sv=0)
        result = detect_closer_role(p)
        assert result["role"] == "Middle"

    def test_zero_sv(self):
        p = _make_player(is_hitter=False, positions="RP", sv=0)
        result = detect_closer_role(p)
        assert result["role"] == "Middle"

    def test_none_sv_treated_as_zero(self):
        p = _make_player(is_hitter=False, positions="RP", sv=None)
        result = detect_closer_role(p)
        assert result["role"] == "Middle"


# ===========================================================================
# compute_platoon_risk
# ===========================================================================


class TestPlatoonRisk:
    """Tests for platoon-risk PA discount."""

    def test_left_handed_batter(self):
        assert compute_platoon_risk("L") == pytest.approx(0.90)

    def test_right_handed_batter(self):
        assert compute_platoon_risk("R") == pytest.approx(0.96)

    def test_switch_hitter_s(self):
        assert compute_platoon_risk("S") == pytest.approx(1.0)

    def test_switch_hitter_b(self):
        assert compute_platoon_risk("B") == pytest.approx(1.0)

    def test_lowercase_input(self):
        assert compute_platoon_risk("l") == pytest.approx(0.90)

    def test_padded_input(self):
        assert compute_platoon_risk("  R  ") == pytest.approx(0.96)

    def test_empty_string(self):
        assert compute_platoon_risk("") == pytest.approx(1.0)

    def test_none_input(self):
        assert compute_platoon_risk(None) == pytest.approx(1.0)

    def test_unknown_letter(self):
        assert compute_platoon_risk("X") == pytest.approx(1.0)


# ===========================================================================
# compute_lineup_protection
# ===========================================================================


class TestLineupProtection:
    """Tests for lineup-protection SGP bonus."""

    def test_leadoff_archetype(self):
        p = _make_player(sb=30, obp=0.360)
        bonus = compute_lineup_protection(p)
        # slot 1 = 4.65 PA/G, baseline 4.30 → delta 0.35 * 162 * 0.0018
        expected = (4.65 - 4.30) * 162 * 0.0018
        assert bonus == pytest.approx(expected)
        assert bonus > 0.0

    def test_cleanup_archetype_hr(self):
        p = _make_player(hr=35, rbi=70, sb=5, obp=0.310)
        bonus = compute_lineup_protection(p)
        expected = (4.35 - 4.30) * 162 * 0.0018
        assert bonus == pytest.approx(expected)
        assert bonus > 0.0

    def test_cleanup_archetype_rbi(self):
        p = _make_player(hr=20, rbi=95, sb=5, obp=0.310)
        bonus = compute_lineup_protection(p)
        expected = (4.35 - 4.30) * 162 * 0.0018
        assert bonus == pytest.approx(expected)

    def test_regular_hitter_zero_bonus(self):
        p = _make_player(hr=15, rbi=60, sb=10, obp=0.320)
        bonus = compute_lineup_protection(p)
        assert bonus == pytest.approx(0.0)

    def test_pitcher_zero_bonus(self):
        p = _make_player(is_hitter=False, hr=0, rbi=0, sb=0, obp=0.0)
        bonus = compute_lineup_protection(p)
        assert bonus == pytest.approx(0.0)

    def test_leadoff_needs_both_sb_and_obp(self):
        """High SB but low OBP should not be leadoff."""
        p = _make_player(sb=30, obp=0.300)
        bonus = compute_lineup_protection(p)
        # Not leadoff and not cleanup → regular
        assert bonus == pytest.approx(0.0)

    def test_leadoff_priority_over_cleanup(self):
        """Player with both leadoff AND cleanup stats uses leadoff (higher PA)."""
        p = _make_player(sb=30, obp=0.360, hr=35, rbi=100)
        bonus = compute_lineup_protection(p)
        expected = (4.65 - 4.30) * 162 * 0.0018
        assert bonus == pytest.approx(expected)

    def test_none_stats_treated_as_zero(self):
        p = _make_player(hr=None, rbi=None, sb=None, obp=None)
        bonus = compute_lineup_protection(p)
        assert bonus == pytest.approx(0.0)


# ===========================================================================
# compute_schedule_strength
# ===========================================================================


class TestScheduleStrength:
    """Tests for schedule-strength divisional adjustment."""

    def test_nl_west_team_with_coors(self):
        """COL's divisional opponents play in pitcher-friendly parks overall."""
        factor = compute_schedule_strength("LAD", SAMPLE_PARK_FACTORS)
        # NL West mates for LAD: SD(0.93), ARI(1.06), SF(0.93), COL(1.38)
        avg_pf = (0.93 + 1.06 + 0.93 + 1.38) / 4  # 1.075
        expected = 1.0 + (avg_pf - 1.0) * 0.46
        assert factor == pytest.approx(expected, abs=1e-6)

    def test_col_divisional_opponents(self):
        """COL itself — mates are LAD, SD, ARI, SF (pitcher-friendly)."""
        factor = compute_schedule_strength("COL", SAMPLE_PARK_FACTORS)
        avg_pf = (0.98 + 0.93 + 1.06 + 0.93) / 4  # 0.975
        expected = 1.0 + (avg_pf - 1.0) * 0.46
        assert factor == pytest.approx(expected, abs=1e-6)
        assert factor < 1.0  # pitcher-friendly opponents = slight disadvantage

    def test_al_east_team(self):
        factor = compute_schedule_strength("NYY", SAMPLE_PARK_FACTORS)
        # Mates: BOS(1.04), TOR(1.03), BAL(1.03), TB(0.96)
        avg_pf = (1.04 + 1.03 + 1.03 + 0.96) / 4
        expected = 1.0 + (avg_pf - 1.0) * 0.46
        assert factor == pytest.approx(expected, abs=1e-6)

    def test_unknown_team_returns_neutral(self):
        assert compute_schedule_strength("ZZZ", SAMPLE_PARK_FACTORS) == 1.0

    def test_empty_team_returns_neutral(self):
        assert compute_schedule_strength("", SAMPLE_PARK_FACTORS) == 1.0

    def test_none_team_returns_neutral(self):
        assert compute_schedule_strength(None, SAMPLE_PARK_FACTORS) == 1.0

    def test_empty_park_factors_returns_neutral(self):
        assert compute_schedule_strength("NYY", {}) == 1.0

    def test_aliased_team_resolves(self):
        """WSN should resolve to WSH (NL East)."""
        factor = compute_schedule_strength("WSN", SAMPLE_PARK_FACTORS)
        assert factor != 1.0  # Should resolve, not return neutral

    def test_all_30_teams_have_divisions(self):
        """Every team in MLB_DIVISIONS appears in the reverse lookup."""
        for div, teams in MLB_DIVISIONS.items():
            for team in teams:
                assert team in _TEAM_TO_DIVISION, f"{team} missing from _TEAM_TO_DIVISION"

    def test_factor_in_reasonable_range(self):
        """All schedule factors should be between 0.90 and 1.10."""
        for div, teams in MLB_DIVISIONS.items():
            for team in teams:
                factor = compute_schedule_strength(team, SAMPLE_PARK_FACTORS)
                assert 0.90 <= factor <= 1.10, f"{team} factor {factor} out of range"


# ===========================================================================
# contract_year_boost
# ===========================================================================


class TestContractYearBoost:
    """Tests for contract-year premium."""

    def test_contract_year_hitter(self):
        assert contract_year_boost(True, True) == pytest.approx(1.02)

    def test_contract_year_pitcher(self):
        assert contract_year_boost(True, False) == pytest.approx(1.0)

    def test_no_contract_year_hitter(self):
        assert contract_year_boost(False, True) == pytest.approx(1.0)

    def test_no_contract_year_pitcher(self):
        assert contract_year_boost(False, False) == pytest.approx(1.0)


# ===========================================================================
# _normalize_team helper
# ===========================================================================


class TestNormalizeTeam:
    """Tests for team abbreviation normalization."""

    def test_standard_abbrev(self):
        assert _normalize_team("NYY") == "NYY"

    def test_alias_wsn(self):
        assert _normalize_team("WSN") == "WSH"

    def test_alias_az(self):
        assert _normalize_team("AZ") == "ARI"

    def test_alias_chw(self):
        assert _normalize_team("CHW") == "CWS"

    def test_lowercase(self):
        assert _normalize_team("nyy") == "NYY"

    def test_none_input(self):
        assert _normalize_team(None) == ""

    def test_empty_string(self):
        assert _normalize_team("") == ""


# ===========================================================================
# LINEUP_SLOT_PA constant sanity
# ===========================================================================


class TestLineupSlotPA:
    """Sanity checks for PA/game constants."""

    def test_nine_slots(self):
        assert len(LINEUP_SLOT_PA) == 9

    def test_decreasing_order(self):
        values = [LINEUP_SLOT_PA[i] for i in range(1, 10)]
        for i in range(len(values) - 1):
            assert values[i] >= values[i + 1]

    def test_leadoff_highest(self):
        assert LINEUP_SLOT_PA[1] == 4.65
