"""Tests for Phase 5 pitcher/closer features: K2, I6, L3, L4."""

from __future__ import annotations

import pytest

from src.closer_monitor import compute_committee_risk, detect_opener
from src.optimizer.pivot_advisor import recommend_ip_management_mode
from src.optimizer.projections import compute_fatigue_multiplier

# ── K2: Pitcher Fatigue Multiplier ──────────────────────────────────


class TestFatigueMultiplier:
    """compute_fatigue_multiplier tests."""

    def test_at_100_ip_returns_1(self):
        assert compute_fatigue_multiplier(100.0) == 1.0

    def test_below_100_ip_returns_1(self):
        assert compute_fatigue_multiplier(50.0) == 1.0
        assert compute_fatigue_multiplier(0.0) == 1.0

    def test_at_150_ip_returns_085(self):
        # 50 IP over * 0.003 = 0.15 discount -> 0.85
        result = compute_fatigue_multiplier(150.0)
        assert result == pytest.approx(0.85, abs=0.001)

    def test_at_200_ip_capped_at_085(self):
        # 100 IP over * 0.003 = 0.30, but capped at 0.15 -> 0.85
        result = compute_fatigue_multiplier(200.0)
        assert result == 0.85

    def test_at_120_ip_partial_discount(self):
        # 20 IP over * 0.003 = 0.06 -> 0.94
        result = compute_fatigue_multiplier(120.0)
        assert result == pytest.approx(0.94, abs=0.001)

    def test_elite_pitcher_exempt(self):
        assert compute_fatigue_multiplier(180.0, is_elite=True) == 1.0

    def test_elite_at_100_still_1(self):
        assert compute_fatigue_multiplier(100.0, is_elite=True) == 1.0


# ── I6: IP Management Mode ─────────────────────────────────────────


class TestIPManagementMode:
    """recommend_ip_management_mode tests."""

    def test_winning_era_whip_protect_ratios(self):
        urgency = {"summary": {"winning": ["ERA", "WHIP", "K"], "losing": []}}
        assert recommend_ip_management_mode(urgency) == "PROTECT_RATIOS"

    def test_losing_k_chase_kw(self):
        urgency = {"summary": {"winning": ["SV"], "losing": ["K"]}}
        assert recommend_ip_management_mode(urgency) == "CHASE_KW"

    def test_losing_w_chase_kw(self):
        urgency = {"summary": {"winning": [], "losing": ["W"]}}
        assert recommend_ip_management_mode(urgency) == "CHASE_KW"

    def test_losing_both_k_and_w(self):
        urgency = {"summary": {"winning": [], "losing": ["K", "W"]}}
        assert recommend_ip_management_mode(urgency) == "CHASE_KW"

    def test_mixed_balanced(self):
        urgency = {"summary": {"winning": ["HR"], "losing": ["RBI"]}}
        assert recommend_ip_management_mode(urgency) == "BALANCED"

    def test_empty_summary_balanced(self):
        urgency = {"summary": {}}
        assert recommend_ip_management_mode(urgency) == "BALANCED"

    def test_no_summary_key_balanced(self):
        assert recommend_ip_management_mode({}) == "BALANCED"

    def test_protect_ratios_even_with_losing_k(self):
        # ERA+WHIP winning takes priority over K losing
        urgency = {"summary": {"winning": ["ERA", "WHIP"], "losing": ["K"]}}
        # PROTECT_RATIOS because era_safe and whip_safe checked first
        assert recommend_ip_management_mode(urgency) == "PROTECT_RATIOS"


# ── L3: Committee Risk Score ────────────────────────────────────────


class TestCommitteeRisk:
    """compute_committee_risk tests."""

    def test_primary_closer_80_pct_defined(self):
        dist = {"Closer A": 16, "Setup B": 2, "Setup C": 2}
        result = compute_committee_risk(dist)
        assert result["signal"] == "DEFINED"
        assert result["is_committee"] is False
        assert result["primary_share"] == pytest.approx(0.80, abs=0.01)

    def test_three_closers_equal_committee(self):
        dist = {"A": 5, "B": 5, "C": 5}
        result = compute_committee_risk(dist)
        assert result["signal"] == "COMMITTEE"
        assert result["is_committee"] is True
        assert result["primary_share"] == pytest.approx(0.33, abs=0.01)
        assert result["num_contributors"] == 3

    def test_no_saves_signal(self):
        dist = {"A": 0, "B": 0}
        result = compute_committee_risk(dist)
        assert result["signal"] == "NO_SAVES"
        assert result["is_committee"] is False

    def test_empty_dict_unknown(self):
        result = compute_committee_risk({})
        assert result["signal"] == "UNKNOWN"

    def test_shaky_signal(self):
        # Primary has 65% share (>60% so not committee, <75% so not defined)
        dist = {"A": 13, "B": 4, "C": 3}
        result = compute_committee_risk(dist)
        assert result["signal"] == "SHAKY"
        assert result["is_committee"] is False

    def test_risk_score_inversely_proportional(self):
        dist_clear = {"A": 20, "B": 2}
        dist_messy = {"A": 5, "B": 5, "C": 5}
        r_clear = compute_committee_risk(dist_clear)
        r_messy = compute_committee_risk(dist_messy)
        assert r_clear["risk_score"] < r_messy["risk_score"]


# ── L4: Opener Detection Flag ──────────────────────────────────────


class TestOpenerDetection:
    """detect_opener tests."""

    def test_opener_true(self):
        assert detect_opener(0.40, 20) is True

    def test_low_first_inning_pct_not_opener(self):
        assert detect_opener(0.10, 30) is False

    def test_boundary_pct_not_opener(self):
        # Exactly 30% should NOT trigger (>30 required)
        assert detect_opener(0.30, 15) is False

    def test_too_few_appearances(self):
        assert detect_opener(0.50, 5) is False

    def test_boundary_appearances(self):
        # Exactly 10 appearances + >30% -> opener
        assert detect_opener(0.35, 10) is True
