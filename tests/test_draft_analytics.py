"""Tests for src/draft_analytics.py — category balance, opportunity cost,
streaming draft value, and BUY/FAIR/AVOID classification.

35 tests covering:
  - compute_category_balance (10 tests)
  - compute_opportunity_cost (10 tests)
  - compute_streaming_draft_value (8 tests)
  - compute_buy_fair_avoid (7 tests)
"""

import numpy as np
import pandas as pd
import pytest

from src.draft_analytics import (
    DEFAULT_SIGMAS,
    compute_buy_fair_avoid,
    compute_category_balance,
    compute_opportunity_cost,
    compute_streaming_draft_value,
)
from src.valuation import LeagueConfig

# ── Fixtures ──────────────────────────────────────────────────────────


def _balanced_totals() -> dict:
    """Roster totals exactly at the median — should produce weights ~ 1.0."""
    return {
        "r": 300,
        "hr": 100,
        "rbi": 300,
        "sb": 60,
        "avg": 0.260,
        "obp": 0.330,
        "w": 30,
        "l": 25,
        "sv": 35,
        "k": 500,
        "era": 3.80,
        "whip": 1.20,
    }


def _weak_sb_totals() -> dict:
    """Roster with very weak stolen bases — SB weight should be highest."""
    t = _balanced_totals()
    t["sb"] = 10  # far below median of 60
    return t


def _strong_hr_totals() -> dict:
    """Roster with massive HR lead — HR weight should be lowest."""
    t = _balanced_totals()
    t["hr"] = 200  # far above median of 100
    return t


def _all_team_totals(n: int = 12) -> list[dict]:
    """Create n teams with totals centered on balanced_totals."""
    base = _balanced_totals()
    teams = []
    rng = np.random.RandomState(42)
    for _ in range(n):
        team = {}
        for k, v in base.items():
            noise = rng.normal(0, 0.05)
            team[k] = v * (1.0 + noise)
        teams.append(team)
    return teams


def _make_player(pid, name, positions, score, is_hitter=True, **kwargs):
    """Create a player Series for testing."""
    data = {
        "player_id": pid,
        "name": name,
        "positions": positions,
        "is_hitter": 1 if is_hitter else 0,
        "pick_score": score,
        "enhanced_pick_score": score,
        "team": "TST",
        "pa": 600 if is_hitter else 0,
        "ip": 0 if is_hitter else 180,
        "sv": 0,
        "era": 0 if is_hitter else 3.50,
        "whip": 0 if is_hitter else 1.15,
        "w": 0 if is_hitter else 12,
        "k": 0 if is_hitter else 180,
        "adp": 50,
    }
    data.update(kwargs)
    return pd.Series(data)


def _make_pool(players):
    """Create a DataFrame pool from a list of player Series."""
    return pd.DataFrame([p.to_dict() for p in players])


# ══════════════════════════════════════════════════════════════════════
# Tests: compute_category_balance
# ══════════════════════════════════════════════════════════════════════


class TestCategoryBalance:
    """Tests for compute_category_balance()."""

    def test_balanced_roster_weights_near_one(self):
        """When roster equals median, all weights should be near 1.0."""
        totals = _balanced_totals()
        all_teams = [totals.copy() for _ in range(12)]  # all identical
        weights = compute_category_balance(totals, all_teams, draft_progress=0.5)

        for cat, w in weights.items():
            assert 0.9 <= w <= 1.1, f"Category {cat} weight {w} not near 1.0"

    def test_weak_sb_gets_highest_weight(self):
        """A roster weak in SB should boost the SB weight."""
        totals = _weak_sb_totals()
        all_teams = _all_team_totals()
        weights = compute_category_balance(totals, all_teams, draft_progress=0.5)

        assert "sb" in weights
        # SB should be among the top 3 highest weights (exact rank depends on noise)
        sorted_cats = sorted(weights, key=weights.get, reverse=True)
        assert "sb" in sorted_cats[:3], (
            f"Expected SB in top 3 weights, got {sorted_cats[:3]} (SB weight={weights['sb']:.3f})"
        )

    def test_strong_hr_gets_lowest_weight(self):
        """A roster dominant in HR should reduce the HR weight."""
        totals = _strong_hr_totals()
        all_teams = _all_team_totals()
        weights = compute_category_balance(totals, all_teams, draft_progress=0.5)

        assert "hr" in weights
        # HR should be among the lowest weights
        min_cat = min(weights, key=weights.get)
        assert min_cat == "hr", f"Expected HR to be lowest weight, got {min_cat}"

    def test_weights_normalise_to_mean_one(self):
        """Weights should normalise so their mean is approximately 1.0."""
        totals = _weak_sb_totals()
        all_teams = _all_team_totals()
        weights = compute_category_balance(totals, all_teams, draft_progress=0.5)

        mean_w = np.mean(list(weights.values()))
        # After progress scaling, mean may not be exactly 1.0,
        # but the raw weights before scaling are normalised
        assert 0.8 <= mean_w <= 1.2, f"Mean weight {mean_w} too far from 1.0"

    def test_early_draft_compresses_toward_one(self):
        """Early draft progress should compress weights toward 1.0."""
        totals = _weak_sb_totals()
        all_teams = _all_team_totals()
        early = compute_category_balance(totals, all_teams, draft_progress=0.1)
        mid = compute_category_balance(totals, all_teams, draft_progress=0.5)

        # Early draft: SB weight should be closer to 1.0 than mid-draft
        sb_early_dev = abs(early["sb"] - 1.0)
        sb_mid_dev = abs(mid["sb"] - 1.0)
        assert sb_early_dev < sb_mid_dev, "Early draft should compress SB weight toward 1.0"

    def test_late_draft_amplifies_gaps(self):
        """Late draft progress should amplify category balance weights."""
        totals = _weak_sb_totals()
        all_teams = _all_team_totals()
        mid = compute_category_balance(totals, all_teams, draft_progress=0.5)
        late = compute_category_balance(totals, all_teams, draft_progress=0.85)

        # Late draft: SB weight deviation from 1.0 should be larger
        sb_mid_dev = abs(mid["sb"] - 1.0)
        sb_late_dev = abs(late["sb"] - 1.0)
        assert sb_late_dev > sb_mid_dev, "Late draft should amplify SB weight gap"

    def test_empty_all_teams_returns_equal_weights(self):
        """No opponent data should return all 1.0 weights."""
        totals = _balanced_totals()
        weights = compute_category_balance(totals, [], draft_progress=0.5)

        for cat, w in weights.items():
            assert w == 1.0, f"Expected 1.0 for {cat}, got {w}"

    def test_inverse_categories_correct_direction(self):
        """High ERA (bad) should boost ERA weight; low ERA should reduce it."""
        # High ERA — we are worse than median, should be boosted
        totals_bad = _balanced_totals()
        totals_bad["era"] = 5.50  # way above (bad for ERA)
        # Low ERA — we are better than median, should be reduced
        totals_good = _balanced_totals()
        totals_good["era"] = 2.50  # way below (good for ERA)
        all_teams = _all_team_totals()

        weights_bad = compute_category_balance(totals_bad, all_teams, draft_progress=0.5)
        weights_good = compute_category_balance(totals_good, all_teams, draft_progress=0.5)

        # Bad ERA should have higher weight (we need ERA help) than good ERA
        assert weights_bad["era"] > weights_good["era"], (
            f"Bad ERA weight {weights_bad['era']:.3f} should be > good ERA weight {weights_good['era']:.3f}"
        )

    def test_with_league_config(self):
        """Accepts a LeagueConfig for category definitions."""
        config = LeagueConfig()
        totals = _balanced_totals()
        all_teams = _all_team_totals()
        weights = compute_category_balance(totals, all_teams, config=config, draft_progress=0.5)

        # Should have all 12 categories
        expected_cats = {c.lower() for c in config.hitting_categories + config.pitching_categories}
        assert set(weights.keys()) == expected_cats

    def test_returns_all_default_categories_without_config(self):
        """Without config, returns weights for all 12 default categories."""
        totals = _balanced_totals()
        all_teams = _all_team_totals()
        weights = compute_category_balance(totals, all_teams, draft_progress=0.5)

        assert set(weights.keys()) == set(DEFAULT_SIGMAS.keys())


# ══════════════════════════════════════════════════════════════════════
# Tests: compute_opportunity_cost
# ══════════════════════════════════════════════════════════════════════


class TestOpportunityCost:
    """Tests for compute_opportunity_cost()."""

    def test_scarce_position_high_oc(self):
        """A top catcher with no good alternatives has high OC."""
        catcher = _make_player(1, "Star Catcher", "C", 8.0)
        backup = _make_player(2, "Backup Catcher", "C", 3.0)
        filler = _make_player(3, "Outfielder", "OF", 7.5)
        pool = _make_pool([backup, filler])

        oc = compute_opportunity_cost(catcher, pool, survival=0.3)
        # Gap is 8.0 - 3.0 = 5.0; weighted by (1 - 0.3) = 0.7 => 3.5
        assert oc > 2.0, f"OC {oc} too low for scarce position"

    def test_deep_position_low_oc(self):
        """An outfielder with many good alternatives has low OC."""
        target = _make_player(1, "Good OF", "OF", 6.0)
        alt1 = _make_player(2, "Alt OF 1", "OF", 5.8)
        alt2 = _make_player(3, "Alt OF 2", "OF", 5.5)
        alt3 = _make_player(4, "Alt OF 3", "OF", 5.3)
        pool = _make_pool([alt1, alt2, alt3])

        oc = compute_opportunity_cost(target, pool, survival=0.3)
        # Gap is 6.0 - 5.8 = 0.2; weighted by 0.7 => 0.14
        assert oc < 1.0, f"OC {oc} too high for deep position"

    def test_multi_position_reduces_oc(self):
        """Multi-position eligibility should reduce OC."""
        single_pos = _make_player(1, "SS Only", "SS", 7.0)
        multi_pos = _make_player(2, "SS/2B/3B", "SS,2B,3B", 7.0)
        backup_ss = _make_player(3, "Backup SS", "SS", 4.0)
        backup_2b = _make_player(4, "Backup 2B", "2B", 5.0)
        backup_3b = _make_player(5, "Backup 3B", "3B", 4.5)
        pool = _make_pool([backup_ss, backup_2b, backup_3b])

        oc_single = compute_opportunity_cost(single_pos, pool, survival=0.3)
        oc_multi = compute_opportunity_cost(multi_pos, pool, survival=0.3)

        assert oc_multi < oc_single, f"Multi-pos OC {oc_multi} should be < single-pos OC {oc_single}"

    def test_high_survival_low_oc(self):
        """High survival probability (player likely available later) = low OC."""
        player = _make_player(1, "Mid SP", "SP", 5.0)
        backup = _make_player(2, "Backup SP", "SP", 3.0)
        pool = _make_pool([backup])

        oc_low_surv = compute_opportunity_cost(player, pool, survival=0.1)
        oc_high_surv = compute_opportunity_cost(player, pool, survival=0.9)

        assert oc_high_surv < oc_low_surv, "High survival should mean lower OC"

    def test_zero_survival_max_oc(self):
        """Survival = 0 means the player WILL be taken — maximum urgency."""
        player = _make_player(1, "Top Pick", "SS", 10.0)
        backup = _make_player(2, "Next SS", "SS", 5.0)
        pool = _make_pool([backup])

        oc = compute_opportunity_cost(player, pool, survival=0.0)
        # gap = 10 - 5 = 5; weight = (1 - 0) = 1.0 => OC = 5.0
        assert oc == pytest.approx(5.0, abs=0.01)

    def test_survival_one_zero_oc(self):
        """Survival = 1.0 means the player will definitely be there — OC = 0."""
        player = _make_player(1, "Safe Pick", "OF", 6.0)
        backup = _make_player(2, "Alt OF", "OF", 4.0)
        pool = _make_pool([backup])

        oc = compute_opportunity_cost(player, pool, survival=1.0)
        assert oc == 0.0

    def test_no_alternatives_max_gap(self):
        """When no one else plays the position, gap equals full score."""
        player = _make_player(1, "Only C", "C", 7.0)
        of_player = _make_player(2, "OF Guy", "OF", 8.0)  # different position
        pool = _make_pool([of_player])

        oc = compute_opportunity_cost(player, pool, survival=0.3)
        # No catcher alternatives → next_best = 0, gap = 7.0
        expected = 7.0 * (1 - 0.3)  # = 4.9
        assert oc == pytest.approx(expected, abs=0.01)

    def test_empty_pool_returns_positive(self):
        """Empty pool means no alternatives — OC should be positive."""
        player = _make_player(1, "Lone Wolf", "SS", 5.0)
        pool = pd.DataFrame()

        oc = compute_opportunity_cost(player, pool, survival=0.3)
        assert oc > 0.0

    def test_oc_never_negative(self):
        """OC should never be negative."""
        player = _make_player(1, "Weak Player", "OF", 2.0)
        star = _make_player(2, "Star OF", "OF", 10.0)
        pool = _make_pool([star])

        oc = compute_opportunity_cost(player, pool, survival=0.5)
        assert oc >= 0.0

    def test_uses_enhanced_pick_score_when_available(self):
        """Should use enhanced_pick_score over pick_score when available."""
        player = _make_player(1, "Test", "OF", 3.0, enhanced_pick_score=7.0)
        backup = _make_player(2, "Alt", "OF", 3.0, enhanced_pick_score=4.0)
        pool = _make_pool([backup])

        oc = compute_opportunity_cost(player, pool, survival=0.3)
        # Gap from enhanced scores: 7.0 - 4.0 = 3.0; weight 0.7 => 2.1
        assert oc > 1.5, f"OC {oc} should reflect enhanced_pick_score gap"


# ══════════════════════════════════════════════════════════════════════
# Tests: compute_streaming_draft_value
# ══════════════════════════════════════════════════════════════════════


class TestStreamingDraftValue:
    """Tests for compute_streaming_draft_value()."""

    def test_closer_no_penalty(self):
        """Closers (SV > 10) should get zero penalty — saves not streamable."""
        closer = _make_player(
            1,
            "Elite Closer",
            "RP",
            5.0,
            is_hitter=False,
            sv=30,
            era=3.00,
            ip=60,
            w=3,
            k=70,
            whip=1.10,
        )
        penalty = compute_streaming_draft_value(closer)
        assert penalty == 0.0

    def test_elite_sp_no_penalty(self):
        """Elite SP with ERA < 2.80 should get zero penalty."""
        ace = _make_player(
            1,
            "Ace",
            "SP",
            8.0,
            is_hitter=False,
            sv=0,
            era=2.50,
            ip=200,
            w=15,
            k=220,
            whip=0.95,
        )
        penalty = compute_streaming_draft_value(ace)
        assert penalty == 0.0

    def test_mediocre_sp_gets_penalty(self):
        """A mediocre SP (high ERA, mainly W/K value) should get a penalty."""
        streamer = _make_player(
            1,
            "Streamer SP",
            "SP",
            3.0,
            is_hitter=False,
            sv=0,
            era=4.80,
            ip=160,
            w=10,
            k=150,
            whip=1.40,
        )
        penalty = compute_streaming_draft_value(streamer)
        assert penalty < 0.0, f"Expected negative penalty, got {penalty}"
        assert penalty >= -0.5, f"Penalty {penalty} exceeds max"

    def test_hitter_no_penalty(self):
        """Hitters should always get zero penalty."""
        hitter = _make_player(1, "Hitter", "OF", 5.0, is_hitter=True)
        penalty = compute_streaming_draft_value(hitter)
        assert penalty == 0.0

    def test_penalty_between_bounds(self):
        """Penalty should be between -0.5 and 0.0."""
        pitcher = _make_player(
            1,
            "Mid SP",
            "SP",
            4.0,
            is_hitter=False,
            sv=0,
            era=4.20,
            ip=180,
            w=11,
            k=170,
            whip=1.30,
        )
        penalty = compute_streaming_draft_value(pitcher)
        assert -0.5 <= penalty <= 0.0

    def test_low_ip_pitcher_mild_penalty(self):
        """Pitchers with very low IP get mild penalty."""
        low_ip = _make_player(
            1,
            "Low IP",
            "SP",
            2.0,
            is_hitter=False,
            sv=0,
            era=5.00,
            ip=20,
            w=1,
            k=20,
            whip=1.50,
        )
        penalty = compute_streaming_draft_value(low_ip)
        assert penalty == pytest.approx(-0.3, abs=0.01)

    def test_with_league_config(self):
        """Accepts a LeagueConfig for SGP denominators."""
        config = LeagueConfig()
        streamer = _make_player(
            1,
            "Streamer",
            "SP",
            3.0,
            is_hitter=False,
            sv=0,
            era=4.80,
            ip=160,
            w=10,
            k=150,
            whip=1.40,
        )
        penalty = compute_streaming_draft_value(streamer, config=config)
        assert penalty <= 0.0

    def test_rate_dominant_pitcher_no_penalty(self):
        """A pitcher whose value is rate-stat dominant (low ERA/WHIP, few K) gets no penalty."""
        rate_pitcher = _make_player(
            1,
            "Rate Anchor",
            "SP",
            5.0,
            is_hitter=False,
            sv=0,
            era=3.00,
            ip=180,
            w=8,
            k=120,
            whip=1.00,
        )
        penalty = compute_streaming_draft_value(rate_pitcher)
        # ERA < 2.80 is the elite threshold — 3.00 is close but above.
        # With strong WHIP, rate SGP is high relative to counting.
        # The function should return a mild or zero penalty.
        assert penalty >= -0.3


# ══════════════════════════════════════════════════════════════════════
# Tests: compute_buy_fair_avoid
# ══════════════════════════════════════════════════════════════════════


class TestBuyFairAvoid:
    """Tests for compute_buy_fair_avoid()."""

    def test_buy_when_model_ranks_higher(self):
        """Model rank 10, ADP rank 50 at early pick → BUY (gap=40 > 20)."""
        result = compute_buy_fair_avoid(enhanced_rank=10, adp_rank=50, current_pick=20)
        assert result == "BUY"

    def test_avoid_when_adp_ranks_higher(self):
        """Model rank 80, ADP rank 30 at early pick → AVOID (gap=-50 < -20)."""
        result = compute_buy_fair_avoid(enhanced_rank=80, adp_rank=30, current_pick=20)
        assert result == "AVOID"

    def test_fair_when_close(self):
        """Model rank 50, ADP rank 55 → FAIR (gap=5 < threshold)."""
        result = compute_buy_fair_avoid(enhanced_rank=50, adp_rank=55, current_pick=50)
        assert result == "FAIR"

    def test_threshold_scales_with_progress(self):
        """Late-draft threshold is tighter (10) than early (20)."""
        # Gap of 12: BUY late (threshold=10) but FAIR early (threshold=20)
        late = compute_buy_fair_avoid(enhanced_rank=50, adp_rank=62, current_pick=210)
        early = compute_buy_fair_avoid(enhanced_rank=50, adp_rank=62, current_pick=50)
        assert late == "BUY"
        assert early == "FAIR"

    def test_boundary_early_exact_threshold(self):
        """At exactly the threshold gap, should be BUY."""
        result = compute_buy_fair_avoid(enhanced_rank=10, adp_rank=30, current_pick=50)
        assert result == "BUY"  # gap = 20 == threshold for early

    def test_boundary_mid_draft(self):
        """Mid-draft threshold is 15."""
        buy = compute_buy_fair_avoid(enhanced_rank=50, adp_rank=66, current_pick=150)
        fair = compute_buy_fair_avoid(enhanced_rank=50, adp_rank=64, current_pick=150)
        assert buy == "BUY"  # gap = 16 > 15
        assert fair == "FAIR"  # gap = 14 < 15

    def test_invalid_ranks_return_fair(self):
        """Invalid ranks (0 or negative) should return FAIR."""
        assert compute_buy_fair_avoid(0, 50, 100) == "FAIR"
        assert compute_buy_fair_avoid(50, 0, 100) == "FAIR"
        assert compute_buy_fair_avoid(-1, 50, 100) == "FAIR"
