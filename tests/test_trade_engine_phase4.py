"""Tests for Trade Analyzer Engine Phase 4: Context Engine.

Spec reference: Section 17 Phase 4 items 17-20

Tests cover:
  - Log5 matchup engine (odds-ratio, park adjustment, game projection)
  - Injury stochastic process (Weibull duration, frailty, availability)
  - Enhanced bench option value (flexibility, injury cushion)
  - Roster concentration risk (HHI, penalty, delta, exposure)
  - Integration with trade evaluator
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ── Matchup Engine ──────────────────────────────────────────────────


class TestMatchupEngine:
    """Spec ref: Section 9 L6 — Log5 Matchup Engine."""

    def test_log5_equal_rates_returns_league_avg(self):
        """When batter == pitcher == league_avg, result should be league_avg."""
        from src.engine.context.matchup import log5_matchup

        result = log5_matchup(0.300, 0.300, 0.300)
        assert abs(result - 0.300) < 0.001

    def test_log5_good_batter_high_result(self):
        """Good batter vs average pitcher should exceed league average."""
        from src.engine.context.matchup import log5_matchup

        result = log5_matchup(0.350, 0.300, 0.300)
        assert result > 0.300

    def test_log5_bad_pitcher_helps_batter(self):
        """Batter should project higher against a bad pitcher."""
        from src.engine.context.matchup import log5_matchup

        # Bad pitcher = high rate against
        result_bad = log5_matchup(0.300, 0.360, 0.300)
        result_avg = log5_matchup(0.300, 0.300, 0.300)
        assert result_bad > result_avg

    def test_log5_edge_zero_rate_clamped(self):
        """Rate at 0.0 should be clamped, no division by zero."""
        from src.engine.context.matchup import log5_matchup

        result = log5_matchup(0.0, 0.300, 0.300)
        assert 0 < result < 1  # Should return a valid probability

    def test_log5_edge_one_rate_clamped(self):
        """Rate at 1.0 should be clamped, no division by zero."""
        from src.engine.context.matchup import log5_matchup

        result = log5_matchup(1.0, 0.300, 0.300)
        assert 0 < result < 1

    def test_lineup_slot_pa_decreasing(self):
        """PA should decrease by lineup position."""
        from src.engine.context.matchup import LINEUP_SLOT_PA

        for slot in range(1, 9):
            assert LINEUP_SLOT_PA[slot] > LINEUP_SLOT_PA[slot + 1]

    def test_park_adjust_amplifies(self):
        """Hitter-friendly park (Coors) should amplify stats."""
        from src.engine.context.matchup import park_adjust

        base_hr = 25.0
        coors_hr = park_adjust(base_hr, 1.38)
        assert coors_hr > base_hr

    def test_park_adjust_neutral(self):
        """Neutral park factor should leave stats unchanged."""
        from src.engine.context.matchup import park_adjust

        assert park_adjust(25.0, 1.0) == 25.0

    def test_matchup_adjustment_no_data_neutral(self):
        """Without schedule data, matchup factor should be 1.0."""
        from src.engine.context.matchup import matchup_adjustment_factor

        result = matchup_adjustment_factor()
        assert result == 1.0

    def test_game_projection_returns_all_stats(self):
        """Game projection should return all expected stat keys."""
        from src.engine.context.matchup import game_projection

        result = game_projection(batter_xwoba=0.350)
        expected_keys = {"hr", "r", "rbi", "sb", "h", "ab", "pa"}
        assert expected_keys.issubset(result.keys())
        assert all(v >= 0 for v in result.values())


# ── Injury Process ──────────────────────────────────────────────────


class TestInjuryProcess:
    """Spec ref: Section 8 L5 — Injury Stochastic Process."""

    def test_injury_duration_hamstring_minimum(self):
        """Hamstring injury should last at least 10 days (loc=10)."""
        from src.engine.context.injury_process import sample_injury_duration

        rng = np.random.RandomState(42)
        durations = [sample_injury_duration("hamstring", rng=rng) for _ in range(100)]
        assert all(d >= 10 for d in durations)

    def test_injury_duration_ucl_long(self):
        """UCL injuries should be very long (loc=300)."""
        from src.engine.context.injury_process import sample_injury_duration

        rng = np.random.RandomState(42)
        durations = [sample_injury_duration("ucl", rng=rng) for _ in range(50)]
        # Median UCL duration should be well over 300 days
        assert np.median(durations) > 300

    def test_injury_duration_unknown_uses_other(self):
        """Unknown injury type should fall back to 'other' parameters."""
        from src.engine.context.injury_process import sample_injury_duration

        rng = np.random.RandomState(42)
        duration = sample_injury_duration("sprained_ankle", rng=rng)
        assert duration >= 10

    def test_frailty_increases_duration(self):
        """Higher frailty should produce longer average durations."""
        from src.engine.context.injury_process import sample_injury_duration

        rng_normal = np.random.RandomState(42)
        rng_fragile = np.random.RandomState(42)

        normal = [sample_injury_duration("hamstring", frailty=1.0, rng=rng_normal) for _ in range(200)]
        fragile = [sample_injury_duration("hamstring", frailty=2.0, rng=rng_fragile) for _ in range(200)]

        assert np.mean(fragile) > np.mean(normal)

    def test_frailty_from_health_score(self):
        """Healthy player = low frailty, fragile player = high frailty."""
        from src.engine.context.injury_process import frailty_from_health_score

        assert frailty_from_health_score(1.0) == 1.0
        assert frailty_from_health_score(0.5) == 2.0
        # Below 0.5 is clamped to frailty=2.0
        assert frailty_from_health_score(0.3) == 2.0

    def test_availability_healthy_player_near_one(self):
        """Healthy player should almost always be fully available."""
        from src.engine.context.injury_process import sample_season_availability

        rng = np.random.RandomState(42)
        avails = [sample_season_availability(health_score=0.95, weeks_remaining=16, rng=rng) for _ in range(200)]
        # Most should be 1.0, average should be high
        assert np.mean(avails) > 0.90

    def test_availability_injury_prone_lower(self):
        """Injury-prone player should have lower average availability."""
        from src.engine.context.injury_process import sample_season_availability

        rng = np.random.RandomState(42)
        avails = [
            sample_season_availability(health_score=0.60, age=34, weeks_remaining=16, rng=rng) for _ in range(500)
        ]
        # Should be noticeably below 1.0
        assert np.mean(avails) < 0.95

    def test_availability_never_negative(self):
        """Availability should always be in [0, 1]."""
        from src.engine.context.injury_process import sample_season_availability

        rng = np.random.RandomState(42)
        for _ in range(100):
            avail = sample_season_availability(
                health_score=0.50,
                age=38,
                is_pitcher=True,
                weeks_remaining=16,
                rng=rng,
            )
            assert 0.0 <= avail <= 1.0

    def test_seed_reproducibility(self):
        """Same seed should produce same duration."""
        from src.engine.context.injury_process import sample_injury_duration

        d1 = sample_injury_duration("hamstring", rng=np.random.RandomState(123))
        d2 = sample_injury_duration("hamstring", rng=np.random.RandomState(123))
        assert d1 == d2

    def test_injury_probability_scales_with_horizon(self):
        """Longer horizon should produce higher injury probability."""
        from src.engine.context.injury_process import estimate_injury_probability

        p_short = estimate_injury_probability(0.80, horizon_days=30)
        p_long = estimate_injury_probability(0.80, horizon_days=120)
        assert p_long > p_short


# ── Enhanced Bench Value ────────────────────────────────────────────


class TestBenchValue:
    """Spec ref: Section 10 L7D — Bench Option Value."""

    def test_enhanced_returns_breakdown(self):
        """Enhanced function should return a dict with component breakdown."""
        from src.engine.context.bench_value import enhanced_bench_option_value

        result = enhanced_bench_option_value(weeks_remaining=16)
        assert "streaming" in result
        assert "hot_fa" in result
        assert "flexibility" in result
        assert "injury_cushion" in result
        assert "total" in result
        assert result["total"] > 0

    def test_flexibility_premium_adds_value(self):
        """Higher roster flexibility should increase bench value."""
        from src.engine.context.bench_value import enhanced_bench_option_value

        no_flex = enhanced_bench_option_value(weeks_remaining=16, roster_flexibility=0.0)
        high_flex = enhanced_bench_option_value(weeks_remaining=16, roster_flexibility=0.5)
        assert high_flex["total"] > no_flex["total"]

    def test_injury_replacement_adds_value(self):
        """Injury replacement value should increase total."""
        from src.engine.context.bench_value import enhanced_bench_option_value

        no_injury = enhanced_bench_option_value(weeks_remaining=16, injury_replacement_value=0.0)
        with_injury = enhanced_bench_option_value(weeks_remaining=16, injury_replacement_value=0.5)
        assert with_injury["total"] > no_injury["total"]

    def test_zero_weeks_near_zero(self):
        """With 0 weeks remaining, most bench value components should be 0."""
        from src.engine.context.bench_value import enhanced_bench_option_value

        result = enhanced_bench_option_value(weeks_remaining=0)
        assert result["streaming"] == 0.0
        assert result["total"] >= 0.0

    def test_roster_flexibility_multi_position(self):
        """Multi-position players contribute more flexibility."""
        from src.engine.context.bench_value import compute_roster_flexibility

        single_pos = pd.DataFrame({"positions": ["1B", "C", "SS"]})
        multi_pos = pd.DataFrame({"positions": ["1B,2B,3B,SS", "C,1B", "SS,2B"]})

        flex_single = compute_roster_flexibility(single_pos)
        flex_multi = compute_roster_flexibility(multi_pos)
        assert flex_multi > flex_single

    def test_roster_flexibility_empty(self):
        """Empty roster should return 0.0 flexibility."""
        from src.engine.context.bench_value import compute_roster_flexibility

        assert compute_roster_flexibility(pd.DataFrame()) == 0.0

    def test_simple_bench_still_works(self):
        """Original bench_option_value() should still be importable and work."""
        from src.engine.portfolio.lineup_optimizer import bench_option_value

        val = bench_option_value(weeks_remaining=16)
        assert val > 0


# ── Concentration Risk ──────────────────────────────────────────────


class TestConcentrationRisk:
    """Spec ref: Section 7 L4B — Roster Concentration Risk."""

    def _make_pool(self, players):
        """Helper to create a player pool DataFrame."""
        return pd.DataFrame(players)

    def test_hhi_single_team_is_one(self):
        """All players from one team → HHI = 1.0."""
        from src.engine.context.concentration import roster_concentration_hhi

        pool = self._make_pool(
            [
                {"player_id": 1, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 2, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 3, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
            ]
        )
        hhi = roster_concentration_hhi([1, 2, 3], pool)
        assert abs(hhi - 1.0) < 0.001

    def test_hhi_perfect_diversification(self):
        """One player per team → HHI = 1/N."""
        from src.engine.context.concentration import roster_concentration_hhi

        pool = self._make_pool(
            [{"player_id": i, "team": f"T{i}", "is_hitter": True, "pa": 500, "ip": 0} for i in range(1, 11)]
        )
        hhi = roster_concentration_hhi(list(range(1, 11)), pool)
        # With 10 equally-weighted teams: HHI = 10 * (1/10)^2 = 0.10
        assert abs(hhi - 0.10) < 0.001

    def test_hhi_moderate_concentration(self):
        """3 of 10 players from same team → moderate HHI."""
        from src.engine.context.concentration import roster_concentration_hhi

        players = [{"player_id": i, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0} for i in range(1, 4)] + [
            {"player_id": i, "team": f"T{i}", "is_hitter": True, "pa": 500, "ip": 0} for i in range(4, 11)
        ]
        pool = self._make_pool(players)
        hhi = roster_concentration_hhi(list(range(1, 11)), pool)
        assert 0.10 < hhi < 0.20

    def test_penalty_below_threshold_is_zero(self):
        """Low HHI → no penalty."""
        from src.engine.context.concentration import concentration_risk_penalty

        assert concentration_risk_penalty(0.10) == 0.0

    def test_penalty_above_threshold_is_positive(self):
        """High HHI → positive penalty."""
        from src.engine.context.concentration import concentration_risk_penalty

        penalty = concentration_risk_penalty(0.25)
        assert penalty > 0.0

    def test_concentration_delta_trade_concentrates(self):
        """Trading for a same-team player should increase HHI."""
        from src.engine.context.concentration import compute_concentration_delta

        pool = self._make_pool(
            [
                {"player_id": 1, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 2, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 3, "team": "BOS", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 4, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
            ]
        )

        # Before: [1, 2, 3] (2 NYY, 1 BOS). After: [1, 2, 4] (3 NYY)
        result = compute_concentration_delta([1, 2, 3], [1, 2, 4], pool)
        assert result["hhi_delta"] > 0  # More concentrated

    def test_concentration_delta_trade_diversifies(self):
        """Trading concentrated player for diverse one should decrease HHI."""
        from src.engine.context.concentration import compute_concentration_delta

        pool = self._make_pool(
            [
                {"player_id": 1, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 2, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 3, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 4, "team": "LAD", "is_hitter": True, "pa": 500, "ip": 0},
            ]
        )

        # Before: [1, 2, 3] (3 NYY). After: [1, 2, 4] (2 NYY, 1 LAD)
        result = compute_concentration_delta([1, 2, 3], [1, 2, 4], pool)
        assert result["hhi_delta"] < 0  # Less concentrated

    def test_team_exposure_correct_counts(self):
        """Should return correct player counts per team."""
        from src.engine.context.concentration import team_exposure_breakdown

        pool = self._make_pool(
            [
                {"player_id": 1, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 2, "team": "NYY", "is_hitter": True, "pa": 500, "ip": 0},
                {"player_id": 3, "team": "BOS", "is_hitter": True, "pa": 500, "ip": 0},
            ]
        )

        breakdown = team_exposure_breakdown([1, 2, 3], pool)
        assert breakdown["NYY"]["count"] == 2
        assert breakdown["BOS"]["count"] == 1

    def test_empty_roster_returns_zero(self):
        """Empty roster HHI should be 0."""
        from src.engine.context.concentration import roster_concentration_hhi

        assert roster_concentration_hhi([], pd.DataFrame()) == 0.0

    def test_pitchers_use_ip(self):
        """Pitcher concentration weight should use IP, not PA."""
        from src.engine.context.concentration import roster_concentration_hhi

        pool = self._make_pool(
            [
                {"player_id": 1, "team": "NYY", "is_hitter": False, "pa": 0, "ip": 200},
                {"player_id": 2, "team": "BOS", "is_hitter": False, "pa": 0, "ip": 100},
            ]
        )

        hhi = roster_concentration_hhi([1, 2], pool)
        # NYY pitcher has 200 IP (2/3 weight), BOS has 100 IP (1/3 weight)
        # HHI = (2/3)^2 + (1/3)^2 = 4/9 + 1/9 = 5/9 ≈ 0.556
        assert abs(hhi - 5.0 / 9.0) < 0.01


# ── Phase 4 Integration ────────────────────────────────────────────


class TestPhase4Integration:
    """Test Phase 4 integration with trade evaluator."""

    def _make_test_pool(self):
        """Create a minimal player pool for integration tests."""
        return pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "name": [
                    "Aaron Judge",
                    "Shohei Ohtani",
                    "Mookie Betts",
                    "Freddie Freeman",
                    "Ronald Acuna",
                    "Mike Trout",
                    "Trea Turner",
                    "Juan Soto",
                    "Corey Seager",
                    "Julio Rodriguez",
                ],
                "team": ["NYY", "LAD", "LAD", "LAD", "ATL", "LAA", "PHI", "NYY", "TEX", "SEA"],
                "positions": ["OF", "DH,SP", "OF,2B", "1B", "OF", "OF", "SS", "OF", "SS", "OF"],
                "is_hitter": [True, True, True, True, True, True, True, True, True, True],
                "is_injured": [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                "pa": [600, 550, 580, 570, 590, 400, 560, 580, 540, 550],
                "ab": [520, 480, 510, 500, 510, 350, 490, 500, 470, 480],
                "h": [145, 160, 155, 165, 150, 100, 140, 155, 135, 140],
                "r": [100, 95, 105, 90, 110, 60, 85, 95, 80, 90],
                "hr": [45, 40, 30, 25, 35, 20, 22, 35, 30, 28],
                "rbi": [110, 100, 90, 95, 95, 55, 70, 100, 85, 80],
                "sb": [5, 10, 15, 8, 30, 10, 25, 5, 3, 20],
                "avg": [0.279, 0.333, 0.304, 0.330, 0.294, 0.286, 0.286, 0.310, 0.287, 0.292],
                "ip": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "w": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "sv": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "k": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "era": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "whip": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "adp": [1, 2, 5, 8, 3, 20, 15, 4, 18, 10],
            }
        )

    def test_evaluate_trade_includes_concentration_keys(self):
        """Result should include Phase 4 concentration risk keys."""
        from unittest.mock import patch

        from src.engine.output.trade_evaluator import evaluate_trade

        pool = self._make_test_pool()

        with patch("src.engine.output.trade_evaluator.load_league_standings") as mock_s:
            mock_s.return_value = pd.DataFrame()

            result = evaluate_trade(
                giving_ids=[1],  # Judge (NYY)
                receiving_ids=[5],  # Acuna (ATL)
                user_roster_ids=[1, 2, 3, 4, 8],  # 2 NYY, 2 LAD, 1 NYY
                player_pool=pool,
                enable_context=True,
            )

        assert "concentration_hhi_before" in result
        assert "concentration_hhi_after" in result
        assert "concentration_delta" in result

    def test_evaluate_trade_context_disabled_backward_compat(self):
        """With enable_context=False, Phase 4 keys should be absent."""
        from unittest.mock import patch

        from src.engine.output.trade_evaluator import evaluate_trade

        pool = self._make_test_pool()

        with patch("src.engine.output.trade_evaluator.load_league_standings") as mock_s:
            mock_s.return_value = pd.DataFrame()

            result = evaluate_trade(
                giving_ids=[1],
                receiving_ids=[5],
                user_roster_ids=[1, 2, 3, 4, 8],
                player_pool=pool,
                enable_context=False,
            )

        # Phase 4 keys should NOT be present
        assert "concentration_hhi_before" not in result
        # But standard keys should still work
        assert "grade" in result
        assert "surplus_sgp" in result

    def test_concentration_diversification_improves_grade(self):
        """Trading away a concentrated player should reduce concentration penalty."""
        from unittest.mock import patch

        from src.engine.output.trade_evaluator import evaluate_trade

        pool = self._make_test_pool()

        with patch("src.engine.output.trade_evaluator.load_league_standings") as mock_s:
            mock_s.return_value = pd.DataFrame()

            # Trade Judge (NYY) for Acuna (ATL) — diversifies away from NYY
            result = evaluate_trade(
                giving_ids=[1],  # Judge (NYY)
                receiving_ids=[5],  # Acuna (ATL)
                user_roster_ids=[1, 8, 2, 3, 4],  # 2 NYY, 2 LAD, 1 other
                player_pool=pool,
                enable_context=True,
            )

        # After trade: 1 NYY (Soto), 2 LAD, 1 ATL (Acuna) → less concentrated
        assert result["concentration_delta"] < 0  # HHI decreased
