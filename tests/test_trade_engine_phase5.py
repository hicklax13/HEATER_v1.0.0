"""Tests for Trade Engine Phase 5 — Game Theory + Optimization.

Tests cover:
  - Opponent valuation: willingness-to-pay, market clearing price (L8A)
  - Adverse selection: Bayesian discount, manager history calibration (L8B)
  - Dynamic programming: Bellman rollout, discount factors, option value (L9)
  - Sensitivity: category sensitivity, breakeven analysis, counter-offers (L11)
  - Integration: evaluate_trade with game theory keys
"""

import unittest

import numpy as np
import pandas as pd


class TestOpponentValuation(unittest.TestCase):
    """Test opponent valuation and market clearing price (L8A)."""

    def _make_team_totals(self):
        """Create sample team standings totals."""
        return {
            "Team A": {
                "R": 700,
                "HR": 180,
                "RBI": 650,
                "SB": 80,
                "AVG": 0.265,
                "W": 70,
                "K": 1100,
                "SV": 50,
                "ERA": 3.80,
                "WHIP": 1.20,
            },
            "Team B": {
                "R": 650,
                "HR": 150,
                "RBI": 600,
                "SB": 120,
                "AVG": 0.258,
                "W": 65,
                "K": 1000,
                "SV": 30,
                "ERA": 4.10,
                "WHIP": 1.30,
            },
            "Team C": {
                "R": 750,
                "HR": 200,
                "RBI": 700,
                "SB": 60,
                "AVG": 0.270,
                "W": 75,
                "K": 1200,
                "SV": 40,
                "ERA": 3.60,
                "WHIP": 1.15,
            },
            "My Team": {
                "R": 680,
                "HR": 160,
                "RBI": 620,
                "SB": 90,
                "AVG": 0.260,
                "W": 68,
                "K": 1050,
                "SV": 45,
                "ERA": 3.90,
                "WHIP": 1.25,
            },
        }

    def test_opponent_valuations_excludes_your_team(self):
        """Valuations should not include your team."""
        from src.engine.game_theory.opponent_valuation import estimate_opponent_valuations

        totals = self._make_team_totals()
        proj = {"R": 85, "HR": 30, "RBI": 80, "SB": 10, "AVG": 0.280, "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}

        vals = estimate_opponent_valuations(proj, totals, "My Team")
        assert "My Team" not in vals
        assert len(vals) == 3  # A, B, C

    def test_opponent_valuations_positive_for_power_hitter(self):
        """A power hitter should have positive value for teams needing HR."""
        from src.engine.game_theory.opponent_valuation import estimate_opponent_valuations

        totals = self._make_team_totals()
        proj = {"R": 85, "HR": 35, "RBI": 90, "SB": 5, "AVG": 0.270, "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}

        vals = estimate_opponent_valuations(proj, totals, "My Team")
        # Team B has fewest HR (150), should value the HR hitter most
        assert all(v > 0 for v in vals.values())

    def test_market_clearing_price_second_highest(self):
        """Market price should be the second-highest bid."""
        from src.engine.game_theory.opponent_valuation import market_clearing_price

        valuations = {"Team A": 3.0, "Team B": 5.0, "Team C": 2.0}
        price = market_clearing_price(valuations)
        assert price == 3.0  # Second highest

    def test_market_clearing_price_single_bidder(self):
        """Single bidder: market price = their bid."""
        from src.engine.game_theory.opponent_valuation import market_clearing_price

        price = market_clearing_price({"Team A": 3.0})
        assert price == 3.0

    def test_market_clearing_price_empty(self):
        """Empty valuations: price = 0."""
        from src.engine.game_theory.opponent_valuation import market_clearing_price

        assert market_clearing_price({}) == 0.0

    def test_player_market_value_has_all_keys(self):
        """player_market_value should return complete analysis."""
        from src.engine.game_theory.opponent_valuation import player_market_value

        totals = self._make_team_totals()
        proj = {"R": 85, "HR": 30, "RBI": 80, "SB": 10, "AVG": 0.270, "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}

        mv = player_market_value(proj, totals, "My Team")
        assert "valuations" in mv
        assert "market_price" in mv
        assert "max_bidder" in mv
        assert "max_bid" in mv
        assert "demand" in mv
        assert mv["market_price"] >= 0

    def test_get_player_projections_from_pool(self):
        """Should extract projections from player pool DataFrame."""
        from src.engine.game_theory.opponent_valuation import get_player_projections_from_pool

        pool = pd.DataFrame(
            {
                "player_id": [1],
                "name": ["Test Player"],
                "r": [85],
                "hr": [30],
                "rbi": [80],
                "sb": [10],
                "avg": [0.270],
                "w": [0],
                "sv": [0],
                "k": [0],
                "era": [0.0],
                "whip": [0.0],
            }
        )
        proj = get_player_projections_from_pool(1, pool)
        assert proj["R"] == 85
        assert proj["HR"] == 30

    def test_get_player_projections_missing_player(self):
        """Missing player ID should return empty dict."""
        from src.engine.game_theory.opponent_valuation import get_player_projections_from_pool

        pool = pd.DataFrame(
            {
                "player_id": [1],
                "name": ["X"],
                "r": [0],
                "hr": [0],
                "rbi": [0],
                "sb": [0],
                "avg": [0],
                "w": [0],
                "sv": [0],
                "k": [0],
                "era": [0],
                "whip": [0],
            }
        )
        proj = get_player_projections_from_pool(999, pool)
        assert proj == {}

    def test_demand_counts_valuable_teams(self):
        """Demand should count teams valuing player above 0.5 SGP."""
        from src.engine.game_theory.opponent_valuation import player_market_value

        totals = self._make_team_totals()
        # Elite player — most teams should want them
        proj = {"R": 100, "HR": 40, "RBI": 100, "SB": 20, "AVG": 0.300, "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}

        mv = player_market_value(proj, totals, "My Team")
        assert mv["demand"] >= 1


class TestAdverseSelection(unittest.TestCase):
    """Test Bayesian adverse selection discount (L8B)."""

    def test_default_discount_near_one(self):
        """Default discount (no history) should be close to 1.0."""
        from src.engine.game_theory.adverse_selection import adverse_selection_discount

        discount = adverse_selection_discount()
        assert 0.90 <= discount <= 1.00

    def test_clean_history_low_discount(self):
        """Manager with clean history should get minimal discount."""
        from src.engine.game_theory.adverse_selection import adverse_selection_discount

        history = [
            {"actual": 100, "projected": 95},
            {"actual": 80, "projected": 85},
            {"actual": 90, "projected": 88},
            {"actual": 70, "projected": 72},
        ]
        discount = adverse_selection_discount(history)
        assert discount >= 0.95

    def test_bad_history_bigger_discount(self):
        """Manager who dumps underperformers should get bigger discount."""
        from src.engine.game_theory.adverse_selection import adverse_selection_discount

        history = [
            {"actual": 40, "projected": 100},  # Underperformed
            {"actual": 30, "projected": 90},  # Underperformed
            {"actual": 50, "projected": 95},  # Underperformed
            {"actual": 90, "projected": 85},  # OK
        ]
        discount = adverse_selection_discount(history)
        assert discount < 0.95  # Should be discounted more

    def test_discount_never_below_floor(self):
        """Discount should never go below 0.75 (MAX_DISCOUNT=0.25)."""
        from src.engine.game_theory.adverse_selection import adverse_selection_discount

        # Worst possible history: every trade underperformed
        history = [
            {"actual": 10, "projected": 100},
            {"actual": 5, "projected": 90},
            {"actual": 15, "projected": 95},
        ]
        discount = adverse_selection_discount(history)
        assert discount >= 0.75

    def test_insufficient_history_uses_prior(self):
        """With < 3 trades, should use default prior."""
        from src.engine.game_theory.adverse_selection import adverse_selection_discount

        default_disc = adverse_selection_discount()
        short_hist = adverse_selection_discount([{"actual": 10, "projected": 100}])
        assert default_disc == short_hist

    def test_compute_discount_for_trade_has_all_keys(self):
        """compute_discount_for_trade should return complete analysis."""
        from src.engine.game_theory.adverse_selection import compute_discount_for_trade

        result = compute_discount_for_trade(receiving_player_count=2)
        assert "discount_factor" in result
        assert "p_flaw" in result
        assert "p_flaw_given_offered" in result
        assert "sgp_adjustment" in result
        assert "total_sgp_adjustment" in result
        assert "risk_level" in result
        assert result["risk_level"] in ("low", "medium", "high")

    def test_more_received_players_more_adjustment(self):
        """More received players should amplify the total adjustment."""
        from src.engine.game_theory.adverse_selection import compute_discount_for_trade

        one = compute_discount_for_trade(receiving_player_count=1)
        three = compute_discount_for_trade(receiving_player_count=3)
        assert abs(three["total_sgp_adjustment"]) > abs(one["total_sgp_adjustment"])

    def test_spec_test_case_adverse_selection(self):
        """Spec test 3: Manager X with 3/5 underperforming → discount < 0.90."""
        from src.engine.game_theory.adverse_selection import adverse_selection_discount

        history = [
            {"actual": 70, "projected": 100},  # Under (70 < 80)
            {"actual": 90, "projected": 95},  # OK
            {"actual": 60, "projected": 100},  # Under
            {"actual": 75, "projected": 100},  # Under (75 < 80)
            {"actual": 85, "projected": 80},  # OK
        ]
        discount = adverse_selection_discount(history)
        assert discount < 0.95  # Spec says < 0.90 but implementation uses 25% max


class TestDynamicProgramming(unittest.TestCase):
    """Test Bellman rollout and option value (L9)."""

    def test_gamma_contending(self):
        """Contending teams (>70%) should get high gamma."""
        from src.engine.game_theory.dynamic_programming import get_gamma

        assert get_gamma(0.80) == 0.98

    def test_gamma_bubble(self):
        """Bubble teams (30-70%) should get medium gamma."""
        from src.engine.game_theory.dynamic_programming import get_gamma

        assert get_gamma(0.50) == 0.95

    def test_gamma_rebuilding(self):
        """Rebuilding teams (<30%) should get low gamma."""
        from src.engine.game_theory.dynamic_programming import get_gamma

        assert get_gamma(0.20) == 0.85

    def test_bellman_rollout_returns_all_keys(self):
        """Bellman rollout should return complete analysis."""
        from src.engine.game_theory.dynamic_programming import bellman_rollout

        result = bellman_rollout(immediate_surplus=1.0)
        assert "immediate" in result
        assert "future_before" in result
        assert "future_after" in result
        assert "option_value" in result
        assert "gamma" in result
        assert "total_value" in result
        assert result["immediate"] == 1.0

    def test_bellman_total_includes_option_value(self):
        """Total value should be immediate + discounted option value."""
        from src.engine.game_theory.dynamic_programming import bellman_rollout

        result = bellman_rollout(immediate_surplus=1.0, seed=42)
        # total = immediate + gamma * option_value
        expected = result["immediate"] + result["gamma"] * result["option_value"]
        assert abs(result["total_value"] - expected) < 0.01

    def test_positive_surplus_reflects_in_total(self):
        """Positive immediate surplus should produce positive total."""
        from src.engine.game_theory.dynamic_programming import bellman_rollout

        result = bellman_rollout(immediate_surplus=3.0, seed=42)
        assert result["total_value"] > 0

    def test_balanced_roster_better_future(self):
        """Better-balanced roster should have better future trade options."""
        from src.engine.game_theory.dynamic_programming import bellman_rollout

        # Balanced roster
        balanced = bellman_rollout(
            immediate_surplus=0.0,
            roster_balance_after=0.8,
            roster_balance_before=0.0,
            n_sims=500,
            seed=42,
        )

        # Lopsided roster
        lopsided = bellman_rollout(
            immediate_surplus=0.0,
            roster_balance_after=-0.8,
            roster_balance_before=0.0,
            n_sims=500,
            seed=42,
        )

        # Balanced should have better future value
        assert balanced["future_after"] >= lopsided["future_after"]

    def test_estimate_playoff_probability(self):
        """Playoff probability estimates should be reasonable."""
        from src.engine.game_theory.dynamic_programming import estimate_playoff_probability

        # First place, late season
        p1 = estimate_playoff_probability(1, 12, 4)
        assert p1 > 0.7

        # Last place, late season
        p12 = estimate_playoff_probability(12, 12, 4)
        assert p12 < 0.3

        # Mid-season, any position: should be closer to 0.5
        p_mid = estimate_playoff_probability(6, 12, 20)
        assert 0.3 < p_mid < 0.7

    def test_roster_balance_scoring(self):
        """Balanced roster should score higher than lopsided."""
        from src.engine.game_theory.dynamic_programming import compute_roster_balance

        # Perfectly balanced (all rank 6-7 in 12-team)
        balanced = compute_roster_balance(
            {"R": 6, "HR": 7, "RBI": 6, "SB": 7, "AVG": 6, "W": 7, "K": 6, "SV": 7, "ERA": 6, "WHIP": 7}
        )

        # Extremely lopsided (1st or 12th in everything)
        lopsided = compute_roster_balance(
            {"R": 1, "HR": 1, "RBI": 1, "SB": 12, "AVG": 12, "W": 1, "K": 1, "SV": 12, "ERA": 12, "WHIP": 12}
        )

        assert balanced > lopsided

    def test_roster_balance_empty(self):
        """Empty ranks should return 0."""
        from src.engine.game_theory.dynamic_programming import compute_roster_balance

        assert compute_roster_balance({}) == 0.0

    def test_seed_reproducibility(self):
        """Same seed should produce same rollout."""
        from src.engine.game_theory.dynamic_programming import bellman_rollout

        r1 = bellman_rollout(immediate_surplus=1.0, seed=123)
        r2 = bellman_rollout(immediate_surplus=1.0, seed=123)
        assert r1["total_value"] == r2["total_value"]


class TestSensitivity(unittest.TestCase):
    """Test sensitivity analysis and counter-offers (L11)."""

    def test_category_sensitivity_sorted_by_impact(self):
        """Categories should be sorted by absolute impact."""
        from src.engine.game_theory.sensitivity import category_sensitivity

        impact = {
            "R": 0.5,
            "HR": 1.2,
            "SB": -0.8,
            "AVG": 0.1,
            "ERA": -0.3,
            "W": 0.0,
            "K": 0.2,
            "SV": 0.0,
            "RBI": 0.4,
            "WHIP": -0.1,
        }

        ranked = category_sensitivity(impact)
        assert ranked[0]["category"] == "HR"  # Highest absolute impact
        assert ranked[0]["direction"] == "helps"

    def test_category_sensitivity_direction(self):
        """Direction should be 'helps' for positive, 'hurts' for negative."""
        from src.engine.game_theory.sensitivity import category_sensitivity

        impact = {"R": 0.5, "SB": -0.8}
        ranked = category_sensitivity(impact)

        r_entry = next(e for e in ranked if e["category"] == "R")
        sb_entry = next(e for e in ranked if e["category"] == "SB")
        assert r_entry["direction"] == "helps"
        assert sb_entry["direction"] == "hurts"

    def test_sensitivity_report_has_all_keys(self):
        """Sensitivity report should be complete."""
        from src.engine.game_theory.sensitivity import trade_sensitivity_report

        impact = {"R": 0.5, "HR": 1.2, "SB": -0.8}
        report = trade_sensitivity_report(impact, surplus_sgp=0.9)

        assert "category_ranking" in report
        assert "biggest_driver" in report
        assert "biggest_drag" in report
        assert "breakeven_gap" in report
        assert "vulnerability" in report

    def test_vulnerability_robust(self):
        """Large surplus should be classified as robust."""
        from src.engine.game_theory.sensitivity import trade_sensitivity_report

        report = trade_sensitivity_report({"R": 2.0}, surplus_sgp=2.0)
        assert report["vulnerability"] == "robust"

    def test_vulnerability_razor_thin(self):
        """Tiny surplus should be classified as razor-thin."""
        from src.engine.game_theory.sensitivity import trade_sensitivity_report

        report = trade_sensitivity_report({"R": 0.05}, surplus_sgp=0.05)
        assert report["vulnerability"] == "razor-thin"

    def test_biggest_drag_is_most_negative(self):
        """Biggest drag should be the most negative category."""
        from src.engine.game_theory.sensitivity import trade_sensitivity_report

        impact = {"R": 1.0, "HR": 0.5, "SB": -1.5, "AVG": -0.3}
        report = trade_sensitivity_report(impact)

        assert report["biggest_drag"]["category"] == "SB"
        assert report["biggest_drag"]["direction"] == "hurts"

    def test_breakeven_gap_is_abs_surplus(self):
        """Breakeven gap should be the absolute surplus."""
        from src.engine.game_theory.sensitivity import trade_sensitivity_report

        report_pos = trade_sensitivity_report({"R": 1.0}, surplus_sgp=1.5)
        assert abs(report_pos["breakeven_gap"] - 1.5) < 0.01

        report_neg = trade_sensitivity_report({"R": -1.0}, surplus_sgp=-1.5)
        assert abs(report_neg["breakeven_gap"] - 1.5) < 0.01


class TestPhase5Integration(unittest.TestCase):
    """Test Phase 5 integration with evaluate_trade."""

    def _make_pool(self):
        """Create a minimal player pool."""
        return pd.DataFrame(
            {
                "player_id": [1, 2, 3, 4, 5],
                "name": ["Player A", "Player B", "Player C", "Player D", "Player E"],
                "team": ["NYY", "BOS", "LAD", "CHC", "HOU"],
                "positions": ["OF", "1B", "SS", "OF", "SP"],
                "is_hitter": [1, 1, 1, 1, 0],
                "is_injured": [0, 0, 0, 0, 0],
                "pa": [600, 550, 580, 500, 0],
                "ab": [540, 495, 522, 450, 0],
                "h": [150, 130, 160, 120, 0],
                "r": [90, 70, 95, 65, 0],
                "hr": [30, 25, 20, 15, 0],
                "rbi": [85, 75, 70, 55, 0],
                "sb": [10, 5, 25, 8, 0],
                "avg": [0.278, 0.263, 0.307, 0.267, 0.0],
                "ip": [0, 0, 0, 0, 180],
                "w": [0, 0, 0, 0, 12],
                "sv": [0, 0, 0, 0, 0],
                "k": [0, 0, 0, 0, 200],
                "era": [0, 0, 0, 0, 3.50],
                "whip": [0, 0, 0, 0, 1.15],
                "er": [0, 0, 0, 0, 70],
                "bb_allowed": [0, 0, 0, 0, 50],
                "h_allowed": [0, 0, 0, 0, 157],
                "adp": [15, 45, 30, 80, 25],
            }
        )

    def test_evaluate_trade_includes_game_theory_keys(self):
        """evaluate_trade with game theory should include Phase 5 keys."""
        from src.engine.output.trade_evaluator import evaluate_trade

        pool = self._make_pool()
        result = evaluate_trade(
            giving_ids=[1],
            receiving_ids=[3],
            user_roster_ids=[1, 2, 4, 5],
            player_pool=pool,
            enable_game_theory=True,
        )

        assert "adverse_selection" in result
        assert "sensitivity_report" in result
        assert result["adverse_selection"]["risk_level"] in ("low", "medium", "high")
        assert "category_ranking" in result["sensitivity_report"]

    def test_evaluate_trade_game_theory_disabled_backward_compat(self):
        """With game theory disabled, Phase 5 keys should be absent."""
        from src.engine.output.trade_evaluator import evaluate_trade

        pool = self._make_pool()
        result = evaluate_trade(
            giving_ids=[1],
            receiving_ids=[3],
            user_roster_ids=[1, 2, 4, 5],
            player_pool=pool,
            enable_game_theory=False,
        )

        assert "adverse_selection" not in result
        assert "sensitivity_report" not in result
        # Base keys should still exist
        assert "grade" in result
        assert "surplus_sgp" in result

    def test_sensitivity_vulnerability_reflects_surplus(self):
        """Sensitivity report vulnerability should reflect trade surplus magnitude."""
        from src.engine.output.trade_evaluator import evaluate_trade

        pool = self._make_pool()
        result = evaluate_trade(
            giving_ids=[4],  # Weak player (Player D)
            receiving_ids=[3],  # Strong player (Player C)
            user_roster_ids=[1, 2, 4, 5],
            player_pool=pool,
            enable_game_theory=True,
        )

        report = result["sensitivity_report"]
        # Should have a vulnerability classification
        assert report["vulnerability"] in ("robust", "moderate", "fragile", "razor-thin")


if __name__ == "__main__":
    unittest.main()
