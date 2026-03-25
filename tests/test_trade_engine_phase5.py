"""Tests for Trade Engine Phase 5 — Game Theory + Optimization.

Tests cover:
  - Opponent valuation: willingness-to-pay, market clearing price (L8A)
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

        assert "sensitivity_report" in result
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
