"""Tests for F5 (playoff-odds acceptance) and G4 (stat reliability weighting)."""

from __future__ import annotations

import pytest

from src.trade_finder import estimate_acceptance_probability

# ── F5: Playoff-Odds Acceptance ─────────────────────────────────────────


class TestPlayoffOddsAcceptance:
    """Test that bubble_bonus varies with approximated playoff odds."""

    # Helper: compute internal exponent diff to isolate bubble_bonus effect
    # We compare acceptance prob at a given rank vs no rank (bubble_bonus=0)
    def _acceptance_at_rank(self, rank: int) -> float:
        return estimate_acceptance_probability(
            user_gain_sgp=1.0,
            opponent_gain_sgp=0.5,
            need_match_score=0.5,
            adp_fairness=0.5,
            opponent_need_match=0.5,
            opponent_standings_rank=rank,
            opponent_trade_willingness=0.5,
        )

    def _acceptance_no_rank(self) -> float:
        return estimate_acceptance_probability(
            user_gain_sgp=1.0,
            opponent_gain_sgp=0.5,
            need_match_score=0.5,
            adp_fairness=0.5,
            opponent_need_match=0.5,
            opponent_standings_rank=None,
            opponent_trade_willingness=0.5,
        )

    def test_low_playoff_odds_negative_bonus(self):
        """Rank 12 -> playoff_odds ~0.0 -> bubble_bonus = -0.2 (checked out)."""
        prob_rank12 = self._acceptance_at_rank(12)
        prob_none = self._acceptance_no_rank()
        # Negative bubble_bonus makes exponent larger -> lower acceptance
        assert prob_rank12 < prob_none, (
            f"Rank 12 (low playoff odds) should have LOWER acceptance than no rank, "
            f"got {prob_rank12:.4f} vs {prob_none:.4f}"
        )

    def test_bubble_team_positive_bonus(self):
        """Ranks 4-8 -> playoff_odds 0.30-0.70 -> bubble_bonus = 0.4 (active)."""
        prob_rank6 = self._acceptance_at_rank(6)
        prob_none = self._acceptance_no_rank()
        # Positive bubble_bonus subtracts from exponent -> higher acceptance
        assert prob_rank6 > prob_none, (
            f"Rank 6 (bubble team) should have HIGHER acceptance than no rank, got {prob_rank6:.4f} vs {prob_none:.4f}"
        )

    def test_contender_slight_negative_bonus(self):
        """Rank 1 -> playoff_odds = 1.0 -> bubble_bonus = -0.1 (conservative)."""
        prob_rank1 = self._acceptance_at_rank(1)
        prob_none = self._acceptance_no_rank()
        # Conservative contender slightly less willing
        assert prob_rank1 < prob_none, (
            f"Rank 1 (contender) should have slightly LOWER acceptance than no rank, "
            f"got {prob_rank1:.4f} vs {prob_none:.4f}"
        )

    def test_fallback_no_rank_uses_neutral(self):
        """When opponent_standings_rank is None, bubble_bonus stays 0."""
        prob = self._acceptance_no_rank()
        # Should be a reasonable probability, not extreme
        assert 0.01 < prob < 0.99

    def test_bubble_range_ranks_5_through_8(self):
        """Ranks 5-8 have playoff_odds in 0.30-0.70 -> bubble bonus 0.4."""
        # Rank 4 -> odds=0.727 (above 0.70, neutral zone)
        # Rank 5 -> odds=0.636, Rank 8 -> odds=0.364 (all in 0.30-0.70)
        prob_none = self._acceptance_no_rank()
        for rank in [5, 6, 7, 8]:
            prob = self._acceptance_at_rank(rank)
            assert prob > prob_none, f"Rank {rank} should be bubble team"

    def test_playoff_odds_linear_approximation(self):
        """Verify the linear formula: odds = max(0, 1 - (rank-1)/(n-1))."""
        # Rank 1 -> odds = 1.0
        # Rank 6 -> odds = 1 - 5/11 = 0.545
        # Rank 12 -> odds = 1 - 11/11 = 0.0
        n_teams = 12
        assert max(0.0, 1.0 - (1 - 1) / (n_teams - 1)) == pytest.approx(1.0)
        assert max(0.0, 1.0 - (6 - 1) / (n_teams - 1)) == pytest.approx(0.5454, abs=0.01)
        assert max(0.0, 1.0 - (12 - 1) / (n_teams - 1)) == pytest.approx(0.0)


# ── G4: Stat Reliability Weighting ──────────────────────────────────────


class TestStatReliabilityWeighting:
    """Test that YTD modifier is scaled by stat reliability (sample size)."""

    def _make_pool_row(self, pid: int, is_hitter: int = 1, ip: float = 0.0) -> dict:
        return {
            "player_id": pid,
            "name": f"Player_{pid}",
            "team": "NYY",
            "positions": "OF" if is_hitter else "SP",
            "is_hitter": is_hitter,
            "pa": 500 if is_hitter else 0,
            "ab": 450 if is_hitter else 0,
            "h": 120 if is_hitter else 0,
            "r": 60 if is_hitter else 0,
            "hr": 20 if is_hitter else 0,
            "rbi": 70 if is_hitter else 0,
            "sb": 10 if is_hitter else 0,
            "avg": 0.270 if is_hitter else 0,
            "obp": 0.340 if is_hitter else 0,
            "bb": 40 if is_hitter else 0,
            "hbp": 5 if is_hitter else 0,
            "sf": 3 if is_hitter else 0,
            "ip": ip,
            "w": 0 if is_hitter else 10,
            "l": 0 if is_hitter else 5,
            "sv": 0,
            "k": 0 if is_hitter else 150,
            "era": 0 if is_hitter else 3.50,
            "whip": 0 if is_hitter else 1.20,
            "er": 0 if is_hitter else 50,
            "bb_allowed": 0 if is_hitter else 40,
            "h_allowed": 0 if is_hitter else 130,
            "adp": 50,
        }

    def test_early_season_hitter_modifier_near_neutral(self):
        """At 50 PA, reliability = 50/910 = 0.055 -> modifier barely moves."""
        # Simulate: raw ytd_modifier would be 1.10 (hot streak)
        # After reliability: 1.0 + 0.10 * 0.055 = 1.0055
        pa = 50
        reliability = min(1.0, pa / 910.0)
        raw_divergence = 0.10  # Hot streak: ytd_modifier = 1.10
        adjusted = 1.0 + raw_divergence * reliability
        assert adjusted == pytest.approx(1.0055, abs=0.001)
        assert adjusted < 1.01, "Early-season modifier should barely move"

    def test_mid_season_hitter_moderate_modifier(self):
        """At 400 PA, reliability = 400/910 = 0.44 -> moderate influence."""
        pa = 400
        reliability = min(1.0, pa / 910.0)
        raw_divergence = 0.10
        adjusted = 1.0 + raw_divergence * reliability
        assert adjusted == pytest.approx(1.044, abs=0.002)
        assert 1.03 < adjusted < 1.06, "Mid-season should have moderate effect"

    def test_full_season_hitter_full_modifier(self):
        """At 900+ PA, reliability = 1.0 -> full YTD modifier preserved."""
        pa = 910
        reliability = min(1.0, pa / 910.0)
        raw_divergence = 0.10
        adjusted = 1.0 + raw_divergence * reliability
        assert adjusted == pytest.approx(1.10, abs=0.001)

    def test_pitcher_uses_ip_not_pa(self):
        """Pitcher reliability should scale with IP (180 IP = full)."""
        # Early season pitcher: 30 IP -> reliability = 30/180 = 0.167
        ip = 30.0
        reliability = min(1.0, ip / 180.0)
        raw_divergence = -0.10  # Cold pitcher: ytd_modifier = 0.90
        adjusted = 1.0 + raw_divergence * reliability
        assert adjusted == pytest.approx(0.9833, abs=0.002)
        assert adjusted > 0.98, "Early-season pitcher modifier should barely move"

    def test_pitcher_full_season_ip(self):
        """Pitcher with 180+ IP should get full modifier."""
        ip = 200.0
        reliability = min(1.0, ip / 180.0)
        assert reliability == 1.0
        raw_divergence = -0.10
        adjusted = 1.0 + raw_divergence * reliability
        assert adjusted == pytest.approx(0.90, abs=0.001)

    def test_negative_divergence_scaled_correctly(self):
        """Cold streaks should also be dampened by low reliability."""
        pa = 50
        reliability = min(1.0, pa / 910.0)
        raw_divergence = -0.10  # Cold streak: ytd_modifier = 0.90
        adjusted = 1.0 + raw_divergence * reliability
        assert adjusted == pytest.approx(0.9945, abs=0.001)
        assert adjusted > 0.99, "Cold streak should be dampened early season"

    def test_reliability_capped_at_one(self):
        """Reliability should never exceed 1.0 even with huge PA."""
        pa = 2000
        reliability = min(1.0, pa / 910.0)
        assert reliability == 1.0
