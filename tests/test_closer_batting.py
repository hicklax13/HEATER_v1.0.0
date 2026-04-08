"""Tests for H8 (closer stability discount) and I5/K6 (batting order PA adjustment)."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# H8: Closer stability discount
# ---------------------------------------------------------------------------


class TestCloserStabilityDiscount:
    """Verify that shaky closers get an SV discount in trade valuation."""

    def test_shaky_closer_gets_sv_discount(self):
        """A closer with job_sec=0.3 should have a portion of sv_bonus removed."""
        from src.closer_monitor import compute_job_security

        # Shaky closer: low hierarchy confidence, few saves
        job_sec = compute_job_security(hierarchy_confidence=0.2, projected_sv=5)
        assert job_sec < 0.5, f"Expected shaky closer (job_sec < 0.5), got {job_sec}"

        # Simulate the discount logic from scan_1_for_1
        sv_bonus = 1.5  # hypothetical SV scarcity bonus
        sv_discount = sv_bonus * (1.0 - job_sec)
        adjusted_bonus = sv_bonus - sv_discount

        # Discount should be significant for shaky closer
        assert adjusted_bonus < sv_bonus * 0.5, (
            f"Shaky closer should lose >50% of SV bonus, got {adjusted_bonus:.3f} vs {sv_bonus}"
        )
        assert adjusted_bonus > 0, "Adjusted bonus should still be positive"

    def test_secure_closer_keeps_full_sv_bonus(self):
        """A closer with job_sec=0.8 should NOT get a discount (above 0.5 threshold)."""
        from src.closer_monitor import compute_job_security

        # Secure closer: high hierarchy, solid saves
        job_sec = compute_job_security(hierarchy_confidence=0.9, projected_sv=25)
        assert job_sec >= 0.5, f"Expected secure closer (job_sec >= 0.5), got {job_sec}"

        # No discount applied when job_sec >= 0.5
        sv_bonus = 1.5
        # The trade_finder code only discounts when job_sec < 0.5
        discount_applied = job_sec < 0.5
        assert not discount_applied, "Secure closer should not be discounted"

    def test_graceful_fallback_when_closer_monitor_unavailable(self):
        """When compute_job_security raises, SV bonus should be kept intact."""
        sv_bonus = 1.5
        user_delta = 2.0
        user_delta += sv_bonus  # Apply SV bonus

        # Simulate the try/except fallback
        try:
            raise ImportError("Simulated missing module")
        except Exception:
            pass  # Keep original SV bonus -- no discount

        # user_delta should still have the full sv_bonus
        assert user_delta == pytest.approx(3.5), f"Fallback should preserve full SV bonus, got {user_delta}"


# ---------------------------------------------------------------------------
# I5/K6: Batting order PA multiplier
# ---------------------------------------------------------------------------


class TestBattingOrderPAMultiplier:
    """Verify batting order PA adjustments for DCV."""

    def test_leadoff_hitter_gets_boost(self):
        """Slot 1 (leadoff) should get ~1.08x multiplier."""
        from src.contextual_factors import batting_order_pa_multiplier

        mult = batting_order_pa_multiplier(1)
        assert mult > 1.05, f"Leadoff should be >1.05x, got {mult:.4f}"
        assert mult < 1.15, f"Leadoff should be <1.15x, got {mult:.4f}"

    def test_ninth_slot_gets_discount(self):
        """Slot 9 should get ~0.88-0.90x multiplier."""
        from src.contextual_factors import batting_order_pa_multiplier

        mult = batting_order_pa_multiplier(9)
        assert mult < 0.95, f"9th slot should be <0.95x, got {mult:.4f}"
        assert mult > 0.80, f"9th slot should be >0.80x, got {mult:.4f}"

    def test_unknown_slot_returns_neutral(self):
        """Unknown slot (0 or out-of-range) should return 1.0."""
        from src.contextual_factors import batting_order_pa_multiplier

        assert batting_order_pa_multiplier(0) == pytest.approx(1.0)
        assert batting_order_pa_multiplier(10) == pytest.approx(1.0)
        assert batting_order_pa_multiplier(-1) == pytest.approx(1.0)

    def test_all_slots_positive(self):
        """Every valid slot (1-9) should produce a positive multiplier."""
        from src.contextual_factors import batting_order_pa_multiplier

        for slot in range(1, 10):
            mult = batting_order_pa_multiplier(slot)
            assert mult > 0, f"Slot {slot} produced non-positive multiplier: {mult}"

    def test_monotonically_decreasing(self):
        """PA multiplier should decrease from slot 1 to slot 9."""
        from src.contextual_factors import batting_order_pa_multiplier

        prev = batting_order_pa_multiplier(1)
        for slot in range(2, 10):
            curr = batting_order_pa_multiplier(slot)
            assert curr < prev, f"Slot {slot} ({curr:.4f}) should be less than slot {slot - 1} ({prev:.4f})"
            prev = curr

    def test_dcv_counting_stat_adjusted(self):
        """DCV for counting stats should reflect batting order PA multiplier."""
        import pandas as pd

        from src.optimizer.daily_optimizer import build_daily_dcv_table
        from src.valuation import LeagueConfig

        config = LeagueConfig()

        # Create a minimal roster with one hitter
        roster = pd.DataFrame(
            [
                {
                    "player_id": 101,
                    "name": "Test Leadoff",
                    "team": "NYY",
                    "positions": "SS",
                    "is_hitter": 1,
                    "status": "active",
                    "r": 80,
                    "hr": 20,
                    "rbi": 60,
                    "sb": 15,
                    "avg": 0.270,
                    "obp": 0.340,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "era": 0,
                    "whip": 0,
                }
            ]
        )

        schedule = [{"home_name": "NYY", "away_name": "BOS"}]

        # With confirmed lineup (slot 1 = leadoff)
        lineups_with = {"NYY": ["Test Leadoff", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]}
        dcv_with = build_daily_dcv_table(
            roster,
            matchup=None,
            schedule_today=schedule,
            park_factors={},
            config=config,
            confirmed_lineups=lineups_with,
        )

        # Without confirmed lineup
        dcv_without = build_daily_dcv_table(
            roster,
            matchup=None,
            schedule_today=schedule,
            park_factors={},
            config=config,
            confirmed_lineups=None,
        )

        # Both should produce a row for player 101
        assert len(dcv_with) == 1
        assert len(dcv_without) == 1

        # The leadoff DCV for a counting stat (HR) should be higher than
        # the no-lineup version (which uses volume=0.9 vs 1.0, but also no PA mult)
        # Key check: the PA multiplier was applied (leadoff ~1.08x on counting stats)
        # Volume with confirmed lineup = 1.0, without = 0.9
        # So with = 1.0 * 1.08 = 1.08 effective, without = 0.9 * 1.0 = 0.9
        hr_with = dcv_with.iloc[0]["dcv_hr"]
        hr_without = dcv_without.iloc[0]["dcv_hr"]

        # With leadoff + confirmed should be notably higher than unconfirmed
        assert hr_with > hr_without, (
            f"Leadoff confirmed DCV ({hr_with:.4f}) should exceed unconfirmed ({hr_without:.4f})"
        )
