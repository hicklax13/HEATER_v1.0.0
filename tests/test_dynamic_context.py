"""Tests for dynamic context — replaces hardcoded weeks_remaining, schedule_strength, etc."""

from datetime import date

import pytest

from src.validation.dynamic_context import (
    compute_injury_exposure,
    compute_momentum,
    compute_schedule_strength,
    compute_weeks_remaining,
)


class TestWeeksRemaining:
    """Replaces hardcoded weeks_remaining=16."""

    def test_preseason_returns_full_season(self):
        """Before Opening Day, full season ahead."""
        result = compute_weeks_remaining(as_of=date(2026, 3, 1), season=2026)
        assert result == 22

    def test_opening_day_returns_full_season(self):
        result = compute_weeks_remaining(as_of=date(2026, 3, 26), season=2026)
        assert result == 22

    def test_mid_season(self):
        """Roughly mid-June, ~13 weeks in, ~9 remaining."""
        result = compute_weeks_remaining(as_of=date(2026, 6, 25), season=2026)
        assert 8 <= result <= 10

    def test_late_season(self):
        """Early September, ~2 weeks left."""
        result = compute_weeks_remaining(as_of=date(2026, 9, 15), season=2026)
        assert 1 <= result <= 4

    def test_post_season_returns_minimum(self):
        result = compute_weeks_remaining(as_of=date(2026, 10, 15), season=2026)
        assert result == 1

    def test_never_returns_zero(self):
        """Minimum is always 1."""
        result = compute_weeks_remaining(as_of=date(2026, 12, 31), season=2026)
        assert result >= 1

    def test_today_is_reasonable(self):
        """Current date (2026-03-24) is pre-season."""
        result = compute_weeks_remaining(as_of=date(2026, 3, 24), season=2026)
        assert result == 22  # Full season ahead


class TestScheduleStrength:
    """Replaces hardcoded schedule_strength=0.5."""

    def test_returns_none_without_data(self):
        """Should return None (honest), not 0.5 (fake)."""
        result = compute_schedule_strength("team1")
        assert result is None

    def test_returns_none_with_empty_standings(self):
        result = compute_schedule_strength("team1", standings={}, remaining_opponents=[])
        assert result is None

    def test_all_strong_opponents(self):
        standings = {
            "opp1": {"wins": 80, "losses": 20},
            "opp2": {"wins": 75, "losses": 25},
        }
        result = compute_schedule_strength("team1", standings=standings, remaining_opponents=["opp1", "opp2"])
        assert result is not None
        assert result > 0.7

    def test_all_weak_opponents(self):
        standings = {
            "opp1": {"wins": 20, "losses": 80},
            "opp2": {"wins": 25, "losses": 75},
        }
        result = compute_schedule_strength("team1", standings=standings, remaining_opponents=["opp1", "opp2"])
        assert result is not None
        assert result < 0.5

    def test_mixed_opponents(self):
        standings = {
            "opp1": {"wins": 50, "losses": 50},
            "opp2": {"wins": 50, "losses": 50},
        }
        result = compute_schedule_strength("team1", standings=standings, remaining_opponents=["opp1", "opp2"])
        assert result is not None
        assert 0.4 <= result <= 0.6


class TestInjuryExposure:
    """Replaces hardcoded injury_exposure=0.1."""

    def test_returns_none_without_data(self):
        """Should return None (honest), not 0.1 (fake)."""
        result = compute_injury_exposure([1, 2, 3])
        assert result is None

    def test_returns_none_for_empty_roster(self):
        result = compute_injury_exposure([])
        assert result is None

    def test_healthy_roster(self):
        injury_data = {1: 0.95, 2: 0.90, 3: 0.85}
        result = compute_injury_exposure([1, 2, 3], injury_data=injury_data)
        assert result is not None
        assert result < 0.2

    def test_injured_roster(self):
        injury_data = {1: 0.30, 2: 0.40, 3: 0.25}
        current_il = {1, 3}
        result = compute_injury_exposure([1, 2, 3], injury_data=injury_data, current_il=current_il)
        assert result is not None
        assert result > 0.5

    def test_current_il_dominates(self):
        """60% weight on current IL status."""
        # All healthy historically but 2/3 currently on IL
        injury_data = {1: 0.95, 2: 0.95, 3: 0.95}
        current_il = {1, 2}
        result = compute_injury_exposure([1, 2, 3], injury_data=injury_data, current_il=current_il)
        assert result is not None
        assert result > 0.3  # IL fraction = 2/3 * 0.6 = 0.4


class TestMomentum:
    """Replaces hardcoded momentum=1.0."""

    def test_returns_none_without_data(self):
        """Should return None (honest), not 1.0 (fake)."""
        result = compute_momentum(None)
        assert result is None

    def test_returns_none_for_empty_list(self):
        result = compute_momentum([])
        assert result is None

    def test_hot_streak(self):
        """Won most categories in last 3 weeks."""
        results = [(9, 3), (10, 2), (8, 4)]  # (cats_won, cats_lost)
        result = compute_momentum(results)
        assert result is not None
        assert result > 1.3

    def test_cold_streak(self):
        """Lost most categories in last 3 weeks."""
        results = [(3, 9), (2, 10), (4, 8)]
        result = compute_momentum(results)
        assert result is not None
        assert result < 0.7

    def test_average(self):
        """Split categories evenly."""
        results = [(6, 6), (6, 6), (6, 6)]
        result = compute_momentum(results)
        assert result is not None
        assert 0.9 <= result <= 1.1

    def test_uses_only_last_n_weeks(self):
        """Should only consider last 3 weeks by default."""
        results = [
            (2, 10),  # Week 1 — bad
            (2, 10),  # Week 2 — bad
            (10, 2),  # Week 3 — good (most recent)
            (10, 2),  # Week 4 — good
            (10, 2),  # Week 5 — good
        ]
        result = compute_momentum(results, n_weeks=3)
        assert result is not None
        assert result > 1.3  # Last 3 weeks were good
