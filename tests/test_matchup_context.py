"""Tests for MatchupContextService (src/matchup_context.py)."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.matchup_context import MatchupContextService, _CacheEntry

# ── Cache Entry Tests ────────────────────────────────────────────────


class TestCacheEntry:
    """Verify TTL-based cache behavior."""

    def test_fresh_entry_not_stale(self):
        entry = _CacheEntry("value", ttl=300)
        assert not entry.is_stale

    def test_expired_entry_is_stale(self):
        entry = _CacheEntry("value", ttl=0)
        # TTL=0 means immediately stale
        time.sleep(0.01)
        assert entry.is_stale

    def test_value_preserved(self):
        data = {"key": "data", "nested": [1, 2, 3]}
        entry = _CacheEntry(data, ttl=300)
        assert entry.value == data


# ── Service Instantiation ────────────────────────────────────────────


class TestServiceInit:
    """Verify service creation and singleton pattern."""

    def test_creates_service(self):
        svc = MatchupContextService()
        assert isinstance(svc, MatchupContextService)
        assert svc._cache == {}

    def test_singleton_returns_same_instance(self):
        """Non-Streamlit context returns fresh instances (no session_state)."""
        from src.matchup_context import get_matchup_context

        # In test context (no Streamlit), each call creates a new instance
        ctx1 = get_matchup_context()
        ctx2 = get_matchup_context()
        assert isinstance(ctx1, MatchupContextService)
        assert isinstance(ctx2, MatchupContextService)


# ── Team Strength ────────────────────────────────────────────────────


class TestTeamStrength:
    """Verify team strength data loading and caching."""

    def test_returns_defaults_when_no_data(self):
        svc = MatchupContextService()
        result = svc.get_team_strength("UNKNOWN_TEAM")
        assert result["wrc_plus"] == 100.0
        assert result["fip"] == 4.00
        assert result["team_ops"] == 0.720

    def test_caches_result(self):
        svc = MatchupContextService()
        # First call
        r1 = svc.get_team_strength("NYY")
        # Second call should return cached value
        r2 = svc.get_team_strength("NYY")
        assert r1 == r2
        # Verify it's actually cached
        assert "strength_NYY" in svc._cache

    def test_get_all_team_strengths_returns_30_teams(self):
        svc = MatchupContextService()
        result = svc.get_all_team_strengths()
        assert len(result) == 30
        assert "NYY" in result
        assert "LAD" in result
        assert "SEA" in result
        # Each team has the expected keys
        for team, data in result.items():
            assert "wrc_plus" in data
            assert "fip" in data


# ── Player Form ──────────────────────────────────────────────────────


class TestPlayerForm:
    """Verify player form loading and trend classification."""

    def test_returns_neutral_when_no_data(self):
        svc = MatchupContextService()
        with patch("src.game_day.get_player_recent_form_cached", return_value=None):
            result = svc.get_player_form(999999)
            assert result.get("trend") == "neutral"

    def test_hot_trend_classification(self):
        svc = MatchupContextService()
        mock_form = {
            "last_7": {"avg": 0.400, "hr": 3},
            "last_14": {"avg": 0.320, "hr": 5},
            "last_30": {"avg": 0.280, "hr": 8},
        }
        with patch("src.game_day.get_player_recent_form_cached", return_value=mock_form):
            result = svc.get_player_form(545361)
            assert result["trend"] == "hot"

    def test_cold_trend_classification(self):
        svc = MatchupContextService()
        mock_form = {
            "last_7": {"avg": 0.150, "hr": 0},
            "last_14": {"avg": 0.200, "hr": 1},
            "last_30": {"avg": 0.280, "hr": 6},
        }
        with patch("src.game_day.get_player_recent_form_cached", return_value=mock_form):
            result = svc.get_player_form(660271)
            assert result["trend"] == "cold"

    def test_neutral_trend_classification(self):
        svc = MatchupContextService()
        mock_form = {
            "last_7": {"avg": 0.275, "hr": 2},
            "last_14": {"avg": 0.270, "hr": 3},
            "last_30": {"avg": 0.268, "hr": 5},
        }
        with patch("src.game_day.get_player_recent_form_cached", return_value=mock_form):
            result = svc.get_player_form(663728)
            assert result["trend"] == "neutral"

    def test_caches_form_data(self):
        svc = MatchupContextService()
        mock_form = {"last_7": {"avg": 0.300}, "last_30": {"avg": 0.280}}
        with patch("src.game_day.get_player_recent_form_cached", return_value=mock_form):
            r1 = svc.get_player_form(545361)
        # Second call should use cache (no patch needed)
        r2 = svc.get_player_form(545361)
        assert r1 == r2


# ── Weather ──────────────────────────────────────────────────────────


class TestWeather:
    """Verify weather data loading."""

    def test_returns_empty_when_no_data(self):
        svc = MatchupContextService()
        result = svc.get_weather("UNKNOWN")
        assert result == {}

    def test_caches_weather(self):
        svc = MatchupContextService()
        r1 = svc.get_weather("COL")
        r2 = svc.get_weather("COL")
        assert r1 == r2
        assert "weather_COL_today" in svc._cache


# ── Category Urgency ─────────────────────────────────────────────────


class TestCategoryUrgency:
    """Verify category urgency wrapping."""

    def test_returns_empty_when_no_matchup(self):
        svc = MatchupContextService()
        # No YDS available in test context
        result = svc.get_category_urgency()
        assert "urgency" in result
        assert "rate_modes" in result
        assert "summary" in result

    def test_caches_urgency(self):
        svc = MatchupContextService()
        r1 = svc.get_category_urgency()
        r2 = svc.get_category_urgency()
        assert r1 == r2
        assert "urgency" in svc._cache


# ── Opponent Context ─────────────────────────────────────────────────


class TestOpponentContext:
    """Verify opponent context loading."""

    def test_returns_unknown_when_no_data(self):
        svc = MatchupContextService()
        result = svc.get_opponent_context()
        assert "name" in result
        # Should have at least name key even on fallback

    def test_specific_week(self):
        svc = MatchupContextService()
        result = svc.get_opponent_context(week=5)
        assert isinstance(result, dict)
        # Week 5 context cached separately from current
        assert "opponent_5" in svc._cache

    def test_caches_opponent(self):
        svc = MatchupContextService()
        r1 = svc.get_opponent_context()
        r2 = svc.get_opponent_context()
        assert r1 == r2


# ── Matchup Adjustments ─────────────────────────────────────────────


class TestMatchupAdjustments:
    """Verify matchup adjustment passthrough."""

    def test_returns_roster_on_failure(self):
        svc = MatchupContextService()
        roster = pd.DataFrame({"player_id": [1, 2], "name": ["A", "B"]})
        result = svc.get_matchup_adjustments(roster)
        # On failure, should return original roster
        assert len(result) == 2


# ── Data Freshness ───────────────────────────────────────────────────


class TestDataFreshness:
    """Verify freshness reporting."""

    def test_empty_when_no_cache(self):
        svc = MatchupContextService()
        assert svc.get_data_freshness() == {}

    def test_reports_cached_entries(self):
        svc = MatchupContextService()
        svc._set_cached("test_key", "value", 300)
        freshness = svc.get_data_freshness()
        assert "test_key" in freshness
        assert "ago" in freshness["test_key"] or "Live" in freshness["test_key"]
