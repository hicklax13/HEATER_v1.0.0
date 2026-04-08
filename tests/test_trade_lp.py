"""Tests for LP-constrained baseline totals in Trade Finder.

Verifies that scan_1_for_1() can use LP-optimal starters-only baseline
instead of raw roster totals (which include bench inflation).
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from src.trade_finder import _lp_constrained_totals, scan_1_for_1
from src.valuation import LeagueConfig


def _make_pool(n_hitters: int = 10, n_pitchers: int = 8) -> pd.DataFrame:
    """Create a minimal player pool for testing."""
    rows = []
    pid = 1
    # Hitters
    for i in range(n_hitters):
        rows.append(
            {
                "player_id": pid,
                "name": f"Hitter_{pid}",
                "player_name": f"Hitter_{pid}",
                "team": "NYY",
                "positions": ["1B", "OF"][i % 2] if i < 8 else "Util",
                "is_hitter": 1,
                "pa": 500 + i * 10,
                "ab": 450 + i * 10,
                "h": 120 + i * 3,
                "r": 70 + i * 2,
                "hr": 20 + i,
                "rbi": 75 + i * 2,
                "sb": 5 + i,
                "avg": 0.267 + i * 0.003,
                "obp": 0.340 + i * 0.002,
                "bb": 50 + i,
                "hbp": 3,
                "sf": 4,
                "ip": 0,
                "w": 0,
                "l": 0,
                "sv": 0,
                "k": 0,
                "era": 0,
                "whip": 0,
                "er": 0,
                "bb_allowed": 0,
                "h_allowed": 0,
                "adp": 50 + i * 10,
                "status": "active",
            }
        )
        pid += 1
    # Pitchers
    for i in range(n_pitchers):
        rows.append(
            {
                "player_id": pid,
                "name": f"Pitcher_{pid}",
                "player_name": f"Pitcher_{pid}",
                "team": "LAD",
                "positions": "SP" if i < 6 else "RP",
                "is_hitter": 0,
                "pa": 0,
                "ab": 0,
                "h": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "ip": 150 + i * 10,
                "w": 10 + i,
                "l": 6 + i,
                "sv": 0 if i < 6 else 25 + i,
                "k": 140 + i * 10,
                "era": 3.50 + i * 0.1,
                "whip": 1.15 + i * 0.02,
                "er": 58 + i * 2,
                "bb_allowed": 45 + i * 3,
                "h_allowed": 130 + i * 5,
                "adp": 30 + i * 15,
                "status": "active",
            }
        )
        pid += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def pool() -> pd.DataFrame:
    """Standard 18-player pool: 10 hitters + 8 pitchers."""
    return _make_pool()


@pytest.fixture()
def config() -> LeagueConfig:
    return LeagueConfig()


class TestLpConstrainedTotals:
    """Tests for _lp_constrained_totals helper."""

    def test_returns_dict_when_pulp_available(self, pool: pd.DataFrame, config: LeagueConfig) -> None:
        """LP totals should return a dict with uppercase category keys."""
        roster_ids = pool["player_id"].tolist()
        result = _lp_constrained_totals(roster_ids, pool, config)
        # PuLP should be available in CI/dev
        try:
            import pulp  # noqa: F401

            assert result is not None
            assert isinstance(result, dict)
            for cat in ["R", "HR", "RBI", "SB", "AVG", "ERA"]:
                assert cat in result, f"Missing category {cat}"
        except ImportError:
            # If PuLP not installed, result should be None (graceful fallback)
            assert result is None

    def test_returns_none_when_pulp_unavailable(self, pool: pd.DataFrame, config: LeagueConfig) -> None:
        """Should return None gracefully when PuLP import fails."""
        roster_ids = pool["player_id"].tolist()
        with patch("src.lineup_optimizer.PULP_AVAILABLE", False):
            result = _lp_constrained_totals(roster_ids, pool, config)
            # The optimizer falls back to greedy which returns status != "Optimal"
            # so we should get None
            assert result is None

    def test_returns_none_on_empty_roster(self, pool: pd.DataFrame, config: LeagueConfig) -> None:
        """Empty roster should return None."""
        result = _lp_constrained_totals([], pool, config)
        assert result is None

    def test_returns_none_on_small_roster(self, pool: pd.DataFrame, config: LeagueConfig) -> None:
        """Very small roster (< 5 players) should return None."""
        result = _lp_constrained_totals([1, 2], pool, config)
        assert result is None

    def test_lp_totals_le_raw_totals_counting_stats(self, pool: pd.DataFrame, config: LeagueConfig) -> None:
        """LP starters-only totals should be <= raw totals for counting stats.

        Because the LP excludes bench players, their counting stat contributions
        are removed. Bench players have >= 0 counting stats, so LP totals <= raw.
        """
        try:
            import pulp  # noqa: F401
        except ImportError:
            pytest.skip("PuLP not installed")

        roster_ids = pool["player_id"].tolist()
        from src.in_season import _roster_category_totals

        raw_totals = _roster_category_totals(roster_ids, pool)
        lp_totals = _lp_constrained_totals(roster_ids, pool, config)

        if lp_totals is None:
            pytest.skip("LP solver did not return optimal result")

        # Counting stats: LP starters <= raw (all players)
        for cat in ["R", "HR", "RBI", "SB", "W", "SV", "K"]:
            assert lp_totals[cat] <= raw_totals[cat] + 0.01, (
                f"LP {cat}={lp_totals[cat]} should be <= raw {cat}={raw_totals[cat]}"
            )

    def test_handles_name_column_alias(self, config: LeagueConfig) -> None:
        """Should handle pool with 'name' column (no 'player_name')."""
        pool = _make_pool(n_hitters=10, n_pitchers=8)
        pool = pool.drop(columns=["player_name"], errors="ignore")
        roster_ids = pool["player_id"].tolist()
        # Should not crash — the function renames 'name' to 'player_name'
        result = _lp_constrained_totals(roster_ids, pool, config)
        # Result depends on PuLP availability, but should not raise
        assert result is None or isinstance(result, dict)


class TestScan1For1LpFallback:
    """Verify scan_1_for_1 still works when LP fails (fallback to raw)."""

    def test_scan_works_without_lp(self, pool: pd.DataFrame, config: LeagueConfig) -> None:
        """scan_1_for_1 should return results even when LP is unavailable."""
        user_ids = pool["player_id"].tolist()[:9]
        opp_ids = pool["player_id"].tolist()[9:]

        with patch("src.trade_finder._lp_constrained_totals", return_value=None):
            results = scan_1_for_1(user_ids, opp_ids, pool, config)
            # Should still produce a list (may be empty due to filters)
            assert isinstance(results, list)

    def test_scan_works_with_lp(self, pool: pd.DataFrame, config: LeagueConfig) -> None:
        """scan_1_for_1 should work when LP baseline is available."""
        user_ids = pool["player_id"].tolist()[:9]
        opp_ids = pool["player_id"].tolist()[9:]

        results = scan_1_for_1(user_ids, opp_ids, pool, config)
        assert isinstance(results, list)
