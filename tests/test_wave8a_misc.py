"""Wave 8a / Group 6 — miscellaneous behavioral fixes.

D5B-037: ip_tracker hardcoded `ip_per_start = ip_season / 30.0` was right
    for full-season SP but wrong for relievers, partial-season SP, and
    SP/RP hybrids. Now uses the player's actual `gs` (games_started)
    when available; falls back to MLB-2024 league-average SP starts
    (~27) when gs is missing.

D5B-014: matchup_context.get_matchup_adjustments returned the
    un-adjusted roster on failure with no signal — callers couldn't
    tell whether adjustments were applied. Now returns
    (roster, applied: bool) AND logs a WARNING on the failure path.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pandas as pd

from src.ip_tracker import compute_weekly_ip_projection
from src.matchup_context import MatchupContextService

# ──────────────────────────────────────────────────────────────────────
# D5B-037: ip_per_start must respect actual games_started
# ──────────────────────────────────────────────────────────────────────


class TestIpPerStartUsesActualGs:
    """When a starter's projection includes a `gs` (games_started) value,
    `ip_per_start` should be derived from it — not from the bare /30.0.
    """

    def test_uses_actual_gs_when_available(self):
        """200 IP across 25 starts -> 8.0 IP/start, NOT 200/30 = 6.67."""
        pitchers = [
            {
                "ip": 200.0,
                "gs": 25,
                "positions": "SP",
                "status": "active",
                "is_starter": True,
            }
        ]
        # 1 SP across 5 days of remaining week -> exactly 1 start expected
        # (days_remaining=5, expected_starts = max(1, 5/5) = 1.0)
        # projected_contribution = ip_per_start * 1.0 = ip_per_start
        result = compute_weekly_ip_projection(pitchers, days_remaining=5)
        # If gs honored: ip_per_start = min(200/25, 7.0) = 7.0 (cap)
        # If bare /30: ip_per_start = min(200/30, 7.0) = 6.67
        # The cap masks the 8.0 case, so use 35 starts for a clearer signal.
        assert result["projected_ip"] > 6.7, (
            f"Expected >6.7 (gs honored), got {result['projected_ip']}; bare /30 fallback gives 6.67"
        )

    def test_uses_actual_gs_below_cap(self):
        """Make a case where gs vs /30 are *both* below the 7.0 cap, so
        we can see a clean difference."""
        pitchers = [
            {
                "ip": 150.0,
                "gs": 24,  # 6.25 IP/start (below 7.0 cap)
                "positions": "SP",
                "status": "active",
                "is_starter": True,
            }
        ]
        # days_remaining=5 -> 1 start expected
        result = compute_weekly_ip_projection(pitchers, days_remaining=5)
        # If gs honored: ip_per_start = 6.25 -> projection ~6.25
        # If bare /30:   ip_per_start = 5.0  -> projection ~5.0
        # Floor of 6.0 cleanly separates the two paths
        assert result["projected_ip"] >= 6.0, (
            f"Expected >=6.0 (gs={24} honored), got {result['projected_ip']}; bare /30 fallback gives ~5.0"
        )

    def test_falls_back_to_league_avg_when_no_gs(self):
        """When gs is missing, fall back to a league-average SP starts
        constant (~27 for MLB-2024 SPs with >=10 GS), NOT the bare 30."""
        pitchers = [
            {
                "ip": 200.0,
                "positions": "SP",
                "status": "active",
                "is_starter": True,
                # no gs
            }
        ]
        # days_remaining=5 -> 1 start expected
        result = compute_weekly_ip_projection(pitchers, days_remaining=5)
        # If fallback honored: ip_per_start = min(200/27, 7.0) = 7.0 (cap)
        # If still /30:        ip_per_start = min(200/30, 7.0) = 6.67
        # Use a partial-season pitcher to dodge the cap.
        assert result["projected_ip"] > 6.7, (
            f"Expected >6.7 (league-avg ~27 fallback), got {result['projected_ip']}; bare /30 fallback gives 6.67"
        )

    def test_fallback_avoids_bare_30(self):
        """A partial-season starter at 130 IP with no gs: the bare /30
        path gives 4.33 IP/start; the ~27 fallback gives ~4.81."""
        pitchers = [
            {
                "ip": 130.0,
                "positions": "SP",
                "status": "active",
                "is_starter": True,
                # no gs
            }
        ]
        # days_remaining=5 -> 1 start expected
        result = compute_weekly_ip_projection(pitchers, days_remaining=5)
        # Fallback (~27): 130/27 = 4.81
        # Bare /30:       130/30 = 4.33
        # Floor of 4.5 cleanly separates the two
        assert result["projected_ip"] >= 4.5, (
            f"Expected >=4.5 (league-avg ~27 fallback), got {result['projected_ip']}; bare /30 fallback gives 4.33"
        )

    def test_gs_zero_falls_back_safely(self):
        """gs=0 (edge case: pitcher mid-rehab) should fall back to the
        league average, not div-by-zero."""
        pitchers = [
            {
                "ip": 100.0,
                "gs": 0,
                "positions": "SP",
                "status": "active",
                "is_starter": True,
            }
        ]
        # Just verify no crash and returns finite IP
        result = compute_weekly_ip_projection(pitchers, days_remaining=7)
        assert result["projected_ip"] >= 0
        assert result["projected_ip"] < 50.0  # sanity bound

    def test_rp_path_unchanged(self):
        """The RP path (ip/60.0) is correct for relievers and must not
        be touched by the gs change."""
        pitchers = [
            {
                "ip": 60.0,  # ~60 appearances over a season
                "positions": "RP",
                "status": "active",
                "is_starter": False,
            }
        ]
        result = compute_weekly_ip_projection(pitchers, days_remaining=7)
        # ip_per_app = min(60/60, 2.0) = 1.0
        # expected_appearances = 7 * 0.55 = 3.85
        # projected_contribution ~ 3.85
        assert 3.5 <= result["projected_ip"] <= 4.2, f"Expected ~3.85 for RP projection, got {result['projected_ip']}"


# ──────────────────────────────────────────────────────────────────────
# D5B-014: get_matchup_adjustments must signal success/failure
# ──────────────────────────────────────────────────────────────────────


class TestMatchupAdjustmentsSignalsFailure:
    """When matchup adjustment fails, the return value must signal it
    (tuple of (df, applied: bool)) AND emit a WARNING-level log."""

    def test_returns_tuple_with_applied_flag(self):
        """Return signature must be (DataFrame, bool)."""
        svc = MatchupContextService()
        roster = pd.DataFrame({"player_id": [1, 2], "name": ["A", "B"]})
        result = svc.get_matchup_adjustments(roster)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result).__name__}"
        assert len(result) == 2, f"Expected 2-tuple, got {len(result)}-tuple"
        df, applied = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(applied, bool)

    def test_failure_sets_applied_false(self):
        """When the underlying call raises, applied must be False."""
        svc = MatchupContextService()
        roster = pd.DataFrame({"player_id": [1, 2], "name": ["A", "B"]})
        # Force the underlying call to raise
        with patch(
            "src.optimizer.matchup_adjustments.compute_weekly_matchup_adjustments",
            side_effect=RuntimeError("boom"),
        ):
            df, applied = svc.get_matchup_adjustments(roster)
        assert applied is False, "Failure path must set applied=False"
        # Roster should be the original (unadjusted) input
        assert len(df) == 2

    def test_failure_logs_warning(self, caplog):
        """The failure path must log at WARNING (not just DEBUG) so the
        silent-failure is visible to operators."""
        svc = MatchupContextService()
        roster = pd.DataFrame({"player_id": [1, 2], "name": ["A", "B"]})
        with patch(
            "src.optimizer.matchup_adjustments.compute_weekly_matchup_adjustments",
            side_effect=RuntimeError("boom"),
        ):
            with caplog.at_level(logging.WARNING, logger="src.matchup_context"):
                svc.get_matchup_adjustments(roster)
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and r.name == "src.matchup_context"]
        assert warnings, "Expected at least one WARNING-level log on adjustment failure"
        assert any("matchup" in r.message.lower() for r in warnings), "Expected the warning to mention 'matchup'"

    def test_success_sets_applied_true(self):
        """When the underlying call succeeds, applied must be True."""
        svc = MatchupContextService()
        roster = pd.DataFrame({"player_id": [1, 2], "name": ["A", "B"]})
        # Mock a successful adjustment that returns a new DataFrame
        mock_adjusted = pd.DataFrame(
            {
                "player_id": [1, 2],
                "name": ["A", "B"],
                "matchup_mult": [1.05, 0.95],
            }
        )
        with patch(
            "src.optimizer.matchup_adjustments.compute_weekly_matchup_adjustments",
            return_value=mock_adjusted,
        ):
            df, applied = svc.get_matchup_adjustments(roster)
        assert applied is True, "Success path must set applied=True"
        assert "matchup_mult" in df.columns
