"""Unit test: Lineup Optimizer weight path uses emoji-reconciled team name.

When the DB stores team names with emoji prefixes (e.g. "🏆 Team Hickey") but
the env seeds a bare name ("Team Hickey"), compute_nonlinear_weights() must
receive the EXACT emoji-prefixed name so the standings lookup hits — otherwise
all 12 category weights collapse to 1.00× (equal-weight fallback).

This proves that pages/2_Line-up_Optimizer.py routes through
resolve_viewer_team_name(rosters), which reconciles the bare env name against
the actual roster names via _normalize_team_name().
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_standings(team_name: str) -> pd.DataFrame:
    """Minimal standings DataFrame with one team varying across categories.

    The team_name gets rank 6 in every category (middle of 12) so the
    marginal SGP weights are meaningfully non-1.0.
    """
    cats = ["r", "hr", "rbi", "sb", "avg", "obp", "w", "l", "sv", "k", "era", "whip"]
    rows = []
    for rank, cat in enumerate(cats, start=1):
        # Add 12 teams; user team sits at rank 6 for every category.
        for i in range(1, 13):
            rows.append(
                {
                    "team_name": team_name if i == 6 else f"Other Team {i}",
                    "category": cat,
                    "total": float(100 - i),
                    "rank": i,
                }
            )
    return pd.DataFrame(rows)


def _make_rosters(team_name: str) -> pd.DataFrame:
    """Minimal rosters DataFrame with the user's team flagged via is_user_team."""
    return pd.DataFrame(
        [
            {"team_name": team_name, "player_id": 1, "is_user_team": 1},
            {"team_name": "Other Team 1", "player_id": 2, "is_user_team": 0},
        ]
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEmojiTeamNameWeights:
    """The weight calculation must use the emoji-reconciled team name."""

    def test_emoji_name_in_standings_gives_non_equal_weights(self):
        """compute_nonlinear_weights() with the emoji name finds the team and
        returns weights that are NOT all 1.0.
        """
        try:
            from src.optimizer.sgp_theory import compute_nonlinear_weights
        except ImportError:
            pytest.skip("sgp_theory not available")

        emoji_name = "🏆 Team Hickey"
        standings = _make_standings(emoji_name)
        weights = compute_nonlinear_weights(standings, emoji_name)

        # At least one weight must differ from 1.0 to confirm the team was found
        assert any(abs(w - 1.0) > 0.01 for w in weights.values()), (
            "All weights are 1.0 — team not found in standings. "
            f"standings team_names={standings['team_name'].unique().tolist()!r}, "
            f"looked up={emoji_name!r}"
        )

    def test_bare_name_not_in_emoji_standings_gives_equal_weights(self):
        """Confirms the bug scenario: bare name misses the emoji standings row
        and returns all 1.0 weights (the equal-weight fallback).
        """
        try:
            from src.optimizer.sgp_theory import compute_nonlinear_weights
        except ImportError:
            pytest.skip("sgp_theory not available")

        emoji_name = "🏆 Team Hickey"
        bare_name = "Team Hickey"
        standings = _make_standings(emoji_name)
        weights = compute_nonlinear_weights(standings, bare_name)

        # The bare name doesn't match the emoji-prefixed standings row → all 1.0
        assert all(abs(w - 1.0) < 1e-9 for w in weights.values()), (
            "Expected equal weights (1.0) when bare name misses emoji standings, "
            f"but got weights with variance: {weights}"
        )

    def test_resolve_viewer_team_name_returns_emoji_name_from_rosters(self):
        """resolve_viewer_team_name(rosters) returns the emoji-prefixed name
        that exists in the rosters frame, not the bare env-seeded name.

        This is the contract that pages/2_Line-up_Optimizer.py relies on.
        """
        try:
            from src.auth import resolve_viewer_team_name
        except ImportError:
            pytest.skip("auth module not available")

        emoji_name = "🏆 Team Hickey"
        rosters = _make_rosters(emoji_name)

        with patch("src.auth.multi_user_enabled", return_value=False):
            resolved = resolve_viewer_team_name(rosters)

        assert resolved == emoji_name, (
            f"Expected emoji name {emoji_name!r} but got {resolved!r}. "
            "The resolver must return the ACTUAL roster name (with emoji prefix) "
            "so downstream lookups against standings/rosters hit correctly."
        )

    def test_resolve_then_compute_weights_pipeline(self):
        """End-to-end: resolve_viewer_team_name -> compute_nonlinear_weights
        produces non-equal weights when the DB name has an emoji prefix.

        This is the exact pipeline in pages/2_Line-up_Optimizer.py lines 220-225
        + 3012.
        """
        try:
            from src.auth import resolve_viewer_team_name
            from src.optimizer.sgp_theory import compute_nonlinear_weights
        except ImportError:
            pytest.skip("auth or sgp_theory not available")

        emoji_name = "🏆 Team Hickey"
        rosters = _make_rosters(emoji_name)
        standings = _make_standings(emoji_name)

        with patch("src.auth.multi_user_enabled", return_value=False):
            team_name = resolve_viewer_team_name(rosters)

        assert team_name == emoji_name

        weights = compute_nonlinear_weights(standings, team_name)

        assert any(abs(w - 1.0) > 0.01 for w in weights.values()), (
            "Pipeline resolve→compute_nonlinear_weights returned all-1.0 weights — "
            "the emoji name reconciliation is not flowing through correctly."
        )
