"""Structural guard: personalized pages resolve the viewer's team via the
canonical resolve_viewer_team_name(), never the legacy is_user_team flag.

This is the guard that was MISSING when the 2026-06-01 launch-blocker shipped:
the auth-aware resolver existed and was unit-tested, but no test checked that
the pages actually used it — so every page kept filtering is_user_team == 1
(pinned to Team Hickey) and every leaguemate saw the wrong team.
"""

from pathlib import Path

import pytest

# The 8 pages that render a personalized "my team" view.
_PERSONALIZED_PAGES = [
    "1_My_Team.py",
    "2_Line-up_Optimizer.py",
    "5_Matchup_Planner.py",
    "6_League_Standings.py",
    "10_Punt_Analyzer.py",
    "11_Trade_Analyzer.py",
    "12_Trade_Finder.py",
    "14_Free_Agents.py",
]

_PAGES_DIR = Path(__file__).resolve().parent.parent / "pages"


@pytest.mark.parametrize("page", _PERSONALIZED_PAGES)
def test_page_uses_viewer_team_resolver(page):
    src = (_PAGES_DIR / page).read_text(encoding="utf-8")
    assert "resolve_viewer_team_name" in src, (
        f"{page} must resolve the viewer's team via resolve_viewer_team_name() "
        "so each logged-in user sees their own assigned team"
    )


@pytest.mark.parametrize("page", _PERSONALIZED_PAGES)
def test_page_has_no_legacy_is_user_team_reference(page):
    src = (_PAGES_DIR / page).read_text(encoding="utf-8")
    assert "is_user_team" not in src, (
        f"{page} must NOT read the legacy is_user_team flag directly — that flag is "
        "pinned to one team, so under MULTI_USER every user would see that team. "
        "Use resolve_viewer_team_name() instead."
    )
