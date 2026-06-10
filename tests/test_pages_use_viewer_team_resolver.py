"""Structural guard: personalized pages resolve the viewer's team via the
canonical resolve_viewer_team_name(), never the legacy is_user_team flag.

This is the guard that was MISSING when the 2026-06-01 launch-blocker shipped:
the auth-aware resolver existed and was unit-tested, but no test checked that
the pages actually used it — so every page kept filtering is_user_team == 1
(pinned to Team Hickey) and every leaguemate saw the wrong team.
"""

import ast
from pathlib import Path

import pytest

# The 9 pages that render a personalized "my team" view.
_PERSONALIZED_PAGES = [
    "1_My_Team.py",
    "2_Line-up_Optimizer.py",
    "4_Pitcher_Streaming.py",
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


@pytest.mark.parametrize("page", _PERSONALIZED_PAGES)
def test_page_passes_frame_to_viewer_team_resolver(page):
    """resolve_viewer_team_name() must be called WITH a frame so its emoji/whitespace
    reconciliation runs.

    A no-arg call returns the raw env-seeded assignment (e.g. "Team Hickey"), which
    won't exact-match a Yahoo team name carrying a leading emoji/whitespace. That was
    the 2026-06-05 "Your Position: Team not found in standings" bug on League Standings
    -- the one personalized page that called the resolver with no frame, so every
    downstream `== user_team` comparison missed the viewer's own (emoji-prefixed) row.
    """
    src = (_PAGES_DIR / page).read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_viewer_team_name"
        and not node.args
        and not node.keywords
    ]
    assert not bad_lines, (
        f"{page}: resolve_viewer_team_name() called with no frame at line(s) "
        f"{bad_lines}. Pass the rosters/records frame so emoji/whitespace team names "
        "reconcile (env-seeded name must match the Yahoo name with its emoji)."
    )
