"""MS-C5: the standings-page category-rank grid excludes ghost teams.

A team present in the standings cache but absent from current rosters must not
appear in (or distort) the per-category rank grid — the standings-page analogue
of the Bug-D engine ghost filter (`test_no_ghost_team_rank` covers the engine path).
"""

from __future__ import annotations

import pandas as pd

from src.standings_utils import filter_standings_to_valid_teams


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team_name": "Real A", "category": "R", "total": 50},
            {"team_name": "Real B", "category": "R", "total": 40},
            {"team_name": "Ghost Z", "category": "R", "total": 99},
        ]
    )


def test_ghost_team_excluded():
    out = filter_standings_to_valid_teams(_frame(), {"Real A", "Real B"})
    assert set(out["team_name"]) == {"Real A", "Real B"}
    assert "Ghost Z" not in set(out["team_name"])


def test_none_valid_teams_passes_through():
    f = _frame()
    out = filter_standings_to_valid_teams(f, None)
    assert len(out) == len(f)


def test_empty_frame_is_safe():
    assert filter_standings_to_valid_teams(pd.DataFrame(), {"Real A"}).empty
