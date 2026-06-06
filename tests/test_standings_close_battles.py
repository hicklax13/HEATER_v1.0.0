"""Standings banner cosmetic (2026-06-05): the matchup teaser rendered
"Close battles: , , ." when categories had missing/empty names.

`close_battle_categories()` returns the names of close (small-margin) matchup
categories, SKIPPING any entry with a missing/empty name, so the banner never
shows empty commas. The page's `if close_cats:` guard then reflects real close
battles only.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.standings_utils import close_battle_categories


def test_close_battles_filters_empty_and_missing_names():
    cats = [
        {"name": "", "user_val": 0, "opp_val": 0},  # no name, no data
        {"name": None, "user_val": 5, "opp_val": 5},  # None name (close but unnamed)
        {"name": "   ", "user_val": 1, "opp_val": 1},  # whitespace-only name
        {"name": "AVG", "user_val": 0.300, "opp_val": 0.301},  # real close battle
    ]
    assert close_battle_categories(cats) == ["AVG"]


def test_close_battles_returns_only_close_ones():
    cats = [
        {"name": "HR", "user_val": 10, "opp_val": 2},  # not close
        {"name": "SB", "user_val": 5, "opp_val": 5},  # close (tied)
    ]
    assert close_battle_categories(cats) == ["SB"]


def test_close_battles_empty_inputs():
    assert close_battle_categories([]) == []
    assert close_battle_categories(None) == []


def test_close_battles_tolerates_malformed_rows():
    # Non-numeric values / non-dict rows must not raise.
    cats = [
        {"name": "ERA", "user_val": "n/a", "opp_val": "n/a"},
        {"name": "WHIP"},  # missing values
    ]
    result = close_battle_categories(cats)
    assert isinstance(result, list)
