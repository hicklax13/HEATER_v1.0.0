"""2026-06-16 owner finding: early-week matchup undercount.

When a team has 0 IP (no pitcher has thrown yet this week), Yahoo returns "-"
for ERA/WHIP (undefined ratio). Yahoo scores that as a LOSS for the 0-IP team
(the opponent has a real ratio and wins) — a TIE only if BOTH teams have 0 IP.
HEATER's matchup loop skipped any category where one side was "-", dropping
those losses: the owner's Week-13 board read 5-3-2 ("Winning") when Yahoo had
it 5-5 (tied). `_score_matchup_category` must score "-" the way Yahoo does.
"""

from __future__ import annotations

from src.valuation import LeagueConfig
from src.yahoo_api import _score_matchup_category

_INV = set(LeagueConfig().inverse_stats)  # {"L", "ERA", "WHIP"}


def test_undefined_rate_stat_is_a_loss_not_a_skip():
    # 0 IP -> ERA/WHIP "-"; opponent has a real ratio -> the 0-IP team LOSES.
    assert _score_matchup_category("ERA", "-", "5.40", _INV) == "LOSS"
    assert _score_matchup_category("WHIP", "-", "2.00", _INV) == "LOSS"
    assert _score_matchup_category("AVG", "-", ".147", _INV) == "LOSS"


def test_having_rate_stat_beats_opponent_without():
    assert _score_matchup_category("ERA", "3.00", "-", _INV) == "WIN"
    assert _score_matchup_category("AVG", ".250", "-", _INV) == "WIN"


def test_both_undefined_not_counted():
    # Pre-event / both teams 0 IP: undecided, not counted (preserves the prior
    # 0-0-0 pre-event tally; the headline W-L is unaffected either way).
    assert _score_matchup_category("ERA", "-", "-", _INV) == "-"
    assert _score_matchup_category("WHIP", "", "", _INV) == "-"


def test_both_defined_inverse_and_normal():
    assert _score_matchup_category("L", "0", "1", _INV) == "WIN"  # fewer losses wins (inverse)
    assert _score_matchup_category("ERA", "5.40", "3.00", _INV) == "LOSS"  # higher ERA loses
    assert _score_matchup_category("R", "7", "4", _INV) == "WIN"
    assert _score_matchup_category("W", "0", "0", _INV) == "TIE"


def test_full_week13_board_matches_yahoo_5_5_2():
    """The owner's live Week-13 board. Yahoo shows 5-5 (2 ties). HEATER showed
    5-3-2 by skipping ERA/WHIP where the owner had 0 IP."""
    rows = [
        ("R", "7", "4"),
        ("HR", "1", "0"),
        ("RBI", "1", "2"),
        ("SB", "0", "1"),
        ("AVG", ".290", ".147"),
        ("OBP", ".371", ".256"),
        ("W", "0", "0"),
        ("L", "0", "1"),
        ("SV", "0", "0"),
        ("K", "0", "2"),
        ("ERA", "-", "5.40"),
        ("WHIP", "-", "2.00"),
    ]
    w = l = t = 0
    for cat, you, opp in rows:
        r = _score_matchup_category(cat, you, opp, _INV)
        w += r == "WIN"
        l += r == "LOSS"
        t += r == "TIE"
    assert (w, l, t) == (5, 5, 2)
