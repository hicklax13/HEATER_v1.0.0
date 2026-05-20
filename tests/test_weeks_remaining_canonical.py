"""Section 5: weeks_remaining canonical helper + 22→26 default fix.

The bonus fix: src/validation/dynamic_context.py:32 had
FANTASY_REGULAR_SEASON_WEEKS = 22 (a generic Yahoo H2H default) — but
FourzynBurn uses 26 weeks. Every caller of compute_weeks_remaining()
that didn't pass an explicit total_weeks silently inherited the wrong
horizon. PR #33's L11 fix did NOT catch this site.
"""

from __future__ import annotations

from datetime import date

import pytest


def test_dynamic_context_default_is_26_not_22():
    """FourzynBurn is a 26-week H2H league; the old 22-week default was wrong."""
    from src.validation.dynamic_context import FANTASY_REGULAR_SEASON_WEEKS

    assert FANTASY_REGULAR_SEASON_WEEKS == 26, (
        "FANTASY_REGULAR_SEASON_WEEKS must be 26 (FourzynBurn) — see CLAUDE.md "
        "'Roster' section + LeagueConfig.season_weeks."
    )


def test_league_rules_weeks_remaining_uses_league_config_season_weeks():
    """The canonical wrapper sources total_weeks from LeagueConfig (26)."""
    from src.league_rules import weeks_remaining
    from src.valuation import LeagueConfig

    # Pre-season → returns full season
    result = weeks_remaining(as_of=date(2026, 2, 1))
    assert result == LeagueConfig().season_weeks


def test_weeks_remaining_postseason_returns_1():
    from src.league_rules import weeks_remaining

    assert weeks_remaining(as_of=date(2026, 11, 1)) == 1


def test_weeks_remaining_mid_season_decreasing():
    from src.league_rules import weeks_remaining

    early = weeks_remaining(as_of=date(2026, 4, 15))
    late = weeks_remaining(as_of=date(2026, 8, 15))
    assert early > late
    assert 1 <= late <= 26
    assert 1 <= early <= 26


@pytest.mark.parametrize(
    "as_of,expected_range",
    [
        # Pre-season: 26 weeks
        (date(2026, 1, 1), (26, 26)),
        # Just before season start (2026-03-26): still 26
        (date(2026, 3, 25), (26, 26)),
        # Mid-season: 1-25
        (date(2026, 6, 15), (1, 25)),
        # End of season (2026-09-27): 1
        (date(2026, 9, 27), (1, 1)),
        # Post-season: 1
        (date(2026, 12, 31), (1, 1)),
    ],
)
def test_weeks_remaining_boundaries(as_of, expected_range):
    from src.league_rules import weeks_remaining

    result = weeks_remaining(as_of=as_of)
    lo, hi = expected_range
    assert lo <= result <= hi, f"as_of={as_of}: got {result}, expected in [{lo}, {hi}]"
