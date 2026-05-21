"""PR17 follow-on: _roster_category_totals must read expected_return_days
from each row and pass to _il_weight_from_status."""

import pandas as pd

from src.in_season import _roster_category_totals


def test_roster_totals_uses_return_date_column():
    """A 'Suspended' player WITH expected_return_days=1 contributes
    ~95% of his projection (not 0%, the pre-PR17 behavior)."""
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Valdez",
                "is_hitter": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "ab": 0,
                "h": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "w": 10,
                "l": 5,
                "sv": 0,
                "k": 150,
                "ip": 100,
                "er": 30,
                "bb_allowed": 25,
                "h_allowed": 80,
                "status": "Suspended",
                "expected_return_days": 1.0,
            },
        ]
    )
    totals = _roster_category_totals([1], pool)
    # 1-day suspension -> ~0.95 weight -> K = 150 * 0.95 = 142.5
    assert totals["K"] > 130, (
        f"1-day suspension should still contribute ~95% of K projection; "
        f"got K={totals['K']:.1f}. Pre-PR17 behavior would be K=0."
    )


def test_roster_totals_no_return_date_column_unchanged():
    """When the column is absent, behavior is identical to pre-PR17 (string-
    based weighting only)."""
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Active SP",
                "is_hitter": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "ab": 0,
                "h": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "w": 10,
                "l": 5,
                "sv": 0,
                "k": 150,
                "ip": 100,
                "er": 30,
                "bb_allowed": 25,
                "h_allowed": 80,
                "status": "active",
            },
        ]
    )
    totals = _roster_category_totals([1], pool)
    # Active player full weight
    assert totals["K"] == 150


def test_roster_totals_indefinite_suspension_zeroed():
    """A 'Suspended' player WITHOUT expected_return_days contributes 0%."""
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Indef Suspended",
                "is_hitter": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "ab": 0,
                "h": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "w": 10,
                "l": 5,
                "sv": 0,
                "k": 150,
                "ip": 100,
                "er": 30,
                "bb_allowed": 25,
                "h_allowed": 80,
                "status": "Suspended",  # no expected_return_days column
            },
        ]
    )
    totals = _roster_category_totals([1], pool)
    assert totals["K"] == 0  # indefinite -> 0%
