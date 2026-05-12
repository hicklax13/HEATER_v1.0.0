"""BUG-020 fix: closer_monitor matches names robustly via normalization."""

import pandas as pd


def test_closer_with_accented_name_matched():
    """A closer in depth_data with no accent should still match the player_pool
    row that has accents (or vice versa)."""
    from src.closer_monitor import build_closer_grid

    depth_data = {
        "NYM": {"closer": "Edwin Diaz", "setup": [], "closer_confidence": 0.95},
    }
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Edwin Díaz",
                "team": "NYM",
                "sv": 30,
                "era": 1.50,
                "whip": 0.90,
                "mlb_id": 621242,
            },
        ]
    )
    grid = build_closer_grid(depth_data, player_pool=pool)
    assert len(grid) == 1, f"Expected 1 team in grid; got {grid}"
    row = grid[0]
    assert row["closer_name"] == "Edwin Diaz"
    assert row["projected_sv"] == 30, (
        f"BUG-020: projected_sv = {row['projected_sv']} indicates name match "
        f"failed for 'Edwin Diaz' (depth_data) vs 'Edwin Díaz' (pool, accented)"
    )
    assert abs(row["era"] - 1.50) < 0.01
    assert abs(row["whip"] - 0.90) < 0.01


def test_closer_with_suffix_matched():
    """Suffix mismatch should not break the match."""
    from src.closer_monitor import build_closer_grid

    depth_data = {
        "TBR": {"closer": "Pete Fairbanks", "setup": [], "closer_confidence": 0.85},
    }
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Pete Fairbanks Jr.",
                "team": "TBR",
                "sv": 25,
                "era": 2.00,
                "whip": 1.00,
                "mlb_id": 668941,
            },
        ]
    )
    grid = build_closer_grid(depth_data, player_pool=pool)
    row = grid[0]
    assert row["projected_sv"] == 25, (
        f"BUG-020: projected_sv = {row['projected_sv']} indicates suffix variation broke the match"
    )


def test_closer_exact_match_still_works():
    """Sanity: a clean exact match still works (no regression)."""
    from src.closer_monitor import build_closer_grid

    depth_data = {
        "BAL": {"closer": "Felix Bautista", "setup": [], "closer_confidence": 0.90},
    }
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Felix Bautista",
                "team": "BAL",
                "sv": 28,
                "era": 1.85,
                "whip": 0.95,
                "mlb_id": 669456,
            },
        ]
    )
    grid = build_closer_grid(depth_data, player_pool=pool)
    row = grid[0]
    assert row["projected_sv"] == 28
