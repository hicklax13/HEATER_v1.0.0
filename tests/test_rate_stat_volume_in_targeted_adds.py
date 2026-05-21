"""PR10 Part B: compute_matchup_targeted_adds must use volume-weighted
rate-stat aggregation (sum(h)/sum(ab) for AVG, etc.), not direct addition."""

import pandas as pd
import pytest


def test_targeted_adds_uses_volume_weighted_rates():
    """Check that the function doesn't naively add AVG/OBP/ERA/WHIP across
    multiple players (which is mathematically wrong)."""
    # Read the function source to verify it doesn't have the naive bug
    import inspect

    from src import waiver_wire

    if not hasattr(waiver_wire, "compute_matchup_targeted_adds"):
        pytest.skip("compute_matchup_targeted_adds not in module")

    src = inspect.getsource(waiver_wire.compute_matchup_targeted_adds)
    # The buggy pattern: `team_totals["AVG"] += ` directly accumulating rate
    # The fix: accumulate h/ab/bb/etc separately then compute rate at the end
    naive_patterns = [
        'team_totals["AVG"] +=',
        'team_totals["OBP"] +=',
        'team_totals["ERA"] +=',
        'team_totals["WHIP"] +=',
    ]
    found_naive = [p for p in naive_patterns if p in src]
    assert not found_naive, (
        f"compute_matchup_targeted_adds still has naive rate-stat addition: {found_naive}. "
        f"Rate stats must use volume-weighted aggregation."
    )
