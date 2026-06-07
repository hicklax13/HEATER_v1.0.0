"""MS-C4 fix: momentum adjustment respects inverse-category sign.

simulate_season_enhanced applied `adjustment = 1.0 + 0.05*(mom-1.0)` uniformly
to every category mean (`val *= adjustment`). For a hot team (mom > 1) that
RAISES the mean — correct for counting cats (R, HR...), but WRONG for inverse
cats (ERA, WHIP, L) where lower is better: a hot team would get a worse
projected ERA/WHIP/L.

The fix inverts the adjustment for cats in the LeagueConfig inverse set so a
hot team's inverse-cat mean DECREASES (improves). The inverse set is derived
from LeagueConfig (no hardcoded category list — structural guard
test_no_hardcoded_categories_in_src applies).
"""

from __future__ import annotations

from src.standings_engine import INVERSE_CATS, _apply_momentum_to_mean
from src.valuation import LeagueConfig


def test_inverse_set_matches_league_config():
    """The module's inverse set must be the canonical LeagueConfig one."""
    assert INVERSE_CATS == set(LeagueConfig().inverse_stats)
    # Sanity: the canonical inverse cats are present.
    assert {"ERA", "WHIP", "L"} <= INVERSE_CATS


def test_hot_team_counting_cat_mean_increases():
    """mom > 1 on a counting cat (R) raises the mean (more is better)."""
    base = 100.0
    out = _apply_momentum_to_mean(base, mom=2.0, is_inverse=False)
    assert out > base


def test_hot_team_inverse_cat_mean_decreases():
    """mom > 1 on an inverse cat (ERA) LOWERS the mean (less is better)."""
    base = 4.00
    out = _apply_momentum_to_mean(base, mom=2.0, is_inverse=True)
    assert out < base, f"MS-C4: a hot team's ERA mean should improve (decrease); got {out} >= {base}"


def test_cold_team_inverse_cat_mean_increases():
    """mom < 1 on an inverse cat (WHIP) RAISES the mean (gets worse)."""
    base = 1.25
    out = _apply_momentum_to_mean(base, mom=0.0, is_inverse=True)
    assert out > base


def test_cold_team_counting_cat_mean_decreases():
    """mom < 1 on a counting cat (HR) lowers the mean (gets worse)."""
    base = 30.0
    out = _apply_momentum_to_mean(base, mom=0.0, is_inverse=False)
    assert out < base


def test_neutral_momentum_is_identity_both_directions():
    """mom == 1.0 leaves the mean unchanged regardless of direction."""
    assert _apply_momentum_to_mean(50.0, mom=1.0, is_inverse=False) == 50.0
    assert _apply_momentum_to_mean(3.5, mom=1.0, is_inverse=True) == 3.5


def test_adjustment_magnitude_is_symmetric_5pct():
    """The +/-5% magnitude is preserved; only the sign flips for inverse."""
    base = 100.0
    up = _apply_momentum_to_mean(base, mom=2.0, is_inverse=False)  # +5%
    # Inverse cat with same momentum should move the same magnitude DOWN.
    inv = _apply_momentum_to_mean(base, mom=2.0, is_inverse=True)  # -5%
    assert abs((up - base) + (inv - base)) < 1e-9  # equal & opposite
