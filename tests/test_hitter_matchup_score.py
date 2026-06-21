"""Real-shape tests for compute_hitter_matchup_score — the calibrated inverse of the
pitcher matchup scorer. Calls the REAL compute_pitcher_matchup_score (no mocks)."""

from src.optimizer.stream_analyzer import compute_hitter_matchup_score
from src.two_start import compute_pitcher_matchup_score

# k_bb_pct, xfip, csw_pct: an ace vs a batting-practice arm.
_TOUGH_SP = {"k_bb_pct": 0.25, "xfip": 2.80, "csw_pct": 0.33}
_WEAK_SP = {"k_bb_pct": 0.03, "xfip": 5.20, "csw_pct": 0.24}
# team_offense: wrc_plus + k_pct in PERCENT (matches get_opponent_offense_context output).
_STRONG_OFF = {"wrc_plus": 120.0, "k_pct": 18.0}
_WEAK_OFF = {"wrc_plus": 82.0, "k_pct": 27.0}


def _inverse(sp, wrc, k_pct_frac, pf, sp_is_home):
    p = compute_pitcher_matchup_score(
        sp,
        opponent_team_stats={"wrc_plus": wrc, "k_pct": k_pct_frac},
        park_factor=pf,
        is_home=sp_is_home,
    )
    return round(max(0.0, min(100.0, (10.0 - p) * 10.0)), 1)


def test_weak_pitcher_is_a_better_hitting_matchup_than_an_ace():
    tough = compute_hitter_matchup_score(_TOUGH_SP, _STRONG_OFF, park_factor=1.0, hitters_home=True)
    weak = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, park_factor=1.0, hitters_home=True)
    assert 0.0 <= tough <= 100.0 and 0.0 <= weak <= 100.0
    assert weak > tough


def test_strong_offense_scores_higher_than_weak_vs_same_pitcher():
    strong = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, hitters_home=True)
    weak = compute_hitter_matchup_score(_WEAK_SP, _WEAK_OFF, hitters_home=True)
    assert strong > weak


def test_hitter_friendly_park_raises_difficulty():
    coors = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, park_factor=1.30, hitters_home=True)
    pitcher_park = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, park_factor=0.92, hitters_home=True)
    assert coors > pitcher_park


def test_home_bats_beat_away_bats_same_inputs():
    home = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, hitters_home=True)
    away = compute_hitter_matchup_score(_WEAK_SP, _STRONG_OFF, hitters_home=False)
    assert home > away


def test_percent_to_fraction_boundary():
    # team_offense.k_pct is PERCENT (22.0); the pitcher scorer expects a FRACTION (0.22).
    got = compute_hitter_matchup_score(_WEAK_SP, {"wrc_plus": 100.0, "k_pct": 22.0}, park_factor=1.0, hitters_home=True)
    # hitters_home=True -> the SP is AWAY -> is_home=False
    assert got == _inverse(_WEAK_SP, 100.0, 0.22, 1.0, False)


def test_exact_complement_of_board_matchup():
    sp = {"k_bb_pct": 0.15, "xfip": 3.6, "csw_pct": 0.30}
    off = {"wrc_plus": 105.0, "k_pct": 21.0}
    got = compute_hitter_matchup_score(sp, off, park_factor=1.05, hitters_home=True)
    assert got == _inverse(sp, 105.0, 0.21, 1.05, False)


def test_nan_and_missing_inputs_never_raise_and_stay_in_range():
    nan = float("nan")
    a = compute_hitter_matchup_score({"k_bb_pct": nan, "xfip": nan, "csw_pct": nan}, {"wrc_plus": nan, "k_pct": nan})
    b = compute_hitter_matchup_score({}, {})
    assert isinstance(a, float) and 0.0 <= a <= 100.0
    assert isinstance(b, float) and 0.0 <= b <= 100.0
    # all-missing -> engine league defaults -> the two are identical
    assert a == b
