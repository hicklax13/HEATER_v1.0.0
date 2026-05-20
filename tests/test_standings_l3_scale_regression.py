"""Regression lock for L3 (standings_engine divisor → config.season_weeks).

PR #47 changed `_estimate_team_weekly_stats` to divide by `config.season_weeks` (26)
instead of `weeks_remaining`. Silent-failure-hunter (run 2026-05-19 after merge)
flagged this as H2 — concerned that tau (variance) would be miscalibrated relative
to the new mean scale, compressing all win probabilities toward 0.5.

Empirical check: with the L3 fix, a clearly-elite team (910 R/season → 35 R/wk)
vs a weak team (650 R/season → 25 R/wk) gives:

    USER weekly R = 35.08, OPP weekly R = 24.92
    P(USER wins R) = 0.990  ← DECISIVE, not compressed
    Overall win pct = 0.9991

H2 was a false positive. This test locks the empirical p_win values to detect
future drift if the divisor or tau scale ever changes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.standings_engine import _estimate_team_weekly_stats, compute_category_win_probabilities
from src.valuation import LeagueConfig


@pytest.fixture
def synthetic_unequal_rosters():
    """16 hitters + 16 pitchers, split into clearly-unequal teams.

    USER team: elite — 910 R, 200 HR, 640 RBI, 96 SB seasonal
    OPP team:  weak — 650 R, 120 HR, 480 RBI, 64 SB seasonal
    """
    rows = []
    for pid in range(1, 9):
        rows.append(
            {
                "player_id": pid,
                "name": f"UH{pid}",
                "is_hitter": 1,
                "r": 114,
                "hr": 25,
                "rbi": 80,
                "sb": 12,
                "ab": 550,
                "h": 150,
                "bb": 50,
                "hbp": 5,
                "sf": 5,
                "pa": 610,
                "avg": 0.273,
                "obp": 0.336,
            }
        )
    for pid in range(9, 17):
        rows.append(
            {
                "player_id": pid,
                "name": f"UP{pid}",
                "is_hitter": 0,
                "w": 12,
                "l": 6,
                "sv": 0,
                "k": 180,
                "ip": 180,
                "er": 70,
                "bb_allowed": 55,
                "h_allowed": 165,
                "era": 3.50,
                "whip": 1.22,
            }
        )
    for pid in range(17, 25):
        rows.append(
            {
                "player_id": pid,
                "name": f"OH{pid}",
                "is_hitter": 1,
                "r": 81,
                "hr": 15,
                "rbi": 60,
                "sb": 8,
                "ab": 530,
                "h": 130,
                "bb": 40,
                "hbp": 3,
                "sf": 4,
                "pa": 577,
                "avg": 0.245,
                "obp": 0.305,
            }
        )
    for pid in range(25, 33):
        rows.append(
            {
                "player_id": pid,
                "name": f"OP{pid}",
                "is_hitter": 0,
                "w": 8,
                "l": 10,
                "sv": 0,
                "k": 140,
                "ip": 170,
                "er": 90,
                "bb_allowed": 70,
                "h_allowed": 185,
                "era": 4.76,
                "whip": 1.50,
            }
        )
    return pd.DataFrame(rows)


def test_l3_weekly_mean_uses_season_weeks(synthetic_unequal_rosters):
    """USER 910 R / 26 = 35 R/wk; OPP 650 R / 26 = 25 R/wk."""
    config = LeagueConfig()
    pool = synthetic_unequal_rosters
    user_ids = list(range(1, 17))
    opp_ids = list(range(17, 33))

    user_stats = _estimate_team_weekly_stats(user_ids, pool, config, weeks_remaining=16)
    opp_stats = _estimate_team_weekly_stats(opp_ids, pool, config, weeks_remaining=16)

    # Counting-stat weekly means (910/26 ≈ 35, 650/26 ≈ 25).
    assert user_stats["R"] == pytest.approx(35.0, abs=0.5), (
        f"L3 regression: USER R weekly mean should be ~35 (910/26), got {user_stats['R']:.2f}. "
        f"If this drifted, check src/standings_engine.py:188 divisor."
    )
    assert opp_stats["R"] == pytest.approx(25.0, abs=0.5), (
        f"L3 regression: OPP R weekly mean should be ~25 (650/26), got {opp_stats['R']:.2f}."
    )
    # weeks_remaining=16 must NOT affect the divisor (L3 design decision).
    user_stats_alt = _estimate_team_weekly_stats(user_ids, pool, config, weeks_remaining=8)
    assert user_stats_alt["R"] == pytest.approx(user_stats["R"], abs=0.01), (
        "L3 regression: weeks_remaining parameter must NOT influence weekly mean "
        "(divisor is config.season_weeks). Check src/standings_engine.py:188."
    )


def test_l3_win_prob_decisive_for_unequal_rosters(synthetic_unequal_rosters):
    """H2 lock: tau/sigma_diff stays calibrated; decisive p_win for clear winners.

    Silent-failure-hunter H2 was concerned that the L3 divisor change would
    compress p_win values toward 0.5. Empirically it doesn't — this test
    locks the decisive-win behavior.
    """
    config = LeagueConfig()
    pool = synthetic_unequal_rosters
    user_ids = list(range(1, 17))
    opp_ids = list(range(17, 33))

    result = compute_category_win_probabilities(
        user_ids,
        opp_ids,
        pool,
        config,
        weeks_played=10,
        weeks_remaining=16,
    )

    # Decisive overall win — should be very high given the lopsided rosters.
    assert result["overall_win_pct"] > 0.95, (
        f"H2 regression: overall_win_pct should be >0.95 for clearly-elite vs weak rosters; "
        f"got {result['overall_win_pct']:.3f}. If compressed near 0.5, the L3 divisor + "
        f"CALIBRATED_WEEKLY_TAU are out of scale. See PR #47 H2 finding."
    )

    cat_probs = {c["name"]: c["win_pct"] for c in result["categories"]}

    # USER decisively wins counting stats with ~10 R/wk advantage.
    assert cat_probs["R"] > 0.95, f"P(user wins R) should be >0.95; got {cat_probs['R']:.3f}"
    assert cat_probs["RBI"] > 0.95, f"P(user wins RBI) should be >0.95; got {cat_probs['RBI']:.3f}"
    assert cat_probs["HR"] > 0.85, f"P(user wins HR) should be >0.85; got {cat_probs['HR']:.3f}"
    assert cat_probs["K"] > 0.95, f"P(user wins K) should be >0.95; got {cat_probs['K']:.3f}"

    # USER decisively wins inverse rate stats (lower better).
    assert cat_probs["ERA"] > 0.95, f"P(user wins ERA) should be >0.95; got {cat_probs['ERA']:.3f}"
    assert cat_probs["WHIP"] > 0.95, f"P(user wins WHIP) should be >0.95; got {cat_probs['WHIP']:.3f}"

    # SV tied at 0 → p_win ≈ 0.5.
    assert 0.4 < cat_probs["SV"] < 0.6, f"SV tied at 0; p_win should be ~0.5, got {cat_probs['SV']:.3f}"
