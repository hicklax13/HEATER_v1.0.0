"""Regression lock for L3 (standings_engine divisor → config.season_weeks).

PR #47 changed `_estimate_team_weekly_stats` to divide by `config.season_weeks` (26)
instead of `weeks_remaining`. Silent-failure-hunter (run 2026-05-19 after merge)
flagged this as H2 — concerned that tau (variance) would be miscalibrated relative
to the new mean scale, compressing all win probabilities toward 0.5.

Empirical check: with the L3 fix, a clearly-elite team (910 R/season → 35 R/wk)
vs a weak team (650 R/season → 25 R/wk) is the clear matchup favorite — H2 was a
false positive (p_win is NOT compressed toward 0.5).

MS-E1 (2026-06-07) then unified the three divergent per-module weekly-variance
tables onto one canonical source (h2h_engine.default_weekly_sigmas). The prior
standings_engine taus were implausibly tight (R=1.6, K=1.2), which saturated a
10 R/wk edge to ~0.99. Under the canonical, empirically-grounded SDs (R sd=15,
K sd=12):

    USER weekly R = 35.08, OPP weekly R = 24.92
    P(USER wins R) ≈ 0.75   ← clear favorite, REALISTICALLY calibrated
    Overall win pct > 0.95  ← decisive, driven by the rate-stat sweep

This test locks the realistic p_win bands to detect future drift if the divisor
or the canonical weekly-SD scale ever changes.
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
    """L3 + MS-E1 lock: the elite roster is clearly favored, with REALISTICALLY
    calibrated per-category win-probs.

    Silent-failure-hunter H2 was concerned the L3 divisor change would compress
    p_win toward 0.5; it does not — the elite roster decisively wins the matchup
    overall. MS-E1 then replaced the implausibly-tight per-module tau (R=1.6,
    K=1.2 — which saturated a 10 R/wk edge to ~0.99) with the canonical,
    empirically-grounded weekly SDs (R sd=15, K sd=12). A 10-runs/week edge is a
    clear-favorite ~0.75, NOT a near-certain 0.99 — that is the correct, sane
    calibration this test now locks. The overall win-prob stays decisive because
    the rate stats (AVG/OBP/ERA/WHIP) are swept by wide margins.
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

    # Decisive overall win — still very high given the lopsided rosters (driven
    # by the rate-stat sweep), even with realistically-wider counting-stat SDs.
    assert result["overall_win_pct"] > 0.95, (
        f"overall_win_pct should be >0.95 for clearly-elite vs weak rosters; "
        f"got {result['overall_win_pct']:.3f}. If compressed near 0.5, the L3 divisor + "
        f"canonical weekly SDs (h2h_engine.default_weekly_sigmas) are out of scale."
    )

    cat_probs = {c["name"]: c["win_pct"] for c in result["categories"]}

    # USER is the clear favorite in counting stats with a ~10 R/wk advantage —
    # favored but NOT near-certain, the empirically-correct band (~0.66-0.85).
    assert 0.65 < cat_probs["R"] < 0.85, f"P(user wins R) should be a clear favorite ~0.75; got {cat_probs['R']:.3f}"
    assert 0.60 < cat_probs["RBI"] < 0.80, f"P(user wins RBI) should be ~0.67; got {cat_probs['RBI']:.3f}"
    assert 0.65 < cat_probs["HR"] < 0.90, f"P(user wins HR) should be ~0.78; got {cat_probs['HR']:.3f}"
    assert 0.75 < cat_probs["K"] < 0.95, f"P(user wins K) should be ~0.85; got {cat_probs['K']:.3f}"

    # Rate stats are swept by wide margins relative to their (small) weekly SDs —
    # these remain decisively in the user's favor and drive the overall win.
    assert cat_probs["AVG"] > 0.90, f"P(user wins AVG) should be decisive; got {cat_probs['AVG']:.3f}"
    assert cat_probs["ERA"] > 0.90, f"P(user wins ERA) should be decisive; got {cat_probs['ERA']:.3f}"
    assert cat_probs["WHIP"] > 0.90, f"P(user wins WHIP) should be decisive; got {cat_probs['WHIP']:.3f}"

    # USER decisively wins inverse rate stats (lower better).
    assert cat_probs["ERA"] > 0.95, f"P(user wins ERA) should be >0.95; got {cat_probs['ERA']:.3f}"
    assert cat_probs["WHIP"] > 0.95, f"P(user wins WHIP) should be >0.95; got {cat_probs['WHIP']:.3f}"

    # SV tied at 0 → p_win ≈ 0.5.
    assert 0.4 < cat_probs["SV"] < 0.6, f"SV tied at 0; p_win should be ~0.5, got {cat_probs['SV']:.3f}"
