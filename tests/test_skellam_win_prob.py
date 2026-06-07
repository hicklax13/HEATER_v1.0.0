"""Guard: Skellam skew-aware win probability for counting categories.

Item 8 (2026-05-25): per report Sections C.5 / H.9, low-count categories
(SB, SV) are right-skewed and the Gaussian approximation in the "normal"
variance model mis-handles the asymmetry. The "skellam" model treats each
side's weekly count as Poisson and uses the Skellam distribution (the exact
difference of two Poissons) for the head-to-head probability. Rate stats
keep the normal model regardless (Skellam is integer-only).
"""

from __future__ import annotations

import pandas as pd

from src.engine.output.weekly_matrix import (
    _category_win_prob,
    _category_win_prob_skellam,
    _per_week_means,
    compute_weekly_matrix,
)
from src.valuation import LeagueConfig

CONFIG = LeagueConfig()


def test_skellam_equal_means_near_half() -> None:
    p = _category_win_prob_skellam(5.0, 5.0, inverse=False)
    assert 0.45 <= p <= 0.55


def test_skellam_higher_mean_wins() -> None:
    assert _category_win_prob_skellam(8.0, 4.0, inverse=False) > 0.6
    assert _category_win_prob_skellam(4.0, 8.0, inverse=False) < 0.4


def test_skellam_inverse_flips() -> None:
    """For an inverse counting cat (Losses), the lower count wins."""
    higher = _category_win_prob_skellam(8.0, 4.0, inverse=False)
    inv = _category_win_prob_skellam(8.0, 4.0, inverse=True)
    assert inv < 0.5 < higher
    # Symmetry: inverse(a,b) ≈ 1 - normal(a,b) up to the tie split
    assert abs((higher + inv) - 1.0) < 1e-6


def test_skellam_handles_zero_mean() -> None:
    """mu=0 must not raise (Skellam needs strictly positive rates)."""
    p = _category_win_prob_skellam(0.0, 3.0, inverse=False)
    assert 0.0 <= p <= 0.5


def test_dispatch_uses_skellam_only_for_counting() -> None:
    """variance_model='skellam' routes counting cats to Skellam, rates to normal."""
    # Rate cat (inverse=False, counting=False): must equal the normal-model value.
    normal_rate = _category_win_prob(0.30, 0.27, cv=0.05, inverse=False, variance_model="normal", counting=False)
    skellam_rate = _category_win_prob(0.30, 0.27, cv=0.05, inverse=False, variance_model="skellam", counting=False)
    assert normal_rate == skellam_rate

    # Counting cat: skellam path should differ from the normal-CV path.
    normal_cnt = _category_win_prob(6.0, 4.0, cv=0.25, inverse=False, variance_model="normal", counting=True)
    skellam_cnt = _category_win_prob(6.0, 4.0, cv=0.25, inverse=False, variance_model="skellam", counting=True)
    assert normal_cnt != skellam_cnt


def _mini_pool() -> tuple[pd.DataFrame, dict[str, list[int]]]:
    rows = []
    rosters: dict[str, list[int]] = {"Me": [], "Opp": []}
    pid = 1
    for team, scale in (("Me", 1.0), ("Opp", 0.8)):
        for _ in range(3):
            rows.append(
                {
                    "player_id": pid,
                    "name": f"P{pid}",
                    "is_hitter": 1,
                    "positions": "OF",
                    "r": 80 * scale,
                    "hr": 22 * scale,
                    "rbi": 80 * scale,
                    "sb": 14 * scale,
                    "h": 140 * scale,
                    "ab": 520,
                    "bb": 55,
                    "hbp": 5,
                    "sf": 5,
                    "pa": 585,
                    "avg": 0.27 * scale,
                    "obp": 0.34 * scale,
                    "w": 0,
                    "l": 0,
                    "sv": 0,
                    "k": 0,
                    "ip": 0,
                    "er": 0,
                    "bb_allowed": 0,
                    "h_allowed": 0,
                    "era": 0,
                    "whip": 0,
                }
            )
            rosters[team].append(pid)
            pid += 1
    return pd.DataFrame(rows), rosters


def test_compute_weekly_matrix_skellam_method_tag() -> None:
    pool, rosters = _mini_pool()
    schedule = {10: "Opp", 11: "Opp"}
    result = compute_weekly_matrix(rosters["Me"], pool, schedule, rosters, config=CONFIG, variance_model="skellam")
    assert result["method"] == "skellam"
    # Matrix is well-formed and bounded.
    mat = result["matrix"]
    assert mat.shape[0] == 2
    assert ((mat.fillna(0.5) >= 0.0) & (mat.fillna(0.5) <= 1.0)).all().all()


# ── TE-E4: Skellam is the DEFAULT for low-count counting cats ────────────


def test_weekly_matrix_default_routes_low_count_to_skellam() -> None:
    """By default (no variance_model), a low-count counting cat (SB) routes
    through Skellam while a rate cat (AVG) stays Normal. Compared against the
    explicit all-Normal override, SB must differ and AVG must match.
    """
    pool, rosters = _mini_pool()
    schedule = {10: "Opp"}
    default = compute_weekly_matrix(rosters["Me"], pool, schedule, rosters, config=CONFIG)
    all_normal = compute_weekly_matrix(rosters["Me"], pool, schedule, rosters, config=CONFIG, variance_model="normal")
    # SB is a low-count counting cat → default uses Skellam, override uses Normal.
    assert default["matrix"].loc[10, "SB"] != all_normal["matrix"].loc[10, "SB"]
    # AVG is a rate cat → both models identical.
    assert abs(default["matrix"].loc[10, "AVG"] - all_normal["matrix"].loc[10, "AVG"]) < 1e-12


def test_weekly_matrix_default_method_tag_flags_skellam_lowcount() -> None:
    """The auto-routing default carries a distinct method tag (not 'cv-based',
    which is reserved for the all-Normal override, nor bare 'skellam')."""
    pool, rosters = _mini_pool()
    schedule = {10: "Opp"}
    default = compute_weekly_matrix(rosters["Me"], pool, schedule, rosters, config=CONFIG)
    assert "skellam" in default["method"]
    assert default["method"] != "skellam"  # not the force-all-counting-skellam tag
    assert default["method"] != "cv-based"  # not the all-Normal override


def test_weekly_matrix_all_normal_override_matches_legacy() -> None:
    """variance_model='normal' forces every cat (incl. SB/W/L/SV) through the
    Normal model and keeps the legacy 'cv-based' method tag for back-compat."""
    pool, rosters = _mini_pool()
    schedule = {10: "Opp"}
    res = compute_weekly_matrix(rosters["Me"], pool, schedule, rosters, config=CONFIG, variance_model="normal")
    assert res["method"] == "cv-based"
    # Every SB cell must equal the direct Normal-model helper value.
    user_means = _per_week_means(rosters["Me"], pool, list(CONFIG.all_categories), int(CONFIG.season_weeks))
    opp_means = _per_week_means(rosters["Opp"], pool, list(CONFIG.all_categories), int(CONFIG.season_weeks))
    expected_sb = _category_win_prob(
        mu_h=user_means["SB"], mu_o=opp_means["SB"], cv=0.25, inverse=False, variance_model="normal", counting=True
    )
    assert abs(res["matrix"].loc[10, "SB"] - expected_sb) < 1e-12
