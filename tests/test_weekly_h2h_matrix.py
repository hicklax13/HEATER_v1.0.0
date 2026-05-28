"""Structural-invariant guard: 26-week × 12-cat H2H win-probability matrix.

Feature 2 (2026-05-23): per report Section B.5, for each remaining
matchup week w and category c, compute p_{w,c} = Phi((mu_H - mu_O) /
sqrt(sigma_H^2 + sigma_O^2)) — the probability of winning category c
in week w against the scheduled opponent.

Design choices baked in:
  - Per-category CV calibration (rate stats stable, pitching W/L volatile)
  - Inverse cats (L, ERA, WHIP) flipped: 1 - p so "good" is always >0.5
  - Volume-weighted rate stats (h/ab, etc.) not naive mean of rates
  - Min sigma floor 0.01 to avoid divide-by-zero at zero means
  - Default behavior when opponent missing: 0.5 neutral (no fake data)

These tests cover the module in isolation; the evaluate_trade integration
test verifies wiring + opt-in flag behavior.
"""

from __future__ import annotations

import pandas as pd

from src.engine.output.trade_evaluator import evaluate_trade
from src.engine.output.weekly_matrix import (
    _DEFAULT_CV,
    _category_win_prob,
    _per_week_means,
    compute_trade_weekly_delta,
    compute_weekly_matrix,
)
from src.valuation import LeagueConfig

CONFIG = LeagueConfig()


# ── Pure helper tests ────────────────────────────────────────────────


def test_category_win_prob_higher_mean_wins() -> None:
    """When mu_h > mu_o, p > 0.5 for normal cats; p < 0.5 for inverse."""
    # Normal cat (HR): higher = good
    p_normal = _category_win_prob(mu_h=30.0, mu_o=20.0, cv=0.20, inverse=False)
    assert p_normal > 0.5, f"Higher mean should win normal cat; got p={p_normal}"

    # Inverse cat (ERA): higher = bad → p < 0.5
    p_inverse = _category_win_prob(mu_h=4.50, mu_o=3.50, cv=0.18, inverse=True)
    assert p_inverse < 0.5, f"Higher mean (worse) should lose inverse cat; got p={p_inverse}"


def test_category_win_prob_equal_means_is_50_50() -> None:
    """Symmetric noise + identical means → exactly 0.5."""
    p = _category_win_prob(mu_h=20.0, mu_o=20.0, cv=0.20, inverse=False)
    assert abs(p - 0.5) < 1e-9, f"Equal means must give 0.5; got {p}"

    p_inv = _category_win_prob(mu_h=3.50, mu_o=3.50, cv=0.18, inverse=True)
    assert abs(p_inv - 0.5) < 1e-9


def test_category_win_prob_clamped_to_unit_interval() -> None:
    """Probabilities must be in [0, 1] even for extreme inputs."""
    # Enormous gap
    p = _category_win_prob(mu_h=1000.0, mu_o=1.0, cv=0.01, inverse=False)
    assert 0.0 <= p <= 1.0


def test_category_win_prob_zero_means_no_crash() -> None:
    """Both means zero → degenerate but must return a valid probability."""
    p = _category_win_prob(mu_h=0.0, mu_o=0.0, cv=0.20, inverse=False)
    assert 0.0 <= p <= 1.0
    # With min_sigma floor, both sigmas are 0.01 → z=0 → 0.5
    assert abs(p - 0.5) < 0.01


def test_default_cv_covers_all_12_categories() -> None:
    """_DEFAULT_CV must have an entry for every league category."""
    for cat in CONFIG.all_categories:
        assert cat in _DEFAULT_CV, f"_DEFAULT_CV missing entry for {cat}"


# ── _per_week_means ──────────────────────────────────────────────────


def _two_hitter_pool() -> pd.DataFrame:
    """Two hitters with different stats so the team aggregate is non-trivial."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Power",
                "is_hitter": 1,
                "positions": "OF",
                "r": 80,
                "hr": 35,
                "rbi": 95,
                "sb": 5,
                "h": 130,
                "ab": 500,
                "bb": 60,
                "hbp": 5,
                "sf": 5,
                "pa": 570,
                "avg": 0.260,
                "obp": 0.340,
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
            },
            {
                "player_id": 2,
                "name": "Speed",
                "is_hitter": 1,
                "positions": "OF",
                "r": 100,
                "hr": 10,
                "rbi": 60,
                "sb": 40,
                "h": 165,
                "ab": 550,
                "bb": 50,
                "hbp": 3,
                "sf": 5,
                "pa": 608,
                "avg": 0.300,
                "obp": 0.360,
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
            },
        ]
    )


def test_per_week_means_counting_stats_divide_by_season_weeks() -> None:
    """R total = 180 across 26 weeks → 6.92 R/week."""
    pool = _two_hitter_pool()
    means = _per_week_means([1, 2], pool, CONFIG.all_categories, season_weeks=26)
    assert abs(means["R"] - (180.0 / 26.0)) < 1e-6
    assert abs(means["HR"] - (45.0 / 26.0)) < 1e-6
    assert abs(means["SB"] - (45.0 / 26.0)) < 1e-6


def test_per_week_means_rate_stats_volume_weighted() -> None:
    """AVG = sum(H) / sum(AB), not mean of player AVGs."""
    pool = _two_hitter_pool()
    means = _per_week_means([1, 2], pool, CONFIG.all_categories, season_weeks=26)
    expected_avg = (130 + 165) / (500 + 550)  # = 0.2810
    assert abs(means["AVG"] - expected_avg) < 1e-6
    # OBP weighted
    expected_obp = (130 + 60 + 5 + 165 + 50 + 3) / (500 + 60 + 5 + 5 + 550 + 50 + 3 + 5)
    assert abs(means["OBP"] - expected_obp) < 1e-6


def test_per_week_means_empty_roster_returns_zeros() -> None:
    pool = _two_hitter_pool()
    means = _per_week_means([], pool, CONFIG.all_categories, season_weeks=26)
    assert all(v == 0.0 for v in means.values())


# ── compute_weekly_matrix ────────────────────────────────────────────


def _league_setup() -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """Build a 12-team mini-league with realistic per-team strengths.

    User (Team_0) gets moderate hitters; opponents range from worse (Team_1)
    to better (Team_11) so we can verify the matrix is sensitive to opponent.
    """
    rows = []
    rosters: dict[str, list[int]] = {}
    pid = 1
    for team_idx in range(12):
        team_name = f"Team_{team_idx}"
        rosters[team_name] = []
        # 2 hitters per team, scaled by team_idx (worse team idx 0 → better team idx 11)
        scale = 0.7 + 0.05 * team_idx  # 0.70 to 1.25
        for _ in range(2):
            rows.append(
                {
                    "player_id": pid,
                    "name": f"P{pid}",
                    "is_hitter": 1,
                    "positions": "OF",
                    "r": 80 * scale,
                    "hr": 22 * scale,
                    "rbi": 80 * scale,
                    "sb": 12 * scale,
                    "h": 130 * scale,
                    "ab": 500,
                    "bb": 55,
                    "hbp": 5,
                    "sf": 5,
                    "pa": 570,
                    "avg": 0.260 * scale,
                    "obp": 0.330 * scale,
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
            rosters[team_name].append(pid)
            pid += 1
    return pd.DataFrame(rows), rosters


def test_matrix_shape_matches_schedule() -> None:
    """Returned matrix is indexed by week, columns are 12 cats."""
    pool, rosters = _league_setup()
    schedule = {8: "Team_5", 9: "Team_7", 10: "Team_3"}
    result = compute_weekly_matrix(
        user_roster_ids=rosters["Team_0"],
        player_pool=pool,
        schedule=schedule,
        all_team_rosters=rosters,
    )
    m = result["matrix"]
    assert list(m.index) == [8, 9, 10]
    assert set(m.columns) == set(CONFIG.all_categories)
    assert m.shape == (3, len(CONFIG.all_categories))


def test_matrix_against_stronger_opponent_has_lower_win_probs() -> None:
    """A team facing a STRONGER opponent should have lower hitting win probs."""
    pool, rosters = _league_setup()
    schedule_easy = {8: "Team_1"}  # weak opponent
    schedule_hard = {8: "Team_11"}  # strong opponent

    res_easy = compute_weekly_matrix(rosters["Team_5"], pool, schedule_easy, rosters)
    res_hard = compute_weekly_matrix(rosters["Team_5"], pool, schedule_hard, rosters)

    # User's HR win prob should be higher vs easy opponent
    assert res_easy["matrix"].loc[8, "HR"] > res_hard["matrix"].loc[8, "HR"]
    assert res_easy["matrix"].loc[8, "R"] > res_hard["matrix"].loc[8, "R"]


def test_unknown_opponent_returns_neutral_05() -> None:
    """If schedule says opponent we don't have data for, return 0.5."""
    pool, rosters = _league_setup()
    schedule = {8: "Team_DOES_NOT_EXIST"}
    res = compute_weekly_matrix(rosters["Team_0"], pool, schedule, rosters)
    assert all(res["matrix"].loc[8, :] == 0.5)


def test_inverse_cats_are_flipped() -> None:
    """ERA win prob > 0.5 means user has LOWER ERA than opp (good)."""
    # Build a small pool where Team_A has lower ERA than Team_B
    rows = []
    for i in range(1, 3):
        rows.append(
            {
                "player_id": i,
                "name": f"GoodPitcher{i}",
                "is_hitter": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "h": 0,
                "ab": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "pa": 0,
                "w": 15,
                "l": 6,
                "sv": 0,
                "k": 200,
                "ip": 180,
                "er": 60,
                "bb_allowed": 45,
                "h_allowed": 140,
                "era": 3.00,
                "whip": 1.03,
            }
        )
    for i in range(3, 5):
        rows.append(
            {
                "player_id": i,
                "name": f"BadPitcher{i}",
                "is_hitter": 0,
                "r": 0,
                "hr": 0,
                "rbi": 0,
                "sb": 0,
                "avg": 0,
                "obp": 0,
                "h": 0,
                "ab": 0,
                "bb": 0,
                "hbp": 0,
                "sf": 0,
                "pa": 0,
                "w": 8,
                "l": 10,
                "sv": 0,
                "k": 130,
                "ip": 180,
                "er": 100,
                "bb_allowed": 70,
                "h_allowed": 180,
                "era": 5.00,
                "whip": 1.39,
            }
        )
    pool = pd.DataFrame(rows)
    rosters = {"GoodTeam": [1, 2], "BadTeam": [3, 4]}
    schedule = {8: "BadTeam"}

    res = compute_weekly_matrix(rosters["GoodTeam"], pool, schedule, rosters)
    # GoodTeam ERA = 3.00, BadTeam ERA = 5.00 → GoodTeam should win ERA
    assert res["matrix"].loc[8, "ERA"] > 0.5, (
        f"Good ERA team should win ERA category; got p={res['matrix'].loc[8, 'ERA']}"
    )
    # Similarly for WHIP and L
    assert res["matrix"].loc[8, "WHIP"] > 0.5
    assert res["matrix"].loc[8, "L"] > 0.5


# ── compute_trade_weekly_delta ───────────────────────────────────────


def test_delta_no_change_yields_zero_delta() -> None:
    """Identical before/after rosters → zero delta everywhere."""
    pool, rosters = _league_setup()
    schedule = {8: "Team_5", 9: "Team_7"}
    res = compute_trade_weekly_delta(
        before_roster_ids=rosters["Team_0"],
        after_roster_ids=rosters["Team_0"],
        player_pool=pool,
        schedule=schedule,
        all_team_rosters=rosters,
    )
    delta = res["delta"]
    # All deltas should be ~0 (might have tiny floating point noise)
    assert (delta.abs() < 1e-9).all().all()


def test_delta_summary_includes_expected_cat_wins() -> None:
    """Per-week summary tracks expected cat-wins before/after/delta + opponent."""
    pool, rosters = _league_setup()
    schedule = {8: "Team_5", 9: "Team_7"}
    res = compute_trade_weekly_delta(
        before_roster_ids=rosters["Team_0"],
        after_roster_ids=rosters["Team_0"],
        player_pool=pool,
        schedule=schedule,
        all_team_rosters=rosters,
    )
    summary = res["summary"]
    assert "expected_cat_wins_before" in summary.columns
    assert "expected_cat_wins_after" in summary.columns
    assert "delta_cat_wins" in summary.columns
    assert "opponent" in summary.columns
    # Expected cat wins should be in [0, 12]
    assert (summary["expected_cat_wins_before"] >= 0).all()
    assert (summary["expected_cat_wins_before"] <= 12).all()


# ── evaluate_trade integration ───────────────────────────────────────


def test_evaluate_trade_opt_in_only() -> None:
    """When enable_weekly_matrix=False (default), no weekly_matrix in result."""
    pool, rosters = _league_setup()
    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_0"],
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert "weekly_matrix" not in result, "weekly_matrix must not be in result when enable_weekly_matrix=False"


def test_evaluate_trade_with_weekly_matrix_enabled() -> None:
    """When opt-in + schedule + rosters provided, result has weekly_matrix."""
    pool, rosters = _league_setup()
    schedule = {8: "Team_5", 9: "Team_7"}
    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_0"],
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
        enable_weekly_matrix=True,
        weekly_schedule=schedule,
        league_rosters=rosters,
    )
    assert "weekly_matrix" in result
    wm = result["weekly_matrix"]
    assert "before" in wm
    assert "after" in wm
    assert "delta" in wm
    assert "summary" in wm


def test_evaluate_trade_opt_in_without_schedule_skips_gracefully() -> None:
    """enable_weekly_matrix=True but no schedule → no weekly_matrix key,
    but evaluate_trade still succeeds."""
    pool, rosters = _league_setup()
    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_0"],
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
        enable_weekly_matrix=True,
        weekly_schedule=None,  # missing
        league_rosters=None,
    )
    # Must NOT crash; weekly_matrix simply absent
    assert "grade" in result  # baseline output still works
    assert "weekly_matrix" not in result
