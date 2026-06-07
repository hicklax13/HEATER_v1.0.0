"""Structural-invariant guard: playoff + championship probability bracket sim.

Feature 3 (2026-05-23): per report Section B.10 + Q(a), the engine's
PRIMARY objective. Simulates the rest of the regular season + 2-round
bracket (top-4 → 1v4, 2v3 → re-seeded final) and reports the user's
probability of making the playoffs and winning the championship.

Design choices baked in:
  - FourzynBurn / Yahoo standard: top-4 playoff cutoff, 2-round bracket
    (1v4, 2v3 in round 1; winners re-seeded in final)
  - 20,000 MC sims per user choice 2026-05-23 Q4
  - Paired MC discipline across before/after arms (same seed)
  - Hickey-centric scope: user's per-week probabilities from weekly_matrix
    schedule-aware computation; opponents approximated via Binomial(N, p_avg)
    where p_avg derives from team's avg win-rate vs league average
  - Bradley-Terry style matchup prob: p_a / (p_a + p_b), bounded [0.05, 0.95]

These tests verify the sim is well-formed without requiring it to match a
specific oracle value (the sim has 20K-sim noise; assertions are on
direction, range, and invariants).
"""

from __future__ import annotations

import pandas as pd

from src.engine.output.playoff_sim import (
    _PLAYOFF_SPOTS,
    _matchup_prob,
    _prob_majority_cat_wins,
    simulate_playoff_outcomes,
    simulate_trade_playoff_delta,
)
from src.engine.output.trade_evaluator import evaluate_trade
from src.valuation import LeagueConfig

CONFIG = LeagueConfig()


# ── Helper formula tests ─────────────────────────────────────────────


def test_matchup_prob_equal_strength_is_50_50() -> None:
    assert _matchup_prob(0.55, 0.55) == 0.5


def test_matchup_prob_stronger_wins() -> None:
    assert _matchup_prob(0.70, 0.40) > 0.5
    assert _matchup_prob(0.40, 0.70) < 0.5


def test_matchup_prob_clamped() -> None:
    """No matchup is 100% certain even with massive strength gap."""
    p_extreme = _matchup_prob(0.99, 0.01)
    assert 0.05 <= p_extreme <= 0.95


def test_matchup_prob_both_zero_is_50_50() -> None:
    """Degenerate case must return 0.5, not divide-by-zero."""
    assert _matchup_prob(0.0, 0.0) == 0.5


def test_prob_majority_cat_wins_basic() -> None:
    """If every cat is 50/50, P(majority) ≈ 0.5."""
    import numpy as np

    p = np.full((3, 12), 0.5)  # 3 weeks, all cats 50/50
    result = _prob_majority_cat_wins(p)
    # Result is approximately 0.5 (continuity-corrected, slight deviation OK)
    assert all(0.40 < r < 0.60 for r in result)


def test_prob_majority_cat_wins_dominant() -> None:
    """If every cat is heavily favored, P(majority) → ~1.0."""
    import numpy as np

    p = np.full((1, 12), 0.95)
    result = _prob_majority_cat_wins(p)
    assert result[0] > 0.95


# ── Mini-league for sim tests ────────────────────────────────────────


def _build_mini_league(n_teams: int = 12) -> tuple[pd.DataFrame, dict[str, list[int]], dict[str, int]]:
    """12-team mini-league with scaled per-team strengths."""
    rows = []
    rosters: dict[str, list[int]] = {}
    current_wins: dict[str, int] = {}
    pid = 1
    for team_idx in range(n_teams):
        team_name = f"Team_{team_idx}"
        rosters[team_name] = []
        current_wins[team_name] = 4 + team_idx // 3  # 4-7 wins each
        scale = 0.70 + 0.05 * team_idx  # 0.70 to 1.25
        for _ in range(3):
            rows.append(
                {
                    "player_id": pid,
                    "name": f"P{pid}",
                    "is_hitter": 1 if pid % 2 == 0 else 0,
                    "positions": "OF" if pid % 2 == 0 else "SP",
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
                    "w": 12 * scale,
                    "l": 8 / scale,  # better team has fewer losses
                    "sv": 0,
                    "k": 180 * scale,
                    "ip": 180,
                    "er": 70 / scale,  # better team allows fewer ER
                    "bb_allowed": 50,
                    "h_allowed": 160,
                    "era": 3.50 / scale,
                    "whip": 1.17 / scale,
                }
            )
            rosters[team_name].append(pid)
            pid += 1
    pool = pd.DataFrame(rows)
    return pool, rosters, current_wins


def _build_schedule(user_team: str, opponents: list[str], start_week: int, end_week: int) -> dict[int, str]:
    weeks = list(range(start_week, end_week + 1))
    return {w: opponents[(w - start_week) % len(opponents)] for w in weeks}


# ── simulate_playoff_outcomes ────────────────────────────────────────


def test_simulate_returns_required_keys() -> None:
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule("Team_5", [t for t in rosters if t != "Team_5"], 10, 26)
    result = simulate_playoff_outcomes(
        user_roster_ids=rosters["Team_5"],
        user_team_name="Team_5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=2000,  # smaller for test speed
    )
    for key in (
        "playoff_prob",
        "champ_prob",
        "playoff_seed_distribution",
        "mean_regular_season_wins",
        "n_sims",
        "method",
    ):
        assert key in result, f"Missing key {key!r} in simulate result"


def test_probabilities_in_unit_interval() -> None:
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule("Team_5", [t for t in rosters if t != "Team_5"], 10, 26)
    result = simulate_playoff_outcomes(
        user_roster_ids=rosters["Team_5"],
        user_team_name="Team_5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=2000,
    )
    assert 0.0 <= result["playoff_prob"] <= 1.0
    assert 0.0 <= result["champ_prob"] <= 1.0
    # Champ prob can't exceed playoff prob (you must make playoffs to win)
    assert result["champ_prob"] <= result["playoff_prob"] + 1e-9


def test_stronger_team_has_higher_playoff_prob() -> None:
    """Best team (Team_11) should have higher playoff prob than worst (Team_0)."""
    pool, rosters, wins = _build_mini_league()
    opp_for_strong = [t for t in rosters if t != "Team_11"]
    opp_for_weak = [t for t in rosters if t != "Team_0"]
    sched_strong = _build_schedule("Team_11", opp_for_strong, 10, 26)
    sched_weak = _build_schedule("Team_0", opp_for_weak, 10, 26)

    strong_res = simulate_playoff_outcomes(
        user_roster_ids=rosters["Team_11"],
        user_team_name="Team_11",
        all_team_rosters=rosters,
        user_schedule=sched_strong,
        current_wins=wins,
        player_pool=pool,
        n_sims=3000,
        seed=123,
    )
    weak_res = simulate_playoff_outcomes(
        user_roster_ids=rosters["Team_0"],
        user_team_name="Team_0",
        all_team_rosters=rosters,
        user_schedule=sched_weak,
        current_wins=wins,
        player_pool=pool,
        n_sims=3000,
        seed=123,
    )
    # Strong team should have meaningfully higher playoff prob
    assert strong_res["playoff_prob"] > weak_res["playoff_prob"], (
        f"Stronger team should have higher playoff prob. "
        f"Strong={strong_res['playoff_prob']:.3f} weak={weak_res['playoff_prob']:.3f}"
    )


def test_seed_distribution_sums_to_playoff_prob() -> None:
    """Sum of P(seed=1..4) should equal playoff_prob (you're in top-4 in exactly one seed)."""
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule("Team_8", [t for t in rosters if t != "Team_8"], 10, 26)
    result = simulate_playoff_outcomes(
        user_roster_ids=rosters["Team_8"],
        user_team_name="Team_8",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=3000,
    )
    seed_sum = sum(result["playoff_seed_distribution"].values())
    assert abs(seed_sum - result["playoff_prob"]) < 1e-6


def test_user_not_in_rosters_returns_zero() -> None:
    """If user_team_name not in rosters, return zero probabilities (no crash)."""
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule("Team_5", list(rosters.keys()), 10, 26)
    result = simulate_playoff_outcomes(
        user_roster_ids=[],
        user_team_name="DOES_NOT_EXIST",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=1000,
    )
    assert result["playoff_prob"] == 0.0
    assert result["champ_prob"] == 0.0


# ── simulate_trade_playoff_delta (paired-MC) ─────────────────────────


def test_identical_rosters_zero_delta() -> None:
    """Before == After → delta is exactly 0 (paired-MC discipline)."""
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule("Team_5", [t for t in rosters if t != "Team_5"], 10, 26)
    result = simulate_trade_playoff_delta(
        before_roster_ids=rosters["Team_5"],
        after_roster_ids=rosters["Team_5"],  # SAME roster
        user_team_name="Team_5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=2000,
    )
    # Same seed + same inputs + paired MC → delta should be exactly 0
    assert result["delta_playoff_prob"] == 0.0
    assert result["delta_champ_prob"] == 0.0


# ── evaluate_trade integration ───────────────────────────────────────


def test_evaluate_trade_opt_in_only() -> None:
    """enable_playoff_sim=False → no playoff_sim key in result."""
    pool, rosters, wins = _build_mini_league()
    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_5"],
        player_pool=pool,
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
    )
    assert "playoff_sim" not in result
    assert "delta_playoff_prob" not in result


def test_evaluate_trade_with_playoff_sim() -> None:
    """All flags → result has playoff_sim + top-level delta fields."""
    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule("Team_5", [t for t in rosters if t != "Team_5"], 10, 26)
    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_5"],
        player_pool=pool,
        user_team_name="Team_5",
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
        enable_playoff_sim=True,
        weekly_schedule=schedule,
        league_rosters=rosters,
        current_wins=wins,
        playoff_n_sims=1000,  # smaller for test speed
    )
    assert "playoff_sim" in result
    assert "delta_playoff_prob" in result
    assert "delta_champ_prob" in result
    ps = result["playoff_sim"]
    assert "before" in ps and "after" in ps
    assert "delta_playoff_prob" in ps and "delta_champ_prob" in ps


def test_evaluate_trade_playoff_sim_missing_inputs_skips_gracefully() -> None:
    """enable_playoff_sim=True but missing current_wins → skip, don't crash."""
    pool, rosters, _ = _build_mini_league()
    schedule = _build_schedule("Team_5", [t for t in rosters if t != "Team_5"], 10, 26)
    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_5"],
        player_pool=pool,
        user_team_name="Team_5",
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
        enable_playoff_sim=True,
        weekly_schedule=schedule,
        league_rosters=rosters,
        current_wins=None,  # missing
    )
    assert "grade" in result  # baseline still works
    assert "playoff_sim" not in result


def test_playoff_spots_constant_is_four() -> None:
    """FourzynBurn / CLAUDE.md says playoff field is 4 teams."""
    assert _PLAYOFF_SPOTS == 4


# ── TE-E3: schedule-aware Path A is the default when data is available ──


def _build_full_schedule(team_names: list[str], start_week: int, end_week: int) -> dict[int, list[tuple[str, str]]]:
    """Circle-method round-robin full league schedule for an even team count."""
    n = len(team_names)
    fixed = team_names[0]
    others = team_names[1:]
    full: dict[int, list[tuple[str, str]]] = {}
    for wk_idx, wk in enumerate(range(start_week, end_week + 1)):
        rotation = wk_idx % (n - 1)
        rotated = others[rotation:] + others[:rotation]
        matchups: list[tuple[str, str]] = [(fixed, rotated[0])]
        for i in range(1, (n - 1) // 2 + 1):
            matchups.append((rotated[i], rotated[-i]))
        full[wk] = matchups
    return full


def test_evaluate_trade_auto_uses_full_schedule_from_db(monkeypatch) -> None:
    """TE-E3: when the caller doesn't pass full_league_schedule but the DB
    cache has one, evaluate_trade auto-loads it so Path A is the default."""
    import src.database as _db

    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule("Team_5", [t for t in rosters if t != "Team_5"], 10, 25)
    full_sched = _build_full_schedule(sorted(rosters.keys()), 10, 25)
    monkeypatch.setattr(_db, "load_league_schedule_full", lambda: full_sched)

    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_5"],
        player_pool=pool,
        user_team_name="Team_5",
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
        enable_playoff_sim=True,
        weekly_schedule=schedule,
        league_rosters=rosters,
        current_wins=wins,
        playoff_n_sims=600,
        full_league_schedule=None,  # caller did NOT supply it
    )
    assert "playoff_sim" in result
    # Path A method tag (a "-corr" suffix may follow when the copula path is on).
    assert result["playoff_sim"]["before"]["method"].startswith("hickey-centric-full-sched-opp")


def test_evaluate_trade_falls_back_to_path_b_when_no_schedule(monkeypatch) -> None:
    """TE-E3: when neither the caller nor the DB has a full schedule, Path B
    (Binomial league-average) is the documented fallback."""
    import src.database as _db

    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule("Team_5", [t for t in rosters if t != "Team_5"], 10, 25)
    monkeypatch.setattr(_db, "load_league_schedule_full", lambda: {})

    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_5"],
        player_pool=pool,
        user_team_name="Team_5",
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
        enable_playoff_sim=True,
        weekly_schedule=schedule,
        league_rosters=rosters,
        current_wins=wins,
        playoff_n_sims=600,
        full_league_schedule=None,
    )
    assert "playoff_sim" in result
    assert result["playoff_sim"]["before"]["method"].startswith("hickey-centric-binomial-opp")


def test_evaluate_trade_explicit_full_schedule_not_overwritten(monkeypatch) -> None:
    """TE-E3: a caller-supplied full_league_schedule must take precedence over
    the DB auto-load (no DB read needed when the caller already passed one)."""
    import src.database as _db

    pool, rosters, wins = _build_mini_league()
    schedule = _build_schedule("Team_5", [t for t in rosters if t != "Team_5"], 10, 25)
    full_sched = _build_full_schedule(sorted(rosters.keys()), 10, 25)

    def _boom() -> dict:
        raise AssertionError("DB auto-load must not run when caller supplies full_league_schedule")

    monkeypatch.setattr(_db, "load_league_schedule_full", _boom)

    result = evaluate_trade(
        giving_ids=[],
        receiving_ids=[],
        user_roster_ids=rosters["Team_5"],
        player_pool=pool,
        user_team_name="Team_5",
        enable_mc=False,
        enable_context=False,
        enable_game_theory=False,
        apply_ytd_blend=False,
        enable_playoff_sim=True,
        weekly_schedule=schedule,
        league_rosters=rosters,
        current_wins=wins,
        playoff_n_sims=600,
        full_league_schedule=full_sched,
    )
    assert result["playoff_sim"]["before"]["method"].startswith("hickey-centric-full-sched-opp")
