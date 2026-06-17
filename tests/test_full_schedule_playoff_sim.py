"""Structural-invariant guard: full opponent schedule simulation per report B.10.

Phase 3 (2026-05-24) — replaces the Binomial(N, avg_p) approximation in
playoff_sim with per-team-per-week schedule-aware simulation using
load_league_schedule_full(). Each opponent's wins are now computed as
the sum of per-week P(win) against their actual scheduled opponent.

Contract locked:
  - simulate_playoff_outcomes accepts full_league_schedule kwarg (optional)
  - When provided: result['method'] starts with 'hickey-centric-full-sched-opp'
    (a '-corr' suffix is appended when the TE-E2 copula-correlated category
    path is active, which is the default)
  - When None: result['method'] starts with 'hickey-centric-binomial-opp' (legacy)
  - _compute_per_week_per_team_p_win returns symmetric (a wins, b loses)
  - Trade Analyzer page loads + passes full_league_schedule
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.output.playoff_sim import (
    _compute_per_week_per_team_p_win,
    simulate_playoff_outcomes,
    simulate_trade_playoff_delta,
)
from src.valuation import LeagueConfig

PAGE_PATH = Path(__file__).resolve().parent.parent / "pages" / "11_Trade_Analyzer.py"


def _make_hitter(pid: int, name: str, hr: int = 20) -> dict:
    return {
        "player_id": pid,
        "name": name,
        "player_name": name,
        "is_hitter": 1,
        "positions": "OF",
        "status": "active",
        "r": 70,
        "hr": hr,
        "rbi": 75,
        "sb": 8,
        "h": 130,
        "ab": 500,
        "bb": 55,
        "hbp": 5,
        "sf": 5,
        "pa": 565,
        "avg": 0.260,
        "obp": 0.330,
        "w": 0,
        "l": 0,
        "sv": 0,
        "k": 0,
        "ip": 0,
        "ytd_ip": 0,
        "er": 0,
        "bb_allowed": 0,
        "h_allowed": 0,
        "era": 0,
        "whip": 0,
    }


def _build_league():
    rows = []
    rosters: dict[str, list[int]] = {}
    current_wins: dict[str, int] = {}
    pid = 1
    for team_idx in range(12):
        name = f"T{team_idx}"
        rosters[name] = []
        current_wins[name] = 4 + team_idx // 3
        for _ in range(3):
            rows.append(_make_hitter(pid, f"P{pid}", hr=15 + team_idx))
            rosters[name].append(pid)
            pid += 1
    pool = pd.DataFrame(rows)
    user_schedule = {w: f"T{(w % 11) + 1}" for w in range(9, 25)}
    # Full schedule: simple circle-method round-robin for 12 teams.
    # Fix team 0, rotate the other 11. Each week pairs (0, k), then pairs
    # (k+i, k-i mod 11) for i=1..5 — standard round-robin construction.
    full_schedule: dict[int, list[tuple[str, str]]] = {}
    n = 12
    fixed = "T0"
    others = [f"T{i}" for i in range(1, n)]
    for wk_idx, wk in enumerate(range(9, 25)):
        rotation = wk_idx % (n - 1)
        rotated = others[rotation:] + others[:rotation]
        matchups: list[tuple[str, str]] = [(fixed, rotated[0])]
        for i in range(1, (n - 1) // 2 + 1):
            matchups.append((rotated[i], rotated[-i]))
        full_schedule[wk] = matchups
    return pool, rosters, current_wins, user_schedule, full_schedule


# ── Helper function tests ────────────────────────────────────────────


def test_per_week_per_team_returns_correct_shape() -> None:
    """Output shape must be (n_teams, n_weeks_remaining)."""
    pool, rosters, _, _, full_schedule = _build_league()
    team_names = sorted(rosters.keys())
    weeks = list(range(9, 25))
    matrix = _compute_per_week_per_team_p_win(
        full_league_schedule=full_schedule,
        all_team_rosters=rosters,
        player_pool=pool,
        config=LeagueConfig(),
        variance_cv={cat: 0.15 for cat in LeagueConfig().all_categories},
        weeks_to_include=weeks,
        team_names=team_names,
    )
    assert matrix.shape == (12, 16)


def test_per_week_per_team_values_in_unit_interval() -> None:
    """All P(win) values must be in [0, 1]."""
    pool, rosters, _, _, full_schedule = _build_league()
    team_names = sorted(rosters.keys())
    matrix = _compute_per_week_per_team_p_win(
        full_league_schedule=full_schedule,
        all_team_rosters=rosters,
        player_pool=pool,
        config=LeagueConfig(),
        variance_cv={cat: 0.15 for cat in LeagueConfig().all_categories},
        weeks_to_include=list(range(9, 25)),
        team_names=team_names,
    )
    assert (matrix >= 0.0).all()
    assert (matrix <= 1.0).all()


def test_per_week_per_team_symmetric_pairs_sum_to_one() -> None:
    """When team A plays team B, P(A wins) + P(B wins) = 1 (symmetric)."""
    pool, rosters, _, _, full_schedule = _build_league()
    team_names = sorted(rosters.keys())
    matrix = _compute_per_week_per_team_p_win(
        full_league_schedule=full_schedule,
        all_team_rosters=rosters,
        player_pool=pool,
        config=LeagueConfig(),
        variance_cv={cat: 0.15 for cat in LeagueConfig().all_categories},
        weeks_to_include=list(range(9, 25)),
        team_names=team_names,
    )
    team_idx = {t: i for i, t in enumerate(team_names)}
    weeks = list(range(9, 25))
    for w_idx, wk in enumerate(weeks):
        for a, b in full_schedule.get(wk, []):
            if a in team_idx and b in team_idx:
                p_a = matrix[team_idx[a], w_idx]
                p_b = matrix[team_idx[b], w_idx]
                assert abs(p_a + p_b - 1.0) < 1e-9, (
                    f"P(A wins) + P(B wins) != 1 for {a} vs {b} week {wk}: {p_a} + {p_b} = {p_a + p_b}"
                )


# ── simulate_playoff_outcomes integration ────────────────────────────


def test_simulate_returns_full_sched_method_when_provided() -> None:
    """When full_league_schedule provided, method tag reflects it."""
    pool, rosters, wins, user_sched, full_sched = _build_league()
    result = simulate_playoff_outcomes(
        user_roster_ids=rosters["T5"],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=user_sched,
        current_wins=wins,
        player_pool=pool,
        n_sims=500,
        full_league_schedule=full_sched,
    )
    # startswith: the path-selection tag may carry a "-corr" suffix when the
    # TE-E2 copula-correlated category path is active (now the default).
    assert result["method"].startswith("hickey-centric-full-sched-opp")


def test_simulate_falls_back_to_binomial_without_full_sched() -> None:
    """When full_league_schedule is None, method tag indicates Binomial fallback."""
    pool, rosters, wins, user_sched, _ = _build_league()
    result = simulate_playoff_outcomes(
        user_roster_ids=rosters["T5"],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=user_sched,
        current_wins=wins,
        player_pool=pool,
        n_sims=500,
        full_league_schedule=None,  # explicit None
    )
    assert result["method"].startswith("hickey-centric-binomial-opp")


def test_simulate_trade_delta_propagates_full_sched() -> None:
    """simulate_trade_playoff_delta passes full_league_schedule through."""
    pool, rosters, wins, user_sched, full_sched = _build_league()
    result = simulate_trade_playoff_delta(
        before_roster_ids=rosters["T5"],
        after_roster_ids=rosters["T5"][:2] + [rosters["T8"][0]],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=user_sched,
        current_wins=wins,
        player_pool=pool,
        n_sims=500,
        full_league_schedule=full_sched,
    )
    assert result["before"]["method"].startswith("hickey-centric-full-sched-opp")
    assert result["after"]["method"].startswith("hickey-centric-full-sched-opp")


# ── UI wiring ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def page_source() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


def test_page_loads_full_league_schedule(page_source: str) -> None:
    """Trade Analyzer must load full schedule via load_league_schedule_full."""
    assert "load_league_schedule_full" in page_source, (
        "Trade Analyzer must call load_league_schedule_full() so playoff sim "
        "uses per-team-per-week opponent matchups, not Binomial approx"
    )


def test_page_passes_full_league_schedule_to_evaluate(page_source: str) -> None:
    """The full_league_schedule kwarg must be passed to evaluate_trade."""
    assert "full_league_schedule=" in page_source, (
        "evaluate_trade() must receive full_league_schedule= for accurate non-user-team simulation"
    )


def test_playoff_sim_caps_schedule_to_weeks_remaining() -> None:
    """#S1365 (2026-06-16): evaluate_trade must cap the playoff sim to the
    REMAINING weeks, not the full season schedule.

    load_league_schedule() returns the user's ENTIRE season (~24 matchup weeks).
    Passing it through unfiltered set n_weeks_remaining = the whole season,
    projecting impossible 30+-win finals (live: a team at 11 wins projected to
    32.6) and burying a competitive user at ~0% playoff odds. The fix caps the
    schedule to the last `weeks_remaining` weeks.
    """
    from src.engine.output.trade_evaluator import evaluate_trade

    pool, rosters, wins, user_sched, full_sched = _build_league()
    assert len(user_sched) == 16, "fixture sanity: user schedule spans weeks 9..24"

    result = evaluate_trade(
        giving_ids=[rosters["T5"][0]],
        receiving_ids=[rosters["T8"][0]],
        user_roster_ids=rosters["T5"],
        player_pool=pool,
        user_team_name="T5",
        weeks_remaining=6,  # far fewer than the 16-week schedule
        enable_playoff_sim=True,
        enable_weekly_matrix=False,
        weekly_schedule=user_sched,
        league_rosters=rosters,
        current_wins=wins,
        playoff_n_sims=200,
        full_league_schedule=full_sched,
    )
    ps = result.get("playoff_sim")
    assert ps is not None, "playoff sim should have run (enable_playoff_sim=True)"
    assert ps["before"]["n_weeks_remaining"] == 6, (
        "playoff sim must simulate only the last `weeks_remaining`=6 weeks; got "
        f"{ps['before']['n_weeks_remaining']} (the full season schedule leaked through)"
    )
