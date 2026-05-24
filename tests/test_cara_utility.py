"""Structural-invariant guard: CARA mean-CVaR utility on Δchamp_prob.

Phase 2 (2026-05-24) — per Enhanced Trade Engine report Section B.9 + Q(a),
the engine's recommended primary decision metric is a risk-adjusted utility
on per-sim championship-probability deltas:

    CARA = E[Δchamp] − (λ/2) × Var[Δchamp]   with λ = 0.15
    CVaR₂₀ = mean(worst 20% of per-sim Δchamp values)

These are computed from per-sim outcomes (paired-MC discipline keeps
before/after on the same RNG seed → most per-sim deltas are 0, deviation
sims contribute the variance term).

Contract locked here:
  - simulate_playoff_outcomes exposes per-sim champ_outcomes + playoff_outcomes
  - simulate_trade_playoff_delta computes cara_utility, var_champ,
    cvar20_champ, mean_playoff_delta_per_sim
  - Lambda sourced from LeagueConfig.risk_aversion (default 0.15)
  - Identical rosters → variance 0, CARA == 0, CVaR_20 == 0
  - Trade Analyzer page surfaces CARA + CVaR alongside Δchamp_prob
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.engine.output.playoff_sim import simulate_playoff_outcomes, simulate_trade_playoff_delta
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


def _build_league() -> tuple[pd.DataFrame, dict[str, list[int]], dict[str, int], dict[int, str]]:
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
    schedule = {w: f"T{(w % 11) + 1}" for w in range(9, 25)}
    return pool, rosters, current_wins, schedule


# ── simulate_playoff_outcomes exposes per-sim outcomes ───────────────


def test_simulate_playoff_outcomes_returns_per_sim_arrays() -> None:
    """champ_outcomes + playoff_outcomes must be present as numpy arrays."""
    pool, rosters, wins, schedule = _build_league()
    result = simulate_playoff_outcomes(
        user_roster_ids=rosters["T5"],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=1000,
    )
    assert "champ_outcomes" in result, "Missing per-sim champ_outcomes (needed for CARA)"
    assert "playoff_outcomes" in result, "Missing per-sim playoff_outcomes"
    champ = result["champ_outcomes"]
    playoff = result["playoff_outcomes"]
    assert len(champ) == 1000, f"champ_outcomes length must equal n_sims, got {len(champ)}"
    assert len(playoff) == 1000
    # Values must be 0/1 Bernoulli outcomes
    assert set(np.unique(champ).tolist()) <= {0, 1}, "champ_outcomes must be 0/1 Bernoulli"
    assert set(np.unique(playoff).tolist()) <= {0, 1}


# ── simulate_trade_playoff_delta produces CARA + CVaR ────────────────


def test_delta_returns_cara_and_cvar20_keys() -> None:
    """Required keys for the report Section B.9 utility."""
    pool, rosters, wins, schedule = _build_league()
    result = simulate_trade_playoff_delta(
        before_roster_ids=rosters["T5"],
        after_roster_ids=rosters["T5"][:2] + [rosters["T8"][0]],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=1000,
    )
    for key in ("cara_utility", "var_champ", "cvar20_champ", "lambda_risk_aversion"):
        assert key in result, f"Missing required CARA key {key!r}"


def test_identical_rosters_have_zero_variance_and_cara() -> None:
    """Paired-MC + identical rosters → per-sim Δ is 0 every sim, so
    variance = 0, CARA = 0, CVaR_20 = 0."""
    pool, rosters, wins, schedule = _build_league()
    result = simulate_trade_playoff_delta(
        before_roster_ids=rosters["T5"],
        after_roster_ids=rosters["T5"],  # SAME
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=500,
    )
    assert result["var_champ"] == 0.0
    assert result["cara_utility"] == 0.0
    assert result["cvar20_champ"] == 0.0
    assert result["delta_champ_prob"] == 0.0


def test_cara_formula_matches_definition() -> None:
    """CARA = E[Δ] − (λ/2) × Var[Δ]. Verify the algebra holds with the
    reported lambda, variance, and CARA values."""
    pool, rosters, wins, schedule = _build_league()
    result = simulate_trade_playoff_delta(
        before_roster_ids=rosters["T5"],
        after_roster_ids=rosters["T5"][:2] + [rosters["T11"][0]],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=1500,
    )
    e_delta = result["delta_champ_prob"]
    var = result["var_champ"]
    lam = result["lambda_risk_aversion"]
    cara = result["cara_utility"]
    expected = e_delta - (lam / 2.0) * var
    # Allow tiny float tolerance from round()
    assert abs(cara - expected) < 5e-4, (
        f"CARA formula broken: got {cara}, expected {expected:.6f} = {e_delta} − ({lam}/2) × {var}"
    )


def test_lambda_sourced_from_league_config() -> None:
    """λ must equal LeagueConfig.risk_aversion (default 0.15 per report B.9)."""
    pool, rosters, wins, schedule = _build_league()
    cfg = LeagueConfig()
    result = simulate_trade_playoff_delta(
        before_roster_ids=rosters["T5"],
        after_roster_ids=rosters["T5"][:2] + [rosters["T9"][0]],
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        config=cfg,
        n_sims=500,
    )
    assert result["lambda_risk_aversion"] == pytest.approx(cfg.risk_aversion)
    assert result["lambda_risk_aversion"] == 0.15  # current report calibration


def test_cvar20_at_most_zero_or_negative_when_no_upside_sims() -> None:
    """CVaR_20 looks at worst 20% — for Bernoulli deltas in {-1, 0, +1}
    where 0 dominates, the worst 20% sit at ≤ 0."""
    pool, rosters, wins, schedule = _build_league()
    result = simulate_trade_playoff_delta(
        before_roster_ids=rosters["T5"],
        after_roster_ids=rosters["T5"][:2] + [rosters["T1"][0]],  # downgrade
        user_team_name="T5",
        all_team_rosters=rosters,
        user_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        n_sims=1000,
    )
    # Worst-20% mean must be ≤ E[Δ] (CVaR ≤ mean is always true for tail).
    # For 0/±1 outcomes the worst-20% averaging tail is always ≤ 0.
    assert result["cvar20_champ"] <= max(result["delta_champ_prob"], 0.0) + 1e-9


# ── UI wiring guard ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trade_analyzer_source() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


def test_page_renders_cara_utility(trade_analyzer_source: str) -> None:
    """Trade Analyzer must surface CARA utility metric."""
    assert "cara_utility" in trade_analyzer_source, (
        "Trade Analyzer page must render cara_utility — the report's Section B.9 risk-adjusted decision metric"
    )


def test_page_renders_cvar20(trade_analyzer_source: str) -> None:
    """Trade Analyzer must surface CVaR_20 metric."""
    assert "cvar20_champ" in trade_analyzer_source, (
        "Trade Analyzer page must render cvar20_champ — downside-risk diagnostic"
    )
