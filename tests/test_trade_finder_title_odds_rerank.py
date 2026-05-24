"""Structural-invariant guard: Trade Finder title-odds-aware re-rank.

Phase 2 (2026-05-24) — per Enhanced Trade Engine report Q(a), the engine's
PRIMARY objective is Δ-championship-probability. Trade Finder's fast
composite_score (SGP + ADP + acceptance + cat-fit) is great for surfacing
candidates but doesn't directly measure title impact. This module's
rerank_by_title_odds() function deep-evaluates the top N candidates with
simulate_trade_playoff_delta and re-ranks by a blend of Δchamp_prob (60%)
and composite_score (40%).

This guard locks the contract:
  - rerank_by_title_odds enriches top-N candidates with delta_playoff_prob,
    delta_champ_prob, title_odds_score, ranked_by keys
  - top-N are re-sorted by title_odds_score
  - remaining candidates keep original composite-score order, appended
  - find_trade_opportunities accepts enable_title_odds_rerank flag and
    its supporting kwargs (weekly_schedule, current_wins, etc.)
  - Trade Finder page wires the toggle + columns
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.trade_finder import find_trade_opportunities, rerank_by_title_odds

PAGE_PATH = Path(__file__).resolve().parent.parent / "pages" / "12_Trade_Finder.py"


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
    """Build a minimal 12-team mini-league + schedule."""
    rows = []
    rosters: dict[str, list[int]] = {}
    current_wins: dict[str, int] = {}
    pid = 1
    for team_idx in range(12):
        team_name = f"Team_{team_idx}"
        rosters[team_name] = []
        current_wins[team_name] = 4 + (team_idx // 3)
        for _ in range(3):
            rows.append(_make_hitter(pid, f"P{pid}", hr=15 + team_idx))
            rosters[team_name].append(pid)
            pid += 1
    pool = pd.DataFrame(rows)
    schedule = {
        wk: f"Team_{(wk % 11) + 1}"  # rotate non-user teams
        for wk in range(9, 25)
    }
    return pool, rosters, current_wins, schedule


# ── Unit tests on rerank_by_title_odds ───────────────────────────────


def test_rerank_returns_same_length() -> None:
    """Re-rank preserves all input candidates, just changes order of top N."""
    pool, rosters, wins, schedule = _build_league()
    # Build 3 synthetic candidates
    candidates = [
        {
            "giving_ids": [1],
            "receiving_ids": [4],
            "composite_score": 0.7,
            "giving_names": ["P1"],
            "receiving_names": ["P4"],
        },
        {
            "giving_ids": [2],
            "receiving_ids": [5],
            "composite_score": 0.5,
            "giving_names": ["P2"],
            "receiving_names": ["P5"],
        },
        {
            "giving_ids": [3],
            "receiving_ids": [6],
            "composite_score": 0.3,
            "giving_names": ["P3"],
            "receiving_names": ["P6"],
        },
    ]
    result = rerank_by_title_odds(
        candidates=candidates,
        user_team_name="Team_0",
        user_roster_ids=rosters["Team_0"],
        all_team_rosters=rosters,
        weekly_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        top_n=2,
        n_sims=500,  # small for test speed
    )
    assert len(result) == 3, "Re-rank must preserve total candidate count"


def test_rerank_enriches_top_n_with_title_odds_keys() -> None:
    """Top-N enriched with delta_playoff_prob, delta_champ_prob, title_odds_score, ranked_by."""
    pool, rosters, wins, schedule = _build_league()
    candidates = [
        {
            "giving_ids": [1],
            "receiving_ids": [4],
            "composite_score": 0.7,
            "giving_names": ["P1"],
            "receiving_names": ["P4"],
        },
        {
            "giving_ids": [2],
            "receiving_ids": [5],
            "composite_score": 0.5,
            "giving_names": ["P2"],
            "receiving_names": ["P5"],
        },
        {
            "giving_ids": [3],
            "receiving_ids": [6],
            "composite_score": 0.3,
            "giving_names": ["P3"],
            "receiving_names": ["P6"],
        },
    ]
    result = rerank_by_title_odds(
        candidates=candidates,
        user_team_name="Team_0",
        user_roster_ids=rosters["Team_0"],
        all_team_rosters=rosters,
        weekly_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        top_n=2,
        n_sims=500,
    )
    # First 2 (top_n) should have title-odds keys
    for i in range(2):
        assert "delta_playoff_prob" in result[i], f"Top-{i + 1} missing delta_playoff_prob"
        assert "delta_champ_prob" in result[i], f"Top-{i + 1} missing delta_champ_prob"
        assert "title_odds_score" in result[i], f"Top-{i + 1} missing title_odds_score"
        assert result[i].get("ranked_by") == "title_odds_blend"
    # 3rd should remain composite-only
    assert result[2].get("ranked_by") == "composite_score_only"


def test_rerank_top_n_sorted_by_title_odds_score() -> None:
    """After re-rank, the top N are in title_odds_score descending order."""
    pool, rosters, wins, schedule = _build_league()
    candidates = [
        {
            "giving_ids": [1],
            "receiving_ids": [4],
            "composite_score": 0.7,
            "giving_names": ["P1"],
            "receiving_names": ["P4"],
        },
        {
            "giving_ids": [2],
            "receiving_ids": [5],
            "composite_score": 0.5,
            "giving_names": ["P2"],
            "receiving_names": ["P5"],
        },
        {
            "giving_ids": [3],
            "receiving_ids": [6],
            "composite_score": 0.3,
            "giving_names": ["P3"],
            "receiving_names": ["P6"],
        },
    ]
    result = rerank_by_title_odds(
        candidates=candidates,
        user_team_name="Team_0",
        user_roster_ids=rosters["Team_0"],
        all_team_rosters=rosters,
        weekly_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        top_n=3,  # all candidates
        n_sims=500,
    )
    # All 3 should now be in title_odds_score order (descending)
    scores = [r["title_odds_score"] for r in result]
    assert scores == sorted(scores, reverse=True), f"Top-N not sorted by title_odds_score descending: {scores}"


def test_rerank_missing_schedule_returns_unchanged() -> None:
    """If schedule/current_wins missing, return candidates unchanged."""
    pool, rosters, wins, _ = _build_league()
    candidates = [
        {
            "giving_ids": [1],
            "receiving_ids": [4],
            "composite_score": 0.7,
            "giving_names": ["P1"],
            "receiving_names": ["P4"],
        },
    ]
    # No schedule supplied
    result = rerank_by_title_odds(
        candidates=candidates,
        user_team_name="Team_0",
        user_roster_ids=rosters["Team_0"],
        all_team_rosters=rosters,
        weekly_schedule={},  # empty
        current_wins=wins,
        player_pool=pool,
        top_n=1,
        n_sims=500,
    )
    assert result == candidates, "Empty schedule must return candidates unchanged"


def test_rerank_empty_candidates_no_crash() -> None:
    """Edge case: empty input list."""
    pool, rosters, wins, schedule = _build_league()
    result = rerank_by_title_odds(
        candidates=[],
        user_team_name="Team_0",
        user_roster_ids=rosters["Team_0"],
        all_team_rosters=rosters,
        weekly_schedule=schedule,
        current_wins=wins,
        player_pool=pool,
        top_n=15,
        n_sims=500,
    )
    assert result == []


# ── find_trade_opportunities integration ─────────────────────────────


def test_find_trade_opportunities_accepts_title_odds_flag() -> None:
    """The new flag + supporting kwargs are accepted (no TypeError)."""
    pool, rosters, wins, schedule = _build_league()
    # Won't actually find trades with this minimal setup; just check the
    # flag plumbing doesn't crash
    try:
        find_trade_opportunities(
            user_roster_ids=rosters["Team_0"],
            player_pool=pool,
            user_team_name="Team_0",
            league_rosters=rosters,
            all_team_totals={t: {"R": 100} for t in rosters},
            max_results=5,
            top_partners=3,
            enable_title_odds_rerank=True,
            weekly_schedule=schedule,
            current_wins=wins,
            title_odds_top_n=5,
            title_odds_n_sims=200,
        )
    except TypeError as e:
        pytest.fail(f"find_trade_opportunities rejects new kwargs: {e}")


# ── Trade Finder page wiring guard ───────────────────────────────────


@pytest.fixture(scope="module")
def page_source() -> str:
    return PAGE_PATH.read_text(encoding="utf-8")


def test_page_wires_title_odds_flag(page_source: str) -> None:
    """Page must opt into title-odds re-rank via enable_title_odds_rerank=."""
    assert "enable_title_odds_rerank" in page_source, (
        "Trade Finder page must wire enable_title_odds_rerank= to find_trade_opportunities"
    )


def test_page_loads_weekly_schedule_for_rerank(page_source: str) -> None:
    """Page must source weekly_schedule for the re-rank to work."""
    assert "load_league_schedule" in page_source, (
        "Trade Finder page must call load_league_schedule() for the title-odds re-rank"
    )


def test_page_loads_current_wins_for_rerank(page_source: str) -> None:
    """Page must build current_wins from standings."""
    assert "WINS" in page_source, "Trade Finder page must build current_wins from standings WINS rows"


def test_page_renders_title_odds_columns(page_source: str) -> None:
    """Page must render ΔPlayoff% and ΔChamp% columns in the trade table."""
    assert re.search(r"ΔPlayoff%|delta_playoff_prob", page_source), (
        "Trade Finder page must render delta_playoff_prob column"
    )
    assert re.search(r"ΔChamp%|delta_champ_prob", page_source), "Trade Finder page must render delta_champ_prob column"


def test_page_has_user_toggle_for_title_odds(page_source: str) -> None:
    """Users can opt out of the slower re-rank via a toggle."""
    assert re.search(r"toggle\(|checkbox\(", page_source), (
        "Trade Finder page should have a user toggle to enable/disable title-odds re-rank"
    )
