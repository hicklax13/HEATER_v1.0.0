"""Matchup-impact guards: streaming a probable must be valued against the
LIVE matchup, with-vs-without (owner request 2026-06-10).

``compute_matchup_impact`` compares the current matchup's category win
probabilities (canonical ``estimate_h2h_win_probability`` — the same engine
the Lineup Optimizer uses) before and after adding one streamed start's
expected line. Inverse cats must move the right way, the no-op line must
produce exact-zero deltas (paired-MC discipline: same seed both arms), and
missing matchup data must yield None — never a guess.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.optimizer.constants_registry import CONSTANTS_REGISTRY
from src.optimizer.stream_analyzer import compute_matchup_impact

_PAGE = Path(__file__).resolve().parents[1] / "pages" / "4_Pitcher_Streaming.py"


def _totals():
    my = {
        "r": 20.0,
        "hr": 6.0,
        "rbi": 19.0,
        "sb": 3.0,
        "avg": 0.262,
        "obp": 0.330,
        "w": 2.0,
        "l": 2.0,
        "sv": 3.0,
        "k": 38.0,
        "era": 4.10,
        "whip": 1.28,
    }
    opp = {
        "r": 21.0,
        "hr": 7.0,
        "rbi": 20.0,
        "sb": 4.0,
        "avg": 0.258,
        "obp": 0.325,
        "w": 3.0,
        "l": 2.0,
        "sv": 2.0,
        "k": 44.0,
        "era": 3.80,
        "whip": 1.21,
    }
    return my, opp


_GEM = {"ip": 7.0, "k": 9.0, "er": 1.0, "win_prob": 0.62}
_BLOWUP = {"ip": 3.0, "k": 1.0, "er": 7.0, "win_prob": 0.10}
_NOOP = {"ip": 0.0, "k": 0.0, "er": 0.0, "win_prob": 0.0}


def test_loss_share_constant_registered():
    entry = CONSTANTS_REGISTRY["stream_loss_decision_share"]
    assert entry.lower_bound <= entry.value <= entry.upper_bound
    assert entry.citation.strip()
    assert "stream_analyzer" in entry.module


def test_gem_start_helps_k_w_and_expected_wins():
    my, opp = _totals()
    impact = compute_matchup_impact(my, opp, _GEM, pitcher_whip=1.05, team_ip=40.0)
    assert impact is not None
    assert impact["per_cat"]["k"]["delta"] > 0
    assert impact["per_cat"]["w"]["delta"] > 0
    assert impact["per_cat"]["era"]["delta"] > 0, (
        "a 1-ER/7-IP start lowers team ERA — the ERA win probability must RISE (inverse-cat sign discipline)"
    )
    assert impact["expected_wins_delta"] > 0
    # Hitting cats untouched by a pitcher stream.
    assert impact["per_cat"]["hr"]["delta"] == pytest.approx(0.0)


def test_blowup_start_hurts_rates():
    my, opp = _totals()
    impact = compute_matchup_impact(my, opp, _BLOWUP, pitcher_whip=1.90, team_ip=40.0)
    assert impact is not None
    assert impact["per_cat"]["era"]["delta"] < 0
    assert impact["per_cat"]["whip"]["delta"] < 0
    assert impact["per_cat"]["l"]["delta"] < 0, "a likely loss raises team L — the L win probability must FALL"


def test_noop_line_is_exact_zero_paired():
    """Identical before/after totals ⇒ every delta exactly 0.0 — locks the
    paired-seed discipline (different seeds would leave MC jitter)."""
    my, opp = _totals()
    impact = compute_matchup_impact(my, opp, _NOOP, pitcher_whip=0.0, team_ip=40.0)
    assert impact is not None
    assert impact["expected_wins_delta"] == pytest.approx(0.0, abs=1e-12)
    assert impact["overall_win_prob_delta"] == pytest.approx(0.0, abs=1e-12)
    for cat in impact["per_cat"].values():
        assert cat["delta"] == pytest.approx(0.0, abs=1e-12)


def test_missing_matchup_returns_none():
    assert compute_matchup_impact({}, {}, _GEM, pitcher_whip=1.05) is None
    my, _ = _totals()
    assert compute_matchup_impact(my, {}, _GEM, pitcher_whip=1.05) is None


def test_two_starts_scale_counting_impact():
    my, opp = _totals()
    one = compute_matchup_impact(my, opp, _GEM, pitcher_whip=1.05, team_ip=40.0, num_starts=1)
    two = compute_matchup_impact(my, opp, _GEM, pitcher_whip=1.05, team_ip=40.0, num_starts=2)
    assert two["per_cat"]["k"]["after"] > one["per_cat"]["k"]["after"]


def test_page_wires_matchup_impact():
    src = _PAGE.read_text(encoding="utf-8")
    assert "compute_matchup_impact" in src, (
        "the Stream Finder must surface the with-vs-without matchup impact (owner request 2026-06-10)"
    )
