"""Inverse-stat guards for the Stream Analyzer (the FA PR #99 lesson).

ERA/WHIP/L must enter the Stream Score through canonical SGP paths
(compute_streaming_value → SGPCalculator semantics) so their contribution is
NEGATIVE for bad rates. The earlier FA-engine inline formula
``value += (stat / denom) * weight`` scored a default-bad unknown pitcher
(era=9.0, whip=2.0) at +38.8 when it should have been -38.8 — this module
locks the stream engine against the same class of bug, structurally and
behaviorally.
"""

from __future__ import annotations

import ast
from pathlib import Path

from src.optimizer.stream_analyzer import score_stream_candidate
from src.valuation import LeagueConfig

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "optimizer" / "stream_analyzer.py"


def _start():
    return {
        "game_date": "2026-06-12",
        "opponent": "CHW",
        "is_home": True,
        "venue": "SEA",
        "park_factor": 1.00,
        "weather": {},
        "confidence": "HIGH",
        "num_starts": 1,
    }


def _opp():
    return {
        "wrc_plus": 100.0,
        "k_pct": 22.0,
        "bb_pct": 8.0,
        "iso": None,
        "l14_wrc_plus": None,
        "split_source": "overall",
    }


def test_default_bad_pitcher_sgp_component_negative():
    """era=9.0 / whip=2.0 (the unknown-pitcher default line) must score negative."""
    phantom = {
        "player_id": 99,
        "player_name": "Unknown Phantom",
        "team": "SEA",
        "throws": "R",
        "era": 9.0,
        "whip": 2.0,
        "k": 40.0,
        "w": 2.0,
        "ip": 60.0,
        "ip_per_start": 5.0,
    }
    res = score_stream_candidate(phantom, _start(), _opp(), LeagueConfig())
    assert res["components"]["sgp"] < 0.0, (
        "bad-rate pitcher must have NEGATIVE SGP component — inverse stats "
        "(ERA/WHIP) are contributing with the wrong sign"
    )
    assert res["net_sgp"] < 0.0


def test_ace_sgp_component_positive():
    ace = {
        "player_id": 7,
        "player_name": "Ace",
        "team": "SEA",
        "throws": "R",
        "era": 2.60,
        "whip": 0.98,
        "k": 220.0,
        "w": 16.0,
        "ip": 190.0,
        "ip_per_start": 6.2,
    }
    res = score_stream_candidate(ace, _start(), _opp(), LeagueConfig())
    assert res["components"]["sgp"] > 0.0
    assert res["net_sgp"] > 0.0
    assert res["stream_score"] > 50.0


def test_ace_outscores_phantom():
    ace = {
        "player_name": "Ace",
        "team": "SEA",
        "era": 2.60,
        "whip": 0.98,
        "k": 220.0,
        "w": 16.0,
        "ip": 190.0,
    }
    phantom = {
        "player_name": "Unknown Phantom",
        "team": "SEA",
        "era": 9.0,
        "whip": 2.0,
        "k": 40.0,
        "w": 2.0,
        "ip": 60.0,
    }
    cfg = LeagueConfig()
    assert (
        score_stream_candidate(ace, _start(), _opp(), cfg)["stream_score"]
        > score_stream_candidate(phantom, _start(), _opp(), cfg)["stream_score"]
    )


def test_no_local_sgp_denominator_table():
    """The module must not regrow its own MODULE-LEVEL denominator table —
    SGP math lives in streaming.py / SGPCalculator, which own the
    inverse-stat signs. (Function-local variables passing config-derived
    denominators through are fine; a dict-literal table is not.)"""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Name) and "DENOM" in tgt.id.upper():
                    raise AssertionError(
                        f"stream_analyzer.py defines {tgt.id} — SGP denominators "
                        "must come from LeagueConfig/streaming, never a local table"
                    )


def test_imports_canonical_streaming_value():
    src = _MODULE_PATH.read_text(encoding="utf-8")
    assert "compute_streaming_value" in src, (
        "stream_analyzer must delegate SGP math to streaming.compute_streaming_value"
    )


def test_no_hardcoded_category_collections():
    """No literal list/set/tuple containing the UPPERCASE category names
    'ERA' and 'WHIP' together — category lists derive from LeagueConfig
    (structural-invariant house rule). Lowercase stat-key tuples (dict
    field extraction like ("era", "whip")) are not category lists."""
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            values = {el.value for el in node.elts if isinstance(el, ast.Constant) and isinstance(el.value, str)}
            assert not ({"ERA", "WHIP"} <= values), (
                "hardcoded category collection found in stream_analyzer.py — derive categories from LeagueConfig"
            )
