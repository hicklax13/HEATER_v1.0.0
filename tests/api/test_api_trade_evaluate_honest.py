"""TASK 1+2 — honest YTD + need-weighted Build-tab trade valuation, and Compare
reading ACTUAL 2026 YTD stats.

These are DB-FREE: they monkeypatch the pool loader / Yahoo service / the engine
and (for the headline math) the shared Finder per-cat SGP resolver so grade
boundaries are deterministic without the 26 MB live DB."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from api.contracts.compare import CompareResponse
from api.contracts.trade import TradeEvaluationResponse
from api.services import compare_service as compare_mod
from api.services import trade_service as trade_mod
from api.services.compare_service import CompareService
from api.services.trade_service import TradeService

# ── fixtures / fakes ──────────────────────────────────────────────────────────


class _FakeYDS:
    """Stand-in for the Yahoo data service. Returns the rosters frame it was given."""

    def __init__(self, rosters):
        self._rosters = rosters

    def get_rosters(self):
        return self._rosters


def _install_trade_stubs(monkeypatch, *, per_cat_map, pool=None, rosters=None, all_team_totals=None, evaluate_fn=None):
    """Wire TradeService.evaluate's collaborators so the honest headline math is
    deterministic. ``per_cat_map`` = {player_id: {cat: sgp}} drives the YTD value;
    a 0-arg ``evaluate_fn`` supplies the SUPPLEMENTARY engine result (or raises)."""
    pool = pool if pool is not None else pd.DataFrame({"player_id": list(per_cat_map.keys())})

    monkeypatch.setattr(trade_mod, "load_player_pool", lambda: pool, raising=False)
    # The service imports these lazily *inside* evaluate via ``from src... import``.
    import src.database
    import src.standings_utils
    import src.yahoo_data_service

    monkeypatch.setattr(src.database, "load_player_pool", lambda: pool, raising=False)
    monkeypatch.setattr(src.yahoo_data_service, "get_yahoo_data_service", lambda: _FakeYDS(rosters), raising=False)
    monkeypatch.setattr(
        src.standings_utils,
        "get_all_team_totals",
        lambda **kw: (all_team_totals or {}),
        raising=False,
    )

    # The shared Finder resolver — patch the staticmethod so headline cat_net is exact.
    def _fake_lookup(_pool):
        def per_cat(pid):
            return dict(per_cat_map.get(int(pid), {}))

        return per_cat

    from api.services.trade_finder_service import TradeFinderService

    monkeypatch.setattr(TradeFinderService, "_player_sgp_lookup", staticmethod(_fake_lookup))

    # Supplementary engine path.
    import src.engine.output.trade_evaluator as te

    if evaluate_fn is None:
        evaluate_fn = lambda **kw: {}  # noqa: E731 — engine adds nothing extra
    monkeypatch.setattr(te, "evaluate_trade", lambda **kw: evaluate_fn(**kw), raising=False)


# ── TASK 1: symmetric grade boundaries ────────────────────────────────────────


@pytest.mark.parametrize(
    "gain, grade, verdict",
    [
        (3.5, "A+", "You win"),
        (2.5, "A", "You win"),
        (1.3, "B+", "You win"),
        (0.7, "B", "You win"),
        (0.0, "C", "Even / fair value"),
        (-0.4, "C", "Even / fair value"),
        (-1.0, "D", "You lose"),
        (-2.5, "F", "You lose"),
    ],
)
def test_symmetric_grade_boundaries(monkeypatch, gain, grade, verdict):
    """Grade ladder must handle NEGATIVE gains (proposed trades can lose). Drive the
    weighted gain by making the single received player's HR SGP exactly ``gain`` and
    giving nothing (no slot credit, need weights default 1.0 → weighted == cat_net)."""
    _install_trade_stubs(
        monkeypatch,
        per_cat_map={1: {}, 2: {"HR": gain}},
        rosters=pd.DataFrame({"team_name": [], "player_id": []}),
    )
    resp = TradeService().evaluate("Team Hickey", giving_ids=[1], receiving_ids=[2])
    assert isinstance(resp, TradeEvaluationResponse)
    assert resp.grade == grade
    assert resp.verdict == verdict
    assert resp.surplus_sgp == pytest.approx(round(gain, 3))


def test_lopsided_loss_grades_F_you_lose(monkeypatch):
    """Give two studs (5 SGP each), get one scrub (1 SGP). cat_net = 1 - 10 = -9,
    slot_credit = max(0, 2-1)*1.0 = +1 → weighted_gain = -8 → F / You lose."""
    _install_trade_stubs(
        monkeypatch,
        per_cat_map={10: {"HR": 5.0}, 11: {"R": 5.0}, 20: {"HR": 1.0}},
        rosters=pd.DataFrame({"team_name": [], "player_id": []}),
    )
    resp = TradeService().evaluate("Team Hickey", giving_ids=[10, 11], receiving_ids=[20])
    assert resp.grade == "F"
    assert resp.verdict == "You lose"
    assert resp.surplus_sgp < 0


def test_upgrade_grades_A_you_win(monkeypatch):
    """Give a scrub (1 SGP), get a stud (5 SGP). cat_net = +4, no slot credit
    (1-for-1) → A / You win."""
    _install_trade_stubs(
        monkeypatch,
        per_cat_map={20: {"HR": 1.0}, 10: {"HR": 5.0}},
        rosters=pd.DataFrame({"team_name": [], "player_id": []}),
    )
    resp = TradeService().evaluate("Team Hickey", giving_ids=[20], receiving_ids=[10])
    assert resp.grade in {"A", "A+"}
    assert resp.verdict == "You win"
    assert resp.surplus_sgp == pytest.approx(4.0)


def test_category_impacts_correct_signs(monkeypatch):
    """category_impacts = cat_net (Σreceiving − Σgiving). Receive HR, give SB →
    HR positive, SB negative; magnitudes finite + >= 0.01."""
    _install_trade_stubs(
        monkeypatch,
        per_cat_map={1: {"SB": 3.0}, 2: {"HR": 4.0}},
        rosters=pd.DataFrame({"team_name": [], "player_id": []}),
    )
    resp = TradeService().evaluate("Team Hickey", giving_ids=[1], receiving_ids=[2])
    impacts = {ci.cat: ci.delta for ci in resp.category_impacts}
    assert impacts["HR"] == pytest.approx(4.0)
    assert impacts["SB"] == pytest.approx(-3.0)
    for ci in resp.category_impacts:
        assert math.isfinite(ci.delta)
        assert abs(ci.delta) >= 0.01


def test_confidence_scales_with_magnitude(monkeypatch):
    """confidence_pct = min(95, 50 + |gain|*15). A near-even trade → ~50; a big win → 95."""
    _install_trade_stubs(
        monkeypatch,
        per_cat_map={1: {}, 2: {"HR": 0.0}},
        rosters=pd.DataFrame({"team_name": [], "player_id": []}),
    )
    even = TradeService().evaluate("Team Hickey", giving_ids=[1], receiving_ids=[2])
    assert even.confidence_pct == pytest.approx(50.0)

    _install_trade_stubs(
        monkeypatch,
        per_cat_map={1: {}, 2: {"HR": 10.0}},
        rosters=pd.DataFrame({"team_name": [], "player_id": []}),
    )
    big = TradeService().evaluate("Team Hickey", giving_ids=[1], receiving_ids=[2])
    assert big.confidence_pct == pytest.approx(95.0)  # clamped


def test_need_weighting_applied(monkeypatch):
    """A category the user is WEAK in gets weight > 1.0 → amplifies a gain in it.
    Build totals where the user is a dead-last HR OUTLIER so the weight clamps to 1.6
    (a tight strong cluster + a far-below user → big negative z → the +0.30*|z| clamp)."""
    # 11 strong teams clustered at ~30 HR, user far below at 0 → HR weight clamps to 1.6.
    totals = {f"T{i}": {"HR": 30.0 + (i % 3)} for i in range(1, 12)}
    totals["Team Hickey"] = {"HR": 0.0}
    _install_trade_stubs(
        monkeypatch,
        per_cat_map={1: {}, 2: {"HR": 2.0}},
        rosters=pd.DataFrame({"team_name": ["Team Hickey"], "player_id": [1]}),
        all_team_totals=totals,
    )
    resp = TradeService().evaluate("Team Hickey", giving_ids=[1], receiving_ids=[2])
    # weighted_gain = 2.0 * 1.6 = 3.2 → A+, vs the context-free 2.0 (A).
    assert resp.surplus_sgp > 2.0
    assert resp.grade == "A+"


# ── TASK 1: engine failure must not sink the honest headline ───────────────────


def test_engine_raise_still_returns_honest_headline(monkeypatch):
    """When the supplementary evaluate_trade raises, evaluate() still returns the
    honest YTD/need-weighted headline (NOT the default 'Could not evaluate')."""

    def _boom(**kw):
        raise RuntimeError("engine exploded")

    _install_trade_stubs(
        monkeypatch,
        per_cat_map={1: {"HR": 1.0}, 2: {"HR": 5.0}},
        rosters=pd.DataFrame({"team_name": [], "player_id": []}),
        evaluate_fn=_boom,
    )
    resp = TradeService().evaluate("Team Hickey", giving_ids=[1], receiving_ids=[2])
    assert resp.verdict == "You win"  # honest headline survived
    assert resp.grade in {"A", "A+", "B+"}
    assert resp.surplus_sgp == pytest.approx(4.0)


def test_engine_warnings_passed_through_when_present(monkeypatch):
    """Supplementary engine context (risk_flags / reshuffle / ip-floor /
    playoff deltas) is surfaced when the engine succeeds."""

    def _engine(**kw):
        return {
            "risk_flags": ["Roster below IP floor."],
            "reshuffle": {"reshuffle_pct": 40.0},
            "ip_floor_detail": {"risk": "Weekly IP short."},
            "delta_playoff_prob": 0.05,
            "delta_champ_prob": 0.02,
        }

    _install_trade_stubs(
        monkeypatch,
        per_cat_map={1: {"HR": 1.0}, 2: {"HR": 5.0}},
        rosters=pd.DataFrame({"team_name": [], "player_id": []}),
        evaluate_fn=_engine,
    )
    resp = TradeService().evaluate("Team Hickey", giving_ids=[1], receiving_ids=[2])
    assert "Roster below IP floor." in resp.warnings
    assert any("reshuffle" in w.lower() for w in resp.warnings)
    assert "Weekly IP short." in resp.warnings
    assert resp.delta_playoff_prob == pytest.approx(0.05)
    assert resp.delta_champ_prob == pytest.approx(0.02)
    # Headline still from honest math, NOT the engine.
    assert resp.verdict == "You win"
    assert resp.surplus_sgp == pytest.approx(4.0)


def test_mc_confidence_override(monkeypatch):
    """With enable_mc=True, the engine's MC confidence may override the magnitude-
    derived confidence; the grade/verdict/surplus stay honest."""

    def _engine(**kw):
        return {"confidence_pct": 88.0}

    _install_trade_stubs(
        monkeypatch,
        per_cat_map={1: {"HR": 1.0}, 2: {"HR": 5.0}},
        rosters=pd.DataFrame({"team_name": [], "player_id": []}),
        evaluate_fn=_engine,
    )
    resp = TradeService().evaluate("Team Hickey", giving_ids=[1], receiving_ids=[2], enable_mc=True)
    assert resp.confidence_pct == pytest.approx(88.0)
    assert resp.verdict == "You win"
    assert resp.surplus_sgp == pytest.approx(4.0)


def test_evaluate_never_raises_on_total_failure(monkeypatch):
    """Total failure (pool load blows up) → the default 'Could not evaluate'."""

    def _boom():
        raise RuntimeError("db down")

    import src.database

    monkeypatch.setattr(src.database, "load_player_pool", _boom, raising=False)
    resp = TradeService().evaluate("Team Hickey", giving_ids=[1], receiving_ids=[2])
    assert isinstance(resp, TradeEvaluationResponse)
    assert resp.verdict == "Could not evaluate"


# ── TASK 2: Compare reads ACTUAL YTD ──────────────────────────────────────────


def test_compare_reads_ytd_columns(monkeypatch):
    """Each stat must come from ytd_<col>, NOT the projection col, so the head-to-head
    shows real 2026 production. Build a pool where proj != ytd and assert ytd wins."""
    pool = pd.DataFrame(
        {
            "player_id": [1, 2],
            "player_name": ["Alpha", "Beta"],
            "positions": ["1B", "OF"],
            "team": ["ATL", "COL"],
            "mlb_id": [100, 200],
            # projection columns (should NOT be read)
            "hr": [16, 13],
            "avg": [0.249, 0.261],
            "r": [70, 60],
            # ytd columns (SHOULD be read)
            "ytd_hr": [19, 12],
            "ytd_avg": [0.271, 0.280],
            "ytd_r": [44, 30],
        }
    )
    monkeypatch.setattr(compare_mod, "load_player_pool", lambda: pool, raising=False)
    import src.database

    monkeypatch.setattr(src.database, "load_player_pool", lambda: pool, raising=False)

    resp = CompareService().compare([1, 2])
    assert isinstance(resp, CompareResponse)
    by_name = {p.player.name: p.stats for p in resp.players}
    # YTD values, not projections.
    assert by_name["Alpha"]["HR"] == pytest.approx(19.0)
    assert by_name["Beta"]["HR"] == pytest.approx(12.0)
    assert by_name["Alpha"]["AVG"] == pytest.approx(0.271)
    assert by_name["Alpha"]["R"] == pytest.approx(44.0)


def test_compare_falls_back_to_projection_when_no_ytd_col(monkeypatch):
    """A stat with no ytd_ variant in the pool falls back to the projection col so the
    table never goes blank. Here HR has a ytd col; SB does NOT (only projection)."""
    pool = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["Alpha"],
            "positions": ["OF"],
            "team": ["ATL"],
            "mlb_id": [100],
            "hr": [16],
            "ytd_hr": [19],
            "sb": [25],  # no ytd_sb column at all
        }
    )
    monkeypatch.setattr(compare_mod, "load_player_pool", lambda: pool, raising=False)
    import src.database

    monkeypatch.setattr(src.database, "load_player_pool", lambda: pool, raising=False)

    resp = CompareService().compare([1])
    stats = resp.players[0].stats
    assert stats["HR"] == pytest.approx(19.0)  # ytd
    assert stats["SB"] == pytest.approx(25.0)  # projection fallback


def test_compare_nan_guard(monkeypatch):
    """A NaN ytd value → 0.0 (not NaN) in the response."""
    pool = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["Alpha"],
            "positions": ["OF"],
            "team": ["ATL"],
            "mlb_id": [100],
            "ytd_hr": [float("nan")],
            "hr": [float("nan")],
        }
    )
    monkeypatch.setattr(compare_mod, "load_player_pool", lambda: pool, raising=False)
    import src.database

    monkeypatch.setattr(src.database, "load_player_pool", lambda: pool, raising=False)

    resp = CompareService().compare([1])
    assert resp.players[0].stats.get("HR", 0.0) == 0.0
