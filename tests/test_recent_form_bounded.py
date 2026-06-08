"""#4 QA finding (2026-06-07): the Free Agents / Lineup Optimizer recent-form
loader made UNBOUNDED live MLB-StatsAPI calls during page render.

`build_optimizer_context` -> `_load_recent_form` loops the user's ~25 roster
players, each calling `get_player_recent_form` -> `statsapi.player_stat_data`
(requests.get with NO timeout). If statsapi is slow/unreachable the page hangs
forever (a real 17-min hang surfaced in QA); plus an unconditional 0.5s sleep per
player. Fix: a per-call timeout (no infinite hang) + a total budget on the loop.
Recent-form is an enhancement, so timed-out players degrade to projection.
"""

import time
import types

import pandas as pd

import src.game_day as game_day
from src.optimizer import shared_data_layer as sdl


def test_get_player_recent_form_does_not_hang_on_slow_statsapi(monkeypatch):
    """A hanging statsapi must NOT hang the render — get_player_recent_form
    returns (empty) within ~the per-call timeout, not forever."""
    import threading

    import statsapi

    blocker = threading.Event()  # never set -> the call would block indefinitely

    def _hang(*a, **k):
        blocker.wait()
        return {}

    monkeypatch.setattr(game_day, "_RECENT_FORM_CALL_TIMEOUT", 0.5)
    monkeypatch.setattr(statsapi, "player_stat_data", _hang)

    t0 = time.monotonic()
    out = game_day.get_player_recent_form(12345)
    elapsed = time.monotonic() - t0
    blocker.set()  # release the leaked worker thread

    assert elapsed < 4.0, f"recent-form hung {elapsed:.1f}s — per-call timeout not enforced"
    assert out["player_type"] == "unknown" and out["l7"] == {}  # graceful empty


def test_get_player_recent_form_fast_path_still_returns_data(monkeypatch):
    """The timeout wrapper must not break the normal (fast) path."""
    import statsapi

    def _ok(mlb_id, group, **k):
        if group == "hitting":
            return {"stats": [{"stats": {"avg": ".300", "homeRuns": 2, "atBats": 20, "hits": 6}}]}
        return {"stats": []}

    monkeypatch.setattr(statsapi, "player_stat_data", _ok)
    out = game_day.get_player_recent_form(777)
    assert out["player_type"] == "hitter"
    assert "l7" in out and out["l7"]  # aggregated something


def _ctx(n):
    return types.SimpleNamespace(
        roster=pd.DataFrame([{"player_id": i, "mlb_id": 1000 + i} for i in range(n)]),
        recent_form={},
    )


def test_load_recent_form_respects_total_budget(monkeypatch):
    """_load_recent_form stops after its total budget so a slow statsapi can't
    make the render take minutes; remaining players degrade to projection."""

    def _slow(mlb_id):
        time.sleep(0.2)
        return {"l7": {}, "l14": {}, "l30": {}, "player_type": "hitter", "mlb_id": mlb_id}

    monkeypatch.setattr(sdl, "_RECENT_FORM_TOTAL_BUDGET_S", 0.5)
    monkeypatch.setattr("src.game_day.get_player_recent_form_cached", _slow)

    ctx = _ctx(10)
    t0 = time.monotonic()
    sdl._load_recent_form(ctx)
    elapsed = time.monotonic() - t0

    assert elapsed < 2.0, f"_load_recent_form ran {elapsed:.1f}s — budget not enforced"
    assert len(ctx.recent_form) < 10, "budget should have stopped the loop before all 10 players"
