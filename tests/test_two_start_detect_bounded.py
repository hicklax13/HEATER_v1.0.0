"""#4 QA finding Q2-3 (2026-06-08): bound two-start detection on the render path.

build_optimizer_context (used by Free Agents + Lineup Optimizer) calls
_detect_two_start_pitchers -> two_start.identify_two_start_pitchers, which makes
slow live fetches (statsapi schedule over 7 days + fetch_team_batting_stats). The
existing try/except catches exceptions but does NOT bound TIME, so a slow MLB
data source makes the page render exceed the cap (surfaced as the disconnected FA
render failure; a real production slowness for members). Two-start data is an
enhancement, so on timeout it is skipped gracefully (ctx.two_start_pitchers empty).
"""

import time
import types

import pandas as pd

from src.optimizer import shared_data_layer as sdl


def _ctx():
    return types.SimpleNamespace(
        scope="rest_of_season",
        player_pool=pd.DataFrame([{"name": "Joe Pitcher", "player_id": 1, "is_hitter": 0}]),
        two_start_pitchers=[],
    )


def test_detect_two_start_bounded_on_slow(monkeypatch):
    """A slow identify_two_start_pitchers must NOT block the render — it's skipped
    within ~the budget and two_start_pitchers stays empty (graceful degradation)."""
    import src.two_start as ts

    def _slow(*a, **k):
        time.sleep(5)
        return [{"pitcher_name": "Joe Pitcher"}]

    monkeypatch.setattr(ts, "identify_two_start_pitchers", _slow)
    monkeypatch.setattr(sdl, "_TWO_START_DETECT_TIMEOUT_S", 0.5)

    ctx = _ctx()
    t0 = time.monotonic()
    sdl._detect_two_start_pitchers(ctx)
    elapsed = time.monotonic() - t0

    assert elapsed < 2.5, f"_detect_two_start_pitchers ran {elapsed:.1f}s — budget not enforced"
    assert ctx.two_start_pitchers == []  # skipped, no two-start data


def test_detect_two_start_fast_path_maps_ids(monkeypatch):
    """When detection is fast, two-start pitcher ids are mapped from the pool."""
    import src.two_start as ts

    monkeypatch.setattr(ts, "identify_two_start_pitchers", lambda **k: [{"pitcher_name": "Joe Pitcher"}])
    ctx = _ctx()
    sdl._detect_two_start_pitchers(ctx)
    assert ctx.two_start_pitchers == [1]  # mapped Joe Pitcher -> player_id 1


def test_detect_two_start_skipped_for_today_scope():
    ctx = _ctx()
    ctx.scope = "today"
    sdl._detect_two_start_pitchers(ctx)
    assert ctx.two_start_pitchers == []
