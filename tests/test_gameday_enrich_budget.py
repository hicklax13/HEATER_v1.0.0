"""Item A (2026-06-08): render-time budget on build_optimizer_context's game-day
enrichment.

In production the scheduler warms schedule/lineups/team-strength into SQLite, so
member renders read them fast. But in a brief cold-cache window (right after a
deploy, before the scheduler's first cycle) every game-day fetch goes live and the
render can take minutes. A monotonic deadline on the context bounds the TOTAL
game-day enrichment: it's checked before each step + inside the looping steps
(lineups boxscore-per-game, team-strength per-team); once past, the remaining
enrichments are skipped (best-effort — the page still renders with projections).
"""

import time
import types

import pandas as pd

import src.game_day as gd
from src.optimizer import shared_data_layer as sdl


def test_gameday_time_left_true_when_no_deadline():
    assert sdl._gameday_time_left(types.SimpleNamespace(gameday_deadline=None)) is True


def test_gameday_time_left_false_when_past():
    assert sdl._gameday_time_left(types.SimpleNamespace(gameday_deadline=time.monotonic() - 1)) is False


def test_gameday_time_left_true_when_before():
    assert sdl._gameday_time_left(types.SimpleNamespace(gameday_deadline=time.monotonic() + 5)) is True


def test_team_strength_loop_breaks_at_deadline(monkeypatch):
    """The team-strength per-team loop stops at the render deadline."""

    class _MCS:
        def get_team_strength(self, t):
            time.sleep(0.15)
            return {"wrc_plus": 100}

    monkeypatch.setattr("src.matchup_context.get_matchup_context", lambda: _MCS())
    ctx = types.SimpleNamespace(
        roster=pd.DataFrame([{"team": f"T{i}"} for i in range(20)]),
        team_strength={},
        gameday_deadline=time.monotonic() + 0.4,
    )
    t0 = time.monotonic()
    sdl._load_team_strength(ctx)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, f"team-strength ran {elapsed:.1f}s — deadline not honored"
    assert len(ctx.team_strength) < 20  # stopped early


def test_get_todays_lineups_breaks_at_deadline(monkeypatch):
    """get_todays_lineups stops fetching boxscores at the deadline."""
    calls = {"n": 0}

    def _slow_box(pk):
        calls["n"] += 1
        time.sleep(0.15)
        return {"home": {}, "away": {}}

    monkeypatch.setattr(gd, "_statsapi", types.SimpleNamespace(boxscore_data=_slow_box))
    schedule = [{"game_id": i + 1, "home_name": "A", "away_name": "B"} for i in range(20)]
    t0 = time.monotonic()
    gd.get_todays_lineups(schedule, deadline=time.monotonic() + 0.4)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5, f"lineups ran {elapsed:.1f}s — deadline not honored"
    assert calls["n"] < 20  # stopped early


def test_get_todays_lineups_no_deadline_processes_all(monkeypatch):
    """Backward compat: no deadline => process every game (current behavior)."""
    calls = {"n": 0}

    def _box(pk):
        calls["n"] += 1
        return {"home": {}, "away": {}}

    monkeypatch.setattr(gd, "_statsapi", types.SimpleNamespace(boxscore_data=_box))
    schedule = [{"game_id": i + 1, "home_name": "A", "away_name": "B"} for i in range(5)]
    gd.get_todays_lineups(schedule)
    assert calls["n"] == 5
