"""F4 (2026-06-02 silent-failure sweep): compute_drop_cost must derive the
league-average AVG/OBP used for its rate-stat-drag adjustment from
``standings_utils.get_all_team_totals()`` — NOT from ``st.session_state
["_cached_team_totals"]``, a key nothing in the codebase ever writes (the cache
is a module-level global in standings_utils). Because that read always missed,
the rate-drag adjustment silently used the hardcoded .250/.320 prior-season
defaults for every player, on every team, forever.

These tests force the dynamic and default branches apart: a .260/.330 hitter sits
ABOVE the static defaults (no penalty) but BELOW a patched dynamic league average
of .270/.340 (penalty fires). If the drop cost is identical across the two, the
dynamic average is not being read — the F4 bug.
"""

from __future__ import annotations

import pandas as pd

import src.standings_utils as standings_utils
from src.valuation import LeagueConfig
from src.waiver_wire import compute_drop_cost


def _mid_hitter_pool() -> pd.DataFrame:
    """One hitter at AVG .260 / OBP .330 — between the static .250/.320 defaults
    and the dynamic .270/.340 average used in the test. sb/hr are kept high so the
    SB-dead-weight and HR<5 adjustments don't fire and confound the rate-drag arm;
    a single non-DH position avoids the DH penalty and the 3+ multi-pos bonus."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Mid Hitter",
                "positions": "2B",
                "is_hitter": 1,
                "r": 70,
                "hr": 20,
                "rbi": 65,
                "sb": 15,
                "ab": 500,
                "h": 130,
                "bb": 45,
                "hbp": 4,
                "sf": 3,
                "avg": 0.260,
                "obp": 0.330,
                "status": "active",
            }
        ]
    )


def test_drop_cost_uses_dynamic_league_averages(monkeypatch):
    """Dynamic league averages from get_all_team_totals() must change the
    rate-stat-drag adjustment vs the static-default fallback."""
    pool = _mid_hitter_pool()
    cfg = LeagueConfig()

    # Scenario A — no league data: get_all_team_totals() returns {} → the
    # adjustment falls back to .250/.320. .260 > .250 and .330 > .320, so NO
    # rate-drag penalty fires.
    monkeypatch.setattr(standings_utils, "get_all_team_totals", lambda *a, **k: {})
    cost_default = compute_drop_cost(1, [1], pool, cfg)

    # Scenario B — dynamic league averages .270/.340: .260 < .270 and
    # .330 < .340, so the rate-drag penalty (-1.0 AVG, -0.5 OBP) fires and the
    # drop cost drops by ~1.5 relative to Scenario A.
    monkeypatch.setattr(
        standings_utils,
        "get_all_team_totals",
        lambda *a, **k: {
            "Team A": {"AVG": 0.270, "OBP": 0.340},
            "Team B": {"AVG": 0.270, "OBP": 0.340},
        },
    )
    cost_dynamic = compute_drop_cost(1, [1], pool, cfg)

    assert cost_dynamic < cost_default - 1.0, (
        "Dynamic league averages (.270/.340) should trigger the rate-stat-drag "
        "penalty for a .260/.330 hitter, lowering the drop cost ~1.5 below the "
        f"static-default scenario. default={cost_default:.2f} dynamic={cost_dynamic:.2f} "
        "(if equal, compute_drop_cost is still reading the never-written "
        "session_state key instead of get_all_team_totals())."
    )


def test_drop_cost_calls_get_all_team_totals(monkeypatch):
    """Wiring guard: compute_drop_cost must actually invoke get_all_team_totals()
    for a hitter (the rate-drag arm), proving it no longer reads the dead
    session_state key."""
    pool = _mid_hitter_pool()
    cfg = LeagueConfig()
    calls = {"n": 0}

    def _spy(*a, **k):
        calls["n"] += 1
        return {}

    monkeypatch.setattr(standings_utils, "get_all_team_totals", _spy)
    compute_drop_cost(1, [1], pool, cfg)
    assert calls["n"] >= 1, "compute_drop_cost did not call get_all_team_totals() for a hitter"
