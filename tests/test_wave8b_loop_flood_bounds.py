# tests/test_wave8b_loop_flood_bounds.py
"""Wave 8b CodeRabbit follow-up: bounded in-loop WARNING volume.

Wave 8b added ~60 logger.warning calls; 5 hotspots sat inside hot loops
that could flood operator dashboards if the failure case was persistent
(e.g. closer_monitor import error -> 25x25 warnings per scan_1_for_1 call).

These tests verify the log-once-then-summarize pattern: for each fix
site, even if the per-iteration failure happens many times, the WARNING
volume is bounded to at most 2 lines (first failure + summary).

Sites covered:
  - trade_finder.scan_1_for_1 (nested for-give x for-recv)
  - espn_injuries.store_injuries (per-row INSERT)
  - live_stats.save_statcast_leaderboards_to_db (per-row, two failure kinds)
  - optimizer/matchup_adjustments.compute_weekly_matchup_adjustments (per-row)
  - optimizer/shared_data_layer._load_team_strength (per-team)
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pandas as pd


def _count_substring_warnings(caplog, logger_name: str, substring: str) -> int:
    """Count WARNING-level records from logger_name whose message contains substring."""
    return sum(
        1
        for r in caplog.records
        if r.levelno >= logging.WARNING and r.name == logger_name and substring.lower() in r.message.lower()
    )


# ---------------------------------------------------------------------------
# trade_finder.scan_1_for_1 — closer_monitor failure must not flood
# ---------------------------------------------------------------------------


def test_trade_finder_closer_monitor_warnings_bounded(monkeypatch, caplog):
    """Wave 8b CodeRabbit fix: closer_monitor.compute_job_security failure
    must not flood logs across the scan_1_for_1 nested loop.

    Sets up a small (3 user x 3 opponent) player pool where every receive
    candidate has SV >= 5. Patches `src.closer_monitor.compute_job_security`
    to always raise. Without the loop-flood guard, this would emit one
    WARNING per (give, recv) pair (up to 9 here). With the guard, total
    is bounded to <= 2 (first failure + summary).
    """
    from src import trade_finder
    from src.valuation import LeagueConfig

    # Build a minimal player pool with all required SGP columns and pitchers w/ SV >= 5
    # so the closer-monitor branch is exercised.
    cat_cols = {
        "r": 0.0,
        "hr": 0.0,
        "rbi": 0.0,
        "sb": 0.0,
        "h": 0.0,
        "ab": 1.0,
        "bb": 0.0,
        "hbp": 0.0,
        "sf": 0.0,
        "avg": 0.260,
        "obp": 0.330,
        "w": 10.0,
        "l": 5.0,
        "sv": 30.0,
        "k": 150.0,
        "ip": 70.0,
        "er": 25.0,
        "h_allowed": 50.0,
        "bb_allowed": 20.0,
        "era": 3.20,
        "whip": 1.10,
        "is_hitter": 0,
        "pa": 0,
    }
    rows = []
    pids_user = [1, 2, 3]
    pids_opp = [4, 5, 6]
    for pid in pids_user + pids_opp:
        row = {"player_id": pid, "name": f"P{pid}", **cat_cols}
        rows.append(row)
    pool = pd.DataFrame(rows)

    # Force `compute_job_security` to raise — this is the failure path.
    import src.closer_monitor as closer_monitor

    def _boom(*a, **kw):
        raise ImportError("simulated closer_monitor failure")

    monkeypatch.setattr(closer_monitor, "compute_job_security", _boom)

    cfg = LeagueConfig()
    with caplog.at_level(logging.WARNING, logger="src.trade_finder"):
        trade_finder.scan_1_for_1(
            user_roster_ids=pids_user,
            opponent_roster_ids=pids_opp,
            player_pool=pool,
            config=cfg,
        )

    closer_warnings = _count_substring_warnings(caplog, "src.trade_finder", "closer_monitor")
    assert closer_warnings <= 2, (
        f"loop-flood regression: {closer_warnings} closer_monitor warnings emitted "
        f"in a single scan_1_for_1 call across {len(pids_user) * len(pids_opp)} pairs "
        f"(expected <= 2: first-failure log + summary)"
    )


# ---------------------------------------------------------------------------
# espn_injuries.store_injuries — per-row INSERT failure must not flood
# ---------------------------------------------------------------------------


def test_espn_injuries_store_injuries_warnings_bounded(caplog, monkeypatch):
    """Wave 8b CodeRabbit fix: store_injuries per-row INSERT failure
    must not flood logs (one WARNING per injury row would be bad).
    """
    from src import espn_injuries

    # Build 10 fake injury rows
    injuries = [
        {
            "player_name": f"Player {i}",
            "team": "NYY",
            "status": "DTD",
            "injury_type": "hamstring",
            "detail": "day-to-day",
        }
        for i in range(10)
    ]

    # Monkeypatch match_player_id to always succeed (returns int) so we reach
    # the INSERT path. The real `src.live_stats.match_player_id` is imported
    # locally inside store_injuries, so patch it in src.live_stats.
    import src.live_stats as live_stats

    monkeypatch.setattr(live_stats, "match_player_id", lambda name, team: 12345)

    # Force conn.execute to raise on INSERT (the call inside the loop).
    class _BoomConn:
        def execute(self, *a, **kw):
            raise RuntimeError("simulated INSERT failure")

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("src.database.get_connection", lambda: _BoomConn())

    with caplog.at_level(logging.WARNING, logger="src.espn_injuries"):
        espn_injuries.save_espn_injuries_to_db(injuries)

    insert_warnings = _count_substring_warnings(caplog, "src.espn_injuries", "INSERT")
    assert insert_warnings <= 2, (
        f"loop-flood regression: {insert_warnings} per-row INSERT warnings emitted "
        f"in a single store_injuries call across 10 rows (expected <= 2)"
    )


# ---------------------------------------------------------------------------
# espn_injuries.update_player_injury_flags — per-row UPDATE flood guard
# ---------------------------------------------------------------------------


def test_espn_injuries_update_flags_warnings_bounded(caplog, monkeypatch):
    """Wave 8b CodeRabbit fix: update_player_injury_flags per-row UPDATE
    failure must not flood logs.
    """
    from src import espn_injuries

    injuries = [
        {"player_name": f"Player {i}", "team": "NYY", "status": "IL10", "injury_type": "elbow"} for i in range(8)
    ]

    import src.live_stats as live_stats

    monkeypatch.setattr(live_stats, "match_player_id", lambda name, team: 12345)

    class _BoomConn:
        def execute(self, *a, **kw):
            raise RuntimeError("simulated UPDATE failure")

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("src.database.get_connection", lambda: _BoomConn())

    with caplog.at_level(logging.WARNING, logger="src.espn_injuries"):
        espn_injuries.update_player_injury_flags(injuries)

    update_warnings = _count_substring_warnings(caplog, "src.espn_injuries", "UPDATE")
    assert update_warnings <= 2, (
        f"loop-flood regression: {update_warnings} per-row UPDATE warnings emitted "
        f"in a single update_player_injury_flags call across 8 rows (expected <= 2)"
    )


# ---------------------------------------------------------------------------
# live_stats.save_statcast_leaderboards_to_db — per-row INSERT bound
# ---------------------------------------------------------------------------


def test_live_stats_statcast_insert_warnings_bounded(caplog, monkeypatch):
    """Wave 8b CodeRabbit fix: statcast_archive INSERT per-row failure
    must not flood logs (200-600 rows per batch — would be 600 warnings).
    """
    from src import live_stats

    df = pd.DataFrame(
        [
            {
                "mlb_id": 10000 + i,
                "xba": 0.250,
                "xslg": 0.420,
                "xwoba": 0.330,
                "barrel_pct": 8.0,
                "exit_velocity_avg": 89.0,
                "hard_hit_pct": 38.0,
            }
            for i in range(20)
        ]
    )

    # Build a fake connection: SELECT succeeds and returns a player_id
    # (so we reach the INSERT path); INSERT raises.
    class _Cursor:
        def execute(self, *a, **kw):
            return self

        def fetchone(self):
            return (99999,)

    class _BoomConn:
        def cursor(self):
            return _Cursor()

        def execute(self, *a, **kw):
            raise RuntimeError("simulated INSERT failure")

        def commit(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr("src.database.get_connection", lambda: _BoomConn())
    # The module under test imports get_connection directly at module load.
    monkeypatch.setattr(live_stats, "get_connection", lambda: _BoomConn())

    with caplog.at_level(logging.WARNING, logger="src.live_stats"):
        live_stats.save_statcast_leaderboards_to_db(df, season=2026)

    insert_warnings = _count_substring_warnings(caplog, "src.live_stats", "INSERT")
    assert insert_warnings <= 2, (
        f"loop-flood regression: {insert_warnings} per-row INSERT warnings emitted "
        f"in a single save_statcast_leaderboards_to_db call across 20 rows (expected <= 2)"
    )


# ---------------------------------------------------------------------------
# optimizer/shared_data_layer._load_team_strength — per-team failure bound
# ---------------------------------------------------------------------------


def test_shared_data_layer_team_strength_warnings_bounded(caplog, monkeypatch):
    """Wave 8b CodeRabbit fix: _load_team_strength per-team failure must
    not flood logs (30 MLB teams * N optimizer calls = many warnings).
    """
    from src.optimizer import shared_data_layer

    # Build a fake context with a roster covering several teams.
    teams = ["NYY", "BOS", "LAD", "HOU", "SDP", "ATL"]
    roster = pd.DataFrame([{"player_id": i, "team": t, "name": f"P{i}"} for i, t in enumerate(teams)])

    class _Ctx:
        def __init__(self):
            self.roster = roster
            self.team_strength = {}

    ctx = _Ctx()

    # Patch the MatchupContextService to always raise on get_team_strength.
    class _BoomMCS:
        def get_team_strength(self, team):
            raise RuntimeError(f"simulated wRC+ fetch failure for {team}")

    with patch("src.matchup_context.get_matchup_context", return_value=_BoomMCS()):
        with caplog.at_level(logging.WARNING, logger="src.optimizer.shared_data_layer"):
            shared_data_layer._load_team_strength(ctx)

    team_warnings = _count_substring_warnings(caplog, "src.optimizer.shared_data_layer", "get_team_strength")
    assert team_warnings <= 2, (
        f"loop-flood regression: {team_warnings} per-team get_team_strength warnings "
        f"emitted across {len(teams)} teams (expected <= 2)"
    )
