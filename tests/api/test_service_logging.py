"""tests/api/test_service_logging.py

TDD tests for U1: Backend observability sweep.
Every swallow site must log a WARNING before degrading gracefully.

Pattern per test:
  1. Monkeypatch the FIRST src-import / engine call inside the try block to raise.
  2. Call the service method.
  3. Assert: (a) return is still the graceful empty/default shape, AND
             (b) a WARNING containing the service name + "failed" is in caplog.
"""

from __future__ import annotations

import logging


def _raise(*a, **k):
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# FreeAgentService
# ---------------------------------------------------------------------------


def test_fa_service_logs_on_failure(monkeypatch, caplog):
    from api.services import fa_service

    svc = fa_service.FreeAgentService()
    # The first call inside the try block is get_yahoo_data_service()
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", _raise, raising=False)

    with caplog.at_level(logging.WARNING):
        out = svc.get_free_agents(team_name="Team Hickey")

    assert out.recommendations == []
    assert any("FreeAgentService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# LeadersService
# ---------------------------------------------------------------------------


def test_leaders_service_logs_on_failure(monkeypatch, caplog):
    from api.services import leaders_service

    svc = leaders_service.LeadersService()
    # The first call inside the try is get_connection()
    monkeypatch.setattr("src.database.get_connection", _raise, raising=False)

    with caplog.at_level(logging.WARNING):
        out = svc.get_leaders(category="HR")

    assert out.rows == []
    assert any("LeadersService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# StandingsService
# ---------------------------------------------------------------------------


def test_standings_service_logs_on_failure(monkeypatch, caplog):
    from api.services import standings_service

    svc = standings_service.StandingsService()
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", _raise, raising=False)

    with caplog.at_level(logging.WARNING):
        out = svc.get_standings()

    assert out.teams == []
    assert any("StandingsService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# StreamingService — get_streaming
# ---------------------------------------------------------------------------


def test_streaming_service_get_streaming_logs_on_failure(monkeypatch, caplog):
    from api.services import streaming_service

    svc = streaming_service.StreamingService()
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", _raise, raising=False)

    with caplog.at_level(logging.WARNING):
        out = svc.get_streaming()

    assert out.candidates == []
    assert any("StreamingService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# StreamingService — analyze_pitcher (LOW-6: found=False path)
# ---------------------------------------------------------------------------


def test_streaming_service_analyze_pitcher_logs_on_failure(monkeypatch, caplog):
    from api.contracts.streaming import StreamAnalyzeRequest
    from api.services import streaming_service

    svc = streaming_service.StreamingService()
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", _raise, raising=False)

    req = StreamAnalyzeRequest(pitcher_id=1, date="2026-06-22")
    with caplog.at_level(logging.WARNING):
        out = svc.analyze_pitcher(req)

    assert out.found is False
    assert out.scorecard is None
    assert any("StreamingService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# CloserService
# ---------------------------------------------------------------------------


def test_closer_service_logs_on_failure(monkeypatch, caplog):
    from api.services import closers_service

    svc = closers_service.CloserService()
    monkeypatch.setattr("src.closer_monitor.build_depth_data_from_db", _raise, raising=False)

    with caplog.at_level(logging.WARNING):
        out = svc.get_closers()

    assert out.entries == []
    assert any("CloserService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# PuntService
# ---------------------------------------------------------------------------


def test_punt_service_logs_on_failure(monkeypatch, caplog):
    from api.services import punt_service

    svc = punt_service.PuntService()
    monkeypatch.setattr("src.standings_utils.get_all_team_totals", _raise, raising=False)

    with caplog.at_level(logging.WARNING):
        out = svc.get_punt(team_name="Team Hickey")

    assert out.categories == []
    assert any("PuntService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# DraftService — recommend (invalid-state and engine paths)
# ---------------------------------------------------------------------------


def test_draft_service_recommend_invalid_state_logs(monkeypatch, caplog):
    """_rebuild_state raises → returns the invalid-state summary AND logs."""
    from api.contracts.draft import DraftConfig, DraftRecommendRequest
    from api.services import draft_service

    svc = draft_service.DraftService()
    monkeypatch.setattr(draft_service.DraftService, "_rebuild_state", staticmethod(_raise))

    req = DraftRecommendRequest(
        config=DraftConfig(num_teams=12, num_rounds=23, user_team_index=0),
        pick_log=[],
    )
    with caplog.at_level(logging.WARNING):
        out = svc.recommend(req)

    assert out.recommendations == []
    assert "Invalid draft state" in out.summary
    assert any("DraftService" in r.message and "invalid draft state" in r.message for r in caplog.records)


def test_draft_service_recommend_engine_failure_logs(monkeypatch, caplog):
    """_run_engine raises → returns the engine-unavailable summary AND logs."""
    from api.contracts.draft import DraftConfig, DraftRecommendRequest
    from api.services import draft_service
    from src.draft_state import DraftState

    svc = draft_service.DraftService()

    # _rebuild_state must succeed (return a real DraftState)
    ds = DraftState(num_teams=12, num_rounds=23, user_team_index=0)
    monkeypatch.setattr(draft_service.DraftService, "_rebuild_state", staticmethod(lambda req: ds))
    monkeypatch.setattr(draft_service.DraftService, "_run_engine", staticmethod(_raise))

    req = DraftRecommendRequest(
        config=DraftConfig(num_teams=12, num_rounds=23, user_team_index=0),
        pick_log=[],
    )
    with caplog.at_level(logging.WARNING):
        out = svc.recommend(req)

    assert out.recommendations == []
    assert "unavailable" in out.summary.lower()
    assert any("DraftService" in r.message and "failed" in r.message for r in caplog.records)


def test_draft_service_simulate_picks_logs_on_failure(monkeypatch, caplog):
    from api.contracts.draft import DraftConfig, DraftSimulatePicksRequest
    from api.services import draft_service
    from src.draft_state import DraftState

    svc = draft_service.DraftService()

    ds = DraftState(num_teams=12, num_rounds=23, user_team_index=0)
    monkeypatch.setattr(draft_service.DraftService, "_rebuild_state", staticmethod(lambda req: ds))
    monkeypatch.setattr("src.simulation.auto_pick_opponents", _raise, raising=False)

    req = DraftSimulatePicksRequest(
        config=DraftConfig(num_teams=12, num_rounds=23, user_team_index=0),
        pick_log=[],
        seed=42,
    )
    with caplog.at_level(logging.WARNING):
        out = svc.simulate_picks(req)

    assert out.picks == []
    assert "unavailable" in out.summary.lower()
    assert any("DraftService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# MatchupService — get_matchup
# ---------------------------------------------------------------------------


def test_matchup_service_get_matchup_logs_on_failure(monkeypatch, caplog):
    from api.services import matchup_service

    svc = matchup_service.MatchupService()
    monkeypatch.setattr("src.yahoo_data_service.get_yahoo_data_service", _raise, raising=False)

    with caplog.at_level(logging.WARNING):
        out = svc.get_matchup(team_name="Team Hickey")

    assert out.categories == []
    assert any("MatchupService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# MatchupService — _build_roster_tables (inner except pass)
# ---------------------------------------------------------------------------


def test_matchup_service_build_roster_tables_logs_on_failure(monkeypatch, caplog):
    """_build_roster_tables raises → outer caller catches it, fields stay empty, and logs."""
    from api.services import matchup_service

    svc = matchup_service.MatchupService()

    # Patch _build_roster_tables itself to raise
    monkeypatch.setattr(
        matchup_service.MatchupService,
        "_build_roster_tables",
        staticmethod(_raise),
    )

    # Also patch the outer get_yahoo_data_service to return a mock with get_matchup
    class _FakeMatchup:
        def get_matchup(self):
            return {"week": 1, "opp_name": "Opp Team", "categories": []}

        def get_standings(self):
            return None

    monkeypatch.setattr(
        "src.yahoo_data_service.get_yahoo_data_service",
        lambda: _FakeMatchup(),
        raising=False,
    )

    with caplog.at_level(logging.WARNING):
        out = svc.get_matchup(team_name="Team Hickey")

    # Roster tables should be empty (degraded), not raise
    assert out.hitters == []
    assert out.pitchers == []
    assert any("MatchupService" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# TeamService — get_my_team (LOW-5: Yahoo calls degrade to None, no 500)
# ---------------------------------------------------------------------------


def test_team_service_get_matchup_failure_degrades(monkeypatch, caplog):
    """get_matchup() raises → TeamService.get_my_team must NOT raise, and must log."""
    from api.services import team_service

    svc = team_service.TeamService()

    class _FakeYds:
        def get_matchup(self):
            raise RuntimeError("yahoo down")

        def get_standings(self):
            return None

    monkeypatch.setattr(
        "src.yahoo_data_service.get_yahoo_data_service",
        lambda: _FakeYds(),
        raising=False,
    )
    # Prevent the heavy optimizer context call from running
    monkeypatch.setattr(team_service.TeamService, "_build_ctx", staticmethod(lambda *a, **k: None))
    # Prevent DB roster call from failing
    monkeypatch.setattr(team_service.TeamService, "_roster", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(team_service.TeamService, "_roster_ids", staticmethod(lambda *a, **k: []))

    with caplog.at_level(logging.WARNING):
        out = svc.get_my_team(team_name="Team Hickey")

    # Must not raise; team_name must be propagated
    assert out.team_name == "Team Hickey"
    assert any(
        "TeamService" in r.message and "get_matchup" in r.message and "failed" in r.message for r in caplog.records
    )


def test_team_service_get_standings_failure_degrades(monkeypatch, caplog):
    """get_standings() raises → TeamService.get_my_team must NOT raise, and must log."""
    from api.services import team_service

    svc = team_service.TeamService()

    class _FakeYds:
        def get_matchup(self):
            return None  # fine — won't raise

        def get_standings(self):
            raise RuntimeError("standings down")

    monkeypatch.setattr(
        "src.yahoo_data_service.get_yahoo_data_service",
        lambda: _FakeYds(),
        raising=False,
    )
    monkeypatch.setattr(team_service.TeamService, "_build_ctx", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(team_service.TeamService, "_roster", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(team_service.TeamService, "_roster_ids", staticmethod(lambda *a, **k: []))

    with caplog.at_level(logging.WARNING):
        out = svc.get_my_team(team_name="Team Hickey")

    assert out.team_name == "Team Hickey"
    assert any(
        "TeamService" in r.message and "get_standings" in r.message and "failed" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# src/ai/keys.py — get_key and get_admin_shared_key decrypt failures
# ---------------------------------------------------------------------------


def test_keys_get_key_logs_decrypt_failure(monkeypatch, caplog):
    """_decrypt raising inside get_key → returns None AND logs a WARNING."""
    import src.ai.keys as keys_mod

    # Provide a fake DB row so the code reaches the _decrypt call
    class _FakeConn:
        def execute(self, *a, **k):
            return self

        def fetchone(self):
            return {"encrypted_key": "fakeciphertext"}

        def close(self):
            pass

    monkeypatch.setattr("src.database.get_connection", lambda: _FakeConn(), raising=False)
    monkeypatch.setattr(keys_mod, "_decrypt", _raise, raising=False)

    with caplog.at_level(logging.WARNING):
        result = keys_mod.get_key(user_id=42, provider="anthropic")

    assert result is None
    assert any("get_key" in r.message and "decrypt" in r.message for r in caplog.records)


def test_keys_get_admin_shared_key_logs_decrypt_failure(monkeypatch, caplog):
    """_decrypt raising inside get_admin_shared_key → returns None AND logs a WARNING."""
    import json

    import src.ai.keys as keys_mod

    # Return a valid JSON map so the code reaches _decrypt. Patch the name BOUND
    # IN keys (it does `from src.app_settings import get_setting`), not the source
    # module — otherwise the patch misses and the test leans on shared DB state
    # (passes alone, fails in the full suite).
    monkeypatch.setattr(
        keys_mod,
        "get_setting",
        lambda key: json.dumps({"openai": "fakeciphertext"}),
        raising=False,
    )
    monkeypatch.setattr(keys_mod, "_decrypt", _raise, raising=False)

    with caplog.at_level(logging.WARNING):
        result = keys_mod.get_admin_shared_key(provider="openai")

    assert result is None
    assert any("get_admin_shared_key" in r.message and "decrypt" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# DraftService — simulate_picks FIRST catch (invalid state path)
# ---------------------------------------------------------------------------


def test_draft_service_simulate_picks_invalid_state_logs(monkeypatch, caplog):
    """_rebuild_state raises inside simulate_picks FIRST catch → graceful return + WARNING.

    The existing test_draft_service_simulate_picks_logs_on_failure only exercises
    the SECOND catch (engine failure after _rebuild_state succeeds).  This test
    forces the FIRST catch by monkeypatching _rebuild_state to raise.
    """
    from api.contracts.draft import DraftConfig, DraftSimulatePicksRequest
    from api.services import draft_service

    svc = draft_service.DraftService()
    monkeypatch.setattr(draft_service.DraftService, "_rebuild_state", staticmethod(_raise))

    req = DraftSimulatePicksRequest(
        config=DraftConfig(num_teams=12, num_rounds=23, user_team_index=0),
        pick_log=[],
        seed=42,
    )
    with caplog.at_level(logging.WARNING):
        out = svc.simulate_picks(req)

    # Graceful shape — same as the "Invalid draft state." return
    assert out.picks == []
    assert "Invalid draft state" in out.summary
    # The new WARNING must appear
    assert any(
        "DraftService" in r.message and "simulate_picks" in r.message and "invalid draft state" in r.message
        for r in caplog.records
    )


# ---------------------------------------------------------------------------
# StreamingService — probables inner build failure
# ---------------------------------------------------------------------------


def test_streaming_service_probables_build_failure_logs(monkeypatch, caplog):
    """Inner probables build_stream_board raises → probables=[] + WARNING logged."""
    from api.services import streaming_service

    svc = streaming_service.StreamingService()

    # Build a fake context that passes the outer try block but fails the inner one.
    class _FakeCtx:
        adds_remaining_this_week = 10
        category_gaps = {}

    call_count = {"n": 0}

    def _fake_board(ctx, date, include_rostered=False, **kw):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call (main board) → return empty frame so we don't need pandas
            import pandas as pd

            return pd.DataFrame()
        # Second call (probables with include_rostered=True) → raise
        raise RuntimeError("probables boom")

    monkeypatch.setattr(
        "src.optimizer.shared_data_layer.build_optimizer_context",
        lambda **k: _FakeCtx(),
        raising=False,
    )
    monkeypatch.setattr(
        "src.optimizer.stream_analyzer.build_stream_board",
        _fake_board,
        raising=False,
    )
    monkeypatch.setattr(
        "src.game_day.get_target_game_date",
        lambda: "2026-06-22",
        raising=False,
    )
    monkeypatch.setattr(
        "src.yahoo_data_service.get_yahoo_data_service",
        lambda: None,
        raising=False,
    )
    monkeypatch.setattr(
        "src.valuation.LeagueConfig",
        lambda: None,
        raising=False,
    )

    with caplog.at_level(logging.WARNING):
        out = svc.get_streaming()

    assert out.probables == []
    assert any(
        "StreamingService" in r.message and "probables" in r.message and "failed" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# MatchupService — _league outer failure
# ---------------------------------------------------------------------------


def test_matchup_service_league_logs_on_failure(monkeypatch, caplog):
    """_league outer except raises → returns [] AND logs a WARNING."""
    from api.services import matchup_service

    svc = matchup_service.MatchupService()
    # Patch the schedule loader to raise inside _league
    monkeypatch.setattr(
        "src.database.load_league_schedule_full",
        _raise,
        raising=False,
    )

    with caplog.at_level(logging.WARNING):
        result = svc._league(team_name="Team Hickey", opponent="Opp", week=1, you_score=5, opp_score=4)

    assert result == []
    assert any(
        "MatchupService" in r.message and "_league" in r.message and "failed" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# MatchupService — _team_side failure
# ---------------------------------------------------------------------------


def test_matchup_service_team_side_logs_on_failure(monkeypatch, caplog):
    """_team_side except pass raises → still returns TeamSide AND logs a WARNING."""
    from api.services import matchup_service

    # Patch get_connection to raise so the inner except is triggered
    monkeypatch.setattr(
        "src.database.get_connection",
        _raise,
        raising=False,
    )

    with caplog.at_level(logging.WARNING):
        result = matchup_service.MatchupService._team_side("Team Hickey", 5)

    # Still returns a valid TeamSide with defaults for manager/record
    assert result.name == "Team Hickey"
    assert result.score == 5
    assert result.manager == ""
    assert any(
        "MatchupService" in r.message and "_team_side" in r.message and "failed" in r.message for r in caplog.records
    )


# ---------------------------------------------------------------------------
# DraftService — grade failure
# ---------------------------------------------------------------------------


def test_draft_service_grade_logs_on_failure(monkeypatch, caplog):
    """grade() except returns DraftGradeResponse() AND logs a WARNING."""
    from api.contracts.draft import DraftConfig, DraftGradeRequest
    from api.services import draft_service

    svc = draft_service.DraftService()
    # Force the inner pool load to raise to hit the except path
    monkeypatch.setattr("src.database.load_player_pool", _raise, raising=False)
    monkeypatch.setattr("src.valuation.LeagueConfig", _raise, raising=False)

    req = DraftGradeRequest(
        config=DraftConfig(num_teams=12, num_rounds=23, user_team_index=0),
        pick_log=[],
    )
    with caplog.at_level(logging.WARNING):
        out = svc.grade(req)

    # Graceful empty grade
    assert out.overall_grade == "N/A"
    assert any("DraftService" in r.message and "grade" in r.message and "failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# CloserService — pool load DEBUG log
# ---------------------------------------------------------------------------


def test_closer_service_pool_load_failure_logs_debug(monkeypatch, caplog):
    """Inner pool load raising → DEBUG line emitted; get_closers still returns entries."""
    from api.services import closers_service

    svc = closers_service.CloserService()

    # Make the depth data succeed (return a valid list)
    monkeypatch.setattr(
        "src.closer_monitor.build_depth_data_from_db",
        lambda: [
            {
                "team": "NYY",
                "closer_name": "Closer A",
                "setup_names": [],
                "job_security": 0.8,
                "security_color": "green",
                "projected_sv": 20,
                "era": 3.0,
                "whip": 1.1,
                "mlb_id": None,
            }
        ],
        raising=False,
    )
    # Make the pool load raise (the non-fatal inner try)
    monkeypatch.setattr("src.database.load_player_pool", _raise, raising=False)
    # Make grid build succeed with no pool
    monkeypatch.setattr(
        "src.closer_monitor.build_closer_grid",
        lambda depth_data, player_pool=None: depth_data,
        raising=False,
    )

    with caplog.at_level(logging.DEBUG):
        out = svc.get_closers()

    # Grid still returns entries (pool miss is non-fatal)
    assert len(out.entries) >= 1
    # DEBUG line — not WARNING
    assert any(
        "CloserService" in r.message
        and "pool" in r.message.lower()
        and "failed" in r.message
        and r.levelno == logging.DEBUG
        for r in caplog.records
    )
    # Must NOT emit a WARNING for the pool miss (it's expected fallback)
    assert not any(
        "CloserService" in r.message and "pool" in r.message.lower() and r.levelno == logging.WARNING
        for r in caplog.records
    )
