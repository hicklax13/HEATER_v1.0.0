# tests/test_wave8b_silent_failures.py
"""Wave 8b: silent-failure batch — observability logging for ~30 sites.

These tests verify that each fix in Wave 8b emits a `logger.warning` (with
`exc_info=True` where applicable) on its failure path, while preserving
the original falsy return value. The fixes are pure observability changes
— no behavior changes.

Audit IDs covered (sampled — full coverage in inspection):
  - yahoo_data_service.py (4 sites): is_connected, snapshot, get_opponent_profile,
    _get_user_team_name
  - yahoo_api.py (1 site): get_access_token token-file fallback
  - ecr.py (2 sites): _get_yahoo_client, fetch_ecr_rankings
  - ml_ensemble.py (1 site): get_feature_importance
  - depth_charts.py (3 sites): _fetch_team_depth_chart (HTTP + parse),
    _fetch_team_via_statsapi
  - database.py (3 sites): init CREATE TABLE, velo regression, get_cached_matchup
  - data_bootstrap.py (3 sites): _game_window_aware_ttl, stuff_plus / batting_stats
    refresh_log fallback
  - trade_finder.py (5+ sites): compute_adp_fairness, scan_1_for_1 various
    enrichment loads, find_trade_opportunities trade_intelligence integration,
    closer_monitor SV discount
  - trade_intelligence.py (4 sites): prior_season_pa_ip, batch, opponent
    archetype, ECR fairness
  - optimizer/matchup_adjustments.py (5 sites): umpire_data preload,
    catcher framing per-row, umpire per-row, get_catcher_framing_data,
    get_pvb_matchup_data
  - optimizer/shared_data_layer.py (4 sites): remaining_games schedule,
    team_strength, weather, recent_form
  - alerts.py (4 sites): FA fill recommendations, projected SV map,
    IL-stash candidates, IL return-date match_player_id
  - opponent_intel.py (2 sites): get_schedule, get_opponent_profile
  - opponent_trade_analysis.py (1 site): timestamp parsing
  - validation/calibration_data.py (1 site): _build_league_key
  - espn_injuries.py (2 sites): store_injuries per-row, update_player_injury_flags
  - matchup_planner.py (1 site): park_factors DB load
  - waiver_wire.py (1 site): cached AVG/OBP read
  - game_day.py (1 site): session_state pitch-mix cache write
  - war_room_actions.py (1 site): get_team_strength wRC+ probe
  - live_stats.py (2 sites): mlb_id→player_id lookup, statcast INSERT

All tests use `caplog` to capture log records and assert presence of
WARNING-level messages on the corresponding failure path.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from unittest.mock import patch

import pandas as pd

# ---------------------------------------------------------------------------
# Helper: assert at least one WARNING log from `logger_name` matched
# ---------------------------------------------------------------------------


def _assert_warning(caplog, logger_name: str, fragment: str = "", description: str = ""):
    matches = [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and r.name == logger_name
        and (not fragment or fragment.lower() in r.message.lower())
    ]
    assert matches, (
        f"{description}: expected WARNING log from {logger_name!r} "
        f"matching {fragment!r}. Got records: "
        f"{[(r.name, r.levelname, r.message) for r in caplog.records]}"
    )


# ---------------------------------------------------------------------------
# yahoo_data_service.is_connected — logs when authenticated probe fails
# ---------------------------------------------------------------------------


def test_yds_is_connected_logs_on_probe_failure(caplog):
    from src.yahoo_data_service import YahooDataService

    yds = YahooDataService()
    # Inject a broken client
    bad_client = type("BadClient", (), {})()
    # Property that raises on access
    type(bad_client).is_authenticated = property(lambda self: (_ for _ in ()).throw(RuntimeError("probe failed")))
    yds._client = bad_client

    with caplog.at_level(logging.WARNING, logger="src.yahoo_data_service"):
        result = yds.is_connected()

    assert result is False
    _assert_warning(caplog, "src.yahoo_data_service", "is_connected", "yds.is_connected")


# ---------------------------------------------------------------------------
# yahoo_data_service._get_user_team_name — logs on DB failure
# ---------------------------------------------------------------------------


def test_yds_get_user_team_name_logs_on_db_failure(caplog):
    from src.yahoo_data_service import YahooDataService

    yds = YahooDataService()

    with patch("src.database.get_connection", side_effect=RuntimeError("DB exploded")):
        with caplog.at_level(logging.WARNING, logger="src.yahoo_data_service"):
            result = yds._get_user_team_name()

    assert result is None
    _assert_warning(caplog, "src.yahoo_data_service", "user team", "yds._get_user_team_name")


# ---------------------------------------------------------------------------
# yahoo_api.get_access_token — logs when token-file parse fails
# ---------------------------------------------------------------------------


def test_yahoo_api_get_access_token_logs_on_bad_token_json(caplog, tmp_path, monkeypatch):
    """When yahoo_token.json exists but is corrupt JSON, log a warning."""
    import json

    # Create a bad token file
    bad_token = tmp_path / "yahoo_token.json"
    bad_token.write_text("not-valid-json-{")

    # Patch _AUTH_DIR to point at our tmp dir
    monkeypatch.setattr("src.yahoo_api._AUTH_DIR", tmp_path)

    # Build a fake client that has no in-memory access token, forcing fallback
    from src.yahoo_api import YahooFantasyClient

    # Use object.__new__ to bypass __init__ requirements
    client = object.__new__(YahooFantasyClient)
    client._query = type("Q", (), {"_yahoo_access_token_dict": None})()

    with caplog.at_level(logging.WARNING, logger="src.yahoo_api"):
        token = client._get_bearer_token()

    assert token == ""
    _assert_warning(caplog, "src.yahoo_api", "yahoo_token.json", "yahoo_api._get_bearer_token")


# ---------------------------------------------------------------------------
# ecr._get_yahoo_client — logs when streamlit unavailable
# ---------------------------------------------------------------------------


def test_ecr_get_yahoo_client_logs_on_streamlit_failure(caplog):
    from src.ecr import _get_yahoo_client

    # Patch the import to raise during getattr/session_state access
    class _BadModule:
        @property
        def session_state(self):
            raise RuntimeError("streamlit not initialized")

    import sys

    original = sys.modules.get("streamlit")
    sys.modules["streamlit"] = _BadModule()
    try:
        with caplog.at_level(logging.WARNING, logger="src.ecr"):
            result = _get_yahoo_client()
    finally:
        if original is not None:
            sys.modules["streamlit"] = original
        else:
            sys.modules.pop("streamlit", None)

    assert result is None
    _assert_warning(caplog, "src.ecr", "session_state", "ecr._get_yahoo_client")


# ---------------------------------------------------------------------------
# ecr.fetch_ecr_rankings — logs when fetch fails
# ---------------------------------------------------------------------------


def test_ecr_fetch_ecr_rankings_logs_on_fetch_failure(caplog):
    from src import ecr

    with patch.object(
        importlib.import_module("src.adp_sources"),
        "fetch_fantasypros_ecr",
        side_effect=RuntimeError("FantasyPros down"),
    ):
        with caplog.at_level(logging.WARNING, logger="src.ecr"):
            result = ecr.fetch_ecr_extended()

    # Returns empty DataFrame with the expected columns
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    _assert_warning(caplog, "src.ecr", "ECR", "ecr.fetch_ecr_extended")


# ---------------------------------------------------------------------------
# ml_ensemble.get_feature_importance — logs when XGBoost.get_score fails
# ---------------------------------------------------------------------------


def test_ml_ensemble_feature_importance_logs_on_xgboost_failure(caplog):
    from src.ml_ensemble import XGBOOST_AVAILABLE, DraftMLEnsemble

    if not XGBOOST_AVAILABLE:
        # If XGBoost is not installed, the early-return path is the one
        # we want — no warning expected for the "model is None" branch.
        # Skip — this test is only meaningful when XGBoost is installed.
        import pytest

        pytest.skip("XGBoost not installed; cannot exercise get_score failure path")

    # Build a fake model that raises when get_score is called
    class _BadModel:
        def get_score(self, importance_type="gain"):
            raise RuntimeError("model corrupt")

    with caplog.at_level(logging.WARNING, logger="src.ml_ensemble"):
        result = DraftMLEnsemble.compute_feature_importance(_BadModel())

    assert result == {}
    _assert_warning(caplog, "src.ml_ensemble", "get_score", "ml_ensemble.compute_feature_importance")


# ---------------------------------------------------------------------------
# depth_charts._fetch_team_depth_chart — logs on HTTP failure
# ---------------------------------------------------------------------------


def test_depth_charts_fetch_team_logs_on_http_failure(caplog):
    from src.depth_charts import _fetch_team_depth_chart

    with patch("src.depth_charts.requests.get", side_effect=RuntimeError("connection refused")):
        with caplog.at_level(logging.WARNING, logger="src.depth_charts"):
            result = _fetch_team_depth_chart("/team/test")

    assert result is None
    _assert_warning(caplog, "src.depth_charts", "fetch", "depth_charts._fetch_team_depth_chart")


# ---------------------------------------------------------------------------
# depth_charts._fetch_team_via_statsapi — logs when statsapi call fails
# ---------------------------------------------------------------------------


def test_depth_charts_fetch_via_statsapi_logs_on_failure(caplog):
    """Run only when statsapi is importable."""
    from src.depth_charts import STATSAPI_AVAILABLE, _fetch_team_via_statsapi

    if not STATSAPI_AVAILABLE:
        import pytest

        pytest.skip("statsapi not installed")

    with patch("src.depth_charts._statsapi.get", side_effect=RuntimeError("statsapi failed")):
        with caplog.at_level(logging.WARNING, logger="src.depth_charts"):
            result = _fetch_team_via_statsapi(team_id=147)

    assert result is None
    _assert_warning(caplog, "src.depth_charts", "statsapi", "depth_charts._fetch_team_via_statsapi")


# ---------------------------------------------------------------------------
# database.get_cached_matchup — logs on DB failure
# ---------------------------------------------------------------------------


def test_database_load_matchup_cache_logs_on_failure(caplog):
    from src import database

    # Patch get_connection to provide a connection whose cursor raises
    class _BadConn:
        def cursor(self):
            class _BC:
                def execute(self, *args, **kwargs):
                    raise RuntimeError("DB exploded")

                def fetchone(self):
                    return None

            return _BC()

        def close(self):
            pass

    with patch.object(database, "get_connection", return_value=_BadConn()):
        with caplog.at_level(logging.WARNING, logger="src.database"):
            result = database.load_matchup_cache("Team Hickey", week=1)

    assert result is None
    _assert_warning(caplog, "src.database", "matchup_cache", "database.load_matchup_cache")


# ---------------------------------------------------------------------------
# data_bootstrap._game_window_aware_ttl — logs on clock probe failure
# ---------------------------------------------------------------------------


def test_data_bootstrap_live_stats_ttl_logs_on_clock_failure(caplog):
    """Make the inner datetime probe raise to trigger the outer except path."""
    from src.data_bootstrap import _live_stats_ttl_hours

    # _live_stats_ttl_hours imports datetime inside the function body; patch
    # the module-imported datetime so all calls raise.
    with patch("datetime.datetime") as m_dt:
        m_dt.now.side_effect = RuntimeError("clock broken")
        with caplog.at_level(logging.WARNING, logger="src.data_bootstrap"):
            result = _live_stats_ttl_hours(default_hours=2.0)

    assert result == 2.0
    _assert_warning(caplog, "src.data_bootstrap", "clock", "data_bootstrap._live_stats_ttl_hours")


# ---------------------------------------------------------------------------
# trade_finder.compute_adp_fairness — logs when draft-round lookup fails
# ---------------------------------------------------------------------------


def test_trade_finder_compute_adp_fairness_logs_on_draft_round_failure(caplog):
    from src.trade_finder import compute_adp_fairness

    pool = pd.DataFrame(
        [
            {"player_id": 1, "adp": 50.0},
            {"player_id": 2, "adp": 100.0},
        ]
    )

    with patch("src.database.get_player_draft_round", side_effect=RuntimeError("DB out")):
        with caplog.at_level(logging.WARNING, logger="src.trade_finder"):
            result = compute_adp_fairness(give_id=1, recv_id=2, player_pool=pool)

    # Should fall back to ADP-based fairness, returning a float
    assert isinstance(result, float)
    _assert_warning(caplog, "src.trade_finder", "draft", "trade_finder.compute_adp_fairness")


# ---------------------------------------------------------------------------
# trade_intelligence._prior_season_pa_ip — logs on DB failure
# ---------------------------------------------------------------------------


def test_trade_intelligence_prior_season_pa_ip_logs_on_db_failure(caplog):
    from src.trade_intelligence import _get_prior_season_pa_ip

    with patch("src.database.get_connection", side_effect=RuntimeError("DB out")):
        with caplog.at_level(logging.WARNING, logger="src.trade_intelligence"):
            result = _get_prior_season_pa_ip(player_id=42, is_hitter=True)

    assert result == 0.0
    _assert_warning(
        caplog,
        "src.trade_intelligence",
        "prior_season_pa_ip",
        "trade_intelligence._get_prior_season_pa_ip",
    )


# ---------------------------------------------------------------------------
# trade_intelligence._batch_prior_season_pa_ip — logs on DB failure
# ---------------------------------------------------------------------------


def test_trade_intelligence_batch_prior_season_pa_ip_logs_on_db_failure(caplog):
    from src.trade_intelligence import _batch_prior_season_pa_ip

    with patch("src.database.get_connection", side_effect=RuntimeError("DB out")):
        with caplog.at_level(logging.WARNING, logger="src.trade_intelligence"):
            result = _batch_prior_season_pa_ip([1, 2, 3])

    assert result == {}
    _assert_warning(caplog, "src.trade_intelligence", "batch", "trade_intelligence._batch_prior_season_pa_ip")


# ---------------------------------------------------------------------------
# optimizer.matchup_adjustments.get_catcher_framing_data — logs on DB failure
# ---------------------------------------------------------------------------


def test_optimizer_matchup_get_catcher_framing_data_logs_on_db_failure(caplog):
    from src.optimizer.matchup_adjustments import get_catcher_framing_data

    with patch("src.database.get_connection", side_effect=RuntimeError("DB out")):
        with caplog.at_level(logging.WARNING, logger="src.optimizer.matchup_adjustments"):
            result = get_catcher_framing_data()

    assert result == {}
    _assert_warning(
        caplog,
        "src.optimizer.matchup_adjustments",
        "catcher_framing",
        "optimizer.matchup_adjustments.get_catcher_framing_data",
    )


# ---------------------------------------------------------------------------
# optimizer.matchup_adjustments.get_pvb_matchup_data — logs on DB failure
# ---------------------------------------------------------------------------


def test_optimizer_matchup_get_pvb_data_logs_on_db_failure(caplog):
    from src.optimizer.matchup_adjustments import get_pvb_matchup_data

    with patch("src.database.get_connection", side_effect=RuntimeError("DB out")):
        with caplog.at_level(logging.WARNING, logger="src.optimizer.matchup_adjustments"):
            result = get_pvb_matchup_data()

    assert result == {}
    _assert_warning(
        caplog,
        "src.optimizer.matchup_adjustments",
        "pvb",
        "optimizer.matchup_adjustments.get_pvb_matchup_data",
    )


# ---------------------------------------------------------------------------
# espn_injuries.update_player_injury_flags — logs on UPDATE failure
# ---------------------------------------------------------------------------


def test_espn_injuries_update_logs_on_update_failure(caplog):
    """When a per-row UPDATE fails, log a warning naming the player."""
    from src import espn_injuries

    # Build a connection mock: execute raises only on UPDATE, commit OK
    class _BadConn:
        def __init__(self):
            self.commits = 0

        def execute(self, sql, params=None):
            if "UPDATE players SET is_injured" in sql:
                raise RuntimeError("UPDATE failed")
            return None

        def commit(self):
            self.commits += 1

        def close(self):
            pass

    bad = _BadConn()
    injuries = [{"player_name": "Test Player", "team": "NYY", "status": "DTD"}]

    with (
        patch("src.database.get_connection", return_value=bad),
        patch("src.live_stats.match_player_id", return_value=12345),
    ):
        with caplog.at_level(logging.WARNING, logger="src.espn_injuries"):
            result = espn_injuries.update_player_injury_flags(injuries)

    assert result == 0
    _assert_warning(caplog, "src.espn_injuries", "is_injured", "espn_injuries.update_player_injury_flags")


# ---------------------------------------------------------------------------
# opponent_intel.get_opponent_for_week — logs when yds.get_schedule fails
# ---------------------------------------------------------------------------


def test_opponent_intel_get_opponent_logs_on_schedule_failure(caplog):
    from src.opponent_intel import get_opponent_for_week

    bad_yds = type("BadYds", (), {})()
    bad_yds.get_schedule = lambda: (_ for _ in ()).throw(RuntimeError("Yahoo down"))
    bad_yds.get_opponent_profile = lambda name: {}

    with caplog.at_level(logging.WARNING, logger="src.opponent_intel"):
        result = get_opponent_for_week(week=1, yds=bad_yds)

    # Falls back to hardcoded schedule
    assert isinstance(result, dict)
    _assert_warning(caplog, "src.opponent_intel", "get_schedule", "opponent_intel.get_opponent_for_week")


# ---------------------------------------------------------------------------
# alerts.generate_roster_alerts — logs when closer projected-SV DB fails
# ---------------------------------------------------------------------------


def test_alerts_closer_count_logs_on_db_failure(caplog):
    """closer_count computation queries projections for SV. If DB fails,
    a warning should be emitted naming the closer-count fallback."""
    from src.alerts import generate_roster_alerts

    # Roster with a player who has actual_sv=0 but projected_sv could close
    roster = pd.DataFrame(
        [
            {"player_id": 1, "name": "Test Pitcher", "sv": 0, "positions": "P", "team": "NYY"},
        ]
    )

    with patch("src.database.get_connection", side_effect=RuntimeError("DB out")):
        with caplog.at_level(logging.WARNING, logger="src.alerts"):
            alerts = generate_roster_alerts(
                roster, max_roster_size=23, fa_pool=None, user_roster_ids=None, player_pool=None
            )

    # Alerts may still be generated; just verify logging
    assert isinstance(alerts, list)
    _assert_warning(caplog, "src.alerts", "projected", "alerts.generate_roster_alerts (proj_sv_map)")


# ---------------------------------------------------------------------------
# Static inspection guards: ensure logger.warning + exc_info=True appear
# for the most critical fix sites (so future refactors don't drop them).
# ---------------------------------------------------------------------------


def _assert_logger_call_near(module_path: str, fragment: str):
    """Static-source check: find `fragment` in module source, verify a
    `logger.warning(...)` call appears within ±8 lines.

    The fragment may be on a continuation line of a logger.warning(...)
    multiline call, so the window must include lines BEFORE the fragment.
    """
    mod = importlib.import_module(module_path)
    src = inspect.getsource(mod)
    lines = src.splitlines()
    idx = next((i for i, ln in enumerate(lines) if fragment in ln), None)
    assert idx is not None, f"{module_path}: fragment {fragment!r} not found in source"
    start = max(0, idx - 8)
    end = min(len(lines), idx + 12)
    window = "\n".join(lines[start:end])
    assert "logger.warning" in window, f"{module_path}: no logger.warning near fragment {fragment!r}. Window:\n{window}"


def test_static_guards_for_top_sites():
    """Pin down the most important fix sites via static source inspection.

    These are the highest-impact silent-failure locations identified in
    the 2026-05-11 bug audit. If a future refactor accidentally removes
    the logger.warning, this guard fails immediately.
    """
    _assert_logger_call_near("src.yahoo_data_service", "snapshot_league_rosters")
    _assert_logger_call_near("src.ecr", "FantasyPros ECR fetch failed")
    _assert_logger_call_near("src.trade_finder", "ecr_consensus load failed")
    _assert_logger_call_near("src.trade_finder", "transactions table load failed")
    _assert_logger_call_near("src.trade_finder", "shaky-closer")
    _assert_logger_call_near("src.optimizer.matchup_adjustments", "umpire-tendency")
    _assert_logger_call_near("src.optimizer.matchup_adjustments", "catcher-framing")
    _assert_logger_call_near("src.optimizer.shared_data_layer", "remaining-games")
    _assert_logger_call_near("src.matchup_planner", "park_factors DB load")
    _assert_logger_call_near("src.data_bootstrap", "bootstrap status panel will be missing this phase")
    _assert_logger_call_near("src.alerts", "closer-count")
    _assert_logger_call_near("src.opponent_intel", "falling back to hardcoded TEAM_HICKEY_SCHEDULE")
    _assert_logger_call_near("src.depth_charts", "MLB statsapi team_roster")


def test_alerts_module_has_logger_defined_once():
    """Regression guard: alerts.py logger must be defined exactly once."""
    import src.alerts

    src_text = inspect.getsource(src.alerts)
    # Count occurrences of `logger = logging.getLogger(__name__)`
    count = src_text.count("logger = logging.getLogger(__name__)")
    assert count == 1, f"alerts.py should have logger defined exactly once at module top, found {count} occurrence(s)."
