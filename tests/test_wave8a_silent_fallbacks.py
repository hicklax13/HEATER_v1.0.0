# tests/test_wave8a_silent_fallbacks.py
"""Wave 8a / Group 2: silent fallbacks and error-swallowing.

Audit IDs covered:
  - D5B-031 (`src/weekly_report.py:184`) — `check_daily_lineup`
    returns `[]` when `todays_games is None` with no log.
  - D5B-034 (`src/alerts.py:443`) — `compute_swap_impacts` outer
    bare `except Exception: return []` swallows any failure.
  - D5B-042 (`src/leaders.py:517`) — `compute_projection_skew`
    bare `except: pass` swallows DB-failure → all players get
    `projection_skew=""`.
  - D4B-005 (`src/engine/output/trade_evaluator.py:846`) — DB roster
    rebuild silently swallows errors → uniform category weights
    without UI signal.
  - D4B-020 (`src/trade_finder.py:62`) — `_player_sgp_volume_aware`
    returns `0.0` silently when player_id not in pool.

Each fix preserves the existing return value (falsy default) but
adds a `logger.warning` / `logger.info` call so operators observe
the failure path. Tests assert both the return shape AND the log.
"""

from __future__ import annotations

import logging

import pandas as pd

# ---------------------------------------------------------------------------
# D5B-031 — weekly_report.check_daily_lineup logs when todays_games is None
# ---------------------------------------------------------------------------


def test_d5b031_check_daily_lineup_logs_when_no_schedule(caplog):
    """When todays_games is None, the function should log INFO explaining
    why it's skipping (and still return [])."""
    from src.weekly_report import check_daily_lineup

    roster = pd.DataFrame(
        [
            {"name": "Test Player", "positions": "1B", "team": "NYY", "roster_slot": "1B"},
        ]
    )

    with caplog.at_level(logging.INFO, logger="src.weekly_report"):
        result = check_daily_lineup(roster, todays_games=None)

    assert result == [], f"D5B-031: expected empty list, got {result!r}"

    log_messages = [r.message for r in caplog.records if r.levelno >= logging.INFO]
    matched = any(
        ("schedule" in m.lower() or "todays_games" in m.lower() or "today" in m.lower()) for m in log_messages
    )
    assert matched, (
        f"D5B-031: expected INFO log mentioning schedule/today when todays_games is None. Got: {log_messages!r}"
    )


# ---------------------------------------------------------------------------
# D5B-034 — alerts.compute_swap_impacts logs on failure
# ---------------------------------------------------------------------------


def test_d5b034_compute_swap_impacts_logs_on_failure(caplog):
    """When the inner pipeline throws (e.g. broken player_pool), the outer
    except branch should LOG a WARNING with the traceback, not silently
    swallow."""
    from src.alerts import compute_swap_impacts

    # Construct inputs that will pass the empty-check but blow up downstream:
    # roster has player_ids, pool has them too, but a row has no usable cols
    # which forces sgp_calc to throw during player_sgp.
    roster = pd.DataFrame(
        [
            {"player_id": 1, "selected_position": "1B"},
            {"player_id": 2, "selected_position": "BN"},
        ]
    )
    # player_pool with broken/missing columns — sgp_calc.player_sgp will choke
    # because we have no proj/ytd stat columns at all.
    pool = pd.DataFrame(
        [
            {"player_id": 1, "is_hitter": "not-a-number", "name": "Broken Hitter"},
            {"player_id": 2, "is_hitter": "not-a-number", "name": "Broken Bench"},
        ]
    )

    # Force the import inside compute_swap_impacts to fail by patching it out.
    # The function does `from src.valuation import LeagueConfig, SGPCalculator`
    # inside the try block. Patching the SGPCalculator to raise on construction
    # is the cleanest way to trigger the outer except path.
    from unittest.mock import patch

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated DB failure")

    with patch("src.valuation.SGPCalculator", side_effect=_boom):
        with caplog.at_level(logging.WARNING, logger="src.alerts"):
            result = compute_swap_impacts(roster, pool, config=None)

    assert result == [], f"D5B-034: expected empty list on failure, got {result!r}"

    log_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING and r.name == "src.alerts"]
    assert log_messages, (
        f"D5B-034: expected WARNING log when compute_swap_impacts fails. "
        f"Got no warnings from src.alerts. caplog.records: "
        f"{[(r.name, r.levelname, r.message) for r in caplog.records]}"
    )
    matched = any(
        ("swap_impacts" in m.lower() or "compute_swap" in m.lower() or "failed" in m.lower()) for m in log_messages
    )
    assert matched, f"D5B-034: WARNING log should mention compute_swap_impacts / failure. Got: {log_messages!r}"


# ---------------------------------------------------------------------------
# D5B-042 — leaders.compute_projection_skew logs on DB failure
# ---------------------------------------------------------------------------


def test_d5b042_compute_projection_skew_logs_on_db_failure(caplog):
    """When the projections DB query fails, the function should log a
    WARNING and still return the pool with empty projection_skew (not
    raise)."""
    from src.leaders import compute_projection_skew

    pool = pd.DataFrame([{"player_id": 1, "name": "Test"}])

    # Patch get_connection to raise, simulating DB failure
    from unittest.mock import patch

    with patch("src.database.get_connection", side_effect=RuntimeError("DB exploded")):
        with caplog.at_level(logging.WARNING, logger="src.leaders"):
            result = compute_projection_skew(pool)

    # Falsy default behavior preserved
    assert "projection_skew" in result.columns
    assert (result["projection_skew"] == "").all()

    log_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING and r.name == "src.leaders"]
    assert log_messages, "D5B-042: expected WARNING log when DB query fails. Got no warnings from src.leaders."
    matched = any(
        ("projection" in m.lower() or "skew" in m.lower() or "db" in m.lower() or "failed" in m.lower())
        for m in log_messages
    )
    assert matched, f"D5B-042: WARNING log should describe the failure. Got: {log_messages!r}"


# ---------------------------------------------------------------------------
# D4B-005 — trade_evaluator DB roster rebuild logs on failure
# ---------------------------------------------------------------------------


def test_d4b005_trade_evaluator_db_rebuild_logs_on_failure(caplog):
    """When standings lack stat categories AND the DB roster rebuild
    throws, the trade evaluator should log a WARNING (or ERROR) so
    operators know category weights are uniform / strategic context is
    missing.

    Regression guard via static inspection of the source: the relevant
    logger call must surround the literal message and use exc_info=True
    so the traceback is captured."""
    import inspect

    import src.engine.output.trade_evaluator as te

    src_text = inspect.getsource(te)
    # The log message exists somewhere in the module
    assert "Failed to compute team totals from league_rosters" in src_text, (
        "D4B-005: trade_evaluator should log when DB roster rebuild fails. "
        "Look for the literal message 'Failed to compute team totals from league_rosters'."
    )
    # Find the line with the message and capture a small surrounding window
    lines = src_text.splitlines()
    log_line_idx = next(
        (i for i, line in enumerate(lines) if "Failed to compute team totals from league_rosters" in line),
        None,
    )
    assert log_line_idx is not None, "D4B-005: log line not found"

    # The logger call may span several lines (logger.warning(\n    "msg",\n    exc,\n    exc_info=True\n))
    # Look at a window of the previous + next 6 lines.
    window_start = max(0, log_line_idx - 2)
    window_end = min(len(lines), log_line_idx + 6)
    window = "\n".join(lines[window_start:window_end])

    assert "logger.warning" in window or "logger.error" in window, (
        f"D4B-005: must use logger.warning or logger.error near the message. Got window:\n{window}"
    )
    assert "exc_info=True" in window, f"D4B-005: must propagate traceback via exc_info=True. Got window:\n{window}"


# ---------------------------------------------------------------------------
# D4B-020 — trade_finder._player_sgp_volume_aware logs when player missing
# ---------------------------------------------------------------------------


def test_d4b020_player_sgp_volume_aware_logs_when_player_missing(caplog):
    """When player_id not in pool, the function should log a WARNING and
    still return 0.0 (don't raise — callers iterate over many ids)."""
    from src.trade_finder import _player_sgp_volume_aware
    from src.valuation import LeagueConfig

    # Pool with player 1 but not 999
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Real Player",
                "is_hitter": 1,
                "r": 50,
                "hr": 10,
                "rbi": 30,
                "sb": 5,
                "ab": 200,
                "h": 60,
                "bb": 25,
                "hbp": 1,
                "sf": 1,
                "pa": 230,
            }
        ]
    )

    with caplog.at_level(logging.WARNING, logger="src.trade_finder"):
        result = _player_sgp_volume_aware(pid=999, player_pool=pool, config=LeagueConfig())

    assert result == 0.0, f"D4B-020: expected 0.0 for missing player, got {result!r}"

    log_messages = [r.message for r in caplog.records if r.levelno >= logging.WARNING and r.name == "src.trade_finder"]
    assert log_messages, (
        "D4B-020: expected WARNING log when player_id missing from pool. Got no warnings from src.trade_finder."
    )
    matched = any(
        ("missing" in m.lower() or "player_id" in m.lower() or "not in pool" in m.lower()) for m in log_messages
    )
    assert matched, f"D4B-020: WARNING log should mention missing player_id. Got: {log_messages!r}"


def test_d4b020_player_sgp_volume_aware_does_not_raise_on_missing(caplog):
    """Regression guard: callers loop over many pids and we don't want
    a single missing id to crash the trade-finder run."""
    from src.trade_finder import _player_sgp_volume_aware
    from src.valuation import LeagueConfig

    empty_pool = pd.DataFrame(columns=["player_id", "is_hitter"])

    # Should NOT raise — just return 0.0 and log
    result = _player_sgp_volume_aware(pid=42, player_pool=empty_pool, config=LeagueConfig())
    assert result == 0.0
