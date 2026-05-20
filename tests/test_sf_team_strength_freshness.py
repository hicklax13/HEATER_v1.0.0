"""Regression test for Bonus 4: shared_data_layer records team_strength freshness.

Pre-fix: build_optimizer_data_context() called _load_team_strength(ctx) but
never invoked tracker.record("team_strength", ...).  Every other major data
source (roster, projections, free_agents, matchup, schedule, opposing_pitchers,
weather, recent_form) recorded freshness.  The optimizer freshness UI showed
UNKNOWN for team_strength forever.

This test verifies that when team_strength is loaded successfully, the
freshness tracker records it under the "team_strength" key.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizer.shared_data_layer import build_optimizer_context


@pytest.fixture
def fake_yds():
    """Stub Yahoo data service that returns empty data — keeps the pipeline cheap."""
    yds = MagicMock()
    yds.get_rosters.return_value = pd.DataFrame()
    yds.get_matchup.return_value = None
    yds.get_free_agents.return_value = pd.DataFrame()
    yds.get_full_league_schedule.return_value = {}
    yds.get_standings.return_value = pd.DataFrame()
    yds.get_transactions.return_value = pd.DataFrame()
    return yds


def test_team_strength_freshness_recorded_when_data_present(fake_yds):
    """When _load_team_strength populates ctx.team_strength, tracker.record fires.

    2026-05-19: patch _populate_from_refresh_log for isolation (see sibling test).
    """

    def _fake_loader(ctx):
        ctx.team_strength = {
            "NYY": {"wrc_plus": 110, "fip": 3.85},
            "BOS": {"wrc_plus": 102, "fip": 4.12},
        }

    with (
        patch(
            "src.optimizer.data_freshness.DataFreshnessTracker._populate_from_refresh_log",
            return_value=None,
        ),
        patch("src.optimizer.shared_data_layer._load_team_strength", side_effect=_fake_loader),
    ):
        ctx = build_optimizer_context(scope="today", yds=fake_yds)

    assert "team_strength" in ctx.data_timestamps, (
        f"Expected 'team_strength' in data_timestamps, got keys: {list(ctx.data_timestamps.keys())}.  "
        f"shared_data_layer.py is failing to call tracker.record('team_strength', ...) after "
        f"_load_team_strength."
    )

    record = ctx.data_timestamps["team_strength"]
    assert record["status"] == "fresh", f"Expected fresh status, got {record['status']}"
    assert record["ttl_hours"] == 24.0
    assert "pybaseball" in record["source_label"].lower()


def test_team_strength_freshness_NOT_recorded_when_empty(fake_yds):
    """When _load_team_strength leaves ctx.team_strength empty, tracker should skip recording.

    2026-05-19: patch DataFreshnessTracker._populate_from_refresh_log to no-op
    so the tracker starts clean. Without this, the local SQLite's refresh_log
    pre-seeds team_strength + ~28 other sources, leaking into the assertion.
    """

    def _empty_loader(ctx):
        ctx.team_strength = {}

    with (
        patch(
            "src.optimizer.data_freshness.DataFreshnessTracker._populate_from_refresh_log",
            return_value=None,
        ),
        patch("src.optimizer.shared_data_layer._load_team_strength", side_effect=_empty_loader),
    ):
        ctx = build_optimizer_context(scope="today", yds=fake_yds)

    if "team_strength" in ctx.data_timestamps:
        pytest.fail(
            "team_strength was recorded even though ctx.team_strength was empty. "
            "The fix should guard with `if ctx.team_strength:` like other call sites."
        )
