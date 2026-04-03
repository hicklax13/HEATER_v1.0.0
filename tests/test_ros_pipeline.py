"""Tests for FanGraphs ROS projection fetching in the data pipeline."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline import (
    ROS_SYSTEM_MAP,
    ROS_SYSTEMS,
    SYSTEM_MAP,
    SYSTEMS,
    fetch_all_projections,
)

# ── ROS_SYSTEM_MAP tests ─────────────────────────────────────────────


def test_ros_system_map_has_3_entries():
    """ROS_SYSTEM_MAP should have exactly 3 entries (steamer, zips, depthcharts)."""
    assert len(ROS_SYSTEM_MAP) == 3


def test_ros_system_map_values():
    """ROS_SYSTEM_MAP values should be steamer_ros, zips_ros, depthcharts_ros."""
    expected_values = {"steamer_ros", "zips_ros", "depthcharts_ros"}
    assert set(ROS_SYSTEM_MAP.values()) == expected_values


def test_ros_systems_list():
    """ROS_SYSTEMS should match the keys of ROS_SYSTEM_MAP."""
    assert set(ROS_SYSTEMS) == set(ROS_SYSTEM_MAP.keys())
    assert len(ROS_SYSTEMS) == len(ROS_SYSTEM_MAP)


# ── fetch_all_projections ROS integration ─────────────────────────────


@patch("src.data_pipeline.fetch_projections")
def test_fetch_all_projections_includes_ros(mock_fetch):
    """fetch_all_projections should call fetch_projections for ROS systems."""
    mock_fetch.return_value = (
        pd.DataFrame({"name": ["Test"]}),
        [{"PlayerName": "Test"}],
    )

    projections, raw = fetch_all_projections()

    # Collect all fg_system args passed to fetch_projections
    called_systems = [call.args[0] for call in mock_fetch.call_args_list]

    # Verify ROS systems were called
    for ros_sys in ROS_SYSTEMS:
        assert ros_sys in called_systems, f"ROS system {ros_sys} was not fetched"

    # Verify base systems were also called
    for base_sys in SYSTEMS:
        assert base_sys in called_systems, f"Base system {base_sys} was not fetched"


@patch("src.data_pipeline.fetch_projections")
def test_ros_fetch_failure_non_fatal(mock_fetch):
    """ROS fetch failure should not prevent base systems from working."""
    from src.data_pipeline import FetchError

    def side_effect(fg_system, stats):
        # Fail all ROS fetches
        if fg_system in ROS_SYSTEMS:
            raise FetchError(f"ROS {fg_system} not available")
        return (
            pd.DataFrame({"name": ["Player"]}),
            [{"PlayerName": "Player"}],
        )

    mock_fetch.side_effect = side_effect

    projections, raw = fetch_all_projections()

    # Base systems should still succeed
    assert len(projections) > 0, "No base projections returned despite ROS failures"

    # Verify no ROS keys in results (since they all failed)
    for key in projections:
        for ros_val in ROS_SYSTEM_MAP.values():
            assert not key.startswith(ros_val), f"ROS key {key} should not be in results after failure"


@patch("src.data_pipeline.fetch_projections")
def test_store_projections_ros_system_name(mock_fetch):
    """ROS projection keys should use the mapped DB system name."""
    mock_fetch.return_value = (
        pd.DataFrame({"name": ["Player"]}),
        [{"PlayerName": "Player"}],
    )

    projections, _ = fetch_all_projections()

    # Check that ROS keys use the DB system name from ROS_SYSTEM_MAP
    expected_ros_keys = set()
    for db_sys in ROS_SYSTEM_MAP.values():
        expected_ros_keys.add(f"{db_sys}_bat")
        expected_ros_keys.add(f"{db_sys}_pit")

    # All expected ROS keys should be present (since mock succeeds)
    for key in expected_ros_keys:
        assert key in projections, f"Expected ROS key {key} not in projections"
