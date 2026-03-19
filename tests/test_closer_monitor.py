"""Tests for closer depth chart monitor."""

from __future__ import annotations

import pandas as pd

from src.closer_monitor import build_closer_grid, compute_job_security, get_security_color


def test_job_security_high():
    sec = compute_job_security(hierarchy_confidence=0.9, projected_sv=35)
    assert sec >= 0.8


def test_job_security_low():
    sec = compute_job_security(hierarchy_confidence=0.2, projected_sv=5)
    assert sec < 0.4


def test_job_security_clamp_upper():
    sec = compute_job_security(hierarchy_confidence=1.0, projected_sv=50)
    assert sec <= 1.0


def test_job_security_clamp_lower():
    sec = compute_job_security(hierarchy_confidence=0.0, projected_sv=0)
    assert sec >= 0.0


def test_security_color_green():
    assert get_security_color(0.8) == "#2d6a4f"


def test_security_color_yellow():
    assert get_security_color(0.5) == "#ff9f1c"


def test_security_color_red():
    assert get_security_color(0.2) == "#e63946"


def test_build_grid_basic():
    depth_data = {
        "NYY": {"closer": "Clay Holmes", "setup": ["Jonathan Loaisiga"], "closer_confidence": 0.8},
        "BOS": {"closer": "Kenley Jansen", "setup": ["Chris Martin"], "closer_confidence": 0.9},
    }
    pool = pd.DataFrame(
        {
            "name": ["Clay Holmes", "Kenley Jansen"],
            "team": ["NYY", "BOS"],
            "sv": [30, 35],
            "era": [3.50, 3.20],
            "whip": [1.20, 1.10],
            "is_hitter": [False, False],
        }
    )
    grid = build_closer_grid(depth_data, pool)
    assert len(grid) == 2
    assert all("team" in item and "closer_name" in item and "job_security" in item for item in grid)


def test_build_grid_empty():
    assert build_closer_grid({}) == []


def test_build_grid_committee():
    depth_data = {"CHC": {"closer": "Committee", "setup": ["A", "B"], "closer_confidence": 0.3}}
    grid = build_closer_grid(depth_data)
    assert len(grid) == 1
    assert grid[0]["job_security"] < 0.5
