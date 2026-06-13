"""Task 4.5 — AZ→ARI normalization + header count honesty + dual-SV label.

Three failing tests that drive the three targeted fixes:

1. AZ→ARI stat mismatch: build_closer_grid must normalize the *team* key
   in depth_data before comparing it to pool["team"], so an "AZ" depth entry
   matches the "ARI" row in the pool (or vice versa).

2. Header count honesty: the page must NOT say "30-team closer depth chart"
   when fewer than 30 teams have data.  It must expose the actual count so
   the UI can show "N of 30 teams" or similar.

3. Dual-SV label: the primary stat block SV label must read "PROJ SV"
   (or "PROJ" at minimum) to distinguish it from the "2026 ACTUAL · N SV"
   line, so users can tell the two figures apart.
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

PAGE_SRC = pathlib.Path(__file__).parent.parent / "pages" / "3_Closer_Monitor.py"
_page_text = PAGE_SRC.read_text(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# 1. AZ→ARI normalization in build_closer_grid
# ─────────────────────────────────────────────────────────────────────────────


def _make_ari_depth():
    """Depth entry using the "AZ" abbreviation (Roster Resource / FanGraphs variant)."""
    return {
        "AZ": {
            "closer": "Paul Sewald",
            "setup": ["Kevin Ginkel"],
            "closer_confidence": 0.80,
        }
    }


def _make_ari_pool():
    """Player pool row using the canonical "ARI" abbreviation."""
    return pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Paul Sewald",
                "team": "ARI",  # canonical MLB Stats API code
                "sv": 15,
                "era": 3.47,
                "whip": 0.73,
                "mlb_id": 605218,
            }
        ]
    )


def test_az_key_matches_ari_pool_stats():
    """build_closer_grid with an 'AZ' depth entry must find the 'ARI' pool row.

    This is the ARI blank-vs-populated contradiction: depth_data keyed "AZ" but
    pool row has team="ARI".  The function must normalize the team key before
    doing the pool lookup so projected_sv/era/whip are populated.
    """
    from src.closer_monitor import build_closer_grid

    grid = build_closer_grid(_make_ari_depth(), _make_ari_pool())
    assert len(grid) == 1, f"Expected 1 grid row; got {grid}"
    row = grid[0]
    assert row["projected_sv"] == 15, (
        f"AZ→ARI mismatch: projected_sv={row['projected_sv']} (expected 15). "
        "build_closer_grid must normalize team keys before pool lookup."
    )
    assert abs(row["era"] - 3.47) < 0.01, f"era mismatch: {row['era']}"
    assert abs(row["whip"] - 0.73) < 0.01, f"whip mismatch: {row['whip']}"


def test_ari_key_matches_az_pool_stats():
    """Inverse: depth_data keyed 'ARI', pool row has team='AZ' — same fix covers this."""
    from src.closer_monitor import build_closer_grid

    depth = {
        "ARI": {
            "closer": "Paul Sewald",
            "setup": [],
            "closer_confidence": 0.80,
        }
    }
    pool = pd.DataFrame(
        [
            {
                "player_id": 1,
                "name": "Paul Sewald",
                "team": "AZ",  # non-canonical
                "sv": 15,
                "era": 3.47,
                "whip": 0.73,
                "mlb_id": 605218,
            }
        ]
    )
    grid = build_closer_grid(depth, pool)
    assert len(grid) == 1
    row = grid[0]
    assert row["projected_sv"] == 15, f"ARI→AZ mismatch: projected_sv={row['projected_sv']} (expected 15)"


def test_az_depth_entry_output_team_is_normalized():
    """The grid row's 'team' field must be the canonical 'ARI', not 'AZ'."""
    from src.closer_monitor import build_closer_grid

    grid = build_closer_grid(_make_ari_depth(), _make_ari_pool())
    assert len(grid) == 1
    assert grid[0]["team"] == "ARI", f"Expected grid row team='ARI', got '{grid[0]['team']}'"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Header count honesty
# ─────────────────────────────────────────────────────────────────────────────


def test_page_does_not_hardcode_30_team_in_reco_banner():
    """render_reco_banner must NOT say '30-team closer depth chart'.

    The page may only have ~21 teams with data, so '30-team' is misleading.
    Replace the literal with a dynamic count or 'Closer depth — N of 30 teams'.
    """
    # The offending string we found in the source.
    assert "30-team closer depth chart" not in _page_text, (
        "pages/3_Closer_Monitor.py must not hardcode '30-team closer depth chart' "
        "in render_reco_banner — use a dynamic count like 'N of 30 teams with data'."
    )


def test_page_shows_of_30_teams_pattern():
    """Page source must reference '30' alongside a dynamic count marker
    (e.g. 'of 30 teams' or '{len(grid)} of 30') rather than just 'N teams'."""
    assert "of 30" in _page_text or "/ 30" in _page_text, (
        "pages/3_Closer_Monitor.py must include 'of 30' (or '/ 30') to give "
        "context about how many of the 30 MLB teams have data."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dual-SV label — projected SV must be labeled distinctly
# ─────────────────────────────────────────────────────────────────────────────


def test_page_primary_sv_label_is_proj():
    """The primary stat block SV column header must say 'PROJ SV' (not bare 'SV').

    There are two SV figures on each card:
      - The main stat block (projected/blended SV from player_pool)
      - The '2026 ACTUAL · N SV' green line below

    The primary block currently labels its header as just 'SV', which is
    indistinguishable from the actual figure.  It must say 'PROJ SV' or
    include 'PROJ' so users can tell them apart.
    """
    # The page renders the SV column header in the stat block as plain "SV" today.
    # After the fix it must include "PROJ" nearby.
    assert (
        "PROJ SV" in _page_text
        or "PROJ</div>" in _page_text
        or ">PROJ SV<" in _page_text
        or "PROJ&nbsp;SV" in _page_text
    ), (
        "pages/3_Closer_Monitor.py: the primary SV stat column header must read "
        "'PROJ SV' (not bare 'SV') to distinguish it from the '2026 ACTUAL' line. "
        "Expected 'PROJ SV' somewhere in the page source."
    )
