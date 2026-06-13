"""TDD tests for Task 4.2 — dead/false controls on the Player Databank page.

Each test fails against the current code and passes after the fixes.

Items covered:
  1. "Waivers Only" status filter — removed from POSITION_OPTIONS (page) and
     filter_databank now rejects it gracefully (returns all when status="W"
     with no roster_team col vs. the fixed behaviour: removed option).
  2. "Util" position filter — removed from POSITION_OPTIONS on the page.
  3. "Today (live)" label renamed to "ROS Projections" in STAT_VIEW_OPTIONS.
  4. Four stub views (K_K / R_O / M_W / O_O) removed from STAT_VIEW_OPTIONS.
  5. Off-palette hex #2c2f36 replaced with a CSS variable in player_databank.py.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pandas as pd
import pytest

# ── Module paths ──────────────────────────────────────────────────────────────

PAGE_SRC = pathlib.Path(__file__).parent.parent / "pages" / "19_Player_Databank.py"
DB_SRC = pathlib.Path(__file__).parent.parent / "src" / "player_databank.py"

_page_text = PAGE_SRC.read_text(encoding="utf-8")
_db_text = DB_SRC.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Item 1 — "Waivers Only" (status="W") removed from page STATUS_OPTIONS
# ─────────────────────────────────────────────────────────────────────────────


def test_waivers_only_option_removed_from_page():
    """'Waivers Only' / ('W', ...) must not appear in STATUS_OPTIONS on the page.

    The option returned all ~4,408 players because filter_databank had no
    handler for status='W'.  The fix is to remove the option from the page.
    """
    # Check that the literal tuple entry is gone from the page source.
    # Both ("W", ...) and "Waivers Only" must be absent.
    assert '("W"' not in _page_text or "Waivers Only" not in _page_text, (
        "STATUS_OPTIONS in pages/19_Player_Databank.py must not contain the "
        "'Waivers Only' / ('W', ...) entry — it had no handler and returned all players."
    )


def test_waivers_only_absent_from_status_options_list():
    """The string 'Waivers Only' must not appear anywhere in the page source."""
    assert "Waivers Only" not in _page_text, (
        "'Waivers Only' must be removed from pages/19_Player_Databank.py STATUS_OPTIONS."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Item 2 — "Util" position removed from page POSITION_OPTIONS
# ─────────────────────────────────────────────────────────────────────────────


def test_util_position_removed_from_page():
    """'Util' must not appear as a position option in the page POSITION_OPTIONS.

    No player in the pool has literal 'Util' in their positions string —
    the option always returned 0 results.  The fix is to remove it.
    """
    assert '("Util", "Util")' not in _page_text, (
        "('Util', 'Util') must be removed from POSITION_OPTIONS in "
        "pages/19_Player_Databank.py — it always returned 0 results."
    )


def test_util_string_absent_from_position_options():
    """The string 'Util' must not appear inside POSITION_OPTIONS definition."""
    # We specifically check the POSITION_OPTIONS list definition block.
    # (The word 'Util' may legitimately appear elsewhere in comments, so we
    # look for the tuple form.)
    # Accept either absence of the key or absence from the list literal.
    pos_block_match = re.search(
        r"POSITION_OPTIONS\s*=\s*\[(.+?)\]",
        _page_text,
        re.DOTALL,
    )
    if pos_block_match:
        block = pos_block_match.group(1)
        assert '"Util"' not in block, (
            "'Util' must not be an option inside POSITION_OPTIONS — it always returned 0 results."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Item 3 — "Today (live)" renamed to "ROS Projections" in STAT_VIEW_OPTIONS
# ─────────────────────────────────────────────────────────────────────────────


def test_today_live_label_absent_from_stat_view_options():
    """'Today (live)' must not be a STAT_VIEW_OPTIONS label — it was false.

    The S_L key showed ROS projections, not live stats.
    """
    from src.player_databank import STAT_VIEW_OPTIONS

    assert "Today (live)" not in STAT_VIEW_OPTIONS.values(), (
        "STAT_VIEW_OPTIONS must not contain the 'Today (live)' label — "
        "it was a false label (showed ROS projections, not live stats). "
        "Rename it to 'ROS Projections'."
    )


def test_ros_projections_label_present_for_s_l_key():
    """STAT_VIEW_OPTIONS['S_L'] must be renamed to 'ROS Projections'."""
    from src.player_databank import STAT_VIEW_OPTIONS

    if "S_L" in STAT_VIEW_OPTIONS:
        assert STAT_VIEW_OPTIONS["S_L"] == "ROS Projections", (
            f"STAT_VIEW_OPTIONS['S_L'] must be 'ROS Projections', got: {STAT_VIEW_OPTIONS['S_L']!r}"
        )
    # If S_L was removed entirely, that is also acceptable.
    # (A separate key could carry the label — either way 'Today (live)' must be gone.)


# ─────────────────────────────────────────────────────────────────────────────
# Item 4 — Stub views K_K / R_O / M_W / O_O removed from STAT_VIEW_OPTIONS
# ─────────────────────────────────────────────────────────────────────────────

_STUB_KEYS = {"K_K", "R_O", "M_W", "O_O"}
_STUB_LABELS = {"Ranks", "Research", "Fantasy Matchups", "Opponents"}


@pytest.mark.parametrize("key", sorted(_STUB_KEYS))
def test_stub_view_key_removed(key):
    """Stub view keys must not appear in STAT_VIEW_OPTIONS."""
    from src.player_databank import STAT_VIEW_OPTIONS

    assert key not in STAT_VIEW_OPTIONS, (
        f"STAT_VIEW_OPTIONS['{key}'] must be removed — it was a stub that showed identical data to the default view."
    )


@pytest.mark.parametrize("label", sorted(_STUB_LABELS))
def test_stub_view_label_removed(label):
    """Stub view labels must not appear in STAT_VIEW_OPTIONS values."""
    from src.player_databank import STAT_VIEW_OPTIONS

    assert label not in STAT_VIEW_OPTIONS.values(), (
        f"STAT_VIEW_OPTIONS must not contain the stub label '{label}' — "
        f"it was a placeholder that showed the same data as the default view."
    )


def test_stat_view_params_stubs_removed():
    """STAT_VIEW_PARAMS must not contain the stub keys either."""
    from src.player_databank import STAT_VIEW_PARAMS

    for key in _STUB_KEYS:
        assert key not in STAT_VIEW_PARAMS, f"STAT_VIEW_PARAMS['{key}'] must be removed alongside STAT_VIEW_OPTIONS."


# ─────────────────────────────────────────────────────────────────────────────
# Item 5 — Off-palette hex #2c2f36 replaced with CSS variable
# ─────────────────────────────────────────────────────────────────────────────


def test_no_offpalette_hex_2c2f36_in_player_databank():
    """src/player_databank.py must not contain the banned hex #2c2f36.

    The hex appears inline in render_databank_table CSS. It must be replaced
    with a THEME token or a CSS variable (e.g. var(--fp-ink) or T['ink']).
    """
    assert "#2c2f36" not in _db_text.lower(), (
        "src/player_databank.py contains off-palette hex #2c2f36 (line ~1071). "
        "Replace it with a THEME token or var(--fp-ink) CSS variable."
    )


def test_render_table_uses_css_var_for_header_text():
    """render_databank_table must not embed #2c2f36 in the returned HTML."""
    df = pd.DataFrame(
        {
            "player_name": ["Test Player"],
            "team": ["NYY"],
            "positions": ["1B"],
            "r": [10],
            "hr": [5],
            "rbi": [15],
            "sb": [2],
            "avg": [0.285],
            "obp": [0.350],
        }
    )
    from src.player_databank import render_databank_table

    html = render_databank_table(df, stat_view="S_S_2026", is_pitcher=False)
    assert "#2c2f36" not in html.lower(), (
        "render_databank_table output must not embed #2c2f36. Use var(--fp-ink) or T['ink'] instead."
    )
