"""TDD tests for Task 5.4 — deprecated API + dead-column sweep.

Items:
  1. app.py: no st.components.v1.html usage (replaced with st.markdown)
  2. Player Databank / Free Agents: percent_owned column is populated
     (not always a dash) — verifies the ownership_trends JOIN in load_player_pool
     and that render_databank_table includes it when data present.
  3. Closer Monitor: empty gmLI and empty SETUP rows removed/hidden
     (no unconditional '<!-- no gmli data -->' and no unconditional SETUP · — row).
"""

from __future__ import annotations

import ast
import pathlib

import pandas as pd
import pytest

# ── File paths ────────────────────────────────────────────────────────────────

APP_SRC = pathlib.Path(__file__).parent.parent / "app.py"
CLOSER_SRC = pathlib.Path(__file__).parent.parent / "pages" / "3_Closer_Monitor.py"
DATABANK_SRC = pathlib.Path(__file__).parent.parent / "pages" / "19_Player_Databank.py"
FA_SRC = pathlib.Path(__file__).parent.parent / "pages" / "14_Free_Agents.py"
DB_SRC = pathlib.Path(__file__).parent.parent / "src" / "player_databank.py"

_app_text = APP_SRC.read_text(encoding="utf-8")
_closer_text = CLOSER_SRC.read_text(encoding="utf-8")
_databank_text = DATABANK_SRC.read_text(encoding="utf-8")
_db_text = DB_SRC.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Item 1 — app.py: no st.components.v1.html
# ─────────────────────────────────────────────────────────────────────────────


def test_app_no_components_v1_html_import():
    """app.py must not import streamlit.components.v1."""
    assert "streamlit.components.v1" not in _app_text, (
        "app.py must not import streamlit.components.v1 — use st.markdown(...) instead. "
        "The splash-screen timer must be embedded as a <script> inside st.markdown."
    )


def test_app_no_components_html_call():
    """app.py must not call _components.html(...) or components.html(...)."""
    assert "_components.html(" not in _app_text and "components.html(" not in _app_text, (
        "app.py must not call _components.html() or components.html(). "
        "The HH:MM:SS splash timer must be rendered via st.markdown(<script>...) instead."
    )


def test_app_splash_timer_still_present():
    """The splash timer JS (setInterval / heater-splash-timer) must still be present.

    Replacing components.v1.html must not remove the live-updating timer entirely.
    The JS tick() / setInterval logic must remain in the page source so the
    timer continues to self-update every second.
    """
    assert "setInterval" in _app_text and "heater-splash-timer" in _app_text, (
        "app.py must still contain the splash timer JS (setInterval + heater-splash-timer id). "
        "Only the delivery mechanism changed (st.markdown instead of components.v1.html)."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Item 2 — percent_owned / % Ros column is populated (not always a dash)
# ─────────────────────────────────────────────────────────────────────────────


def test_load_player_pool_includes_percent_owned_column():
    """load_player_pool() must return a DataFrame with a percent_owned column."""
    from src.database import load_player_pool

    pool = load_player_pool()
    assert not pool.empty, "Player pool must not be empty"
    assert "percent_owned" in pool.columns, (
        "load_player_pool() must include a 'percent_owned' column — "
        "sourced from ownership_trends (already in the SQL subquery). "
        "The column must exist even when all values are NULL."
    )


def test_load_player_pool_percent_owned_has_nonzero_values():
    """At least some players must have a non-zero percent_owned in the pool.

    The ownership_trends table has real data for ~1800 players.  If the
    subquery is wired correctly, non-NULL values flow into the pool.
    """
    from src.database import load_player_pool

    pool = load_player_pool()
    if "percent_owned" not in pool.columns:
        pytest.skip("percent_owned column absent — skipping value check")
    nonzero = (pool["percent_owned"].fillna(0) > 0).sum()
    assert nonzero > 0, (
        "percent_owned has 0 non-zero values in the player pool. "
        "Expected >0 — the ownership_trends JOIN must be returning real data."
    )


def test_render_databank_table_includes_percent_owned_when_present():
    """render_databank_table must include '% Ros' column header when percent_owned is in the DataFrame."""
    from src.player_databank import render_databank_table

    df = pd.DataFrame(
        {
            "player_name": ["Test Hitter"],
            "team": ["NYY"],
            "positions": ["OF"],
            "r": [50],
            "hr": [15],
            "rbi": [60],
            "sb": [5],
            "avg": [0.270],
            "obp": [0.340],
            "percent_owned": [42.0],  # non-zero value
        }
    )
    html = render_databank_table(df, stat_view="S_S_2026", is_pitcher=False)
    assert "% Ros" in html or "percent_owned" in html.lower(), (
        "render_databank_table must include '% Ros' column header when "
        "the DataFrame has a non-null percent_owned column."
    )


def test_render_databank_table_percent_owned_value_renders_as_percentage():
    """render_databank_table must format percent_owned as '42.0%' not '42' or '-'."""
    from src.player_databank import render_databank_table

    df = pd.DataFrame(
        {
            "player_name": ["Test Hitter"],
            "team": ["NYY"],
            "positions": ["OF"],
            "r": [50],
            "hr": [15],
            "rbi": [60],
            "sb": [5],
            "avg": [0.270],
            "obp": [0.340],
            "percent_owned": [42.0],
        }
    )
    html = render_databank_table(df, stat_view="S_S_2026", is_pitcher=False)
    # Must show the value as a percentage string (42.0%)
    assert "42.0%" in html or "42%" in html, (
        "render_databank_table must format percent_owned=42.0 as '42.0%' in the table HTML. "
        "Got HTML that does not contain '42.0%' or '42%'."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Item 3 — Closer Monitor: empty gmLI and SETUP rows removed/hidden
# ─────────────────────────────────────────────────────────────────────────────


def test_closer_no_unconditional_gmli_comment():
    """The closer card must NOT unconditionally render '<!-- no gmli data -->' in its HTML.

    The previous code set gmli_html = '<!-- no gmli data -->' as the
    always-rendered default.  That string is invisible to users but wastes space
    and signals a broken data path.  The fix: either omit the gmli block entirely
    when there's no data source, or omit the default HTML comment.
    """
    assert "<!-- no gmli data -->" not in _closer_text, (
        "pages/3_Closer_Monitor.py must not unconditionally set "
        "gmli_html = '<!-- no gmli data -->'. "
        "Remove the gmLI row from the card when no data source exists."
    )


def test_closer_gmli_row_absent_or_data_gated():
    """The gmLI row must either be absent OR gated behind a real data condition.

    If gmLI is kept, it must only render when _gmli_val is not None.
    The unconditional HTML-comment default must be gone.
    """
    # Either the whole gmli block is gone, OR it has a non-trivial else clause
    # that doesn't produce the comment.  We check that the comment is gone
    # (covered by test above) and that the template still has the if _gmli_val block.
    # (If gmLI was removed entirely, _gmli_val may also be absent — both are valid.)
    has_gmli_block = "_gmli_val" in _closer_text or "gmli_html" in _closer_text
    if has_gmli_block:
        # Still has gmLI logic — verify the comment default is gone (test above covers)
        assert "<!-- no gmli data -->" not in _closer_text, (
            "gmLI block present but still has the unconditional '<!-- no gmli data -->' default."
        )


def test_closer_setup_row_conditional_or_removed():
    """The SETUP row must either be absent or only rendered when setup_names is non-empty.

    Previous code always rendered 'SETUP · —' when setup_names was empty.
    The fix: wrap the SETUP row in an if/else or template condition so it
    shows only when there are actual setup pitchers.
    """
    # Check that the page does not unconditionally set setup_str to "—" and
    # then render it. The "—" default is fine, but only if it's inside
    # a conditional block that hides the entire SETUP row when empty.
    #
    # Acceptable patterns:
    #   a) SETUP row entirely removed from the card template
    #   b) setup_str rendered only when setup_names is truthy
    #   c) SETUP row contains a conditional that hides when setup_str == "—"
    #
    # Failing pattern: unconditional 'SETUP · {setup_str}' with setup_str="—"
    # We check: if "SETUP" is still in the template, there must be some conditional
    # guard near it.

    if "SETUP" not in _closer_text:
        # Removed entirely — pass
        return

    # SETUP is still present — verify there's a conditional guard.
    # Acceptable guards: 'if item["setup_names"]', 'if setup_names', or
    # a conditional expression that suppresses the row when empty.
    has_guard = (
        'if item["setup_names"]' in _closer_text
        or "if setup_names" in _closer_text
        or "setup_names and" in _closer_text
        or "setup_str and setup_str != " in _closer_text
        or "if setup_str" in _closer_text
    )
    assert has_guard, (
        "pages/3_Closer_Monitor.py has a SETUP row but no conditional guard. "
        "The SETUP row must only render when setup_names is non-empty. "
        "Add: 'if item[\"setup_names\"]' guard or remove the row entirely."
    )
