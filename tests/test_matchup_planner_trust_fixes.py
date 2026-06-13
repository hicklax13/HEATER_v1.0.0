"""Tests for trust/comprehension fixes on pages/5_Matchup_Planner.py.

Tasks covered:
  3.6 [BLOCKER] — schedule_warning must be rendered (st.warning) when built.
  3.1 — render_data_freshness_chip imported and called near header.
  3.3 — jargon_help tooltips on coined labels; render_glossary_expander called.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "pages" / "5_Matchup_Planner.py"


def _source() -> str:
    return PAGE.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Task 3.6 [BLOCKER] — schedule_warning must be rendered
# ---------------------------------------------------------------------------


def test_schedule_warning_is_rendered():
    """When schedule_warning is set, it must be rendered via st.warning().

    The bug: schedule_warning is built but the st.warning(schedule_warning)
    call is missing, so users see 27 'Avoid' rows with no explanation.
    """
    src = _source()
    lines = src.splitlines()

    # Find where schedule_warning is assigned
    assign_line = None
    for i, line in enumerate(lines):
        if "schedule_warning" in line and "=" in line and "st.warning" not in line:
            assign_line = i
            break

    assert assign_line is not None, "Could not find schedule_warning assignment in page"

    # Within the next 10 lines after assignment there must be an
    # st.warning(schedule_warning) or equivalent render call
    window = lines[assign_line : assign_line + 10]
    rendered = any(
        "st.warning(schedule_warning)" in l or ("schedule_warning" in l and "st.warning" in l) for l in window
    )
    assert rendered, (
        f"schedule_warning is assigned at line {assign_line + 1} but never "
        "rendered.  Add 'if schedule_warning: st.warning(schedule_warning)' "
        "immediately after the assignment so users see the explanation for "
        "why they're seeing 27 'Avoid' rows."
    )


def test_schedule_warning_rendered_conditionally():
    """The render must be guarded: only shown when schedule_warning is set
    (i.e. the block reads 'if schedule_warning: st.warning(...)').
    """
    src = _source()
    assert "if schedule_warning" in src or "schedule_warning and st.warning" in src, (
        "schedule_warning render must be conditional so it doesn't show on happy-path (when a real schedule loaded)."
    )


# ---------------------------------------------------------------------------
# Task 3.1 — render_data_freshness_chip imported and called
# ---------------------------------------------------------------------------


def test_data_freshness_chip_imported_in_matchup_planner():
    """pages/5_Matchup_Planner.py must import render_data_freshness_chip."""
    src = _source()
    assert "render_data_freshness_chip" in src, (
        "render_data_freshness_chip must be imported from src.ui_shared and called on the Matchup Planner page"
    )


def test_data_freshness_chip_called_in_matchup_planner():
    """render_data_freshness_chip must be called (not just imported)."""
    src = _source()
    call_sites = [
        line
        for line in src.splitlines()
        if "render_data_freshness_chip(" in line
        and not line.strip().startswith("from ")
        and not line.strip().startswith("#")
    ]
    assert call_sites, "render_data_freshness_chip(...) must be called at least once in the Matchup Planner page"


# ---------------------------------------------------------------------------
# Task 3.3 — jargon_help tooltips + render_glossary_expander
# ---------------------------------------------------------------------------


def test_jargon_help_imported_in_matchup_planner():
    """jargon_help must be imported from src.ui_shared in the Matchup Planner."""
    src = _source()
    assert "jargon_help" in src, (
        "jargon_help must be imported from src.ui_shared to provide help= tooltips "
        "on coined labels (Smash, Win Prob, etc.)"
    )


def test_glossary_expander_called_in_matchup_planner():
    """render_glossary_expander must be called once on the Matchup Planner page
    so users can look up coined terms like 'Smash', 'Rating', etc.
    """
    src = _source()
    assert "render_glossary_expander" in src, (
        "render_glossary_expander(...) must be called on the Matchup Planner page "
        "to expose a 'What do these numbers mean?' expander for first-time users."
    )


def test_smash_tier_has_help_tooltip():
    """The 'Smash' tier label (or the Days-to-look-ahead selectbox or at least
    one coined widget) must have a help= kwarg using jargon_help().
    """
    src = _source()
    # Check that jargon_help is actually used in a help= context
    has_help_call = "help=jargon_help(" in src
    assert has_help_call, (
        "At least one widget must use help=jargon_help(...) so coined labels like 'Smash' have discoverable tooltips."
    )
