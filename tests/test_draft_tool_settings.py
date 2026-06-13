"""Regression + correctness tests for the Draft Tool settings wizard.

Task 1.1 — Next button navigation (verify, not fix)
Task 1.3 — Manual SGP inputs cover all 12 LeagueConfig categories (OBP + L missing)
Task 1.4 — SGP spinbuttons carry an explicit format= arg; WHIP display uses .2f
"""

import ast
import re
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parent.parent / "app.py"
_SRC = APP.read_text(encoding="utf-8")


# ── Helpers ────────────────────────────────────────────────────────────


def _render_step_settings_src() -> str:
    """Return the source text of render_step_settings() only."""
    tree = ast.parse(_SRC)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "render_step_settings":
            seg = ast.get_source_segment(_SRC, node)
            assert seg, "render_step_settings source extraction failed"
            return seg
    raise AssertionError("render_step_settings not found in app.py")


def _render_category_balance_src() -> str:
    """Return the source text of render_category_balance() only."""
    tree = ast.parse(_SRC)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "render_category_balance":
            seg = ast.get_source_segment(_SRC, node)
            assert seg, "render_category_balance source extraction failed"
            return seg
    raise AssertionError("render_category_balance not found in app.py")


def _build_player_pool_src() -> str:
    """Return the source text of _build_player_pool() only."""
    tree = ast.parse(_SRC)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_build_player_pool":
            seg = ast.get_source_segment(_SRC, node)
            assert seg, "_build_player_pool source extraction failed"
            return seg
    raise AssertionError("_build_player_pool not found in app.py")


# ── Task 1.1 — "Next →" button navigates to Step 2 ──────────────────


def test_next_button_sets_setup_step_to_2():
    """The Next → callback must set setup_step = 2 then rerun.

    AST-check: find the "Next →" button in render_step_settings and confirm
    its handler sets setup_step = 2 and calls st.rerun() immediately after.
    Uses the position of the LAST st.button("Next") to avoid false-positives
    from earlier st.rerun() calls in the Yahoo OAuth block.
    """
    src = _render_step_settings_src()
    # The button handler should assign 2 to setup_step
    assert "setup_step" in src, "setup_step never referenced in render_step_settings"
    assert "= 2" in src, "setup_step is never set to 2 inside render_step_settings"

    # Find the "Next →" button — search from the end (it's the last button)
    next_pos = src.rfind("Next")
    assert next_pos != -1, '"Next" button label not found'

    # From the Next button onwards, both the assignment and rerun must appear
    tail = src[next_pos:]
    assign_pos = tail.find("setup_step = 2")
    rerun_pos = tail.find("st.rerun()")
    assert assign_pos != -1, "setup_step = 2 not found after the Next → button"
    assert rerun_pos != -1, "st.rerun() not found after the Next → button"
    assert rerun_pos > assign_pos, "st.rerun() must appear after setup_step = 2"


# ── Task 1.3 — All 12 categories have manual SGP inputs ──────────────


def test_manual_sgp_inputs_include_obp():
    """render_step_settings must render an OBP number_input when auto_sgp is off."""
    src = _render_step_settings_src()
    # Look for a number_input that references OBP (key or label)
    assert "sgp_obp" in src or "On-Base" in src, (
        "No OBP SGP denominator number_input found in render_step_settings. "
        "OBP must be in the manual input block (was missing before this fix)."
    )


def test_manual_sgp_inputs_include_losses():
    """render_step_settings must render an L (Losses) number_input when auto_sgp is off."""
    src = _render_step_settings_src()
    # Look for a number_input that references Losses / sgp_l
    assert "sgp_l" in src or "Losses" in src, (
        "No Losses (L) SGP denominator number_input found in render_step_settings. "
        "L must be in the manual input block (was missing before this fix)."
    )


def test_manual_sgp_inputs_count_covers_all_12_categories():
    """The manual SGP block must have number_inputs for all 12 categories.

    We count st.number_input calls inside the `if not auto_sgp:` branch.
    The minimum expected count is 12 (one per LeagueConfig category).
    """
    from src.valuation import LeagueConfig

    lc = LeagueConfig()
    expected = set(lc.all_categories)  # 12 categories

    src = _render_step_settings_src()
    # Extract SGP key names used in number_input calls, e.g. key="sgp_r"
    keys_found = set(re.findall(r'key=["\']sgp_(\w+)["\']', src))
    # Map key suffixes back to uppercase category codes
    key_to_cat = {k.upper(): k for k in keys_found}
    cats_found = set(key_to_cat.keys())
    missing = expected - cats_found
    assert not missing, (
        f"Manual SGP inputs are missing categories: {sorted(missing)}. "
        f"Keys found: {sorted(cats_found)}. Expected: {sorted(expected)}."
    )


def test_build_player_pool_reads_obp_from_session_state():
    """_build_player_pool must read sgp_obp from session_state when auto_sgp is off."""
    src = _build_player_pool_src()
    assert "sgp_obp" in src, (
        "_build_player_pool does not read sgp_obp from session_state. "
        "OBP denominator would be silently ignored in manual mode."
    )


def test_build_player_pool_reads_l_from_session_state():
    """_build_player_pool must read sgp_l from session_state when auto_sgp is off."""
    src = _build_player_pool_src()
    assert "sgp_l" in src, (
        "_build_player_pool does not read sgp_l from session_state. "
        "L (Losses) denominator would be silently ignored in manual mode."
    )


# ── Task 1.4 — Float formatting on SGP inputs + WHIP display ─────────


def test_sgp_avg_has_format_arg():
    """AVG number_input must pass an explicit format= so no raw float tail renders."""
    src = _render_step_settings_src()
    # Find the AVG input block
    assert 'key="sgp_avg"' in src or "key='sgp_avg'" in src, "sgp_avg key not found"
    # The format= kwarg must be present near the sgp_avg key
    # Extract a generous window around sgp_avg
    idx = src.find("sgp_avg")
    window = src[max(0, idx - 300) : idx + 50]
    assert "format=" in window, (
        "sgp_avg number_input is missing a format= arg — raw float tails will render "
        "(e.g. 0.00800000037997961 instead of 0.008)."
    )


def test_sgp_obp_has_format_arg():
    """OBP number_input must pass an explicit format= so no raw float tail renders."""
    src = _render_step_settings_src()
    assert "sgp_obp" in src, "sgp_obp key not found (see test_manual_sgp_inputs_include_obp)"
    idx = src.find("sgp_obp")
    window = src[max(0, idx - 300) : idx + 50]
    assert "format=" in window, "sgp_obp number_input is missing a format= arg — raw float tails will render."


def test_sgp_era_has_format_arg():
    """ERA number_input must pass an explicit format=."""
    src = _render_step_settings_src()
    assert "sgp_era" in src, "sgp_era key not found"
    idx = src.find("sgp_era")
    window = src[max(0, idx - 300) : idx + 50]
    assert "format=" in window, "sgp_era number_input is missing a format= arg"


def test_sgp_whip_has_format_arg():
    """WHIP number_input must pass an explicit format=."""
    src = _render_step_settings_src()
    assert "sgp_whip" in src, "sgp_whip key not found"
    idx = src.find("sgp_whip")
    window = src[max(0, idx - 300) : idx + 50]
    assert "format=" in window, "sgp_whip number_input is missing a format= arg"


def test_category_balance_whip_uses_2f_not_3f():
    """render_category_balance must display WHIP with .2f precision, not .3f.

    Both audits flagged WHIP showing 3 decimal places (e.g. 1.234) — the
    correct display is 2 decimal places (e.g. 1.23) matching format_stat convention.
    """
    src = _render_category_balance_src()
    # Find the WHIP tuple in the cats list
    # Pattern: something like ("...WHIP...", ..., ".3f")
    # We want to assert ".3f" is NOT used with WHIP
    whip_line_match = re.search(r"WHIP[^\n]*\n?[^\n]*", src)
    assert whip_line_match, "WHIP not found in render_category_balance cats list"
    # The fragment around WHIP must not contain ".3f"
    whip_ctx = src[max(0, whip_line_match.start() - 20) : whip_line_match.end() + 60]
    assert '".3f"' not in whip_ctx and "'.3f'" not in whip_ctx, (
        f"WHIP in render_category_balance is formatted with .3f — must be .2f.\nContext: {whip_ctx!r}"
    )
    # And .2f must appear somewhere in the category balance function
    assert ".2f" in src, "render_category_balance has no .2f format string — ERA and WHIP both need 2-decimal display."
