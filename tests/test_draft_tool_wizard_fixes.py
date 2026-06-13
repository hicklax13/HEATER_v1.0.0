"""Tests for draft tool wizard fixes — Tasks 1.1, 1.3, 1.4.

Task 1.1 — Verify "Next →" button navigates from Step 1 to Step 2.
Task 1.3 — Manual SGP inputs cover ALL 12 categories (OBP + L were missing).
Task 1.4 — SGP spinbuttons have explicit format= arg; Category Balance WHIP
            uses .2f (not .3f).
"""

import ast
import re
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app.py"


def _app_src() -> str:
    return APP.read_text(encoding="utf-8")


def _func_src(name: str) -> str:
    """Return the source text of a top-level function in app.py."""
    src = _app_src()
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, node)


# ── Task 1.1: Next → button ─────────────────────────────────────────


def test_next_button_sets_setup_step_2():
    """'Next →' click handler must set setup_step = 2 and call st.rerun()."""
    src = _func_src("render_step_settings")
    # The button label
    assert "Next →" in src, "render_step_settings() must contain a 'Next →' button"
    # Sets step to 2
    assert "setup_step" in src and "2" in src, (
        "render_step_settings() must set st.session_state.setup_step = 2 on button click"
    )
    # Calls rerun so the page transitions
    assert "rerun" in src, "render_step_settings() must call st.rerun() after setting setup_step"


def test_render_setup_page_routes_step2_to_launch():
    """render_setup_page() must call render_step_launch() when setup_step == 2."""
    src = _func_src("render_setup_page")
    assert "render_step_launch" in src, "render_setup_page() must call render_step_launch() for step == 2"


# ── Task 1.3: All 12 categories in manual SGP block ─────────────────


def test_manual_sgp_block_includes_obp():
    """OBP (On-Base Percentage) number_input must be present in the manual SGP block."""
    src = _func_src("render_step_settings")
    # Key used for OBP
    assert "sgp_obp" in src, "render_step_settings() must include an st.number_input with key='sgp_obp' for OBP"


def test_manual_sgp_block_includes_l():
    """L (Losses) number_input must be present in the manual SGP block."""
    src = _func_src("render_step_settings")
    assert "sgp_l" in src, "render_step_settings() must include an st.number_input with key='sgp_l' for Losses"


def test_manual_sgp_all_12_categories_have_session_key():
    """_build_player_pool (or the manual-SGP block) must read all 12 category keys
    from session_state when auto_sgp is False.  OBP and L are the historically
    missing ones."""
    from src.valuation import LeagueConfig

    lc = LeagueConfig()
    all_cats = lc.all_categories  # ['R','HR','RBI','SB','AVG','OBP','W','L','SV','K','ERA','WHIP']
    src = _app_src()

    # For each category, either a key like sgp_r / sgp_hr / sgp_obp must appear
    # inside the manual-SGP block / build section of app.py.
    cat_to_key = {
        "R": "sgp_r",
        "HR": "sgp_hr",
        "RBI": "sgp_rbi",
        "SB": "sgp_sb",
        "AVG": "sgp_avg",
        "OBP": "sgp_obp",
        "W": "sgp_w",
        "L": "sgp_l",
        "SV": "sgp_sv",
        "K": "sgp_k",
        "ERA": "sgp_era",
        "WHIP": "sgp_whip",
    }
    missing = [cat for cat, key in cat_to_key.items() if key not in src]
    assert not missing, f"app.py is missing manual-SGP session keys for: {missing}"


def test_build_player_pool_reads_obp_and_l_from_session():
    """_build_player_pool must apply sgp_obp and sgp_l from session_state
    when auto_sgp is False (same as it does for the other 10 categories)."""
    src = _func_src("_build_player_pool")
    assert "sgp_obp" in src, "_build_player_pool() must read 'sgp_obp' from session_state when auto_sgp is off"
    assert "sgp_l" in src, "_build_player_pool() must read 'sgp_l' from session_state when auto_sgp is off"


# ── Task 1.4: float formatting in SGP spinbuttons + WHIP in cat balance ─


def test_sgp_counting_inputs_have_format_arg():
    """Counting-stat SGP number_inputs (R, HR, RBI, SB, W, L, SV, K) must
    not be raw floats — each st.number_input must specify format= so the
    display doesn't show floating-point noise like 32.000000000001."""
    src = _func_src("render_step_settings")
    # Counting stats — no decimal format needed, but the widget needs format
    # so it doesn't display "32.0" or worse "32.000000037".
    # We check that every number_input call in the manual-SGP block has format=
    # by counting number_input calls and format= occurrences in the block.
    # Locate the `if not auto_sgp:` block
    block_start = src.find("if not auto_sgp:")
    assert block_start != -1, "Could not find 'if not auto_sgp:' block in render_step_settings()"
    # Count number_input calls and format= args in the block
    block = src[block_start:]
    n_inputs = block.count("st.number_input(")
    n_format = block.count("format=")
    # Every number_input in the SGP block should have a format= argument
    assert n_format >= n_inputs, (
        f"Not all SGP st.number_input() calls have a format= argument: "
        f"{n_inputs} inputs vs {n_format} format= args in the 'if not auto_sgp:' block"
    )


def test_category_balance_whip_uses_2f():
    """Category Balance renders WHIP — it must use .2f (or format_stat), not .3f."""
    src = _func_src("render_category_balance")
    # The cats tuple for WHIP should use ".2f", not ".3f"
    # Find the tuple entry for WHIP
    whip_idx = src.find('"Walks + Hits per Inning Pitched"')
    assert whip_idx != -1, "WHIP entry not found in render_category_balance cats list"
    # Extract the 60 chars after the WHIP label to find its format string
    snippet = src[whip_idx : whip_idx + 80]
    assert '".3f"' not in snippet and "'.3f'" not in snippet, (
        f"WHIP in Category Balance uses '.3f' — must use '.2f' (format_stat('WHIP') → 2dp). Snippet: {snippet!r}"
    )
    # Optionally confirm .2f is present (covers both inline and format_stat paths)
    assert '".2f"' in snippet or "format_stat" in src, (
        "WHIP in Category Balance should use '.2f' or route through format_stat"
    )
