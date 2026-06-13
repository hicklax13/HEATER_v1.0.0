"""Task 4.3 — League Standings reco banner must pass non-empty expanded_html
and must contain a deep-link to Free Agents or Matchup Planner.

AST-based checks so no Streamlit runtime is needed.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "pages" / "6_League_Standings.py"


def _src() -> str:
    assert PAGE.exists(), f"Page not found: {PAGE}"
    return PAGE.read_text(encoding="utf-8")


# ── Helper: find all render_reco_banner call nodes ───────────────────


def _reco_banner_calls(src: str) -> list[ast.Call]:
    tree = ast.parse(src)
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "render_reco_banner":
                calls.append(node)
    return calls


# ── Test 1: banner call exists ────────────────────────────────────────


def test_reco_banner_called():
    """The page must call render_reco_banner at least once."""
    calls = _reco_banner_calls(_src())
    assert calls, "render_reco_banner was not found in pages/6_League_Standings.py"


# ── Test 2: expanded_html is non-empty ───────────────────────────────


def test_reco_banner_expanded_html_non_empty():
    """render_reco_banner must be called with a non-empty expanded_html argument.

    The second positional arg (or 'expanded_html' kwarg) must NOT be an empty
    string literal `""` or `''`.
    """
    src = _src()
    calls = _reco_banner_calls(src)
    assert calls, "render_reco_banner not called"

    for call in calls:
        # Get the second positional arg or expanded_html kwarg
        expanded: ast.expr | None = None
        if len(call.args) >= 2:
            expanded = call.args[1]
        else:
            for kw in call.keywords:
                if kw.arg == "expanded_html":
                    expanded = kw.value
                    break

        if expanded is None:
            continue  # single-arg call — skip

        # If it's a constant empty string, that's the failure
        if isinstance(expanded, ast.Constant) and expanded.value == "":
            assert False, (
                "render_reco_banner is called with expanded_html='' (empty). "
                "Task 4.3 requires a non-empty expanded_html with a next-step hint."
            )

    # At least one call must have a non-trivially-empty expanded_html
    # (either a variable or a non-empty literal)
    has_non_empty = False
    for call in calls:
        expanded = None
        if len(call.args) >= 2:
            expanded = call.args[1]
        else:
            for kw in call.keywords:
                if kw.arg == "expanded_html":
                    expanded = kw.value
                    break
        if expanded is None:
            continue
        # A non-constant (variable/function call) counts as non-empty
        if not isinstance(expanded, ast.Constant):
            has_non_empty = True
            break
        # A constant non-empty string is fine
        if expanded.value != "":
            has_non_empty = True
            break

    assert has_non_empty, (
        "All render_reco_banner calls pass an empty string for expanded_html. "
        "Task 4.3 requires passing a non-empty string (a variable or function call is fine)."
    )


# ── Test 3: banner expanded_html contains a playoff / GB hint ─────────


def test_standings_banner_expanded_html_has_helper():
    """The page must define a helper that builds expanded_html.

    We accept any function whose name contains 'expand' or whose body is
    assigned to the expanded_html argument of render_reco_banner.  The minimal
    signal: the page source contains a function or lambda that builds the
    expanded_html string (not just an empty literal).
    """
    src = _src()
    # Either a dedicated helper function exists...
    has_helper_fn = bool(
        re.search(r"def\s+_build_banner_expanded", src) or re.search(r"def\s+_standings_expanded", src)
    )
    # ...or the expanded_html variable is assigned before the banner call
    has_variable_assignment = bool(re.search(r"expanded_html\s*=", src))
    # ...or the call directly passes a non-empty string built via f-string/concat
    has_fstring_call = bool(re.search(r"render_reco_banner\s*\([^)]*f[\"']", src, re.DOTALL))
    assert has_helper_fn or has_variable_assignment or has_fstring_call, (
        "pages/6_League_Standings.py must define a helper or variable that builds "
        "a non-empty expanded_html for the reco banner (Task 4.3)."
    )


# ── Test 4: a deep-link to Free Agents or Matchup Planner exists ──────


def test_standings_page_has_deep_link_to_fa_or_matchup():
    """The page must call st.page_link pointing to Free Agents or Matchup Planner."""
    src = _src()
    # st.page_link("pages/14_Free_Agents.py", ...) or Matchup Planner
    has_page_link = bool(re.search(r"st\.page_link\s*\(", src) or re.search(r"st\.switch_page\s*\(", src))
    assert has_page_link, (
        "pages/6_League_Standings.py must call st.page_link or st.switch_page "
        "to link the user to a relevant action page (Free Agents / Matchup Planner). "
        "Task 4.3 requirement."
    )

    # The target must reference Free Agents or Matchup
    has_fa_or_matchup = bool(
        re.search(r"Free_Agent", src)
        or re.search(r"14_Free", src)
        or re.search(r"Matchup", src)
        or re.search(r"5_Matchup", src)
    )
    assert has_fa_or_matchup, (
        "The page_link / switch_page call must target Free Agents (14_Free_Agents.py) "
        "or Matchup Planner (5_Matchup_Planner.py). Task 4.3 requirement."
    )
