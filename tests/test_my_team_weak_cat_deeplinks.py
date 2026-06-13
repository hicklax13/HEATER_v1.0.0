"""Task 4.3 — My Team page must render deep-links (st.page_link) for
weak-category / flippable-category suggestions from the War Room.

AST-based checks — no Streamlit runtime required.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "pages" / "1_My_Team.py"


def _src() -> str:
    assert PAGE.exists(), f"Page not found: {PAGE}"
    return PAGE.read_text(encoding="utf-8")


# ── Test 1: page_link / switch_page call exists ───────────────────────


def test_my_team_has_page_link():
    """My Team must call st.page_link or st.switch_page at least once."""
    src = _src()
    has_link = bool(re.search(r"st\.page_link\s*\(", src) or re.search(r"st\.switch_page\s*\(", src))
    assert has_link, (
        "pages/1_My_Team.py must call st.page_link or st.switch_page to deep-link "
        "weak-category suggestions to Free Agents or Pitcher Streaming. Task 4.3."
    )


# ── Test 2: deep-link targets Free Agents ────────────────────────────


def test_my_team_deep_link_to_free_agents():
    """My Team must have a deep-link to the Free Agents page."""
    src = _src()
    has_fa = bool(re.search(r"Free_Agent", src) or re.search(r"14_Free", src))
    assert has_fa, (
        "pages/1_My_Team.py must link to Free Agents (14_Free_Agents.py) "
        "for hitter weak-category suggestions (SB/HR/RBI/R). Task 4.3."
    )


# ── Test 3: deep-link targets Pitcher Streaming ───────────────────────


def test_my_team_deep_link_to_pitcher_streaming():
    """My Team must have a deep-link to Pitcher Streaming for K/W/ERA/WHIP."""
    src = _src()
    has_stream = bool(re.search(r"Pitcher_Stream", src) or re.search(r"4_Pitcher", src))
    assert has_stream, (
        "pages/1_My_Team.py must link to Pitcher Streaming (4_Pitcher_Streaming.py) "
        "for pitching weak-category suggestions (K/W/ERA/WHIP). Task 4.3."
    )


# ── Test 4: links are near the flippable categories section ──────────


def test_my_team_deeplinks_near_flippable_cats():
    """The deep-links must appear after/near the Flippable Categories section.

    Heuristic: the source text that contains the page_link call must come
    AFTER the line that references ``get_flippable_categories`` or
    ``fl["suggestion"]`` so the links are contextually placed.
    """
    src = _src()
    lines = src.splitlines()

    # Find last line index of flippable-cat rendering
    flip_line = max(
        (i for i, ln in enumerate(lines) if "flippables" in ln or 'fl["suggestion"]' in ln),
        default=-1,
    )
    assert flip_line >= 0, (
        "Could not locate Flippable Categories section in pages/1_My_Team.py. "
        "The deep-links test relies on flippables rendering being present."
    )

    # Find first page_link call line
    link_line = next(
        (i for i, ln in enumerate(lines) if "page_link" in ln or "switch_page" in ln),
        -1,
    )
    assert link_line >= 0, "No page_link/switch_page found in pages/1_My_Team.py"

    assert link_line >= flip_line, (
        f"st.page_link (line {link_line + 1}) appears BEFORE the Flippable Categories "
        f"section (line {flip_line + 1}). The deep-link should be placed after/near the "
        "suggestion text so it is contextually relevant. Task 4.3."
    )


# ── Test 5: render_reco_banner on My Team passes something for expanded_html ─


def test_my_team_reco_banner_has_expanded_html():
    """The render_reco_banner call on My Team must pass a non-empty expanded_html.

    Task 4.3 asks the banner's 'Details' expander to carry a next-step link.
    This test checks the call signature — not the runtime value.
    """
    src = _src()

    tree = ast.parse(src)
    banner_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "render_reco_banner":
                banner_calls.append(node)

    assert banner_calls, "render_reco_banner not called on My Team page"

    # At least one call must have a non-empty expanded_html
    has_non_empty = False
    for call in banner_calls:
        expanded: ast.expr | None = None
        if len(call.args) >= 2:
            expanded = call.args[1]
        else:
            for kw in call.keywords:
                if kw.arg == "expanded_html":
                    expanded = kw.value
                    break
        if expanded is None:
            continue
        if isinstance(expanded, ast.Constant) and expanded.value == "":
            continue  # explicitly empty — doesn't count
        has_non_empty = True
        break

    assert has_non_empty, (
        "render_reco_banner on My Team is called with expanded_html='' (empty). "
        "Task 4.3 requires a non-empty expanded_html so the Details expander shows "
        "a next-step link."
    )
