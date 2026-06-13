"""Trust & comprehension additions — Leaders page (Tasks 3.1, 3.3, 3.5).

Tests are AST/source-text based so they run without Streamlit or a real DB.

Task 3.1  — render_data_freshness_chip is called near the Leaders header.
Task 3.3  — jargon_help is used for "Value", "Sell-High", and "Breakout";
            render_glossary_expander([...]) appears exactly once.
Task 3.5  — Leaderboard entries include an is_mine / roster-membership flag
            that gets highlighted. A My Roster / Free Agents / All radio/
            segmented control appears on the tab to filter leaderboard scope.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "pages" / "17_Leaders.py"


def _page_text() -> str:
    assert PAGE.exists(), f"Page not found: {PAGE}"
    return PAGE.read_text(encoding="utf-8")


# ── Task 3.1: Data freshness chip ────────────────────────────────────────────


def test_leaders_imports_render_data_freshness_chip():
    """render_data_freshness_chip must be imported from src.ui_shared."""
    text = _page_text()
    assert "render_data_freshness_chip" in text, "Task 3.1: render_data_freshness_chip not found in leaders page"


def test_leaders_freshness_chip_called():
    """render_data_freshness_chip must be called at least once on the leaders page."""
    text = _page_text()
    pat = re.compile(r"render_data_freshness_chip\(")
    assert pat.search(text), "Task 3.1: render_data_freshness_chip() not called on leaders page"


# ── Task 3.3: Jargon tooltips & glossary ─────────────────────────────────────


def test_leaders_imports_jargon_help():
    """jargon_help must be imported from src.ui_shared."""
    text = _page_text()
    assert "jargon_help" in text, "Task 3.3: jargon_help not imported on leaders page"


def test_leaders_imports_render_glossary_expander():
    """render_glossary_expander must be imported from src.ui_shared."""
    text = _page_text()
    assert "render_glossary_expander" in text, "Task 3.3: render_glossary_expander not imported on leaders page"


def test_leaders_jargon_help_used_for_sell_high():
    """jargon_help('Sell-High') must be called on the leaders page."""
    text = _page_text()
    pat = re.compile(r"jargon_help\(\s*['\"]Sell-High['\"]")
    assert pat.search(text), "Task 3.3: jargon_help('Sell-High') not found in leaders page"


def test_leaders_glossary_expander_called_once():
    """render_glossary_expander must be called exactly once on the leaders page."""
    text = _page_text()
    count = text.count("render_glossary_expander(")
    assert count == 1, f"Task 3.3: expected render_glossary_expander called once, found {count} times"


def test_leaders_glossary_includes_sell_high():
    """The glossary call must include 'Sell-High' in the terms list."""
    text = _page_text()
    pat = re.compile(r"render_glossary_expander\(([^)]*)\)", re.DOTALL)
    m = pat.search(text)
    assert m, "render_glossary_expander call not found"
    call_args = m.group(1)
    assert "Sell-High" in call_args, "Task 3.3: 'Sell-High' not in leaders glossary term list"


# ── Task 3.5: Highlight viewer's rostered players + lens control ─────────────


def test_leaders_imports_resolve_viewer_team_name():
    """resolve_viewer_team_name must be imported (for roster membership check)."""
    text = _page_text()
    assert "resolve_viewer_team_name" in text, "Task 3.5: resolve_viewer_team_name not imported on leaders page"


def test_leaders_has_roster_lens_control():
    """A My Roster / Free Agents / All filter control must exist on the leaders tab."""
    text = _page_text()
    # Accept radio, selectbox, or segmented control with these option labels
    has_my_roster = "My Roster" in text
    has_free_agents = "Free Agent" in text or "Free Agents" in text
    assert has_my_roster and has_free_agents, (
        "Task 3.5: Leaders page missing 'My Roster' / 'Free Agents' lens control. "
        "Add a radio/selectbox with ['All', 'My Roster', 'Free Agents'] options."
    )


def test_leaders_has_is_mine_highlight_in_leaderboard():
    """The leaderboard HTML builder or table must mark rostered players (is_mine)."""
    text = _page_text()
    assert "is_mine" in text or "rostered_by" in text, (
        "Task 3.5: No is_mine flag or rostered_by column used to highlight viewer's players"
    )


def test_leaders_is_mine_star_or_class_in_leaderboard_html():
    """_build_leaderboard_html must mark the viewer's rostered players visually.

    Accepts either 'is_mine' flag logic or a star/badge marker in the HTML output.
    """
    text = _page_text()
    # The leaderboard HTML function or its caller must check is_mine / viewer_ids
    has_marker = (
        "is_mine" in text or "viewer_ids" in text or "viewer_team" in text or "mine_ids" in text or "my_player" in text
    )
    assert has_marker, (
        "Task 3.5: _build_leaderboard_html must accept/use a set of viewer player IDs "
        "to visually mark the viewer's rostered players."
    )
