"""Task 4.3 (Punt Analyzer slice) — deep-link to Free Agents.

Guards:
1. The page exposes a navigational affordance (st.page_link or st.switch_page
   or a button/link) pointing to '14_Free_Agents' after the punt analysis.
2. The active (non-punted) categories are stashed into session_state so the
   Free Agents page can pre-filter to those categories.
3. The link appears after the punt category selection is complete (i.e., after
   `punt_cats` is defined), not in the cold state where no categories are selected.
"""

from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "pages" / "10_Punt_Analyzer.py"


def _src() -> str:
    return PAGE.read_text(encoding="utf-8")


# ── 1. Free Agents navigational affordance ───────────────────────────────────


def test_free_agents_link_present():
    """A link or button targeting Free Agents must be present in Punt Analyzer."""
    src = _src()
    has_link = bool(
        re.search(r"14_Free_Agents", src)
        or re.search(r"Free.{0,5}Agent", src, re.IGNORECASE)
        and re.search(r"switch_page|page_link|st\.link_button|href.*free|Free.*href", src, re.IGNORECASE)
    )
    # More permissive: just check for "Free Agent" + some navigation concept
    has_fa_mention = bool(re.search(r"Free.{0,10}Agent", src, re.IGNORECASE))
    has_nav = bool(re.search(r"switch_page|page_link|st\.link_button", src) or re.search(r"14_Free_Agents", src))
    assert has_fa_mention and has_nav, (
        "pages/10_Punt_Analyzer.py must include a navigational affordance to the "
        "Free Agents page (e.g. st.page_link('pages/14_Free_Agents.py', ...) or "
        "st.switch_page('pages/14_Free_Agents.py')) after the punt analysis so "
        "users can go browse FAs for their active (non-punted) categories."
    )


def test_free_agents_page_path_referenced():
    """The Free Agents page path or label must be explicitly referenced."""
    src = _src()
    has_path = bool(re.search(r"14_Free_Agents", src) or re.search(r"[Ff]ree.{0,5}[Aa]gents", src))
    assert has_path, (
        "The Free Agents deep-link in Punt Analyzer must reference '14_Free_Agents' "
        "or 'Free Agents' so navigation targets the correct page."
    )


# ── 2. Active categories stashed for FA pre-filter ───────────────────────────


def test_active_cats_stashed_for_free_agents():
    """The active (non-punted) categories must be stashed in session_state.

    Key: '_punt_active_cats' (or similar) so the Free Agents page can read it
    and pre-filter recommendations to the active categories.
    """
    src = _src()
    has_stash = bool(
        re.search(r"_punt_active_cats|punt_active_cats", src)
        or re.search(r"session_state.*active_cats|active_cats.*session_state", src, re.IGNORECASE)
    )
    assert has_stash, (
        "Punt Analyzer must write the active (non-punted) categories to "
        "st.session_state['_punt_active_cats'] so the Free Agents page can "
        "pre-filter recommendations to only those categories."
    )


# ── 3. Link appears after punt_cats is defined ───────────────────────────────


def test_link_appears_after_punt_cat_selection():
    """The Free Agents link must appear after punt_cats is defined (not in cold state)."""
    src = _src()
    # punt_cats is assigned via st.multiselect
    punct_cats_idx = src.find("punt_cats = st.multiselect")
    if punct_cats_idx == -1:
        punct_cats_idx = src.find("punt_cats")
    assert punct_cats_idx != -1, "Could not find punt_cats definition."

    post_selection = src[punct_cats_idx:]
    has_link_after = bool(
        re.search(r"14_Free_Agents", post_selection)
        or (
            re.search(r"Free.{0,10}Agent", post_selection, re.IGNORECASE)
            and re.search(r"switch_page|page_link|st\.link_button", post_selection)
        )
    )
    assert has_link_after, (
        "The Free Agents deep-link must appear AFTER punt_cats is defined, "
        "so it is only shown once categories have been selected."
    )
