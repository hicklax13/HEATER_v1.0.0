"""Task 4.3 (Trade Finder slice) — deep-link to Trade Analyzer.

Guards:
1. The page stashes giving_ids + receiving_ids into session_state under
   the key '_tf_prefill' (a dict with keys 'giving_ids' and 'receiving_ids').
2. A navigational affordance (st.page_link or st.switch_page or a button
   that calls st.switch_page) targeting '11_Trade_Analyzer' is present.
3. The stash key and the link are co-located (within the Trade Recs tab).
"""

from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "pages" / "12_Trade_Finder.py"


def _src() -> str:
    return PAGE.read_text(encoding="utf-8")


# ── 1. Session-state stash ────────────────────────────────────────────────────


def test_tf_prefill_key_written_to_session_state():
    """The page must write giving_ids + receiving_ids into session_state.

    Key name: '_tf_prefill' containing {'giving_ids': [...], 'receiving_ids': [...]}.
    This allows the Trade Analyzer to pre-populate from the stash on next load.
    """
    src = _src()
    assert "_tf_prefill" in src, (
        "pages/12_Trade_Finder.py must write trade player IDs into "
        "st.session_state['_tf_prefill'] so the Trade Analyzer can pre-populate."
    )


def test_tf_prefill_contains_giving_ids():
    """The stash dict must contain 'giving_ids'."""
    src = _src()
    assert "giving_ids" in src, (
        "The '_tf_prefill' session_state stash must include 'giving_ids' so the "
        "Trade Analyzer knows which players the user is giving."
    )


def test_tf_prefill_contains_receiving_ids():
    """The stash dict must contain 'receiving_ids'."""
    src = _src()
    assert "receiving_ids" in src, (
        "The '_tf_prefill' session_state stash must include 'receiving_ids' so the "
        "Trade Analyzer knows which players the user is receiving."
    )


# ── 2. Navigation affordance ──────────────────────────────────────────────────


def test_trade_analyzer_link_present():
    """A link or switch to the Trade Analyzer page must be present.

    Accept: st.switch_page, st.page_link, or a button label referencing
    'Trade Analyzer'.
    """
    src = _src()
    has_link = bool(re.search(r"switch_page", src) or re.search(r"page_link", src) or re.search(r"Trade Analyzer", src))
    assert has_link, (
        "pages/12_Trade_Finder.py must include a link or button to the Trade Analyzer "
        "(e.g. st.page_link or st.switch_page targeting '11_Trade_Analyzer')."
    )


def test_trade_analyzer_page_path_referenced():
    """The navigational affordance must reference the Trade Analyzer page path."""
    src = _src()
    has_path = bool(re.search(r"11_Trade_Analyzer", src) or re.search(r"Trade.{0,10}Analyzer", src))
    assert has_path, (
        "The Trade Analyzer deep-link must reference the page path or label "
        "'11_Trade_Analyzer' / 'Trade Analyzer' so navigation lands on the right page."
    )


# ── 3. Co-location in Recs tab ────────────────────────────────────────────────


def test_prefill_stash_near_analyze_button():
    """The session_state stash and the navigation affordance must be co-located.

    Both '_tf_prefill' and the navigation call (switch_page/page_link) must
    appear after 'tab_recs' is opened in the source, not at some earlier
    unrelated scope.
    """
    src = _src()
    tab_recs_idx = src.find("tab_recs")
    assert tab_recs_idx != -1, "Could not find 'tab_recs' in Trade Finder page."

    post_tab = src[tab_recs_idx:]
    assert "_tf_prefill" in post_tab, (
        "The '_tf_prefill' stash must appear after 'tab_recs' is opened (i.e., inside the Trade Recommendations tab)."
    )
    has_nav = bool(re.search(r"switch_page|page_link|Trade Analyzer", post_tab))
    assert has_nav, (
        "The Trade Analyzer navigation affordance must appear after 'tab_recs' "
        "(i.e., inside the Trade Recommendations tab, near the rows being displayed)."
    )
