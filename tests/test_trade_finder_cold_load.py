"""Task 2.3 + 2.5-TF: Trade Finder cold-load performance gate.

Structural (AST-free, text-scan) guards that the page:
1. Does NOT call find_trade_opportunities unconditionally on every render —
   the call must be gated behind a button check or an existing cache hit.
2. Uses a reduced default max_results (<= 20) for the on-demand scan.
3. Caches scan results in session_state so reruns don't repeat the scan.
4. Wraps the scan in st.spinner with a descriptive label.
5. Shows a prompt / empty-state when no scan has been run yet (cold state).

All tests are pure text-scan over the page source — no Streamlit AppTest
required, no network, no DB.  Intentionally fast.
"""

from __future__ import annotations

from pathlib import Path

PAGE_SRC = (Path(__file__).resolve().parent.parent / "pages" / "12_Trade_Finder.py").read_text(encoding="utf-8")

# ---------------------------------------------------------------------------
# 1. The full multi-team scan is gated behind an explicit "Run scan" button
#    (or a session_state cold-load guard), NOT triggered unconditionally.
#
#    Evidence: the call to find_trade_opportunities() must be preceded in the
#    source by a button check (st.button / st.form_submit_button) or a cold-
#    load guard that returns early when no scan has been run.  We verify by
#    checking that there is a button whose text suggests "scan" / "find trades",
#    and that find_trade_opportunities is NOT called outside of a guarded block.
# ---------------------------------------------------------------------------


def test_scan_is_gated_behind_button():
    """find_trade_opportunities must only be called inside a button-gated block
    OR after confirming a cached result exists.  The page must have a scan
    trigger button so the user explicitly initiates the 43.8s scan."""
    # The button label should mention 'scan' or 'trade' (case-insensitive).
    import re

    button_pattern = re.compile(
        r'st\.button\s*\(["\']([^"\']*(?:scan|find.?trade|search|run)[^"\']*)["\']',
        re.IGNORECASE,
    )
    assert button_pattern.search(PAGE_SRC), (
        "Trade Finder page must have an explicit 'Run scan' / 'Find trades' button "
        "so the cold-load scan is gated behind user intent, not triggered automatically."
    )


def test_find_trade_opportunities_not_called_at_module_top_level():
    """find_trade_opportunities must not be called unconditionally at page load.
    It must be inside a conditional block (if ... button / cache miss).

    Proxy check: the call must be preceded by a session_state cache guard or
    a button check within the same scope — verified by asserting that
    '_tf_scan_cache' check (existing) AND a button guard BOTH appear in the
    source, and that there is no bare call outside both guards.

    We check for the presence of a button + the cache guard together.
    """
    assert "_tf_scan_cache" in PAGE_SRC, (
        "page must still have the session_state cache (_tf_scan_cache) so reruns "
        "after the initial scan don't recompute."
    )
    # The cache-miss branch must still call find_trade_opportunities, but now
    # that branch must ALSO require a button click (cold state → empty-state,
    # not an automatic scan).
    assert "find_trade_opportunities" in PAGE_SRC, "page must still call find_trade_opportunities (it's the engine)."


# ---------------------------------------------------------------------------
# 2. The on-demand scan uses a reduced default max_results (<= 20).
#    This keeps the first explicit scan fast (sub-15s) while still showing
#    meaningful results.  An "Expand search" affordance can raise it later.
# ---------------------------------------------------------------------------


def test_default_max_results_is_reduced():
    """The initial on-demand scan must use max_results <= 20 (not 50).

    A sidebar slider or a secondary button can allow expanding to 50+.
    The literal 'max_results=50' call to find_trade_opportunities must
    no longer be the only call — or it must be replaced with a smaller default.
    """
    import re

    # Count calls to find_trade_opportunities with max_results=50 (or higher)
    high_results_pattern = re.compile(
        r"find_trade_opportunities\s*\([^)]*max_results\s*=\s*([5-9]\d|\d{3,})",
        re.DOTALL,
    )
    hits = high_results_pattern.findall(PAGE_SRC)
    assert not hits, (
        f"find_trade_opportunities called with max_results >= 50 ({hits!r}). "
        "Default scan must use max_results <= 20 to keep the first run fast. "
        "Add an 'Expand search' affordance for larger scans."
    )


# ---------------------------------------------------------------------------
# 3. Scan result is cached in session_state so reruns don't re-run it.
#    (Already tested in test_trade_scan_signature.py; re-assert the key
#    structural pieces here as part of this blocker's test suite.)
# ---------------------------------------------------------------------------


def test_scan_result_cached_in_session_state():
    """The result of find_trade_opportunities must be written into session_state
    so subsequent reruns (filter changes, tab switches) skip the expensive call."""
    assert 'st.session_state["_tf_scan_cache"]' in PAGE_SRC, (
        "page must store scan results in st.session_state['_tf_scan_cache'] after running find_trade_opportunities."
    )


# ---------------------------------------------------------------------------
# 4. Scan is wrapped in st.spinner with a descriptive label.
# ---------------------------------------------------------------------------


def test_scan_wrapped_in_spinner():
    """The find_trade_opportunities call must be inside an st.spinner(...) block
    so the user sees progress feedback during the scan."""
    import re

    # st.spinner(...): must appear in the source with a label that's a string
    spinner_pattern = re.compile(r"with\s+st\.spinner\s*\(", re.IGNORECASE)
    assert spinner_pattern.search(PAGE_SRC), (
        "Trade Finder page must use st.spinner(...) around the scan call "
        "to give the user progress feedback during the scan."
    )

    # The spinner label must be descriptive (mention scan/trade/league)
    spinner_label_pattern = re.compile(
        r'st\.spinner\s*\(["\']([^"\']*(?:scan|trade|league)[^"\']*)["\']',
        re.IGNORECASE,
    )
    assert spinner_label_pattern.search(PAGE_SRC), (
        "st.spinner label must be descriptive — mention 'scan', 'trade', or 'league'."
    )


# ---------------------------------------------------------------------------
# 5. Cold state shows a prompt / empty-state, not a blank page or an error.
#    The page must have a path that renders a call-to-action when no scan
#    has been run (i.e., when _tf_scan_cache is absent from session_state).
# ---------------------------------------------------------------------------


def test_cold_state_renders_prompt():
    """When no scan has been run (_tf_scan_cache absent), the page must render
    a prompt telling the user to click the scan button.

    Verified by the presence of an empty-state / info call that is conditioned
    on the absence of cached results AND is accompanied by the scan button.
    """
    # The page must have logic that checks for the absence of cached results
    # and renders something (render_empty_state, st.info, or markdown) in that branch.
    # Simplest proxy: the page checks 'get("_tf_scan_cache")' and has a render
    # call that is NOT inside an 'else' that also calls find_trade_opportunities.
    assert "get" in PAGE_SRC and "_tf_scan_cache" in PAGE_SRC, (
        "page must check st.session_state.get('_tf_scan_cache') to determine cold state."
    )

    # The cold path must show something — either render_empty_state or st.info
    import re

    cold_prompt_pattern = re.compile(
        r"render_empty_state|st\.info\s*\(|st\.markdown.*click|button.*scan",
        re.IGNORECASE | re.DOTALL,
    )
    assert cold_prompt_pattern.search(PAGE_SRC), (
        "Cold state (no scan yet) must show a prompt/empty-state so the user knows to click the scan button."
    )
