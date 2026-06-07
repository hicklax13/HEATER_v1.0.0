"""BR-7: the post-LP IP-budget line must not false-warn on the Today scope.

The "Post-LP Starters X / 54 IP" line in the IP-budget card is compared
against the WEEKLY target. The post-LP IP recompute only runs under the
Today (daily) scope and sums only TODAY's LP-selected pitchers, which
understates weekly IP and produces a false low-IP / forfeit-risk warning.

These structural guards lock the fix: the write site records the optimize
scope alongside _post_lp_ip, and the render gates the post-LP line so it
only shows for weekly / rest-of-season scopes (suppressed on Today).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "pages" / "2_Line-up_Optimizer.py"


def _page_text() -> str:
    assert PAGE.exists()
    return PAGE.read_text(encoding="utf-8")


def test_post_lp_write_site_records_scope():
    """When the optimizer stores _post_lp_ip, it must also store the scope it
    was computed under (_post_lp_scope) so the renderer can gate the display."""
    text = _page_text()
    assert 'st.session_state["_post_lp_ip"]' in text or "_post_lp_ip'" in text
    assert "_post_lp_scope" in text, (
        "BR-7: the post-LP IP write site must record _post_lp_scope so the "
        "IP-budget card can suppress the weekly comparison on the Today scope."
    )


def test_post_lp_line_scope_gated_to_weekly():
    """The post-LP IP line (_post_lp_line) must be gated by a scope check that
    restricts it to weekly / rest-of-season scopes — never the Today daily view."""
    text = _page_text()
    lines = text.splitlines()

    # Find where the post-LP line is constructed.
    constr_idx = next(
        (i for i, ln in enumerate(lines) if "_post_lp_line = (" in ln),
        None,
    )
    assert constr_idx is not None, "Could not locate _post_lp_line construction."

    # Look back a small window for the gating condition.
    window = "\n".join(lines[max(0, constr_idx - 8) : constr_idx + 1])
    assert "_post_lp_scope" in window, (
        "BR-7: the post-LP IP line must be gated on _post_lp_scope so it is "
        "suppressed on the Today daily view (false weekly-forfeit warning)."
    )
    assert re.search(r"rest_of_week|rest_of_season", window), (
        "BR-7: the post-LP IP line gate must restrict to weekly / "
        "rest-of-season scopes (the weekly IP target only applies there)."
    )
