"""The Home-page "Connect Yahoo Fantasy" expander is v1-only (2026-06-10).

Under MULTI_USER the server scheduler owns the Yahoo token (admins paste it in
Admin Controls) and member sessions are READ-ONLY. The per-session OAuth
expander in app.py's setup wizard is therefore redundant noise on the hosted
app — and its connect flow calls ``client.sync_to_db()``, a session-side DB
write that would violate the sole-writer invariant. The expander block must be
gated on ``not multi_user_enabled()``.

Owner request 2026-06-10 ("yes hide or remove it").
"""

import re
from pathlib import Path

_APP = (Path(__file__).resolve().parent.parent / "app.py").read_text(encoding="utf-8")


def test_yahoo_expander_gated_on_v1():
    """The gating `if` directly above the expander must check the flag."""
    idx = _APP.find('st.expander("Connect Yahoo Fantasy (optional)"')
    assert idx != -1, "the v1 Yahoo connect expander should still exist for local single-user mode"
    # Inspect the ~600 chars before the expander for its guard condition.
    guard_window = _APP[max(0, idx - 600) : idx]
    assert "not multi_user_enabled()" in guard_window, (
        "the Yahoo connect expander must be gated on `not multi_user_enabled()` — "
        "under MULTI_USER the scheduler owns the token and sessions are read-only"
    )


def test_authorize_link_label_survives_content_link_rule():
    """The 'Authorize with Yahoo' anchor styles its label with inline
    `!important` color + text-decoration, otherwise the global content-link
    rule (orange + underline, also !important) paints orange-on-orange and
    the button renders blank (2026-06-10 prod report)."""
    m = re.search(r"Authorize with Yahoo", _APP)
    assert m, "the Authorize with Yahoo link should exist (v1 path)"
    window = _APP[max(0, m.start() - 700) : m.start()]
    assert re.search(r"color:\{T\['ink'\]\}\s*!important", window), (
        "the anchor label color must carry inline !important to beat the content-link rule"
    )
    assert "text-decoration:none !important" in window, (
        "the anchor must suppress the content-link underline with inline !important"
    )
