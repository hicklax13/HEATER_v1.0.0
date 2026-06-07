"""Guard against THEME-key KeyError crashes on the League Standings page.

Regression test for BR-4 (2026-06-07 audit): the Playoff Odds tab crashed with
`KeyError: 'accent'` because `pages/6_League_Standings.py` used
`T.get("primary", T["accent"])` — and Python evaluates the `.get` default
(`T["accent"]`) eagerly, so a direct subscript on a key absent from THEME raises
every time, regardless of the `.get` key.

This test asserts every direct `T["<key>"]` subscript in the page references a key
that actually exists in the shared THEME, so a missing-key subscript can never ship.
"""

from __future__ import annotations

import re
from pathlib import Path

from src.ui_shared import THEME

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "6_League_Standings.py"


def test_standings_page_theme_subscripts_exist_in_theme():
    src = _PAGE.read_text(encoding="utf-8")
    # Direct subscripts T["key"] are the crash-prone form (T.get(...) is safe).
    subscript_keys = set(re.findall(r'T\["([^"]+)"\]', src))
    missing = sorted(k for k in subscript_keys if k not in THEME)
    assert not missing, (
        f"League Standings page uses T[...] with keys absent from THEME: {missing}. "
        f"Use an existing THEME key (e.g. 'primary') or T.get('key', T['primary'])."
    )
