"""BR-2: the My Team category-gap callout must not be mislabeled.

The red callout renders only the top-2 losing categories (a focused "priority
targets" subset, `_losing_sorted[:2]`), but it used to be headed "Losing
Categories" — implying the complete list, which contradicted both the page's own
per-category detail and the League Standings ("trailing N-N-N"). This guards that
the misleading subset header does not return.
"""

from __future__ import annotations

from pathlib import Path

_PAGE = Path(__file__).resolve().parent.parent / "pages" / "1_My_Team.py"


def test_priority_callout_not_mislabeled_as_full_losing_list():
    src = _PAGE.read_text(encoding="utf-8")
    # The exact misleading subset header (a 2-item list titled as if it were the
    # full set of losing categories) must not ship.
    assert 'f"Losing Categories</div>"' not in src, (
        "My Team renders a 2-item priority subset under a 'Losing Categories' "
        "header — relabel it (e.g. 'Priority Targets') so it doesn't imply the full list."
    )
