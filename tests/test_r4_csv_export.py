"""R-4 CSV export — structural TDD tests for Free Agents + Leaders pages.

Verifies that each page calls st.download_button(...) with:
  - file_name ending in ".csv"
  - mime="text/csv"
  - a unique key= argument

These are AST source-scan tests (no Streamlit runtime required), following the
same pattern as test_wave9_free_agents_level_filter.py.

Written first (TDD); they fail until the implementation is added.
"""

from __future__ import annotations

import ast
from pathlib import Path

_PAGES = Path(__file__).resolve().parents[1] / "pages"


# ---------------------------------------------------------------------------
# Shared AST helpers
# ---------------------------------------------------------------------------


def _source(glob: str) -> str:
    hits = list(_PAGES.glob(glob))
    assert hits, f"No page matching {glob!r} found in pages/"
    return hits[0].read_text(encoding="utf-8")


def _dl_calls(source: str) -> list[ast.Call]:
    tree = ast.parse(source)
    out: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (
            isinstance(fn, ast.Attribute)
            and fn.attr == "download_button"
            and isinstance(fn.value, ast.Name)
            and fn.value.id == "st"
        ):
            out.append(node)
    return out


def _kw_str(call: ast.Call, name: str) -> str | None:
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _csv_calls(source: str) -> list[ast.Call]:
    return [
        c
        for c in _dl_calls(source)
        if (_kw_str(c, "file_name") or "").endswith(".csv") and _kw_str(c, "mime") == "text/csv"
    ]


# ---------------------------------------------------------------------------
# Free Agents tests
# ---------------------------------------------------------------------------


def test_fa_has_csv_download_button():
    """pages/14_Free_Agents.py must have st.download_button with .csv file_name
    and mime='text/csv'."""
    src = _source("*Free_Agents*.py")
    assert _csv_calls(src), (
        "14_Free_Agents.py: missing st.download_button(file_name='*.csv', mime='text/csv'). "
        "Add a CSV export for the ranked FA table."
    )


def test_fa_csv_button_has_key_fa_csv():
    """The FA CSV button must use key='fa_csv' (as specified in the task)."""
    src = _source("*Free_Agents*.py")
    keyed = [c for c in _csv_calls(src) if _kw_str(c, "key") == "fa_csv"]
    assert keyed, "14_Free_Agents.py: CSV download_button must use key='fa_csv'."


def test_fa_csv_file_name():
    """The FA CSV download must use file_name='heater_free_agents.csv'."""
    src = _source("*Free_Agents*.py")
    named = [c for c in _csv_calls(src) if _kw_str(c, "file_name") == "heater_free_agents.csv"]
    assert named, "14_Free_Agents.py: CSV download_button should use file_name='heater_free_agents.csv'."


# ---------------------------------------------------------------------------
# Leaders tests
# ---------------------------------------------------------------------------


def test_leaders_has_csv_download_buttons():
    """pages/17_Leaders.py must have at least one st.download_button with .csv
    file_name and mime='text/csv'."""
    src = _source("*Leaders*.py")
    assert _csv_calls(src), (
        "17_Leaders.py: missing st.download_button(file_name='*.csv', mime='text/csv'). "
        "Add CSV exports for the leaderboard tabs."
    )


def test_leaders_csv_buttons_have_unique_keys():
    """All CSV download_buttons in 17_Leaders.py must have unique key= args."""
    src = _source("*Leaders*.py")
    calls = _csv_calls(src)
    keys = [_kw_str(c, "key") for c in calls]
    assert all(k is not None for k in keys), (
        f"17_Leaders.py: some CSV download_button is missing a key= argument. Keys: {keys}"
    )
    assert len(keys) == len(set(keys)), f"17_Leaders.py: duplicate CSV download_button keys detected: {keys}"


def test_leaders_csv_covers_category_leaders_tab():
    """17_Leaders.py must have a CSV export whose file_name contains 'leader'
    or 'category' (covers the Category Leaders tab)."""
    src = _source("*Leaders*.py")
    tab1 = [
        c
        for c in _csv_calls(src)
        if any(kw in (_kw_str(c, "file_name") or "").lower() for kw in ("leader", "category"))
    ]
    assert tab1, (
        "17_Leaders.py: no CSV download_button with 'leader' or 'category' in file_name. "
        "Add one inside the Category Leaders (tab1) section."
    )


def test_leaders_csv_covers_value_tab():
    """17_Leaders.py must have a CSV export whose file_name contains 'value'
    (covers the Category Value tab)."""
    src = _source("*Leaders*.py")
    tab2 = [c for c in _csv_calls(src) if "value" in (_kw_str(c, "file_name") or "").lower()]
    assert tab2, (
        "17_Leaders.py: no CSV download_button with 'value' in file_name. "
        "Add one inside the Category Value (tab2) section."
    )
