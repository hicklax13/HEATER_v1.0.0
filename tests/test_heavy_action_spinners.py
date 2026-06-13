"""Structural guard: heavy compute actions must be wrapped in st.spinner.

Task 2.5 — progress feedback on heavy actions (HIGH):
- pages/2_Line-up_Optimizer.py "Optimize Lineup" click must use
  st.spinner("Optimizing lineup…")
- pages/6_League_Standings.py projection/season sim must use
  st.spinner("Simulating remaining season…")

Uses a simple regex scan (same pattern as test_lineup_optimizer_26_weeks.py)
so no Streamlit runtime is required and the test stays fast.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_OPT = REPO_ROOT / "pages" / "2_Line-up_Optimizer.py"
PAGE_STD = REPO_ROOT / "pages" / "6_League_Standings.py"

# Pattern: st.spinner("Optimizing lineup…") — allow straight or curly ellipsis
_SPINNER_OPT = re.compile(r'st\.spinner\s*\(\s*["\']Optimizing lineup[…\.]{1,3}["\']')
# Pattern: st.spinner("Simulating remaining season…")
_SPINNER_STD = re.compile(r'st\.spinner\s*\(\s*["\']Simulating remaining season[…\.]{1,3}["\']')


def test_optimizer_page_exists():
    assert PAGE_OPT.exists(), "pages/2_Line-up_Optimizer.py must exist"


def test_standings_page_exists():
    assert PAGE_STD.exists(), "pages/6_League_Standings.py must exist"


def test_optimizer_has_spinner():
    """The 'Optimize Lineup' click path must be wrapped in st.spinner('Optimizing lineup…')."""
    text = PAGE_OPT.read_text(encoding="utf-8")
    assert _SPINNER_OPT.search(text), (
        "pages/2_Line-up_Optimizer.py: optimize-click path must contain "
        'st.spinner("Optimizing lineup…") — user has no feedback during the LP solve.'
    )


def test_standings_has_spinner():
    """The season-projection sim path must be wrapped in st.spinner('Simulating remaining season…')."""
    text = PAGE_STD.read_text(encoding="utf-8")
    assert _SPINNER_STD.search(text), (
        "pages/6_League_Standings.py: projection sim path must contain "
        'st.spinner("Simulating remaining season…") — user has no feedback during MC sim.'
    )
