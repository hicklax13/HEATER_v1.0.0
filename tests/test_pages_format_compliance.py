"""Pages must use format_stat for rate stats, not inline f-strings."""

import re
from pathlib import Path

PAGES = [
    "pages/9_Weekly_Recap.py",
    "pages/8_Weekly_Dashboard.py",
    "pages/10_Punt_Analyzer.py",
]


def test_no_inline_3f_format_for_era_or_whip():
    """f'{x:.3f}' is wrong precision for ERA/WHIP. Pages should use format_stat."""
    bad = []
    for p in PAGES:
        path = Path(p)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        # Look for .3f format applied near era/whip references
        for m in re.finditer(r":\.3f", text):
            # context: 100 chars before, 20 after
            ctx = text[max(0, m.start() - 100) : m.end() + 20].lower()
            if "era" in ctx or "whip" in ctx:
                bad.append(f"{p}:offset_{m.start()}")
    assert bad == [], f"Inline .3f near ERA/WHIP context: {bad}"
