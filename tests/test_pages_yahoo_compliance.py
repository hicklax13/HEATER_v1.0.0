"""Pages must use yahoo_data_service, not raw SQL load_league_*."""

import re
from pathlib import Path

PAGES_TO_CHECK = [
    "pages/15_Weekly_Recap.py",
    "pages/7_Weekly_Dashboard.py",
    "pages/10_Category_Tracker.py",
    "pages/9_Waiver_Wire.py",
    "pages/8_Trade_Values.py",
    "pages/14_Punt_Analyzer.py",
    "pages/12_Playoff_Odds.py",
]


def test_pages_dont_call_load_league_rosters_directly():
    bad = []
    for p in PAGES_TO_CHECK:
        path = Path(p)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bload_league_rosters\s*\(", text):
            bad.append(p)
    assert bad == [], f"These pages still call load_league_rosters directly: {bad}"


def test_pages_dont_call_load_league_standings_directly():
    bad = []
    for p in PAGES_TO_CHECK:
        path = Path(p)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"\bload_league_standings\s*\(", text):
            bad.append(p)
    assert bad == [], f"These pages still call load_league_standings directly: {bad}"
