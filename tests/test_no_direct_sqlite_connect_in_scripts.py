"""Scripts must not call sqlite3.connect directly — use get_connection."""

import re
from pathlib import Path

SCRIPTS_TO_CHECK = [
    "scripts/draft_vs_current.py",
    "scripts/extract_trade_data.py",
    "scripts/optimal_roster_sim.py",
]


def test_no_direct_sqlite_connect_in_scripts():
    """Bypassing get_connection() drops WAL+busy_timeout protections.
    SF-4's root cause was concurrent writers without busy_timeout."""
    bad = []
    for s in SCRIPTS_TO_CHECK:
        p = Path(s)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        # Strip comments to avoid false positives
        text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
        # Strip strings
        text_no_strings = re.sub(r'"[^"]*"', "", text_no_comments)
        text_no_strings = re.sub(r"'[^']*'", "", text_no_strings)
        if re.search(r"\bsqlite3\.connect\s*\(", text_no_strings):
            bad.append(s)
    assert bad == [], (
        f"These scripts still call sqlite3.connect directly (bypasses WAL+busy_timeout): {bad}\n"
        f"Use `from src.database import get_connection; conn = get_connection()` instead."
    )
