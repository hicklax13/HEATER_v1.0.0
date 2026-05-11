"""Permanent guard against BUG-003 regression.

Asserts that no source file references `lr.name` or `WHERE name = ?` patterns
in the context of league_rosters (which has NO `name` column).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_files_under(*dirs: str):
    for d in dirs:
        for p in (REPO_ROOT / d).rglob("*.py"):
            # Skip the guard test itself (this file)
            if p.name == "test_no_lr_name_column_refs.py":
                continue
            yield p


def test_no_lr_name_join_in_src():
    """`lr.name = ...` patterns reference a non-existent column on league_rosters."""
    offenders: list[tuple[Path, int, str]] = []
    pat = re.compile(r"\blr\.name\b")
    for p in _python_files_under("src", "scripts", "pages"):
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            if pat.search(line):
                offenders.append((p, lineno, line.strip()))
    assert not offenders, (
        "BUG-003 regression: `lr.name` reference(s) found — league_rosters has no "
        "`name` column. Offenders:\n" + "\n".join(f"  {p}:{n}: {ln}" for p, n, ln in offenders)
    )


def test_no_league_rosters_update_by_name():
    """`UPDATE league_rosters SET ... WHERE name = ?` is a BUG-003 pattern.

    Constrains the wildcard to non-quote characters so the match cannot cross
    a string-literal boundary (e.g. two separate SQL statements on adjacent
    lines whose combined text happens to spell out the bad pattern).
    """
    pat = re.compile(r"UPDATE\s+league_rosters\s+SET[^\"']*?WHERE\s+name\s*=", re.IGNORECASE)
    offenders: list[tuple[Path, str]] = []
    for p in _python_files_under("src", "scripts", "pages"):
        text = p.read_text(encoding="utf-8")
        for m in pat.finditer(text):
            offenders.append((p, m.group(0)[:120]))
    assert not offenders, (
        "BUG-003 regression: `UPDATE league_rosters SET ... WHERE name = ?` "
        f"pattern found. Use player_id. Offenders: {offenders}"
    )
