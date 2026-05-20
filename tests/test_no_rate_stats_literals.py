"""D6 invariant: no hardcoded {"AVG", "OBP", "ERA", "WHIP"} rate-stat literals
outside LeagueConfig.

The rate-stat set is league-format-specific. Hardcoding it leaks scoring
knowledge into modules that should read from LeagueConfig.rate_stats.

Allowed:
  - src/valuation.py — defines LeagueConfig.rate_stats (canonical)
  - tests/ — test literals are acceptable
  - docs/ — documentation

NOT flagged:
  - Pitching-rate-only sets {"ERA", "WHIP"} (D4 territory or pitching-specific)
  - Subsets containing some but not all 4 rate cats
"""

from __future__ import annotations

import re
from pathlib import Path

# Match a SET literal containing EXACTLY 4 quoted string elements, all
# from {AVG, OBP, ERA, WHIP} (any order).
RATE_4ELT_SET = re.compile(
    r"""
    \{                                              # opening brace
    \s*
    "(?P<a>AVG|OBP|ERA|WHIP)"
    \s*,\s*
    "(?P<b>AVG|OBP|ERA|WHIP)"
    \s*,\s*
    "(?P<c>AVG|OBP|ERA|WHIP)"
    \s*,\s*
    "(?P<d>AVG|OBP|ERA|WHIP)"
    \s*,?\s*
    \}
    """,
    re.VERBOSE,
)

ALLOWED_FILES = {Path("src/valuation.py")}
ALLOWED_DIRS = {"tests", "docs"}


def _line_has_rate_4elt(line: str) -> bool:
    for m in RATE_4ELT_SET.finditer(line):
        elts = {m.group("a"), m.group("b"), m.group("c"), m.group("d")}
        if elts == {"AVG", "OBP", "ERA", "WHIP"}:
            return True
    return False


def test_no_rate_stats_literal_outside_league_config():
    """Hardcoded {AVG, OBP, ERA, WHIP} rate-stat literals must use LeagueConfig.rate_stats."""
    root = Path(__file__).resolve().parent.parent
    targets = []
    for sub in ("src", "pages", "scripts"):
        d = root / sub
        if d.exists():
            targets.extend(d.rglob("*.py"))
    offenders: list[str] = []
    for f in targets:
        rel = f.relative_to(root)
        if rel in ALLOWED_FILES or rel.parts[0] in ALLOWED_DIRS:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                continue
            # Strip inline comments before matching.
            code_part = line.split("#", 1)[0]
            if _line_has_rate_4elt(code_part):
                offenders.append(f"{rel}:{lineno}: {stripped[:120]}")
    assert not offenders, (
        "Hardcoded rate-stat literal {AVG, OBP, ERA, WHIP} found outside LeagueConfig. "
        "Use `LeagueConfig().rate_stats` instead (or `(config or LeagueConfig()).rate_stats` "
        "if a config may be passed):\n  " + "\n  ".join(offenders)
    )
