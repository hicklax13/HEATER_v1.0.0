"""D4 invariant: no hardcoded {"L", "ERA", "WHIP"} inverse-stat literals
outside LeagueConfig.

The inverse-stat set is FourzynBurn-specific (other leagues may use different
inverse-stat cats). Hardcoding it leaks league-specific knowledge into
modules that should read from LeagueConfig.inverse_stats.

Allowed:
  - src/valuation.py — defines LeagueConfig.inverse_stats (canonical)
  - tests/ — test literals are acceptable
  - docs/ — documentation

NOT flagged (different concept):
  - SUPERSETS of {L, ERA, WHIP} like pitching-cat lists `{W, L, SV, K, ERA, WHIP}`
    (these are PITCHING category lists, not inverse-stat sets)
  - {"ERA", "WHIP"} without L (separate D6 concern about rate-stats, not D4)
"""

from __future__ import annotations

import re
from pathlib import Path

# Match a SET literal containing EXACTLY 3 quoted string elements, all
# from {L, ERA, WHIP} (any order). Permits whitespace + trailing comma.
INVERSE_3ELT_SET = re.compile(
    r"""
    \{                                  # opening brace
    \s*                                  # optional whitespace
    "(?P<a>L|ERA|WHIP)"                  # 1st element (quoted)
    \s*,\s*                              # comma separator
    "(?P<b>L|ERA|WHIP)"                  # 2nd element
    \s*,\s*                              # comma separator
    "(?P<c>L|ERA|WHIP)"                  # 3rd element
    \s*,?\s*                             # optional trailing comma + ws
    \}                                   # closing brace
    """,
    re.VERBOSE,
)

# Also frozenset({...}) form
INVERSE_3ELT_FROZENSET = re.compile(
    r"""
    frozenset\s*\(\s*\{                  # frozenset({
    \s*
    "(?P<a>L|ERA|WHIP)"
    \s*,\s*
    "(?P<b>L|ERA|WHIP)"
    \s*,\s*
    "(?P<c>L|ERA|WHIP)"
    \s*,?\s*
    \}\s*\)                              # })
    """,
    re.VERBOSE,
)

ALLOWED_FILES = {Path("src/valuation.py")}
ALLOWED_DIRS = {"tests", "docs"}


def _line_has_inverse_3elt(line: str) -> bool:
    for pat in (INVERSE_3ELT_SET, INVERSE_3ELT_FROZENSET):
        for m in pat.finditer(line):
            elts = {m.group("a"), m.group("b"), m.group("c")}
            if elts == {"L", "ERA", "WHIP"}:
                return True
    return False


def test_no_inverse_cats_literal_outside_league_config():
    """Hardcoded {L, ERA, WHIP} inverse-stat literals must use LeagueConfig.inverse_stats."""
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
            # Strip inline comment (# ...) so doc-only literals don't trigger.
            # Naive: split on '#' not inside a string. For the patterns we
            # care about, the code never has '#' inside the set-literal itself,
            # so first-'#' split is safe enough.
            code_part = line.split("#", 1)[0]
            if _line_has_inverse_3elt(code_part):
                offenders.append(f"{rel}:{lineno}: {stripped[:120]}")
    assert not offenders, (
        "Hardcoded inverse-stat literal {L, ERA, WHIP} found outside LeagueConfig. "
        "Use `LeagueConfig().inverse_stats` instead (or `(config or LeagueConfig()).inverse_stats` "
        "if a config may be passed):\n  " + "\n  ".join(offenders)
    )
