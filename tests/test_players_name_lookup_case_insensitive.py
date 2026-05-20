"""SFH LOW-2 guard (2026-05-20): every `SELECT FROM players WHERE name = ?`
in `src/` must use `COLLATE NOCASE`.

Background: the `players` table has no UNIQUE constraint on `name`. Different
upstream sources can write the same player with subtly different casing
("Ronald Acuna Jr." vs "Ronald Acuña Jr." with different casing). A
case-sensitive `WHERE name = ?` lookup will miss the existing row, the
upsert path will silently create a duplicate, and downstream joins will
silently match the wrong `player_id` — the same Muncy DNA that prompted
the original April 8 stat-mix-up.

PR #49 (H1) first introduced COLLATE NOCASE in the news-fetcher entry
point. PR #62 (L-2) extended it to `match_player_id`. PR #74 follow-up
patched the remaining 12 sites across `data_bootstrap`, `data_pipeline`,
`database`, `league_manager`, and `live_stats`. This guard prevents
regression — if a future PR adds a new case-sensitive lookup, the test
fails with a clear pointer.

Allowlist policy: there is no allowlist. Every `WHERE name = ?` query
against the `players` table must be case-insensitive. If you have a
genuine case-sensitive requirement (binary comparison), document the
exact contract and request explicit review.
"""

from __future__ import annotations

import pathlib
import re

_SRC_DIR = pathlib.Path("src")

# Match `WHERE name = ?` followed by something OTHER than `COLLATE` (case-
# insensitive for the keyword check, but the bug class is specifically about
# missing COLLATE — note the next non-whitespace token).
# We deliberately keep the regex narrow to the players-table lookup pattern;
# unrelated tables (yahoo_free_agents, league_rosters, refresh_log, etc.)
# are tracked separately.
_BAD_PATTERN = re.compile(
    r"FROM\s+players\s+WHERE\s+name\s*=\s*\?(?!\s*COLLATE\s+NOCASE)",
    re.IGNORECASE,
)


def test_no_case_sensitive_players_name_lookups_in_src():
    """Every `WHERE name = ?` against the players table in src/ must
    use `COLLATE NOCASE`. New violations indicate a regression of the
    SFH LOW-2 cleanup from 2026-05-20."""
    offenders: list[str] = []

    for py_file in _SRC_DIR.rglob("*.py"):
        # Skip __pycache__ etc.
        if "__pycache__" in py_file.parts:
            continue
        text = py_file.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _BAD_PATTERN.search(line):
                offenders.append(f"  {py_file.as_posix()}:{lineno} — {line.strip()}")

    assert not offenders, (
        "\nCase-sensitive `WHERE name = ?` lookups against the `players` table found. "
        "The `players` table has no UNIQUE constraint on `name`, so case-sensitive "
        "lookups silently miss the canonical row and create duplicates (Muncy-bug "
        "DNA from April 8 2026). Add `COLLATE NOCASE` after the `?`:\n\n" + "\n".join(offenders)
    )
