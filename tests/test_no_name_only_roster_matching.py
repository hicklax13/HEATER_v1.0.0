"""PR22 structural invariant (2026-05-21): roster-sync code paths must
not silently resolve Yahoo→DB player matches via name-only SQL queries.
They must call `match_player_id(name, editorial_team_abbr)` so the DNA
collision warning fires when Yahoo's team disagrees with the stored
player's team.

Background — the Muncy DNA collision:
  The roster sync in `src/yahoo_api.py:write_to_database` already goes
  through `match_player_id`, which has the warning. But a future PR
  could add a NEW roster-sync code path that bypasses `match_player_id`
  and does its own `cursor.execute("SELECT player_id FROM players
  WHERE name = ?", (name,))`. That bypass would silently re-introduce
  the Muncy bug. This guard catches it at CI time.

How the guard works:
  Parse the AST of the three modules that write to `league_rosters`.
  For each function, if it contains a string literal mentioning
  `INSERT INTO league_rosters` (or calls `upsert_league_roster_entry`),
  scan it for any `WHERE name = ?` string literal. If found, the
  function is doing inline name-resolution — fail unless allowlisted.

Allowlist:
  - `src/league_manager.py:import_league_rosters_csv` — legacy CSV
    importer. CSV format has no team column, so name-only is the only
    available signal. CSV imports are operator-driven (not periodic
    Yahoo sync) and the failure mode is loud (no team → DNA collision
    warning impossible by construction).

Files watched: yahoo_data_service.py, league_manager.py, data_bootstrap.py
(yahoo_api.py uses match_player_id correctly and is NOT in this guard.)
"""

from __future__ import annotations

import ast
import pathlib
import re

_WATCHED = [
    pathlib.Path("src/yahoo_data_service.py"),
    pathlib.Path("src/league_manager.py"),
    pathlib.Path("src/data_bootstrap.py"),
]

# Functions explicitly exempt from this guard. Format: "module_basename:func_name".
# Keep this list MINIMAL. Each entry is a deliberate design decision documented
# in the calling-site comments.
_ALLOWLIST: set[str] = {
    # Legacy CSV importer: CSV format has no team column, so name-only is
    # the only signal available. Operator-driven, not periodic. Not the
    # Muncy DNA vector (that's Yahoo's auto-sync path).
    "league_manager.py:import_league_rosters_csv",
}

_NAME_ONLY_PATTERN = re.compile(r"WHERE\s+name\s*=\s*\?", re.IGNORECASE)
_ROSTER_WRITE_MARKERS = (
    "INSERT INTO league_rosters",
    "upsert_league_roster_entry",
)


def _function_text(source: str, fn: ast.FunctionDef) -> str:
    """Return the source text for a function (best-effort, line-based)."""
    lines = source.splitlines()
    start = fn.lineno - 1
    end = fn.end_lineno if fn.end_lineno is not None else fn.lineno
    return "\n".join(lines[start:end])


def test_no_name_only_roster_matching():
    offenders: list[str] = []

    for path in _WATCHED:
        assert path.exists(), f"Watched file missing: {path}"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)

        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            fn_text = _function_text(source, node)
            # Only flag functions that ALSO write to league_rosters — i.e. they
            # are roster-sync sites, not unrelated lookups.
            if not any(marker in fn_text for marker in _ROSTER_WRITE_MARKERS):
                continue

            key = f"{path.name}:{node.name}"
            if key in _ALLOWLIST:
                continue

            if _NAME_ONLY_PATTERN.search(fn_text):
                offenders.append(
                    f"  {path.as_posix()}::{node.name} — function writes to "
                    "league_rosters AND contains a raw `WHERE name = ?` SQL "
                    "match. Use `match_player_id(name, editorial_team_abbr)` "
                    "instead so the DNA collision warning fires when Yahoo's "
                    "team disagrees with the stored player's team."
                )

    assert not offenders, (
        "\nPR22 structural invariant violation. Roster-sync code paths must "
        "not silently resolve Yahoo→DB player matches via name-only SQL. The "
        "Muncy DNA collision (April 8 2026) happened because the (name+team) "
        "match found nothing, the name-only fallback silently picked the wrong "
        "player, and no warning fired. Route all roster-sync resolution through "
        "`match_player_id(name, editorial_team_abbr)` which now warns on team "
        "mismatch. Offenders:\n\n" + "\n".join(offenders)
    )
