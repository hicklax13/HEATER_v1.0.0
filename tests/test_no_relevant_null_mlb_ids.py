"""Guard: meaningfully-owned MLB players must have a resolvable ``mlb_id``.

The React frontend resolves player headshots (``mlb_id``) and team logos
(``team_id``) from ``PlayerRef``. A free-agent-pool / roster player with
``mlb_id`` NULL/0 renders a broken (404) team logo — handled gracefully by an
avatar fallback, so cosmetic, but the data is incomplete. The CMO's M3 live
verification hit exactly this. ``scripts/backfill_player_mlb_ids.py`` backfills
the missing ids; this guard keeps the high-relevance subset from silently
regressing.

DB-aware: SKIPS when the player pool is empty / lacks the needed columns. Git
worktrees and CI ship an empty ``draft_tool.db`` (see the
``reference_worktree_empty_db`` memory), so the assertion only runs against a
real, populated DB. It is count-based, so it is never flaky.
"""

from __future__ import annotations

import sqlite3

import pytest

from src.database import load_player_pool

# Players owned in >= this share of leagues are established MLB players that MUST
# have an mlb_id. Below this floor sit pre-debut prospects (e.g. unsigned 2025
# draft picks) that legitimately have NO MLB id yet, so they are out of scope —
# resolving them would require an id that does not exist. 5% cleanly separates
# the established players from those prospects.
_MIN_OWNED_PCT = 5.0


def _safe(text: object) -> str:
    """ascii-safe rendering so a failure message can't crash the cp1252 console."""
    return str(text).encode("ascii", "replace").decode("ascii")


def test_no_relevant_null_mlb_ids():
    # Unbuilt / missing DB (git worktrees & CI): SKIP. A real schema/DB
    # regression raises a different exception and MUST fail loudly, not skip
    # (the normal empty-DB case returns an empty DataFrame, handled below).
    try:
        pool = load_player_pool()
    except (sqlite3.OperationalError, FileNotFoundError) as exc:  # pragma: no cover
        pytest.skip(f"player DB unavailable ({type(exc).__name__}) — nothing to guard")

    if pool is None or getattr(pool, "empty", True):
        pytest.skip("player pool empty (worktree/CI empty DB) — nothing to guard")
    if "percent_owned" not in pool.columns or "mlb_id" not in pool.columns:
        pytest.skip("player pool missing percent_owned/mlb_id columns — nothing to guard")

    owned = pool["percent_owned"].fillna(0)
    missing = pool["mlb_id"].isna() | (pool["mlb_id"] == 0)
    offenders = pool[(owned >= _MIN_OWNED_PCT) & missing]

    if offenders.empty:
        return

    name_col = "player_name" if "player_name" in pool.columns else "name"
    detail = ", ".join(
        f"{_safe(r.get(name_col))} ({_safe(r.get('team'))}, {float(r['percent_owned']):.0f}%)"
        for _, r in offenders.iterrows()
    )
    raise AssertionError(
        f"{len(offenders)} player(s) owned >= {_MIN_OWNED_PCT}% have a null/0 mlb_id "
        f"(headshots/logos will 404). Run scripts/backfill_player_mlb_ids.py: {detail}"
    )
