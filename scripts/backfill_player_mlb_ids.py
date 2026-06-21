#!/usr/bin/env python
"""Backfill missing ``mlb_id`` on the ``players`` table.

WHY
    The React frontend resolves player headshots (``mlb_id``) and team logos
    (``team_id``) from ``PlayerRef``. A player that reaches the free-agent pool
    or a roster with ``mlb_id`` NULL/0 renders a broken (404) team logo — handled
    gracefully by an avatar fallback, so cosmetic, but the data is incomplete.
    This script resolves the ``mlb_id`` for the *relevant* subset (rostered /
    meaningfully-owned players) and writes it back.

SAFETY (Muncy-DNA discipline)
    * Only a CONFIDENT, UNIQUE (name [+ team]) match is written. Ambiguous or
      zero matches are SKIPPED and logged — never guessed. Assigning the WRONG
      ``mlb_id`` is strictly worse than leaving it NULL.
    * Never overwrites a non-null ``mlb_id`` (idempotent; re-runnable).
    * Refuses to write an ``mlb_id`` already owned by a different ``player_id``
      (that would create a DNA collision) — skips + logs instead.
    * Constrained to MLB (``sportId=1``).

RESOLUTION
    ``statsapi.lookup_player(name, season=...)`` with a season fallback
    (2026 has no lookup data yet -> 2025 -> 2024). statsapi name search is
    accent-insensitive, so correctly-accented names ("Pablo Lopez", "Felix
    Bautista") resolve directly. A unique result is accepted (after a light
    surname sanity check); when a name is shared by several players the match is
    disambiguated by the player's MLB team, and if that is still ambiguous the
    player is skipped.

    NOTE: the names in this DB are stored correctly with their accents (340 of
    9,888 rows carry valid non-ASCII characters; zero rows contain a U+FFFD
    replacement char). An earlier investigation read the console's ASCII display
    artifact ("Pablo L?pez") as data corruption — it is not. There is no encoding
    bug to repair here. A few low-ownership entries (e.g. unsigned 2025 draft
    picks) have no MLB id at all and are correctly left NULL.

PROPAGATION
    A local DB patch does NOT propagate to Railway production. This script is the
    operator's re-runnable tool — run it against the production DB too. Wiring it
    into the bootstrap ``mlb_id``-enrichment phase is an owner-gated follow-up.

USAGE
    python scripts/backfill_player_mlb_ids.py             # relevant subset (owned), write
    python scripts/backfill_player_mlb_ids.py --dry-run   # show what would change, write nothing
    python scripts/backfill_player_mlb_ids.py --all       # every null/0-mlb_id row (slow)
    python scripts/backfill_player_mlb_ids.py --min-owned 5  # ownership floor (default 0 -> >0)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path so `src` imports resolve when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.database import get_connection, load_player_pool  # noqa: E402
from src.player_id_resolver import resolve_mlb_id  # noqa: E402


def _safe(text: object) -> str:
    """cp1252-safe rendering for the Windows console (accents -> '?')."""
    return str(text).encode("ascii", "replace").decode("ascii")


def _load_targets(process_all: bool, min_owned: float) -> list[dict]:
    """Rows with null/0 mlb_id, driven by the enriched pool (carries percent_owned)."""
    pool = load_player_pool()
    if pool is None or pool.empty:
        return []
    name_col = "player_name" if "player_name" in pool.columns else "name"
    missing = pool["mlb_id"].isna() | (pool["mlb_id"] == 0)
    if not process_all:
        owned = pool.get("percent_owned")
        if owned is None:
            return []
        missing &= owned.fillna(0) > min_owned
    rows = []
    for _, r in pool[missing].iterrows():
        try:
            pid = int(r["player_id"])
        except (TypeError, ValueError):
            continue
        rows.append(
            {
                "player_id": pid,
                "name": str(r.get(name_col, "")),
                "team": (str(r.get("team")) if r.get("team") is not None else ""),
                "percent_owned": float(r.get("percent_owned") or 0),
            }
        )
    # Highest-ownership first so the most-relevant players are resolved first.
    rows.sort(key=lambda d: d["percent_owned"], reverse=True)
    return rows


def _existing_mlb_id_owners() -> dict[int, int]:
    """Map of mlb_id -> player_id for every row that already has a non-null id."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        rows = cur.execute("SELECT player_id, mlb_id FROM players WHERE mlb_id IS NOT NULL AND mlb_id != 0").fetchall()
    finally:
        conn.close()
    return {int(mlb_id): int(pid) for pid, mlb_id in rows}


def _write(updates: list[tuple[int, int]]) -> int:
    """Persist (player_id, mlb_id) updates; only fills NULL/0. Returns rows changed."""
    conn = get_connection()
    changed = 0
    try:
        cur = conn.cursor()
        for player_id, mlb_id in updates:
            cur.execute(
                "UPDATE players SET mlb_id = ? WHERE player_id = ? AND (mlb_id IS NULL OR mlb_id = 0)",
                (mlb_id, player_id),
            )
            changed += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backfill missing players.mlb_id.")
    parser.add_argument("--dry-run", action="store_true", help="resolve + report, write nothing")
    parser.add_argument(
        "--all",
        action="store_true",
        help="process EVERY null/0-mlb_id row, not just owned players (slow)",
    )
    parser.add_argument(
        "--min-owned",
        type=float,
        default=0.0,
        help="percent_owned floor (default 0 -> strictly >0); ignored with --all",
    )
    args = parser.parse_args(argv)

    targets = _load_targets(process_all=args.all, min_owned=args.min_owned)
    scope = "ALL null/0-mlb_id" if args.all else f"owned (>{args.min_owned}%)"
    print(f"Scanning {len(targets)} {scope} players with missing mlb_id...\n")
    if not targets:
        print("Nothing to do (no targets, or empty DB/pool).")
        return 0

    taken = _existing_mlb_id_owners()
    resolved: list[tuple[int, int]] = []
    assigned_this_run: dict[int, int] = {}
    skipped: list[tuple[dict, str]] = []

    for t in targets:
        mlb_id, reason = resolve_mlb_id(t["name"], t["team"])
        if mlb_id is None:
            skipped.append((t, reason))
            continue
        # Collision guard: never duplicate an mlb_id already in use elsewhere.
        owner = taken.get(mlb_id) or assigned_this_run.get(mlb_id)
        if owner is not None and owner != t["player_id"]:
            skipped.append((t, f"collision: mlb_id {mlb_id} already on player_id {owner}"))
            continue
        resolved.append((t["player_id"], mlb_id))
        assigned_this_run[mlb_id] = t["player_id"]
        print(
            f"  RESOLVED  {_safe(t['name']):<22} {t['team']:<4} "
            f"({t['percent_owned']:>4.1f}%) -> mlb_id {mlb_id}  [{reason}]"
        )

    for t, reason in skipped:
        print(f"  SKIPPED   {_safe(t['name']):<22} {t['team']:<4} ({t['percent_owned']:>4.1f}%)  [{reason}]")

    print(f"\nResolved {len(resolved)}, skipped {len(skipped)}.")
    if args.dry_run:
        print("Dry-run: no changes written.")
        return 0
    if not resolved:
        print("No confident matches to write.")
        return 0
    changed = _write(resolved)
    print(f"Wrote {changed} mlb_id value(s) to the players table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
