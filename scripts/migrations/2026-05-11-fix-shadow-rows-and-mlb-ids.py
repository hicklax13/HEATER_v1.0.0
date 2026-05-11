#!/usr/bin/env python
"""Migration 2026-05-11: fix shadow rows + backfill missing mlb_ids.

Addresses BUG-001 and BUG-002 from the 2026-05-11 bug audit.

BUG-001 root cause: 33 player rows exist with team='MLB' and fabricated
mlb_id values in [600000, 601999]. These IDs resolve to DSL/VSL minor-league
prospects when queried against MLB Stats API. 4 of these rows are currently
on league rosters (Burnes pid 110, Jared Jones pid 130, Schwellenbach pid 164,
Joyce pid 187), so any live-stats refresh pulls Brian Escolastico's stats
into Teoscar's row, etc.

BUG-002: 3 rostered players have NULL mlb_id (Greene pid 4643, Pepiot 4641,
Crews 922), invisible to mlb_id-keyed bootstrap phases.

Migration is idempotent. Always pass `--dry-run` first to preview changes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.database import get_connection  # noqa: E402

logger = logging.getLogger("migrate.shadow_rows")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# Known rostered players with NULL mlb_id (BUG-002).
# mlb_ids verified via statsapi.lookup_player at audit time.
KNOWN_NULL_MLB_BACKFILL: dict[int, int] = {
    4643: 668881,  # Hunter Greene (CIN, RHP)
    4641: 686752,  # Ryan Pepiot (TBR, RHP)
    922: 686611,  # Dylan Crews (WSN, OF) — verified 2026-05-11 (NOT 696285 which is Jacob Young)
}

# Rostered shadow rows that need IN-PLACE mlb_id repair (BUG-001 primary).
# These shadow player_ids HAVE no real twin in the players table, so they
# cannot be repointed-and-deleted. Instead we repair the existing row by
# replacing the fake mlb_id with the verified real one. Once mlb_id is
# outside the SHADOW_PLAYER_ID_RANGE, the row no longer matches the
# structural-guard definition of "shadow".
#
# Real mlb_ids verified via MLB Stats API on 2026-05-11.
# team='MLB' is intentionally NOT corrected here — the next bootstrap
# refresh of `players` will overwrite it with the current team.
KNOWN_SHADOW_REPAIRS: dict[int, int] = {
    110: 669203,  # Corbin Burnes
    130: 683003,  # Jared Jones
    164: 680885,  # Spencer Schwellenbach
    187: 690829,  # Ben Joyce
}

# Shadow rows are identified by team='MLB' AND mlb_id BETWEEN 600000 AND 601999.
SHADOW_PLAYER_ID_RANGE = (600000, 601999)


def find_shadow_rows(conn) -> list[tuple[int, str, int]]:
    """Returns list of (player_id, name, mlb_id) for shadow rows."""
    cur = conn.execute(
        """SELECT player_id, name, mlb_id FROM players
           WHERE team = 'MLB' AND mlb_id BETWEEN ? AND ?
           ORDER BY player_id""",
        SHADOW_PLAYER_ID_RANGE,
    )
    return list(cur.fetchall())


def find_real_twin(conn, shadow_name: str, shadow_pid: int) -> int | None:
    """Find a non-shadow player with the same name. Returns real player_id or None."""
    cur = conn.execute(
        """SELECT player_id FROM players
           WHERE name = ? AND player_id != ?
             AND NOT (team = 'MLB' AND mlb_id BETWEEN ? AND ?)
           ORDER BY player_id LIMIT 1""",
        (shadow_name, shadow_pid, *SHADOW_PLAYER_ID_RANGE),
    )
    row = cur.fetchone()
    return row[0] if row else None


def find_rostered_null_mlb(conn) -> list[tuple[int, str]]:
    """Returns (player_id, name) for rostered players with NULL mlb_id."""
    cur = conn.execute(
        """SELECT DISTINCT p.player_id, p.name FROM players p
           JOIN league_rosters lr ON p.player_id = lr.player_id
           WHERE p.mlb_id IS NULL
           ORDER BY p.player_id"""
    )
    return list(cur.fetchall())


def run_migration(dry_run: bool = True) -> dict[str, int]:
    """Run the migration. Returns dict of action counts."""
    counts = {
        "rostered_shadows_repaired": 0,
        "shadow_rows_found": 0,
        "league_rosters_repointed": 0,
        "shadow_rows_deleted": 0,
        "null_mlb_backfilled": 0,
        "null_mlb_still_missing": 0,
    }
    conn = get_connection()
    try:
        # === Phase 0: Repair rostered shadow rows in place (BUG-001 primary) ===
        logger.info("Phase 0: repair rostered shadow rows in place (UPDATE mlb_id)")
        # Track which shadow_pids were (or would be) repaired in place so that
        # Phase 2 can filter them out. In --commit mode, the UPDATE is visible
        # to subsequent SELECTs on the same connection; in --dry-run mode it
        # isn't, so we filter explicitly to keep both modes consistent.
        repaired_in_place_pids: set[int] = set()
        for shadow_pid, real_mlb_id in KNOWN_SHADOW_REPAIRS.items():
            cur = conn.execute("SELECT name, team, mlb_id FROM players WHERE player_id = ?", (shadow_pid,))
            row = cur.fetchone()
            if row is None:
                logger.warning("  pid=%d not found in players (skip)", shadow_pid)
                continue
            name, team, current_mlb = row
            if current_mlb == real_mlb_id:
                logger.info(
                    "  pid=%d %s already has mlb_id=%d (idempotent skip)",
                    shadow_pid,
                    name,
                    real_mlb_id,
                )
                continue
            # Sanity check: only repair if current mlb_id IS in the shadow range
            in_shadow_range = (
                current_mlb is not None and SHADOW_PLAYER_ID_RANGE[0] <= current_mlb <= SHADOW_PLAYER_ID_RANGE[1]
            )
            if not in_shadow_range:
                logger.warning(
                    "  pid=%d %s has unexpected mlb_id=%s (not in shadow range); manual review, skipping",
                    shadow_pid,
                    name,
                    current_mlb,
                )
                continue
            logger.info(
                "  pid=%d %s team=%s fake_mlb=%d → real_mlb=%d",
                shadow_pid,
                name,
                team,
                current_mlb,
                real_mlb_id,
            )
            if not dry_run:
                conn.execute(
                    "UPDATE players SET mlb_id = ? WHERE player_id = ? AND mlb_id = ?",
                    (real_mlb_id, shadow_pid, current_mlb),
                )
            counts["rostered_shadows_repaired"] += 1
            repaired_in_place_pids.add(shadow_pid)

        # === Phase 1: Backfill known NULL mlb_ids (BUG-002) ===
        logger.info("Phase 1: backfill known NULL mlb_ids")
        for pid, real_mlb_id in KNOWN_NULL_MLB_BACKFILL.items():
            cur = conn.execute("SELECT name, mlb_id FROM players WHERE player_id = ?", (pid,))
            row = cur.fetchone()
            if row is None:
                logger.warning("  pid=%d not found in players (skip)", pid)
                continue
            name, current_mlb = row
            if current_mlb == real_mlb_id:
                logger.info("  pid=%d %s already has mlb_id=%d (idempotent skip)", pid, name, real_mlb_id)
                continue
            if current_mlb is not None:
                logger.warning(
                    "  pid=%d %s has unexpected mlb_id=%s (expected NULL); manual review needed, skipping",
                    pid,
                    name,
                    current_mlb,
                )
                continue
            logger.info("  pid=%d %s NULL → mlb_id=%d", pid, name, real_mlb_id)
            if not dry_run:
                conn.execute(
                    "UPDATE players SET mlb_id = ? WHERE player_id = ? AND mlb_id IS NULL",
                    (real_mlb_id, pid),
                )
            counts["null_mlb_backfilled"] += 1

        # Report any remaining NULL-mlb_id rostered players (not in known list)
        remaining_nulls = find_rostered_null_mlb(conn)
        remaining_unknown = [r for r in remaining_nulls if r[0] not in KNOWN_NULL_MLB_BACKFILL]
        if remaining_unknown:
            logger.warning("Rostered players with NULL mlb_id not in backfill table (manual fix needed):")
            for pid, name in remaining_unknown:
                logger.warning("  pid=%d name=%s", pid, name)
            counts["null_mlb_still_missing"] = len(remaining_unknown)

        # === Phase 2: Identify shadow rows + repoint league_rosters refs (BUG-001) ===
        logger.info("Phase 2: identify shadow rows and repoint league_rosters refs")
        shadow_rows = find_shadow_rows(conn)
        # In --commit mode the Phase 0 UPDATEs above are visible to this SELECT and
        # the repaired rows already drop out naturally. In --dry-run mode we must
        # filter them explicitly so the preview matches what --commit would do.
        if repaired_in_place_pids:
            shadow_rows = [r for r in shadow_rows if r[0] not in repaired_in_place_pids]
        counts["shadow_rows_found"] = len(shadow_rows)
        logger.info("  found %d shadow rows (after excluding Phase 0 repairs)", len(shadow_rows))

        # Track shadow_pids whose refs were (or would be) repointed in Phase 2.
        # In dry-run we don't actually UPDATE, so Phase 3's "still referenced?"
        # check would falsely keep these alive. Tracking the set lets Phase 3
        # predict the post-commit deletable count consistently for both modes.
        repointed_pids: set[int] = set()

        for shadow_pid, shadow_name, shadow_mlb in shadow_rows:
            twin_pid = find_real_twin(conn, shadow_name, shadow_pid)
            lr_refs = conn.execute(
                "SELECT id, team_name FROM league_rosters WHERE player_id = ?",
                (shadow_pid,),
            ).fetchall()
            if not lr_refs:
                logger.info(
                    "  shadow pid=%d %s (fake_mlb=%d) — no league_rosters refs, deletable",
                    shadow_pid,
                    shadow_name,
                    shadow_mlb,
                )
                continue
            if twin_pid is None:
                logger.warning(
                    "  shadow pid=%d %s (fake_mlb=%d) HAS %d league_rosters refs but NO real twin in "
                    "players table. Cannot safely delete. Manual fix: add real player + repoint.",
                    shadow_pid,
                    shadow_name,
                    shadow_mlb,
                    len(lr_refs),
                )
                continue
            logger.info(
                "  shadow pid=%d %s (fake_mlb=%d) → repoint %d league_rosters rows to real pid=%d",
                shadow_pid,
                shadow_name,
                shadow_mlb,
                len(lr_refs),
                twin_pid,
            )
            if not dry_run:
                conn.execute(
                    "UPDATE league_rosters SET player_id = ? WHERE player_id = ?",
                    (twin_pid, shadow_pid),
                )
            counts["league_rosters_repointed"] += len(lr_refs)
            repointed_pids.add(shadow_pid)

        # === Phase 3: Delete shadow rows with no remaining refs ===
        logger.info("Phase 3: delete shadow rows with no remaining league_rosters refs")
        deletable = []
        for shadow_pid, shadow_name, _ in shadow_rows:
            if shadow_pid in repointed_pids:
                # Phase 2 already repointed (or will repoint in dry-run) all refs
                # for this shadow row, so it is deletable regardless of what the
                # live DB still shows.
                deletable.append((shadow_pid, shadow_name))
                continue
            still_refed = conn.execute(
                "SELECT 1 FROM league_rosters WHERE player_id = ? LIMIT 1",
                (shadow_pid,),
            ).fetchone()
            if not still_refed:
                deletable.append((shadow_pid, shadow_name))
        logger.info("  %d shadow rows are now safely deletable", len(deletable))
        for shadow_pid, shadow_name in deletable:
            logger.info("  DELETE shadow pid=%d %s", shadow_pid, shadow_name)
            if not dry_run:
                conn.execute("DELETE FROM players WHERE player_id = ?", (shadow_pid,))
            counts["shadow_rows_deleted"] += 1

        if not dry_run:
            conn.commit()
            logger.info("Migration committed.")
        else:
            logger.info("DRY RUN — no changes committed.")
    finally:
        conn.close()
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description="Migrate shadow rows + backfill mlb_ids")
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Preview changes without committing (recommended first run)",
    )
    p.add_argument(
        "--commit", action="store_true", default=False, help="Actually apply the migration. Required to write."
    )
    args = p.parse_args()

    if not args.dry_run and not args.commit:
        logger.error("Must pass one of --dry-run or --commit")
        return 2
    if args.dry_run and args.commit:
        logger.error("Cannot pass both --dry-run and --commit")
        return 2

    counts = run_migration(dry_run=args.dry_run)
    logger.info("Migration summary: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
