#!/usr/bin/env python3
"""Recalibrate `repl_avg / repl_obp / repl_era / repl_whip` in
``src/optimizer/constants_registry.py`` from FourzynBurn standings.

These four replacement-level baselines drive every rate-stat DCV in the
daily optimizer (see `apply_stud_floor` + `build_daily_dcv_table` rate
branch in `src/optimizer/daily_optimizer.py`). A 0.005 absolute shift in
``repl_avg`` moves per-player AVG DCV by 16-33% (audit DCV-A1-001), so
the values must reflect the league's actual replacement tier — not a
generic 12-team H2H rule of thumb.

Methodology (OQ-1, 2026-05-15 resolution):
- Read the 12-team-season ``league_standings`` long-format table.
- For each rate cat (AVG, OBP, ERA, WHIP), take the rank=12 team's
  category total. That row represents the "replacement tier" — the team
  whose rate stats you'd inherit by punting that category.
- Print the recalibrated values + the suggested registry diff.
- ``--apply`` (default off) edits ``src/optimizer/constants_registry.py``
  in place; without it the script is a dry-run that just prints.

Run after each FourzynBurn season ends (week 26 championship) and before
the next year's draft. The registry citation lines reference this script
explicitly so future maintainers can refresh annually.

Usage:
    python scripts/calibrate_repl_baselines.py            # dry-run
    python scripts/calibrate_repl_baselines.py --apply    # edit registry
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.database import get_connection  # noqa: E402

REGISTRY_PATH = REPO_ROOT / "src" / "optimizer" / "constants_registry.py"

# Mapping: registry key → (long-format category column value, dp for display)
_TARGETS: dict[str, tuple[str, int]] = {
    "repl_avg": ("AVG", 3),
    "repl_obp": ("OBP", 3),
    "repl_era": ("ERA", 2),
    "repl_whip": ("WHIP", 2),
}


def fetch_bottom_team_rates() -> dict[str, float]:
    """Read league_standings and return the rank=12 team's rate values."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT category, total FROM league_standings WHERE rank = 12 AND category IN ('AVG','OBP','ERA','WHIP')"
        ).fetchall()
    finally:
        conn.close()

    out: dict[str, float] = {}
    for cat, total in rows:
        try:
            out[str(cat).upper()] = float(total)
        except (TypeError, ValueError):
            continue
    return out


def patch_registry(new_values: dict[str, float]) -> int:
    """Rewrite the four `value=` lines in CONSTANTS_REGISTRY in place.
    Returns the number of entries successfully patched."""
    text = REGISTRY_PATH.read_text(encoding="utf-8")
    patched = 0
    for key, (cat, dp) in _TARGETS.items():
        new_v = new_values.get(cat)
        if new_v is None:
            continue
        # Locate the entry block and replace its `value=...` line.
        block_re = re.compile(
            rf'("{key}":\s*ConstantEntry\(\s*\n\s*value=)([0-9.]+)(,)',
            re.MULTILINE,
        )
        formatted = f"{new_v:.{dp}f}"
        new_text, n = block_re.subn(rf"\g<1>{formatted}\g<3>", text)
        if n == 1:
            text = new_text
            patched += 1
        else:
            print(f"  warn: failed to patch {key} (regex matched {n} times)")
    REGISTRY_PATH.write_text(text, encoding="utf-8")
    return patched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Edit src/optimizer/constants_registry.py in place; default is dry-run",
    )
    args = ap.parse_args()

    rates = fetch_bottom_team_rates()
    if not rates:
        print(
            "ERROR: league_standings table is empty or has no rank=12 entries\n"
            "       for AVG/OBP/ERA/WHIP. Run a full season first.",
            file=sys.stderr,
        )
        return 2

    print("Replacement-level baselines from FourzynBurn standings (rank=12):")
    print(f"  AVG : {rates.get('AVG', float('nan')):.3f}")
    print(f"  OBP : {rates.get('OBP', float('nan')):.3f}")
    print(f"  ERA : {rates.get('ERA', float('nan')):.2f}")
    print(f"  WHIP: {rates.get('WHIP', float('nan')):.2f}")
    print()

    if not args.apply:
        print("Dry-run; pass --apply to update src/optimizer/constants_registry.py.")
        return 0

    n = patch_registry(rates)
    print(f"Patched {n}/{len(_TARGETS)} entries in {REGISTRY_PATH.relative_to(REPO_ROOT)}")
    return 0 if n == len(_TARGETS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
