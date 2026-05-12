"""Wave 7 fix: source modules emit "ATH" not "OAK" for Athletics.

Wave 1 D1A-008 established ATH as canonical (matches MLB Stats API +
_PARK_FACTORS_EMERGENCY_2026 + Wave 1's DB migration). Wave 4 fixed the
Streaming-tab UI map. This guards that source modules don't regress
back to emitting "OAK" for current-team values.

Note: `valuation.py` TEAM_CODE_CANONICAL and `fa_recommender.py`
_TEAM_EQUIVALENCES intentionally reference "OAK" as the old-code input
that maps TO canonical "ATH" — those are normalization maps, not
emissions. They are excluded here.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Files where "OAK" as a current-team string literal must NOT appear.
GUARDED_FILES = [
    "src/player_databank.py",
    "src/data_2026.py",
    "src/ecr.py",
    "src/prospect_engine.py",
]


def test_no_oak_team_value_in_source_modules():
    """A bare \"OAK\" string literal in player/team data implies pre-2024
    Athletics naming; canonical 2026 is \"ATH\"."""
    offenders: list[tuple[str, int, str]] = []
    pat = re.compile(r'"OAK"')
    for rel in GUARDED_FILES:
        p = REPO_ROOT / rel
        if not p.exists():
            continue
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pat.search(line):
                offenders.append((rel, lineno, stripped))
    assert not offenders, (
        'Wave 7 regression: source modules still emit "OAK" for Athletics. '
        'Use "ATH" (canonical 2026 per Wave 1 D1A-008). Offenders:\n'
        + "\n".join(f"  {p}:{n}: {ln}" for p, n, ln in offenders)
    )
