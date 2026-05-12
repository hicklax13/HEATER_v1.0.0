"""BUG-021 fix: playoff_sim uses canonical season_weeks=26 and _PLAYOFF_SPOTS=4."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PSIM = REPO_ROOT / "src" / "playoff_sim.py"


def test_no_hardcoded_22_season_weeks():
    """season_weeks should be 26 (CLAUDE.md canonical: 26-week MLB regular
    season), not 22."""
    assert PSIM.exists()
    text = PSIM.read_text(encoding="utf-8")
    bad: list[tuple[int, str]] = []
    pat = re.compile(r"\bseason_weeks\s*=\s*([0-9.]+)\b")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        m = pat.search(line)
        if m:
            val = float(m.group(1))
            if abs(val - 26.0) > 0.01:
                bad.append((lineno, stripped))
    assert not bad, (
        f"BUG-021 regression: season_weeks should be 26 (CLAUDE.md canonical), not other value. Offenders: {bad}"
    )


def test_no_hardcoded_6_playoff_spots():
    """_PLAYOFF_SPOTS should be 4 (FourzynBurn league: top-4 playoff per
    CLAUDE.md), not 6."""
    assert PSIM.exists()
    text = PSIM.read_text(encoding="utf-8")
    bad: list[tuple[int, str]] = []
    pat = re.compile(r"\b_PLAYOFF_SPOTS\s*=\s*([0-9]+)\b")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        m = pat.search(line)
        if m:
            val = int(m.group(1))
            if val != 4:
                bad.append((lineno, stripped))
    assert not bad, (
        f"BUG-021 regression: _PLAYOFF_SPOTS should be 4 (FourzynBurn top-4 playoff per CLAUDE.md). Offenders: {bad}"
    )
