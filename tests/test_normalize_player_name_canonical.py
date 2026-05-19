"""D2 consolidation: normalize_player_name() lives in src.valuation only.

Guards against re-introducing per-module name-normalization implementations.
The canonical impl is the SUPERSET of three pre-consolidation impls:
  - src/league_manager.py:206 (_normalize_name): NFD + suffix strip
  - src/live_stats.py:57 (_normalize_name): NFKD + Yahoo (Pitcher) regex
  - src/optimizer/daily_optimizer.py:184 (_normalize_pitcher_name): all of above + punctuation
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.valuation import normalize_player_name


@pytest.mark.parametrize(
    "raw, expected",
    [
        # Accents (NFKD): Iván → Ivan
        ("Iván Rodríguez", "ivan rodriguez"),
        ("José Ramírez", "jose ramirez"),
        # Generational suffixes
        ("Vladimir Guerrero Jr.", "vladimir guerrero"),
        ("Robinson Canó III", "robinson cano"),
        ("Cal Ripken Sr.", "cal ripken"),
        ("Ronald Acuña Jr.", "ronald acuna"),
        # Yahoo parenthetical role markers
        ("Shohei Ohtani (Pitcher)", "shohei ohtani"),
        ("Shohei Ohtani (Batter)", "shohei ohtani"),
        ("Shohei Ohtani (P)", "shohei ohtani"),
        ("Shohei Ohtani (B)", "shohei ohtani"),
        # Punctuation
        ("Mike O'Neill", "mike oneill"),
        ("Hyun-Jin Ryu", "hyunjin ryu"),
        ("A.J. Burnett", "aj burnett"),
        # Edge cases
        ("", ""),
        (None, ""),
        (123, ""),  # non-string input
        ("   Mike   Trout   ", "mike trout"),  # whitespace collapse
    ],
)
def test_canonical_normalization(raw, expected):
    """The single canonical normalize_player_name handles all match dimensions."""
    assert normalize_player_name(raw) == expected


def test_no_duplicate_normalize_name_defs_in_src():
    """Structural guard: no other `def _normalize_name(` or `def _normalize_pitcher_name(`
    in src/. The canonical normalize_player_name in valuation.py is the only impl.

    Exempt: src/valuation.py itself (where the canonical lives).
    """
    src_dir = Path(__file__).resolve().parent.parent / "src"
    pattern = re.compile(r"^def (_normalize_name|_normalize_pitcher_name)\b", re.MULTILINE)
    offenders: list[str] = []
    for f in src_dir.rglob("*.py"):
        if f.name == "valuation.py" and f.parent == src_dir:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            offenders.append(str(f.relative_to(src_dir.parent)))
    assert not offenders, (
        "Duplicate name-normalization impl(s) found. Consolidate into "
        "src.valuation.normalize_player_name (D2 invariant):\n  " + "\n  ".join(offenders)
    )
