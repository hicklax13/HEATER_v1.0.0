"""MS-C1 fix: League Standings page sources total weeks from LeagueConfig
(canonical season_weeks=26), not a hardcoded literal 24.

Mirrors the Lineup Optimizer guard (test_lineup_optimizer_26_weeks.py): the
synthetic-schedule fallback (range(1, _TOTAL_WEEKS+1)) and the magic-number
horizon (_TOTAL_WEEKS - current_week + 1) must use the canonical 26-week
season, not the 24 that drifted in.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "pages" / "6_League_Standings.py"


def test_total_weeks_not_hardcoded_literal():
    """`_TOTAL_WEEKS = 24` (or any bare integer literal) must not appear —
    it should be assigned from LeagueConfig().season_weeks."""
    assert PAGE.exists()
    text = PAGE.read_text(encoding="utf-8")
    bad: list[tuple[int, str]] = []
    pat = re.compile(r"\b_TOTAL_WEEKS\s*=\s*([0-9.]+)\b")
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        m = pat.search(line)
        if m:
            bad.append((lineno, stripped))
    assert not bad, (
        f"MS-C1 regression: _TOTAL_WEEKS should be sourced from "
        f"LeagueConfig().season_weeks, not a hardcoded literal. Offenders: {bad}"
    )


def test_total_weeks_sourced_from_league_config():
    """The page must assign _TOTAL_WEEKS from LeagueConfig().season_weeks."""
    assert PAGE.exists()
    text = PAGE.read_text(encoding="utf-8")
    pat = re.compile(r"_TOTAL_WEEKS\s*=\s*[A-Za-z_][\w.]*\(\)\.season_weeks")
    assert pat.search(text), (
        "MS-C1: _TOTAL_WEEKS must be assigned from LeagueConfig().season_weeks "
        "(mirrors pages/2_Line-up_Optimizer.py's _LC_W().season_weeks pattern)."
    )


def test_total_weeks_resolves_to_26():
    """The canonical season length is 26 weeks (FourzynBurn)."""
    from src.valuation import LeagueConfig

    assert LeagueConfig().season_weeks == 26
