"""BUG-018 fix: Lineup Optimizer scales counting stats by 26 (canonical), not 24."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE = REPO_ROOT / "pages" / "2_Line-up_Optimizer.py"


def test_lineup_optimizer_uses_26_weeks_for_weekly_proj():
    """The Projected Weekly Category Totals scaler should be 26 (canonical
    per CLAUDE.md "Counting stats divided by 26 weeks"), not 24."""
    assert PAGE.exists()
    text = PAGE.read_text(encoding="utf-8")
    bad: list[tuple[int, str]] = []
    pat = re.compile(r"\bWEEKS_IN_SEASON\s*=\s*([0-9.]+)\b")
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
        f"BUG-018 regression: WEEKS_IN_SEASON should be 26.0 "
        f"(per CLAUDE.md and src/optimizer/backtest_runner.py). Offenders: {bad}"
    )
