"""BUG-010 fix: strategy and UI modules must not define _LC singletons."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GUARDED_FILES = [
    "src/standings_engine.py",
    "src/standings_projection.py",
    "src/war_room.py",
    "src/leaders.py",
    "src/player_databank.py",
    "src/player_card.py",
    "src/lineup_optimizer.py",
    "src/ui_shared.py",
]


def test_no_lc_singletons_in_strategy_modules():
    offenders: list[tuple[str, int, str]] = []
    pat = re.compile(r"^_LC\s*=\s*", re.MULTILINE)
    for rel in GUARDED_FILES:
        p = REPO_ROOT / rel
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        text_no_comments = re.sub(r"#.*$", "", text, flags=re.MULTILINE)
        for m in pat.finditer(text_no_comments):
            lineno = text_no_comments[: m.start()].count("\n") + 1
            line_str = text_no_comments.splitlines()[lineno - 1].strip()
            offenders.append((rel, lineno, line_str))
    assert not offenders, (
        f"BUG-010 regression: _LC = ... module-level singleton found in strategy/UI. "
        f"Use _LC_ONCE + del pattern. Offenders: {offenders}"
    )
