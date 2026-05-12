"""BUG-010 fix: optimizer modules must not define _LC singletons at module level."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GUARDED_FILES = [
    "src/optimizer/pipeline.py",
    "src/optimizer/projections.py",
    "src/optimizer/scenario_generator.py",
    "src/optimizer/h2h_engine.py",
    "src/optimizer/sgp_theory.py",
    "src/optimizer/advanced_lp.py",
    "src/optimizer/dual_objective.py",
    "src/optimizer/shared_data_layer.py",
]


def test_no_lc_singletons_in_optimizer_modules():
    """Each guarded optimizer module must not have a top-level `_LC = ...`
    assignment (the SF-21 anti-pattern). Use a function-local instance
    or _LC_ONCE + del pattern instead."""
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
        f"BUG-010 regression: _LC = ... module-level singleton found in optimizer. "
        f"Use _LC_ONCE + del pattern (see Wave 6 plan). Offenders: {offenders}"
    )
