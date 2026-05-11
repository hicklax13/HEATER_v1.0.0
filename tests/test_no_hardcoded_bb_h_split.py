"""Permanent guard against BUG-012 ROS-path regression.

Hardcoded `0.35`/`0.65` BB:H split in Bayesian rate-stat derivation silently
ignores observed BB:H ratio. Must use observed-share-with-regression toward
0.35 with stabilization=50 baserunners (see bayesian.py:457-474 and :771-787).
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_hardcoded_bb_h_split_in_bayesian():
    """Reject `total_baserunners * 0.35` / `total_br * 0.35` patterns in
    src/bayesian.py production code. The fix should use observed-share."""
    bayesian_path = REPO_ROOT / "src" / "bayesian.py"
    text = bayesian_path.read_text(encoding="utf-8")
    # Catch both `total_baserunners * 0.35` and `total_br * 0.35` patterns
    pat = re.compile(r"total_(?:baserunners|br)\s*\*\s*0\.(?:35|65)\b")
    offenders: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.strip().startswith("#"):
            continue
        if pat.search(line):
            offenders.append((lineno, line.strip()))
    assert not offenders, (
        "BUG-012 regression: hardcoded `total_baserunners * 0.35` / `* 0.65` "
        "BB:H split found. Must use observed-share-with-regression toward 0.35. "
        f"Offenders in src/bayesian.py: {offenders}"
    )
