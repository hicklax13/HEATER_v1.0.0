"""Permanent guard against BUG-004 regression.

The MLB Stats API formats innings-pitched as outs notation, not decimals.
Any direct `float(...inningsPitched...)` coercion in production code is a
BUG-004 regression — must use _ip_outs_to_decimal() helper.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_no_direct_float_innings_pitched():
    """Reject `float(...inningsPitched...)` patterns in production code."""
    offending = []
    for d in ("src", "scripts"):
        for p in (REPO_ROOT / d).rglob("*.py"):
            text = p.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                stripped = line.strip()
                # Skip comments
                if stripped.startswith("#"):
                    continue
                # The helper definition itself can mention both terms inside a string
                # docstring — skip the file that defines _ip_outs_to_decimal.
                if "_ip_outs_to_decimal" in line:
                    continue
                if "float(" in line and "inningsPitched" in line:
                    offending.append((str(p), lineno, stripped))
    assert not offending, (
        "BUG-004 regression: direct `float(...inningsPitched...)` parse found. "
        "Use _ip_outs_to_decimal() from src.live_stats instead. Offenders:\n"
        + "\n".join(f"  {p}:{n}: {ln}" for p, n, ln in offending)
    )
