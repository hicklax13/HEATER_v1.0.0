"""Guard: no leftover merge conflict markers in source files.

A leftover '<<<<<<<', '=======', or '>>>>>>>' marker would cause runtime
ImportError. This caught us during the SF-15..SF-28 cleanup; this test
prevents recurrence.
"""

import re
from pathlib import Path

# Files to scan
SCAN_GLOBS = [
    "src/**/*.py",
    "pages/**/*.py",
    "tests/**/*.py",
    "app.py",
    "scripts/**/*.py",
]

CONFLICT_PATTERNS = [
    re.compile(r"^<{7} ", re.MULTILINE),  # <<<<<<<<<
    re.compile(r"^={7}$", re.MULTILINE),  # =======
    re.compile(r"^>{7} ", re.MULTILINE),  # >>>>>>>>>
]


def _iter_source_files():
    repo_root = Path(__file__).resolve().parent.parent
    for pattern in SCAN_GLOBS:
        for path in repo_root.glob(pattern):
            if path.is_file():
                yield path


def test_no_merge_conflict_markers():
    """Every source file must be free of leftover merge conflict markers."""
    repo_root = Path(__file__).resolve().parent.parent
    bad: list[str] = []
    for path in _iter_source_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        for pattern in CONFLICT_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text[: match.start()].count("\n") + 1
                bad.append(f"{path.relative_to(repo_root)}:{line_no}")
    assert bad == [], "Merge conflict markers found:\n  " + "\n  ".join(bad) + "\n\nResolve them before committing."
