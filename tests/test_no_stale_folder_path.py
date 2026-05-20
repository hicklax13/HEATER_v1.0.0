"""Section 8 invariant: no stale HEATER_v1.0.0 LOCAL folder-path references.

The local folder was renamed from HEATER_v1.0.0 → HEATER_v1.0.1 in PR #46
(relocation from OneDrive). The GitHub repo NAME stays at HEATER_v1.0.0.

This guard distinguishes:
  - LOCAL PATHS (HEATER_v1.0.0 used as a filesystem directory name)
    → must be updated to HEATER_v1.0.1
  - GITHUB URLs (github.com/hicklax13/HEATER_v1.0.0 or hicklax13/HEATER_v1.0.0)
    → must NOT be touched (repo name unchanged)
  - CLAUDE.md OneDrive incompatibility WARNING block
    → intentional historical reference (documents what NOT to do)
  - Documentation in docs/archive/, docs/superpowers/plans/, docs/superpowers/specs/
    → session artifacts / historical docs (intentional context)

Files SCANNED:
  src/, pages/, scripts/, .github/, .streamlit/, top-level docs/*.md
  (no subdirs), CLAUDE.md, README.md, AGENTS.md, GEMINI.md

Files NOT scanned:
  .claude/, .venv/, tests/, docs/archive/, docs/superpowers/,
  __pycache__/, build/, dist/, *.pyc, *.egg-info/
"""

from __future__ import annotations

import re
from pathlib import Path

# Patterns that are ALWAYS allowed (GitHub URL forms).
GITHUB_URL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"github\.com/hicklax13/HEATER_v1\.0\.0"),
    re.compile(r"hicklax13/HEATER_v1\.0\.0(?:\.git)?"),
    re.compile(r"repos/hicklax13/HEATER_v1\.0\.0"),
    re.compile(r"GitHub repo NAME (?:remains|is) `?HEATER_v1\.0\.0`?"),
)

# Marker for CLAUDE.md's intentional OneDrive incompatibility WARNING block.
# The block starts at the line containing this phrase and ends at the next
# top-level (## or #) markdown header or two blank lines.
ONEDRIVE_WARN_MARKER = "OneDrive's Cloud Files API"


def _is_in_onedrive_warn_block(lines: list[str], lineno: int) -> bool:
    """Return True if line `lineno` (1-indexed) is inside the CLAUDE.md OneDrive warning block.

    Block heuristic: scan backwards from the line; if we hit ONEDRIVE_WARN_MARKER
    before a top-level header (^## or ^#), we're inside the block.
    """
    for i in range(lineno - 1, max(0, lineno - 30), -1):
        line = lines[i]
        if ONEDRIVE_WARN_MARKER in line:
            return True
        if line.startswith("## ") or line.startswith("# "):
            return False
    return False


def _is_allowed_line(line: str) -> bool:
    """Return True if the line's HEATER_v1.0.0 reference is allowed (GitHub URL)."""
    return any(p.search(line) for p in GITHUB_URL_PATTERNS)


def _collect_targets(root: Path) -> list[Path]:
    """Files this guard scans."""
    targets: list[Path] = []
    for sub in ("src", "pages", "scripts", ".github", ".streamlit"):
        d = root / sub
        if d.exists():
            targets.extend(p for p in d.rglob("*") if p.is_file())
    # Top-level docs only (NOT recursive into docs/archive, docs/superpowers).
    docs = root / "docs"
    if docs.exists():
        for p in docs.iterdir():
            if p.is_file():
                targets.append(p)
    # Root-level docs files.
    for name in ("CLAUDE.md", "README.md", "AGENTS.md", "GEMINI.md"):
        p = root / name
        if p.exists() and p.is_file():
            targets.append(p)
    return targets


def test_no_stale_local_folder_path():
    """No `HEATER_v1.0.0` LOCAL path references in scanned files.

    The GitHub repo NAME is allowed (full URL or hicklax13/ owner-name forms);
    the CLAUDE.md OneDrive incompatibility WARNING block is allowed.
    """
    root = Path(__file__).resolve().parent.parent
    targets = _collect_targets(root)
    offenders: list[str] = []
    skip_suffixes = {".png", ".jpg", ".jpeg", ".gif", ".db", ".sqlite", ".pyc", ".lock"}

    for f in targets:
        if f.suffix.lower() in skip_suffixes:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if "HEATER_v1.0.0" not in text:
            continue
        lines = text.splitlines()
        is_claude_md = f.name == "CLAUDE.md"
        for lineno, line in enumerate(lines, start=1):
            if "HEATER_v1.0.0" not in line:
                continue
            if _is_allowed_line(line):
                continue
            if is_claude_md and _is_in_onedrive_warn_block(lines, lineno):
                continue
            offenders.append(f"{f.relative_to(root)}:{lineno}: {line.strip()[:140]}")

    assert not offenders, (
        "Stale `HEATER_v1.0.0` LOCAL folder-path reference(s) found. The local "
        "folder is `HEATER_v1.0.1`; only the GitHub repo NAME stays at v1.0.0.\n  " + "\n  ".join(offenders)
    )
