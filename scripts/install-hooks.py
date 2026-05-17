#!/usr/bin/env python3
"""Install git hooks for the HEATER project.

Run once after cloning:
    python scripts/install-hooks.py

Installs both:
- ``pre-commit`` — ruff format + lint check on staged Python files (fast)
- ``pre-push``   — structural-invariant test suite (~2s, runs the
                   test_no_*, test_pages_*, test_wave* families that
                   enforce CLAUDE.md's architectural guarantees)
"""

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOKS_DIR = REPO_ROOT / ".git" / "hooks"

# (source_basename, destination_basename) for each hook to install.
_HOOKS = [
    ("pre-commit", "pre-commit"),
    ("pre-push", "pre-push"),
]


def main():
    if not (REPO_ROOT / ".git").is_dir():
        print("Error: not a git repository.")
        return

    for src_name, dst_name in _HOOKS:
        src = REPO_ROOT / "scripts" / src_name
        dst = HOOKS_DIR / dst_name
        if not src.exists():
            print(f"warn: {src} not found, skipping {dst_name}")
            continue
        shutil.copy2(src, dst)
        # Make executable on Unix (no-op on Windows; Git Bash respects the
        # script extension)
        dst.chmod(0o755)
        print(f"✅ Installed {dst_name} hook to {dst}")


if __name__ == "__main__":
    main()
