#!/usr/bin/env python3
"""Install git pre-commit hooks for the HEATER project.

Run once after cloning:
    python scripts/install-hooks.py
"""

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOOK_SRC = REPO_ROOT / "scripts" / "pre-commit"
HOOK_DST = REPO_ROOT / ".git" / "hooks" / "pre-commit"


def main():
    if not (REPO_ROOT / ".git").is_dir():
        print("Error: not a git repository.")
        return

    shutil.copy2(HOOK_SRC, HOOK_DST)
    # Make executable on Unix
    HOOK_DST.chmod(0o755)
    print(f"✅ Installed pre-commit hook to {HOOK_DST}")


if __name__ == "__main__":
    main()
