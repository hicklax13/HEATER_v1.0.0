#!/usr/bin/env python3
"""Phase 0 preflight checks for HEATER deep-audit completion session.

Exit codes:
    0  all checks pass
    1  hard failure (fix before proceeding)
    2  passed with warnings (review but may proceed)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

OK_, WARN_, FAIL_ = "[OK]  ", "[WARN]", "[FAIL]"
errors: list[str] = []
warnings: list[str] = []


def check(name: str, ok: bool, *, severity: str = "error", detail: str = "") -> bool:
    marker = OK_ if ok else (FAIL_ if severity == "error" else WARN_)
    suffix = f" — {detail}" if detail and not ok else ""
    print(f"{marker} {name}{suffix}")
    if not ok:
        (errors if severity == "error" else warnings).append(name)
    return ok


print("== file-system access ==")
probe = Path(".access_probe")
try:
    probe.write_text("ok")
    assert probe.read_text() == "ok"
    probe.unlink()
    check("read/write/delete in repo root", True)
except Exception as e:
    check("read/write/delete in repo root", False, detail=str(e))

print("\n== python + deps ==")
v = sys.version_info
check(f"python {v.major}.{v.minor}", v >= (3, 12), detail="need 3.12+")

r = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
check("pip check (no broken deps)", r.returncode == 0, detail=r.stdout.strip()[:200])

for mod in (
    "pandas",
    "numpy",
    "scipy",
    "plotly",
    "streamlit",
    "statsapi",
    "pybaseball",
    "pulp",
    "arviz",
    "yfpy",
    "streamlit_oauth",
):
    try:
        __import__(mod)
        check(f"import {mod}", True)
    except Exception as e:
        check(f"import {mod}", False, detail=str(e))

for mod in ("pymc", "xgboost", "weasyprint"):
    try:
        __import__(mod)
        check(f"import {mod} (optional)", True)
    except Exception:
        check(f"import {mod} (optional)", False, severity="warn", detail="not installed")

print("\n== secrets / tokens / env ==")
try:
    from dotenv import load_dotenv

    load_dotenv()
    check(".env loaded (if present)", True)
except ImportError:
    check("python-dotenv import", False, severity="warn")

for k in ("YAHOO_CLIENT_ID", "YAHOO_CLIENT_SECRET", "YAHOO_LEAGUE_KEY"):
    check(f"env {k}", bool(os.getenv(k)), severity="warn", detail="not set (OK if Yahoo disconnected)")

token = Path("data/yahoo_token.json")
if token.exists():
    try:
        t = json.loads(token.read_text())
        required = {"access_token", "refresh_token", "consumer_key", "consumer_secret", "token_time", "expires_in"}
        missing = required - set(t.keys())
        check(
            f"yahoo_token.json shape ({len(t)} keys)",
            not missing,
            severity="warn",
            detail=f"missing: {sorted(missing)}" if missing else "",
        )
        try:
            tt = float(t.get("token_time", 0))
            check(
                "yahoo_token.json token_time parseable",
                tt > 0,
                severity="warn",
                detail="invalid token_time (yfpy will auto-refresh on first call)",
            )
        except (TypeError, ValueError):
            check("yahoo_token.json token_time parseable", False, severity="warn")
    except Exception as e:
        check("yahoo_token.json parse", False, detail=str(e))
else:
    check("yahoo_token.json present", False, severity="warn", detail="Yahoo disconnected (OK for code-only changes)")

print("\n== github access ==")
if shutil.which("gh"):
    r = subprocess.run(["gh", "auth", "status"], capture_output=True, text=True)
    check("gh auth status", r.returncode == 0, detail=(r.stderr or r.stdout).strip()[:200])
    r = subprocess.run(
        ["gh", "api", "repos/hicklax13/HEATER_v1.0.0", "--jq", ".permissions"], capture_output=True, text=True
    )
    check("gh push permission on hicklax13/HEATER_v1.0.0", '"push":true' in r.stdout, detail=r.stdout.strip()[:200])
    r = subprocess.run(["gh", "api", "user", "--jq", ".login"], capture_output=True, text=True)
    check(f"gh identity = {r.stdout.strip()}", r.returncode == 0)
else:
    check("gh CLI on PATH", False)

print("\n== git state ==")
hook = Path(".git/hooks/pre-commit")
check(
    "pre-commit hook installed",
    hook.exists() and os.access(hook, os.X_OK),
    severity="warn",
    detail="run: python scripts/install-hooks.py",
)

r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
check("git working tree clean", r.stdout.strip() == "", detail=r.stdout.strip()[:200])

r = subprocess.run(["git", "worktree", "list"], capture_output=True, text=True)
wt_count = len([ln for ln in r.stdout.strip().splitlines() if ln])
if wt_count > 1:
    check(
        f"worktree count = {wt_count}",
        True,
        severity="warn",
        detail=f"Phase 99 will prune {wt_count - 1} non-main worktrees",
    )
else:
    check(f"worktree count = {wt_count}", True)

print("\n== summary ==")
print(f"errors: {len(errors)} | warnings: {len(warnings)}")
if errors:
    print("\nHARD FAIL — fix the above before proceeding:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
if warnings:
    print("\nWARNINGS — review before proceeding:")
    for w in warnings:
        print(f"  - {w}")
    sys.exit(2)
print("All checks pass.")
sys.exit(0)
