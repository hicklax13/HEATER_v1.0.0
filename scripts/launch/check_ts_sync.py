"""Fail if web/src/lib/api/generated.ts is not regenerable from api/openapi.json
without drift. Used locally; CI runs the same check via `pnpm gen:api` +
`git diff --exit-code` (see .github/workflows/ci.yml)."""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WEB = _ROOT / "web"
_GENERATED = _WEB / "src" / "lib" / "api" / "generated.ts"


def main() -> int:
    before = _GENERATED.read_text(encoding="utf-8") if _GENERATED.exists() else ""
    # Resolve pnpm explicitly: on Windows it is a .cmd/.ps1 shim that bare
    # subprocess cannot find (WinError 2). shutil.which finds the real path
    # cross-platform; absent pnpm, degrade cleanly (CI is the enforcing surface).
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        print("pnpm not found on PATH; skipping (CI enforces this check).", file=sys.stderr)
        return 2
    try:
        subprocess.run([pnpm, "gen:api"], cwd=_WEB, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"could not run `pnpm gen:api`: {exc}", file=sys.stderr)
        return 2
    after = _GENERATED.read_text(encoding="utf-8")
    if before != after:
        print(
            "generated.ts drifted from api/openapi.json. Run `cd web && pnpm gen:api` and commit the result.",
            file=sys.stderr,
        )
        return 1
    print("generated.ts is in sync with api/openapi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
