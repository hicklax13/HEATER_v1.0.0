"""Capture an authoritative, reproducible baseline of the current product.

Pure helpers (testable) + a report writer that records git SHA, tool versions,
the OpenAPI route inventory, and an operation count into a deterministic markdown
file under docs/launch/baseline/.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

_HTTP_METHODS = {"get", "post", "put", "patch", "delete"}
_ROOT = pathlib.Path(__file__).resolve().parents[2]


def openapi_operation_count(openapi: dict) -> int:
    """Count HTTP operations (path+method pairs) in an OpenAPI document."""
    return sum(1 for ops in openapi.get("paths", {}).values() for method in ops if method.lower() in _HTTP_METHODS)


def route_inventory(openapi: dict) -> list[dict]:
    """Return a stable, sorted list of {method, path, operation_id} rows."""
    rows: list[dict] = []
    for path, ops in openapi.get("paths", {}).items():
        for method, op in ops.items():
            if method.lower() not in _HTTP_METHODS:
                continue
            rows.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "operation_id": (op or {}).get("operationId", ""),
                }
            )
    # Sort by path then method for deterministic output.
    rows.sort(key=lambda r: (r["path"], r["method"]))
    return rows


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _tool_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for mod in ("fastapi", "httpx", "pydantic"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 - best-effort version capture
            versions[mod] = "unavailable"
    return versions


def build_report(openapi: dict, *, sha: str, versions: dict[str, str]) -> str:
    """Render the baseline markdown from already-collected facts (pure)."""
    inv = route_inventory(openapi)
    lines = [
        "# HEATER baseline report",
        "",
        f"- git SHA: `{sha}`",
        f"- OpenAPI operations: {openapi_operation_count(openapi)}",
        "- tool versions:",
        *[f"  - {k}: {v}" for k, v in sorted(versions.items())],
        "",
        "## Route inventory",
        "",
        "| Method | Path | operationId |",
        "|---|---|---|",
        *[f"| {r['method']} | `{r['path']}` | `{r['operation_id']}` |" for r in inv],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    openapi = json.loads((_ROOT / "api" / "openapi.json").read_text(encoding="utf-8"))
    report = build_report(openapi, sha=_git_sha(), versions=_tool_versions())
    out_dir = _ROOT / "docs" / "launch" / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Single overwritten file (the SHA lives in the content) so re-runs across
    # commits refresh one artifact rather than accumulating per-SHA files.
    out = out_dir / "baseline.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
