"""Load, validate, and summarize the launch evidence registry.

The registry (docs/launch/evidence_registry.yaml) is the single machine-readable
source of truth for "are we at 100/100?". Every requirement/gate is one row.
"""

from __future__ import annotations

import argparse
import pathlib

import yaml

VALID_STATUSES = {
    "planned",
    "in_progress",
    "passing",
    "failing",
    "deferred",
    "waived",
}

_REQUIRED_FIELDS = (
    "id",
    "category",
    "phase",
    "description",
    "status",
    "subsystem",
    "verify",
    "metric",
    "evidence",
    "external_review",
    "blocking_ring",
    "last_verified",
    "score_contribution",
)


def load_registry(path: pathlib.Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def validate(registry: dict) -> list[str]:
    """Return a list of human-readable validation errors (empty == valid)."""
    errors: list[str] = []
    rows = registry.get("requirements", [])
    seen: set[str] = set()
    for i, row in enumerate(rows):
        rid = row.get("id", f"<row {i}>")
        for field in _REQUIRED_FIELDS:
            if field not in row or row[field] in (None, ""):
                errors.append(f"{rid}: missing required field '{field}'")
        status = row.get("status")
        if status is not None and status not in VALID_STATUSES:
            errors.append(f"{rid}: invalid status '{status}' (must be one of {sorted(VALID_STATUSES)})")
        if rid in seen:
            errors.append(f"{rid}: duplicate id")
        seen.add(rid)
    return errors


def summarize(registry: dict) -> dict[str, int]:
    counts = {s: 0 for s in VALID_STATUSES}
    for row in registry.get("requirements", []):
        status = row.get("status")
        if status in counts:
            counts[status] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="HEATER launch evidence registry")
    parser.add_argument("--summary", action="store_true", help="print a status rollup")
    parser.add_argument(
        "--path",
        default=str(pathlib.Path(__file__).resolve().parents[2] / "docs" / "launch" / "evidence_registry.yaml"),
    )
    args = parser.parse_args()
    registry = load_registry(pathlib.Path(args.path))
    errors = validate(registry)
    if errors:
        print("INVALID registry:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    if args.summary:
        total = len(registry.get("requirements", []))
        print(f"evidence registry: {total} requirements")
        for status, n in sorted(summarize(registry).items()):
            print(f"  {status:12} {n}")


if __name__ == "__main__":
    main()
