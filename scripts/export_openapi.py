"""Dump the FastAPI OpenAPI schema to api/openapi.json. The frontend
(Sub-project A) generates web/lib/data/types.ts from this file, so the
API contract has exactly one source of truth. Run after any contract change."""

from __future__ import annotations

import json
import pathlib
import sys

# Ensure the project root is on sys.path so `api` is importable when the script
# is invoked with `python scripts/export_openapi.py` from any working directory.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api.main import create_app  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "api" / "openapi.json"


def main() -> None:
    schema = create_app().openapi()
    OUT.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
