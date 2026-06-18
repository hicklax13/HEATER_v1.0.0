"""Guard: the committed api/openapi.json matches the live schema. If this
fails, a contract changed without regenerating — run scripts/export_openapi.py
and commit, then tell the frontend to regenerate types.ts."""

import json
import pathlib

from api.main import create_app

SCHEMA = pathlib.Path(__file__).resolve().parents[2] / "api" / "openapi.json"


def test_openapi_snapshot_is_current():
    live = json.loads(json.dumps(create_app().openapi(), sort_keys=True))
    committed = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert committed == live, "api/openapi.json is stale — run `python scripts/export_openapi.py` and commit"
