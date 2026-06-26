# Phase 0a — Rebaseline, Evidence Registry & Structural Guards — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish one authoritative, reproducible baseline of the current product and a machine-readable evidence registry that turns the 100/100 launch standards into trackable gates — plus the structural guards that can be enforced today.

**Architecture:** Pure additive tooling and tests. No application behavior changes. A `scripts/launch/` package captures current state into committed reports; a YAML `evidence_registry` is the single source of truth for "are we at 100/100?"; new `tests/launch/` guards fail when specific drift is introduced. The one behavioral fix is regenerating the stale committed OpenAPI snapshot so the contract gate is green.

**Tech Stack:** Python 3.12 (CI) / 3.14 (local), pytest, PyYAML, FastAPI (read-only here), `scripts/export_openapi.py`, `openapi-typescript` (via `pnpm gen:api`).

**Master spec:** `docs/superpowers/specs/2026-06-26-heater-public-commercial-launch-program-design.md` (Phase 0).

**Scope note:** This is plan **0a** of Phase 0. Plan **0b** (separate) builds the API contract foundation (error envelope, correlation-ID middleware, OpenAPI bearer-security scheme, versioning policy, idempotency framework, async-job contract). Guards that reference concepts not yet in the codebase (tenant scoping, recommendation evidence, calibrated-model provenance, frontend no-mock-fallback) are **registered as `planned`** here and implemented in the phase that introduces the concept — they are deliberately not faked in this plan.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `api/openapi.json` | Committed OpenAPI contract snapshot | Modify (regenerate) |
| `web/src/lib/api/generated.ts` | Generated TS types from the snapshot | Modify (regenerate) |
| `docs/launch/README.md` | Index of the launch program (links spec, registry, baselines) | Create |
| `scripts/launch/__init__.py` | Marks the launch tooling package | Create |
| `scripts/launch/freeze_baseline.py` | Capture current state → committed markdown report | Create |
| `scripts/launch/evidence_registry.py` | Load + validate + summarize the evidence registry | Create |
| `docs/launch/evidence_registry.yaml` | The machine-readable requirement/gate registry | Create |
| `docs/launch/baseline/` | Timestamped baseline reports | Create (dir) |
| `tests/launch/__init__.py` | Marks the launch test package | Create |
| `tests/launch/test_freeze_baseline.py` | Unit tests for freeze helpers | Create |
| `tests/launch/test_evidence_registry.py` | Validates the registry + helper logic | Create |
| `tests/launch/test_guard_routers_mounted.py` | Guard: every `api/routers/*.py` is mounted | Create |
| `.github/workflows/ci.yml` | Add the OpenAPI↔TS sync check step | Modify |
| `scripts/launch/check_ts_sync.py` | Convenience: regenerate TS + report drift | Create |

---

## Task 1: Fix the stale OpenAPI snapshot

**Files:**
- Verify: `requirements.txt` (fastapi/httpx pins) vs installed
- Modify: `api/openapi.json`, `web/src/lib/api/generated.ts`
- Test: `tests/api/test_openapi_contract.py` (existing)

- [ ] **Step 1: Confirm the test fails and confirm the installed versions match the pins**

Run:
```bash
python -m pytest tests/api/test_openapi_contract.py -q
python -c "import fastapi, httpx; print('fastapi', fastapi.__version__); print('httpx', httpx.__version__)"
grep -E "^(fastapi|httpx)==" requirements.txt
```
Expected: the test FAILS (snapshot stale); `fastapi` prints `0.137.1` and `httpx` prints `0.28.1`, matching `requirements.txt`. **If the installed versions do not match the pins, stop** — the drift is an environment problem, not a snapshot problem. Run `pip install -r requirements.txt` to align, then re-check before regenerating (regenerating under the wrong FastAPI version would bake a non-canonical schema into the snapshot).

- [ ] **Step 2: Regenerate the committed snapshot and the TS types**

Run:
```bash
python scripts/export_openapi.py
cd web && pnpm gen:api && cd ..
```
Expected: `api/openapi.json` and `web/src/lib/api/generated.ts` are rewritten.

- [ ] **Step 3: Verify the contract test passes**

Run: `python -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add api/openapi.json web/src/lib/api/generated.ts
git commit -m "fix(api): regenerate stale OpenAPI snapshot + frontend types (Phase 0a)"
```

---

## Task 2: Program scaffolding

**Files:**
- Create: `scripts/launch/__init__.py`, `tests/launch/__init__.py`, `docs/launch/README.md`
- Create (dir): `docs/launch/baseline/` (with a `.gitkeep`)

- [ ] **Step 1: Create the package markers and the baseline dir keeper**

`scripts/launch/__init__.py`:
```python
"""Tooling for the HEATER public commercial launch program (see
docs/superpowers/specs/2026-06-26-heater-public-commercial-launch-program-design.md)."""
```

`tests/launch/__init__.py`:
```python
```
(empty file)

`docs/launch/baseline/.gitkeep`:
```
```
(empty file)

- [ ] **Step 2: Write the launch index**

`docs/launch/README.md`:
```markdown
# HEATER Public Commercial Launch — operating index

- **Master spec:** [`../superpowers/specs/2026-06-26-heater-public-commercial-launch-program-design.md`](../superpowers/specs/2026-06-26-heater-public-commercial-launch-program-design.md)
- **Evidence registry (source of truth for the score):** [`evidence_registry.yaml`](evidence_registry.yaml)
- **Baseline reports:** [`baseline/`](baseline/)
- **Phase plans:** `../superpowers/plans/2026-06-*-heater-launch-phase*.md`

## How to read the registry

Each requirement row has a `status`: `planned` (not started), `in_progress`,
`passing` (its `verify` command is green), `failing`, `deferred` (intentionally
later, e.g. needs a concept from a later phase), or `waived`. Run
`python -m scripts.launch.evidence_registry --summary` for a status rollup.

## How to refresh the baseline

`python -m scripts.launch.freeze_baseline` writes a timestamped report into `baseline/`.
```

- [ ] **Step 3: Commit**

```bash
git add scripts/launch/__init__.py tests/launch/__init__.py docs/launch/README.md docs/launch/baseline/.gitkeep
git commit -m "chore(launch): scaffold launch program tooling + docs/launch index (Phase 0a)"
```

---

## Task 3: Baseline freeze tool — pure helpers (TDD)

**Files:**
- Create: `scripts/launch/freeze_baseline.py`
- Test: `tests/launch/test_freeze_baseline.py`

- [ ] **Step 1: Write the failing tests**

`tests/launch/test_freeze_baseline.py`:
```python
from scripts.launch.freeze_baseline import (
    openapi_operation_count,
    route_inventory,
)

_FAKE_OPENAPI = {
    "paths": {
        "/healthz": {"get": {"operationId": "healthz_healthz_get"}},
        "/api/standings": {"get": {"operationId": "get_standings"}},
        "/api/lineup/set": {
            "post": {"operationId": "set_lineup"},
            "parameters": [],  # not an HTTP method — must be ignored
        },
    }
}


def test_openapi_operation_count_counts_only_http_methods():
    assert openapi_operation_count(_FAKE_OPENAPI) == 3


def test_route_inventory_is_sorted_and_typed():
    rows = route_inventory(_FAKE_OPENAPI)
    assert rows == [
        {"method": "GET", "path": "/api/standings", "operation_id": "get_standings"},
        {"method": "POST", "path": "/api/lineup/set", "operation_id": "set_lineup"},
        {"method": "GET", "path": "/healthz", "operation_id": "healthz_healthz_get"},
    ]


def test_route_inventory_ignores_non_method_keys():
    rows = route_inventory(_FAKE_OPENAPI)
    assert all(r["method"] in {"GET", "POST", "PUT", "PATCH", "DELETE"} for r in rows)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/launch/test_freeze_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (functions not defined).

- [ ] **Step 3: Write the minimal implementation**

`scripts/launch/freeze_baseline.py`:
```python
"""Capture an authoritative, reproducible baseline of the current product.

Pure helpers (testable) + a report writer that records git SHA, tool versions,
the OpenAPI route inventory, test counts, and the DB table list into a
timestamped markdown file under docs/launch/baseline/.
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
    return sum(
        1
        for ops in openapi.get("paths", {}).values()
        for method in ops
        if method.lower() in _HTTP_METHODS
    )


def route_inventory(openapi: dict) -> list[dict]:
    """Return a stable, sorted list of {method, path, operation_id} rows."""
    rows: list[dict] = []
    for path, ops in sorted(openapi.get("paths", {}).items()):
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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip()
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
    # Deterministic name from the SHA so re-runs on the same commit overwrite,
    # not accumulate. (No wall-clock — keeps the artifact reproducible.)
    out = out_dir / f"baseline-{_git_sha()[:12]}.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/launch/test_freeze_baseline.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/launch/freeze_baseline.py tests/launch/test_freeze_baseline.py
git commit -m "feat(launch): baseline freeze tool with tested pure helpers (Phase 0a)"
```

---

## Task 4: Generate and commit the first baseline report

**Files:**
- Create: `docs/launch/baseline/baseline-<sha>.md` (generated)

- [ ] **Step 1: Run the freeze tool**

Run: `python -m scripts.launch.freeze_baseline`
Expected: prints `wrote docs/launch/baseline/baseline-<sha>.md`.

- [ ] **Step 2: Sanity-check the report**

Run: `python -c "import pathlib,glob; print(pathlib.Path(sorted(glob.glob('docs/launch/baseline/baseline-*.md'))[-1]).read_text(encoding='utf-8')[:400])"`
Expected: shows the git SHA, an OpenAPI operation count > 40, and the start of the route-inventory table.

- [ ] **Step 3: Commit**

```bash
git add docs/launch/baseline/
git commit -m "docs(launch): first frozen baseline report (Phase 0a)"
```

---

## Task 5: Evidence registry loader + validator (TDD)

**Files:**
- Create: `scripts/launch/evidence_registry.py`
- Test: `tests/launch/test_evidence_registry.py`

- [ ] **Step 1: Write the failing tests**

`tests/launch/test_evidence_registry.py`:
```python
import pathlib

import pytest

from scripts.launch.evidence_registry import (
    VALID_STATUSES,
    load_registry,
    summarize,
    validate,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REGISTRY = _ROOT / "docs" / "launch" / "evidence_registry.yaml"


def _row(**over):
    base = {
        "id": "P0-X",
        "category": "maintainability",
        "phase": 0,
        "description": "desc",
        "status": "passing",
        "subsystem": "api",
        "verify": "pytest -q",
        "metric": "green",
        "evidence": "tests/foo.py",
        "external_review": "none",
        "blocking_ring": "ci",
        "last_verified": "2026-06-26",
        "score_contribution": "maintainability",
    }
    base.update(over)
    return base


def test_valid_registry_has_no_errors():
    assert validate({"requirements": [_row()]}) == []


def test_missing_required_field_is_reported():
    bad = _row()
    del bad["status"]
    errors = validate({"requirements": [bad]})
    assert any("status" in e for e in errors)


def test_invalid_status_is_reported():
    errors = validate({"requirements": [_row(status="done")]})
    assert any("status" in e and "done" in e for e in errors)


def test_duplicate_ids_are_reported():
    errors = validate({"requirements": [_row(id="DUP"), _row(id="DUP")]})
    assert any("DUP" in e for e in errors)


def test_summarize_counts_by_status():
    reg = {"requirements": [_row(status="passing"), _row(id="P0-Y", status="planned")]}
    summary = summarize(reg)
    assert summary["passing"] == 1
    assert summary["planned"] == 1


def test_status_enum_is_the_documented_set():
    assert VALID_STATUSES == {
        "planned",
        "in_progress",
        "passing",
        "failing",
        "deferred",
        "waived",
    }


@pytest.mark.skipif(not _REGISTRY.exists(), reason="registry not created yet")
def test_committed_registry_validates():
    assert validate(load_registry(_REGISTRY)) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/launch/test_evidence_registry.py -v`
Expected: FAIL with `ImportError` (module not defined). The committed-registry test is skipped (file absent).

- [ ] **Step 3: Write the minimal implementation**

`scripts/launch/evidence_registry.py`:
```python
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
            errors.append(
                f"{rid}: invalid status '{status}' (must be one of {sorted(VALID_STATUSES)})"
            )
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
        default=str(
            pathlib.Path(__file__).resolve().parents[2]
            / "docs"
            / "launch"
            / "evidence_registry.yaml"
        ),
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
```

- [ ] **Step 4: Confirm PyYAML is a declared dependency**

Run: `grep -i "pyyaml\|^yaml" requirements.txt`
Expected: PyYAML is present. **If it is not**, add `PyYAML>=6.0` to `requirements.txt` and `pip install PyYAML>=6.0` before continuing (it is a common transitive dep but the registry now depends on it directly).

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/launch/test_evidence_registry.py -v`
Expected: PASS (the committed-registry test still SKIPS — file created in Task 6).

- [ ] **Step 6: Commit**

```bash
git add scripts/launch/evidence_registry.py tests/launch/test_evidence_registry.py requirements.txt
git commit -m "feat(launch): evidence registry loader/validator with status enum (Phase 0a)"
```

---

## Task 6: Seed the evidence registry

**Files:**
- Create: `docs/launch/evidence_registry.yaml`

- [ ] **Step 1: Write the seed registry**

Create `docs/launch/evidence_registry.yaml`. Seed it with: one row per phase's top-level internal gate (0–15, status `planned` except Phase 0a rows), the deferred external certifications (status `deferred`), and the guards built in this plan (status `passing`). Use this exact starting content (extend later as phases land):

```yaml
meta:
  program: HEATER Public Commercial Launch
  spec: docs/superpowers/specs/2026-06-26-heater-public-commercial-launch-program-design.md
  updated: "2026-06-26"

requirements:
  # ---- Phase 0a gates (this plan) ----
  - id: P0A-OPENAPI-CURRENT
    category: maintainability
    phase: 0
    description: Committed api/openapi.json matches the live FastAPI schema.
    status: passing
    subsystem: api
    verify: "python -m pytest tests/api/test_openapi_contract.py -q"
    metric: "snapshot test green"
    evidence: "tests/api/test_openapi_contract.py"
    external_review: none
    blocking_ring: ci
    last_verified: "2026-06-26"
    score_contribution: maintainability
  - id: P0A-ROUTERS-MOUNTED
    category: maintainability
    phase: 0
    description: Every api/routers/*.py with a router is mounted in api/main.py.
    status: passing
    subsystem: api
    verify: "python -m pytest tests/launch/test_guard_routers_mounted.py -q"
    metric: "guard green"
    evidence: "tests/launch/test_guard_routers_mounted.py"
    external_review: none
    blocking_ring: ci
    last_verified: "2026-06-26"
    score_contribution: maintainability
  - id: P0A-OPENAPI-TS-SYNC
    category: maintainability
    phase: 0
    description: web/src/lib/api/generated.ts is regenerable from api/openapi.json with no drift.
    status: passing
    subsystem: web
    verify: "cd web && pnpm gen:api && git diff --exit-code src/lib/api/generated.ts"
    metric: "no TS drift in CI"
    evidence: ".github/workflows/ci.yml (openapi-ts-sync step)"
    external_review: none
    blocking_ring: ci
    last_verified: "2026-06-26"
    score_contribution: maintainability
  - id: P0A-BASELINE
    category: maintainability
    phase: 0
    description: A reproducible baseline report exists for the audited commit.
    status: passing
    subsystem: launch
    verify: "python -m scripts.launch.freeze_baseline"
    metric: "report written under docs/launch/baseline/"
    evidence: "docs/launch/baseline/"
    external_review: none
    blocking_ring: local
    last_verified: "2026-06-26"
    score_contribution: maintainability

  # ---- Per-phase top-level internal gates (planned) ----
  - id: P0B-API-CONTRACT-FOUNDATION
    category: api
    phase: 0
    description: Error envelope, correlation-ID, OpenAPI bearer scheme, versioning, idempotency, async-job contract live + enforced.
    status: planned
    subsystem: api
    verify: "tbd in plan 0b"
    metric: "contract tests green"
    evidence: "docs/superpowers/plans/2026-06-..-heater-launch-phase0b-*.md"
    external_review: none
    blocking_ring: ci
    last_verified: ""
    score_contribution: api
  - id: P1-LEGAL-DATA-RIGHTS
    category: data_quality
    phase: 1
    description: Every production source approved/replaced/disabled; policies drafted; privacy export+delete pass.
    status: planned
    subsystem: legal
    verify: "manual + privacy workflow staging tests"
    metric: "no feature on an unresolved source"
    evidence: "docs/legal/, docs/launch/source_inventory.md"
    external_review: lawyer
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: data_quality
  - id: P2-POSTGRES
    category: persistence
    phase: 2
    description: All production paths use PostgreSQL; no runtime DDL; behavior-equivalent; restore drill passes.
    status: planned
    subsystem: backend
    verify: "alembic upgrade head on PG + parity suite"
    metric: "parity + restore drill green"
    evidence: "alembic/, tests/db/"
    external_review: none
    blocking_ring: owner_prod
    last_verified: ""
    score_contribution: persistence
  - id: P3-TENANT-ISOLATION
    category: multi_tenancy
    phase: 3
    description: tenant_id/league_id enforced; RLS on; cross-tenant adversarial suite passes; multi-league switch works.
    status: planned
    subsystem: backend
    verify: "tests/tenancy/ adversarial matrix"
    metric: "zero cross-tenant exposure"
    evidence: "tests/tenancy/"
    external_review: pentest
    blocking_ring: existing_league
    last_verified: ""
    score_contribution: multi_tenancy
  - id: P4-FORMAT-GENERALIZATION
    category: analytical_value
    phase: 4
    description: Engines accept arbitrary league configs; Roto + Points engines correct; no hardcoded format assumptions.
    status: planned
    subsystem: engine
    verify: "per-format validation harness"
    metric: "correct across format corpus"
    evidence: "tests/engine/formats/"
    external_review: none
    blocking_ring: invite_beta
    last_verified: ""
    score_contribution: analytical_value
  - id: P5-CONNECTORS
    category: integrations
    phase: 5
    description: LeagueConnector interface; Yahoo per-user OAuth + write; >=1 guided import; capability flags honored.
    status: planned
    subsystem: integrations
    verify: "recorded-fixture + sandbox connector tests"
    metric: "connect/reconnect without admin"
    evidence: "tests/connectors/"
    external_review: provider_terms
    blocking_ring: invite_beta
    last_verified: ""
    score_contribution: integrations
  - id: P6-DATA-QUALITY
    category: data_quality
    phase: 6
    description: Canonical identity graph; ingestion quality gates; freshness/lineage on every payload; immutable snapshots.
    status: planned
    subsystem: data
    verify: "data-quality gate suite"
    metric: "identity coverage target; quality blocks publish"
    evidence: "tests/data_quality/"
    external_review: none
    blocking_ring: invite_beta
    last_verified: ""
    score_contribution: data_quality
  - id: P7-PREDICTION-VALIDATION
    category: prediction_accuracy
    phase: 7
    description: Point-in-time backtests run the real engines; constants calibrated or labeled heuristic; model cards complete.
    status: planned
    subsystem: engine
    verify: "backtest + calibration reports"
    metric: "holdout metrics beat baselines with CIs"
    evidence: "docs/launch/validation/"
    external_review: statistician
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: prediction_accuracy
  - id: P8-RECOMMENDATION-TRUST
    category: recommendation_trust
    phase: 8
    description: Immutable recommendation records + evidence cards + abstention + conflict detection + outcome tracking.
    status: planned
    subsystem: backend
    verify: "tests/recommendations/"
    metric: "100% recs have evidence; replay-equivalent"
    evidence: "tests/recommendations/"
    external_review: none
    blocking_ring: invite_beta
    last_verified: ""
    score_contribution: recommendation_trust
  - id: P9-RELIABILITY
    category: reliability
    phase: 9
    description: Redis/Arq workers; async job API; SLOs/alerts; backups/DR drill; load+chaos pass; no unbounded request compute.
    status: planned
    subsystem: platform
    verify: "load/chaos + restore drill"
    metric: "SLO targets; restore drill green"
    evidence: "docs/launch/ops/"
    external_review: none
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: reliability
  - id: P10-SECURITY
    category: security
    phase: 10
    description: Authz policy engine; rate limits; credential envelope-encryption; API/import/AI hardening; SDLC scans.
    status: planned
    subsystem: platform
    verify: "negative authz tests + SDLC scans"
    metric: "zero internal Critical/High"
    evidence: "tests/security/"
    external_review: pentest
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: security
  - id: P11-BILLING
    category: billing
    phase: 11
    description: Subscription state machine; webhook dedup/ordering; tiered+metered entitlements; portal; daily reconciliation.
    status: planned
    subsystem: backend
    verify: "Stripe certification + webhook replay"
    metric: "reconciliation clean; cost limits unbypassable"
    evidence: "tests/billing/"
    external_review: stripe_live_cert
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: billing
  - id: P12-ONBOARDING
    category: onboarding
    phase: 12
    description: Self-service onboarding + recovery states + education + support ops + privacy-safe analytics.
    status: planned
    subsystem: frontend
    verify: "onboarding E2E + recovery-state tests"
    metric: "unassisted onboarding + activation targets"
    evidence: "web/tests/onboarding/"
    external_review: none
    blocking_ring: invite_beta
    last_verified: ""
    score_contribution: onboarding
  - id: P13-FRONTEND-A11Y
    category: ui_ux
    phase: 13
    description: IA/responsive/type/design-system/states; WCAG 2.2 AA internal; CWV good p75; frontend CI suite.
    status: planned
    subsystem: frontend
    verify: "axe + playwright + responsive-overflow + CWV"
    metric: "zero overflow; internal a11y clean; CWV good"
    evidence: "web/tests/"
    external_review: a11y_audit
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: ui_ux
  - id: P14-AI-SAFETY
    category: ai
    phase: 14
    description: Bubba grounded in canonical services; tenant-scoped read-only tools; prompt-injection red-team; eval system.
    status: planned
    subsystem: ai
    verify: "AI eval set + red-team suite"
    metric: "eval thresholds met; cost limits unbypassable"
    evidence: "tests/ai/"
    external_review: none
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: ai
  - id: P15-LAUNCH
    category: onboarding
    phase: 15
    description: Public landing + final pricing + all external certs cleared + ring progression to GA + Streamlit retired.
    status: planned
    subsystem: gtm
    verify: "ring criteria + external cert register clear"
    metric: "purchase works; all certs green"
    evidence: "docs/launch/evidence_registry.yaml"
    external_review: all
    blocking_ring: general_availability
    last_verified: ""
    score_contribution: onboarding

  # ---- Deferred external certifications (owner-arranged pre-launch) ----
  - id: EXT-LEGAL
    category: security
    phase: 15
    description: Lawyer review of ToS/Privacy/AUP + commercial data licenses signed off.
    status: deferred
    subsystem: legal
    verify: "signed external review on file"
    metric: "policies + licenses approved"
    evidence: "docs/legal/"
    external_review: lawyer
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: security
  - id: EXT-PENTEST
    category: security
    phase: 15
    description: Independent penetration test with zero Critical/High findings.
    status: deferred
    subsystem: platform
    verify: "external pentest report"
    metric: "zero Critical/High"
    evidence: "docs/launch/audits/"
    external_review: pentest
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: security
  - id: EXT-STATS
    category: prediction_accuracy
    phase: 15
    description: Independent statistical methodology + accuracy-claim sign-off.
    status: deferred
    subsystem: engine
    verify: "external statistician report"
    metric: "methodology approved"
    evidence: "docs/launch/validation/"
    external_review: statistician
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: prediction_accuracy
  - id: EXT-A11Y
    category: ui_ux
    phase: 15
    description: Independent WCAG 2.2 AA accessibility audit passes.
    status: deferred
    subsystem: frontend
    verify: "external a11y audit report"
    metric: "zero Critical/High a11y"
    evidence: "docs/launch/audits/"
    external_review: a11y_audit
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: ui_ux
  - id: EXT-STRIPE-LIVE
    category: billing
    phase: 15
    description: Live-mode end-to-end purchase certification (real card).
    status: deferred
    subsystem: backend
    verify: "owner live-mode purchase run"
    metric: "real purchase + entitlement + refund"
    evidence: "docs/launch/audits/"
    external_review: stripe_live_cert
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: billing

  # ---- Guards deferred to their owning phase (concept does not exist yet) ----
  - id: GUARD-TENANT-SCOPE
    category: multi_tenancy
    phase: 3
    description: Guard fails when a tenant-owned table lacks tenant_id / a league-owned table lacks league_id.
    status: deferred
    subsystem: backend
    verify: "tests/tenancy/test_guard_tenant_scope.py (Phase 3)"
    metric: "guard active"
    evidence: "Phase 3 plan"
    external_review: none
    blocking_ring: existing_league
    last_verified: ""
    score_contribution: multi_tenancy
  - id: GUARD-RECO-EVIDENCE
    category: recommendation_trust
    phase: 8
    description: Guard fails when a recommendation-producing path lacks evidence metadata.
    status: deferred
    subsystem: backend
    verify: "tests/recommendations/test_guard_evidence.py (Phase 8)"
    metric: "guard active"
    evidence: "Phase 8 plan"
    external_review: none
    blocking_ring: invite_beta
    last_verified: ""
    score_contribution: recommendation_trust
  - id: GUARD-CALIBRATED-PROVENANCE
    category: prediction_accuracy
    phase: 7
    description: Guard fails when a constant marked calibrated lacks dataset/method/metrics/reviewer provenance.
    status: deferred
    subsystem: engine
    verify: "tests/validation/test_guard_calibrated_provenance.py (Phase 7)"
    metric: "guard active"
    evidence: "Phase 7 plan"
    external_review: none
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: prediction_accuracy
  - id: GUARD-NO-MOCK-FALLBACK
    category: ui_ux
    phase: 13
    description: Guard fails when a live frontend fetcher converts an error into mock data.
    status: deferred
    subsystem: frontend
    verify: "web/tests/guard-no-mock-fallback (Phase 13)"
    metric: "guard active"
    evidence: "Phase 13 plan"
    external_review: none
    blocking_ring: limited_public
    last_verified: ""
    score_contribution: ui_ux
```

- [ ] **Step 2: Validate the seeded registry**

Run: `python -m scripts.launch.evidence_registry --summary`
Expected: prints a total count and a per-status rollup with no `INVALID registry` output (exit 0).

- [ ] **Step 3: Run the registry tests (the committed-registry test now runs)**

Run: `python -m pytest tests/launch/test_evidence_registry.py -v`
Expected: PASS, including `test_committed_registry_validates` (no longer skipped).

- [ ] **Step 4: Commit**

```bash
git add docs/launch/evidence_registry.yaml
git commit -m "docs(launch): seed evidence registry with 16-phase gates + deferred certs/guards (Phase 0a)"
```

---

## Task 7: Guard — every router is mounted (TDD)

**Files:**
- Create: `tests/launch/test_guard_routers_mounted.py`

- [ ] **Step 1: Write the failing test**

`tests/launch/test_guard_routers_mounted.py`:
```python
"""Guard: every api/routers/*.py that defines a `router` is imported AND
included in api/main.py. Catches the 'feature exists in code but is not
mounted / not in the contract' drift (Codex Phase 0 inventory guard)."""

from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_ROUTERS_DIR = _ROOT / "api" / "routers"
_MAIN = _ROOT / "api" / "main.py"


def _router_modules_on_disk() -> set[str]:
    names: set[str] = set()
    for p in _ROUTERS_DIR.glob("*.py"):
        if p.name == "__init__.py":
            continue
        text = p.read_text(encoding="utf-8")
        # Only count modules that actually define a top-level `router`.
        if re.search(r"^router\s*=", text, re.MULTILINE):
            names.add(p.stem)
    return names


def _router_modules_imported_in_main() -> set[str]:
    text = _MAIN.read_text(encoding="utf-8")
    return set(re.findall(r"from api\.routers\.(\w+) import router", text))


def _router_aliases_included_in_main() -> int:
    text = _MAIN.read_text(encoding="utf-8")
    return len(re.findall(r"app\.include_router\(", text))


def test_every_router_module_is_imported_in_main():
    on_disk = _router_modules_on_disk()
    imported = _router_modules_imported_in_main()
    missing = on_disk - imported
    assert not missing, (
        "api/routers modules define a `router` but are not imported in api/main.py: "
        f"{sorted(missing)}. Mount them or remove the dead router."
    )


def test_import_and_include_counts_match():
    # Every imported router alias should also be passed to include_router.
    imported = _router_modules_imported_in_main()
    includes = _router_aliases_included_in_main()
    assert includes >= len(imported), (
        f"{len(imported)} routers imported but only {includes} include_router(...) calls "
        "in api/main.py — a router was imported but never mounted."
    )
```

- [ ] **Step 2: Run to verify it passes immediately (current code is clean)**

Run: `python -m pytest tests/launch/test_guard_routers_mounted.py -v`
Expected: PASS — this guard ratchets the *current* clean state. (If it FAILS, that is a real pre-existing drift: a router exists on disk but is unmounted. Investigate and either mount it in `api/main.py` or delete the dead module before continuing — do not weaken the guard.)

- [ ] **Step 3: Prove the guard bites (temporary negative check)**

Temporarily add a throwaway `api/routers/_zzz_guard_probe.py` containing `router = 1`, run the guard, confirm it FAILS naming `_zzz_guard_probe`, then delete the probe file and confirm the guard PASSES again.
```bash
printf 'router = 1\n' > api/routers/_zzz_guard_probe.py
python -m pytest tests/launch/test_guard_routers_mounted.py::test_every_router_module_is_imported_in_main -q   # expect FAIL naming _zzz_guard_probe
rm api/routers/_zzz_guard_probe.py
python -m pytest tests/launch/test_guard_routers_mounted.py -q   # expect PASS
```

- [ ] **Step 4: Commit**

```bash
git add tests/launch/test_guard_routers_mounted.py
git commit -m "test(launch): guard that every api router is mounted in main.py (Phase 0a)"
```

---

## Task 8: CI guard — OpenAPI↔generated-TS sync

**Files:**
- Create: `scripts/launch/check_ts_sync.py`
- Modify: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the convenience checker**

`scripts/launch/check_ts_sync.py`:
```python
"""Fail if web/src/lib/api/generated.ts is not regenerable from api/openapi.json
without drift. Used locally; CI runs the same check via `pnpm gen:api` +
`git diff --exit-code` (see .github/workflows/ci.yml)."""

from __future__ import annotations

import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_WEB = _ROOT / "web"
_GENERATED = _WEB / "src" / "lib" / "api" / "generated.ts"


def main() -> int:
    before = _GENERATED.read_text(encoding="utf-8") if _GENERATED.exists() else ""
    try:
        subprocess.run(["pnpm", "gen:api"], cwd=_WEB, check=True)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"could not run `pnpm gen:api`: {exc}", file=sys.stderr)
        return 2
    after = _GENERATED.read_text(encoding="utf-8")
    if before != after:
        print(
            "generated.ts drifted from api/openapi.json. Run `cd web && pnpm gen:api` "
            "and commit the result.",
            file=sys.stderr,
        )
        return 1
    print("generated.ts is in sync with api/openapi.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Inspect the existing CI workflow**

Run: `python -c "import pathlib; print(pathlib.Path('.github/workflows/ci.yml').read_text(encoding='utf-8'))"`
Read it to find where Node/pnpm is (or is not) already set up and where to add a job/step. The web app currently has no CI; this step adds the first web check.

- [ ] **Step 3: Add the sync step to CI**

Add a job to `.github/workflows/ci.yml` (adapt indentation/keys to the existing file's style; this is the canonical content):
```yaml
  openapi-ts-sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pnpm/action-setup@v4
        with:
          version: 9
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: pnpm
          cache-dependency-path: web/pnpm-lock.yaml
      - name: Install web deps
        working-directory: web
        run: pnpm install --frozen-lockfile
      - name: Regenerate API types
        working-directory: web
        run: pnpm gen:api
      - name: Fail on OpenAPI/TS drift
        run: git diff --exit-code web/src/lib/api/generated.ts
```

- [ ] **Step 4: Verify the check locally (best-effort — needs pnpm)**

Run: `python scripts/launch/check_ts_sync.py`
Expected: prints `generated.ts is in sync with api/openapi.json` (after Task 1 regenerated it). If `pnpm` is unavailable locally, the script exits 2 with a clear message — that is acceptable; CI is the enforcing surface.

- [ ] **Step 5: Validate the workflow YAML parses**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml',encoding='utf-8')); print('ci.yml parses')"`
Expected: `ci.yml parses`.

- [ ] **Step 6: Commit**

```bash
git add scripts/launch/check_ts_sync.py .github/workflows/ci.yml
git commit -m "ci(launch): enforce OpenAPI->generated.ts sync; add local checker (Phase 0a)"
```

---

## Task 9: Final verification + push

**Files:** none (verification + integration)

- [ ] **Step 1: Run the full launch test set + the contract test**

Run:
```bash
python -m pytest tests/launch tests/api/test_openapi_contract.py -q
python -m scripts.launch.evidence_registry --summary
```
Expected: all launch tests PASS; the registry summary prints with exit 0 and shows `passing` ≥ 4 and `planned`/`deferred` rows for the later phases.

- [ ] **Step 2: Run the structural-invariant subset (what pre-push runs) to confirm no regression**

Run: `python -m pytest tests/ -k "no_ or guard or launch" -q`
Expected: PASS (no new failures introduced by the launch tooling).

- [ ] **Step 3: Refresh the baseline now that 0a tooling exists**

Run: `python -m scripts.launch.freeze_baseline`
Expected: rewrites the baseline report for the current SHA. Stage it if it changed.
```bash
git add docs/launch/baseline/
git diff --cached --quiet || git commit -m "docs(launch): refresh baseline after Phase 0a tooling (Phase 0a)"
```

- [ ] **Step 4: Push (pre-push hook runs the structural suite)**

Run: `git push origin master`
Expected: pre-push structural suite PASSES; push succeeds. If the hook fails, fix the named issue — never `--no-verify`.

---

## Self-review notes

- **Spec coverage (Phase 0):** rebaseline → Tasks 3–4 (freeze tool + report); evidence registry → Tasks 5–6; structural guards (the implementable-now subset) → Tasks 1 (OpenAPI snapshot), 7 (router-mount), 8 (OpenAPI↔TS); the not-yet-applicable guards (tenant scope, recommendation evidence, calibrated provenance, frontend no-mock-fallback) → registered `deferred` in Task 6 with their owning phase. The API contract foundation portion of Phase 0 is **plan 0b** (out of scope here by design).
- **No placeholders:** every step has exact commands or complete code. The two "if X is missing" branches (PyYAML dep, version-pin mismatch) are explicit, recoverable instructions, not deferrals.
- **Type/name consistency:** `openapi_operation_count`, `route_inventory`, `build_report` (freeze); `load_registry`, `validate`, `summarize`, `VALID_STATUSES` (registry) — used identically in their tests and the seed/CLI. Registry `status` values in the YAML are all within `VALID_STATUSES`.
```
