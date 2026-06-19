# M0 Slice 1 — PlayerRef contract extension + display enrichment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the API's `PlayerRef` model with three optional display fields (`mlb_id`, `team_abbr`, `team_id`) so the frontend can render player headshots and MLB team logos, and populate them in the five service sites that already have the data in hand.

**Architecture:** Add the fields to `api/contracts/common.py` (all optional, default `None` → fully backward-compatible). Add ONE shared builder `api/services/player_ref.py` that normalizes the fields and derives `team_id` from `team_abbr` by reusing the single-source 30-team map already in `src/depth_charts.py` (no duplicate map). Wire the builder into the five service mapping functions that expose an injectable row/pool (trade, trade_finder, databank, closers, streaming). Regenerate the OpenAPI snapshot.

**Tech Stack:** FastAPI + Pydantic v2 contracts; pandas DataFrames in the service layer; pytest. Engines in `src/` are NOT modified (one read-only constant is imported, lazily). Routers are NOT touched (they stay logic-free — AST-guarded).

**Scope note (firm):** This slice covers ONLY the contract + helper + the 5 injectable sites. The remaining `PlayerRef` sites — `compare_service` (inline, DB-backed), `leaders_service` (DB query — needs `p.mlb_id, p.team` added to its SELECT), `lineup_service`, `fa_service`, `draft_service` (engine dicts that carry only `player_id`, need a new pool lookup) — are DEFERRED to a follow-up slice (M0.2). They keep emitting `null` for the new fields (valid + backward-compatible) until then.

**Interpreter for ALL commands:** `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe` (the shared venv). Run from the worktree root (cwd) so the worktree's `api/`/`src/`/`tests/` are imported.

---

## File Structure

- `api/contracts/common.py` — **modify**: add 3 optional fields to `PlayerRef`.
- `api/services/player_ref.py` — **create**: `team_id_for()` + `make_player_ref()` (the one place that maps team_abbr→team_id and normalizes the enrichment fields).
- `api/services/trade_service.py` — **modify**: enrich the found-row branch of `_build_player_refs`.
- `api/services/trade_finder_service.py` — **modify**: enrich the found-row branch of `_build_player_refs`.
- `api/services/databank_service.py` — **modify**: enrich the found-row branch of `DatabankService._build_ref`.
- `api/services/closers_service.py` — **modify**: enrich the closer `PlayerRef` in `_to_entry`.
- `api/services/streaming_service.py` — **modify**: enrich the `PlayerRef` in `_to_candidate` (team only — stream board has no mlb_id).
- `api/openapi.json` — **regenerate** via `scripts/export_openapi.py`.
- `tests/api/test_api_player_ref.py` — **create**: helper unit tests.
- `tests/api/test_api_player_ref_enrichment.py` — **create**: per-site enrichment tests.

---

### Task 1: Extend `PlayerRef` + create the shared builder

**Files:**
- Modify: `api/contracts/common.py`
- Create: `api/services/player_ref.py`
- Test: `tests/api/test_api_player_ref.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_api_player_ref.py`:

```python
"""Unit tests for the shared PlayerRef builder + team-id mapping (M0 slice 1)."""

from __future__ import annotations

from api.contracts.common import PlayerRef
from api.services.player_ref import make_player_ref, team_id_for


def test_team_id_for_known_abbr():
    assert team_id_for("NYY") == 147
    assert team_id_for("LAD") == 119
    assert team_id_for("ATH") == 133  # codebase-canonical (not "OAK")


def test_team_id_for_is_case_and_whitespace_insensitive():
    assert team_id_for(" nyy ") == 147


def test_team_id_for_unknown_or_blank_returns_none():
    assert team_id_for("ZZZ") is None
    assert team_id_for("") is None
    assert team_id_for(None) is None


def test_make_player_ref_populates_enrichment():
    ref = make_player_ref(
        id=42, name="Aaron Judge", positions="OF", mlb_id=592450, team_abbr="NYY"
    )
    assert isinstance(ref, PlayerRef)
    assert ref.id == 42
    assert ref.mlb_id == 592450
    assert ref.team_abbr == "NYY"
    assert ref.team_id == 147
    assert ref.positions == "OF"


def test_make_player_ref_normalizes_missing_data():
    ref = make_player_ref(id=0, name="x", positions="", mlb_id=0, team_abbr="  ")
    assert ref.mlb_id is None  # 0 -> None (not a real MLB id)
    assert ref.team_abbr is None  # blank -> None
    assert ref.team_id is None


def test_make_player_ref_handles_float_and_nan_mlb_id():
    assert make_player_ref(id=1, name="x", positions="", mlb_id=592450.0).mlb_id == 592450
    assert make_player_ref(id=1, name="x", positions="", mlb_id=float("nan")).mlb_id is None


def test_make_player_ref_unknown_team_keeps_abbr_but_null_id():
    ref = make_player_ref(id=1, name="x", positions="", team_abbr="ZZZ")
    assert ref.team_abbr == "ZZZ"
    assert ref.team_id is None


def test_make_player_ref_defaults_all_enrichment_to_none():
    ref = make_player_ref(id=3, name="y", positions="SP")
    assert ref.mlb_id is None
    assert ref.team_abbr is None
    assert ref.team_id is None
    assert ref.yahoo_player_key is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'api.services.player_ref'` (and/or `PlayerRef` has no `mlb_id`).

- [ ] **Step 3: Add the fields to `PlayerRef`**

Replace the body of `api/contracts/common.py` `PlayerRef` so it reads exactly:

```python
class PlayerRef(BaseModel):
    id: int
    mlb_id: int | None = None
    name: str
    positions: str
    team_abbr: str | None = None
    team_id: int | None = None
    yahoo_player_key: str | None = None
```

(Only the class body changes — keep the module's existing imports/docstring.)

- [ ] **Step 4: Create the shared builder**

Create `api/services/player_ref.py` with exactly:

```python
"""Shared PlayerRef construction with display enrichment (mlb_id + team).

The frontend renders headshots (via mlb_id) and team logos (via team_abbr /
team_id) from PlayerRef. This module centralizes:
  - team_abbr -> MLB Stats API numeric team_id, reusing the single-source
    30-team map in src.depth_charts (no duplicate map).
  - normalization of mlb_id (0 / negative / NaN -> None) and team_abbr
    (blank -> None).

It lives in api/services because that is the one layer permitted to import
from src/. The src import is lazy (inside team_id_for) to keep this module's
import graph light and avoid any startup-import surprise.
"""

from __future__ import annotations

from api.contracts.common import PlayerRef


def team_id_for(team_abbr: str | None) -> int | None:
    """Map a canonical MLB team abbreviation to its MLB Stats API numeric id.

    Returns None for unknown/blank abbreviations. Reuses the single-source
    `_STATSAPI_TEAM_IDS` map in src.depth_charts so the abbr->id mapping is
    not duplicated.
    """
    if not team_abbr:
        return None
    from src.depth_charts import _STATSAPI_TEAM_IDS

    return _STATSAPI_TEAM_IDS.get(team_abbr.strip().upper())


def _coerce_mlb_id(value) -> int | None:
    """Coerce a raw mlb_id (int/float/str/None/NaN) to a positive int or None."""
    if value is None:
        return None
    try:
        ivalue = int(value)
    except (TypeError, ValueError):
        return None
    return ivalue if ivalue > 0 else None


def make_player_ref(
    *,
    id: int,
    name: str,
    positions: str,
    mlb_id=None,
    team_abbr=None,
    yahoo_player_key: str | None = None,
) -> PlayerRef:
    """Build a PlayerRef with display enrichment populated where available.

    mlb_id is normalized (0 / negative / NaN -> None); team_abbr is stripped
    (blank -> None) and team_id is derived from it. All enrichment fields
    default to None so callers that lack the data still emit a valid ref.
    """
    abbr = (str(team_abbr).strip() if team_abbr is not None else "") or None
    return PlayerRef(
        id=int(id) if id is not None else 0,
        mlb_id=_coerce_mlb_id(mlb_id),
        name=str(name) if name is not None else "",
        positions=str(positions) if positions is not None else "",
        team_abbr=abbr,
        team_id=team_id_for(abbr),
        yahoo_player_key=yahoo_player_key,
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref.py -q`
Expected: PASS (8 tests).

- [ ] **Step 6: Commit**

```bash
git add api/contracts/common.py api/services/player_ref.py tests/api/test_api_player_ref.py
git commit -m "feat(api): extend PlayerRef with mlb_id/team_abbr/team_id + shared builder"
```

---

### Task 2: Enrich the pool-row sites (trade, trade_finder, databank)

These three map a player-pool DataFrame row; the pool has `mlb_id` and `team` columns. Each has an injectable function so it is directly unit-testable.

**Files:**
- Modify: `api/services/trade_service.py` (function `_build_player_refs`, the found-row branch ~line 184-190)
- Modify: `api/services/trade_finder_service.py` (function `_build_player_refs`, the found-row branch ~line 116-123)
- Modify: `api/services/databank_service.py` (`DatabankService._build_ref`, the found-row branch ~line 83-87)
- Test: `tests/api/test_api_player_ref_enrichment.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_api_player_ref_enrichment.py`:

```python
"""Per-site PlayerRef enrichment tests (M0 slice 1).

Each test injects a fabricated pool row / grid dict and asserts the service's
mapping function fills mlb_id / team_abbr / team_id.
"""

from __future__ import annotations

import pandas as pd

from api.services.databank_service import DatabankService
from api.services.trade_finder_service import _build_player_refs as finder_refs
from api.services.trade_service import _build_player_refs as trade_refs


def _pool_one(pid: int, mlb_id: int, team: str) -> pd.DataFrame:
    return pd.DataFrame(
        [{"player_id": pid, "name": "Test Player", "positions": "OF", "mlb_id": mlb_id, "team": team}]
    )


def test_trade_build_player_refs_enriches_from_pool():
    refs = trade_refs([5], _pool_one(5, 605141, "LAD"))
    assert len(refs) == 1
    assert refs[0].mlb_id == 605141
    assert refs[0].team_abbr == "LAD"
    assert refs[0].team_id == 119


def test_trade_finder_build_player_refs_enriches_from_pool():
    refs = finder_refs([8], _pool_one(8, 660271, "LAA"))
    assert refs[0].mlb_id == 660271
    assert refs[0].team_abbr == "LAA"
    assert refs[0].team_id == 108


def test_databank_build_ref_enriches_from_pool():
    ref = DatabankService._build_ref(7, _pool_one(7, 456789, "BOS"))
    assert ref.mlb_id == 456789
    assert ref.team_abbr == "BOS"
    assert ref.team_id == 111
```

NOTE for the implementer: confirm `_build_player_refs` is importable as a module-level function in both `trade_service.py` and `trade_finder_service.py` (it is per the source). If either is actually a method, import/reference it accordingly and keep the assertions identical.

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py -q`
Expected: FAIL — `assert None == 605141` (fields not populated yet).

- [ ] **Step 3: Edit the three found-row branches**

In each file: add the import `from api.services.player_ref import make_player_ref` (top of file with the other imports), then in the branch that has the found pool row `r`, replace the `PlayerRef(...)` constructor with `make_player_ref(...)` passing the SAME `id`/`name`/`positions` arguments PLUS `mlb_id=r.get("mlb_id"), team_abbr=r.get("team")`. Leave the empty-row fallback `PlayerRef(id=..., name=f"Player {...}", positions="")` calls UNCHANGED.

`trade_service.py` — the found branch becomes:

```python
                r = row.iloc[0]
                refs.append(
                    make_player_ref(
                        id=pid,
                        name=str(r.get(name_col, f"Player {pid}") or f"Player {pid}"),
                        positions=str(r.get("positions", "") or ""),
                        mlb_id=r.get("mlb_id"),
                        team_abbr=r.get("team"),
                    )
                )
```

`trade_finder_service.py` — apply the identical change to its found branch (same variable names: `r`, `pid`, `name_col`).

`databank_service.py` — `_build_ref` found branch becomes:

```python
                    r = row_df.iloc[0]
                    name_col = "player_name" if "player_name" in pool.columns else "name"
                    return make_player_ref(
                        id=player_id,
                        name=str(r.get(name_col, f"Player {player_id}") or f"Player {player_id}"),
                        positions=str(r.get("positions", "") or ""),
                        mlb_id=r.get("mlb_id"),
                        team_abbr=r.get("team"),
                    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add api/services/trade_service.py api/services/trade_finder_service.py api/services/databank_service.py tests/api/test_api_player_ref_enrichment.py
git commit -m "feat(api): enrich pool-row PlayerRefs (trade, trade_finder, databank)"
```

---

### Task 3: Enrich the dict sites (closers, streaming)

`closers_service._to_entry(row)` and `streaming_service._to_candidate(row)` are static methods taking a row dict. The closer grid row carries `mlb_id` + `team`; the stream board row carries `team` only (no mlb_id, which stays `None`).

**Files:**
- Modify: `api/services/closers_service.py` (`_to_entry`, the `closer_ref = PlayerRef(...)` at ~line 48)
- Modify: `api/services/streaming_service.py` (`_to_candidate`, the `player = PlayerRef(...)` at ~line 55)
- Test: append to `tests/api/test_api_player_ref_enrichment.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_api_player_ref_enrichment.py`:

```python
from api.services.closers_service import ClosersService
from api.services.streaming_service import StreamingService


def test_closers_to_entry_enriches_closer_ref():
    row = {
        "team": "NYY",
        "closer_name": "Devin Williams",
        "mlb_id": 642207,
        "job_security": 0.8,
        "setup_names": [],
    }
    entry = ClosersService._to_entry(row)
    assert entry.closer is not None
    assert entry.closer.mlb_id == 642207
    assert entry.closer.team_abbr == "NYY"
    assert entry.closer.team_id == 147


def test_streaming_to_candidate_enriches_team_only():
    row = {"player_id": 9, "player_name": "Tarik Skubal", "team": "DET", "stream_score": 80.0}
    cand = StreamingService._to_candidate(row)
    assert cand.player.team_abbr == "DET"
    assert cand.player.team_id == 116
    assert cand.player.mlb_id is None  # stream board carries no mlb_id
```

NOTE for the implementer: confirm the class names `ClosersService` / `StreamingService` and that `_to_entry` / `_to_candidate` are `@staticmethod` (they are per the source). Adjust the references if the file uses different class names; keep the assertions identical.

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py -q`
Expected: FAIL — closer/candidate enrichment fields are None.

- [ ] **Step 3: Edit the two sites**

`closers_service.py` — add `from api.services.player_ref import make_player_ref` with the other imports; replace the closer_ref construction:

```python
        closer_ref: PlayerRef | None = None
        if closer_name and closer_name.lower() not in ("unknown", ""):
            closer_ref = make_player_ref(
                id=int(mlb_id) if mlb_id is not None else 0,
                name=closer_name,
                positions="RP",
                mlb_id=mlb_id,
                team_abbr=team,
            )
```

Leave the handcuffs list (`PlayerRef(id=0, name=str(name), positions="RP")`) UNCHANGED — they carry no id/team data.

`streaming_service.py` — add the same import; replace the player construction:

```python
        player = make_player_ref(id=pid_int, name=name, positions=positions, team_abbr=g("team"))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_api_player_ref_enrichment.py -q`
Expected: PASS (5 tests total in the file).

- [ ] **Step 5: Commit**

```bash
git add api/services/closers_service.py api/services/streaming_service.py tests/api/test_api_player_ref_enrichment.py
git commit -m "feat(api): enrich closer + streaming PlayerRefs (team logos; closers also mlb_id)"
```

---

### Task 4: Regenerate the OpenAPI snapshot + full api-suite green

The `PlayerRef` schema changed, so `api/openapi.json` (snapshot-guarded by `tests/api/test_openapi_contract.py`) must be regenerated.

**Files:**
- Regenerate: `api/openapi.json`

- [ ] **Step 1: Confirm the snapshot test currently fails (proving the change is real)**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: FAIL (the committed snapshot lacks the new `PlayerRef` fields).

- [ ] **Step 2: Regenerate the snapshot**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe scripts/export_openapi.py`
Expected: rewrites `api/openapi.json`. Confirm the `PlayerRef` schema in the file now contains `mlb_id`, `team_abbr`, `team_id`.

- [ ] **Step 3: Run the snapshot test to verify it passes**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/test_openapi_contract.py -q`
Expected: PASS.

- [ ] **Step 4: Run the FULL api suite + fix any exact-equality fallout**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m pytest tests/api/ -q`
Expected: PASS (all api tests, including the new ones).

If any pre-existing test fails because it asserts EXACT equality on a response containing a `PlayerRef` (the response now includes `"mlb_id": null, "team_abbr": null, "team_id": null` for fake data that doesn't set them), update that test's expected dict to include the three new keys (value `null`/`None` unless the fake data sets them). Do NOT weaken assertions beyond adding the new keys.

- [ ] **Step 5: Lint**

Run: `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff format api/ tests/api/` then `C:/Users/conno/Code/HEATER_v1.0.1/.venv/Scripts/python.exe -m ruff check api/ tests/api/`
Expected: both clean (no diffs needing attention / no lint errors).

- [ ] **Step 6: Commit**

```bash
git add api/openapi.json
git commit -m "chore(api): regenerate OpenAPI snapshot for PlayerRef enrichment fields"
```

(If Step 4 required editing any test file, include it in this commit.)

---

## Self-Review (completed by plan author)

**Spec coverage** (vs `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md` "Tier 1 — Universal fix"):
- `PlayerRef` gains `mlb_id`, `team_abbr`, `team_id` (all optional) — Task 1. ✔ Matches the spec's recommended shape.
- `positions` is NOT renamed to `pos` — intentional: the spec calls it a "minor rename" handled by the frontend adapter/type-gen; renaming would break existing tests + every other contract. ✔
- `yahoo_player_key` retained. ✔
- Enrichment populated in the 5 data-in-hand sites; the 5 DB/lookup sites explicitly deferred to M0.2 (documented in the Scope note). ✔ (Partial rollout is by design — the CONTRACT, the M0 deliverable, lands fully here.)

**Placeholder scan:** No TBD/"handle edge cases"/uncoded steps. Every code step shows full code. ✔

**Type consistency:** `make_player_ref(*, id, name, positions, mlb_id=None, team_abbr=None, yahoo_player_key=None)` and `team_id_for(team_abbr)` are used identically in Tasks 2-3 as defined in Task 1. Field names (`mlb_id`/`team_abbr`/`team_id`) match across the contract, the helper, and the tests. ✔
