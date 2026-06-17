# HEATER Backend API — Slice 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Add three read endpoints — `GET /api/standings`, `GET /api/closers`, `GET /api/leaders` — over the existing engines, using the exact contract-first, logic-free-router pattern Slices 1–2 established.

**Architecture:** Identical to Slice 2. Each endpoint = a Pydantic contract (`api/contracts/`) + a service that calls the engine and maps to the contract (`api/services/`) + a thin router (`api/routers/`) + a DI provider (`api/deps.py`) + a fake-service test (`tests/api/`).

**Tech Stack:** FastAPI · Pydantic v2 · pytest · existing `src/` engines (standings, closer_monitor, leaders). SQLite (unchanged).

> **⚠ B1 LESSON — read before coding:** implement the contract models **EXACTLY as written below. Do NOT redesign them** (no flattening `PlayerRef` into strings, no renaming/removing fields, no making optionals required). The frontend generates its TypeScript types from the OpenAPI these produce, so the shapes are load-bearing. Reuse the existing `api/contracts/common.py::PlayerRef`.

**Pattern templates (read first, mirror exactly):** `api/contracts/free_agents.py`, `api/services/fa_service.py`, `api/routers/free_agents.py`, `api/deps.py`, `api/main.py`, `tests/api/test_free_agents.py`.

---

## File Structure

| File | Responsibility |
|---|---|
| `api/contracts/standings.py` | `TeamStanding`, `StandingsResponse`. |
| `api/contracts/closers.py` | `CloserEntry`, `ClosersResponse`. |
| `api/contracts/leaders.py` | `LeaderRow`, `LeadersResponse`. |
| `api/services/standings_service.py` | `StandingsService.get_standings()`. |
| `api/services/closers_service.py` | `CloserService.get_closers()`. |
| `api/services/leaders_service.py` | `LeadersService.get_leaders(category, limit)`. |
| `api/routers/standings.py` / `closers.py` / `leaders.py` | thin routers. |
| `api/deps.py` | ADD `get_standings_service`, `get_closer_service`, `get_leaders_service`. |
| `api/main.py` | ADD the three routers. |
| `tests/api/test_standings.py` / `test_closers.py` / `test_leaders.py` | contract + endpoint tests (fake services). |
| `api/openapi.json` | regenerated. |

---

### Task 1: Standings endpoint

**Files:** Create `api/contracts/standings.py`, `api/services/standings_service.py`, `api/routers/standings.py`, `tests/api/test_standings.py`; Modify `api/deps.py`, `api/main.py`.

**Contract (implement EXACTLY):**
```python
# api/contracts/standings.py
"""Contract models for the League Standings page."""

from __future__ import annotations

from pydantic import BaseModel


class TeamStanding(BaseModel):
    rank: int
    team_name: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points: float = 0.0
    category_ranks: dict[str, int] = {}


class StandingsResponse(BaseModel):
    teams: list[TeamStanding]
```

**Service:** mirror `fa_service.py` structure (lazy `src` imports, try/except → empty list on cold env). Call `get_yahoo_data_service().get_standings()` (a DataFrame). INSPECT its actual columns at build time (the standings frame is keyed per team × category with rank/total); aggregate to one `TeamStanding` per team — derive `rank`/`wins`/`losses`/`ties` from the overall-record row and fill `category_ranks` from the per-category rows. If the frame shape is unclear, map what you can and return the rest as defaults; flag uncertainty in your report.

**Router (mirror `routers/free_agents.py` — thin, no `src` import):** `GET /api/standings` → `StandingsResponse`, `service=Depends(get_standings_service)`.

**DI:** add `get_standings_service` to `api/deps.py`. **main.py:** include the router.

**Test (mirror `test_free_agents.py`):** a contract-shape test + an endpoint test that injects a fake `StandingsService` via `app.dependency_overrides` returning one `TeamStanding`, asserting `body["teams"][0]["team_name"]`.

- [ ] Step 1: write the failing contract + endpoint tests → run, confirm fail
- [ ] Step 2: implement contract, service, router, deps, main → run tests, confirm pass
- [ ] Step 3: `git commit -m "feat(api): GET /api/standings (B-slice3 task 1)"`

---

### Task 2: Closer Monitor endpoint

**Files:** Create `api/contracts/closers.py`, `api/services/closers_service.py`, `api/routers/closers.py`, `tests/api/test_closers.py`; Modify `api/deps.py`, `api/main.py`.

**Contract (implement EXACTLY; reuse `PlayerRef`):**
```python
# api/contracts/closers.py
"""Contract models for the Closer Monitor page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class CloserEntry(BaseModel):
    team: str
    closer: PlayerRef | None = None
    role: str = ""           # e.g. "Closer", "Committee", "Setup"
    confidence: str = ""     # e.g. "Firm", "Shaky"
    handcuffs: list[PlayerRef] = []


class ClosersResponse(BaseModel):
    entries: list[CloserEntry]
```

**Service:** mirror `fa_service.py`. Call `src.closer_monitor.build_closer_grid(...)` — INSPECT its signature + output at build time and map each team's row to a `CloserEntry` (nested `PlayerRef` for the closer + each handcuff). Resilient: cold env → empty list.

**Router:** `GET /api/closers` → `ClosersResponse`. **DI + main.py:** add `get_closer_service` + include router.

**Test:** contract-shape + endpoint test with a fake `CloserService` returning one `CloserEntry` (with a nested `PlayerRef` closer).

- [ ] Step 1: failing tests → confirm fail
- [ ] Step 2: implement → confirm pass
- [ ] Step 3: `git commit -m "feat(api): GET /api/closers (B-slice3 task 2)"`

---

### Task 3: Leaders endpoint

**Files:** Create `api/contracts/leaders.py`, `api/services/leaders_service.py`, `api/routers/leaders.py`, `tests/api/test_leaders.py`; Modify `api/deps.py`, `api/main.py`.

**Contract (implement EXACTLY; reuse `PlayerRef`):**
```python
# api/contracts/leaders.py
"""Contract models for the Leaders page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class LeaderRow(BaseModel):
    rank: int
    player: PlayerRef
    value: float


class LeadersResponse(BaseModel):
    category: str
    rows: list[LeaderRow]
```

**Service:** `get_leaders(self, category: str, limit: int = 25)`. Mirror `fa_service.py`. INSPECT `src/leaders.py` at build time for the right function (e.g. a leaders/top-N-by-category accessor) and the player pool; build a `LeaderRow` per player (nested `PlayerRef`, `value` = the category stat). Resilient: cold env / unknown category → empty `rows`.

**Router:** `GET /api/leaders?category=HR&limit=25` → `LeadersResponse`. **DI + main.py:** add `get_leaders_service` + include router.

**Test:** contract-shape + endpoint test with a fake `LeadersService` returning a `LeadersResponse` for `category="HR"`, asserting `body["rows"][0]["player"]["name"]`.

- [ ] Step 1: failing tests → confirm fail
- [ ] Step 2: implement → confirm pass
- [ ] Step 3: `git commit -m "feat(api): GET /api/leaders (B-slice3 task 3)"`

---

### Task 4: Regenerate OpenAPI + full verification

- [ ] Step 1: `python scripts/export_openapi.py` (adds the 3 new paths to `api/openapi.json`)
- [ ] Step 2 — verify:
  - `python -m pytest tests/api/ -v` → ALL pass (Slices 1–3 + the structural guards + openapi snapshot)
  - `python -m ruff format api/ scripts/ tests/api/ && python -m ruff check api/ scripts/ tests/api/` → clean
  - `git diff --name-only master -- src/` → empty (engines untouched)
- [ ] Step 3: `git commit -m "chore(api): regenerate OpenAPI for B-slice3 endpoints (task 4)"`

---

## Self-Review
- **Spec coverage:** these are the read endpoints for League Standings, Closer Monitor, and Leaders (Sub-project C surface), same safe additive pattern as Slices 1–2. ✓
- **Placeholders:** the three contracts are given in full; services/routers/tests mirror the concrete on-disk `free_agents` template; the engine-output mappings are best-effort integration seams confined to each service (the proven Slice-1/2 pattern) and bypassed by the dependency-override tests. ✓
- **Type consistency:** reuses `PlayerRef` from `api.contracts.common`; new models + `get_*_service` providers + `*Service` classes are referenced identically across each task. ✓
- **Invariants:** routers stay logic-free (existing guard auto-covers them); `src/` untouched (Task 4 asserts); OpenAPI regenerated + snapshot-guarded.

## Leaves for later
Remaining page endpoints (Matchup Planner, Trade Analyzer, Trade Finder, Pitcher Streaming, Punt, Player Compare, Databank, Draft Sim) — same pattern, future slices. Postgres/workers/auth = B2/B3/B4 (deferred).
