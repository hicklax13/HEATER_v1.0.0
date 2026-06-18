# HEATER Backend API — Slice 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Add three read endpoints — `GET /api/matchup`, `GET /api/streaming`, `GET /api/punt` — over the existing engines, using the contract-first, logic-free-router pattern Slices 1–3 established.

**Architecture:** Identical to Slices 1–3. Each endpoint = a Pydantic contract (`api/contracts/`) + a service that calls the engine and maps to the contract (`api/services/`) + a thin router (`api/routers/`) + a DI provider (`api/deps.py`) + a fake-service test (`tests/api/`). Mount each router in `api/main.py`.

**Tech Stack:** FastAPI · Pydantic v2 · pytest · existing `src/` engines (matchup, stream_analyzer, punt). SQLite (unchanged).

> **⚠ LESSON — read before coding:** implement the contract models **EXACTLY as written below. Do NOT redesign them** — no flattening `PlayerRef` to strings, no renaming/removing fields, no making optionals required. The frontend generates its TypeScript types from the OpenAPI these produce. Reuse `api/contracts/common.py::PlayerRef`.

**Pattern templates (read first, mirror exactly):** `api/contracts/standings.py`, `api/services/standings_service.py`, `api/routers/standings.py`, `api/deps.py`, `api/main.py`, `tests/api/test_standings.py`. (Every endpoint added in Slices 1–3 follows this shape — copy it.)

---

## File Structure

| File | Responsibility |
|---|---|
| `api/contracts/matchup.py` | `MatchupCategory`, `MatchupResponse`. |
| `api/contracts/streaming.py` | `StreamCandidate`, `StreamingResponse`. |
| `api/contracts/punt.py` | `PuntCategory`, `PuntResponse`. |
| `api/services/matchup_service.py` / `streaming_service.py` / `punt_service.py` | engine seams. |
| `api/routers/matchup.py` / `streaming.py` / `punt.py` | thin routers. |
| `api/deps.py` | ADD `get_matchup_service`, `get_streaming_service`, `get_punt_service`. |
| `api/main.py` | ADD the three routers to `create_app()`. |
| `tests/api/test_matchup.py` / `test_streaming.py` / `test_punt.py` | contract + endpoint tests (fake services). |
| `api/openapi.json` | regenerated. |

---

### Task 1: Matchup Planner endpoint

**Contract (implement EXACTLY):**
```python
# api/contracts/matchup.py
"""Contract models for the Matchup Planner page."""

from __future__ import annotations

from pydantic import BaseModel


class MatchupCategory(BaseModel):
    cat: str
    you: float
    opp: float
    win_prob: float
    inverse: bool = False


class MatchupResponse(BaseModel):
    team_name: str
    opponent: str = ""
    week: int = 0
    projected_cat_wins: float = 0.0
    win_prob: float = 0.0          # P(win the weekly matchup overall)
    categories: list[MatchupCategory] = []
```

**Service** (`MatchupService.get_matchup(team_name)`): mirror `standings_service.py` (lazy `src` imports, try/except → empty `categories` on cold env). Source the weekly matchup via the Yahoo data service + the matchup engine (INSPECT `src/matchup_context.py` / `src/matchup_planner.py` / `get_yahoo_data_service().get_matchup()` at build time); map each category to `MatchupCategory` and the overall `win_prob`. Best-effort seam; flag uncertainty.

**Router:** `GET /api/matchup?team_name=...` → `MatchupResponse`, thin, `service=Depends(get_matchup_service)`. **DI + main.py:** add provider + include router.

**Test:** contract-shape + endpoint test injecting a fake `MatchupService` via `app.dependency_overrides` (one `MatchupCategory`), asserting `body["categories"][0]["cat"]`.

- [ ] Step 1: failing tests → confirm fail · Step 2: implement → confirm pass · Step 3: `git commit -m "feat(api): GET /api/matchup (B-slice4 task 1)"`

---

### Task 2: Pitcher Streaming endpoint

**Contract (implement EXACTLY; reuse `PlayerRef`):**
```python
# api/contracts/streaming.py
"""Contract models for the Pitcher Streaming page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class StreamCandidate(BaseModel):
    player: PlayerRef
    team: str = ""
    opponent: str = ""
    score: float = 0.0
    actionable: bool = True
    status: str = ""        # e.g. "OK", "LOCKED", "FINAL"
    reason: str = ""


class StreamingResponse(BaseModel):
    date: str
    candidates: list[StreamCandidate]
```

**Service** (`StreamingService.get_streaming(date=None, limit=25)`): mirror the pattern. Call `src.optimizer.stream_analyzer.build_stream_board(...)` (returns a DataFrame) — INSPECT its signature + columns at build time (needs an optimizer context + a target date; use `src.game_day.get_target_game_date()` for the default date). Map each row → `StreamCandidate` (nested `PlayerRef`). Resilient: cold env → empty `candidates`.

**Router:** `GET /api/streaming?date=&limit=25` → `StreamingResponse`. **DI + main.py:** add + include.

**Test:** contract-shape + endpoint test with a fake `StreamingService` returning one `StreamCandidate` (nested `PlayerRef`), asserting `body["candidates"][0]["player"]["name"]`.

- [ ] Step 1: failing tests → confirm fail · Step 2: implement → confirm pass · Step 3: `git commit -m "feat(api): GET /api/streaming (B-slice4 task 2)"`

---

### Task 3: Punt Analyzer endpoint

**Contract (implement EXACTLY):**
```python
# api/contracts/punt.py
"""Contract models for the Punt Analyzer page."""

from __future__ import annotations

from pydantic import BaseModel


class PuntCategory(BaseModel):
    cat: str
    current_rank: int
    gainable: bool
    recommendation: str = ""    # e.g. "Punt", "Contend", "Hold"


class PuntResponse(BaseModel):
    team_name: str
    punt_candidates: list[str] = []     # category names recommended to punt
    categories: list[PuntCategory] = []
```

**Service** (`PuntService.get_punt(team_name)`): mirror the pattern. INSPECT the punt logic (`pages/10_Punt_Analyzer.py` + `src/standings_utils.py`/`standings_engine.py`; CLAUDE.md: punt detection requires `gainable_positions == 0 AND rank >= 10`). Compute each category's `current_rank`/`gainable`/`recommendation` and the `punt_candidates` list. Resilient: cold env → empty lists.

**Router:** `GET /api/punt?team_name=...` → `PuntResponse`. **DI + main.py:** add + include.

**Test:** contract-shape + endpoint test with a fake `PuntService` returning one `PuntCategory` + a `punt_candidates` entry, asserting `body["punt_candidates"]`.

- [ ] Step 1: failing tests → confirm fail · Step 2: implement → confirm pass · Step 3: `git commit -m "feat(api): GET /api/punt (B-slice4 task 3)"`

---

### Task 4: Regenerate OpenAPI + full verification

- [ ] Step 1: `python scripts/export_openapi.py` (adds the 3 new paths to `api/openapi.json`)
- [ ] Step 2 — verify:
  - `python -m pytest tests/api/ -v` → ALL pass (Slices 1–4 + structural guards + openapi snapshot)
  - `python -m ruff format api/ scripts/ tests/api/ && python -m ruff check api/ scripts/ tests/api/` → clean
  - `git diff --name-only origin/master -- src/` → empty (engines untouched)
- [ ] Step 3: `git commit -m "chore(api): regenerate OpenAPI for B-slice4 endpoints (task 4)"`

---

## Self-Review
- **Spec coverage:** read endpoints for Matchup Planner, Pitcher Streaming, Punt Analyzer (Sub-project C surface), same safe additive pattern as Slices 1–3. ✓
- **Placeholders:** the three contracts are given in full; services/routers/tests mirror the concrete on-disk `standings`/`free_agents` template; engine mappings are best-effort seams confined to each service (proven pattern) and bypassed by the dependency-override tests. ✓
- **Type consistency:** reuses `PlayerRef`; new models + `get_*_service` + `*Service` classes referenced identically per task. ✓
- **Invariants:** routers stay logic-free (existing guard auto-covers them); `src/` untouched (Task 4 asserts); OpenAPI regenerated + snapshot-guarded.

## Leaves for later
Trade Analyzer (`POST /api/trade/evaluate`) + Trade Finder (their own slice — heavier engines, MC opt-in, async in B3), Player Compare, Player Databank, Draft Simulator. Postgres/workers/auth = B2/B3/B4 (deferred per owner).
