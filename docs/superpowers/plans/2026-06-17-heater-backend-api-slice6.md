# HEATER Backend API — Slice 6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Add three read endpoints — `GET /api/trade-finder`, `GET /api/compare`, `GET /api/databank` — over the existing engines, using the contract-first, logic-free-router pattern Slices 1–5 established. This completes the in-season page read surface.

**Architecture:** Identical to prior slices. Contract (`api/contracts/`) + service (`api/services/`) + thin router (`api/routers/`) + DI provider + fake-service test. Mount each router in `api/main.py`.

**Tech Stack:** FastAPI · Pydantic v2 · pytest · existing `src/` engines (trade_finder, player pool, player_databank). SQLite (unchanged).

> **⚠ LESSON:** implement the contract models **EXACTLY as written below. Do NOT redesign them.** The frontend generates its types from the OpenAPI these produce. Reuse `api/contracts/common.py::PlayerRef`.

**Pattern templates (read first, mirror exactly):** `api/contracts/standings.py`, `api/services/standings_service.py`, `api/routers/standings.py`, `api/deps.py`, `api/main.py`, `tests/api/test_standings.py`. (Eleven endpoints already follow this shape — copy it.)

---

## File Structure

| File | Responsibility |
|---|---|
| `api/contracts/trade_finder.py` | `TradeSuggestion`, `TradeFinderResponse`. |
| `api/contracts/compare.py` | `ComparePlayer`, `CompareResponse`. |
| `api/contracts/databank.py` | `SeasonStat`, `DatabankResponse`. |
| `api/services/trade_finder_service.py` / `compare_service.py` / `databank_service.py` | engine seams. |
| `api/routers/trade_finder.py` / `compare.py` / `databank.py` | thin routers. |
| `api/deps.py` | ADD `get_trade_finder_service`, `get_compare_service`, `get_databank_service`. |
| `api/main.py` | ADD the three routers to `create_app()`. |
| `tests/api/test_trade_finder.py` / `test_compare.py` / `test_databank.py` | contract + endpoint tests (fake services). |
| `api/openapi.json` | regenerated. |

---

### Task 1: Trade Finder endpoint

**Contract (implement EXACTLY; reuse `PlayerRef`):**
```python
# api/contracts/trade_finder.py
"""Contract models for the Trade Finder page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class TradeSuggestion(BaseModel):
    partner_team: str
    giving: list[PlayerRef] = []      # players YOU give
    receiving: list[PlayerRef] = []   # players you receive
    net_sgp: float = 0.0
    rationale: str = ""


class TradeFinderResponse(BaseModel):
    team_name: str
    suggestions: list[TradeSuggestion] = []
```

**Service** (`TradeFinderService.get_suggestions(team_name, limit=10)`): mirror `standings_service.py`. Call `src.trade_finder.find_trade_opportunities(user_roster_ids, player_pool, config=LeagueConfig(), all_team_totals=None, league_rosters=..., max_results=limit)` — INSPECT its signature + output at build time; resolve `user_roster_ids`/`league_rosters` from the rosters frame and `player_pool` from `load_player_pool()`; map each opportunity → `TradeSuggestion` (nested `PlayerRef`s). Resilient: cold env / heavy-scan failure → empty `suggestions` (never a 500).

**Router:** `GET /api/trade-finder?team_name=...&limit=10` → `TradeFinderResponse`, thin. **DI + main.py:** add provider + mount router.

**Test:** contract-shape + endpoint test injecting a fake `TradeFinderService` returning one `TradeSuggestion`, asserting `body["suggestions"][0]["partner_team"]`.

- [ ] Step 1: failing tests → confirm fail · Step 2: implement → confirm pass · Step 3: `git commit -m "feat(api): GET /api/trade-finder (B-slice6 task 1)"`

---

### Task 2: Player Compare endpoint

**Contract (implement EXACTLY; reuse `PlayerRef`):**
```python
# api/contracts/compare.py
"""Contract models for the Player Compare page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class ComparePlayer(BaseModel):
    player: PlayerRef
    stats: dict[str, float] = {}      # category -> projected value


class CompareResponse(BaseModel):
    categories: list[str] = []        # the categories compared, in order
    players: list[ComparePlayer] = []
```

**Service** (`CompareService.compare(player_ids: list[int])`): mirror the pattern. Load `player_pool = load_player_pool()`, slice to the requested ids, and build a `ComparePlayer` per id (nested `PlayerRef` + a `stats` dict keyed by the league's categories, values pulled from the pool's stat columns). Use `LeagueConfig().all_categories` for the `categories` list. Resilient: unknown ids → skipped; cold env → empty.

**Router:** `GET /api/compare?ids=1,2,3` → `CompareResponse`. Parse the comma-separated `ids` query param to `list[int]` IN THE ROUTER ONLY as trivial parsing (a `.split(",")` + `int()` is acceptable transport-layer parsing, NOT analytics — keep all real work in the service). **DI + main.py:** add + mount.

**Test:** contract-shape + endpoint test with a fake `CompareService` returning two `ComparePlayer`s, asserting `body["players"][0]["player"]["name"]` and `body["categories"]`.

- [ ] Step 1: failing tests → confirm fail · Step 2: implement → confirm pass · Step 3: `git commit -m "feat(api): GET /api/compare (B-slice6 task 2)"`

---

### Task 3: Player Databank endpoint

**Contract (implement EXACTLY; reuse `PlayerRef`):**
```python
# api/contracts/databank.py
"""Contract models for the Player Databank page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class SeasonStat(BaseModel):
    year: int
    stats: dict[str, float] = {}


class DatabankResponse(BaseModel):
    player: PlayerRef
    seasons: list[SeasonStat] = []    # newest first
```

**Service** (`DatabankService.get_player(player_id: int)`): mirror the pattern. INSPECT `src/player_databank.py` at build time for the historical multi-year lookup (e.g. a per-season stats accessor); build a `SeasonStat` per season (year + stats dict) and the player's `PlayerRef`. Resilient: unknown id / cold env → empty `seasons` with a best-effort `PlayerRef` (id only if name unknown).

**Router:** `GET /api/databank?player_id=123` → `DatabankResponse`. **DI + main.py:** add + mount.

**Test:** contract-shape + endpoint test with a fake `DatabankService` returning a `DatabankResponse` with one `SeasonStat`, asserting `body["seasons"][0]["year"]`.

- [ ] Step 1: failing tests → confirm fail · Step 2: implement → confirm pass · Step 3: `git commit -m "feat(api): GET /api/databank (B-slice6 task 3)"`

---

### Task 4: Regenerate OpenAPI + full verification

- [ ] Step 1: `python scripts/export_openapi.py` (adds the 3 new paths to `api/openapi.json`)
- [ ] Step 2 — verify:
  - `python -m pytest tests/api/ -v` → ALL pass (Slices 1–6 + structural guards + openapi snapshot)
  - `python -m ruff format api/ scripts/ tests/api/ && python -m ruff check api/ scripts/ tests/api/` → clean
  - `git diff --name-only origin/master -- src/` → empty (engines untouched)
- [ ] Step 3: `git commit -m "chore(api): regenerate OpenAPI for B-slice6 endpoints (task 4)"`

---

## Self-Review
- **Spec coverage:** read endpoints for Trade Finder, Player Compare, Player Databank — completes the in-season page read surface, same safe additive pattern as Slices 1–5. ✓
- **Placeholders:** the three contracts are given in full; services/routers/tests mirror the on-disk `standings` template; engine mappings are best-effort seams confined to each service (proven pattern) and bypassed by the dependency-override tests. ✓
- **Type consistency:** reuses `PlayerRef`; new models + `get_*_service` + `*Service` classes referenced identically per task. ✓
- **Invariants:** routers stay logic-free (existing guard auto-covers them; the `/compare` ids `.split(",")` is trivial transport parsing, not analytics); `src/` untouched (Task 4 asserts); OpenAPI regenerated + snapshot-guarded.

## Leaves for later
- **Draft Simulator** — deferred: it's an interactive, stateful preseason tool (AI opponents, live MC pick recommendations), NOT a single read; needs its own stateful API design (start sim / make pick / get rec), not this slice's GET pattern.
- B2 Postgres / B3 workers (trade MC job) / B4 auth+multi-tenancy / B5 strangler proof — deferred per owner.
