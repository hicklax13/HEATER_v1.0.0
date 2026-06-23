# Pre-launch Hardening Round 2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the buildable-now MED/LOW pre-launch backlog (`Outstanding_June26.md`) + the successful-empty→mock frontend fast-follow, without touching owner-gated M4/M5 infra or the live Streamlit/`src/` engine math.

**Architecture:** 8 sequential units. Backend first (observability, write isolation, tests), then contracts (type-design + openapi regen), then the frontend re-syncs to the contracts and applies the behavioral fixes (empty-fix, OpsCards, error states, identity, a11y), then docs. Each unit is additive and preserves current behavior except where a fix is the explicit goal.

**Tech Stack:** FastAPI + Pydantic (`api/`), pytest (`tests/api/`), Next.js 16 + React 19 + TypeScript + Tailwind (`web/`), `openapi-typescript` codegen.

**Spec:** `docs/superpowers/specs/2026-06-22-heater-prelaunch-round2-hardening-design.md`

---

## File structure (what changes)

**Backend (api/, src/):**
- `api/services/{fa,leaders,standings,streaming,closers,punt,draft,team}_service.py` — add module loggers + `logger.warning` at swallow sites (U1); `team_service` wraps the two Yahoo calls (U1).
- `api/services/matchup_service.py` — log its two swallow sites (U1); `Literal`/`StatItem` mapper updates (U4).
- `src/ai/keys.py` — log Fernet decrypt failures (U1).
- `api/services/roster_write_service.py` + `api/routers/roster_write.py` — Clerk-gated cross-team guard + traceback logging (U2).
- `api/contracts/{streaming,matchup,my_team,playoff,chat,common}.py` — Literal/structured types (U4).
- `api/openapi.json` — regenerated (U4).

**Tests:**
- `tests/api/test_service_logging.py` — new (U1).
- `tests/api/test_api_roster_write_guard.py` — new (U2).
- `tests/api/test_punt.py`, `tests/api/test_standings.py` — extend (U3).
- `tests/api/test_live_boxscore.py` — new (U3).
- `tests/api/test_contracts_typedesign.py` — new (U4).

**Frontend (web/):**
- `web/src/lib/api/generated.ts` — regenerated (U5).
- `web/src/lib/api/adapters.ts`, `web/src/lib/types.ts` — sync to U4 + OpsCards formatter (U5).
- `web/src/lib/{standings,players,streaming,closers,probables,punt,hitter-matchups,compare,trades,research}-data.ts`, `web/src/lib/data.ts` — empty→null (U5).
- `web/src/components/myteam/OpsCards.tsx` — `fmtOpsNum` (U5).
- `web/src/lib/use-draft.ts`, `web/src/app/draft/page.tsx`, `web/src/lib/databank-data.ts`, `web/src/app/databank/page.tsx` — error states (U6).
- `web/src/components/chrome/TopBar.tsx` — Clerk identity (U6).
- `web/src/app/matchup/page.tsx`, `web/src/components/streaming/AnalyzeStarter.tsx` — a11y (U7).

**Docs:** `CLAUDE.md` (U8).

---

## Conventions (read once)

- **House logging pattern:** `logger.warning("ServiceName.method failed: %s", exc)` — exception object only (NOT `exc_info`), except the write path (U2) which uses `exc_info=True`.
- **DB-free tests:** the worktree/CI DB is empty (`reference_worktree_empty_db`). Never rely on real data. Monkeypatch the engine seam to raise, or build synthetic inputs.
- **Run a single api test:** `python -m pytest tests/api/test_x.py::test_name -v`
- **Run the api suite:** `python -m pytest tests/api/ -q`
- **Web typecheck/build (from `web/`):** `pnpm exec tsc --noEmit` ; `pnpm build` ; codegen `pnpm gen:api`.
- **Literal-ize safely:** when a contract field becomes a `Literal`, the mapper MUST coerce unknown raw values to the default (e.g. `status = raw if raw in _ALLOWED else ""`) so an unexpected engine value degrades gracefully instead of raising a Pydantic ValidationError → 500.

---

## Task 1 (U1): Backend observability sweep

**Files:**
- Modify: `api/services/fa_service.py`, `leaders_service.py`, `standings_service.py`, `streaming_service.py`, `closers_service.py`, `punt_service.py`, `draft_service.py`, `matchup_service.py`, `team_service.py`, `src/ai/keys.py`
- Test: `tests/api/test_service_logging.py` (new)

**Pattern A — add a module logger** (top of file, after imports, only where one doesn't exist):
```python
import logging

logger = logging.getLogger(__name__)
```

**Pattern B — log before degrading** (at each swallow site; keep the existing graceful return):
```python
except Exception as exc:
    logger.warning("FreeAgentService.get_free_agents failed: %s", exc)
    recs = []   # <-- existing graceful fallback, unchanged
```

**Swallow sites (verified 2026-06-22 — Read each file first; line numbers may have shifted):**

| File | Method / site | Message string |
|---|---|---|
| `fa_service.py` | `get_free_agents` (~line 30) | `"FreeAgentService.get_free_agents failed: %s"` |
| `leaders_service.py` | `get_leaders` (~82) | `"LeadersService.get_leaders failed: %s"` |
| `standings_service.py` | `get_standings` (~21) | `"StandingsService.get_standings failed: %s"` |
| `streaming_service.py` | board build (~161) | `"StreamingService.get_streaming failed: %s"` |
| `streaming_service.py` | `analyze_pitcher` (~255, `found=False`) | `"StreamingService.analyze_pitcher failed: %s"` |
| `closers_service.py` | `get_closers` (~28) | `"CloserService.get_closers failed: %s"` |
| `punt_service.py` | `get_punt` (~61) | `"PuntService.get_punt failed: %s"` |
| `draft_service.py` | recommend invalid-state (~38) | `"DraftService.recommend: invalid draft state: %s"` |
| `draft_service.py` | recommend engine (~49) | `"DraftService.recommend engine failed: %s"` |
| `draft_service.py` | simulate_picks (~79) | `"DraftService.simulate_picks failed: %s"` |
| `matchup_service.py` | `get_matchup` (~315) — HAS logger already | `"MatchupService.get_matchup failed: %s"` |
| `matchup_service.py` | `_build_roster_tables` (~347, `pass`) | `"MatchupService._build_roster_tables failed: %s"` |
| `src/ai/keys.py` | get_key decrypt (~70) | `"keys.get_key: decrypt failed for user_id=%s provider=%s: %s"` (args: user_id, provider, exc) |
| `src/ai/keys.py` | admin_shared_key decrypt (~147) | `"keys.get_admin_shared_key: decrypt failed for provider=%s: %s"` |

**`draft_service.py` sites ~49/79:** keep the existing user-facing summary string (behavior unchanged) — only add the log line.

**`team_service.py` LOW-5 wrap** (`get_my_team`, ~lines 103-104): the two Yahoo calls sit outside any try/except. Wrap them so a Yahoo-singleton failure degrades instead of 500ing the dashboard:
```python
try:
    raw_matchup = yds.get_matchup()
except Exception as exc:
    logger.warning("TeamService.get_my_team: get_matchup failed: %s", exc)
    raw_matchup = None
try:
    standings = yds.get_standings()
except Exception as exc:
    logger.warning("TeamService.get_my_team: get_standings failed: %s", exc)
    standings = None
```
(Downstream code already guards `None` — verify it still does after the edit.)

- [ ] **Step 1: Write the failing tests** — `tests/api/test_service_logging.py`. One test per service: construct the service, monkeypatch the engine/Yahoo seam it calls inside the try to raise, assert (a) the graceful empty/`found=False` shape is returned AND (b) a WARNING containing the service name + "failed" was logged. Read each service to find the exact symbol to monkeypatch (the first `src.`/`get_yahoo_data_service`/engine call inside the try). Template (adapt the seam + return assertion per service):

```python
import logging
import pytest


def _raise(*a, **k):
    raise RuntimeError("boom")


def test_fa_service_logs_on_failure(monkeypatch, caplog):
    from api.services import fa_service
    svc = fa_service.FreeAgentService()
    # monkeypatch the engine call inside the try (read fa_service.py for the exact symbol)
    monkeypatch.setattr(fa_service, "build_optimizer_context", _raise, raising=False)
    monkeypatch.setattr(fa_service, "recommend_fa_moves", _raise, raising=False)
    with caplog.at_level(logging.WARNING):
        out = svc.get_free_agents(...)  # use the real method signature / minimal args
    assert out.recs == []
    assert any("FreeAgentService" in r.message and "failed" in r.message
               for r in caplog.records)
```

Repeat for leaders, standings, streaming(get_streaming + analyze_pitcher), closers, punt, draft(recommend + simulate_picks), matchup(get_matchup + _build_roster_tables), team(get_my_team → assert no raise + None-degrade), and `src/ai/keys.py` (monkeypatch `_decrypt` to raise; assert `get_key`/`get_admin_shared_key` return None AND log a WARNING).

- [ ] **Step 2: Run tests, verify they FAIL** — `python -m pytest tests/api/test_service_logging.py -q` → FAIL (no WARNING captured yet).

- [ ] **Step 3: Apply Pattern A + B** at every site in the table above; apply the `team_service` wrap.

- [ ] **Step 4: Run tests, verify PASS** — `python -m pytest tests/api/test_service_logging.py -q` → PASS. Then `python -m pytest tests/api/ -q` → no regressions.

- [ ] **Step 5: Commit**
```bash
git add api/services/*.py src/ai/keys.py tests/api/test_service_logging.py
git commit -m "fix(api): log service swallow sites + wrap team_service Yahoo calls (MED-2/LOW-3/LOW-5/LOW-6)"
```

**Reviewer after this task:** `pr-review-toolkit:silent-failure-hunter` on the diff.

---

## Task 2 (U2): Write isolation + traceback

**Files:**
- Modify: `api/services/roster_write_service.py`, `api/routers/roster_write.py`
- Test: `tests/api/test_api_roster_write_guard.py` (new)

**Design:** when Clerk is live, refuse a write whose caller team ≠ the token-owner team. When Clerk is off (dormant), no-op (byte-for-byte unchanged).

**Router change** (`api/routers/roster_write.py`) — add the viewer context and pass the caller's team to the service:
```python
from api.tenancy import require_viewer_context, ViewerContext

@router.post("/lineup/set", ...)
def set_lineup(req: LineupSetRequest,
               ctx: ViewerContext = Depends(require_viewer_context),
               service=Depends(get_roster_write_service)) -> MutationResult:
    return service.set_lineup(req, caller_team=ctx.team_name)

@router.post("/transactions/add-drop", ...)
def add_drop(req: AddDropRequest,
             ctx: ViewerContext = Depends(require_viewer_context),
             service=Depends(get_roster_write_service)) -> MutationResult:
    return service.add_drop(req, caller_team=ctx.team_name)
```
(Keep `dependencies=[Depends(require_principal)]` — the guard is defense-in-depth on top of auth.)

**Service change** (`api/services/roster_write_service.py`):
```python
import logging
from api.auth import clerk_configured  # the shared predicate (verify import path)

logger = logging.getLogger(__name__)

_FORBIDDEN = "Not authorized to modify another team's roster."

def _owner_team(self) -> str | None:
    try:
        from src.yahoo_data_service import get_yahoo_data_service
        return get_yahoo_data_service()._get_user_team_name()
    except Exception as exc:
        logger.warning("RosterWriteService._owner_team failed: %s", exc)
        return None

def _authorized(self, caller_team: str | None) -> bool:
    # Dormant single-owner path: no multi-tenant risk -> allow (unchanged behavior).
    if not clerk_configured():
        return True
    owner = self._owner_team()
    return bool(caller_team) and bool(owner) and caller_team == owner

def set_lineup(self, req, caller_team=None) -> MutationResult:
    if not self._authorized(caller_team):
        logger.warning("RosterWriteService.set_lineup refused: caller_team=%r", caller_team)
        return MutationResult(ok=False, error=_FORBIDDEN, status="forbidden")
    # ... existing body ...

def add_drop(self, req, caller_team=None) -> MutationResult:
    if not self._authorized(caller_team):
        logger.warning("RosterWriteService.add_drop refused: caller_team=%r", caller_team)
        return MutationResult(ok=False, error=_FORBIDDEN, status="forbidden")
    # ... existing body ...
```

**MED-3 traceback** — in the exception→MutationResult path (~line 52) and the `_to_result` passthrough:
```python
except Exception as exc:
    logger.warning("RosterWriteService.<method> failed: %s", exc, exc_info=True)
    return MutationResult(ok=False, error=f"Write failed: {type(exc).__name__}", status=None)
```
And in `_to_result`, if the result is `ok=False`, `logger.warning("RosterWriteService write returned ok=False: %s", result.error)`.

- [ ] **Step 1: Write failing tests** — `tests/api/test_api_roster_write_guard.py`. Use a fake/stubbed service or monkeypatch `clerk_configured` and `_owner_team`. Cases:
```python
import logging

def _svc():
    from api.services.roster_write_service import RosterWriteService
    return RosterWriteService()

def test_refuses_cross_team_when_clerk_on(monkeypatch):
    svc = _svc()
    monkeypatch.setattr("api.services.roster_write_service.clerk_configured", lambda: True)
    monkeypatch.setattr(svc, "_owner_team", lambda: "Team Hickey")
    out = svc.set_lineup(_lineup_req(), caller_team="Other Team")
    assert out.ok is False and out.status == "forbidden"

def test_refuses_unassigned_when_clerk_on(monkeypatch):
    svc = _svc()
    monkeypatch.setattr("api.services.roster_write_service.clerk_configured", lambda: True)
    monkeypatch.setattr(svc, "_owner_team", lambda: "Team Hickey")
    out = svc.add_drop(_adddrop_req(), caller_team=None)
    assert out.ok is False and out.status == "forbidden"

def test_allows_owner_when_clerk_on(monkeypatch):
    svc = _svc()
    monkeypatch.setattr("api.services.roster_write_service.clerk_configured", lambda: True)
    monkeypatch.setattr(svc, "_owner_team", lambda: "Team Hickey")
    monkeypatch.setattr(svc, "_client", lambda: _fake_client_ok())  # stub the Yahoo client
    out = svc.set_lineup(_lineup_req(), caller_team="Team Hickey")
    assert out.ok is True  # proceeds to (stubbed) write

def test_dormant_clerk_off_is_noop(monkeypatch):
    svc = _svc()
    monkeypatch.setattr("api.services.roster_write_service.clerk_configured", lambda: False)
    monkeypatch.setattr(svc, "_client", lambda: _fake_client_ok())
    out = svc.set_lineup(_lineup_req(), caller_team="anything-or-none")
    assert out.ok is True  # guard skipped, behaves exactly as today

def test_exception_logs_traceback(monkeypatch, caplog):
    svc = _svc()
    monkeypatch.setattr("api.services.roster_write_service.clerk_configured", lambda: False)
    monkeypatch.setattr(svc, "_client", _raise)
    with caplog.at_level(logging.WARNING):
        out = svc.set_lineup(_lineup_req(), caller_team=None)
    assert out.ok is False
    assert any("RosterWriteService" in r.message for r in caplog.records)
```
Build `_lineup_req()`/`_adddrop_req()` from the real `LineupSetRequest`/`AddDropRequest` shapes (read the contracts). `_fake_client_ok()` returns an object whose `set_lineup`/`add_drop` return a success-shaped result.

- [ ] **Step 2: Run, verify FAIL** — `python -m pytest tests/api/test_api_roster_write_guard.py -q`.
- [ ] **Step 3: Implement** the router + service changes above. Verify the `clerk_configured` import path (it's the same predicate used by `get_auth_verifier`); if it lives in `api/auth.py`, import from there.
- [ ] **Step 4: Run, verify PASS** + `python -m pytest tests/api/ -q` (incl. `test_no_logic_in_routers.py` — the authz logic is in the SERVICE, so this must stay green).
- [ ] **Step 5: Commit**
```bash
git add api/services/roster_write_service.py api/routers/roster_write.py tests/api/test_api_roster_write_guard.py
git commit -m "fix(api): Clerk-gated cross-team write guard + traceback logging (MED-1/MED-3)"
```

**Reviewer after this task:** `security-review` (Skill) — focus on the auth/isolation boundary + the dormant-path no-op.

---

## Task 3 (U3): Real service test coverage

**Files:**
- Test: `tests/api/test_punt.py` (extend), `tests/api/test_standings.py` (extend), `tests/api/test_live_boxscore.py` (new)
- Modify (only if a test surfaces a real bug): the corresponding service.

**Approach:** static-method / pure-function tests with synthetic inputs (mirror `tests/api/test_lineup.py`), DB-free.

- [ ] **Step 1: PuntService tests** — read `api/services/punt_service.py:16-69`; identify the pure mapping helper(s) (rank/gainable extraction, `is_punt` branch). If `get_punt` calls an engine, monkeypatch it to return a synthetic structure and assert the contract mapping (categories, is_punt true/false boundary, gainable extraction). Write 3-4 assertions covering: a punt category (rank≥10 AND gainable==0), a non-punt category, and the empty-engine path → empty contract.

- [ ] **Step 2: StandingsService tests** — read `api/services/standings_service.py:25-99`; test `_build_teams` (or the static parsing helpers) directly with a synthetic standings DataFrame/dict: a `"W-L-T"` string splits to the right ints; an OVERALL/RECORD row is detected/handled; a NaN field is guarded (no raise, coerced). Assert the produced team rows.

- [ ] **Step 3: live_boxscore tests** — read `api/services/live_boxscore.py`; test `_hitter_line`/`_pitcher_line` with synthetic statsapi-shaped dicts: AVG/OBP math is correct, a missing field degrades, the line shape matches what the matchup page expects.

- [ ] **Step 4: Run** — `python -m pytest tests/api/test_punt.py tests/api/test_standings.py tests/api/test_live_boxscore.py -q` → PASS. If any test reveals a real mapping bug, fix the service minimally and note it in the commit body.

- [ ] **Step 5: Commit**
```bash
git add tests/api/test_punt.py tests/api/test_standings.py tests/api/test_live_boxscore.py
git commit -m "test(api): real-class coverage for Punt/Standings mappers + live_boxscore parsers (MED-4/MED-5)"
```

**Reviewer after this task:** `pr-review-toolkit:pr-test-analyzer`.

---

## Task 4 (U4): Type-design contract hardening + openapi regen

**Files:**
- Modify: `api/contracts/common.py`, `streaming.py`, `matchup.py`, `my_team.py`, `playoff.py`, `chat.py`; the mappers in `api/services/{streaming,matchup,team,playoff,chat}_service.py`.
- Regenerate: `api/openapi.json`
- Test: `tests/api/test_contracts_typedesign.py` (new)

**Prereq:** confirm `python -c "import fastapi; print(fastapi.__version__)"` == `0.137.1`.

**common.py — add shared `Record`** (StatItem already exists):
```python
class Record(BaseModel):
    wins: int = 0
    losses: int = 0
    ties: int = 0
```

**streaming.py:**
```python
from typing import Literal
status: Literal["", "PROBABLE", "LOCKED", "FINAL", "OPEN"] = ""
confidence: Literal["", "HIGH", "MEDIUM", "LOW"] = ""
ip_pace: float | None = None
```
Mapper `streaming_service.py:198-199` — coerce to the allowed set:
```python
_ALLOWED_STATUS = {"PROBABLE", "LOCKED", "FINAL", "OPEN"}
_ALLOWED_CONF = {"HIGH", "MEDIUM", "LOW"}
raw_status = str(g("status", "") or "")
status = raw_status if raw_status in _ALLOWED_STATUS else ""
raw_conf = str(g("confidence", "") or "")
confidence = raw_conf if raw_conf in _ALLOWED_CONF else ""
```
Mapper `streaming_service.py:61` — `ip_pace=None` (was `0.0`).

**matchup.py:**
```python
from api.contracts.common import StatItem, Record
class MatchPlayer(BaseModel):
    ...
    stats: list[StatItem] = []
class MatchupCategory(BaseModel):
    win: Literal["", "you", "opp"] = ""
class TeamSide(BaseModel):
    record: str = ""              # keep display string
    record_wlt: Record | None = None   # additive structured
# SideTotals stat lists -> list[StatItem] too
```
Mappers `_fmt_hitter_stats`/`_fmt_pitcher_stats` (matchup_service.py:116-139) — pair each value with its column label, returning `list[StatItem]`:
```python
def _fmt_hitter_stats(...) -> list[StatItem]:
    vals = [...7 formatted strings...]   # existing
    return [StatItem(label=c, value=v) for c, v in zip(_HITTER_COLUMNS, vals)]
```
`_team_side` (matchup_service.py:63-69) — set both `record` (existing f-string) and `record_wlt=Record(wins=w, losses=l, ties=t)`.

**my_team.py:**
```python
trend: Literal["up", "down", "flat"] = "flat"      # Mover
tag: Literal["hot", "cold", ""] = ""               # Mover
status: Literal["ok", "warn", "info"] = "info"     # StatusChip
status: Literal["ok", "warn", "danger"] = "ok"     # OpsCard
```
(Mappers in `team_service.py` already set known values; add coercion only if a value is a raw passthrough — verify lines 452-453, 462-465, 272/289/316 emit only allowed values.)

**playoff.py:** add `projected_record_wlt: Record | None = None` (keep `projected_record` string). Mapper `playoff_service.py:195` sets both.

**chat.py:**
```python
class ToolTraceEntry(BaseModel):
    name: str = ""
    args: dict = {}
tool_trace: list[ToolTraceEntry] = []
```
Mapper (chat service, where `tool_trace` is built ~adapters/chat_service) — wrap each dict: `ToolTraceEntry(name=d.get("name",""), args=d.get("args",{}))`. (Keep tolerant of already-dict input.)

- [ ] **Step 1: Write failing contract tests** — `tests/api/test_contracts_typedesign.py`:
```python
import pytest
from pydantic import ValidationError

def test_streaming_status_literal_rejects_unknown():
    from api.contracts.streaming import StreamCandidate
    with pytest.raises(ValidationError):
        StreamCandidate(..., status="BOGUS")  # minimal valid other fields

def test_streaming_status_allows_known():
    from api.contracts.streaming import StreamCandidate
    assert StreamCandidate(..., status="PROBABLE").status == "PROBABLE"

def test_matchup_stats_are_statitems():
    from api.contracts.matchup import MatchPlayer
    from api.contracts.common import StatItem
    mp = MatchPlayer(..., stats=[StatItem(label="HR", value="3")])
    assert mp.stats[0].label == "HR"

def test_record_struct_present():
    from api.contracts.common import Record
    assert Record(wins=4, losses=7, ties=1).losses == 7

def test_tool_trace_entry_typed():
    from api.contracts.chat import ToolTraceEntry
    assert ToolTraceEntry(name="web_search", args={"q": "x"}).name == "web_search"
```
Also a mapper-coercion test: feed the streaming mapper a synthetic row with `status="WEIRD"` and assert the contract gets `status == ""` (graceful, no raise).

- [ ] **Step 2: Run, verify FAIL** — `python -m pytest tests/api/test_contracts_typedesign.py -q`.
- [ ] **Step 3: Implement** the contract + mapper changes above. Read each contract/mapper first; preserve all other fields.
- [ ] **Step 4: Regenerate openapi + run full api suite**
```bash
python scripts/export_openapi.py
python -m pytest tests/api/ -q     # incl. test_openapi_contract.py (must be green) + test_contracts_typedesign.py
```
- [ ] **Step 5: Commit**
```bash
git add api/contracts/*.py api/services/*.py api/openapi.json tests/api/test_contracts_typedesign.py
git commit -m "refactor(api): Literal/structured contract types + mapper coercion + openapi regen (type-design)"
```

**Reviewer after this task:** `pr-review-toolkit:type-design-analyzer` + `feature-dev:code-reviewer`.

---

## Task 5 (U5): Frontend contract-sync + empty-fix + OpsCards

**Files:**
- Regenerate: `web/src/lib/api/generated.ts`
- Modify: `web/src/lib/api/adapters.ts`, `web/src/lib/types.ts`, `web/src/components/myteam/OpsCards.tsx`, and the fetchers listed below.

**Depends on Task 4 (committed openapi.json).**

- [ ] **Step 1: Regenerate types** — from `web/`: `pnpm gen:api`. Then `pnpm exec tsc --noEmit` to surface every consumer broken by U4 (this is the worklist).
- [ ] **Step 2: Fix adapters for U4 structured types** — `adapters.ts`:
  - matchup `stats`: read `{label,value}` objects instead of positional strings; the matchup page `StatCells` (U-note: matchup/page.tsx) reads `s.value`.
  - `ip_pace`: already null-safe (`Math.round(b?.ip_pace ?? 0)`) — confirm.
  - records: keep reading the display string; optionally consume `record_wlt`.
  - `tool_trace`: read `{name,args}` typed entries.
  - Literal fields (streaming status/confidence, my_team trend/tag/status): the existing lowercasing maps still work; confirm the union types line up.
- [ ] **Step 3: OpsCards formatter (MED-6)** — `web/src/components/myteam/OpsCards.tsx`:
```tsx
const fmtOpsNum = (n: number) => (Number.isInteger(n) ? String(n) : n.toFixed(1));
// in render: {fmtOpsNum(card.value)} / {fmtOpsNum(card.total)}
// ring math unchanged: Math.min(100, Math.round((card.value / card.total) * 100))
```
- [ ] **Step 4: Empty→null (decision 1)** — in each fetcher, change the empty false-branch from the MOCK constant to `null`. Files + current empty-branch:
  - `standings-data.ts` (~85-107): `... : STANDINGS` → `... : null`
  - `players-data.ts` (~79-91): `... : PLAYERS` → `... : null`
  - `streaming-data.ts` (~217-225): `... : STREAMING` → `... : null`
  - `closers-data.ts` (~83-91): `... : CLOSERS` → `... : null`
  - `probables-data.ts` (~217-225): `... : PROBABLES_MOCK` → `... : null`
  - `punt-data.ts` (~78-86): `... : PUNT` → `... : null`
  - `hitter-matchups-data.ts` (~209): `... : HITTER_MATCHUPS_MOCK` → `... : null`
  - `compare-data.ts` (~118): `... : mockCompare(players)` → `... : null`
  - `trades-data.ts` (~140-150): `... : mock()` → `... : null`
  - `research-data.ts` (~90-98): `... : RESEARCH` → `... : null`
  - `data.ts:fetchMyTeam` (~108-117): add a full-empty guard — if the adapted result has no movers AND no ops AND no categories → `return null`.
  - The function return types must allow `null` (e.g. `Promise<XData | null>`); `usePageData` already maps `null → empty`. Keep the MOCK constant for the `liveOrMock` `mock` arm (demo mode).
  - **Confirm each page renders the `empty` state.** If a page component doesn't handle `status === "empty"`, add an `EmptyState` render (Team page `app/page.tsx` is the reference).
- [ ] **Step 5: Verify** — from `web/`: `pnpm exec tsc --noEmit` (clean), `pnpm build` (green). Then **preview** (`reference_preview_tool_gotchas`): start the dev server; with live OFF confirm mock still renders (demo mode unchanged); simulate an empty live response (or point at the live API for a page that's currently empty) and confirm the `empty` state renders, NOT fabricated data; confirm OpsCards show `18.2 / 53.8` style. Capture a screenshot/snapshot as evidence.
- [ ] **Step 6: Commit**
```bash
git add web/src/lib web/src/components/myteam/OpsCards.tsx
git commit -m "fix(web): live-never-fabricates empty state + contract sync + OpsCards rounding (fast-follow/MED-6)"
```

**Reviewer after this task:** `feature-dev:code-reviewer`.

---

## Task 6 (U6): Frontend error states + identity

**Files:**
- Modify: `web/src/lib/use-draft.ts`, `web/src/app/draft/page.tsx`, `web/src/lib/databank-data.ts`, `web/src/app/databank/page.tsx`, `web/src/components/chrome/TopBar.tsx`

- [ ] **Step 1: Draft error phase (MED-7)** — `use-draft.ts`: add `"error"` to the phase union; in the `start()` catch, on a non-402 error set `phase:"error"` (keep 402→`locked`); in the `pick()` catch, on a non-402 error set `phase:"error"` (don't leave a silent optimistic pick). `draft/page.tsx`: render the `error` phase with a retry that calls `start(config)` again.
- [ ] **Step 2: Databank error (MED-7)** — `databank-data.ts`: distinguish outage from "no history". Simplest: let the fetcher rethrow on error (remove the bare `catch { return null }`) so the page can tell empty (`null` from a 2xx with no seasons) from error. `databank/page.tsx`: wrap the fetch; on throw show an error/retry card; keep "No history found" only for a successful-but-empty result.
- [ ] **Step 3: TopBar identity (LOW-1)** — `TopBar.tsx:172-185`: replace the literals. Use `useUser()` from `@clerk/nextjs` for the display name (fallback to a neutral label when signed-out/Clerk-off) and `getViewerTeam()` (`web/src/lib/viewer-team.ts`) for the team; derive initials from the name. Render must be safe when Clerk is not configured (no crash, neutral fallback).
- [ ] **Step 4: Verify** — `pnpm exec tsc --noEmit`, `pnpm build`, preview: trigger a draft start failure → error+retry shows; databank outage vs no-history render differently; TopBar shows the resolved name/team (or neutral fallback). Capture evidence.
- [ ] **Step 5: Commit**
```bash
git add web/src/lib/use-draft.ts web/src/app/draft web/src/lib/databank-data.ts web/src/app/databank web/src/components/chrome/TopBar.tsx
git commit -m "fix(web): Draft/Databank error states + TopBar Clerk identity (MED-7/LOW-1)"
```

**Reviewer after this task:** `feature-dev:code-reviewer`.

---

## Task 7 (U7): Accessibility (Matchup-first)

**Files:**
- Modify: `web/src/app/matchup/page.tsx`, `web/src/components/streaming/AnalyzeStarter.tsx` (+ optional `StandingsTable.tsx`, `ComparePanel.tsx`, `web/src/app/layout.tsx`).

- [ ] **Step 1: Non-color winner cue** — `matchup/page.tsx` (~the `bg-heat/10` per-category winner, ~line 226): add a non-color indicator (a ▲/✓ glyph or "WIN" text + `aria-label`) so the winner is distinguishable without color and announced to screen readers.
- [ ] **Step 2: Real date tabs** — `matchup/page.tsx` (~143-153): convert the inert `<span>` date tabs to `<button type="button">` with keyboard focus + `aria-current`/`aria-selected`; wire the existing click/select behavior.
- [ ] **Step 3: Label the Analyze select** — `AnalyzeStarter.tsx:67-77`: add a `<label htmlFor>` (or `aria-label`) to the `<select>`.
- [ ] **Step 4 (optional, if low-risk):** add `scope="col"` / `<th>` to `StandingsTable.tsx` + a `<thead>` to `ComparePanel.tsx`; add a skip-to-content link in `layout.tsx`.
- [ ] **Step 5: Verify** — `pnpm exec tsc --noEmit`, `pnpm build`, preview: Tab-navigate the matchup date tabs (focus visible, Enter switches date), the winner cue is visible in a grayscale check, the select is labeled. Capture evidence.
- [ ] **Step 6: Commit**
```bash
git add web/src/app/matchup web/src/components/streaming/AnalyzeStarter.tsx
git commit -m "fix(web): matchup a11y (non-color winner cue + keyboard date tabs) + label Analyze select (MED-8)"
```

**Reviewer after this task:** `feature-dev:code-reviewer`.

---

## Task 8 (U8): Docs

**Files:** Modify `CLAUDE.md`.

- [ ] **Step 1: Fix counts** — update the Sub-project B status line: "23 endpoints"→"26", "6 require_pro"/"6 compute-heavy"→"7", and note `POST /api/draft/grade` exists. (Search CLAUDE.md for "23 `/api/*`" and "6 compute-heavy".)
- [ ] **Step 2: Commit**
```bash
git add CLAUDE.md
git commit -m "docs(claude): correct endpoint/require_pro counts (23->26, 6->7, note draft/grade)"
```

---

## Final: integration verify + merge

- [ ] `python -m pytest tests/api/ -q` green; `python -m pytest tests/ -q -k "structural or no_logic or openapi" ` (structural guards) green.
- [ ] From `web/`: `pnpm exec tsc --noEmit` clean; `pnpm build` green.
- [ ] Pre-push structural suite green; merge `feat/prelaunch-round2-hardening` → master; push origin/master (standing rule `feedback_always_merge_and_push`).
- [ ] Update CLAUDE.md status + memory (`project_prelaunch_audit_2026_06_22`).

---

## Self-review notes (author)

- **Spec coverage:** U1↔MED-2/LOW-3/LOW-5/LOW-6; U2↔MED-1/MED-3; U3↔MED-4/MED-5; U4↔type-design + LOW-2-safe regen; U5↔fast-follow/MED-6; U6↔MED-7/LOW-1; U7↔MED-8; U8↔LOW-9 (CLAUDE.md only — web doc-comments are accurate, intentionally skipped). All spec §4 units mapped.
- **Literal safety:** every Literal-ized field that takes a raw engine value has mapper coercion (streaming status/confidence) so a surprise value degrades to default, never 500s.
- **Dormant-path safety:** U2 guard is `clerk_configured()`-gated; a dedicated test locks the no-op.
- **Type consistency:** `Record{wins,losses,ties}` and `StatItem{label,value}` live in `common.py` and are reused; `caller_team` is the param name in both router and service; `fmtOpsNum` is the single OpsCards helper.
