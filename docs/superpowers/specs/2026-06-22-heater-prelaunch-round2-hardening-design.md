# HEATER Pre-launch Hardening — Round 2 (MED/LOW backlog)

**Date:** 2026-06-22
**Track:** CEO / platform — pre-launch hardening, round 2 (after the 3 HIGH fixes shipped at master `51ed170`).
**Source backlog:** `Outstanding_June26.md` (repo root) — the 2026-06-22 pre-launch audit.
**Scope:** buildable-now MED/LOW items + the "successful-but-empty → mock" frontend fast-follow. **Excludes** owner-gated M4/M5 infra and the launch-day 12-leaguemate assignment.

---

## 1 — Context & baseline

The single-league beta is live: Vercel frontend (`heater-v1-0-1.vercel.app`) proxies `/api/*` server-side to the Railway FastAPI API over the unchanged `src/` engines. Master auto-deploys.

**HIGH-fix verification (2026-06-22, re-probed this session) — GREEN:**

| Check | Result |
|---|---|
| `git`: master == origin/master @ `51ed170` | HIGH-1/2/3 commits present + pushed |
| API `/healthz` | 200 `{"status":"ok"}` |
| API `/api/standings` / `/api/leaders/overall` / `/api/players/search` | real data (BUBBA CROSBY…; James Wood; Mike Trout `mlb_id 545361`) |
| Clerk gate `/api/me/team` | **401 "Login required."** |
| Vercel prod deploy | master prod alias, ● Ready → HIGH-1/3 frontend live; `/api/*` proxy returns real data |
| Railway API exact-SHA | API provably healthy; exact-SHA confirmation needs the ephemeral Railway token (HIGH-2 change is behind the auth gate, not externally probe-able) |

**Audit-staleness correction:** the audit's frontend line numbers predate the HIGH-3 fix. This spec is grounded in a fresh read of the **current** tree (three parallel read-only explorers, 2026-06-22). Notable corrections baked in below:

- **LOW-2 (fastapi pin) is a non-issue in this env** — local `fastapi==0.137.1` matches the pin; `tests/api/test_openapi_contract.py` passes. Regenerating `openapi.json` (U4) is safe.
- **LOW-9 shrank** — the web doc-comments the audit flagged (`standings-data.ts`, `matchup-data.ts`) are now *accurate*. Only the CLAUDE.md counts drift remains.
- **MED-6 confirmed real** — `OpsCards.tsx:36-42` renders `{card.value}/{card.total}` as raw floats; the ring-% rounding is a separate, correct calc.
- **The "empty→mock" gap is real but inverted from the audit's framing** — see §3 decision 1.

---

## 2 — Out of scope (do NOT touch this round)

- Owner-gated M4 infra: B2.2–B2.5 Postgres, B3 Redis/Arq workers, B4 `league_id` tagging / connectors / per-user Yahoo OAuth, B4 multi-league switch-on.
- M5 public launch (open signup, connectors, marketing site, retire Streamlit).
- M6 owner-manual: the 12-leaguemate assignment, Yahoo `fspt-w` write scope, DNS/domain.
- The live Streamlit app (`app.py`, `pages/`) and `src/` engines (except the two additive, behavior-preserving touches: `src/ai/keys.py` logging in U1, and any test-surfaced bug in U3). No engine math changes.

---

## 3 — Key design decisions (owner-approved)

**Decision 1 — Live never fabricates.** In live mode (`NEXT_PUBLIC_HEATER_LIVE === "1"`), a successful (2xx) response whose primary payload is empty resolves to the honest `empty` state (`return null` → `usePageData` `empty`), **never** the demo mock. Mock data remains ONLY for non-live demo mode (the `mock` arm of `liveOrMock`). Mechanically: in each affected fetcher, the empty false-branch of `cond ? adapt(api) : MOCK` changes from the mock constant to `null`; `data.ts:fetchMyTeam` gains an explicit full-empty check.

**Decision 2 — Write-guard is Clerk-gated.** The cross-team write refusal (MED-1) enforces only when `clerk_configured()` is true (live multi-user). When Clerk is off (the current dormant single-owner path), the guard is a no-op so the write path is byte-for-byte unchanged. When live: a caller whose resolved team (`ctx.team_name`) is `None` or ≠ the token-owner's team gets a graceful `MutationResult(ok=False, error=…)` (matches the established HTTP-200-with-ok pattern; no new 4xx shape).

**Decision 3 — OpsCards numeric formatting.** Integers render plain (`7 / 10`), non-integers to one decimal (`18.2 / 53.8`). A small `fmtOpsNum(n)` helper: `Number.isInteger(n) ? String(n) : n.toFixed(1)`. The ring-% math stays on the raw float values (no precision loss).

---

## 4 — Units (sequential; each gets a two-stage implement → review gate)

### U1 — Backend observability sweep  *(MED-2, LOW-3, LOW-5, LOW-6)*
**Goal:** every service swallow site logs before degrading; no behavior change (still returns empty/graceful). Removes the documented silent-failure blind spot.

**House style (from well-behaved siblings):** `logger = logging.getLogger(__name__)` at module top; `logger.warning("ServiceName.method[ context] failed: %s", [ctx,] exc)` — exception object, not full traceback (write path is the one exception — see U2).

**Sites (ground-truth file:line, 2026-06-22):**
| File | Line(s) | Current | Add logger? |
|---|---|---|---|
| `api/services/fa_service.py` | 30 | `except: recs=[]` | yes (no module logger) |
| `api/services/leaders_service.py` | 82 | `except: rows=[]` | yes |
| `api/services/standings_service.py` | 21 | `except: teams=[]` | yes |
| `api/services/streaming_service.py` | 161 (board), 255 (`analyze_pitcher`→`found=False`) | swallow | yes |
| `api/services/closers_service.py` | 28 | `except: entries=[]` | yes |
| `api/services/punt_service.py` | 61 | `except: []`,`[]` | yes |
| `api/services/draft_service.py` | 38, 49, 79 | error responses (49/79 emit a generic "no pool data" summary for *any* failure) | yes |
| `api/services/matchup_service.py` | 315 (`get_matchup`→`[]`), 347 (`_build_roster_tables`→`pass`) | swallow | **has** logger (line 22) |
| `src/ai/keys.py` | ~70, ~147 | Fernet decrypt failure → `return None` silently | yes (no module logger) — LOW-3 |
| `api/services/team_service.py` | 103-104 | `yds.get_matchup()`/`get_standings()` called OUTSIDE try/except in `get_my_team` | wrap (LOW-5) so a Yahoo-singleton failure can't 500 the dashboard; module already resilient elsewhere |

**Notes:** draft_service 49/79 — keep the graceful user-facing summary (behavior unchanged) but log the real `exc`. `streaming_service` line 159 inner `probables` except may also get a debug log (optional). LOW-6 folds into the line-255 warning (an engine error currently masquerades as `found=False`; we log it — keeping the `found=False` contract).

**TDD:** per-site `caplog` test asserting (a) a `WARNING` is emitted when the underlying engine call raises, and (b) the return value is still the graceful empty/`found=False`/default shape. DB-free (monkeypatch the engine seam to raise). `team_service` test: a raising `get_standings` no longer propagates.

**Reviewer:** `pr-review-toolkit:silent-failure-hunter`.

### U2 — Write isolation + traceback  *(MED-1, MED-3)*
**Goal:** an authenticated caller cannot mutate another team's roster once Clerk is live; failed writes leave an operator trace.

**Current state:** `api/routers/roster_write.py` gates with `require_principal` (auth) but has **no** team-authorization and no viewer context; `api/services/roster_write_service.py` uses the single server `get_yahoo_data_service()._client` (the token-owner's team) and never consults a viewer context; exceptions → `MutationResult(ok=False, error="Write failed: <ClassName>")` (line ~52) with no logging.

**Design:**
- Router: add `ctx = Depends(require_viewer_context)`; pass `ctx.team_name` (the caller's resolved team) into the service methods.
- Service: resolve the **token-owner team** via the existing seam (`get_yahoo_data_service()._get_user_team_name()`, which reads `league_teams.is_user_team=1` in v1 / the Clerk membership team in multi-user). If `clerk_configured()` AND (`caller_team is None` OR `caller_team != owner_team`) → return `MutationResult(ok=False, error="Not authorized to modify another team's roster.", status="forbidden")` BEFORE any Yahoo call. When Clerk off → skip the check entirely (dormant path unchanged).
- Service: on the exception path, `logger.warning("RosterWriteService.<method> failed: %s", exc, exc_info=True)` before returning the graceful `MutationResult`; also log the `ok=False` passthrough in `_to_result` (MED-3).
- Keep the router thin: the authz comparison lives in the **service** (which already owns the `src/` seam) so `tests/api/test_no_logic_in_routers.py` stays green; the router only wires `ctx` → service.

**TDD (DB-free, fake service/seam):**
- Clerk-on + caller_team ≠ owner_team → `ok=False`, status forbidden, **no Yahoo write attempted**.
- Clerk-on + caller_team == owner_team → write proceeds (delegates to client).
- Clerk-on + caller_team None (unassigned) → refused.
- Clerk-off (dormant) → guard is a no-op; write proceeds exactly as today.
- Exception path → `caplog` WARNING with traceback; graceful `MutationResult(ok=False)`.

**Reviewer:** `security-review` (Skill) — auth/isolation focus.

### U3 — Real service test coverage  *(MED-4, MED-5)*
**Goal:** the real engine→contract mappers for two core launch surfaces, plus the largest untested real-logic service, get static unit coverage (the "false-green synthetic test" class CLAUDE.md warns about).

- `PuntService.get_punt` (`api/services/punt_service.py:16-69`): rank/gainable extraction, `is_punt` branch.
- `StandingsService._build_teams` (`api/services/standings_service.py:25-99`): `"W-L-T"` split, OVERALL/RECORD row detection, NaN guards.
- `live_boxscore` line parsers (`api/services/live_boxscore.py`): `_hitter_line`/`_pitcher_line`, AVG/OBP math, statsapi traversal shape.

**Approach:** static-method / pure-function unit tests with synthetic inputs (mirroring `tests/api/test_lineup.py`), **DB-free** (worktree/CI DB is empty — see `reference_worktree_empty_db`). No source change unless a test surfaces a real bug (then fix + note).

**Reviewer:** `pr-review-toolkit:pr-test-analyzer`.

### U4 — Type-design contract hardening  *(type-design)*
**Goal:** replace stringly-typed enum/record fields with `Literal`/structured types so a case/order mismatch fails typecheck, not silently at render. **Highest churn — touches contracts consumed by the frontend.**

**Prereq:** confirm local `fastapi==0.137.1` (verified) before regenerating.

**Changes (contract → mapper):**
1. `api/contracts/streaming.py:27-28` `status`/`confidence` → `Literal`. Engine vocab: status `{"PROBABLE","LOCKED","FINAL"}` (+ "OPEN" in the adapter map), confidence `{"HIGH","MEDIUM","LOW"}`. **Keep emitting the raw engine case** (frontend lowercases in `adapters.ts`); the `Literal` documents/enforces the exact set. Mapper: `streaming_service.py:198-199`.
2. `api/contracts/streaming.py:50` `ip_pace: float = 0.0` → `float | None = None` (deferred field; "not computed" ≠ "0"). Mapper `streaming_service.py:61`; frontend `adapters.ts:161` null-safe.
3. `api/contracts/matchup.py:24` `MatchPlayer.stats: list[str]` (+ `SideTotals` lists) → `list[StatItem]` (reuse `api/contracts/common.py` `StatItem{label,value}`). Pair each value with its column label so the order-coupling to `_HITTER_COLUMNS`/`_PITCHER_COLUMNS` (`matchup_service.py:24-25`) becomes explicit. Mappers `_fmt_hitter_stats`/`_fmt_pitcher_stats`. Frontend `matchup/page.tsx` `StatCells` unpacks `{label,value}`.
4. `api/contracts/matchup.py:16` `MatchupCategory.win: str` → `Literal["you","opp",""]`. Server-internal; low frontend churn.
5. `api/contracts/my_team.py` `Mover.trend`→`Literal["up","down","flat"]`, `Mover.tag`→`Literal["hot","cold",""]`, `StatusChip.status`→`Literal["ok","warn","info"]`, `OpsCard.status`→`Literal["ok","warn","danger"]`. Unify vocab (audit flagged `danger` vs `warn`/`info` inconsistency — keep each field's own closed set; document it). Mappers in `team_service.py` (452-453, 462-465, 272/289/316).
6. `api/contracts/playoff.py:16` + `api/contracts/matchup.py:37` records: **additively** introduce a structured `{wins,losses,ties}` field; **keep** the existing display string (`projected_record`, `TeamSide.record` with its `· 8th` suffix) so no downstream split breaks. The frontend may migrate to the structured field opportunistically in U5 but is not required to. (No destructive rename.)
7. `api/contracts/chat.py:32` `tool_trace: list[dict]` → a typed `ToolTraceEntry{name:str, args:dict}` list (shape from `src/ai/providers.py`).

**Regen:** `python scripts/export_openapi.py` → commit `api/openapi.json`; `tests/api/test_openapi_contract.py` must stay green. Contract unit tests for each new type.

**Reviewer:** `pr-review-toolkit:type-design-analyzer` + `feature-dev:code-reviewer`.

### U5 — Frontend contract-sync + empty-fix + OpsCards  *(fast-follow, MED-6)*
**Depends on U4.**
- `cd web && pnpm gen:api` → regenerate `web/src/lib/api/generated.ts`; update `adapters.ts`/`types.ts` for U4's structured `stats`/records/`ip_pace|null`/`Literal`s.
- **Empty-fix (decision 1):** in the ~10 fetchers that currently return a mock constant on empty-200, change the empty branch to `null`. Files (current behavior verified): `standings-data.ts:85-107`, `players-data.ts:79-91`, `streaming-data.ts:217-225`, `closers-data.ts:83-91`, `probables-data.ts:217-225`, `punt-data.ts:78-86`, `hitter-matchups-data.ts:~209`, `compare-data.ts:118`, `trades-data.ts:140-150`, `research-data.ts:90-98`. Plus `data.ts:fetchMyTeam:108-117` — add an explicit full-empty check (no movers/ops/categories) → `null`. Confirm each page renders the `empty` state (Team page is the reference).
- **OpsCards (decision 3):** `fmtOpsNum` helper; apply in `OpsCards.tsx:36-42`; ring math unchanged.

**Verify:** `pnpm exec tsc --noEmit` clean, `pnpm build` green, **preview**: live + empty + mock paths (web/ has no test runner → preview is the gate; heed `reference_preview_tool_gotchas`).
**Reviewer:** `feature-dev:code-reviewer`.

### U6 — Frontend error states + identity  *(MED-7, LOW-1)*
- **Draft (MED-7):** `web/src/lib/use-draft.ts` — add an `error` phase so a failed `start()`/`pick()` (non-402) surfaces a retryable error instead of silently resetting to `setup` / leaving an optimistic pick. Render it in `web/src/app/draft/page.tsx`.
- **Databank (MED-7):** distinguish outage from "no history": `web/src/lib/databank-data.ts` rethrow (or signal) on error so `web/src/app/databank/page.tsx` shows an error/retry, not "No history found".
- **TopBar identity (LOW-1):** `web/src/components/chrome/TopBar.tsx:172-185` — replace literal "Connor Hickey"/"Team Hickey"/"CH" with `useUser()` (`@clerk/nextjs`) for the name + `getViewerTeam()` (`web/src/lib/viewer-team.ts`) for the team; derive initials. Graceful fallback when Clerk off / team not yet resolved.

**Verify:** tsc + build + preview. **Reviewer:** `feature-dev:code-reviewer`.

### U7 — Accessibility, Matchup-first  *(MED-8)*
- `web/src/app/matchup/page.tsx`: add a non-color winner cue (text/icon, not just `bg-heat/10`) for per-category winners; convert the inert date "tabs" (`<span>`) to real keyboard-focusable buttons.
- `web/src/components/streaming/AnalyzeStarter.tsx:67-77`: label the `<select>`.
- Opportunistic cheap wins if low-risk: table `scope`/`<th>` on `StandingsTable`/`ComparePanel`, a skip-to-content link in `layout.tsx`. (Bubba focus-management deferred unless trivial.)

**Verify:** preview (keyboard nav + visible non-color cue). **Reviewer:** `feature-dev:code-reviewer`.

### U8 — Docs
- `CLAUDE.md`: endpoint count 23→26, `require_pro` 6→7, note `POST /api/draft/grade`. (No web doc-comment edits — they're accurate.)

---

## 5 — Sequencing & dependencies

```
U1 ─┐
U2 ─┼─ (backend, mostly disjoint; run sequentially with review gates; U3 may overlap U2 — tests-only)
U3 ─┘
U4 ─── (contracts; must precede U5) ─── U5 ─── U6 ─── U7   (web/, sequential — no parallel implementers on web/)
U8 ─── (anytime; trivial)
```
- U1 must precede U4 where they touch the same file (`streaming_service.py`, `matchup_service.py`) to avoid edit conflicts.
- U4 must precede U5 (generated.ts depends on the regenerated openapi.json).
- All web units (U5/U6/U7) run sequentially.

## 6 — Definition of done

- `python -m pytest tests/api/ -q` green; structural-invariant guards green; new U1/U2/U3 tests green.
- `api/openapi.json` regenerated; `test_openapi_contract.py` green.
- `web/`: `pnpm exec tsc --noEmit` clean; `pnpm build` green; U5/U6/U7 preview-verified (evidence captured).
- Each unit passed its mapped two-stage review; findings applied or consciously deferred.
- Branch `feat/prelaunch-round2-hardening` merged to local master AND pushed to origin/master (standing rule); pre-push structural suite green.
- `CLAUDE.md` + memory updated.

## 7 — Risks

- **U4 churn:** contract changes ripple to mappers + frontend adapters. Mitigation: U4 regenerates openapi + has type-design review; U5 immediately re-syncs the frontend and gates on `tsc`/`build`. For records (item 6), prefer additive structured fields to avoid breaking any frontend split.
- **Dormant write path:** U2's guard must be `clerk_configured()`-gated or it breaks v1/dormant writes. Locked by a dedicated test + security-review.
- **web/ has no test runner:** U5–U7 correctness rests on preview + tsc + build. Heed the preview gotchas (snapshot > screenshot after many HMR rebuilds).
- **Empty-state coverage:** decision 1 assumes every page renders `empty`. U5 verifies per-page in preview; if a page lacks an `empty` render, add it.
