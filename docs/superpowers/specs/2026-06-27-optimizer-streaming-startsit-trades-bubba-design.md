# Optimizer / Streaming / Start-Sit / Trades + Bubba Omniscience — Design

**Date:** 2026-06-27
**Status:** Approved (design forks locked) — pending spec review
**Track:** CEO/CMO joint — React frontend (`web/`) + FastAPI (`api/`) over the UNCHANGED `src/` engines
**Branch (planned):** `feat/optimizer-streaming-startsit-trades-bubba`

## Context

Five workstreams on the live React product. Four are wiring/frontend over engines that already exist; one (Start/Sit) is net-new; one (Bubba) is a cross-cutting AI-context enhancement. The Streamlit app and `src/` engines are NOT modified except two additive, behavior-preserving extensions (`start_sit_recommendation` player cap 4→6; new explainer **tools** in `src/ai/tools.py`). Investigation confirmed the engines (`stream_analyzer`, `start_sit`, `daily_optimizer`, `trade_finder`, `evaluate_trade`) already produce everything these features need.

## Goals

1. **Optimizer** — render the roster the way Yahoo shows it (grouped by Yahoo slot, batters/pitchers split, decision-colored), mirroring the old Streamlit layout.
2. **Streaming** — a date picker (today → +7) that re-queries pitcher streams for the chosen day, FA-filtered, weighted by the current week's matchup needs.
3. **Start/Sit (new page)** — pick a date (today → +7) and up to 6 players (roster or FA); get a ranked start/sit verdict AND an "apply to my open slots" optimizer pass.
4. **Trades** — fix the empty Finder (root cause: team-name resolution bug), enrich the Finder cards with engine fields the contract currently drops, verify Compare + Build end-to-end.
5. **Bubba** — make the assistant automatically aware of every page's rendered data ("see the screen") and able to explain how any number was calculated ("behind the scenes").

## Non-goals (YAGNI)

- No rebuild of Finder into the 5-tab Streamlit workbench (owner chose "fix + verify wiring").
- No per-suggestion playoff-delta in Finder (a full sim per card is too slow).
- No screenshot-vision auto-capture for Bubba (owner chose structured page data; manual screenshot already exists).
- No Postgres/workers/multi-tenancy (M4, owner-gated).
- No new Yahoo write paths (app is read-only; `fspt-w` unavailable).

## Locked design decisions

| Fork | Decision |
|------|----------|
| Start/Sit interaction | **Both** — ranked compare verdict + "apply to open slots" LP optimize action |
| Trades scope | **Fix + verify wiring** — debug empty Finder, enrich contract, verify Compare/Build |
| Optimizer layout | **Yahoo roster, annotated** — group by current Yahoo slot, batters/pitchers split, decision colors |
| Matchup weighting | **Yes, weight by it** — this-week category urgency biases Optimizer / Streaming / Start-Sit |
| Bubba perception | **Structured page data** — auto-attach the JSON each page rendered (screenshot stays on-demand) |
| Bubba explainers | **Both** — inline breakdowns on touched features + on-demand explainer tools |

## Cross-cutting: matchup-urgency weighting

The unifying thread. `src/optimizer/category_urgency.compute_urgency_weights(matchup, config)` already converts the live H2H matchup (which cats you're winning/losing/tied) into per-category 0–1 urgency. The daily lineup path already uses it. WS2 and WS3 must thread the same matchup into `build_optimizer_context` so `ctx.category_weights` reflect this week (today they're neutral for streaming — `streaming_service.py:149` builds context with no matchup). One helper, three consumers, identical semantics. When no live matchup is available, fall back to neutral/standings weights (never raise).

---

## WS1 — Optimizer: Yahoo-slot layout (frontend-only)

**Gap:** React renders two flat tables (`Starting Lineup`, `Bench`). Streamlit groups by Yahoo slot, splits batters/pitchers, sorts by (decision, slot order, DCV), and color-codes rows. The API already returns every field needed.

**Change:** `web/src/components/optimizer/LineupTable.tsx` + `web/src/app/optimizer/page.tsx`.
- Default the roster view to **daily mode** (`mode="daily"`) — the DCV start/sit view the Streamlit screenshot shows. **Keep standard/ROS optimize reachable** (a mode toggle) so the existing weekly-LP capability isn't dropped.
- Take the **union of `slots` + `bench`**, group each player by **`current_slot`** (their Yahoo slot), order by Yahoo slot order:
  - Batters: `C, 1B, 2B, 3B, SS, OF, OF, OF, Util, Util, BN, IL`
  - Pitchers: `SP, SP, RP, RP, P, P, P, P, BN, IL`
- Two sections (**Batters** / **Pitchers**), each row colored by decision: Start=green, forced-Start=orange (`forced_start`), Bench=blue, IL=gray.
- Columns: **Slot · Player · Eligibility · Value (0–100) · Matchup · Decision**.
- Flag suggested swaps where `current_slot` ≠ the optimizer's `slot`.

**API:** none. `LineupSlot` already carries `slot`, `current_slot`, `action`, `status`, `value`, `matchup`, `forced_start` (`api/contracts/lineup.py:17`).

**Inline explainer (WS5):** each row exposes its DCV value; the per-category DCV contributions are added to the daily slot payload for Bubba/tooltip (see WS5 inline breakdowns).

**Verification:** live API daily mode populates `current_slot`/`value`/`matchup` only with live Yahoo (matchup + schedule); locally degrade gracefully. Verify ordering/coloring against the live API; `pnpm build` + `tsc` gate.

---

## WS2 — Streaming: date picker + matchup-aware (frontend + small backend)

**Gap:** `GET /api/streaming?date=` already exists and the engine handles any date today→+7, but the React page never sends a date and the context isn't matchup-weighted.

**Frontend:** `web/src/app/streaming/page.tsx` + `web/src/lib/streaming-data.ts`.
- Add a 7-day date strip (reuse the Probables-page pattern: `today … today+7` buttons), default today.
- `fetchStreaming(date)` → `apiGet("/streaming", { date })`. Re-fetch on selection.

**Backend:** `api/services/streaming_service.py`.
- Build the optimizer context **matchup-aware**: resolve the viewer's live matchup, pass it so `ctx.category_weights` come from `compute_urgency_weights(matchup, config)`. Now `compute_streaming_value` weights the categories you need this week (high-K streamers rise when you need K; lost ERA/WHIP stop being protected). Fall back to neutral weights when no matchup.
- FA filtering already default (`include_rostered=False`). The `budget` strip already surfaces adds-left + cats-in-play.

**API contract:** unchanged shape; `date` param already accepted. Optionally surface the resolved `urgency` map on `StreamingResponse` for display + Bubba (small additive field).

**Inline explainer (WS5):** `factors[]` (value + registry weight + detail) already present per candidate — keep, and ensure `/api/streaming/analyze` is reachable from the page for any picked pitcher.

**Verification:** live local API for a few dates; confirm ranking shifts with matchup urgency; DB-free service test for the matchup-weight wiring; `pnpm build` + `tsc`.

---

## WS3 — Start/Sit: new page + new endpoints (net-new — largest WS)

**Engine (exists):** `src/start_sit.start_sit_recommendation(player_ids, pool, config, inputs=…)` — 3-layer H2H model (weekly projection × urgency × matchup factors → start_score; risk-adjusted; per-category SGP). `src/start_sit_widget.quick_start_sit` adds density/overlap. `src/optimizer/daily_optimizer.build_daily_dcv_table` + the daily LP fill open slots. **Extension:** raise the player cap 4 → 6 (additive; guard test).

**New endpoint A — compare/verdict:**
`POST /api/start-sit/compare`
- Request `StartSitCompareRequest{ team_name: str | None, date: str, player_ids: list[int]  # 2..6 }`
- Response `StartSitCompareResponse`:
  - `date`
  - `candidates: list[StartSitCandidate]` — `{ player: PlayerRef, start_score: float 0-100, rank: int, projected: list[StatItem], category_impact: list[StatItem], matchup: str, reason: str, playable: bool }`
  - `verdict: { start_ids: list[int], sit_ids: list[int], reasoning: str }`
  - `open_slots: dict[str, int]` — open lineup slots by position for that date
  - `confidence: float` + `confidence_label: str` ("Clear" / "Lean" / "Toss-up")
- Logic: build matchup-aware context (cross-cutting urgency); score each selected player via `start_sit_recommendation` + daily DCV for the date; rank by start_score; the verdict marks the top-K as **start** where K = how many of the selected players' positions can fill open lineup slots (a small position→slot assignment bounded by the league's starting slots, given the rest of the roster). The slot feasibility in `/compare` is a clearly-scoped heuristic; `/optimize` is authoritative.

**New endpoint B — apply to open slots:**
`POST /api/start-sit/optimize`
- Request `StartSitOptimizeRequest{ team_name: str | None, date: str, player_ids: list[int] }`
- Response reuses lineup contracts: `{ date, slots: list[LineupSlot], bench: list[LineupSlot], summary: str, daily: DailyMeta }`
- Logic: run the daily LP with the selected candidates added to the eligible pool (FAs included as hypothetical adds), filling the user's open slots optimally for the date. Reuses the `mode="daily"` machinery in `lineup_service`/`pipeline`.

**Service seam:** new `api/services/start_sit_service.py` (the ONE place importing `src.start_sit` + daily optimizer for this feature). Thin routers (`api/routers/start_sit.py`), DI provider in `api/deps.py`, fake-service DI tests (`tests/api/test_api_start_sit_*.py`), mount in `api/main.py`, regen `openapi.json`.

**Player selection:** the page composes existing endpoints — `GET /api/players/search?q=` (any roster/FA player), `GET /api/free-agents/pool` (ranked FAs), `GET /api/league/rosters`. No new selection API.

**Frontend:** new route `web/src/app/start-sit/page.tsx` + `web/src/lib/start-sit-data.ts` + components. Nav tab added to `web/src/components/chrome/TopBar.tsx`. UI: date strip · player multi-select (search roster + FAs, max 6, mixed positions) · comparison cards (start_score heat bar, projected line, per-category impact, matchup, reason) · ranked start/sit verdict bounded by open slots · "Apply to open slots" button → calls `/optimize` → shows the filled lineup.

**Inline explainer (WS5):** `category_impact` + `start_score` components are returned structured so Bubba/tooltips can explain "why start X over Y."

**Verification:** DB-free fake-service tests for both endpoints; live local smoke for a real date + real player ids (search → compare → optimize); `pnpm build` + `tsc`.

---

## WS4 — Trades: fix the wiring (debug + enrich + verify)

**Root cause of empty Finder (to confirm via systematic-debugging):** `api/services/trade_finder_service.py:34` does `user_roster_ids = league_rosters.get(team_name, [])` — a raw dict lookup. Yahoo team-name keys carry emoji/whitespace (e.g. "🏆 Team Hickey"); the request `team_name` won't match exactly → empty `user_roster_ids` → the engine has no roster to trade from → zero suggestions. The service also passes `all_team_totals=None`. This service bypasses the `effective_team`/reconciliation layer every other personalized router was fixed to use.

**Fix:**
- Resolve the user's team via the viewer-context reconciliation (emoji/whitespace-tolerant match against the actual roster keys), not a raw `.get`. Route the Finder through `require_viewer_context` / `effective_team` like the other personalized routers.
- Compute `all_team_totals` from `standings_utils.get_all_team_totals` (the playoff-odds service already does this) and pass it to `find_trade_opportunities`.

**Enrich the contract** (`api/contracts/trade_finder.py` + `api/services/trade_finder_service.py` + `web/src/lib/api/adapters.ts`):
- Add to `TradeSuggestion`: `grade` (derive from `net_sgp` via the same grade fn the evaluator uses), `partner_record` (from `load_league_records`), `category_impacts: list[CategoryImpact]` (per-cat SGP deltas, if the engine surfaces them cheaply).
- Frontend `TradeCard` renders the now-real grade/impact instead of deriving verdict-only.
- **Defer** per-suggestion `playoffDelta` (a sim per card is too slow) — documented.

**Verify Compare + Build:**
- Compare (`GET /api/compare`) — not gated; confirm picker (search) + table on live data.
- Build (`POST /api/trade/evaluate`) — Pro-gated but **dormant** on the free beta (Stripe unset → `billing_env_configured()` False → `require_pro` no-ops), so friends won't 402. Keep Monte Carlo **opt-in** (`enable_mc=false` default) to avoid the 45s hang; confirm the page default.

**Verification:** the Finder fix needs **real league data** — verify against the live API (or the main-checkout 26 MB DB), NOT the empty worktree DB. DB-free fake-service test for the resolution fix (assert the reconciliation is invoked + non-empty roster path). `pnpm build` + `tsc`.

---

## WS5 — Bubba: omniscient page context + explainers (cross-cutting)

Three pieces. New tools live in `src/ai/tools.py` so they work in BOTH the Streamlit app and the API (one-engine principle). The live Streamlit `chat()` path stays behavior-identical (additive tools; equivalence guarded).

### A. Auto page-context ("see the screen") — structured data

- **Frontend:** a global `BubbaContext` provider (mounted in `web/src/app/layout.tsx` wrapping both `{children}` and `<Bubba/>`). Centralize the publish in `usePageData` (`web/src/lib/use-page-data.ts`) so EVERY page auto-exposes `{ pageId (from route), data }` to the context on each `loaded` state — near-zero per-page code, covers all 14+ pages at once.
- **Bubba** reads the current page context and auto-attaches a **size-capped** JSON serialization to every message (default ON; a quiet toggle defaults on). Distinct from the existing manual `attached_text` tag flow (which stays).
- **API:** add `page_context: PageContext | None` to `ChatSendRequest` (`api/contracts/chat.py`): `{ page: str, data_json: str (capped, e.g. ≤16 KB) }`. `ChatService._build_user_content` / system-prompt assembly folds it in: "Data currently displayed on the `<page>` page (may be truncated): …". Truncation is explicit so Bubba knows to use tools for more.

### B. Explainer tools ("behind the scenes") — on-demand

New tools registered in `src/ai/tools.py` (`tool_specs()` + `dispatch_tool()`):
- `explain_constant(name)` → the `CONSTANTS_REGISTRY[name]` entry: `value`, `lower/upper_bound`, `citation`, `module`, `sensitivity`, `description`. Lets Bubba cite "matchup weight 0.28, per <citation>."
- `list_constants(module?, sensitivity?)` → browse/discover registered constants.
- `explain_metric(kind, params)` → `kind ∈ {stream_score, dcv, trade_grade, start_score}`. Returns the formula string + each component's value + its weight + the input variables that fed it, by calling the relevant service in an "explain" mode (reusing the breakdowns the engines already compute — e.g. streaming `_factors`, trade `category_impacts`, daily DCV per-category). Graceful "not found" on bad params.

### C. System-prompt upgrade

`src/ai/chat.build_system_prompt(page, viewer_team)` gains: a line that the page's currently-displayed data is attached, and that Bubba has explainer tools (`explain_constant` / `explain_metric`) to show how any number was derived and should cite constants/formulas when explaining "how." Persona unchanged otherwise (still reactive; no unsolicited trade opinions).

### Inline breakdowns (the "Both" half)

On the four touched features, ensure the response payloads carry the calculation breakdown so it's visible on the page (and to Bubba's auto-context) without a tool round-trip:
- Streaming: `factors[]` (already present) — keep.
- Optimizer daily: add per-category DCV contributions to the slot payload (small additive field).
- Start/Sit: `category_impact` + `start_score` components (in the new contract).
- Trades: `category_impacts` on both evaluate (present) and finder (added in WS4).

**Verification:** DB-free tests for the new tools (`tests/api/` + `tests/` for `src/ai/tools.py`); a `chat()` equivalence test proving the live Streamlit path is byte-identical with the new tools registered; frontend `BubbaContext` covered by `pnpm build` + `tsc`; live smoke: ask Bubba "what's on this page?" and "how was this stream score computed?" against the local stack.

---

## Contracts summary (new / changed)

| Contract | Change |
|----------|--------|
| `api/contracts/start_sit.py` (new) | `StartSitCompareRequest/Response`, `StartSitCandidate`, `StartSitOptimizeRequest`; reuse `LineupSlot`/`DailyMeta`, `StatItem`, `PlayerRef` |
| `api/contracts/trade_finder.py` | `TradeSuggestion` += `grade`, `partner_record`, `category_impacts` |
| `api/contracts/streaming.py` | `StreamingResponse` += optional `urgency: dict[str,float]` (display) |
| `api/contracts/lineup.py` | `LineupSlot` += optional per-category DCV breakdown (additive) |
| `api/contracts/chat.py` | `ChatSendRequest` += `page_context: PageContext|None`; new `PageContext{page, data_json}` |
| `openapi.json` | regenerated ONCE after all backend routes land; TS types regenerated from it |

All additions are backward-compatible (optional fields / new routes). `fastapi==0.137.1` + `httpx==0.28.1` pins respected (openapi snapshot guard).

## Execution model

- **One branch**, subagent-driven in a single worktree (shared files make parallel worktrees conflict-prone).
- **File-ownership split** to parallelize safely: WS1 (optimizer FE), WS2 (streaming FE + streaming_service), WS3 (start_sit_service + routers + start-sit page), WS4 (trade_finder_service + trade contracts/adapter), WS5 (chat_service + tools + BubbaContext). Shared touch-points are serialized:
  - `api/main.py` (route mounts), `openapi.json`, generated TS types → **one integrator pass at the end** (per the parallel-reconcile lesson: resolve generated files by regenerating, never hand-merge).
  - `web/src/components/chrome/TopBar.tsx` (nav) → WS3 only.
  - `web/src/lib/api/adapters.ts` → WS1/WS2/WS4 append disjoint adapters; integrator reconciles.
- **TDD per slice** (the proven pattern): contract → service seam → thin logic-free router (AST-guarded) → DI provider → fake-service DI test (`tests/api/test_api_*.py`) → mount → regen openapi.
- **Finish:** merge to local master + push origin/master; run silent-failure-hunter on the non-trivial backend changes (Finder fix, Bubba tools).

## Testing & verification

- **API:** DB-free fake-service tests (worktree DB is empty); structural router guards stay green; openapi snapshot regenerated.
- **Frontend:** `pnpm build` (the gate — `web/` has no CI), `pnpm exec tsc --noEmit`, `pnpm run lint`.
- **Live:** run the API locally (`uvicorn api.main:create_app --factory --port 8000`) + React preview against it (per the live-verification local-stack memory). The empty-Finder fix and matchup-weighting REQUIRE real data — verify against the live API or the main-checkout DB. Live-only Yahoo fields (matchup/schedule) degrade gracefully where the local DB can't populate them.

## Risks & open items

- **Finder may also depend on live-only league data** — if local `league_rosters`/`all_team_totals` are thin, verify against the live Railway API. The resolution fix is the primary lever; `all_team_totals` is secondary.
- **Daily-mode + streaming live fields** (matchup, schedule, current_slot) only fully populate with live Yahoo; locally they degrade — confirm full population against the live API.
- **Bubba always-on context = more tokens** — fine on the free/BYO-key beta; size-cap the payload; revisit for paid metered tiers (the B3 tier-aware caps already exist).
- **Shared-file conflicts** — mitigated by the single-branch integrator pass + regenerating generated files.
- **Start/Sit `/compare` slot-feasibility heuristic** vs `/optimize` LP — the compare verdict is a bounded heuristic; the optimize action is authoritative. Documented so they can't be confused.
