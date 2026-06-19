# HEATER — CMO/Frontend Track · Next-Session Handoff Prompt

> Paste **everything below the line** into a fresh Claude Code chat. Same environment, same repo (`C:\Users\conno\Code\HEATER_v1.0.1`, branch `master`). The previous session ended here.

---

You are continuing the **CMO / frontend track** of HEATER (a fantasy-baseball app being re-platformed into a monetized public product). You are paired with a **parallel CEO / backend track** (another Claude session, relayed through Connor). Your lane is `web/` (Next.js frontend) + `docs/superpowers/` + `.claude/launch.json`. **Never edit `api/` or `src/`** — that's the CEO's lane and they are active there.

## Skills / plugins / MCPs to use (invoke when applicable — do NOT skip the design gate)

- **superpowers:brainstorming** — design + get Connor's approval BEFORE building anything non-trivial (hard gate). Every feature/page starts here.
- **superpowers:writing-plans** — turn an approved spec into a bite-sized task plan.
- **superpowers:subagent-driven-development** — execute a plan task-by-task with fresh subagents + two-stage review (the mode used all last session). For verification-heavy tasks, the controller (you) drives the API/preview while subagents do code tasks.
- **superpowers:executing-plans** — alternative inline (in-session) plan execution with checkpoints.
- **superpowers:dispatching-parallel-agents** — M1 is 14 independent pages; parallelize the independent per-page work.
- **superpowers:verification-before-completion** — never claim done without running the gate + showing evidence.
- **superpowers:finishing-a-development-branch** — wrap up + merge (then push, per the standing rule below).
- **superpowers:requesting-code-review / receiving-code-review** — review discipline.
- **superpowers:systematic-debugging** — for any bug/test failure.
- **superpowers:using-git-worktrees** — optional isolation for parallel work.
- **frontend-design:frontend-design** — building the ~8 new React pages in M1 (distinctive, production-grade UI within the locked design system).
- **design:design-critique** — critique passes to find rough edges (used last session on Trades/Players).
- **design:design-system** — keep the "Combustion" system consistent.
- **design:accessibility-review** — a11y pass on new surfaces.
- **MCP `Claude_Preview`** (`preview_start` / `preview_screenshot` / `preview_eval` / `preview_inspect` / `preview_network` / `preview_console_logs` / `preview_resize` / `preview_stop`) — run the dev server and **VISUALLY verify every change**. Connor explicitly wants to see real renders.
- **MCP `visualize`** (`mcp__visualize__read_me` then `show_widget`) — mockups/diagrams during brainstorming.

Tooling note for this track: the codegen task below uses the npm package **`openapi-typescript`**.

## Standing rules (NON-NEGOTIABLE — Connor set these)

1. **Merge + push EVERYTHING.** Finish every task by merging to local `master` AND pushing to `origin/master`. No more "ask before pushing" — that rule is SUPERSEDED (memory `feedback_always_merge_and_push`).
2. **Reconcile before pushing.** `origin/master` moves because the CEO track keeps pushing (`api/`, `src/`, `tests/`). Files are disjoint (CMO = `web/`+`docs/`, CEO = `api/`+`src/`), so: `git pull --no-rebase --no-edit origin master` → clean merge → `git push origin master`. The **pre-push hook runs ~393 structural-invariant tests**; green = safe.
3. **Branch first.** On `master`, create a feature branch before working; merge it back at the end (then push).
4. **`web/` gate (there is NO unit-test runner):** `pnpm exec tsc --noEmit` + `pnpm run lint` + `pnpm build`, all run **from `web/`**. The production build is the real gate. Plus Claude_Preview screenshots for anything visual.
5. **Lane discipline.** Write only in `web/`, `docs/superpowers/`, `.claude/launch.json`. Never touch `api/`/`src/`. Never touch Postgres (it's dormant — see CEO protocol).
6. **Connor is a CLI/coding novice.** Respond in the **shortest, simplest, plainest** way possible — one action per step, no jargon, no filler (his CLAUDE.md rule). Execute terminal work autonomously (he's granted that).
7. **Read `web/AGENTS.md`** — this is Next.js 16 with breaking changes; read `node_modules/next/dist/docs/` before writing Next-specific code.

## Where we are — the migration roadmap (read it: `docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md`)

`M0 contract-lock → M1 14-page parity → M2 auth+billing → M3 single-league beta (current SQLite infra) → M4 Postgres+workers+multi-tenancy → M5 public launch.` Critical path to a proven sellable product = M0→M3; public market revenue = M4→M5. **We are in M0.**

## The CEO coordination protocol (just agreed — follow exactly)

- **The API grows to match the frontend** (not the reverse), but only with **engine-derivable** fields — the CEO won't fabricate data. They'll flag any mock field that isn't derivable so you either define its derivation or cut it.
- **CEO owns the contract:** they extend `api/contracts/*` + `api/services/*`, regenerate `api/openapi.json`, and FF-push. **You own:** the frontend codegen + the `fetchX()` wiring in `web/src/lib/*-data.ts`. **CEO never touches `web/`; you never touch `api/`/`src/`.**
- **`api/openapi.json` is the single source of truth** — committed + snapshot-guarded (`tests/api/test_openapi_contract.py`), regenerated via `scripts/export_openapi.py`. **Build codegen against the COMMITTED `api/openapi.json`**, not live FastAPI introspection (`fastapi==0.137.1`+`httpx==0.28.1` are pinned so the snapshot stays deterministic).
- **Seam handoff (per page / per gap-spec group):** CEO pushes contract + regenerated `openapi.json` → signals via Connor "page X contract is on master" → you pull, re-run `pnpm gen:api`, wire `fetchX()` for that page, verify. **Never edit the same file concurrently** — `openapi.json` is a sequential handoff (CEO writes, you consume after their push).
- **First CEO contract change:** `mlb_id` + `team` on `PlayerRef` (already in `load_player_pool`; ripples to every endpoint).
- **B2 Postgres is dormant/staging until M4.** `get_connection()` still defaults to SQLite; nothing runs on Postgres. The M3 beta runs on **current SQLite** — that assumption holds. Do not touch Postgres.

## Your immediate task — M0 frontend codegen (the CEO green-lit "build it now")

This is pure frontend-lane prep that de-risks the contract-lock before the CEO extends `PlayerRef`. The previous session had just branched + confirmed `api/openapi.json` exists with the expected schemas (`LeadersResponse`, `LeaderRow`, `PlayerRef`, `FreeAgentsResponse`, `MyTeamResponse`, … — all present). Steps:

1. Brainstorm-lite is fine here (the approach is agreed) — but present it to Connor and get a quick nod before building (hard gate).
2. Branch (e.g. `feat/web-api-codegen`).
3. `pnpm add -D openapi-typescript` (in `web/`).
4. Add a script to `web/package.json`: `"gen:api": "openapi-typescript ../api/openapi.json -o src/lib/api/generated.ts"` (run from `web/`; consumes the **committed** repo-root `api/openapi.json`).
5. Run `pnpm gen:api` → generates `web/src/lib/api/generated.ts` (OpenAPI schemas as TS, e.g. `components["schemas"]["LeadersResponse"]`).
6. Rewire `web/src/lib/api/types.ts` to thin aliases over the generated schemas so `client.ts`/`adapters.ts` keep working unchanged:
   `export type ApiLeadersResponse = components["schemas"]["LeadersResponse"];` (and `ApiLeaderRow`, `ApiPlayerRef`).
7. Verify: tsc + lint + build green; the generated types reproduce the hand-written ones (so the Research live wire still type-checks). Then run the API + the live preview to confirm Research still shows live leaders:
   - Run the API: `.venv/Scripts/python -m uvicorn api.main:create_app --factory --host 127.0.0.1 --port 8000` (background).
   - `preview_start` the `heater-web-live` launch config (flag `NEXT_PUBLIC_HEATER_LIVE=1` + `HEATER_API_BASE=http://localhost:8000` already wired), navigate `/research`, screenshot live leaders; stop the API → reload → confirm mock fallback.
8. Commit → merge → push (standing rule).
9. **When the CEO lands `mlb_id`+`team` on `PlayerRef`** and signals: re-run `pnpm gen:api` → the new fields appear → update `web/src/lib/api/adapters.ts` to map the real `mlbId`/`team` (instead of the `0`/`""` gap-placeholders) → the live Research page gains **real headshots + logos**. That's the first visible M0 payoff.

## Then — M1 page parity (the big lift)

Wire all 14 pages to live data, page by page, as the CEO lands each contract:
- **The 6 built pages** (`/`, `/optimizer`, `/matchup`, `/trades`, `/players`, `/research`): swap each `web/src/lib/*-data.ts` mock for the live client using the **slice-0 pattern** (already proven on Research): Next proxy → typed client (`apiGet`) → adapter → mock fallback, gated by `NEXT_PUBLIC_HEATER_LIVE`. Use the per-page gap spec to know what each endpoint must provide.
- **The ~8 missing surfaces** (Closer Monitor, Pitcher Streaming, standalone Standings, Punt Analyzer, Draft Simulator, + any others from the 14): build them as new React pages (use `frontend-design` + the Combustion primitives) and wire them live. Each gets its own brainstorm → plan → build → verify cycle; parallelize independent pages with `dispatching-parallel-agents`.

## Patterns & primitives to reuse (all in `web/`, on master)

- **State machine:** `usePageData` (`web/src/lib/use-page-data.ts`) — `loading|error|empty|loaded` + dev `?state=` override; `PageError`/`PageEmpty` (`web/src/components/ui/PageStates.tsx`). Every page uses these.
- **The slice-0 wiring:** `web/next.config.ts` (proxy `/api/*` → FastAPI), `web/src/lib/api/{client,types,adapters}.ts`, the flag-gated live path in `web/src/lib/research-data.ts`. Copy this pattern for each page.
- **Design system "Combustion":** navy `#0a1f3a` / heat `#ff5c10`, Archivo+Inter, `heatColor(pct)` ramp, `HeroNum` (Archivo width-axis numerals), `useCountUp`, `Card` (`interactive` for clickable), `PlayerDialog`/`PlayerAvatar` (the canonical player card; `mlbId=0` → initials fallback). Tokens in `web/src/lib/tokens.ts`, motion in `web/src/lib/motion.ts`.
- **Mock-data guard:** `web/scripts/audit-mock-ids.mjs` (`node web/scripts/audit-mock-ids.mjs` from repo root) verifies every hand-typed `mlbId` vs MLB's API — run it if you add mock players. (Last session fixed 8 wrong ids + 5 stale teams + 2 positions.)

## What's already DONE this session (do NOT redo)

- State-machine standardization on all 6 pages; Players + Trades uplift to the Optimizer's polish bar; mock-data integrity fixes; **API-wiring slice 0** (Research→`/api/leaders` proven live + fallback); the **per-page contract-gap spec**; the **migration roadmap**. All on master + GitHub.
- Two small **design-critique nits logged for later**: the off-token `#ff7a2e` hex on the Retry button + the doubled `RefreshCw` icon in `PageError`. Fold into a design pass when convenient.

## Read these first (in order)

1. `CLAUDE.md` — ★ NORTH STAR + "Web Frontend (`web/`)" + "Sub-project B status" sections.
2. `docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md` — the master sequence (M0–M5).
3. `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md` — what each endpoint must provide (M0/M1 requirements).
4. `docs/superpowers/specs/2026-06-19-frontend-api-wiring-slice0-design.md` + `web/src/lib/research-data.ts` — the wiring pattern to copy.
5. Memory: `feedback_always_merge_and_push`, `project_cmo_web_frontend_2026_06_19` (the divergence finding + session state).

**Start by:** reading the docs above, giving Connor a 3-bullet status, then proposing how you'll tackle the M0 codegen task (and get his nod before building).
