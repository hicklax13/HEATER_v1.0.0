# HEATER M1 — React Parity (14 Streamlit surfaces → live API) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Ship all 14 HEATER Streamlit surfaces as React pages in `web/`, each wired to the live `/api/*` contract and passing the four-state machine, on current SQLite infra (Streamlit stays live — strangler-fig).

**Architecture:** Next 16 App Router page → `usePageData` four-state hook → module-level `fetchX()` in `lib/*-data.ts` → typed client (`apiGet/apiPost`) → adapter (`Api* → view-model`) → graceful mock fallback (gated by `NEXT_PUBLIC_HEATER_LIVE=1`). API types are generated from `api/openapi.json` via `pnpm gen:api`; never hand-edited.

**Tech Stack:** Next 16.2.9, React 19.2.4, TypeScript 5.9, Tailwind v4, framer-motion, pnpm. Design system "Combustion" (navy `#0a1f3a` / orange `#ff5c10`; tokens in `lib/tokens.ts`; motion in `lib/motion.ts`). Canonical components: `PlayerLink`/`PlayerDialog`, `Card`, `EmptyState`, viz in `components/viz/`.

**Verification gate (the frontend "test"):** every page change must pass, in order:
1. `pnpm exec tsc --noEmit` (types)
2. `pnpm run lint`
3. `pnpm build` (the production gate — `web/` has NO CI)
4. Live preview: `preview_snapshot` + `preview_console_logs` (zero errors) + `preview_screenshot`; click/fill interactive bits; confirm parity vs the Streamlit page.

**Lane boundary:** touch `web/` + `docs/` ONLY. Never edit `api/`, `src/`, `tests/`. If a page needs a backend change, STOP and note it in `docs/` for the CEO track.

---

## Status snapshot (2026-06-20)

| # | Surface | Route | Endpoint(s) | State |
|---|---------|-------|-------------|-------|
| 1 | My Team | `/` | `GET /api/me/team`, `/api/playoff-odds` | ✅ wired — VERIFY |
| 2 | Lineup Optimizer | `/optimizer` | `POST /api/lineup/optimize` (daily) | ⚠️ wired, `daily{}` partial — FINISH + VERIFY |
| 3 | Closer Monitor | `/closers` | `GET /api/closers` | 🔨 BUILD (revive archived branch) |
| 4 | Pitcher Streaming | `/streaming` | `GET /api/streaming`, `POST /api/streaming/analyze` | ✅ wired — VERIFY |
| 5 | Matchup Planner | `/matchup` | `GET /api/matchup` | ✅ wired — VERIFY |
| 6 | League Standings | `/standings` | `GET /api/standings`, `/api/playoff-odds` | ✅ wired — VERIFY |
| 7 | Punt Analyzer | `/punt` | `GET /api/punt` | ✅ wired — VERIFY |
| 8 | Trade Analyzer | `/trades` (Build) | `POST /api/trade/evaluate` | ✅ wired — VERIFY |
| 9 | Trade Finder | `/trades` (Finder) | `GET /api/trade-finder` | ✅ wired — VERIFY |
| 10 | Free Agents | `/players` | `GET /api/free-agents/pool` | ✅ wired — VERIFY |
| 11 | Player Compare | `/trades` (Compare) | `GET /api/compare`, `/api/players/search` | ✅ wired — VERIFY |
| 12 | Leaders | `/research` | `GET /api/leaders/overall` | ✅ wired — VERIFY |
| 13 | Player Databank | `/databank` | `GET /api/databank`, `/api/players/search` | ✅ wired — VERIFY |
| 14 | Draft Simulator | `/draft` | `POST /api/draft/recommend`, `/api/draft/simulate-picks` | 🔨 BUILD NEW |

**Net remaining build work:** Draft Simulator (new), Closer Monitor (revive), Optimizer `daily{}` finish. Everything else is verification + contract hygiene.

---

## Execution order

1. **Phase 0 — Contract hygiene (Task C).** Regenerate `generated.ts`, confirm zero drift, confirm every page on typed client. (Fast, de-risks everything downstream.)
2. **Phase 1 — Task A finish.** Optimizer `daily{}` gap. (Databank + Compare already done — verify only.)
3. **Phase 2 — Build the 2 pages (Task B).** Closer Monitor (revive) then Draft Simulator (own plan).
4. **Phase 3 — Verify all 14 (Tasks D + E).** Four-state coverage + live preview pass.
5. **Phase 4 — Integration.** Final `pnpm build`, merge to master, push.

---

## Phase 0 — Contract hygiene (Task C)

**Files:** `web/src/lib/api/generated.ts` (regen), `web/src/lib/api/types.ts` (verify aliases).

- [ ] **Step 1:** Regenerate API types from the committed contract.
  Run (from `web/`): `pnpm gen:api`
  Expected: `generated.ts` rewritten from `../api/openapi.json`; `git diff` shows only additive/no change (M0 backend is frozen on master).
- [ ] **Step 2:** Confirm no hand-written API request/response types exist outside `generated.ts`. `api/types.ts` must be aliases only; `lib/types.ts` must be view-models only.
  Run: `pnpm exec tsc --noEmit`
  Expected: PASS.
- [ ] **Step 3:** Confirm the draft schemas needed for Task B exist in `generated.ts` (`DraftRecommendRequest`, `DraftRecommendResponse`, `DraftSimulatePicksRequest`, `DraftSimulatePicksResponse`, `DraftConfig`, `DraftPick`, `DraftClock`, `DraftRecommendation`) and the closer schema (`ClosersResponse`/closer item).
  Run: grep `generated.ts` for `Draft` and `Closer`.
  Expected: present (M0 shipped 23 endpoints incl. draft + closers). If a name differs, record the real name for the build tasks.
- [ ] **Step 4:** Commit if `generated.ts` changed.
  `git add web/src/lib/api/generated.ts && git commit -m "chore(web): regen API types from openapi.json (M1 contract lock)"`

---

## Phase 1 — Optimizer `daily{}` finish (Task A)

**Context:** The page sends `mode:"daily"` and renders per-slot `value`/`matchup` + `ip_pace`/`swaps`/`impact`. The backend `daily` object ALSO carries `urgency`, `rate_modes` (ERA/WHIP protect|compete|abandon), `winning/losing/tied`, `recommendations` — currently dropped by the adapter. Surface them.

**Files:**
- Modify: `web/src/lib/api/adapters.ts` (`apiOptimizeToData` — map the full `daily` object)
- Modify: `web/src/lib/optimizer-data.ts` (view-model type for the daily meta)
- Modify: `web/src/app/optimizer/page.tsx` (render the new daily meta)
- Verify against: `web/src/lib/api/generated.ts` (`LineupOptimizeResponse.daily` / `DailyMeta` shape)

- [ ] **Step 1:** Read the real `daily` shape in `generated.ts` (`DailyMeta`): confirm field names for `urgency`, `rate_modes`, `winning`/`losing`/`tied`, `recommendations`, `swaps`, `ip_pace`. Record exact names/types.
- [ ] **Step 2:** Extend the optimizer view-model (in `optimizer-data.ts`) with `urgency`, `rateModes`, `gameState` (winning/losing/tied), `recommendations` — all optional (graceful when engine returns empty).
- [ ] **Step 3:** Extend `apiOptimizeToData` to map those fields NaN/null-safe (mirror the existing `ip_pace`/`swaps` mapping). Keep the existing fields untouched.
- [ ] **Step 4:** Add a compact "Daily context" panel to `optimizer/page.tsx` rendering urgency + rate-mode chips + game-state + recommendations list. Render nothing when all are empty (graceful — matches the live-data degradation the backend documents).
- [ ] **Step 5:** Gate: `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`. Expected: all PASS.
- [ ] **Step 6:** Preview-verify `/optimizer` (mock fallback shows the panel; live shows real values). `preview_console_logs` zero errors; `preview_screenshot`.
- [ ] **Step 7:** Commit. `git commit -m "feat(web): optimizer renders full daily{} (urgency/rate-modes/game-state/recs)"`

---

## Phase 2a — Closer Monitor revive (Task B.1)

**Context:** Built + verified on branch `feat/web-closer-monitor` (1 commit `a74a4d5`) but archived because the static grid is low-value; archive spec at `docs/superpowers/specs/2026-06-19-heater-closer-monitor-archived-saves-finder.md`. The branch is based on an ancient master — do NOT merge it (it would revert M0/M1 work). Instead harvest its 3 files and re-fit to current master. Backend `/api/closers` is live (≈21 real closers, minimal fields; `handcuffs[]`/stats/real `confidence` not yet exposed — the card degrades gracefully). M1 = parity (ship the grid); the high-value "Saves Finder" is a deferred future item documented in the archive spec.

**Files (harvest from branch, re-fit on master):**
- Create: `web/src/app/closers/page.tsx`
- Create: `web/src/components/closers/CloserCard.tsx`
- Create: `web/src/lib/closers-data.ts`
- Modify: `web/src/lib/api/types.ts` (add `ApiClosersResponse` alias if missing)
- Modify: `web/src/lib/api/adapters.ts` (add `apiClosersToData`)
- Modify: `web/src/components/chrome/TopBar.tsx` (add "Closers" nav item)
- Optional: `web/scripts/audit-mock-ids.mjs` (extend to closers-data — the archive spec flags this as independently valuable; it caught 2 wrong headshots)

- [ ] **Step 1:** Extract the 3 archived files to inspect (don't checkout the branch):
  `git show feat/web-closer-monitor:web/src/app/closers/page.tsx`, `:web/src/components/closers/CloserCard.tsx`, `:web/src/lib/closers-data.ts`.
- [ ] **Step 2:** Recreate the 3 files on master, re-fitting imports to the CURRENT `adapters.ts`/`types.ts`/`client.ts`/`use-page-data.ts` (the archived versions targeted an older shape). Use `usePageData` four-state; grid sorted by job-security desc; card degrades when stats/handcuffs absent.
- [ ] **Step 3:** Add `ApiClosersResponse` alias in `api/types.ts` (from `generated.ts` — confirm the real schema name in Phase 0 Step 3) and `apiClosersToData` in `adapters.ts` (snake→camel, null-safe `toPlayerRef`; map `confidence`→security via a `CONFIDENCE_SECURITY` table).
- [ ] **Step 4:** Add the "Closers" nav item to `TopBar.tsx` (after Streaming — closers are a pitching surface).
- [ ] **Step 5:** Gate: `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`. Expected: all PASS.
- [ ] **Step 6:** Preview-verify `/closers`: mock fallback grid renders; with live flag, real closers + headshots, zero console errors. Force `?state=empty|error|loading` to confirm four-state. `preview_screenshot`.
- [ ] **Step 7:** Commit. `git commit -m "feat(web): revive Closer Monitor grid wired to /api/closers (M1 parity)"`

---

## Phase 2b — Draft Simulator (Task B.2)

**This page is net-new and UX-complex → it gets its own dedicated plan**, authored after a brainstorming pass on the draft-board UX. Write `docs/superpowers/plans/2026-06-20-heater-m1-draft-simulator.md` covering:

- **Route:** `/draft` (new). Add to `TopBar.tsx` (Preseason context — eyebrow "SCOUTING" mirrors Streamlit in-season).
- **Stateless model:** client owns `config` (`{num_teams, num_rounds, user_team_index, roster_config}`) + `pick_log` (`DraftPick[]`). Server replays the log every call. `user_team_index` is 0-based (UI input is 1-based → subtract 1). `pick` field = `pick_log.length` at time of pick. `team_index` is authoritative — echo back what the server returned; for user picks compute snake order locally (`round = pick // num_teams; pos = pick % num_teams; team = round%2==0 ? pos : num_teams-1-pos`).
- **Two-call turn loop:** (1) `POST /api/draft/recommend` when `clock.is_user_turn` → top-N rec cards (player, score, projected_sgp, confidence, tag, reason). (2) On user pick: append `DraftPick` locally → `POST /api/draft/simulate-picks` (advances AI to next user turn) → append response `picks[]` → render new `clock`. Never call simulate-picks when already the user's turn (harmless no-op, wasted round-trip).
- **Setup screen:** num_teams (default 12), num_rounds (default 23), draft position (1-based, default 1). (Streamlit also has sim-depth + engine-mode radios → optional, can default `top_n`/`n_simulations` server-side.)
- **Draft UI:** header (Round N · Pick N; "Your Pick" vs "On the clock · TeamName"); my-roster panel (slot+player via `PlayerLink`); recent-picks feed (user picks highlighted); recommendation cards (top-3 with BUY/FAIR/AVOID pill + score) + a draft-any-player search (reuse `lib/player-search.ts`); "Simulate Opponent Pick" / auto-advance; Reset + Undo controls.
- **Server never 500s** → on empty recommendations show "recommendations unavailable" fallback, not a crash.
- **Graceful:** flag-gated live wire + a mock draft (deterministic seed for the demo) fallback so the page is demoable offline.

The dedicated plan will TDD-decompose this into setup → state hook (`useDraft` holding config+pick_log) → recommend wiring → simulate-picks wiring → draft board UI → controls → verify.

---

## Phase 3 — Verify all 14 (Tasks D + E)

**Four-state coverage (Task D):** confirm each page handles loading/empty/error/loaded. 8/10 existing pages use `usePageData`. Databank + Trades sub-tabs (Compare/Build) use manual state — confirm they show a loading indicator, a graceful empty ("no results"), and an error path (search/fetch rejection) rather than blank/crash. Patch any gap (frontend-only).

**Live preview pass (Task E):** with the dev server running, for EACH of the 14 routes:
- [ ] `preview_snapshot` — content + structure present
- [ ] `preview_console_logs` — zero errors
- [ ] Exercise the interactive bit (search picker, tab switch, optimize, draft pick, lens toggle, trade build)
- [ ] `preview_screenshot` — share as proof
- [ ] `?state=empty` / `?state=error` / `?state=loading` on the `usePageData` pages — confirm each branch renders
- [ ] Confirm parity vs the matching Streamlit page (same data, same decisions surfaced)

Run the pass against BOTH mock (`NEXT_PUBLIC_HEATER_LIVE` unset) and, where the local API is reachable, live (`=1`).

---

## Phase 4 — Integration (Task F)

- [ ] Full gate on the whole app: `pnpm exec tsc --noEmit && pnpm run lint && pnpm build`. Expected: all PASS, zero type/lint/build errors.
- [ ] `git pull --no-rebase origin master` (reconcile CEO track — `web/` vs `api/` disjoint → clean).
- [ ] Merge the work to local `master` and `git push origin master` (pre-push runs ~390 structural tests — all in `tests/`, untouched by this lane, so they stay green).
- [ ] Delete the stale branch `feat/web-wire-research-overall` (fully contained in master) and `feat/web-closer-monitor` once revived.

---

## Self-review notes

- **Spec coverage:** all 14 surfaces accounted for (12 verify, 2 build, 1 finish). Tasks A–F from the M1 brief all mapped (A=Phase 1 + verify; B=Phase 2a/2b; C=Phase 0; D/E=Phase 3; F=Phase 4 + per-task gates).
- **No backend edits:** every task is `web/` or `docs/` only. The Optimizer `daily{}` fields and closer fields are already in the M0 contract — no `api/` change needed.
- **Risk:** Draft Simulator is the only high-uncertainty build (net-new UX + two-call stateless loop). Mitigation: its own brainstorm + dedicated plan + mock-draft fallback for offline demo. If the local API isn't runnable in this env, mock fallback keeps every page demoable and the contract is still exercised by `tsc` against `generated.ts`.
