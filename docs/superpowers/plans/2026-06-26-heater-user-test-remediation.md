# HEATER User-Test Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:test-driven-development for every fix (RED → GREEN), superpowers:systematic-debugging where the current code must be traced first, superpowers:verification-before-completion before claiming done. Steps use `- [ ]` tracking.

**Goal:** Fix the open MEDIUM/LOW findings from the 2026-06-26 user-test (`docs/testing/2026-06-26-comprehensive-user-test-report.md`). The two HIGH bugs (lever, optimizer) are already fixed + merged.

**Architecture:** 3 independent clusters by file-ownership so they can be built in parallel isolated worktrees without collision. Backend fixes follow the `api/services/` seam pattern (DB-free synthetic tests, `tests/api/test_*`). Frontend fixes follow `web/` patterns (no test suite → `pnpm exec tsc --noEmit` + `pnpm build` are the gate). M-1 (live projection gap) is an investigation run by the orchestrator, not in these clusters.

**Tech Stack:** FastAPI + pandas (`api/`, `src/`), Next.js 16 + React 19 + TS (`web/`). Test runner: `PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest`. Reconciliation helper: `api.tenancy.normalize_team_name`.

**Verification note:** worktrees get an EMPTY `draft_tool.db` — write DB-free synthetic tests (pattern: `tests/api/test_lineup.py`). The orchestrator does the real-DB end-to-end verification after merge. If a fix changes a response contract, regenerate `api/openapi.json` (snapshot-guarded by `test_openapi_contract.py`).

---

## CLUSTER A — `api/services/team_service.py` (one file; 4 fixes, sequential)

Owns: `api/services/team_service.py` + `tests/api/test_*team*.py`. Read the file first (it's ~560 lines).

### Task A1 (L-4): reconcile team_name in `_rank_and_record` + `_roster`
- **Root cause:** `_rank_and_record` (`:355-405`) and `_roster` (`:490-503`) match `team_name` with exact `==` against `league_records`/`league_rosters`. A bare "Team Hickey" never matches the Yahoo "🏆 Team Hickey" → 0-0-0 / empty roster. `playoff_service` already uses `src.auth._normalize_team_name` and resolves correctly.
- **Fix:** match using `api.tenancy.normalize_team_name(...)` on both sides (records, standings, league_rosters) so emoji/whitespace/case differences reconcile. Keep an exact-match fast path.
- **Test (DB-free):** feed a `records` frame with `team_name="🏆 Team Hickey"` and call `_rank_and_record(standings, records, "Team Hickey")` → expect rank + "W-L-T" resolved (not `(0,"0-0-0")`). Same for `_roster` via a `load_league_rosters` monkeypatch.

### Task A2 (M-7): populate `matchup.opp_name`
- **Root cause:** `_matchup` (`:407+`) — `MatchupHero.opp_name` returns None even when an opponent exists; the hero reads `opponent` elsewhere. Trace where the opponent name lives in the raw `get_matchup()` dict / `_matchup` build.
- **Fix:** set `opp_name` from the resolved opponent (the same source the hero's `opponent` uses). Keep None only when there's genuinely no opponent.
- **Test:** `_matchup({...opponent fields...}, cfg)` → `hero.opp_name == "<opp>"`; empty matchup → None.

### Task A3 (L-1): News status chip — count distinct recent, not all rows
- **Root cause:** `_news_count` (`~:601`) does `SELECT COUNT(*) FROM player_news WHERE player_id IN (...)` → 930/1490 (all history).
- **Fix:** count distinct players with news in a recency window (e.g. `COUNT(DISTINCT player_id)` within the last N days), so the chip reads a sane small number.
- **Test:** monkeypatch the query/connection to a synthetic `player_news` frame with duplicate + old rows → assert the count is distinct + recency-bounded.

### Task A4 (L-2): `categories[].win_prob` always 0.0 — drop the dead field
- **Root cause:** `~:482` reads `c.get("win_prob", 0.0)` from the raw Yahoo matchup categories, which never carry per-category win_prob → always 0.0. Computing it properly is out of scope (needs the H2H engine per category).
- **Fix (minimal):** since it's always 0.0 and unused meaningfully, set the per-category `win_prob` only when present, else omit/None — and confirm the frontend doesn't depend on a 0.0 (it doesn't render it). If the contract requires a float, leave the field but document it as not-yet-computed (`float | None`), regen openapi if the type changes.
- **Test:** assert `_build_categories(...)` emits None/omitted win_prob (not a misleading 0.0) for the raw-matchup shape.

### Task A5: commit cluster A
- `git add api/services/team_service.py tests/api/...` → `git commit -m "fix(api): team_service — reconcile team_name, opp_name, news chip, win_prob (L-4/M-7/L-1/L-2)"`

---

## CLUSTER B — `api/services/{playoff,standings,trade_finder}_service.py` (3 files; 3 fixes)

Owns those 3 services + their tests. Distinct files → no intra-cluster conflict.

### Task B1 (M-6): playoff `projected_record` string == `projected_record_wlt` object
- **Root cause:** `api/services/playoff_service.py:196-197` — the string uses `f"{lo:.0f}-{t:.0f}"` (round-half-even) while the structured `_wlt` uses `int(...)` (truncate) → off-by-1 on the .5 boundary for ~11/12 teams.
- **Fix:** build BOTH from the same rounded ints (`w=round(...)`, `lo=round(...)`, `t=round(...)`; then `f"{w}-{lo}-{t}"` and `Record(wins=w,...)`).
- **Test:** feed fractional projected w/l/t (e.g. 4.5) → assert the string and the `_wlt` object agree exactly.

### Task B2 (L-3): filter ghost teams out of standings
- **Root cause:** `api/services/standings_service.py:~132` iterates `standings_df["team_name"].dropna().unique()` with NO filter → a renamed/abandoned team present in `league_standings` (e.g. "Twigs") but absent from `league_records` leaks as a 13th team.
- **Fix:** when `records_df` is non-empty, skip any `team_name` not present in `records_df` (mirror the documented Bug-D `valid_teams` filter). Keep the records-empty fallback un-filtered (cold-start). This also corrects per-category ranks (computed over the valid teams) + `_team_count`.
- **Test:** feed a standings frame with a team not in the records frame → assert it's excluded; with empty records → not filtered.

### Task B3 (M-8): Trade Finder empty + POST 405
- **Investigate first** (systematic-debugging): `api/routers/trade_finder.py` — what METHOD/path does it expose, and does `web/src/lib/trades-data.ts` call it correctly? The live `POST /api/trade-finder` returned 405. Determine whether the route is GET vs POST, or the path differs, or the page's empty "No trade ideas yet" is the by-design button-gated scan (CLAUDE.md notes the full scan is button-gated). Trace the real frontend call.
- **Fix:** align the router method/path with the frontend call (or confirm the empty state is intentional + documented). If it's a genuine method mismatch, fix the router or the client. Add/extend a router test asserting the supported method returns 200 with the contract shape.
- **Test:** `TestClient` call with the correct method → 200 + the trade-finder contract; the wrong method → 405 is fine.

### Task B4: commit cluster B
- `git commit -m "fix(api): playoff rounding parity, standings ghost-team filter, trade-finder method (M-6/L-3/M-8)"`

---

## CLUSTER C — `web/` frontend (5 fixes)

Owns `web/`. Read `web/AGENTS.md` first (this Next.js has breaking changes — read `node_modules/next/dist/docs/` before Next-specific code). Gate: `cd web && pnpm exec tsc --noEmit && pnpm build` (both exit 0).

### Task C1 (M-2 + M-3): stop the hard-load/refresh bounce on league-wide routes + fix unlinked onboarding
- **Root cause (investigate):** hard-loading/refreshing `/optimizer`,`/standings`,`/closers`,`/players` redirects to "/" (client-nav works). For an UNLINKED user, `/standings`/`/closers`/`/players` bounce to the unlinked card and `/research` shows "No leaders" — yet the card says "League-wide views (Standings, Leaders, Players) work in the meantime." Trace the route guard / `usePageData` / the 409-handling that redirects, and why league-wide pages get treated as personalized.
- **Fix:** (M-2) league-wide + personalized routes must survive a hard load / refresh — a refresh of `/optimizer` should reload `/optimizer`, not bounce home. (M-3) league-wide pages (Standings, Research/Leaders, Players, Closers) must render real data for an authenticated user regardless of team-link state (they don't need a team); only TEAM-required routes show the unlinked state. Either make the named league-wide views work, or change the unlinked message to only promise what actually works — prefer making them work.
- **Verify:** `pnpm build`; if possible, reason through the guard. (Orchestrator will live-verify on the deployed site.)

### Task C2 (M-4): route `<title>` correct on hard-load
- **Root cause:** `web/src/components/chrome/DocumentTitle.tsx` sets the per-route title in a `useEffect`, but the root layout's static `metadata.title` "HEATER — My Team" wins on first paint and isn't overridden on a hard load (only on client-nav).
- **Fix:** ensure the per-route title applies on initial load (e.g. run the title-set effect on mount with the current pathname, or move titles to per-route `metadata`/`generateMetadata` where the route is a server boundary, or set `document.title` synchronously). Confirm a hard load of `/standings` yields `document.title === "HEATER — Standings"`.

### Task C3 (L-6): de-duplicate the playoff-odds fetch on the Team page
- **Root cause:** the Team page (`web/src/app/page.tsx` / `web/src/lib/data.ts`) fires `GET /api/playoff-odds` twice on load (observed in the network log).
- **Fix:** dedupe (single fetch / shared query). Trace the two call sites and collapse to one.

### Task C4 (L-8): Trades page week consistency ("WEEK 13" vs Week 14 everywhere else)
- **Root cause:** `web/src/lib/trades-data.ts` (eyebrow "TRADE WORKBENCH · WEEK 13") shows a different/hardcoded week than the live Week 14.
- **Fix:** source the week from the same place the rest of the app uses (the live matchup/week), not a hardcoded/stale value.

### Task C5: commit cluster C
- `git commit -m "fix(web): hard-load route bounce + unlinked onboarding, route titles, playoff double-fetch, trades week (M-2/M-3/M-4/L-6/L-8)"`

---

## Orchestrator-run (NOT a cluster)

### M-1 — live FA value / optimizer projections = 0 (investigation)
- Local FA pool has real `value`; live App B returns `value:0` for all + optimizer `projected:0` → the live pool lacks ROS projections. Trace via Railway App B logs (`deploymentLogs`, per `reference-railway-ops-graphql`) + `refresh_log` whether the projection bootstrap phases (FanGraphs/Marcel) ran/populated on Railway. Outcome: either a code fix (projection pipeline) or a deploy/data action for the owner. Done by the orchestrator with the Railway GraphQL helper.

---

## Self-review
- **Coverage:** M-2 ✓C1, M-3 ✓C1, M-4 ✓C2, M-6 ✓B1, M-7 ✓A2, M-8 ✓B3, L-1 ✓A3, L-2 ✓A4, L-3 ✓B2, L-4 ✓A1, L-6 ✓C3, L-8 ✓C4, M-1 ✓orchestrator. Owner-gated (Clerk keys, M-5 Bubba key, Stripe dedup) + skip-list (L-5/L-7/L-9/L-10) intentionally excluded.
- **Conflict check:** A=team_service only; B=playoff/standings/trade_finder (distinct); C=web only. No shared files across clusters. Possible `api/openapi.json` touch (A4 if win_prob type changes, B3 if route changes) → regen on merge if conflicting.
- **No placeholders:** each task has a root cause + file + fix direction + test assertion; capable agents trace exact line/code via systematic-debugging + TDD.
