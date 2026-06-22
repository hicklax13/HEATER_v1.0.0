# HEATER — Outstanding Work & Code Audit (M0–M5)

**Date:** 2026-06-22
**Type:** Pre-launch completeness + independent code audit. **AUDIT ONLY — nothing fixed; this file is the only artifact.**
**Method:** 8 parallel read-only audit agents (plan↔code mapping + silent-failure / security / multi-user / type-design / test / frontend lenses) → all load-bearing claims re-verified by direct file read + live-endpoint probe.
**Scope verified:** `api/`, `src/` (migration modules), `web/`, `tests/`, every file in `docs/superpowers/specs/` (38) + `docs/superpowers/plans/` (68), `CLAUDE.md`, memory.

## Live-deployment evidence (2026-06-22)

Probed the production API `https://celebrated-respect-production.up.railway.app`:

| Endpoint | Result | Meaning |
|---|---|---|
| `GET /healthz` | **200** | API up |
| `GET /api/me/team` | **401** | **Clerk login-gate is LIVE** (`clerk_configured()` is true in prod) |
| `GET /api/standings` | real teams (BUBBA CROSBY, Baty Babies…) | real DB, real league |
| `GET /api/players/search?q=trout` | Mike Trout `mlb_id 545361` LAA | real pool, mlb_id backfill worked |
| `GET /api/leaders/overall?lens=overall` | James Wood 20 HR / 68 R / .270 | engine wired, real data |

The 401 on `/api/me/team` is the pivotal fact: **the require-login flip is already active**, so the tenancy-fallback gap (HIGH-1 below) and the frontend 401-swallow (HIGH-3) are *live-reachable now*, not hypothetical.

---

## TOP FINDINGS (priority order)

1. **HIGH-1 — Multi-user isolation hole, live-reachable in the launch window.** An authenticated-but-*unassigned* user is served **Team Hickey's** data on every personalized endpoint. Clerk is live + the 12 leaguemates aren't assigned yet (assignment is sequenced LAST) → this is the *current* state for anyone who signs in. `api/tenancy.py:60-63`.
2. **HIGH-3 — Frontend swallows 401/5xx → renders fabricated mock data as if real.** 12 of ~16 data modules `catch {}` → return mock. With the login-gate live, a signed-out visitor sees fake stats instead of a login prompt. `web/src/lib/data.ts:114` + 11 siblings.
3. **HIGH-2 — Matchup page silently misclassifies unknown pitchers as hitters.** `_PITCHER_SLOTS` is defined but unused; the fallback `return False` routes any pool-missing pitcher into the hitter table with no log. `api/services/matchup_service.py:519-520`.
4. **MED — Write endpoints target the token-owner's team, not the caller's.** `roster_write_service.py:9-11` (mitigated today only by Yahoo write-scope being off).
5. **MED — ~10 service entry points swallow exceptions with NO logging** — recreating the exact blind spot behind this project's documented silent-failure history.
6. The substantive *engineering* backlog beyond these is small + isolated; everything else outstanding is **owner-gated infra (M4/M5)** or owner manual tasks (M6).

---

## 1 — M0–M5 PLAN ITEMS STILL OUTSTANDING

### M0 (contract lock) — ✅ COMPLETE
All contract items present + mounted; all 23 claimed endpoints exist (26 M0/M1 actually mounted). No outstanding M0 items. (Doc drift only: CLAUDE.md says "23 endpoints" / "6 require_pro" — actually **26** M0/M1 endpoints and **7** `require_pro` gates; `POST /api/draft/grade` was added and is uncounted in the docs.)

### M1 (14-page React parity) — ✅ COMPLETE
All 14 Streamlit surfaces + 3 added features (Probables, Hitter-Matchups, Bubba) exist as React pages, live-wired with mock fallback, `tsc --noEmit` clean. No missing pages. Structural note (not a gap): Player Compare + Trade Analyzer are **tabs inside `/trades`**, and "Compare" has no top-nav entry (`web/src/components/chrome/TopBar.tsx:27-40`).

### M2 (auth + billing) — ✅ COMPLETE (backend + frontend), activation done
Clerk + Stripe + `require_pro` shipped and env-activated 2026-06-22 (Stripe intentionally unset on beta = free friends). No outstanding *build* items.

### M3 (single-league paid beta) — DEPLOYED; residual items
| Item | Source | Owner-gated vs buildable-now | Notes |
|---|---|---|---|
| Assign the 12 leaguemates (`HEATER_ADMIN_CLERK_IDS` + `POST /api/admin/assignments`) | roadmap M6 / `2026-06-21-heater-m3-beta-launch-plan.md` | **owner-gated** | The endpoint is built (`api/routers/admin.py:23`); needs each user's Clerk id (sign-in first). Explicitly the LAST launch step. **Until it runs, HIGH-1 below is live.** |
| Yahoo `fspt-w` write scope | roadmap M6 | **owner-gated** | App is read-only → live writes 401 (`src/yahoo_api.py`). |
| Frontend "Team Hickey" cleanup | `2026-06-21-heater-b4-m3-1-...md` | **buildable-now** | Standings/Trades fixed (f2decf7); **7 live call-sites still pass `team_name:"Team Hickey"`** as a query param (harmless *only because* the backend overrides it — see HIGH-1). Files: `web/src/lib/{data.ts:109,127, matchup-data.ts:234, punt-data.ts:77, standings-data.ts:86, probables-data.ts:218, hitter-matchups-data.ts:206}`. |

### M4 (multi-tenant backend) — ALL OWNER-GATED, not started
| Item | Source | Status |
|---|---|---|
| B2.2→B2.5 Postgres migration (dialect port → keystone → sole-writer relax → data migration → cutover) | `2026-06-19-heater-backend-b2-postgres-migration-plan.md` ("PLAN ONLY — NOT APPROVED") | **owner-gated** — needs owner go + a real Postgres; touches LIVE read/write paths. (B2.0/B2.1 shipped + dormant.) |
| Real `alembic upgrade head` vs live Postgres | `2026-06-19-...-b2-slice1-alembic-baseline.md` | **owner-gated** (verification deferred) |
| B3 Redis/Arq workers | `2026-06-19-heater-backend-b3-workers-plan.md` ("NOT APPROVED; depends on B2") | **owner-gated** |
| B4 M4-1 `league_id` tagging + request-scoped repo filter + tenant-isolation tests | `2026-06-21-heater-b4-multi-tenancy-...-design.md` | **owner-gated** — verified `league_id` absent from all `api/services/*.py` |
| B4 M4-2 `LeagueConnector` (Sleeper/Yahoo/ESPN/CBS/manual) + per-user encrypted creds + per-tenant refresh workers + per-user Yahoo OAuth | same + roadmap M4 | **owner-gated** — "needs owner product input" (roadmap:73); not planned in detail |
| B4 M4-3 multi-league switch-on | same | **owner-gated** |
| Clerk live end-to-end token test | `2026-06-19-...-clerk-auth-wiring-plan.md` | code shipped + now live (401 verified); formal token test still deferred |

### M5 (public launch) — OWNER-GATED, not started
Open signup on M4 backend + multi-platform connectors + landing/marketing site + **retire Streamlit**. Source: roadmap M5. **owner-gated** (gated on M4 + owner go).

### M6 (owner manual tasks)
Clerk/Stripe/Railway/Vercel **DONE** (owner-confirmed 2026-06-22). Remaining: assign 12 leaguemates, Yahoo `fspt-w`, domain/DNS, public-launch ops (marketing/listings/connector creds) — all **owner-gated**. (CORS is moot — frontend proxies `/api/*` server-side via `web/next.config.ts`.)

### Buildable-now follow-ups behind already-shipped endpoints (low urgency, none block beta)
- Streaming `BudgetStrip.ip_pace` is stubbed `0.0` — wire real weekly IP pace. `api/contracts/streaming.py:50`; `2026-06-19-heater-m0-streaming-widen.md`.
- Per-team **champ odds** absent on `/api/playoff-odds` (only user champ%). `2026-06-20-heater-m0-playoff-odds-endpoint.md` → also surfaces as `adapters.ts:213-214` hardcoding `champOdds:0`.
- Team dashboard `trajectory` + `win_prob_trend` + lineup-status chip — need a per-week snapshot table + cron writer (a future B3 slice). `2026-06-20-heater-m0-team-dashboard.md`; live UI hides them (`web/src/lib/api/adapters.ts:467-468`).
- Leaders breakout/sell lens is thin (~1 hitter) — merge pool `xwoba_delta` into season_stats. `2026-06-20-heater-m0-leaders-overall-bugfixes.md`.
- Stripe webhook **event-id dedup + ordering guard**, and a `past_due` grace window. `2026-06-20-heater-m2-stripe-billing-design.md`.
- Draft simulator post-draft **A–F category grade** + "Undo last pick" (React). `2026-06-20-heater-draft-simulator-design.md` (frontend hardcodes `stats:[]`-style gaps).
- Hitter/pitcher matchup grids: optional **L14 recent-form blend** + linear→sigmoid difficulty calibration. `2026-06-21-heater-hitter-matchup-scorer-design.md`.
- Exact point-in-time stream **replay** via a scheduler-written nightly snapshot (currently YTD-proxy, disclosed in-UI). `2026-06-09-pitcher-streaming-analyzer-design.md`.
- Full-surface **OpenAPI→TS** type generation (today `web/src/lib/api/generated.ts` is generated from the committed snapshot; keep regenerating on contract changes). `2026-06-19-frontend-api-wiring-slice0-design.md`.
- Closer "Saves Finder" projected SV/ERA/WHIP — `web/src/lib/api/adapters.ts:649` hardcodes `stats:[]`. Deferred CEO backend item.

---

## 2 — CODE ISSUES FOUND INDEPENDENTLY (not captured by any plan)

### HIGH

**HIGH-1 — Tenancy fallback exposes Team Hickey's data to any authenticated-but-unassigned user.**
`api/tenancy.py:60-63` — `effective_team(self, fallback) -> self.team_name or fallback`. For a logged-in user with no membership, `team_name is None`, so `effective_team("Team Hickey")` (the client query-param) returns `"Team Hickey"`. Every personalized service then slices that team (e.g. `team_service.py:367` `lr[lr["team_name"]==team_name]`). The docstring (`tenancy.py:53-54,73-74`) *claims* "never another user's team" — the code contradicts it. **Live-reachable now** (Clerk is on; the 12 are unassigned until the last launch step), and worse post-M4 (a paying stranger with no assignment → sees Team Hickey). Bounded: read-only, exposes only the one configured owner team, self-closes on assignment. The regression test `tests/api/test_api_tenancy_resolver.py:95-106` currently *asserts the fallback*, so it locks the wrong behavior.
**Severity: High (bounded).** **Action:** when `clerk_configured()` AND an identity exists but membership is None, resolve to None and do NOT fall back to the client param (return a "no team assigned" empty state); invert the test assertion for the Clerk-on case.

**HIGH-2 — Matchup page silently classifies unknown pitchers as hitters.**
`api/services/matchup_service.py:504-520` (`_is_pitcher_by_pool`). A `_PITCHER_SLOTS = {"SP","RP","P"}` set is defined (line 502) but the fallback ignores it: any player absent from the pool, or with `is_hitter is None`, hits `return False` (= hitter). Comment says "pitcher-leaning default" — code does the opposite. A pool-missing rostered pitcher is routed into the hitter table (blank/garbage line) and dropped from `pitcher_totals`, with no log. Same family as the documented `roster_slot`-vs-`selected_position` bug.
**Severity: High** (visible wrong data on a core page, no observability). **Action:** classify by the slot string via `_PITCHER_SLOTS` on pool-miss and `logger.debug` the unresolved pid.

**HIGH-3 — Frontend swallows live 401/5xx/network errors and renders fabricated mock data as real.**
`web/src/lib/use-page-data.ts` only reaches its `error` state if the fetcher rethrows — and most don't. Bare `catch {}` → return page mock: `web/src/lib/data.ts:114` (My Team landing — Judge "3 HR/7 RBI", playoff 12%), `matchup-data.ts:237`, `standings-data.ts:96`, `players-data.ts:84`, `streaming-data.ts:220`, `closers-data.ts:85`, `punt-data.ts:80`, `probables-data.ts:221`, `hitter-matchups-data.ts:208`. Compare (`compare-data.ts:116`) and Draft (`draft-data.ts:113,136`) substitute fabricated stat lines/advice for the *real* players the user picked. **401 is swallowed everywhere** — `web/src/lib/api/errors.ts:21` exports `isAuthRequired` but no fetcher imports it; with the login-gate live, a signed-out visitor sees fake data instead of a login redirect.
**Severity: High** (data-integrity + the login-gate is already live). **Action:** rethrow non-2xx (or at least 401/5xx) so `usePageData` reaches `error`; wire `isAuthRequired` → login.

### MEDIUM

**MED-1 — Write endpoints target the token-owner's team, not the caller's.**
`api/services/roster_write_service.py:9-11,34-43` — `_client()` returns the server's single `get_yahoo_data_service()._client` (Team Hickey's); `set_lineup`/`add_drop` never use the viewer context. `api/routers/roster_write.py:24-38` gates with `require_principal` (authentication) but not authorization-of-team. Any authenticated caller would mutate the owner's roster. Mitigated today only by Yahoo `fspt-w` being off (writes 401) — a config flip from going live. Documented as the M4 seam, but it ships now.
**Severity: Med.** **Action:** until per-user Yahoo OAuth lands, refuse writes when `ctx.team_name` ≠ the token-owner's team.

**MED-2 — ~10 service entry points swallow exceptions with NO logging** (the historical silent-failure mechanism — a code defect is indistinguishable from "cold/empty" in the operator log).
`api/services/fa_service.py:30-31`, `leaders_service.py:82-83`, `standings_service.py:21-22`, `streaming_service.py:161-162` + `255-257` (`analyze_pitcher` → `found=False`), `matchup_service.py:287-288` + `319-320`, `punt_service.py:61-63`, `closers_service.py:28-29`, `draft_service.py:49-54,79-88,116-117` (also *claims* "no pool data" for any failure). The well-behaved siblings prove the fix is house style (`leaders_overall_service.py:166`, `fa_pool_service.py:124`, `playoff_service.py:55`, `trade_service.py:55`, `compare_service.py:77`, `databank_service.py:68`, `team_service`). `streaming_service` + `punt_service` lack even a module logger.
**Severity: Med.** **Action:** add `logger.warning("<svc>.<method> failed: %s", exc)` to each (keep returning empty — just stop returning empty *silently*).

**MED-3 — Write path loses the traceback.**
`api/services/roster_write_service.py:45-53` — the single mutation path converts any exception to `MutationResult(ok=False, error="Write failed: <ClassName>")` with no `logger`/`exc_info`. When a leaguemate's "Apply to Yahoo" fails (token expiry vs `fspt-w` scope vs XML drift), the operator has nothing. **Action:** `logger.warning(..., exc_info=True)` before returning; also log the graceful `ok=False` passthrough in `_to_result`.

**MED-4 — `PuntService.get_punt` + `StandingsService._build_teams` have no real-class test coverage.**
`tests/api/test_punt.py` / `test_standings.py` exercise only the contract shape + a fake service. The real engine→contract mapping is untested: `api/services/punt_service.py:16-69` (rank/gainable extraction, is_punt branch) and `api/services/standings_service.py:25-99` (parses `"W-L-T"` string split, OVERALL/RECORD row detection, NaN guards). Both are core launch surfaces; a mis-mapped field ships green. This is the exact "false-green synthetic test" class CLAUDE.md documents for optimizer daily-slice2. **Action:** add static-method unit tests like `tests/api/test_lineup.py` (no DB needed).

**MED-5 — `api/services/live_boxscore.py` has zero direct tests.**
Real parsing logic (`_hitter_line`/`_pitcher_line`, AVG/OBP math, statsapi traversal, TTL cache, a `reset_cache()` hook nothing calls) feeding the live Matchup page. Exposure bounded (live-only/deferred path) but it's the largest untested real-logic service. **Action:** synthetic-shape tests for the line parsers.

**MED-6 — OpsCards renders a raw float on the flagship dashboard.**
`web/src/components/myteam/OpsCards.tsx:36,40` render `{card.value}` / `{card.total}` raw; `OpsCard.value/total` are floats (`web/src/lib/types.ts:58-59`) passed through unrounded by `web/src/lib/api/adapters.ts:430-439`. The `ip_pace` card → `text-3xl` hero like `18.2345 / 53.80001` on Railway (only with live Yahoo data, so local smoke missed it). **Action:** `.toFixed(1)` in `toOpsCard`.

**MED-7 — Draft + Databank pages have no error state.**
`web/src/app/draft/page.tsx:24-27` (`useDraft` has no `error` phase → a failed `start()` silently resets to setup; a failed `pick()` leaves the optimistic pick). `web/src/app/databank/page.tsx:22-41` (no `.catch` → outage looks like "No history found"). **Action:** add error/retry states.

**MED-8 — Accessibility gaps on a soon-public product.**
- Matchup per-category winner is **color-only** (`web/src/app/matchup/page.tsx:226`, `bg-heat/10`) — invisible to colorblind + screen readers; the page's primary signal. 
- Matchup date "tabs" are inert `<span>`s — unreachable by keyboard AND a functional dead control (`matchup/page.tsx:143-153`).
- `AnalyzeStarter` `<select>` has no label (`web/src/components/streaming/AnalyzeStarter.tsx:67-77`).
- Bubba dialog: no focus-move-in, no focus-restore, no Escape-to-close (`web/src/components/bubba/Bubba.tsx:403-413`).
- Unlabeled inputs: Bubba composer/API-key, players + research search (icon-only `<label>`), CommandPalette.
- Tables missing `scope`/`<th>`: `StandingsTable.tsx:16-28`, `ComparePanel.tsx:203-229` (no `<thead>` at all). No skip-to-content link in `web/src/app/layout.tsx`.
**Severity: Med.** **Action:** add non-color cue + real buttons on Matchup first (highest traffic).

### LOW

- **LOW-1 — TopBar hardcodes the user identity.** `web/src/components/chrome/TopBar.tsx:184-185` shows literal "Connor Hickey / Team Hickey" in the account dropdown (live render path, not wired to Clerk `user`/`getViewerTeam()`). Every leaguemate will see Connor's name. Cosmetic, no data path. **Action:** wire to the resolved identity.
- **LOW-2 — Local dev env is off the pin.** `requirements.txt:60` pins `fastapi==0.137.1` but the local interpreter has `0.126.0`, so `tests/api/test_openapi_contract.py::test_openapi_snapshot_is_current` fails *locally only* (FastAPI auto-generates the `ValidationError` schema; older version omits `ctx`/`input`). CI uses the pin → green; no contract drift (diff = 0 paths/schemas changed). **Action:** `pip install -r requirements.txt` locally; not a code defect.
- **LOW-3 — `src/ai/keys.py:67-72`** swallows a decryption failure (rotated/corrupt Fernet) as "no key" → user told to add a key they already have; no log; `return None` makes the admin-shared-key fallback line unreachable on that path.
- **LOW-4 — `src/user_data.py:70-83,120-133,178-183`** member writes (`add_to_watchlist`/`save_view`/…) ignore `_write`'s success bool and return `None` → UI shows success on a swallowed save (the write IS logged, so operator-visible). Propagate the bool like `chat_service.save_prompt`.
- **LOW-5 — `api/services/team_service.py:101-106`** the flagship `/api/me/team` calls `yds.get_matchup()`/`get_standings()` outside any try/except, contradicting the module's "never raise" docstring (a Yahoo-singleton failure 500s the dashboard). Either wrap or fix the docstring.
- **LOW-6 — `streaming_service.analyze_pitcher`** swallows to `found=False` (`streaming_service.py:255-257`) — an engine error masquerades as a specific "not found" result. Folds into MED-2's logging fix; consider a distinct `error` state.
- **LOW-7 — Stale skip-guards** that go quiet instead of red if a symbol is renamed: `tests/test_yahoo_schedule.py:39,62,75` (the imported DB fns now exist), `tests/test_fa_add_drop_yahoo.py` (6 sites). Convert the existence checks to hard asserts.
- **LOW-8 — DB-integrity guards don't run in CI/worktree** (empty DB): `tests/test_no_relevant_null_mlb_ids.py`, `test_no_shadow_player_rows.py`, `test_task54_fixes.py` skip without the real 26 MB DB (documented `reference_worktree_empty_db`). Launch-blocking data-integrity regressions are only catchable by a manual run in the main checkout.
- **LOW-9 — Stale doc-comments (drift):** `web/src/lib/standings-data.ts:7-10` claims playoff odds are "mock-only" (they're fetched at :86); `matchup-data.ts:4-7` claims `league` "not in the contract" (it's adapted from the response). CLAUDE.md "23 endpoints"/"6 require_pro" are stale (really 26/7).

### Type-design hardening (the recurring case-mismatch bug class — `chat.py` already shows the `Literal[...]` fix)
- `api/contracts/streaming.py:27-28` `status/confidence: str = ""` (raw engine case, frontend lowercases) → `Literal[...]`.
- `api/contracts/matchup.py:24` `MatchPlayer.stats: list[str]` + `SideTotals` `list[str]` — order-coupled to a separate columns list; drift = silent cell mismatch → `list[StatItem]`.
- `api/contracts/matchup.py:16` `MatchupCategory.win: str=""`; `my_team.py:34-36,45,72` `trend/tag/status` str enums (and `OpsCard` uses "danger" while others use "warn"/"info" — inconsistent vocab).
- `api/contracts/playoff.py:18` + `matchup.py:37` `record:"W-L-T"` strings re-split downstream → structured `{wins,losses,ties}`.
- `api/contracts/streaming.py:50` `ip_pace: float = 0.0` for a *deferred* field → `float | None = None` so "not computed" ≠ "0".
- `api/contracts/chat.py:32` `tool_trace: list[dict]` — the lone untyped dict in a response.

---

## 3 — DEFERRED-BY-DESIGN (NOT bugs — cross-checked vs CLAUDE.md "Known Design Choices")

- **`src/db/engine.py` `get_connection()` raises `NotImplementedError` for non-SQLite `DATABASE_URL`** — B2.2 Postgres wiring is intentionally paused → M4. Dormant + SQLite-byte-identical. (CLAUDE.md B2 status.)
- **`scripts/migrate_muncy_dna_2026_05_21.py` obsolete** — the fix is self-healing in `upsert_player_bulk`/`deduplicate_players`. Kept for reference; safe to delete eventually. Not imported. (CLAUDE.md.)
- **`src/contextual_factors.py` / `src/weekly_report.py` / `src/waiver_wire.py` are NOT dead** despite removed pages — actively imported by `fa_recommender.py:38`, `trend_tracker.py:36`, `draft_engine.py:830`, `daily_optimizer.py:1101`, `war_room_actions.py:173`, `pages/1_My_Team.py`. Do not remove.
- **`_f()` / `_safe_float` NaN/inf → default coercion** throughout `api/services/` — required for valid JSON (RFC-8259 bans NaN/inf), not value-hiding. The engines own correctness.
- **Bubba metering under-count** (partial-answer tokens on a mid-stream exception not billed) — documented, logged loudly. `api/services/chat_service.py:241-244`.
- **Bubba degraded over-grant** — a subscription-read failure → admin cap (≈$0.90/user/day over a free user's $0.10); bounded + WARNING-logged. `api/gating.py:61-66`.
- **Chat persistence in `draft_tool.db`** (not `api_state.db`) — documented B-phase follow-up; the `ai_*` FK is declared-unenforced. `chat_service.py:24-29`.
- **Bubba native vision-PDF deferred** (PDFs go as extracted text — universal); **Plus/Max tiers + monthly free taste deferred until Pro sells** (`api/services/ai_allowance.py:3-4`).
- **Matchup live in-game box scores** beyond projected lines + champ-odds league-wide + Team trajectory/win-prob-trend — all need a data source/snapshot table that doesn't exist yet; documented deferrals.
- **External limitations already catalogued** (FanGraphs 403, `bat_speed` unavailable, `season_stats` partial scope, news canonical-name collisions, playing-time grace period, punt-weight 0.05, etc.) — see CLAUDE.md "Known Design Choices"; do not re-flag.

---

## 4 — VERIFIED COMPLETE (evidence-backed confidence)

- **Security core is sound — fail-closed everywhere (no fail-open gate found).** Clerk verifier RS256/iss/aud, every exception → 401 (`api/auth.py:99-163`); `EnvTokenVerifier` deny-by-default + `hmac.compare_digest` (`auth.py:70-93`); Stripe webhook fail-closed 400 + link-only `checkout.session.completed` so out-of-order can't re-activate a canceled user (`billing_service.py:135-178`); CORS explicit allowlist, never `*` (`api/main.py:17-36`); admin assign deny-by-default, empty `HEATER_ADMIN_CLERK_IDS` ⇒ no admins (`api/admin.py:21-32`); write routes carry `require_principal` (`roster_write.py:24-38`); `.env` is gitignored + untracked; no token/secret is logged (only exception class names + opaque subject ids).
- **The PRIMARY isolation property is correct + regression-locked:** an *assigned* logged-in user passing `?team_name=Team%20Hickey` gets **their own** team — the resolved value overrides the client param (`tenancy.py:86-92`; tests `test_api_tenancy_resolver.py:80-92`, `test_api_personalized_team_resolution.py:51-64`). HIGH-1 is strictly the *unassigned* edge.
- **API stores never touch the live `draft_tool.db`** — all 5 (`user/league/membership/subscription/prompt`) use `HEATER_API_DB_PATH` → `data/api_state.db`. No cross-write risk to the league.
- **All 9 personalized routers route through `ctx.effective_team(...)`**; routers are logic-free (no `src.` imports; AST-guarded `test_no_logic_in_routers.py` passes); every `src/` import is inside the `api/services/` seam.
- **M0/M1/M2 are genuinely complete in code** (not just claimed): 26 endpoints mounted, 14 React surfaces live-wired with the 4-state machine, paywall on all 5 Pro surfaces, M2 auth/billing UI env-gated-dormant and correct. `web/` `tsc --noEmit` clean.
- **Bubba arc B1→B3 complete + mounted**; the live Streamlit `chat()` path is guarded by a frozen-reference equivalence test (`tests/test_ai_providers_streaming.py:79-105`); Bubba is read-only on the league.
- **Probable-Pitcher + Hitter-Matchup grids complete full-stack** (endpoints + pages + nav + `compute_hitter_matchup_score`). The roadmap's "NOT built" line is stale.
- **`player_id_resolver` + `_enrich_mlb_ids`** are exemplary error handling (distinguish outage from absence → `refresh_log` error; Muncy-DNA-safe) — the opposite of a silent failure.
- **`tests/api/` = 416 passed, 1 failed** — the 1 failure is the openapi snapshot env-artifact (LOW-2), not a regression.
- **Live deployment serves real data** end-to-end with the Clerk login-gate active (probes above).

---

### Bottom line
The migration is in strong shape: M0–M2 complete, M1 parity real, security fail-closed, the primary multi-user property correct and tested. The **few launch-relevant defects** are HIGH-1 (unassigned-viewer fallback — close it before/with the 12-leaguemate assignment), HIGH-3 (frontend 401/5xx swallow — wire before relying on the live gate), and HIGH-2 (matchup pitcher misclassification). The rest is owner-gated infra (M4/M5), owner manual tasks (M6), the MED observability/coverage hardening, and ~11 small buildable-now follow-ups behind already-shipping endpoints — none blocking the beta.
