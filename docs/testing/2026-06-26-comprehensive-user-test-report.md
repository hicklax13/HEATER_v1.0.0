# HEATER — Comprehensive Autonomous User-Test Report

**Date:** 2026-06-26 · **Branch:** master (HEAD `0444ad2`) · **Tester:** autonomous (Opus)

**Environments exercised:**
- **LOCAL stack** — React `web/` (Claude_Preview) → local FastAPI `api/` on :8000 (Clerk OFF, reads open) → real `draft_tool.db` **refreshed from the live Yahoo league this session** (317 rosters, 12 W-L records, 12 matchups, 608 FAs).
- **LIVE production** — Vercel `heater-v1-0-1.vercel.app` + Railway API `celebrated-respect-production.up.railway.app`, driven through **Connor's authenticated Chrome** (Claude in Chrome MCP). Connor's account was **unlinked**, so mid-session — with his explicit consent — I assigned it to **🏆 Team Hickey** via `POST /api/admin/assignments` to unlock the real-data live pass.
- **7 parallel agents** — A1 (Team/Standings/Playoff API), B1 (re-verify `Bugs_June26`), B2 (re-verify `Outstanding_June26`), B3 (security) **completed**; A2/A3/A4 (matchup/optimizer · trades/FA · streaming/leaders/draft API depth) were still running against the contended local API at report time (their scope was substantially covered independently via live + local authed probes).

> ⚠️ **State change made this session (needs your awareness):** I **linked your live Clerk account to 🏆 Team Hickey** (`validated:true`) so I could test the real authenticated experience. You are now 1 of 12 assigned; the other 11 leaguemates remain unlinked. Unassign/keep as you prefer (`api/admin` assignments).

---

## Executive summary & beta-readiness verdict

**Verdict: close to beta-ready — both HIGH bugs are now FIXED + merged this session; remaining launch items are the live projection gap + onboarding polish.** The foundation is genuinely strong: security is fail-closed (no VULN found), the auth/unlinked/error states are correctly wired (no mock-as-real leaks), records/standings/matchup data is correct for assigned users, and the documented `Bugs_June26` / `Outstanding_June26` HIGH+MED fixes are all genuinely shipped (verified behaviorally, not just by code-read). Most pages render real data on live.

**The two flagship in-season defects — both FIXED this session:**

1. **🔴→✅ HIGH — Lineup Optimizer was non-functional (empty lineup in every environment).** Fixed (`b452155`): now fetches the enriched team roster → real lineups (18 standard / 14 daily, verified end-to-end).
2. **🔴→✅ HIGH — The Team page's "lever" gave wrong-category advice** (always "Stolen bases"). Fixed + deployed + live-verified (`9b55124`): now shows the real weakest category.

**The launch-relevant mediums (still open):**
- **🟠 Live-only: FA "value" and optimizer projections render 0** for every player on production (the analytical differentiator shows 0) — a live data-pipeline gap, not a code bug.
- **🟠 The unlinked onboarding experience is broken** — every friend hits it *before* you assign them, and it contradicts its own message.
- **🟠 Bubba shows "No models — add a key" on live** (no managed AI configured) — the assistant doesn't work out-of-box.
- **🟠 Refresh / deeplink / bookmark of several core pages bounces to "/".**

A reasonable path: ~~fix the lever~~ (done), ~~fix the optimizer~~ (done) — then investigate the live projection gap (M-1), decide on Bubba's managed key (M-5), and smooth the unlinked onboarding (M-3) before inviting the 12.

---

## Findings by severity

### 🔴 HIGH

#### H-1 · Lineup Optimizer returns an empty lineup (core page non-functional) — ALL environments
- **Surface:** `/optimizer`; `POST /api/lineup/optimize`.
- **Problem:** The page shows **"No lineup to optimize — We couldn't find your roster for today."** The API returns **0 slots / 0 starters** with `summary: "Lineup unavailable (no live data in this environment)."` and a **316-player bench** (the entire league).
- **Evidence (live, authed):** `optimize {team_name:"🏆 Team Hickey", mode:"standard"}` → `{slots:0, bench:316, summary:"Lineup unavailable…", benchSample:[Hunter Goodman/COL, Kazuma Okamoto/TOR, Ketel Marte/AZ]}` (5.8s daily / 2.7s standard). **Reproduced locally** with the correct emoji team name → identical `slots:0, bench:317`.
- **Root cause:** `api/services/lineup_service.py:50` (`_optimize_standard`) and `:92` (`_optimize_daily`) do `roster = yds.get_rosters()` — the **entire 12-team league (316 players), with no projection columns** — and pass it straight to `LineupOptimizerPipeline(roster, …)`. The accepted `team_name` param is **never used to filter** the roster. Worse, even a roster *filtered to Team Hickey's 26 players* still yields empty assignments (verified in-process: `pipeline.optimize()` → `assignments:[]`, status `"Optimal"`) — because the bare `league_rosters` rows carry **no projections**, so the LP objective is 0 and it starts nobody. The Streamlit optimizer (`pages/2_Line-up_Optimizer.py`) works because it enriches the roster + forwards `confirmed_lineups`/`recent_form`/`team_strength`; the API path replicates none of that. The "no live data" summary is a **misleading default** (set unconditionally at line 40, only overridden when slots exist) — it is NOT actually about live data.
- **Severity:** HIGH — the flagship daily-action page has effectively never produced a real lineup in the new product. Masked until now because local tests used an empty Yahoo roster.
- **Fix — DONE (master `b452155`):** both `_optimize_standard` and `_optimize_daily` now fetch the user's ENRICHED, team-filtered roster via `yds.get_team_roster(team_name)` (the players/season_stats/projections JOIN in `league_manager.get_team_roster`) — the same roster the working Streamlit optimizer uses — instead of the bare all-teams `get_rosters()`. Also fixed a placeholder-name issue exposed once the lineup populated (the standard LP labels its vars "P0"/"P1"; `_to_slots` now resolves the real name from the pool by player_id for those). **Verified end-to-end against the real DB: standard → 18 starters (real names + mlb_ids), daily → 14 starters; 3 new DB-free regression tests; full api suite 590 green.**

#### H-2 · Team-page "lever" hardcodes "Stolen bases" → wrong-category advice to every user
- **Surface:** `/` (Team dashboard), the single primary CTA.
- **Problem:** The lever always renders **"Stolen bases"** and **"You're N stolen bases behind … Three free agents can close most of it … See The 3 Pickups"** regardless of the user's actual weakest category.
- **Evidence (live, authed):** `GET /api/me/team` → `lever:{category_key:"K", headline:"K is your weakest category", behind_by:21, pickups:[]}`, yet the rendered card says *"Stolen bases / 21 stolen bases behind / Three free agents / See The 3 Pickups."* Two bugs compounded: (a) wrong category label, (b) the card claims "Three free agents / See The 3 Pickups" while `pickups` is **empty**.
- **Root cause:** `web/src/components/myteam/LeverCard.tsx` — line 38 hardcodes the chip `Stolen bases`, lines 44-45 hardcode `stolen bases behind` + `Three free agents…`. The component **accepts the `headline` prop but never uses it** (`page.tsx:113` correctly passes the real `headline`; the adapter `adapters.ts:462-464` correctly maps `categoryKey`/`headline`/`behindBy`). The pickup count "3" is also hardcoded.
- **Severity:** HIGH — flagship page, live-reachable for every real user, gives factually wrong fantasy advice.
- **Fix (simple, low-risk):** render the real category from `headline`/`categoryKey` + a category→unit label map, and derive the pickup count/CTA from `pickups.length` (hide the avatar row + adjust copy when empty). **Fixed in-session** (see "Fixes applied").

---

### 🟠 MEDIUM

#### M-1 · Live-only: FA "value" and optimizer projections render 0 for every player
- **Surface:** `/players` (FA pool), `/optimizer`; live only.
- **Problem:** On live, the Players page renders 100 FAs with **real YTD stats but `value:0` for every one** (TOP PICKUP "Trevor Rogers … VALUE 0"); the optimizer bench shows `projected:0` for all.
- **Evidence:** live `GET /api/free-agents/pool` → `free_agents[*].value === 0` (stats real: `K:54 ERA:5.30 WHIP:1.36`). **Local is correct** — same endpoint returns `value: 100.0 (Gabriel Moreno), 79.7, 77.1, 71.2 …`. So the value=0 is **live-specific**.
- **Root cause (suspected):** the live App B **player pool lacks ROS projections** (marginal-SGP value needs projections; YTD-only ≠ value). Likely the projection bootstrap phases didn't populate on Railway (FanGraphs 403 + Marcel/projection fallback not running), so projection-derived value collapses to 0 while YTD-derived signals (the lever's category gaps) still work. **Investigate App B's projection sync / refresh_log.**
- **Severity:** MED-HIGH — the value-add scoring (HEATER's differentiator) is meaningless on production while looking populated.

#### M-2 · Hard-load / refresh / deeplink of several core pages bounces to "/"
- **Surface:** `/optimizer`, `/standings`, `/closers`, `/players` (likely all personalized + some league-wide).
- **Problem:** Navigating directly to (or **refreshing while on**) these routes redirects to the Team page. **Confirmed:** reload on `/optimizer` → `bouncedHome:true`. **Client-side nav (clicking the top-nav link) works fine** and even sets the correct title.
- **Impact:** bookmarks break; a user who refreshes the page they're on is kicked home. Normal click-navigation is unaffected.
- **Root cause:** a route guard / page-data redirect fires on initial hard load (a Clerk/data race or a redirect on the first fetch). Needs a frontend trace.
- **Severity:** MED.

#### M-3 · Unlinked onboarding is broken and self-contradictory (every friend hits this pre-assignment)
- **Surface:** all pages, authenticated-but-unassigned state (the state of all 12 friends until you assign them — the documented LAST launch step).
- **Problem:** The unlinked card says *"Your team isn't linked yet. League-wide views (Standings, Leaders, Players) work in the meantime."* — but, for an unlinked user: **Standings, Closers, Players bounce to the unlinked card**, **Research/Leaders shows "No leaders to show,"** and the **⌘K command palette returns "No results found" for both players AND page-jumps.** So the three pages the message explicitly names do **not** work, and the global search is dead.
- **Evidence (live, Connor pre-assignment):** `/standings`→bounce, `/closers`→bounce (`/api/me/team`→409), `/players`→bounce, `/research`→"No leaders," palette "trout"/"standing"→"No results found."
- **Severity:** MED — it's the literal first experience of every beta user. (Note: this is *also* why I had to assign Connor to test anything real.)

#### M-4 · Route `<title>` is "HEATER — My Team" on every hard-loaded page
- **Surface:** all routes, on initial load / refresh / shared link.
- **Evidence (live):** `/sign-in`, `/research`, `/databank`, `/draft`, `/standings` all show `document.title === "HEATER — My Team"`. **Client-side nav fixes it** (clicking Optimizer → "HEATER — Lineup Optimizer").
- **Root cause:** `web/src/components/chrome/DocumentTitle.tsx` sets the per-route title in a `useEffect`, but the root layout's static `metadata.title` "HEATER — My Team" wins on first paint and isn't overridden on hard load. (Codex C-UI-01.)
- **Severity:** MED-LOW — cosmetic tab title; affects shared-link/SEO previews.

#### M-5 · Bubba shows "No models — add a key" on live
- **Surface:** Bubba (every page), live.
- **Problem:** For a normal user, the assistant panel says *"No models — add a key / Add your own provider API key (gear icon) to start chatting."* — no managed AI is available out-of-box.
- **Root cause:** the live App B has no admin AI provider key configured (or the managed-tier path isn't surfacing one), so there are zero models. May be an intentional beta cost decision, but the UX is abrupt for a flagship feature.
- **Severity:** MED.

#### M-6 · playoff-odds `projected_record` string ≠ `projected_record_wlt` object (A1)
- **Surface:** `GET /api/playoff-odds`.
- **Evidence:** Over the Rembow `projected_record:"18-4-2"` vs `projected_record_wlt:{18,4,1}`; ~11 of 12 teams mismatch by 1 loss/tie.
- **Root cause:** `api/services/playoff_service.py:196-197` — string uses `f"{x:.0f}"` (round-half-even) while the object uses `int()` (truncate); the sim returns fractional W/L/T so they diverge on the .5 boundary.
- **Fix:** build both from the same rounded ints. **Severity:** MED.

#### M-7 · `/api/me/team` matchup is token-owner-scoped + perf-heavy (A1 + perf)
- **`opp_name` returns `None`** from me/team (the hero opponent is sourced from a separate `opponent` field); and the matchup always reflects the **token-owner's** cached matchup, not the requested team (acceptable single-league; a multi-tenant gap). `team_service.py:116`.
- **Perf:** `GET /api/me/team` measured **17–59s locally** — it runs `build_optimizer_context` synchronously (lever + ops) on every landing-page load. On the single Railway replica, 12 users opening the app at once would serialize heavy compute. Frontend's error-state fallback caught the slow case correctly, but this needs caching / a worker before scale. **Severity:** MED.

#### M-8 · Trade Finder empty on live + POST returns 405
- **Surface:** `/trades` (Finder tab); `POST /api/trade-finder`.
- **Evidence:** page shows "No trade ideas yet — We need a bit more league data"; my authed `POST /api/trade-finder` → **405**. The empty state may be the by-design button-gated scan (per CLAUDE.md the full scan is button-gated), but the 405 indicates a method/route mismatch worth confirming against the real frontend call. **Severity:** MED/verify.

---

### 🟡 LOW

- **L-1 · News status chip counts ALL historical news rows** (1490 local / 930 live), not distinct/recent players. `team_service.py:601` (`SELECT COUNT(*) FROM player_news WHERE player_id IN (...)`). Meaningless badge value.
- **L-2 · me/team `categories[].win_prob` always 0.0** (A1). `team_service.py:482` reads a per-category win prob the raw Yahoo matchup doesn't carry.
- **L-3 · Ghost team "Twigs" in LOCAL standings (13 teams).** `api/services/standings_service.py:132` doesn't filter the team loop to records-present teams. **LIVE has exactly 12 (no Twigs)** — so it's a **local stale-data artifact** of my Yahoo refresh (upsert-without-clear left an old renamed team) + a **latent robustness gap** (a renamed/stale Yahoo team would leak). Not live-reachable now.
- **L-4 · me/team record 0-0-0 for a bare/unreconciled team_name.** `team_service._rank_and_record` (`:369`,`:388`) exact-`==`-matches `league_records`/standings with no emoji/whitespace reconciliation; bare `"Team Hickey"` → `0-0-0` while `"🏆 Team Hickey"` → `5-7-1`. **Local/dormant-only** (live assigned users are correct because the admin assignment reconciles to the canonical name); but brittle on a cold-start assignment, and inconsistent with `playoff_service` which DOES normalize.
- **L-5 · 189 sub-12px text nodes on `/standings`** (Codex C-UI-06) — readability/a11y. Category table also uses horizontal-scroll-in-wrapper on mobile (920px in a 334px wrapper — design choice).
- **L-6 · playoff-odds double-fetched** on the Team page (duplicate request).
- **L-7 · `team-logos/0.svg` → 404** (a player with `team_id:0`; gracefully handled by the avatar fallback).
- **L-8 · Trades page header says "WEEK 13"** while every other surface says Week 14 — cross-page inconsistency.
- **L-9 · Local freshness chip "Data is stale — updated 15d ago"** — stalest-source signal (local-only artifact; non-Yahoo bootstrap data is old; live is fresh).
- **L-10 · "UPGRADE" badge on live vs "PRO" on local** — billing-state display difference.

### Pre-public (NOT beta blockers)
- **Clerk DEV keys in production** — the live site shows a "Development mode" badge + loads `meet-firefly-38.clerk.accounts.dev`. Auth enforcement is sound; must move to a Clerk **production** instance before a public/monetized launch.
- **Stripe webhook lacks event-id dedup/ordering** (dormant on the beta; add before billing goes live).

---

## Verified CORRECT (positives, evidence-backed)

- **Security: no VULN (agent B3).** AI `run_read_only_sql` defended against stacked/CTE/UNION/PRAGMA/ATTACH + a read-only driver backstop; Clerk verifier RS256/iss/aud fail-closed; `EnvTokenVerifier` deny-by-default `hmac.compare_digest`; CORS explicit allowlist (live rejects a forged Origin with no ACAO echo); **the documented unauth public-read leak is CLOSED on live** (all 7 private read routers 401 unauthenticated — `require_login` active); tenancy `?team_name=` spoof closed; write-auth is authorization (cross-team guard), not just authentication; no secret literals; `.env` untracked.
- **Tenancy / 4-state machine works on live:** an authenticated-**unassigned** user gets the proper **"Your team isn't linked yet"** state (HIGH-1 fix) — never mock, never another team's data. A failed fetch renders **"We couldn't load this / Retry"** (HIGH-3) — never mock-as-real.
- **Records correct for assigned users:** live Team hero **"5-7-1 · 8th of 12 · 6 GB from 1st"**; `/api/standings` returns 12 teams with real W-L-T matching Yahoo.
- **Matchup page is excellent on live:** hero **"Team Hickey 9 VS 1 Jonny Jockstrap,"** full Category Battle tug-of-war (R 18-16, K 23-44, ERA 1.98-4.41…), and the per-category winner uses a **non-color ▲ glyph** (a11y fix confirmed) + hitter/pitcher roster tables.
- **PlayerDialog shows REAL data** (Trevor Rogers / SP,P · BAL · #778 rank · 73 IP · Free Agent · 54% rostered) — the "100% mock" Bugs HIGH is **fixed on live** (game-log windows are the deferred Slice 2).
- **Real data renders on live:** Leaders (25), Closers (24), Streaming (5), Probables (16), Databank (search), Draft (simulator setup), Players (100 FAs).
- **NaN guards confirmed fixed** (B1 venv repros): `stream_analyzer._clamp(nan)=0.0`, `draft_state._num(nan)=0`, closers `_f`, leaders `is_hitter`. Win-probability gauge "flicker" (81→83%) is just the count-up animation, not non-determinism.

---

## Coverage matrix

Legend: ✅ pass · 🔴 fail/bug · 🟠 partial-issue · ⚪ not covered this session.

| Surface | Render | Data correct | Interactions | States | A11y | Perf/console | Notes |
|---|---|---|---|---|---|---|---|
| Team `/` | ✅ | 🔴 lever; ✅ record | ✅ | ✅ (unlinked/error/loading) | 🟠 | 🟠 17-59s me/team | H-2, L-1, L-2, M-7 |
| Optimizer `/optimizer` | ✅ fixed | ✅ fixed | ⚪ | ✅ | ⚪ | 🟠 3-6s | **H-1 FIXED** (b452155) |
| Matchup `/matchup` | ✅ | ✅ | ✅ date tabs | ✅ | ✅ ▲ cues | ✅ | strong |
| Standings `/standings` | ✅ | ✅ live (🟠 local ghost) | ✅ | ✅ | 🟡 tiny text | ✅ | L-3, L-5 |
| Trades — Analyzer/Build | 🟠 | ⚪ (A3 pending) | ⚪ | ✅ | ⚪ | ⚪ | M-8 |
| Trades — Finder | 🟠 empty | ⚪ | ⚪ | ✅ | ⚪ | ✅ | M-8 |
| Trades — Compare | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | tab; A3 pending |
| Players/FA `/players` | ✅ | 🔴 value=0 (live) | 🟠 | ✅ | ⚪ | 🟠 5s | **M-1** |
| Streaming `/streaming` | ✅ (n=5) | ✅ | ⚪ | ✅ | ⚪ | ✅ 4.9s | A4 pending depth |
| Probables `/probables` | ✅ (n=16) | ✅ | ⚪ | ✅ | ⚪ | 🟠 6s | |
| Hitter-Matchups | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | ⚪ | not probed |
| Closers `/closers` | ✅ (n=24) | ✅ | ⚪ | ✅ | ⚪ | ✅ | |
| Research/Leaders `/research` | ✅ (n=25) | ✅ | ⚪ tabs | ✅ | ⚪ | ✅ | saturation fixed (B1) |
| Databank `/databank` | ✅ | ⚪ | ✅ search | ✅ | ⚪ | ✅ | works unlinked |
| Draft `/draft` | ✅ | ⚪ | ⚪ run-sim | ✅ | ⚪ | ✅ | works unlinked; A4 pending |
| Bubba | ✅ UI | 🔴 no model (live) | 🟠 | ✅ | ⚪ | ✅ | **M-5** |
| PlayerDialog | ✅ | ✅ (mock fixed) | 🟠 | ✅ | ⚪ | ✅ | game-log deferred |
| Command palette | 🔴 (unlinked) | 🔴 | 🟠 | 🔴 | ⚪ | ✅ | M-3; assigned retest inconclusive |
| Nav / TopBar | ✅ | — | ✅ | ✅ | 🟠 Home=Team dup | ✅ | C-UI-09 |
| Auth (login/unlinked/assign) | ✅ | ✅ | ✅ | ✅ | ⚪ | ✅ | gate + assignment verified |
| Paywall 402 | ⚪ | — | ⚪ | ⚪ | — | — | Stripe off (free beta) |
| Responsive (mobile) | 🟠 | — | — | — | 🟡 tiny text | — | standings only; M-2 blocks others |
| Dark mode | ⚪ | — | — | — | ⚪ | — | not tested |

**Honest coverage gaps:** Hitter-Matchups data, Trades Compare/Analyzer interactions (A3 still running), full a11y audit, dark mode, mobile overflow on `/optimizer` & `/players` (those bounce on hard-load + were degraded locally), and the assigned-state command palette.

---

## Data-correctness section (does it show the RIGHT numbers vs Yahoo?)

| Check | Source of truth | App shows | Verdict |
|---|---|---|---|
| Team Hickey record | Yahoo `league_records` 5-7-1, rank 8 | live "5-7-1 · 8th of 12" | ✅ |
| Standings ranks/records | Over the Rembow 11-2 r1 … Cyrus 3-8 r12 | live 12 teams, exact | ✅ |
| Per-category ranks | computed per-category (HR leader = HR rank 1) | ✅ Team Hickey HR-rank 1 despite overall 8 | ✅ (old clone-overall bug fixed) |
| Matchup category battle | live category totals | R 18-16, K 23-44, ERA 1.98-4.41, 9-1 score | ✅ |
| FA stats (YTD) | Yahoo/StatsAPI | Trevor Rogers K:54 ERA 5.30 | ✅ |
| **FA "value"** | marginal SGP | **live 0 (wrong), local 71-100 (right)** | 🔴 **live gap (M-1)** |
| **Lever weakest category** | API `category_key:"K"` | **UI "Stolen bases" (wrong)** | 🔴 **H-2** |
| **Optimizer lineup** | LP over the roster | 18 standard / 14 daily (fixed) | ✅ **H-1 fixed** (b452155) |
| playoff projected record | sim | string ≠ structured by 1 | 🟠 M-6 |
| Team count in subline | 12 | live "of 12" ✅ / local "of 13" (ghost) | ✅ live / 🟠 local L-3 |
| News chip | recent news | 930/1490 (all rows) | 🟡 L-1 |

---

## Reconciliation vs the three backlogs

**`Bugs_June26.md` (deep-debug, HEAD 7ec4666):** all 4 HIGH **FIXED** and verified — standings/my-team records (live shows real W-L; behavioral test, not just code), streaming NaN→max (`_clamp(nan)=0`), draft NaN (`_num`), player-card mock (live PlayerDialog real). MEDs fixed except the documented closer-adapter + Clerk-dev-keys + (now resolved) viewer-team. **Note:** B1 (code-read) marked records "FIXED" but my *behavioral* test found `/api/me/team` still returns 0-0-0 for a bare name — that's the L-4 reconciliation gap (fixed for the live assigned path, brittle for bare/cold-start).

**`Outstanding_June26.md` (pre-launch audit):** the 3 HIGH (tenancy fallback, matchup pitcher misclass, frontend 401/5xx swallow) + the round-2 MED/LOW are **genuinely shipped** (B2 + my live verification of the unlinked/error states). Only LOW-4 (watchlist write-bool) + 2 stale doc-comments remain; the rest is owner-gated M4/M5.

**Codex critiques (public-SaaS / UI / backend, plan-only):** mostly aspirational public-launch maturity (out of scope for a 12-friend beta). Concrete items **re-verified now:** C-UI-01 route-title **CONFIRMED** (M-4); C-UI-06 tiny-text **CONFIRMED** (189 on standings, L-5); C-UI-09 Home=Team nav duplicate **CONFIRMED**; C-UI-02/03 mobile page-overflow **NOT reproduced on /standings** (contained; unverified on the bouncing /optimizer & /players); C-BE-02/03 missing mlb_id/team is a known local-DB shape note. **What they missed (new this session):** H-1 optimizer, H-2 lever, M-1 live value=0, M-2 refresh-bounce, M-3 unlinked onboarding.

---

## Fixes applied this session (both merged to master)
- **H-2 (LeverCard)** — **FIXED + deployed + live-verified** (`9b55124`): renders the real weakest category + pickup count from the API; the live lever now reads "Strikeouts", not "Stolen bases".
- **H-1 (Optimizer)** — **FIXED + merged** (`b452155`): `lineup_service` now uses the enriched `get_team_roster` (both modes) + a placeholder-name fix; verified 18 standard / 14 daily real-named starters end-to-end; 3 new tests; full api suite 590 green. (Deploys to Railway App B from master.)
- Still open (flagged for follow-up): live **M-1** (App B projection gap → FA value=0), **M-3** (unlinked onboarding), **M-2** (refresh/deeplink bounce), **M-4** (route title on hard-load), **M-5** (Bubba no managed key), plus the pre-public Clerk-prod-keys + Stripe-dedup items.
