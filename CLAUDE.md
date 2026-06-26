# HEATER — Fantasy Baseball In-Season Manager

## Connor's Preferences

**Response style — ALWAYS:** Shortest, least verbose, simplest response possible. Every reply (instructions, answers, status updates, code explanations, everything) must be as brief as possible without dropping important information. One action per step. Plain language. No filler.

## Overview

A fantasy-baseball draft assistant + in-season manager for a 12-team Yahoo Sports H2H Categories snake-draft league. Built with Streamlit, powered by Monte Carlo simulation, Bayesian projection updating, and LP-constrained lineup optimization. **2026 MLB season is active; the app is in in-season mode** — draft completed, all components running against live league + Yahoo data.

> ## ★ NORTH STAR (set 2026-06-16) — the ultimate goal
> **HEATER is being prepared to become a real, monetized, general-public consumer product.** Every future session should treat this as the destination behind feature and architecture decisions. Today HEATER is a private single-league (FourzynBurn) Streamlit app on Railway for 12 users; "public + monetized" is where we are going. "Prepare the app as it stands" = keep the working app running while we build toward the product.
>
> - **Maintenance model:** from launch day 1 it is just the owner (Connor) + Claude, until HEATER earns enough revenue to fund hires. Bias every technical decision toward solo-operator-plus-AI maintainability and low operational overhead until then.
> - **This REVERSES the old "intentionally OUT" calls** (see the _Roadmap status_ line in "Known Design Choices" below): the **FastAPI/Next.js (or Reflex) re-platform** and the **multi-league / multi-tenant data model** are now ON the roadmap as the path to public scale — not off-limits. They were "OUT" only for the private-league context.
> - **Why going public is a re-platform, not a UI tweak:** three hard ceilings — single Railway replica + SQLite sole-writer (→ **Postgres** + external workers), one hardcoded league / Yahoo `game_key 469` (→ **multi-tenant** data model + per-user OAuth), and heavy per-session Streamlit cost. Leading architecture: **keep the Python engine** (`src/engine`, `src/optimizer`, `src/valuation`) behind a **FastAPI** service + **Postgres** + background workers, with a real frontend (**Next.js/React** preferred for a consumer product; **Reflex** the stay-in-Python alternative). The Combustion Index *design language* transfers; the engine is preserved.
> - **Status:** goal recorded 2026-06-16; the phased migration plan is **not yet written** — that is the next strategic step (decouple engine → API + Postgres first, rebuild the frontend page-by-page to parity, then layer multi-tenancy / auth / billing).

### Operating model & roles (set 2026-06-16)

Toward the North Star, HEATER is being driven by **two parallel Claude tracks that must converge**:

- **This track — CEO / Product Officer:** product strategy, system architecture, the migration / re-platform, monetization & business model, and engineering execution. Owns *how HEATER is built, scaled, monetized, and shipped*.
- **Parallel track — CMO:** reimagines the entire look & feel — design, frontend aesthetics, UI/UX, logo, colors, branding — with full creative latitude ("nothing is off limits"). Owns *how HEATER looks and feels*.
- **Overlap (the frontend) is shared:** the CEO track owns the stack, build, and scale; the CMO track owns the visual / brand / UX vision. The two plans MUST reconcile — the chosen frontend stack has to host the CMO's design language, and the migration plan must leave headroom for a full rebrand (so a CMO redesign is a re-skin, not a re-architecture).
- **Planning method:** spec with `brainstorming`, document with `writing-plans`, execute with `dispatching-parallel-agents`. The work is decomposed into **four sub-projects A/B/C/D** (see _Program structure_ next), each its own spec → plan(s) → build cycle.

### Program structure & file map (set 2026-06-17 — read this before planning re-platform work)

The whole re-platform = **4 sub-projects**, split across the two tracks:

- **A — Frontend foundation** (CMO) — design system + Next.js app shell + 3 foundation pages. Specs: `docs/superpowers/specs/2026-06-16-heater-frontend-foundation-design.md` + brand `...-heater-brand-design-vision.md`.
- **B — Backend API + data** (CEO) — FastAPI over the **unchanged** Python engines + Postgres + workers + auth + multi-tenancy. Spec: `docs/superpowers/specs/2026-06-16-heater-backend-api-foundation-design.md`.
- **C — Page migration** (joint) — rebuild the 14 pages in React against B's API.
- **D — Cutover & launch** (CEO) — run both apps in parallel, Stripe billing-live, migrate users, retire Streamlit.

**Hierarchy — do NOT conflate these levels:**
- Program → 4 sub-projects (A/B/C/D)
- each sub-project → **one spec doc** + several **phases**
- each phase → **its own plan file** (in `docs/superpowers/plans/`) → small **tasks** (TDD steps)
- a spec's **"sections"** are just document headings — NOT phases, NOT tasks. _(Example: B's spec has 9 sections; B's build has 6 phases B0–B5.)_

**★ Sub-project B status (updated 2026-06-19) — backend API surface BUILT + on `master`, dormant:** **23 `/api/*` endpoints** (the M0 baseline; the mounted surface has since grown past 26 — `api/openapi.json` is the live source of truth) (+ `/healthz`) over the **UNCHANGED** Python engines, shipped via 9 TDD slices (plans `docs/superpowers/plans/2026-06-17-heater-backend-api-slice{1..8}.md` + `2026-06-19-heater-backend-api-slice9-draft-simulate.md`). The live Streamlit app is untouched — the `api/` code is inert until cutover (Sub-project D). Endpoints: **18 read** (`me/team`, `free-agents`, `free-agents/pool`, `lineup/optimize`, `standings`, `closers`, `leaders`, `leaders/overall`, `matchup`, `streaming`, `punt`, `trade/evaluate`, `trade-finder`, `compare`, `databank`, `playoff-odds`, `players/search`, `league/rosters`) + **2 write** (`POST /api/lineup/set`, `POST /api/transactions/add-drop` — slice 7, over the already-graceful Yahoo write methods; all outcomes HTTP 200 with status in `MutationResult.ok`; **gated by `require_principal`** as of the 2026-06-19 write-auth slice — see below) + **2 draft** (`POST /api/draft/recommend` — slice 8; `POST /api/draft/simulate-picks` — slice 9, the AI-opponent advance-to-user-turn sim; **both stateless**: client holds `config`+`pick_log`, server replays to rebuild `DraftState`) + **1 analyze** (`POST /api/streaming/analyze` — read-like; scores ANY probable pitcher + a `factors[]` breakdown; the "Analyze Any Starter" feature). Slice 9 extracted `auto_pick_opponents` from `pages/20_Draft_Simulator.py` into `src/simulation.py` — the page is now a behavior-identical thin delegate (guarded by a frozen-reference equivalence test). **Write-auth seam (2026-06-19, plan `2026-06-19-heater-backend-api-slice2-write-auth.md`):** `api/auth.py` holds `require_principal` + an injectable `AuthVerifier`; the interim `EnvTokenVerifier` is **deny-by-default** (bearer token vs `HEATER_API_WRITE_TOKEN`, byte-compared so a non-ASCII header can't crash to 500; unset secret → 401), attached to BOTH write routes via `dependencies=[...]` (reads stay open). **Clerk is the B4 drop-in** — swap `get_auth_verifier`'s return; `require_principal` + routers never change. **Proven per-slice pattern (reuse for any new endpoint):** contract (`api/contracts/`) → service seam (`api/services/*_service.py`, the ONE place importing `src/`) → thin logic-free router (`api/routers/`, AST-guarded by `tests/api/test_no_logic_in_routers.py`) → DI provider in `api/deps.py` → fake-service dependency-override test (basename MUST be `tests/api/test_api_*.py`) → mount in `api/main.py` → regen `api/openapi.json` (snapshot-guarded by `test_openapi_contract.py`; **`fastapi==0.137.1` + `httpx==0.28.1` are PINNED** because FastAPI auto-generates the OpenAPI error schemas, so an unpinned drift fails the snapshot test with no code change). **STILL TO DO (each its own slice/decision):** real **Clerk** auth on the write gate (the interim env-token gate is shipped — swap the verifier) + the Yahoo `fspt-w` write scope (app is registered read-only, so live writes 401 until re-authorized); and the **infra migration B2 Postgres → B3 Redis/Arq workers → B4 multi-tenancy → B5 strangler-fig cutover** (each phase OWNER-GATED; B2.4/B2.5 + B3.4 touch the LIVE data path = hard gates). **STATUS (2026-06-19): owner RATIFIED the B2 plan; Phases B2.0 (engine seam) + B2.1 (Alembic baseline) are SHIPPED** (master, full suite 6434/0). **B2.0** = new `src/db/engine.py` SQLAlchemy engine seam (`engine_url()`/`get_engine()`/`reset_engine_cache()`/`is_sqlite_backend()`; backend chosen by `DATABASE_URL`, default = the existing SQLite file via the shared `DB_PATH` constant) + `SQLAlchemy>=2.0`; **`get_connection()` now raises `NotImplementedError` for any non-SQLite `DATABASE_URL`** (Postgres conn wiring = B2.2), SQLite path BYTE-IDENTICAL. **B2.1** = new `src/db/schema.py` (dialect-aware SQLAlchemy MetaData for all **53 tables** + the 62 `_safe_add_column` cols: autoincrement PK→SERIAL, `users.username`→CITEXT via `.with_variant`, 4 partial unique indexes, CHECKs; **FKs intentionally omitted** — live app runs `PRAGMA foreign_keys=OFF`, so enforce later via `op.create_foreign_key`, not in the baseline) + Alembic (`alembic/`, `alembic.ini`, `env.py` wired to `engine_url()`, baseline `0001_baseline` = `CREATE EXTENSION citext` on PG then `metadata.create_all`) + `alembic>=1.13`; **`init_db()` UNCHANGED** (a parity test enforces `schema.py` ≡ `init_db`). Verified server-free: SQLite `alembic upgrade head` + PG-dialect DDL compilation + schema parity. Plans: B2 `docs/superpowers/plans/2026-06-19-heater-backend-b2-postgres-migration-plan.md` (+ phase plan `...-b2-slice0-engine-seam.md`), B3 `...-b3-workers-plan.md`, Clerk-auth `...-clerk-auth-wiring-plan.md`; **B4-multitenancy + B5 NOT yet planned** (need owner product input — how leagues connect: Sleeper/Yahoo/ESPN/CBS + manual import). **⚠️ B2 EXECUTION PAUSED (2026-06-19) — Postgres is now SEQUENCED INTO M4, AFTER the single-league paid beta**, per the CMO M0–M5 product roadmap (`docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md`; owner decision: revenue-first / lean-infra — ship a sellable product on CURRENT SQLite infra BEFORE Postgres/workers/multi-tenancy). B2.0/B2.1 already shipped are additive + dormant (SQLite byte-identical), so they just sit until M4. **Do NOT resume B2.2 without an explicit owner go-ahead.** _(When B2/M4 resumes: B2.2 = dialect port + `read_sql`→engine — touches LIVE read paths AND needs a REAL Postgres to verify dialect ports, so it's where a PG is first required: owner starts Docker → throwaway PG container, or a Neon/Railway dev `DATABASE_URL`. Then B2.3 keystone `_load_player_pool_impl` → B2.4 writers/sole-writer relax → B2.5 data migration + cutover.)_ **ACTIVE backend line = M0 (contract lock). M0 slices 1+2 SHIPPED 2026-06-19** (plans `docs/superpowers/plans/2026-06-19-heater-m0-slice1-playerref-contract.md` + `...-m0-slice2-playerref-enrichment-remaining.md`, on `master`): **slice 1** = `api/contracts/common.py` `PlayerRef` extended with optional `mlb_id`/`team_abbr`/`team_id` (backward-compatible; openapi regenerated) + a shared NaN-safe `api/services/player_ref.py` builder (`make_player_ref`/`team_id_for`, reusing `src.depth_charts._STATSAPI_TEAM_IDS` — NO duplicate map). **slice 2** = a second helper `player_ref_from_pool(player_id, pool, …)` (lookup by id; PREFERS the engine's name/positions, pulls mlb_id+team from the pool) → enrichment now wired into **ALL 10 player-bearing sites** (slice 1: trade, trade_finder, databank, closers, streaming; slice 2: compare, draft, leaders [+`p.mlb_id,p.team` in its SELECT + a new `_to_leader_row` helper], lineup [loads the pool in `optimize()`], fa). **Slice 2 also FIXED a latent bug:** `fa_service` read the wrong move-dict id keys (`add_player_id`/`drop_player_id`, which the engine never emits) so every FA rec's `PlayerRef.id` was always 0 — now reads the engine's real `add_id`/`drop_id`. 109 api tests green; `src/` + `api/routers/` UNTOUCHED; **`api/openapi.json` UNCHANGED by slice 2 (no contract change — slice 1 already added the fields).** **M0 per-page contract fixes** (gap spec `docs/superpowers/specs/2026-06-19-frontend-api-contract-gaps.md`): ✅ **DONE — `GET /api/free-agents/pool`** (ranked FA pool for the Players page, plan `docs/superpowers/plans/2026-06-19-heater-m0-free-agent-pool-endpoint.md`; NaN-safe, composes `src.in_season.rank_free_agents`, contracts `FreeAgentPoolResponse`/`FreeAgentPoolItem`/`StatItem`, `top_need` from `ctx.category_gaps`). ✅ **DONE — `/api/streaming` WIDENED** for the CMO's Pitcher Streaming page (plan `docs/superpowers/plans/2026-06-19-heater-m0-streaming-widen.md`): `StreamCandidate` now carries the full board row (`is_home`/`confidence`/`num_starts`/`net_sgp`/`opp_wrc_plus`/`opp_k_pct`/`park`/`expected_ip·k·er`/`win_pct`/`own_pct`/`risk_flags`/`components`{6 factors}/`expected_line`/`rank`) + new `StreamComponents`/`BudgetStrip` models + `StreamingResponse.top_pick`/`budget`; all NaN-safe via `_f`; mapped from `build_stream_board` (no engine change); `status`/`confidence` emit raw engine values (frontend adapter lowercases). **★ COORDINATION (live, 2026-06-19): the CMO is building the Pitcher Streaming React page NOW (branch `feat/web-pitcher-streaming`) and their spec (`docs/superpowers/specs/2026-06-19-heater-pitcher-streaming-page-design.md`, §"API contract handoff for the CEO track") ASSIGNS the CEO track this backend — so the streaming work directly unblocks their active page (zero overlap).** ✅ **DONE — streaming slice 2** (`POST /api/streaming/analyze` + `probables[]` on `StreamingResponse`, plan `docs/superpowers/plans/2026-06-19-heater-m0-streaming-analyze.md`): the "Analyze Any Starter" feature scores ANY probable (reuses `build_stream_board(include_rostered=True)`, finds the `pitcher_id` row, maps to `PitcherScorecard` = the widened candidate + `factors[]` with `CONSTANTS_REGISTRY["stream_score_w_*"]` weights + composed detail strings; graceful `found=False`; `_f` now also guards `inf`); `probables[]` = the full picker list (a 2nd `include_rostered=True` board build; the existing `candidates` build is unchanged). **★ THE STREAMING BACKEND IS COMPLETE — the CMO's Pitcher Streaming page is fully unblocked.** ✅ **DONE — `GET /api/leaders/overall?lens={overall|hot|cold|breakout|sell}`** for the Research page (plan `docs/superpowers/plans/2026-06-19-heater-m0-leaders-overall.md`): a cross-category 5-lens leaderboard (data-driven `_LENS_META` + one mapper) composing 4 existing engines (`compute_category_value_leaders`/`compute_breakout_scores_batch`/`compute_player_trends`/`detect_sell_high_candidates`); per-row `value` 0-100 (overall normalized over the sum-of-z range [-10,10] — code-quality caught + fixed a saturation bug where the single-z window washed the top of the board to 100), `stats[3]`/`trend`/`tag`/`note`/`hitter`; NaN/inf-safe; new contracts `OverallLeaderRow`/`LeadersOverallResponse`; existing per-category `/api/leaders` untouched. ✅ **2 bugfixes (2026-06-20, plan `docs/superpowers/plans/2026-06-20-heater-m0-leaders-overall-bugfixes.md`):** **(1) breakout hung >12s** — `compute_breakout_scores_batch` is O(n²) (each player percentile-ranked vs the whole input); now bounded by `_filter_breakout_candidates`→top-500 most-relevant (consensus_rank/ownership), live 2.35s, engine untouched; guard test is count-based + DB-free (a wall-clock test calls the real service → needs the 26MB live DB → 0 rows + flaky in worktree/CI). **(2) sell ALWAYS returned 0 rows = a real WIRING bug, NOT data-dependent** — the service's `_load_season_stats` query omitted `ab`/`xfip`/`stuff_plus`, so `compute_sustainability_score` collapsed every score to the neutral 0.5 → never below the <0.45 sell cap; fixed by adding `ab,h,sf,fip,xfip,siera,stuff_plus` (matches the Streamlit Leaders page's `load_season_stats()` columns). Lens is THIN (≈1 today) because `xwoba_delta` (primary hitter signal) lives only in the pool, not season_stats — a pre-existing limitation shared with Streamlit; merging it → ~129 but diverges from the proven path + surfaces small-sample noise (documented follow-up, owner's call). ✅ **DONE — `/api/matchup` roster-comparison tables** (the Matchup page centerpiece; plan `docs/superpowers/plans/2026-06-19-heater-m0-matchup-rosters.md`): extended `MatchupResponse` with `hitters`/`pitchers` (`RosterRow{slot, you, opp: MatchPlayer|None}`), `hitter_columns`/`pitcher_columns`, `date_tabs`, + per-category `win`; `MatchPlayer` = PlayerRef + assigned slot (`selected_position`) + **projected** stat line + game-state (sched/live/final from `statsapi.schedule`) + `badge` (IL/DTD from `league_rosters.status`); classified by pool `is_hitter`; BN+IL rows included; NaN-safe; never-raise; per-category `win` is inverse-aware. **Code-quality caught a CRITICAL bug:** the build first read `roster_slot` (= eligible positions, e.g. "SP,RP"/"2B,3B,SS"), NOT the assigned slot — silently mis-labeling slots + routing `SP,RP` swingmen into the HITTER table; fixed to `selected_position` for the slot + `is_hitter` (pool) for classification. **LIVE in-game stats/status DEFERRED** (no box-score feed in the backend) → follow-up plan `docs/superpowers/plans/2026-06-19-heater-matchup-live-stats-followup.md` (a future MLB-StatsAPI live-boxscore slice). ✅ **DONE — Matchup-B: you/opp `TeamSide` + totals** (plan `docs/superpowers/plans/2026-06-20-heater-m0-matchup-teamside-totals.md`): `TeamSide{name,manager,record,score}` for you+opp (manager from `league_teams`, record from `load_league_records`, score = per-cat win count) + `hitter_totals`/`pitcher_totals` (`SideTotals{you,opp}`) with **WEIGHTED** rate aggregation (AVG=Σh/Σab, OBP=Σ(h+bb+hbp)/Σ(ab+bb+hbp+sf), ERA=Σer·9/Σip, WHIP=Σ(bb_allowed+h_allowed)/Σip — NaN/zero-safe); reuses `_build_roster_tables`'s side-slicing + the pool; never-raise. ✅ **DONE — Matchup-C: league scoreboard** (`league[]` on `/api/matchup`, plan `docs/superpowers/plans/2026-06-20-heater-m0-matchup-c-league-scoreboard.md`): new `LeagueMatchup{a,b}` reusing `TeamSide` (the frontend's `LeagueTeam` shape is identical) + `MatchupResponse.league: list[LeagueMatchup]`; `_league` builds ALL of the week's pairings from `load_league_schedule_full`, name/manager/record via the existing `_team_side`; per-team weekly **scores best-effort** from each pairing's cached matchup (`_score_from_cache` reuses `_build_categories`/`_cat_win`; the user's own pairing reuses the header you/opp scores), `(0,0)` when uncached; never-raise → `[]`. Live-verified: schedule(24wk)+records(12 teams) populate locally → 6 real pairings with real records; per-team scores 0 until `sync_all_team_matchups` caches them on Railway (documented env-dependent degradation). DB-free tests (incl. the b-cache swap path, flagged by review); 184 api tests; openapi regenerated. ✅ **DONE — Team dashboard slice 1: movers + scalars + status chips** (`GET /api/me/team` extension, plan `docs/superpowers/plans/2026-06-20-heater-m0-team-dashboard.md`): `MyTeamResponse` extended (additive/backward-compatible) with `movers` (hot/cold roster players via `trend_tracker.compute_player_trends`, pool-filtered to the roster FIRST since the delta is per-player not pool-relative, enriched mlb_id+team, top 4 by |trend_delta|) + `movers_scope`, `status_chips` (IL + news counts; IL match handles Yahoo free-form "IL10 - 3 days"), `eyebrow`, `subline` (ordinal + games-back), `freshness_minutes` (**stalest** core refresh_log source, clamped ≥0), `playoff_cut_rank` (=`_PLAYOFF_SPOTS` 4); new `Mover`/`StatusChip` contracts. All helpers never-raise (missing live data → empty/None). DB-free tests (engines monkeypatched at source — worktree/CI DB is EMPTY, see new memory `reference_worktree_empty_db`) + live-smoke verified the movers chain on real data; 175 api tests green; openapi regenerated. Code-review (with full DB context) caught 2 real issues pre-merge: freshness was reporting the freshest not stalest source, and IL count missed Yahoo free-form statuses. ✅ **DONE — Team dashboard slice 2: `lever` + structured stats** (`GET /api/me/team`, plan `docs/superpowers/plans/2026-06-20-heater-m0-team-lever-slice2.md`): `Lever{category_key, headline, behind_by, pickups: LeverPickup[]}` — weakest = most-negative `ctx.category_gaps`; pickups = `rank_free_agents` filtered to `best_category` (mirrors `fa_pool_service`). Also moved `StatItem` to `api/contracts/common.py` (shared; re-exported from free_agents, schema name unchanged) and converted `movers[].stats` + lever stats to structured `StatItem{label,value}` per the CMO's contract ask. Fixes: live-smoke caught category_gaps keys lowercase vs `best_category` UPPERCASE → `.upper()`-normalized (else 0 pickups on Railway); code-review caught lever should return None when at-or-ahead in EVERY category (no misleading card) + `_lever_pickups` now logs on failure. Perf note: `_lever` adds one `build_optimizer_context` (~2-4s, same as Players page) — documented for a future lazy/cached split. 188 api tests; openapi regenerated. ✅ **DONE — Team dashboard slice 3: `ops` cards (COMPLETES the Team contract)** (`GET /api/me/team`, plan `docs/superpowers/plans/2026-06-20-heater-m0-team-ops-slice3.md`): `OpsCard{key,label,value,total,verdict,status}` ×3 — `ip_pace` (weekly IP vs target via `ip_tracker.compute_weekly_ip_projection` over the roster's pitchers), `moves_left` (`ctx.adds_remaining_this_week`/10), `roster_health` (IL count via the slice-1 `_il_count`). **Refactor:** hoisted `build_optimizer_context` into ONE `_build_ctx` shared by lever + ops (was two ~2-4s calls → one); `_lever(ctx,cfg)` now takes the shared ctx. Each ops card independently try/except-wrapped (a failing card is skipped, never fatal); `[]` when no ctx. Live-verified on the real pool (ip_pace 18.2/53.8); locally only `moves_left` shows (ctx.roster empty → ip_pace/roster_health None), full set on Railway — documented Yahoo-dependent degradation. 194 api tests; openapi regenerated; code-review found no correctness bugs (applied 3 minor test hardenings). **★ THE TEAM DASHBOARD CONTRACT IS COMPLETE (slices 1+2+3) — the CMO can wire the full flagship Team page.** **DEFERRED (no backing data — need a per-week snapshot table + cron, a future B3-workers slice):** Team `trajectory` (rank history) + `win_prob_trend` + lineup-status chip. ✅ **DONE — `GET /api/playoff-odds` (the 21st endpoint; shared: Standings odds panel + Team)** (plan `docs/superpowers/plans/2026-06-20-heater-m0-playoff-odds-endpoint.md`): per-team **playoff odds** + projected standings from ONE `standings_engine.simulate_season_enhanced` run over per-team weekly totals (`standings_utils.get_all_team_totals`, counting/26 + rate passthrough; current_standings from `load_league_records`; schedule from `load_league_schedule_full`; current_week = max games-played +1). Contracts `PlayoffTeam`/`PlayoffOddsResponse`; rows sorted by odds → rank → in_cut(top-4); `is_user`/`you` via `_normalize_team_name`; never-raise → empty. `n_sims=2000` (fixed seed → deterministic; ~2.4s) tuned for read responsiveness. 201 api tests; openapi regenerated; code-review applied (construct-after-sort no mutation, per-row standings guard, dead-code removal). **Per-team CHAMP odds DEFERRED** (no engine computes them league-wide; only the Hickey-centric `simulate_playoff_outcomes` gives the user's champ%) + the API sim is projection-based go-forward (no Streamlit session → `get_all_team_totals` projection tier; still starts from real current_standings) — both documented follow-ups. ✅ **DONE — Optimizer enrich slice 1 + CRITICAL FIX** (`POST /api/lineup/optimize`, plan `docs/superpowers/plans/2026-06-20-heater-m0-optimizer-enrich-slice1.md`): **the endpoint was BROKEN** — `lineup_service` called `pipeline.run()` (no such method → `result=None` → empty even on Railway); fixed to `pipeline.optimize()` reading `result["lineup"]` = `{assignments,bench,projected_stats,status}` (the real `lineup_optimizer.optimize_lineup` shape). Added (additive): `LineupSlot.status` (start|bench), `CatImpact`, `LineupOptimizeResponse.bench` (roster−starters, id-based), `.optimal` (current Yahoo starters via `selected_position` == optimizer starters), `.impact` (projected category totals); `projected` now defaults 0.0. 206 api tests; code-review applied (lambda loop-var freeze, NaN/inf guard in `_impact`, discard pid 0). **DEFERRED → Optimizer slice 2 (daily mode):** per-slot `value` (0-100 DCV), `matchup` string, `proj` human line, `swaps`, `ip_pace`/`moves_left` — all need `mode="daily"` + `build_daily_dcv_table` (live Yahoo, unverifiable locally; same pattern as the matchup live-stats deferral). ✅ **DONE — player-id endpoints (the 22nd + 23rd; CMO formally flagged them for the Trades evaluator + compare-any-player)** (plan `docs/superpowers/plans/2026-06-20-heater-m0-player-id-endpoints.md`): **`GET /api/players/search?q=&limit=25`** → `PlayerSearchResponse{query, results: PlayerRef[]}` — pool-wide case-insensitive name search sorted by `consensus_rank` (most fantasy-relevant first), limit clamped 1..200; DB-backed → **REAL locally** (live-verified "trout" → Mike Trout id=36 mlb_id=545361 LAA); returns ANY player's id (rostered or FA) → unblocks compare-any-player + "add any player". **`GET /api/league/rosters`** → `LeagueRostersResponse{teams: [{team_name, manager, players: PlayerRef[]}]}` — all teams' rosters via `load_league_rosters` grouped + `player_ref_from_pool` enrichment (league_rosters is `SELECT *`, no name/positions → the pool carries them by player_id) + manager from `league_teams`; Yahoo-dependent (empty locally, real on Railway). New `RosterQueryService` (one src-importing seam, both methods) + thin router + DI + fake-service tests; never-raise → empty. 213 api tests; openapi regenerated; code-review applied (dropped dead `name=`/`positions=` kwargs, added the limit clamp). ✅ **DONE — Optimizer enrich slice 2: daily mode (the LAST M0 backend item)** (`POST /api/lineup/optimize`, plan `docs/superpowers/plans/2026-06-20-heater-m0-optimizer-daily-slice2.md`): request `mode="daily"` (default `"standard"` → byte-for-byte backward-compatible) runs the today's-DCV start/sit path. Additive: `LineupSlot` += `value` (0-100 heat = normalized DCV, best play=100), `matchup` ("vs BOS"/"@ NYY"), `current_slot` (frontend diffs swaps); new `DailyMeta{urgency, rate_modes (ERA/WHIP protect|compete|abandon), winning/losing/tied, ip_pace (IpPace), recommendations, swaps}` on `LineupOptimizeResponse.daily` + `mode` echo. **Robust join (key design point):** the engine's `daily_lineup["starters"]` has NO `player_id` (pipeline.py:883) and the engine is frozen, so PlayerRef comes from `daily_dcv` (has player_id) and the LP start/slot decision joins back by `(normalized_name, round(total_dcv,4))` — a per-player near-unique float → same-name players don't collide, player_id never from the name (Muncy-DNA-safe). **Code-review (full engine context) caught 2 real bugs the empty-DB env hid behind false-green synthetic tests — both fixed + re-verified against the real helpers:** (1) `_matchup_str` read `home_team`/`away_team` but `statsapi.schedule` emits `home_name`/`away_name` (full names) vs the abbr `dcv.team` → matchup was ALWAYS "" → now canonicalizes both via `team_name_to_abbr`+`canonicalize_team` (the engine's own convention); (2) `_daily_meta` read `result["urgency_weights"]["urgency"]` but the pipeline flattens `urgency_weights` to the `{cat:weight}` dict itself (pipeline.py:514) → urgency was ALWAYS `{}` → now iterates it directly. **DEFERRED (documented, graceful engine degradation — NOT bugs):** richer daily inputs (`confirmed_lineups`/`recent_form`/`team_strength`/`park_factors`) — v1 passes matchup + schedule_today (gives the SP probable-gate + locked games + opponent strings); `moves_left` lives on `/api/me/team`. 220 api tests (+7 daily, DB-free synthetic); openapi regenerated; live smoke = daily path runs end-to-end no-Yahoo → graceful empty (field-population verified on Railway when the CMO wires the page). **★ M0 BACKEND IS COMPLETE — every page contract is built (26 M0/M1 endpoints; the live surface is larger now — see `api/openapi.json`).** The only remaining backend lines are owner-gated infra: B4 Clerk verifier + Yahoo `fspt-w` write scope, and B2–B5 (paused → M4). **Pattern: read each CMO page-design spec's §"API contract handoff (for the CEO track)" — they define the exact contract to build to.** Frontend-lane OpenAPI→TS codegen is READY — the CMO already regenerated types off slice 1 + wired the Players page to `/api/free-agents/pool` (both CI-green on master). Also still (B4): swap in the real **Clerk** verifier + the Yahoo `fspt-w` write scope (app registered read-only → live writes 401). The API also now has **CORS** (`api/main.py`, env `HEATER_API_CORS_ORIGINS`, localhost default) + write endpoints **advertise their 401** in `openapi.json`. The CMO frontend (`web/`) can wire against the full contract now.

**★ M2 BACKEND COMPLETE (2026-06-20) — auth + billing + Pro gating, all ADDITIVE + DORMANT until env-configured (live Streamlit + `web/` byte-for-byte untouched).** Specs/plans `docs/superpowers/{specs,plans}/2026-06-20-heater-m2-*`; memory [[project_m2_auth_billing]]. **Slice 1 — Clerk auth:** `ClerkVerifier` (PyJWT-JWKS local RS256, fail-closed, JWKS-outage WARNING) dropped into the existing `api/auth.py` `AuthVerifier` seam; `get_auth_verifier()` returns it when `CLERK_ISSUER` set, else the interim `EnvTokenVerifier`. `Principal.clerk_user_id` added; local `AppUser` provisioning (`api/identity.py::require_app_user` + `api/stores/user_store.py`) in a SEPARATE `data/api_state.db` (env `HEATER_API_DB_PATH`) — NOT the live `draft_tool.db`. **Slice 2 — Stripe (owner-set: hard gate, $7.99/mo, 7-day trial):** `POST /api/billing/checkout-session`, `POST /api/billing/webhook` (signature-verified, fail-closed 400), `GET /api/billing/subscription`; `StripeGateway` Protocol (Live/Null/Fake) + `SubscriptionStore` (api_state.db, tier source of truth) + `BillingService`; `stripe==15.2.1` PINNED. `checkout.session.completed` is link-only (out-of-order can't re-activate a canceled user); unset-webhook-secret logs a WARNING not a silent drop. **Slice 3 — gating:** ONE reusable `require_pro` (`api/gating.py`) on the **7 compute-heavy endpoints** (lineup/optimize, trade/evaluate, playoff-odds, trade-finder, draft/recommend, draft/simulate-picks, draft/grade); structural-guarded (`tests/api/test_api_pro_gating.py` introspects the **router OBJECTS** — FastAPI 0.137 lazy `include_router` hides routes from `app.routes`). **DORMANT until billing is operable** via the shared `api/billing_config.py::billing_env_configured()` (`STRIPE_SECRET_KEY` **and** `STRIPE_PRO_PRICE_ID` — same predicate the billing service uses, so the gate can't paywall before checkout works); 402 = not Pro, 401 = unauth when billing live. 281 api tests; every slice code-reviewed (slices 1–2 also silent-failure-hunter), findings applied. **ACTIVATED 2026-06-22 (was owner-gated):** the Clerk app (`CLERK_ISSUER`/`CLERK_AUDIENCE`) + Stripe product (`STRIPE_SECRET_KEY`/`STRIPE_WEBHOOK_SECRET`/`STRIPE_PRO_PRICE_ID`) are configured + the **M2 frontend** (login/signup/pricing/account UI) is BUILT + deployed on Vercel (see the 2026-06-22 Roadmap-status note below). On the live beta API Stripe is intentionally LEFT UNSET = Pro gate dormant = free for friends. Yahoo `fspt-w` write scope still separate (live writes 401 until re-authorized). B2 Postgres / B3 workers / B4 multi-tenancy remain PAUSED (M4).

**★ POST-M0 FEATURES (2026-06-21) — additive `/api/*` + `web/` surface beyond M0's 23 endpoints (detail in the linked memories/specs).** **Probable Pitcher + Hitter Matchup 7-day grids** ([[project_probable_pitcher_grid]]): `GET /api/schedule/probables` (P1) + `GET /api/schedule/hitter-matchups` (P2) over a new `ScheduleService` (the one `src/`-importing seam; composes `build_stream_board ×7` + a league-tag layer); P2's `compute_hitter_matchup_score` (`src/optimizer/stream_analyzer.py`) = the calibrated INVERSE of the pitcher matchup — mirrors `score_stream_candidate`'s extraction so it exactly complements the Probable grid; frontend `/probables` + `/hitter-matchups`. `schedule_service.py` now has a module logger (was the only un-logged `api/services/` file). **Bubba AI assistant** ([[project_bubba_ai_assistant]]): B1 chat MVP SHIPPED (`/api/chat/*` over the UNCHANGED `src/ai/` + the every-page popup; identity offset `chat_user_id = 1_000_000_000 + AppUser.id`); **B2.1 streaming SHIPPED 2026-06-21** (spec `...-bubba-b2-1-streaming-design.md`, plan `docs/superpowers/plans/2026-06-21-heater-bubba-b2-1-streaming.md`): `src/ai/providers.py` refactored to a `_chat_events` generator with `chat()` now a thin DRAIN returning the byte-identical dict (frozen-reference equivalence test guards the LIVE Streamlit app); new `POST /api/chat/send-stream` (`StreamingResponse` SSE) over `ChatService.send_stream` (meters server-side BEFORE the terminal `done` frame so a tail-disconnect still bills; never 500s → `error` frame); a `thinking_params_for_model(model, effort)` per-provider reasoning dial (Anthropic `thinking` budget / OpenAI `reasoning_effort`; off/low + unsupported → no-op) threaded via `reasoning_effort` on `ChatSendRequest` + `/send` + `/send-stream`; frontend `bubba.sendStream` (fetch + ReadableStream SSE reader) + a streaming Bubba.tsx composer (live tokens, tool-status chips, effort dial, Web/Research toggles). **Critical review fix:** `litellm.stream_chunk_builder` returns `None` on an empty stream → guarded in `_chat_events` (graceful `done`, never `None.choices`) so it can't crash the live app's `chat()` hot path. Known follow-up: partial-answer tokens on a mid-stream exception are not metered (documented under-count). **B2.2 tag-page-context SHIPPED 2026-06-21** (spec `...-bubba-b2-2-select-to-tag-design.md`, plan `docs/superpowers/plans/2026-06-21-heater-bubba-b2-2-select-to-tag.md`): a composer **select-mode** lets the user attach page context to a question 3 ways — highlight text, click a player row (capture-phase `click` reads `data-bubba-tag` off the canonical `PlayerLink`, intercepting before `PlayerDialog` opens), or snapshot the page (`html-to-image`). Backend: `ChatSendRequest` += `attached_text` + `attachments: list[ChatAttachment]`; a shared `ChatService._build_user_content` wraps text exactly like the live Streamlit app (`[Context…][Question…]`) and builds an OpenAI multimodal parts list for images — returns the **bare message string** when no tag/image so today's behavior (and the B2.1 `chat()` equivalence) is byte-identical; **`src/ai/providers.py` UNTOUCHED** (litellm consumes list-content natively); both `/send` + `/send-stream` threaded. Frontend: select-mode toggle + page-edge glow + removable chips + `data-bubba-panel` snapshot-exclusion. Review fixes: snapshot-failure logging + an honest cross-origin fidelity caveat (`html-to-image` renders cross-origin headshots blank without throwing) + defensive `data-bubba-tag` parse. A vision-incapable model + image → graceful `error` (never 500). **B2.3 attachments SHIPPED 2026-06-21** (spec `...-bubba-b2-3-attachments-design.md`, plan `docs/superpowers/plans/2026-06-21-heater-bubba-b2-3-attachments.md`): **frontend-only** (the B2.2 `attached_text`/`attachments` backend is reused unchanged) — three new composer sources: **image files** (paperclip → file picker), **screen capture** (`getDisplayMedia` → canvas frame), **PDFs** (`pdfjs-dist` client-side text extraction → folded into `attached_text`). **Key call: documents go as EXTRACTED TEXT, not raw-PDF passthrough** → universal (works on EVERY model incl. DeepSeek), zero backend change; native vision-PDF deferred. New `web/src/components/bubba/attachments.ts` (`readImageFile`/`captureScreen`/`extractPdfText`, the last dynamic-imported so pdfjs stays out of the main bundle); B2.2 `shots` state renamed `images` (now `{…,source:"snapshot"|"screen"|"file"}`, so the cross-origin caveat keys on `source==="snapshot"` only) + a `docs` state; per-item attach failures logged + surfaced (8 MB cap, unreadable file, empty/scanned PDF, truncation shown on the chip, cancelled screen-share). `pdfjs-dist` 6.0.227; the `new URL(worker, import.meta.url)` pattern bundled cleanly in Next 16. Bubba is READ-ONLY on the league (recommends, never writes). **B2.4 message-queue + saved-prompts SHIPPED 2026-06-21 → Phase B2 COMPLETE** (spec `...-bubba-b2-4-queue-prompts-design.md`, plan `docs/superpowers/plans/2026-06-21-heater-bubba-b2-4-queue-prompts.md`): **(1) message queue** (frontend-only) — while Bubba streams, Enter/Send enqueues the next message; a deferred-`setState` auto-drain effect sends the next queued item when the turn finishes cleanly (pauses + shows a hint on an errored turn); queued items are text-only + edit/delete-able. **(2) saved prompts** — new `api/stores/prompt_store.py` (mirrors `user_store.py`: Protocol + InMemory fake + Sqlite over the SEPARATE `api_state.db`, `owner_id`-scoped, NEVER the live `draft_tool.db`) wrapped by `ChatService` (now takes an injected store; reads degrade `[]`, writes `(ok,msg)`) + 3 routes `GET/POST /api/chat/saved-prompts` + `DELETE /api/chat/saved-prompts/{id}` + a `PromptsMenu` (pick/save/delete). `src/` UNTOUCHED. Review fix: PromptsMenu now surfaces the `{ok,message}` the backend returns (a failed save no longer looks like success). **★ Phase B2 (B2.1 streaming + B2.2 tag-context + B2.3 attachments + B2.4 queue/prompts) is DONE.** **B3 monetization (LEAN cut) SHIPPED 2026-06-22** (spec `...-bubba-b3-tier-aware-ai-design.md`, plan `docs/superpowers/plans/2026-06-22-heater-bubba-b3-tier-aware-ai.md`): managed Bubba chat is now metered BY TIER (Free ≈$0.10/day taste, Pro ≈$2.00/day, own-key unlimited) + an over-cap **upgrade nudge** — **additive + DORMANT until billing is configured** (`billing_env_configured()` off → everyone keeps today's single admin cap, the live Streamlit app byte-for-byte unchanged). Mechanism: `src/ai/budget.is_over_cap` gained an optional `cap_usd` override (no-op for the live app); `api/services/ai_allowance.py` maps tier→daily cap; `api/gating.py::get_managed_ai_cap` (mirrors `require_pro`'s verifier+store pattern; lives in gating.py NOT deps.py to avoid the deps↔identity cycle) resolves the caller's cap (None while dormant → admin cap; never raises — store failure → None = bounded+logged over-grant); the chat router threads it via `Depends(get_managed_ai_cap)` → `ChatService.send/send_stream(cap_usd=…)` → `budget`. Over-cap streaming error carries `code:"over_cap"` → `Bubba.tsx` renders an `OverCapNudge` (Upgrade → `/pricing`, Add-your-own-key → key settings). Both reviewers clean. **Deferred until Pro sells: Plus/Max tiers, monthly free taste, admin-tunable per-tier caps. ★ The whole Bubba arc (B1 → B2.1-B2.4 → B3) is COMPLETE — only owner activation (Clerk + Stripe env) remains.**

**No single "master plan" file** — specs in `docs/superpowers/specs/`, one plan file per phase in `docs/superpowers/plans/`. This CLAUDE.md (★ NORTH STAR + Operating model) is the only top-level index.

**Frontend stack (confirmed):** **Next.js + React + TypeScript** frontend, **FastAPI** backend. **Streamlit stays LIVE through the entire migration (strangler-fig) and is retired only in Sub-project D — it is NOT the long-term frontend.** Monetization = **2-tier Free→Pro** (owner-confirmed 2026-06-17).

### ★ Web Frontend (`web/`) — Sub-project A, CMO track (status 2026-06-19)

The Next.js consumer frontend is **building in `web/`** beside the live Streamlit app (strangler-fig). Foundation + all 6 core pages + a "10/10" design/polish pass are **DONE and on `master`** (united with the backend API in merge `0c06cc1`, 2026-06-19; the `feat/frontend-replatform-foundation` branch is fully contained in master).

- **Stack:** Next.js 16 (App Router, RSC) + React 19 + TypeScript 5.9 + Tailwind CSS v4 (`@theme` tokens in `web/src/app/globals.css`) + framer-motion. Package manager **pnpm**. `web/AGENTS.md` warns this Next.js has breaking changes — read `node_modules/next/dist/docs/` before writing Next-specific code.
- **Run / verify** (from `web/`): `pnpm dev` (preview), `pnpm build` (production gate), `pnpm exec tsc --noEmit` (types), `pnpm run lint`. **`web/` has NO CI yet** — the green Python CI does not test it, so the local production build IS the gate. (Cosmetic build warning: a stray `~/pnpm-lock.yaml` confuses Next's workspace-root inference; harmless.)
- **6 pages** (`web/src/app/`): `/` (Team — showcase + in-season landing), `/optimizer`, `/matchup`, `/trades`, `/players`, `/research`. Route-level crossfade via `template.tsx`; per-page mount-stagger.
- **Design system — "Combustion" light:** navy `#0a1f3a` chrome, orange `#ff5c10` heat, Archivo (display; width-axis hero numerals via `HeroNum`) + Inter (body). `heatColor(pct)` ramp (cool blue → warm orange) drives every viz. Tokens/colors in `web/src/lib/tokens.ts`; motion primitives in `web/src/lib/motion.ts` (`EASE_SNAP`, `staggerContainer/Item`, `useCountUp`).
- **Signature data-viz** (hand-rolled SVG — NO recharts, by design; in `web/src/components/viz/`): `HeatGauge` (Team hero win%), `CategoryRadar` + `WinProbTrend` (Team), `CategoryBattle` (Matchup tug-of-war).
- **Canonical components:** `PlayerDialog`/`PlayerLink` (the ONE player card — every player name/row opens it; keyed by `mlbId` with a `fallback()` for unknowns), `Card` (machined edge + opt-in `interactive` lift), `EmptyState` (shared no-results/error), `HexMesh`, `PlayerAvatar`. Data tables: clickable player cells + row-hover everywhere; mobile hamburger nav in `TopBar`.
- **Mock data = the API contract:** `web/src/lib/*-data.ts` (one per page) holds typed mock data behind a `fetch*()` shim; player detail synthesized in `web/src/lib/player-detail.ts`. **Sub-project B's 17 `/api/*` endpoints are BUILT (see B status above) — the next wiring job is to swap these mock modules for the real client; the shapes ARE the contract.**
- **NEXT (handoff):** (1) error/empty states on Optimizer/Matchup/Trades (Team page `app/page.tsx` is the four-state reference — unreachable on mocks but needed for real data); (2) push Trades/Players to the Optimizer's polish bar; (3) **wire `web/src/lib/*-data.ts` to the live B API** (`api/openapi.json` is the contract). Frontend specs: `docs/superpowers/specs/2026-06-16-heater-frontend-foundation-design.md`; uplift spec/plan in `docs/superpowers/{specs,plans}/2026-06-17-heater-10-of-10-uplift*.md`.

The codebase is organized around 7 feature surfaces:

1. **Draft Tool** (`app.py`) — Heater-themed splash + bootstrap + setup wizard + 3-column draft page with Monte Carlo recommendations.
2. **In-Season Pages** (`pages/`) — 14 pages (13 from the 2026-05-18 Section 5 consolidation + Pitcher Streaming added 2026-06-10): My Team, Lineup Optimizer, Closer Monitor, Pitcher Streaming, Matchup Planner, League Standings (+ Playoff Odds tab), Punt Analyzer, Trade Analyzer, Trade Finder (+ Value Chart tab), Free Agents, Player Compare, Leaders (+ Hot/Cold/Sell-High tabs), Player Databank, Draft Simulator. Bullpen, Weekly_Dashboard, Weekly_Recap, Waiver_Wire pages removed — their underlying engine modules (`src/contextual_factors.py`, `src/weekly_report.py`, `src/waiver_wire.py`) remain callable.
3. **Trade Analyzer Engine** (`src/engine/`) — 6-phase pipeline: deterministic SGP → stochastic MC (paired, true antithetic) → signal intelligence → contextual adjustments → game theory → production convergence/caching.
4. **Lineup Optimizer** (`src/optimizer/`) — 21-module pipeline with PuLP LP, daily category value (DCV) scoring, sigmoid urgency, FA recommender, sensitivity analysis, backtest framework.
5. **Draft Recommendation Engine** (`src/draft_engine.py`) — 8-stage enhancement chain with 3 execution modes (Quick/Standard/Full).
6. **War Room** (`src/war_room*.py`) — Mid-week pivot analysis, category flip probability, hot/cold detection.
7. **Backtesting** (`src/backtesting*.py` + `src/optimizer/backtest_runner.py`) — Historical replay framework validating recommendations against actual outcomes.

## Local Environment

- **Project root (local):** `C:\Users\conno\Code\HEATER_v1.0.1` (note: local folder is `v1.0.1`; GitHub repo NAME remains `HEATER_v1.0.0`).
- **Relocated 2026-05-17** from `C:\Users\conno\OneDrive\Desktop\HEATER_v1.0.0`. OneDrive's Cloud Files API conflicts with Cowork's FUSE/virtiofs mount layer (anthropics/claude-code issues #25293 and #40973), corrupts SQLite WAL files when streamlit runs locally, locks `.venv` operations, and conflicts with `.git` writes. Never store this project under OneDrive, Dropbox, iCloud, or any cloud-sync directory.
- **Local Python:** 3.14 preferred, 3.12 acceptable. Recreate venv with `py -3.14 -m venv .venv` (or `py -3.12 -m venv .venv`).
- **Yahoo OAuth deps:** `yfpy` and `streamlit-oauth` must be installed with `--no-deps` after `pip install -r requirements.txt`, per the comment in `requirements.txt` (python-dotenv pin conflict on 3.14).
- **Hooks:** Reinstall after any fresh venv via `python scripts/install-hooks.py`.
- **Multi-user mode (v2, additive):** Set `MULTI_USER=1` to enable league-mate self-registration + admin-approved team assignment. Off (unset) = single-user v1 behavior, byte-for-byte. Admin is seeded from `ADMIN_USERNAME` / `ADMIN_PASSWORD` / `ADMIN_TEAM_NAME` env vars (idempotent — set once). Auth lives entirely in `src/auth.py`; identity is per-session (`st.session_state["auth_user"]`) and replaces the global `league_teams.is_user_team` flag for personalized surfaces only. Plan 2 adds two additive, MULTI_USER-gated surfaces: a per-feature feedback inbox (`src/feedback.py` → Admin Console "Feedback" tab) and per-page usage logging (`src/usage.py`, one deduped view per session + `last_seen_at`). Both are inert when the flag is off. Plan 3 adds the admin dashboard: role-aware `st.navigation()` replaces Streamlit's automatic `pages/` discovery when the flag is ON (OFF still uses auto-discovery = v1), plus five additive tables (`feature_flags`, `audit_log`, `app_settings`, `sessions`, `page_visits`), four new modules (`src/audit.py`, `src/app_settings.py`, `src/feature_flags.py`, `src/nav.py`) + a rewritten `src/usage.py` (session/dwell timing, `bump_activity()` heartbeat, lazy idle-close, analytics readers), per-page feature flags (`require_page_enabled`, "absence = enabled"), view-as-user impersonation in `src/auth.py`, broadcast + maintenance banners, CSV exports (usage + feedback), and an append-only admin audit log. The Admin Console moved to `pages/_admin_console.py` (joined by `pages/_admin_analytics.py` + `pages/_admin_controls.py`); the leading `_` hides all three from v1 auto-discovery — they route only via `st.navigation()`. Every new surface is inert when the flag is off. **Plan 4 (Railway hosting)** ships the deploy surface: `Dockerfile` (`python:3.12-slim`, `--no-deps` yfpy/streamlit-oauth, headless `$PORT` bind), `railway.toml` (DOCKERFILE builder, `/_stcore/health`, `numReplicas = 1`), and `.dockerignore` (keeps `data/seed/`). Under `MULTI_USER=1` the in-process scheduler thread (`src/scheduler.py`, started once in `main()` after the flag-off early return) is the SOLE SQLite writer running `bootstrap_all_data(force=False)`; the 12 user sessions are read-only — `render_single_user_app()` shows a freshness caption + admin-only manual refresh + a "warming up" gate (driven by `_latest_successful_refresh()`, the same refresh-log signal as the badge) instead of the v1 per-session splash bootstrap. Admins seed headless Yahoo reconnect by pasting `yahoo_token.json` into the Admin Controls "Yahoo token" section (`save_yahoo_token_json` → volume). Single replica is a hard invariant (in-process writer + SQLite). Operator steps live in `docs/deployment/railway-runbook.md`. All Plan 4 deltas are flag-gated; flag-off stays v1 byte-for-byte.

## Commands

```bash
pip install -r requirements.txt        # Install deps (or `uv pip install --system -r requirements.txt` for ~10× faster)
python scripts/install-hooks.py        # Install pre-commit + pre-push hooks (first time, per machine)
python load_sample_data.py             # Load sample data (first time/testing)
streamlit run app.py                   # Run the app
python -m ruff check .                 # Lint
python -m ruff format .                # Format
python -m pytest --ignore=tests/test_cheat_sheet.py  # Full suite (~5361 pass; cheat_sheet skipped on Windows)
python -m pytest tests/test_foo.py -v  # Single test file

# Parallel test execution (matches CI's sharded layout)
python -m pytest tests/ -n auto --dist loadfile      # Run all in parallel locally (~4× faster)
python -m pytest tests/ --splits 4 --group 1 -n 2 --dist loadfile  # Run shard 1 of 4
python -m pytest tests/ --store-durations             # Regenerate .test_durations (weekly cadence)

# Optimizer validation tools
python scripts/run_backtest.py --quick          # Replay historical MLB weeks, score accuracy
python scripts/compute_empirical_stats.py       # Compare correlation/CV defaults vs real MLB data
python scripts/calibrate_sigmoid.py --full      # Grid-search optimal sigmoid k-values
python scripts/calibrate_repl_baselines.py      # Refresh _REPL_* from FourzynBurn standings (annual)
```

## League Context

- **League:** FourzynBurn (Yahoo Sports) | **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds, Head-to-Head Categories
- **Hitting cats (6):** R, HR, RBI, SB, AVG, OBP
- **Pitching cats (6):** W, L, SV, K, ERA, WHIP
- **Inverse cats:** L, ERA, WHIP (lower is better)
- **Rate stats:** AVG, OBP, ERA, WHIP
- **Roster:** C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/6BN/4IL = 28 slots
- **Yahoo game_key:** 469
- **Transactions:** 10 adds+trades combined per matchup week (FCFS waivers)
- **Can't-drop:** Players drafted in rounds 1-3 per team (36 league-wide)

## Tech Stack

- **Framework:** Streamlit (multi-page)
- **Database:** SQLite (`data/draft_tool.db`) with WAL + 60s busy_timeout via `get_connection()`
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Analytics:** PyMC 5 (Bayesian, optional), PuLP (LP optimizer), arviz
- **Live data:** MLB-StatsAPI, pybaseball (FanGraphs/Statcast), Open-Meteo (weather), Baseball Savant (catcher framing)
- **Yahoo API:** yfpy + streamlit-oauth (optional OAuth)
- **Linter:** ruff (lint + format), pre-commit + pre-push hooks enforced
- **CI:** GitHub Actions — ruff lint/format + pytest sharded (4 groups × 2 xdist workers, Python 3.12) + 60% coverage floor; uv-based dep install; concurrency cancels in-flight PR runs; expected ~5-7 min total wall time
- **Python:** Local dev uses 3.14; CI tests on 3.12 (sharded 4 ways via pytest-split)

## File Structure

```
app.py                      — Splash, bootstrap, setup wizard, draft page
requirements.txt            — pip dependencies
load_sample_data.py         — Sample data generator for testing
.streamlit/config.toml      — Native ≥1.47 advanced theming: Combustion palette, fonts-by-URL, [theme.sidebar], chartCategoricalColors

pages/ — 14 in-season pages, ordered for daily workflow (status → daily action → strategy → trades → wire → research → preseason):
  # Status & this-week context
  1_My_Team.py              — War Room, roster, category standings, alerts, Yahoo sync
  2_Line-up_Optimizer.py    — 6-tab optimizer: Start/Sit, Optimize, Manual, Streaming, Daily, Roster
  3_Closer_Monitor.py       — 30-team closer depth chart grid
  4_Pitcher_Streaming.py    — 4-tab stream analyzer: Finder (date-driven Stream Score board) /
                              Microscope (lineup+PvB+game-log deep dive) / Week Planner / Track Record
  5_Matchup_Planner.py      — Category probabilities, player matchups, per-game detail
  6_League_Standings.py     — 3 tabs: Current Standings / Season Projections / Playoff Odds
  10_Punt_Analyzer.py       — Punt-category strategy recommender
  # Trades
  11_Trade_Analyzer.py      — Trade proposal builder + 6-phase engine
  12_Trade_Finder.py        — 5 tabs: Recs / Target / Browse Partners / Readiness / Value Chart
  # Wire
  14_Free_Agents.py         — FA rankings by marginal value + ownership heat
  # Research
  16_Player_Compare.py      — Head-to-head comparison with category fit
  17_Leaders.py             — 7 tabs: Leaders / Value / Breakouts / Prospects / Hot / Cold / Sell-High
  19_Player_Databank.py     — Historical multi-year player lookup
  # Preseason
  20_Draft_Simulator.py     — Draft simulator with AI opponents, MC recommendations

# Section 5 (2026-05-18) consolidated 20 → 13 pages;
# 4_Pitcher_Streaming added 2026-06-10 (slot 4 reused) → 14 pages.
# Removed pages: 4_Bullpen, 7_Playoff_Odds, 8_Weekly_Dashboard,
# 9_Weekly_Recap, 13_Trade_Values, 15_Waiver_Wire, 18_Trends.
# Their core engine logic lives in src/ and remains importable.

src/
  # Core
  valuation.py              — LeagueConfig (sole source of truth), SGPCalculator, VORP
  database.py               — SQLite schema, _build_player_pool (full pool with all ytd_*, statcast)
  data_bootstrap.py         — 33-phase bootstrap orchestrator (3-tier waterfall on most phases)
  data_pipeline.py          — FanGraphs auto-fetch (Steamer/ZiPS/Depth Charts)
  data_fetch_utils.py       — fetch_with_fallback + browser-headers helpers
  yahoo_api.py              — Yahoo Fantasy API OAuth (429 backoff, ghost filtering)
  yahoo_data_service.py     — 3-tier cache: session_state → Yahoo API → SQLite

  # Draft
  draft_engine.py / draft_state.py / draft_analytics.py / draft_grader.py
  draft_order.py / simulation.py / pick_predictor.py

  # Trade system
  trade_finder.py           — V4 trade finder (cosine→1v1→2v1)
  trade_intelligence.py     — Health/scarcity/FA gating/correlation
  trade_value.py            — Universal trade value chart (0-100)
  trade_signals.py          — Kalman + regime trend
  opponent_trade_analysis.py — Opponent needs, archetype detection
  in_season.py              — Legacy trade analyzer, player comparison

  # Lineup optimization
  lineup_optimizer.py       — PuLP LP solver (sole PULP_AVAILABLE source)
  lineup_rl.py              — Category-aware RL optimizer

  # Matchup & strategy
  matchup_context.py        — MatchupContextService (canonical category weights)
  matchup_planner.py / standings_engine.py / standings_utils.py
  weekly_h2h_strategy.py / war_room*.py
  start_sit.py / start_sit_widget.py

  # Player intelligence
  bayesian.py / injury_model.py / ml_ensemble.py / playing_time_model.py
  projection_stacking.py / ecr.py / prospect_engine.py
  player_news.py / news_fetcher.py / news_sentiment.py
  espn_injuries.py / player_card.py / player_tags.py

  # Game-day intelligence
  game_day.py               — Weather, opposing pitchers, team strength, lineups
  closer_monitor.py / schedule_grid.py / contextual_factors.py

  # In-season management
  alerts.py / opponent_intel.py / waiver_wire.py / weekly_report.py
  ip_tracker.py / il_manager.py / two_start.py / power_rankings.py / leaders.py
  league_rules.py           — Undroppable list + 10 txn/week + FCFS

  # Multi-user (v2)
  auth.py                   — register→pending→admin-approve→active; MULTI_USER-gated; per-session identity
  version.py                — APP_VERSION string (env-overridable via HEATER_APP_VERSION); stamped onto feedback
  feedback.py               — per-feature feedback capture + admin inbox helpers + render_feedback_widget popover (MULTI_USER-gated)
  usage.py                  — per-session deduped page-view logging + last_seen_at bump (MULTI_USER-gated)

  # Data sources
  live_stats.py / adp_sources.py / depth_charts.py / contract_data.py
  marcel.py / data_2026.py

  # Backtesting & validation
  backtesting.py / backtesting_framework.py
  validation/               — Calibration, constant optimization, dynamic context, survival

  # UI
  ui_shared.py              — THEME, PAGE_ICONS, format_stat (canonical formatter), Combustion design system CSS (locked by test_combustion_lock.py)
  ui_analytics_badge.py / cheat_sheet.py

  # Misc
  league_manager.py / league_registry.py / live_draft_sync.py
  points_league.py / scheduler.py / analytics_context.py / player_databank.py

  # Sub-packages
  optimizer/                — Enhanced Lineup Optimizer (21 modules)
    pipeline.py             — Master orchestrator (Quick/Standard/Full modes)
    projections.py          — Enhanced projections (Bayesian/Kalman/regime/injury/recent form)
    matchup_adjustments.py  — Park, platoon, weather, umpire, catcher framing, PvB
    h2h_engine.py           — H2H category weights + win probability
    sgp_theory.py           — Non-linear marginal SGP
    streaming.py            — Pitcher streaming + two-start fatigue (uses constants_registry)
    stream_analyzer.py      — Pitcher Streaming page engine: Stream Score board, matchup deep-dive,
                              week planner, historical replay (composes streaming/two_start/matchup_adjustments)
    scenario_generator.py   — Gaussian copula + CVaR + empirical correlation
    dual_objective.py       — H2H/Roto weight blending
    advanced_lp.py          — Maximin, epsilon-constraint, stochastic MIP (imports PULP_AVAILABLE)
    category_urgency.py     — Sigmoid urgency (reads k-values from CONSTANTS_REGISTRY at runtime)
    daily_optimizer.py      — Daily Category Value (DCV) scoring + Stuff+/FIP K-boost
    fa_recommender.py       — FA recommendation engine
    pivot_advisor.py        — Mid-week pivot recommendations
    shared_data_layer.py    — Unified data infrastructure + freshness tracker calls
    data_freshness.py       — Per-source staleness tracking with TTLs + UI badge
    constants_registry.py   — 30+ optimizer constants with citations, bounds, sensitivity
    backtest_validator.py   — RMSE, Spearman, bust rate, lineup grading
    backtest_runner.py      — Historical replay via statsapi game logs
    sensitivity_analysis.py — Mock.patch perturbation of constants
    sigmoid_calibrator.py   — Grid-search calibration
  engine/                   — Trade Analyzer Engine (6 phases, 7 sub-packages)
    portfolio/              — Z-scores, SGP via SGPCalculator (no local _LC singleton)
    projections/            — ROS projections, BMA, KDE marginals
    monte_carlo/            — Paired MC simulator (TRUE antithetic via negate-uniforms)
    signals/                — Statcast, decay, Kalman filter, regime detection
    context/                — Log5 matchup, injury process, bench value
    game_theory/            — opponent_valuation (sgp_denominators required), Bellman DP, sensitivity
    production/             — Convergence diagnostics, cache, adaptive sim scaling
    output/                 — Master trade orchestrator (evaluate_trade)

scripts/
  install-hooks.py / pre-commit
  run_backtest.py / compute_empirical_stats.py / calibrate_sigmoid.py
  draft_vs_current.py / extract_trade_data.py / optimal_roster_sim.py  (all use get_connection)

  # FA engine diagnostics (preserved 2026-05-20 → 2026-05-21 for ongoing debug)
  diag_fa_cross_type.py            — Traces _allow_cross_type for known bad rec pairs
  diag_fa_with_filter.py           — Verifies user-roster slicing fix end-to-end
  diag_fa_full_trace.py            — Stage-by-stage pipeline trace (drops, FAs, evaluate_swaps)
  diag_fa_score_plateau.py         — Inspects _compute_base_value internals for inverse-stat bug
  diag_fa_stage_by_stage.py        — Full recommend_fa_moves trace with intermediate counts
  diag_fa_adds_budget.py           — ctx.adds_remaining_this_week + transactions check
  diag_muncy_data.py               — DB column inspection for Muncy DNA collision
  diag_roster_data_audit.py        — Audits ALL rostered players for DNA collisions + stale YTD
  diag_fa_score_plateau.py         — Why unknown minor leaguers score composite=20.6

  # One-time DB migration (2026-05-21)
  migrate_muncy_dna_2026_05_21.py  — Insert LAD Muncy + repoint league_rosters (idempotent)

tests/                      — 448 test files, ~5361 passing tests
  conftest.py               — Session-scoped league_standings fixture (autouse)
  test_no_*.py              — Structural-invariant guards (see Structural Invariants section)
  test_sf*.py               — TDD tests for SF-1..SF-28 fixes
data/
  draft_tool.db             — SQLite database (created at runtime)
  seed/                     — 2024 seed JSONs (catcher_framing, umpire_tendencies) for SF-7 Tier 3
  backups/                  — Draft state JSON backups
  logs/bootstrap.log        — Persistent bootstrap log (SF-14)
docs/
  architecture.md           — Layered system architecture reference
  Research.md               — Competitive gap analysis vs FantasyPros
  VERIFICATION_LOG.md       — V-001..V-017 browser-verified validation sessions
  2026-04-08-heater-postmortem.md  — Early-season recommendation-failure RCA
  legal/                    — LLC formation, ToS, privacy policy, data sources, etc.
  archive/                  — Stale historical docs (ROADMAP, AUDIT_REPORT, BETA_*)
.github/
  workflows/ci.yml          — CI pipeline (lint + sharded pytest 4×2 workers on 3.12 + coverage floor + build check)
  workflows/refresh.yml     — Scheduled daily data refresh
```

## Architecture

### Core Valuation Pipeline
1. **Projection blending** — Ridge regression weighted stacking of 7 systems (Steamer, ZiPS, Depth Charts, ATC, THE BAT, THE BAT X, Marcel).
2. **Dynamic SGP** — Converts raw stats to standings-point movement; Bayesian-updated denominators from league data.
3. **VORP** — Value Over Replacement Player with graduated positional scarcity (C=1.20×, 2B=1.15×, SS=1.10×).
4. **pick_score** = weighted SGP + positional scarcity + VORP bonus.

### Unified Services (sole sources of truth)
- **`LeagueConfig`** (`src/valuation.py`) — Category lists, inverse stats, rate stats, SGP denominators. **Never hardcode category lists** — structural test guards this in src/, pages/, and tests.
- **`SGPCalculator`** (`src/valuation.py`) — Sole path for SGP computation. The `totals_sgp(totals, weights=None)` method consolidates 7 prior local reinventions across trade_finder, waiver_wire, daily_optimizer, trade_evaluator, trade_simulator, in_season, start_sit. **`marginal_sgp(player, roster_totals, weights)`** handles inverse-stat signs correctly (PR #99 / FA P3.7) — consumed via `_compute_base_value` in `fa_recommender.py`.
- **`MatchupContextService`** (`src/matchup_context.py`) — Single source for category weights with 3 modes: "matchup" (H2H urgency), "standings" (gap analysis), "blended" (alpha-weighted).
- **`standings_utils.get_team_totals()`** — Session-cached roster totals.
- **`get_yahoo_data_service()`** — Singleton; 3-tier cache: session_state → Yahoo API → SQLite. All 11+ pages route through it (no raw `load_league_rosters/standings` in pages).
- **`format_stat(value, stat_type)`** (`src/ui_shared.py`) — Enforced display: AVG/OBP=`.3f`, ERA/WHIP=`.2f`, SGP=`+.2f`.
- **`get_connection()`** (`src/database.py`) — Only sanctioned SQLite access. Sets `PRAGMA journal_mode=WAL`, `busy_timeout=60000`, `synchronous=NORMAL`. Direct `sqlite3.connect` is structurally guarded against.
- **`PULP_AVAILABLE`** — Defined ONCE in `src/lineup_optimizer.py`. `src/optimizer/advanced_lp.py` imports it (no duplicate flag).
- **`CONSTANTS_REGISTRY`** (`src/optimizer/constants_registry.py`) — 50+ optimizer/FA-engine constants with citations, bounds, sensitivity tags. **Never hardcode tunable thresholds in `fa_recommender.py` / `waiver_wire.py`** — register them. After PR #107 the FA engine reads all major thresholds (streaming, playing-time, ownership, floor, position-cap, urgency) from the registry at import time.
- **`compute_positional_scarcity_factor(positions, replacement_levels)`** (`src/valuation.py`) — Shared scarcity math (PR #96). Both `fa_recommender._score_fa_candidates` (adds) and `waiver_wire.compute_drop_cost` (drops) consume it for symmetric scarcity treatment. Moved here from `fa_recommender.py` to break the `waiver_wire ← fa_recommender` import cycle.
- **`get_il_stash_names()`** (`src/alerts.py`) — Dynamic derivation of IL stash names from `league_rosters.status IN (IL10, IL15, IL60, IL, DTD)` (PR #102). The module-level `IL_STASH_NAMES` constant calls this at import time. Pages should call the function directly for fresh data each render. Now surfaces ~52 IL players league-wide vs the previously hardcoded 2 (Bieber + Strider).

### Yahoo Data Service (Live-First Architecture)
3-tier cache: `st.session_state` → Yahoo Fantasy API → SQLite fallback.
TTLs: Rosters 30m, Standings 30m, Matchup 5m, Free Agents 1h, Transactions 15m, Settings/Schedule 24h.
Write-through: every Yahoo fetch writes to SQLite. Singleton via `get_yahoo_data_service()`.

### Bootstrap Pipeline (33 phases, 3-tier waterfall)
Bootstrap runs with `force=True` on every new browser session (per-session guard `bootstrap_complete` prevents re-run on intra-session navigation). The "Refresh All Data" sidebar button passes `force=True`, clears `st.cache_data`, preserves `bootstrap_results` for the Data Status panel, and records a new elapsed timer.

Most phases follow a 3-tier waterfall:
- **Tier 1 (primary)** — Live API (pybaseball / FanGraphs / Yahoo / statsapi)
- **Tier 2 (fallback)** — Browser-headers scraper (where applicable, e.g. Savant for SF-7)
- **Tier 3 (emergency)** — Shipped seed file or hardcoded baseline (catcher_framing, umpire_tendencies, park_factors)

`refresh_log` records `tier='primary'/'fallback'/'emergency'` per phase. Operators can query the full snapshot via `get_refresh_log_snapshot()` (added 2026-05-11).

Per-phase timeouts: default 180s, 300s for game_logs + ROS projections (PyMC), 240s for ECR consensus.

### Player Pool (`_build_player_pool`)
The canonical enriched pool returned by `load_player_pool()` includes:
- Projection-blended stats (R/HR/RBI/SB/AB/H/BB/HBP/SF/AVG/OBP for hitters; W/L/SV/K/IP/ER/BB_allowed/H_allowed/ERA/WHIP for pitchers; FIP/xFIP/SIERA)
- ECR rank + consensus + sources + stddev
- ADP from FanGraphs/FantasyPros/NFBC
- Statcast: xwoba, xba, barrel_pct, hard_hit_pct, ev_mean, stuff_plus, babip, **sprint_speed**
- YTD stats (full set, supports weighted aggregation): `ytd_r/hr/rbi/sb/avg/obp/ab/h/bb/hbp/sf/pa/gp` + `ytd_w/l/sv/k/era/whip/ip/er/bb_allowed/h_allowed`
- Regression flags (`regression_flag`, `babip_regression_flag`, `stuff_regression_flag`)
- `bats`/`throws` (enables platoon splits)
- `percent_owned` (ownership trends)

Pages should consume the pool — direct SQL is structurally guarded against in `pages/1_My_Team.py`, `pages/11_Trade_Analyzer.py`, `pages/16_Player_Compare.py`.

## Key API Signatures

```python
# LeagueConfig + SGPCalculator — sole source of truth
from src.valuation import LeagueConfig, SGPCalculator
cfg = LeagueConfig()
calc = SGPCalculator(cfg)
calc.totals_sgp(totals_dict, weights=None) -> float   # Sole path for roster-totals SGP
calc.player_sgp(player_series) -> dict                # Per-player per-category SGP

# Yahoo Data Service
from src.yahoo_data_service import get_yahoo_data_service
yds = get_yahoo_data_service()
yds.get_rosters() -> pd.DataFrame                     # 30min TTL
yds.get_standings() -> pd.DataFrame                   # 30min TTL
yds.get_matchup() -> dict | None                      # 5min TTL
yds.get_free_agents(max_players=500)                  # 1hr TTL
yds.get_transactions() -> pd.DataFrame                # 15min TTL
yds.is_connected() -> bool

# Player pool
from src.database import load_player_pool, load_season_stats
pool = load_player_pool()                             # Full enriched pool with all ytd_*

# Trade evaluator
from src.engine.output.trade_evaluator import evaluate_trade
evaluate_trade(giving_ids, receiving_ids, user_roster_ids, player_pool,
               config=None, enable_mc=False, enable_context=True, enable_game_theory=True)

# Trade finder
from src.trade_finder import find_trade_opportunities
find_trade_opportunities(user_roster_ids, player_pool, config=None,
    all_team_totals=None, league_rosters=None, max_results=20)

# Engine portfolio (config required for live denominators)
from src.engine.portfolio.category_analysis import compute_marginal_sgp, category_gap_analysis
compute_marginal_sgp(player_totals, config)           # config now required (no module _LC)
category_gap_analysis(user_totals, all_team_totals, user_team_name, weeks_remaining, config)

# Engine game theory (sgp_denominators required)
from src.engine.game_theory.opponent_valuation import estimate_opponent_valuations, player_market_value
estimate_opponent_valuations(rosters, sgp_denominators=cfg.sgp_denominators, ...)

# Lineup optimizer
from src.optimizer.pipeline import LineupOptimizerPipeline
LineupOptimizerPipeline(roster, mode="standard", alpha=0.5, weeks_remaining=16, config=None)

# Bootstrap
from src.data_bootstrap import bootstrap_all_data
bootstrap_all_data(yahoo_client=None, on_progress=None, force=False)

# Refresh log snapshot (ops/debug visibility)
from src.database import get_refresh_log_snapshot
get_refresh_log_snapshot() -> list[dict]              # source, status, tier, message, last_refresh

# MatchupContextService
from src.matchup_context import get_matchup_context
get_matchup_context().get_category_weights(mode="matchup"|"standings"|"blended", ...)

# Auto-advance game date
from src.game_day import get_target_game_date
get_target_game_date()  # Today, or tomorrow if all today's games final

# Stat formatting
from src.ui_shared import format_stat
format_stat(0.275, "AVG")  # "0.275"
format_stat(3.85, "ERA")   # "3.85"
format_stat(2.34, "SGP")   # "+2.34"

# Pitcher Streaming engine (canonical engine for the Pitcher Streaming page;
# also powers the Lineup Optimizer Streaming tab's Section 1 after the Phase 5 DRY)
from src.optimizer.stream_analyzer import (
    build_stream_board,        # (ctx, target_date, schedule=None, include_rostered=False) -> DataFrame
    score_stream_candidate,    # (pitcher_row, start_info, opp_context, config, ...) -> dict (0-100 + components)
    build_week_plan,           # (ctx, schedule=None, max_adds=None, base_weekly_ip=None) -> {"plan", "summary"}
    replay_stream_date,        # (ctx, past_date, top_n=5) -> {"board_then", "actuals", "summary", "proxy_caveat"}
    get_pitcher_vs_team_history,  # statsapi gameLog rows; aggregate_pitcher_history for weighted ERA/WHIP
    compute_lineup_exposure,   # regressed PvB lineup wOBA delta vs league (None = neutral)
)

# Weekly transaction budget — canonical home (streaming.WEEKLY_ADDS_BUDGET mirrors it)
from src.league_rules import WEEKLY_TRANSACTION_LIMIT  # 10

# Optimizer validation tools
from src.optimizer.data_freshness import DataFreshnessTracker
from src.optimizer.constants_registry import CONSTANTS_REGISTRY
from src.optimizer.sensitivity_analysis import run_sensitivity_analysis
from src.optimizer.backtest_runner import run_backtest, format_report
from src.optimizer.sigmoid_calibrator import calibrate_sigmoid_k

# FA recommender (canonical engine for Free Agents page)
from src.optimizer.fa_recommender import recommend_fa_moves, recommend_streaming_moves
from src.optimizer.shared_data_layer import build_optimizer_context

# build_optimizer_context accepts level_filter (default "MLB only") and
# auto-filters multi-team rosters by user_team_name (PR #98 / FA P3.6)
ctx = build_optimizer_context(
    scope="rest_of_season", yds=yds, config=config,
    weeks_remaining=compute_weeks_remaining(),
    user_team_name=user_team_name,
    roster=user_roster,                     # pre-filtered single-team slice preferred
    level_filter="MLB only",                # filters minor leaguers from ctx.player_pool
)
recommend_fa_moves(ctx, max_moves=5)        # post-PR #110 engine

# Shared scarcity factor (used by both add-side and drop-side scoring)
from src.valuation import compute_positional_scarcity_factor
compute_positional_scarcity_factor(positions_str, replacement_levels_dict) -> float  # 1.0 - 1.25x

# Dynamic IL stash (replaces hardcoded {"Bieber", "Strider"})
from src.alerts import get_il_stash_names
get_il_stash_names() -> set[str]            # Fresh from league_rosters.status

# News-derived suspension end-dates (PR #109 / FA P5d)
from src.news_sentiment import parse_suspension_days
parse_suspension_days("Suspended through 5/26", reference_date=datetime.now(UTC)) -> int | None

# IL weighting with return-date awareness (PR #95 + PR #105)
from src.in_season import _il_weight_from_status, _return_date_weight
_il_weight_from_status(status: str, expected_return_days: float | None = None) -> float
# Curve: 0d→1.0, 1d→0.95, 7d→0.85, 14d→0.70 (= IL10), 21d→0.55, 45d→0.30, 60d→0.20
# Also parses day counts inline from status strings (PR #105): "IL10 - 3 days" → 3
```

## Gotchas

### Data & Schema
- **`LeagueConfig` is the single source of truth** — Never hardcode category lists. Structurally guarded by `tests/test_no_hardcoded_categories_in_src.py` + sibling per-file guards.
- **DB column `name` vs UI `player_name`** — Normalized via `rename(columns={"name": "player_name"})` in `_build_player_pool()`.
- **Sample data is `system='blended'` only** — Do NOT call `create_blended_projections()` when only blended data exists.
- **Python 3.14 + SQLite bytes** — Returns bytes for some integer columns. Fixed with `CAST` in SQL and `pd.to_numeric()`.
- **Players table has no UNIQUE constraint on name** — Uses SELECT-first pattern. Do NOT use `ON CONFLICT(name)`.
- **Park factors schema** — Columns: `team_code`, `factor_hitting`, `factor_pitching`. Tier 1 (pybaseball) honored when available; emergency dict when not (SF-22).
- **FanGraphs API SYSTEM_MAP** — FG API uses `"fangraphsdc"`, DB stores `"depthcharts"`. Always use `SYSTEM_MAP`.
- **SQLite WAL mode + busy_timeout** — `get_connection()` sets `PRAGMA journal_mode=WAL`, `busy_timeout=60000`, `synchronous=NORMAL`. Direct `sqlite3.connect()` is structurally guarded against (`tests/test_no_direct_sqlite_connect_in_scripts.py`). Bumped from 30s → 60s in PR #69 after parallel-write contention hit the lower ceiling.
- **FanGraphs 403 handling** — `leaders-legacy.aspx` returns 403 to non-browser scrapers. SF-6 wired browser-headers scraper as Tier 2 (probe still 403). FIP/xFIP proxy serves as K-boost fallback when Stuff+ unavailable.
- **`statcast_archive.sprint_speed` column** — Required for SF-19 player pool SELECT. `init_db` creates it; legacy DBs get it via `_safe_add_column` migration.
- **Player pool ytd_* columns are complete** — Includes all hitting + pitching counting + rate components needed for weighted aggregation. Consumers no longer need separate `load_season_stats(year)` calls.

### Yahoo API
- **OAuth uses oob** — `redirect_uri=oob` required. No localhost redirects.
- **Auto-reconnect** — `_try_reconnect_yahoo()` loads `data/yahoo_token.json`. Token refresh automatic.
- **★ TOKEN RELAY (live prod — `src/token_relay.py` + `scripts/yahoo_token_relay.py`):** Yahoo **blocks OAuth token refresh from Railway's datacenter IP**, so the live API server (App B, `celebrated-respect`) cannot refresh its own token. Instead the owner's **local machine** runs a Windows Task Scheduler job (`\HEATER Yahoo Relay`, every 30 min) that refreshes the token from a residential IP, encrypts it (Fernet `HEATER_RELAY_KEY`), and uploads it to a private GitHub gist (`HEATER_GIST_ID`/`HEATER_GIST_PAT`); App B pulls it via `pull_relayed_token()` (`HEATER_TOKEN_RELAY_URL`+`HEATER_RELAY_KEY`). So App B's `token_time` advancing ≈ the relay working; App B's own `YAHOO_CLIENT_ID/SECRET` env are effectively unused (the pulled token's embedded creds drive everything). **To switch Yahoo apps:** oob re-auth with the new Client ID → write the token to local `data/yahoo_token.json` → run `scripts/yahoo_token_relay.py` once → App B auto-pulls within ~8 min (NO Railway redeploy). The migration roadmap moves this to Postgres + a server-side worker at M4.
- **★ Two versions, two Yahoo apps:** **v1.0.0** = the original Streamlit app (Railway `heaterv100-production`, App A) on the OLD Yahoo app `5dliBbR7`. **v1.0.1** = the version built on going forward = Vercel React (`heater-v1-0-1.vercel.app`) + the FastAPI API (App B) on the NEW Yahoo app `almo0Hnw` (switched 2026-06-25). **Yahoo WRITE (`fspt-w`) is NOT self-serve available** — Yahoo's console shows only "Read" for new AND existing apps, so "Apply to Yahoo" lineup writes can't be enabled (external Yahoo limit, same class as the FanGraphs 403); HEATER stays a read-only advisor.
- **yfpy Roster iteration** — `Roster.__iter__()` yields attribute NAMES (strings), NOT players. Use `getattr(roster, "players", None) or []`.
- **Yahoo game_key** — MLB 2026 = 469. Must set BOTH `self._query.game_id` AND `self._query.league_key`.
- **429 backoff** — `_request_with_backoff()` retries 3× with exponential delay.
- **yfpy eligible_positions bug** — yfpy's `Player.eligible_positions` drops multi-position data. Workaround: read from `player._extracted_data["eligible_positions"]` which preserves the full list. Separator is `,` (not `/`).
- **Optimize click uses cached Yahoo data (T1.21)** — `pages/2_Line-up_Optimizer.py`'s optimize handler does NOT force-refresh Yahoo. Users force-refresh via the "Refresh Yahoo Data" sidebar button. This decoupling closed the 159-second OAuth hang documented in the v2 design spec. `_get_cached` in `src/yahoo_data_service.py` also wraps every Yahoo fetch in a 15-second `ThreadPoolExecutor` timeout so a hung Yahoo call cannot block page rendering. Structurally guarded by `tests/test_oauth_decoupling.py`.

### Streamlit UI
- **Combustion Index design system (2026-06-09, locked; finale layer 2026-06-10)** — Pure-white canvas (owner request 2026-06-10; was off-white + blueprint grid) + pale `#f5f6f8` surface fills with darker bolded text + deep-navy chrome rail + orange `#ff6d00` primary (NO brand red; ember `#e63946` allowed only for functional-negative icons). Fonts: Archivo (display, `--font-display`, variable `wdth` axis 62..125 → `font-stretch` hero numerals), Inter (body, `--font-body`), IBM Plex Mono (figures, `--font-mono`) — Figtree/Bebas retired. `html, body` is root-locked to Inter and Streamlit's BaseWeb widgets are font-locked via a comma-joined selector rule. Clickable text uses the `.heater-link`/`.clickable` helper (orange + underline), scoped to main content — sidebar nav stays bone-on-navy. Full-width layout (`max-width: none`). **Finale micro-detail layer (2026-06-10):** ~~grain + blueprint-grid canvas~~ (REMOVED same day by owner request — `.stApp` is plain white and `test_combustion_lock` now asserts the texture stays gone), machined top-edge insets on cards, motion tokens (`--dur-1/--dur-2/--ease-snap`), brand `::selection` + `:focus-visible` ring, thin scrollbars (standard properties only — setting `scrollbar-color` disables `::-webkit-scrollbar`), `.chip`/`.hero-num`/`.hr-fade`/`.hr-heat`/`.empty-state`/`.t-*` utility classes, animated tab underline (`tab-highlight`), navy tooltip/toast portal chrome, `st.dialog` chrome (`min(92vw, 1100px)`), `:has()` metric self-tinting, `prefers-reduced-motion` block (kept LAST in sheet). NO entrance keyframes on persistent cards (Streamlit remounts DOM every rerun — they replay on every click). `.streamlit/config.toml` carries native ≥1.47 advanced theming (fonts-by-URL incl. Glide canvas first-paint, `chartCategoricalColors` = `THEME["tiers"]`, `[theme.sidebar]`). All machine-checked by `tests/test_combustion_lock.py` (24 tests).
- **Material Symbols spans are the ONE font-lock exception** — Streamlit renders widget glyphs (expander chevron, nav `expand_more`, dialog close) as icon-font ligature TEXT in `span[data-testid="stIconMaterial"]`. Any font-lock reaching them shows the literal ligature name ("keyboard_arrow_right"). `ui_shared.py` re-asserts `"Material Symbols Rounded"` on those spans AFTER the font-lock block — never remove it.
- **Preconnect/`<link>` tags live in their OWN `st.markdown` call** — a `<link>` opener demotes the markdown block to a CommonMark type-6 HTML block (ends at the first blank line), making the `<style>` sheet after it render as literal page text. The stylesheet markdown must START with `<style>` (type-1, raw-to-close).
- **`render_empty_state(title, body, icon_key)`** (`src/ui_shared.py`) — instrument-styled empty-data panel; replaces bare `st.info()` in data-empty contexts ONLY (operational/validation warnings stay native — e.g. the FA legacy-fallback banner). Escapes title/body.
- **`build_compact_table_html(..., html_cols={...})`** — columns whose cells are TRUSTED pre-built HTML (e.g. League Standings `_rank_badge` spans) opt out of escape-by-default cell rendering; all other cells stay escaped.
- **No emoji** — All icons are inline SVGs from `PAGE_ICONS`. Injury badges use CSS dots.
- **CSS `!important` required** — Streamlit's inline CSS has high specificity.
- **HTML sanitization** — `st.markdown('<div>')` auto-closes tags. Use self-contained HTML blocks.
- **`T["ink"]` for text-on-accent** — Not `T["bg"]` (invisible on amber). `T["tx"]` not `T["tx1"]`.
- **Connection leak** — Always wrap `get_connection()` in `try/finally` with `conn.close()`.
- **Splash-screen load timer** — HH:MM:SS live counter. Start in `st.session_state["bootstrap_start_time"]`. Final in `bootstrap_elapsed_secs` / `bootstrap_elapsed_hms`.
- **Format precision** — Use `format_stat(value, stat_type)` for all rate stats. Inline `f"{x:.3f}"` for ERA/WHIP is structurally guarded against (`tests/test_pages_format_compliance.py`).

### Algorithms & Math
- **Rate-stat aggregation** — AVG=sum(h)/sum(ab), OBP=sum(h+bb+hbp)/sum(ab+bb+hbp+sf), ERA=sum(er)*9/sum(ip), WHIP=sum(bb+h)/sum(ip). Weighted, NEVER simple averages.
- **LP inverse stat weighting** — ERA/WHIP weighted by IP. Without it, 1-IP relievers dominate starters.
- **L (Losses) is an inverse counting stat** — Sign-flip in SGP, no IP weighting in LP. Inverse-stat sets must include L (`{"L", "ERA", "WHIP"}`); guards prevent accidental `{"ERA", "WHIP"}` regressions.
- **Punt detection** — Requires BOTH: `gainable_positions == 0 AND rank >= 10`.
- **LP-constrained lineup totals** — Only 18 starters' stats feed SGP. Bench excluded.
- **Paired MC seed discipline** — Same seed for before/after rosters. NEVER different seeds.
- **True antithetic variates** — `trade_simulator.py` negates uniform quantiles (`u → 1-u`, `z → -z`) for paired arms; ~25% variance reduction.
- **SGP through `SGPCalculator.totals_sgp` only** — All 7 prior local reinventions migrated. Structurally guarded.
- **Sigmoid urgency reads from registry at runtime** — `category_urgency.py` reads `sigmoid_k_counting` / `sigmoid_k_rate` from `CONSTANTS_REGISTRY` per call, so `calibrate_sigmoid.py` updates take effect without restart.

### FA Recommender Engine (post-2026-05-21 overhaul)
The Free Agents page (`pages/14_Free_Agents.py`) is the sole consumer of `recommend_fa_moves` in production. The engine's correctness depends on these invariants — guard tests prevent regression:

- **`ctx.user_roster_ids` is the USER's team, not the whole league** — Pages pass a single-team roster slice to `build_optimizer_context`. The builder auto-filters multi-team rosters when `user_team_name` is set (PR #98 / FA P3.6). Without this filter, drop candidates would be drawn from all 12 teams.
- **`ctx.player_pool` respects `level_filter`** — Default `"MLB only"` filters minor leaguers (PR #98). Pages can opt into `"All"` for prospect lookups (Player Databank, Draft Simulator).
- **`_compute_base_value` delegates to canonical `SGPCalculator.marginal_sgp`** (PR #99 / FA P3.7) — Handles inverse-stat signs (ERA, WHIP, L are NEGATIVE contributions) and rate-stat volume weighting. The earlier inline formula `value += (stat/denom) * weight` had the wrong sign for inverse stats — unknown pitchers with default `era=9.0/whip=2.0` scored base=+38.8 (should be -38.8).
- **Position cap boundary semantics** — Block when post-swap count `> cap` (PR #100). Having exactly `cap` players at a position is healthy (`starter + 1 backup`), not a violation. Cap = Yahoo starting slots + 1. Off-by-one (`>=`) was blocking legitimate same-position upgrade swaps.
- **Roster-construction floors** — Never drop below 10 active hitters or 8 active pitchers (FourzynBurn starting lineup count). IL players count toward position caps but NOT toward active-roster floors (PR #97 / FA P3.5).
- **Symmetric positional scarcity** — `_compute_base_value` (adds) AND `compute_drop_cost` (drops) both multiply by `compute_positional_scarcity_factor` (PR #96 / FA P3.5). Without symmetry, backup catchers got over-boosted on the add side.
- **Playing-time gate** — FA composite score multiplied by [0.30, 1.0] based on `ytd_gp / expected_gp` ratio for hitters (or `ytd_ip / expected_ip` for pitchers). Grace period: first 30 days of season inactive. Curve: 0 GP → 0.30x, <30% → 0.60x, 30-59% → 0.85x, ≥60% → 1.0x (PR #101 / FA P3.9).
- **ROS × playing-time scaling** — Before blending, ROS counting stats are scaled by `max(0.20, ytd_gp / expected_gp)` for low-volume players (PR #110 / FA P5c). Stops IL stash phantoms (e.g. Westburg, 0 YTD GP) from riding inflated preseason ROS projections.
- **Canonical blend weights** — 0.70 ROS + 0.20 YTD + 0.10 L14 in `_blend_fa_row` (PRs #92 + #110/P5b). L14 wired from `l14_*` columns when `l14_pa >= 20` (hitters) / `l14_ip >= 5` (pitchers); below the volume gate, ROS+YTD renormalize.
- **Regression flag nudge** — `_score_fa_candidates` multiplies composite by 1.05× for `BUY_LOW`, 0.95× for `SELL_HIGH`, 1.0× for empty/None (PR #106 / FA P10).
- **Punt-category awareness** — When `ctx.h2h_strategy.get("punt", [])` lists a category OR `ctx.urgency_weights["per_cat"][cat]["win_prob"] < 0.10`, its weight is overridden to 0.05× for the duration of FA scoring (PR #110 / FA P5f). Prevents engine from rewarding contributions to conceded cats.
- **Sustainability: counting vs rate** — `compute_sustainability_score` in `waiver_wire.py` combines rate-stat signals (xwOBA-wOBA, SIERA-ERA) with counting-stat signals (HR/FB% for hitters, LOB% for pitchers). PR #108 / FA P5e. Backward compat: missing counting cols → rate-only behavior.
- **Drop candidate diversity** — `_deduplicate_and_limit` walks sorted swaps and surfaces distinct (add, drop) pairs. When all top swaps share a single drop, output collapses to 1; when alternative drops exist, output diversifies (PR #110 / FA P5a is a regression-guard test, no impl change).
- **Roster sync: mlb_id matching** — `src/live_stats.py::match_player_id` emits a WARNING when name-only match resolves to a different team than the caller's `team_abbr` (PR #104 / FA P4). Prevents Muncy-DNA-class bugs (one MLB roster row → two players with the same name).
- **News-derived suspension dates** — `parse_suspension_days()` in `news_sentiment.py` recognizes "Suspended through MM/DD" and "Suspended for N games" patterns (PR #109 / FA P5d). Wired into `expected_return_days` column on `ctx.roster` + `ctx.player_pool` so short suspensions (Valdez-class, ~1-7 days) get appropriate IL-weight curves, not the 0.0 indefinite default.
- **`_il_weight_from_status` parses day counts inline** — Status strings like `"IL10 - 3 days"` now extract the day count before falling through to the IL10/IL15/IL60 string defaults (PR #105 / FA PR8).
- **Legacy fallback is now visible** — When the new `recommend_fa_moves` engine throws, `pages/14_Free_Agents.py` falls back to the legacy `compute_add_drop_recommendations` AND shows a `st.warning(...)` banner with the exception type and message (PR #98). Prevents silent degradation masking new-engine bugs.

### Stream Analyzer (Pitcher Streaming page)
- **`team_strength` k_pct is in PERCENT (22.0); `compute_pitcher_matchup_score` and `compute_bayesian_stream_score` expect FRACTIONS (0.22)** — `score_stream_candidate` divides by 100 at the boundary. Passing percent through unconverted silently doubles the opponent-K signal.
- **`LeagueConfig.sgp_denominators` keys are UPPERCASE; `streaming.py` expects lowercase** — `stream_analyzer._lower_keys` bridges. Passing uppercase keys silently falls back to `DEFAULT_SGP_DENOMS`.
- **Locked rows stay on the board** — status LOCKED/FINAL with `actionable=False`; UI greys them out rather than hiding (transparency). Locking applies only when `target_date == today` so a stale status string can't lock tomorrow's start.
- **The page computes no scores** — every score/rank arrives from `stream_analyzer` / `recommend_streaming_moves`; AST-guarded for BOTH this page and the Lineup Optimizer Streaming tab.
- **Replay is a proxy** — `replay_stream_date` scores past dates with CURRENT projections (`proxy_caveat=True`, rendered in-UI). Exact replay needs a scheduler-written nightly snapshot table (deferred; keep the sole-writer invariant when it lands).
- **SP-focused by design** — closers/saves streaming lives on the Closer Monitor page; the engine scores probables only.

### Lineup Optimizer
- **DTD/IL exclusion** — `_il_statuses` includes `dtd` and `day-to-day`.
- **Recent form blend** — Dynamic weight: linear interpolation from 0.10 (7 games) to scope max (14+ games). Below 7 games = 0 weight.
- **Matchup data before state classification** — `yds.get_matchup()` must be called before `classify_matchup_state()`.
- **IP budget uses projected IP** — Not actual IP pitched.
- **Weekly scaling** — Counting stats divided by 26 weeks. Rate stats unchanged.
- **`sort_roster_for_display()` returns a copy** — Never modifies input DataFrame.
- **Two-start pitcher fatigue** — 0.93× rate stat quality on 2nd start (~7% ERA/WHIP decay).
- **Platoon splits** — LHB vs RHP: 7.5% wOBA advantage. RHB vs LHP: 5.8%.
- **Sigmoid urgency** — `COUNTING_STAT_K=2.0`, `RATE_STAT_K=3.0` (configurable via constants_registry).
- **DCV gate — pure SPs not in probables get volume=0.0** — SP/RP hybrids keep 0.9 (can still relieve). Pure-`P` positions (no SP/RP qualifier) ALSO gate via token-set membership; if positions=`{P}` and player isn't a probable, `volume=0`. Closes the SP-gate-was-dead-code bug.
- **Pitcher probable-starter name matching** — `_normalize_pitcher_name` (accents/suffixes/punctuation stripped).
- **Matchup multiplier resolves opponent + venue from schedule** — Walks `schedule_today`, determines home/away, passes the **home team code** as `opponent_team` to `park_factor_adjustment` (the convention used by that fn). Opposing pitcher throws + xFIP also resolved.
- **Pitcher leave-empty threshold** — `_PITCHER_EMPTY_THRESHOLD = _pitcher_median * 0.20` (was 0.05).
- **FA streaming drop-candidate is slot-aware** — Filters roster by position overlap with FA's positions.
- **Already-played games zero out DCV** — `locked_teams` set (status "in progress"/"final" or game_datetime ≤ now). `volume_factor=0.0`. UI renders `—` (not `0.00`) and a "Reason" column shows LOCKED/IL/OFF_DAY/NOT_PROBABLE.
- **Forced-start flag on lineup output** — `forced_start=True` when LP started a player but `matchup_mult < 0.70` OR `total_dcv < median × 0.5`. Renders as "START ⚠" with orange background.
- **Dynamic streaming SGP threshold for pitchers** — `_STREAM_NET_SGP_RELAXED=0.40` replaces `_STREAM_NET_SGP_MIN=0.70` for PITCHER streams when projected weekly IP < 75% of `_STREAM_IP_TARGET=54`.
- **Yahoo lineup mismatch banner** — Captures user's Yahoo `selected_position` into `yahoo_slot` BEFORE LP overwrites; orange banner if LP wants to start a BN player.
- **Pitcher matchup multiplier includes opposing offense wRC+** — League-avg 100 → 1.0, 120 → ~0.75× (capped 0.80), 80 → ~1.25× (capped 1.20). Sourced from `ctx.team_strength[opp_team]['wrc_plus']`.
- **Post-LP IP budget** — After LP assigns starters, recomputes IP from LP-selected pitchers only into `st.session_state["_post_lp_ip"]`. Header card shows "Post-LP Starters X / 54 IP".
- **SP slot reordering after LP** — Post-processes `_lp_slot_map` so highest-DCV SP-eligible pitchers land in SP slots.

### Dependencies
- **Pre-commit hook** — `scripts/pre-commit` runs ruff format + lint on staged files. Install with `python scripts/install-hooks.py`.
- **PyMC/PuLP/XGBoost optional** — `PYMC_AVAILABLE`/`PULP_AVAILABLE` (canonical in `src/lineup_optimizer.py`)/`XGBOOST_AVAILABLE` flags. CI skips with `@pytest.mark.skipif`.
- **Plotly 6.x** — Does NOT accept 8-digit hex (`#RRGGBBAA`). Use `rgba(r,g,b,a)`.
- **`ruff` on Windows** — Use `python -m ruff check .` instead of bare `ruff`.
- **`datetime.now(UTC)`** — Used everywhere; `datetime.utcnow()` deprecated.
- **Snake draft order** — `round % 2 == 0` forward, `round % 2 == 1` reverse.

## Data Sources

| Source | API | Staleness | Tier 2 fallback | Tier 3 fallback |
|--------|-----|-----------|-----------------|-----------------|
| Players (9K+) | MLB Stats API (40-man + spring training) | 7 days | — | — |
| Projections (7 systems) | FanGraphs JSON API + Marcel local | 24h | — | Marcel local |
| ROS Projections | FanGraphs (Steamer/ZiPS/DC ROS) | 4h | — | — |
| ADP | FanGraphs + FantasyPros ECR + NFBC | 24h | — | — |
| Current stats | MLB Stats API | 1h (15min during games) | — | — |
| Historical (3 years) | MLB Stats API | 30 days | — | — |
| Park factors | pybaseball | 30 days | — | Emergency dict (FG 5yr) |
| League data | Yahoo Fantasy API (optional) | 30 minutes | — | SQLite |
| News | MLB Stats API + Yahoo | 1 hour | — | — |
| Prospects | FanGraphs Board API + MiLB stats | 7 days | — | — |
| ECR (6 sources) | Multi-platform aggregation (Trimmed Borda) | 24 hours | — | Cached |
| Weather | Open-Meteo (30 stadiums, dome detection) | 2 hours | — | — |
| Team strength | pybaseball wRC+/FIP | 24 hours | statsapi fallback | — |
| Opposing pitchers | MLB Stats API + platoon splits | 2 hours | — | — |
| Statcast | pybaseball (EV, barrel, xwOBA, Stuff+, sprint speed) | 24 hours | — | — |
| Stuff+ / Batting Stats | FanGraphs (currently 403) | 24 hours | Browser-headers (still 403) | Neutral defaults; FIP/xFIP K-boost proxy active |
| Catcher framing | pybaseball Savant | 7 days | Browser-headers Savant scrape (200 OK) | `data/seed/catcher_framing_2024.json` (32 catchers) |
| Umpire tendencies | statsapi boxscore HP umpire | 24 hours | Savant `/umpire` (404 — no public endpoint) | `data/seed/umpire_tendencies_2024.json` (34 umpires) |
| Depth charts | Roster Resource scrape (often empty) | 7 days | MLB Stats API team_roster + GS≥5/SV≥5 heuristic | — |
| Game logs | statsapi `person+hydrate(stats)` (per SF-1 fix) | 1 hour | — | — |

## Structural Invariants (machine-checked)

These tests guard against regression of the cleanup work. Adding new code that violates any of these will fail CI:

| Test | Guards |
|------|--------|
| `test_no_hardcoded_categories_in_src.py` | No literal category lists in 13 src/ files; must derive from `LeagueConfig` |
| `test_app_no_hardcoded_categories.py` | Same for `app.py` |
| `test_my_team_no_hardcoded_categories.py` | Same for `pages/1_My_Team.py` |
| `test_sensitivity_categories_canonical.py` | `engine/game_theory/sensitivity.py` `CATEGORIES` from LeagueConfig |
| `test_no_merge_conflict_markers.py` | No leftover `<<<<<<<`/`=======`/`>>>>>>>` in any source file |
| `test_pages_yahoo_compliance.py` | 7 named pages don't call `load_league_rosters/standings` directly (must use `yahoo_data_service`) |
| `test_pages_format_compliance.py` | No inline `f"{x:.3f}"` near ERA/WHIP context (must use `format_stat`) |
| `test_no_direct_sqlite_connect_in_scripts.py` | Scripts **and `app.py`** use `get_connection`, never raw `sqlite3.connect` (app.py checked via AST so an aliased `import sqlite3 as _sql` can't slip past the name-based regex) |
| `test_refresh_log_status_validity.py` | All bootstrap-emitted statuses (`success`/`partial`/`cached`/`skipped`/`no_data`/`error`) round-trip without downgrade |
| `test_my_team_uses_pool.py`, `test_player_compare_uses_pool.py`, `test_trade_analyzer_uses_pool.py` | No raw `SELECT FROM season_stats/ecr_rankings/statcast_archive` in those pages |
| `test_pulp_availability_consolidation.py` | `PULP_AVAILABLE` defined ONCE (`src/lineup_optimizer.py`); other modules import |
| `test_engine_no_fallback_singletons.py` | No `_LC = ...` module-level singletons in 6 engine modules (SF-21 + BUG-010 Wave 6 expansion: 3 original + trade_evaluator + engine/portfolio/valuation + engine/game_theory/sensitivity) |
| `test_opponent_valuation_no_default_denoms.py` | No `DEFAULT_SGP_DENOMS` constant in opponent_valuation; functions require `sgp_denominators` |
| `test_pool_ytd_columns_complete.py` | Pool surfaces all `ytd_*` columns needed for weighted rate-stat aggregation |
| `test_standings_adapter_contract.py` | Standings page adapter shape preserved for `simulate_season_enhanced` + `compute_team_strength_profiles` |
| `test_no_shadow_player_rows.py` | No shadow rows in `players` (team='MLB' + fake mlb_id range); no rostered NULL mlb_ids |
| `test_no_lr_name_column_refs.py` | No `lr.name` joins; no `UPDATE league_rosters WHERE name = ?` patterns |
| `test_dynamic_park_factors_removed.py` | `_bootstrap_dynamic_park_factors` does not exist; orchestrator does not dispatch this source |
| `test_no_ip_decimal_parse.py` | No direct `float(...inningsPitched...)` parses in src/ or scripts/ — must use `_ip_outs_to_decimal()` |
| `test_no_ecr_rank_suffix_in_consensus_keys.py` | `refresh_ecr_consensus` strips `_rank` suffix from `sources` dict keys before calling `_compute_player_consensus` |
| `test_engine_signals_architectural_boundary.py` | Core trade-engine modules do NOT import from `src.engine.signals` (those are UI-helper-only) |
| `test_sigmoid_calibrator_patches_registry.py` | Sigmoid calibrator + sensitivity_analysis patch `CONSTANTS_REGISTRY` (the runtime read-path), not legacy module aliases |
| `test_mc_convergence_wired.py` | `_run_mc_overlay` calls `check_convergence` and returns `convergence_quality`/`convergence_ess`/`convergence_rhat` |
| `test_calibration_data_client_methods.py` | Every method called on `yahoo_client` from calibration code exists on `YahooFantasyClient`; calls pass `league_id` |
| `test_lineup_optimizer_26_weeks.py` | `pages/2_Line-up_Optimizer.py` uses `WEEKS_IN_SEASON = 26.0` (canonical), not 24 |
| `test_lineup_optimizer_team_map_consistent.py` | Both "Athletics" and "Oakland Athletics" map to "ATH" (canonical 2026 MLB Stats API code) in `pages/2_Line-up_Optimizer.py` |
| `test_pages_use_yds_for_rosters.py` | `pages/6_League_Standings.py` and `pages/16_Player_Compare.py` do not call `load_league_rosters` directly |
| `test_optimizer_pipeline_forwards_context.py` | `LineupOptimizerPipeline` forwards `confirmed_lineups`/`recent_form`/`team_strength` to `build_daily_dcv_table` (AST-based check) |
| `test_matchup_planner_demo_banner.py` | `pages/5_Matchup_Planner.py` `pool.head(...)` demo-roster fallback is immediately preceded by `st.warning`/`st.info`/`st.error` (within 3 lines) |
| `test_compute_fa_comparisons_negative_sgp.py` | `compute_fa_comparisons` seeds `best_fa_value` from first candidate (handles negative-SGP players) |
| `test_closer_monitor_name_normalization.py` | `closer_monitor.build_closer_grid` normalizes names (accents, suffixes, punctuation) before matching pool |
| `test_playoff_sim_uses_league_config.py` | `playoff_sim` uses `season_weeks=26` and `_PLAYOFF_SPOTS=4` (FourzynBurn canonical) |
| `test_yahoo_reconnect_in_yahoo_api.py` | `try_reconnect_yahoo` lives in `src/yahoo_api`; `app.py` keeps a thin wrapper; no top-level streamlit import in yahoo_api |
| `test_streamlit_security_settings.py` | `.streamlit/config.toml` has `enableXsrfProtection=true` (CORS default-true also acceptable) |
| `test_no_lc_singletons_in_optimizer.py` | No module-level `_LC = ...` in 8 optimizer modules (SF-21 + BUG-010 Wave 6 expansion) |
| `test_no_lc_singletons_in_strategy.py` | No module-level `_LC = ...` in 8 strategy/UI modules (SF-21 + BUG-010 Wave 6 expansion) |
| `test_survival_calibrator_skips_on_missing_adp.py` | `survival_calibrator._build_prediction_pairs` skips with WARNING when no ADP column (no `actual_pick` → ADP data leakage) |
| `test_no_unguarded_update_refresh_log.py` | Bootstrap phases use `update_refresh_log_auto` (with row-count) not plain `update_refresh_log(..., "success")` (SF-54 / INFRA-F6) |
| `test_umpire_phase_has_timeout.py` | `_bootstrap_umpire_tendencies` Tier 1 uses a wall-clock budget (`time.monotonic()` + elapsed/timeout sentinel) (SF-55 / INFRA-F3) |
| `test_data_freshness_reads_refresh_log.py` | `DataFreshnessTracker` hydrates `_sources` from `refresh_log` on `__init__`; non-success rows skipped (SF-56 / INFRA-F4) |
| `test_no_oak_in_source_modules.py` | `src/player_databank.py`, `src/data_2026.py`, `src/ecr.py`, `src/prospect_engine.py` emit "ATH" not "OAK" (SF-57 / D1A-008 follow-up) |
| `test_wave9_level_column_migration.py` | `players.level` column exists after `init_db()` (SF-78 / INFRA-F5) |
| `test_wave9_minor_league_fetch.py` | `fetch_minor_league_players` caps at `top_n_per_team`, sets `level`, handles failures (SF-79) |
| `test_wave9_bootstrap_phase.py` | `_bootstrap_minor_league_rosters` writes rows with `level` + updates `refresh_log` (SF-80) |
| `test_wave9_pool_includes_level.py` | `load_player_pool()` exposes `level` column from all 3 SELECT paths (SF-81) |
| `test_wave9_free_agents_level_filter.py` | `pages/14_Free_Agents.py` has level selectbox defaulting to "MLB only" (SF-82) |
| `test_wave9_player_compare_level_filter.py` | `pages/16_Player_Compare.py` has level selectbox (SF-83) |
| `test_bootstrap_log_isolation.py` | `src/data_bootstrap.py` skips the RotatingFileHandler when running under pytest — prevents test mocks (e.g. `RuntimeError("DB out")`) from polluting `data/logs/bootstrap.log` (SFH L) |
| `test_bootstrap_refresh_log_on_error.py` | Every outermost `except` in `_bootstrap_*` functions in `src/data_bootstrap.py` either calls `update_refresh_log[_auto]`, re-raises, or catches ImportError-only. AST-checked; ≥25-phase sentinel (SFH D companion to `test_no_unguarded_update_refresh_log.py`) |
| `test_sfh_m3_partial_cached_reads.py` | `check_staleness` force-refreshes only `error`/`unknown`; `partial`/`no_data`/`cached`/`skipped` honor TTL. `DataFreshnessTracker` hydrates from `partial`/`cached`/`skipped` rows (SFH M-3) |
| `test_players_name_lookup_case_insensitive.py` | Every `WHERE name = ?` query against the `players` table in `src/` uses `COLLATE NOCASE`. Prevents Muncy-DNA dedup misses from case-variant duplicates (SFH LOW-2) |
| `test_refresh_log_resilience.py` | `_try_write_refresh_log` retries 3× on `OperationalError("database is locked")`; `_run_with_timeout(source=)` writes "timeout" to refresh_log on budget exceeded; `_reconcile_results_to_refresh_log` overwrites stale-success rows for "Error:"/"Timeout" results (SFH H2 + H3) |
| `test_pvb_splits_commit_per_batter.py` | `_bootstrap_pvb_splits` calls `conn.commit()` INSIDE the outer batter loop, NOT outside — required to release the write lock between Statcast fetches so parallel writers don't starve (SFH M1, root-cause fix for #69's band-aid) |
| `test_pybaseball_batting_stats_no_pos_kwarg.py` | No `batting_stats(..., pos=...)` calls in `src/data_bootstrap.py` — pybaseball 2.2.7 rejects `pos=` with TypeError (SFH M2) |
| `test_yahoo_player_key_column.py` | `league_rosters` has `yahoo_player_key` column after `init_db()`; `upsert_league_roster_entry` writes it; TWP-on-different-teams persists both rows distinguished by yahoo_player_key; UNIQUE(team_name, player_id) constraint preserved (SFH M4) |
| `test_roster_sync_partial_fetch_guard.py` | `YahooFantasyClient.sync_to_db` gates the destructive `clear_league_rosters()` + reinsert on `database.roster_fetch_is_complete(rosters_df)`: a non-empty but INCOMPLETE Yahoo fetch (fewer teams than the cache, or >20% roster-row loss vs `_ROSTER_SYNC_MIN_ROW_FRACTION=0.8`) is skipped (refresh_log `partial`) and the cached rosters are KEPT, never wiped. Empty cache (first load) accepts any non-empty fetch; a complete fetch still clears+reinserts so drops propagate. Catches the 2026-06-02 live incident where a partial fetch right after a Yahoo-token re-paste replaced a full league with an empty user roster ("No players on your roster yet"). The in-process scheduler runs the same sync, so the guard protects all 12 users |
| `test_readonly_member_data_status.py` | `YahooDataService.connection_status()` reports DATA availability under MULTI_USER (read-only member sessions never hold a Yahoo client, so `is_connected()` is always False): `"server"` when any source is fresh in cache, `"warming"` otherwise; v1 returns `"connected"`/`"offline"` from the live session (byte-for-byte). `pages/1_My_Team.py` reads the weekly matchup via `get_matchup()` (SQLite-cache fallback), NOT gated behind `is_connected()`. Catches the 2026-06-03 read-only-member gating bug where every member saw "Yahoo Offline" + a blank matchup despite live scheduler-served data. (Deeper follow-up tracked separately: the scheduler only caches the token-owner's matchup, so per-team weekly matchup for the other 11 members still needs a new bootstrap phase.) |
| `test_draft_pick_name_resolution.py` | `resolve_player_names_by_keys` batch-fetches Yahoo player names (25 per call); `get_draft_results` uses it for entries where yfpy didn't expand the player resource; still-unresolved keys fall back to legacy `"Player {key}"` placeholder (SFH L8) |
| `test_oauth_decoupling.py` | `pages/2_Line-up_Optimizer.py`'s optimize click handler must not call YahooDataService methods with `force_refresh=True`; the "Refresh Yahoo Data" button must be present and invoke `force_refresh_all()`; `_get_cached` must time out at 15s when Yahoo hangs. T1.21 / Batch G of v2. |
| `test_app_main_auth_gate.py` | `app.py main()`: flag-off fast path calls `render_single_user_app()` and returns (v1 unchanged); flag-on path runs `require_auth()` (after `init_db()`, before `st.navigation(`) then `st.navigation(build_pages(...)).run()`. AppTest smoke: flag-on + no session ⇒ login renders and `bootstrap_complete` never set (v2 Plan 3) |
| `test_pages_have_auth_guard.py` | Every interactive `pages/*.py` (those calling `inject_custom_css()`, excluding underscore-prefixed pages) imports + calls `require_auth()` after `inject_custom_css()`. Streamlit runs each page independently, so the gate must be per-page |
| `test_pages_use_viewer_team_resolver.py` + `test_viewer_team_identity.py` | The 8 personalized pages (My Team, Lineup, Matchup, Standings, Punt, Trade Analyzer, Trade Finder, Free Agents) resolve "my team" via `src.auth.resolve_viewer_team_name(rosters)` and contain NO direct `is_user_team` reference. The resolver returns the logged-in session user's admin-assigned team under MULTI_USER, falling back to the legacy `league_teams.is_user_team` flag in v1 (byte-for-byte). Closes the 2026-06-01 launch-blocker where every leaguemate saw the single `is_user_team` team (Team Hickey) instead of their own assigned team — the auth-aware resolver existed (`YahooDataService._get_user_team_name`) but no page called it (engine built, wires never connected; the per-resolver unit test passed so CI stayed green) |
| `test_member_write_safety.py` + `test_refresh_buttons_admin_gated.py` | `src.auth.viewer_can_write()` is the sole gate for member-triggered DB writes: True in single-user v1, else only admins under MULTI_USER (the scheduler is the sole SQLite writer; members are read-only). The 4 member-facing refresh/sync buttons (My Team "Refresh Stats" + "Sync Yahoo", Lineup "Refresh Yahoo Data", Leaders "Refresh Data") are gated by it. Also locks: `bump_activity()` swallows DB-lock errors on the 60s heartbeat; `enter_view_as()` refuses admin targets (audit-log integrity); `resolve_viewer_team_name` empty-team guard returns None (never the global flag) for a teamless member under MULTI_USER. 2026-06-01 pre-launch audit |
| `test_mobile_nav.py` | `src/ui_shared.py` hides the Streamlit header bar ONLY inside `@media (min-width: 768px)` so phone users keep the sidebar/menu toggle — an unconditional hide stranded mobile members on the landing page with no navigation. 2026-06-01 pre-launch audit (mobile) |
| `test_admin_console_guarded.py` | `pages/_admin_console.py` calls `require_admin()` before any `approve_user` action; AppTest smoke confirms non-admins are hard-stopped |
| `test_auth_backcompat.py` | `MULTI_USER` off ⇒ `require_auth()` is a no-op and `_get_user_team_name` uses the `league_teams.is_user_team=1` query (v1 behavior preserved) |
| `test_pages_have_feedback_and_usage.py` | Every interactive `pages/*.py` (those calling `inject_custom_css()`, excluding underscore-prefixed pages) imports `log_page_view` + `render_feedback_widget`; `log_page_view()` is called after `require_auth()`; `render_feedback_widget()` is called on the page. Per-page because Streamlit re-runs each page independently (v2 Plan 2) |
| `test_feedback_usage_tables.py` | `init_db()` creates the `feedback` + `usage_events` tables (additive, idempotent). feedback carries `app_version` + `data_state` (JSON refresh_log snapshot); usage_events dedupes per `(page, action)` per `session_id` (v2 Plan 2) |
| `test_feedback_usage_backcompat.py` | `MULTI_USER` off ⇒ `log_page_view()` and `render_feedback_widget()` are no-ops (zero DB writes, no popover) — v1 byte-for-byte (v2 Plan 2) |
| `test_admin_tables.py` | `init_db()` creates the 5 additive Plan 3 tables (`feature_flags`, `audit_log`, `app_settings`, `sessions`, `page_visits`) idempotently; all start empty/inert until written (v2 Plan 3) |
| `test_streamlit_min_version.py` | `requirements.txt` pins `streamlit>=1.40` — covers `st.navigation` (≥1.36) + `st.fragment(run_every=...)` (≥1.37) (v2 Plan 3) |
| `test_audit.py` | `src/audit.log_action()` is MULTI_USER-gated (off ⇒ no write); `list_audit()` joins usernames + filters by action; append-only (v2 Plan 3) |
| `test_app_settings.py` | `get_broadcast()`/`get_maintenance()` return `{"enabled","message"}` (default disabled); setters are flag-gated and each writes exactly one audit row (v2 Plan 3) |
| `test_feature_flags.py` | `is_page_enabled()` honors "absence = enabled"; `set_page_flag`/`require_page_enabled` flag-gated; flag off ⇒ every page always enabled (v2 Plan 3) |
| `test_nav.py` | `PAGE_REGISTRY` matches the 13 non-underscore season pages on disk; `build_pages(user, draft_page)` adds the Admin group only for `user["is_admin"]`; `filter_enabled_pages` is pure (v2 Plan 3) |
| `test_view_as.py` | `enter_view_as`/`exit_view_as`/`is_viewing_as` stash the real admin and swap `current_user()` to the target; flag-gated (off ⇒ no-op) (v2 Plan 3) |
| `test_usage_sessions_dwell.py` | `usage.log_page_view` opens a `sessions` row + `page_visits` dwell row; `bump_activity()` refreshes `last_activity_at`; idle visits lazily closed at query time (v2 Plan 3) |
| `test_usage_analytics.py` | analytics readers (`dau_series`, `most_used_pages`, `per_user_activity`, `last_seen_summary`, `session_timeline`, `page_dwell_summary`, `usage_csv`) return empty/inert when the flag is off (v2 Plan 3) |
| `test_feedback_csv.py` | `feedback.feedback_csv()` serializes the feedback inbox to CSV; header row always present (flag-off ⇒ header-only) (v2 Plan 3) |
| `test_admin_analytics_page.py` | `pages/_admin_analytics.py` calls `require_admin()` before rendering; AppTest smoke confirms non-admins are hard-stopped (v2 Plan 3) |
| `test_admin_controls_page.py` | `pages/_admin_controls.py` calls `require_admin()` before any flag/broadcast/maintenance/view-as mutation; each mutation writes an audit row (v2 Plan 3) |
| `test_pages_guard_set_page_config.py` | Every season `pages/*.py` guards its `st.set_page_config` with `if not multi_user_enabled():` so `app.py` owns the single config call under `st.navigation()` (v2 Plan 3) |
| `test_admin_pages_flag_enforced.py` | Each season page calls `require_page_enabled("page:<stem>")` after `require_auth()`; the 3 `_`-prefixed admin pages are exempt (v2 Plan 3) |
| `test_admin_backcompat.py` | `MULTI_USER` off ⇒ the whole Plan 3 admin surface is inert: a connection spy proves `is_page_enabled`/`log_action`/`set_setting`/`set_page_flag`/`log_page_view`/`bump_activity` never open a DB connection, the 5 tables stay empty, and `app.py`'s `main()` early-returns before `st.navigation()` — v1 byte-for-byte (v2 Plan 3) |
| `test_network_guard.py` | The `tests/conftest.py` network guard blocks real outbound sockets: `socket.connect`/`connect_ex` to a non-loopback host raises `NetworkBlockedError` by default; loopback passes through as a normal `OSError`; `@pytest.mark.allow_network` restores the real connect within a test. Load-bearing fix for the Windows full-suite hang (2026-05-29) |
| `test_phase1_is_grade_authority.py` | Phase 1 weighted SGP is the AUTHORITY for `grade`/`verdict`/`confidence_pct`. `_run_mc_overlay` MUST rename its `grade`/`verdict`/`confidence_pct` keys → `mc_grade`/`mc_verdict`/`mc_confidence_pct` BEFORE returning so `result.update(mc_result)` in `evaluate_trade` cannot clobber the Phase 1 grade. Catches the 2026-05-23 validation regression where `grade=B` was shown alongside `surplus_sgp=-1.58` (Bug A). |
| `test_reshuffle_warning_promote_or_demote.py` | The LP reshuffle warning in `evaluate_trade` fires whenever `(promoted OR demoted) AND abs(total_surplus) > 0.01` — NOT only on `demoted`. Three message variants: promote+demote, demote-only, promote-only (the promote-only variant tells the user they may capture most of the surplus by setting their lineup without making the trade). `reshuffle_pct` is always populated. Catches the 2026-05-23 validation regression where a +3.97 SGP reshuffle (251% of surplus) was silently hidden (Bug B). |
| `test_opponent_market_value_scale.py` | `estimate_opponent_valuations` routes AVG/OBP through the rate-stat marginal blender parallel to ERA/WHIP (volume-weighted blending with `_DEFAULT_TEAM_AB`/`_DEFAULT_TEAM_PA` fallbacks); per-category contribution clamped to `MAX_PER_CAT_SGP=3.0` as defense-in-depth; `get_player_projections_from_pool` surfaces `ab`/`pa`/`ip` so the blender has the volumes it needs. Catches the 2026-05-23 validation regression where a below-average hitter (Swanson, .215/.288) got `market_price=+54.84` from teams already best in AVG/OBP (Bug C). |
| `test_ip_floor_penalty.py` | Feature 1 (2026-05-23): IP-floor soft penalty per report Section B.6. Constants locked at `_IP_FLOOR_PER_WEEK=20.0` (Yahoo H2H Cats default) and `_IP_FLOOR_KAPPA=0.05` (50% shortfall → 5.0 SGP). Quadratic formula `kappa * (20 - IP_w)^2`. Delta semantics: `max(0, after_penalty - before_penalty)` so trades that don't worsen the IP situation contribute 0; trades that IMPROVE it also contribute 0 (no negative bonus). Risk flag fires when post-trade weekly IP < floor, with text adapted per trade-caused-vs-pre-existing. |
| `test_weekly_h2h_matrix.py` | Feature 2 (2026-05-23): 26-week × 12-cat win-probability matrix per report Section B.5. `p_{w,c} = Phi((mu_H - mu_O) / sqrt(sigma_H^2 + sigma_O^2))`. Per-category CV calibration (`_DEFAULT_CV`: AVG=5%, HR=20%, SB=25%, W/L=30%, etc.). Inverse cats (L/ERA/WHIP) flipped. Volume-weighted rate stats. Min sigma floor 0.01. Unknown opponent → neutral 0.5. Opt-in via `enable_weekly_matrix=True` + `weekly_schedule` + `league_rosters`. `compute_trade_weekly_delta` returns before/after/delta + per-week expected-cat-wins summary. |
| `test_playoff_championship_sim.py` | Feature 3 (2026-05-23): the report's PRIMARY objective per Section B.10 + Q(a). `simulate_playoff_outcomes` returns `playoff_prob` + `champ_prob` + `playoff_seed_distribution` + `mean_regular_season_wins`. `simulate_trade_playoff_delta` returns before/after/`delta_playoff_prob`/`delta_champ_prob` with paired-MC discipline (same seed both arms). Top-4 bracket: 1v4 + 2v3 round 1, re-seeded final. Bradley-Terry matchup probs bounded [0.05, 0.95]. Hickey-centric: user's per-week probs from `weekly_matrix`, opponents' season-end wins approximated via Binomial(N, p_avg). Invariants: `champ_prob ≤ playoff_prob`, seed distribution sums to playoff_prob, identical rosters → exact-zero delta. `_PLAYOFF_SPOTS=4` constant locked. |
| `test_no_ghost_team_rank.py` | Bug D (2026-05-23): `build_standings_totals(standings, valid_teams=...)` filters ghost rows (teams in standings but not in league_rosters, like a renamed/abandoned "Twigs"). `evaluate_trade` always passes the real-team list (from `league_rosters` param if provided, else queried from the DB) so risk flags never show "Ranked 13th" in a 12-team league. |
| `test_grade_range_consistency.py` | Bug E (2026-05-23): post-Bug-A, `result["grade"] == result["grade_range"]["grade"]` is an invariant — both derive from Phase 1 surplus_sgp. Also locks `low ≤ center ≤ high` ordering of grade_range and `confidence ∈ {high, medium, low}`. Fires if MC-override behavior is ever re-introduced. |
| Bug F regression in `test_ip_floor_penalty.py` (4 added tests) | `_compute_weekly_ip(starter_ids, pool, weeks_remaining=...)` accepts `weeks_remaining` (forward-looking horizon), NOT `season_weeks`. When `ytd_ip` column exists, computes `max(0, ip - ytd_ip) / weeks_remaining` so mid-season scaling reflects ROS, not full-season divisor. Floors negative ROS at 0 (over-performers). Falls back to `ip / weeks_remaining` when `ytd_ip` is missing (preseason mode). Detail dict surfaces `weeks_remaining` for UI transparency. |
| `test_trade_analyzer_features_wired.py` | UI wiring (2026-05-23): `pages/11_Trade_Analyzer.py` MUST call `evaluate_trade(... enable_weekly_matrix=..., enable_playoff_sim=..., weekly_schedule=..., league_rosters=..., current_wins=...)`. Page must source `weekly_schedule` from `load_league_schedule()` (Yahoo cache). Page must render `delta_playoff_prob`, `delta_champ_prob`, `ip_floor_detail`/`ip_floor_penalty`, and `weekly_matrix`. Catches drop-on-the-floor regressions where the engine still produces an output but the UI silently stops rendering it. |
| `test_plan4_dockerfile_railway.py` | The three Railway deploy artifacts carry their load-bearing invariants: `Dockerfile` pins `python:3.12-slim` (CI parity) + installs `--no-deps` yfpy/streamlit-oauth + binds `$PORT`/headless/`0.0.0.0`; `.dockerignore` excludes `.venv`/`.git`/`__pycache__`/`data/draft_tool.db`/`data/yahoo_token.json` but KEEPS `data/seed/` (Tier 3 fallbacks); `railway.toml` uses `builder = "DOCKERFILE"`, healthcheck `/_stcore/health`, and `numReplicas = 1` (single-writer + SQLite) (v2 Plan 4) |
| `test_plan4_yahoo_token_paste.py` | `src/yahoo_api.save_yahoo_token_json(text)` validates pasted JSON (object + non-empty `refresh_token`) and writes to `_AUTH_DIR/yahoo_token.json`; rejects non-JSON / missing-token / non-object with NO file written. `pages/_admin_controls.py` calls `require_admin()` + the helper, audit-logs only the action `"yahoo_token_update"` (never the token text — AST-checked), and renders a "Yahoo token" subheader for admins (AppTest smoke) (v2 Plan 4) |
| `test_plan4_scheduler_wiring.py` | `app.py` imports `start_background_refresh`; `main()` calls it AFTER the flag-off early return (v1 never spawns a thread — AST-checked) and `start_background_refresh()` is idempotent (exactly one `heater-refresh` thread across repeat calls) (v2 Plan 4) |
| `test_plan4_bootstrap_suppression.py` | `render_single_user_app()` branches on `multi_user_enabled()`; the flag-on branch calls `_render_multiuser_home_gate()` and does NOT call `render_splash_screen` (no per-session bootstrap under MULTI_USER). `_latest_successful_refresh()` returns None when the refresh-log snapshot is empty or success-free, and the most-recent success timestamp otherwise (the single authoritative "data warm yet?" signal) (v2 Plan 4) |
| `test_plan4_backcompat.py` | `MULTI_USER` off ⇒ `main()` renders the single-user app exactly once and starts the scheduler ZERO times (`require_auth` never reached); flag-off Home keeps `render_splash_screen` (v1 byte-for-byte) (v2 Plan 4) |

**2026-06-10 Pitcher Streaming Analyzer guards (14th page + stream engine):**

| Test | Guards |
|------|--------|
| `test_stream_analyzer_constants_registered.py` | The six `stream_score_w_*` component weights (sum == 1.0, convex blend) + four `stream_risk_*` thresholds live in `CONSTANTS_REGISTRY` with value/citation/bounds/sensitivity |
| `test_stream_analyzer_adds_budget_canonical.py` | Weekly add budget is canonical: `league_rules.WEEKLY_TRANSACTION_LIMIT == 10`; `streaming.WEEKLY_ADDS_BUDGET` mirrors it (was a stale literal 7 — budget-exhausted guard + last-add penalty fired three adds early); `rank_streaming_candidates` honors the 10-add budget + `weekly_adds_budget` override; `stream_analyzer` never imports the legacy alias |
| `test_stream_analyzer_score_components.py` | `score_stream_candidate` returns a 0-100 score decomposed into exactly 6 components each in [-1, 1]; weights read from the registry AT CALL TIME (patch ⇒ score changes); matchup monotonicity (weak offense/pitcher park > elite offense/hitter park); missing lineup/form data ⇒ neutral 0.0, never NaN; each risk flag (HIGH_WHIP/SHORT_LEASH/ELITE_OFFENSE/HITTER_PARK/WIND_OUT/LOW_CONFIDENCE) fires on its registry threshold; WIND_OUT suppressed in domes; L14 form gated at 5 IP |
| `test_stream_analyzer_inverse_stats.py` | ERA/WHIP/L enter the Stream Score through `compute_streaming_value` (canonical signs): the era=9.0/whip=2.0 unknown-pitcher line scores NEGATIVE sgp component (the FA PR #99 lesson); no module-level denominator table; no hardcoded uppercase category collections |
| `test_stream_analyzer_locked_games.py` | `build_stream_board`: in-progress/final games (today only) stay VISIBLE but `actionable=False` with status LOCKED/FINAL; future dates never lock and carry the two_start confidence tier (HIGH/MED/LOW); other teams' rostered pitchers excluded; `include_rostered` admits only the user's SPs; two-start weeks score weekly SGP volume; board sorted by score |
| `test_stream_analyzer_history.py` | `get_pitcher_vs_team_history` parses IP via `_ip_outs_to_decimal` ("6.1" = 6⅓), filters by opponent, returns empty frame (no raise) on statsapi failure; `aggregate_pitcher_history` uses weighted ERA=ER*9/IP, WHIP=(BB+H)/IP — never per-game averages; `compute_lineup_exposure` shrinks PvB samples to 60-PA stabilization and returns None with nothing to reason about |
| `test_stream_week_planner.py` | `build_week_plan` never plans more adds than `ctx.adds_remaining_this_week`; summary reports pacing against `ip_tracker.WEEKLY_TARGET` + `MIN_IP` (never literals); `under_floor` is None when the roster baseline is unknown (never guessed); a pitcher with two window starts is planned at most once; rostered pitchers never planned |
| `test_stream_replay.py` | `replay_stream_date` always carries `proxy_caveat=True` (no point-in-time projections stored — replay uses current data) and the page renders the caveat; actuals aggregated with weighted rates; empty date ⇒ empty results, no raise |
| `test_stream_page_no_inline_scoring.py` | NO score arithmetic in pages: `pages/4_Pitcher_Streaming.py` AND `pages/2_Line-up_Optimizer.py` (whose Streaming-tab `K*1.5 + W*0.3 − penalties` formula was DRY'd into the engine) contain zero `*score*`-named assignments built from BinOp/AugAssign arithmetic; both delegate to `build_stream_board` |
| `test_stream_matchup_impact.py` | `compute_matchup_impact` projects the LIVE matchup with vs without a streamed start via the canonical `estimate_h2h_win_probability` (same engine as the Lineup Optimizer): inverse cats move the right way (gem start RAISES the ERA win prob; likely loss LOWERS the L win prob), a no-op/zero-IP line yields exact-zero deltas (paired seed; no decision charged without innings), missing matchup totals ⇒ None (never guessed), `stream_loss_decision_share` registered; the page wires it |
| `test_lineup_optimizer_rate_modes_rerun.py` | The 2026-06-10 "Refresh Yahoo Data" NameError: `_dcv_rate_modes` is persisted to session state at optimize time and read back in the results render branch (which re-executes on EVERY rerun); structural scan proves no other optimize-click local is consumed unbound in the render branch |

**2026-06-05 launch-hardening guards (post-go-live fixes):**

| Test | Guards |
|------|--------|
| `test_yahoo_token_persistence.py` | Item #2: `YahooFantasyClient.persist_current_token()` reads the LIVE yfpy `oauth` object (post-refresh), writes ATOMICALLY (temp + `os.replace`), refuses an incomplete token (never clobbers a good file), and logs WARNING on failure; `authenticate()`/`refresh_token()`/the scheduler all call it. yfpy uses `store_file=False` so persistence is HEATER's job — the old write-back read a stale dict, wrote non-atomically, and debug-swallowed failures, so the token died ~1h after each paste (refresh storm → Yahoo rate-limit). Module fn `_write_token_file` is the atomic writer (`95912d4`) |
| `test_readonly_member_data_status.py` (connection_status honesty, extended 2026-06-05) | `connection_status()` reports `"server"` ("Live (via server)") only when a CORE real-time source (rosters/standings/matchup/free_agents/transactions) is fresh — NOT the 24h-TTL settings/schedule, which coast for a day and masked a dead scheduler/token (`95912d4`) |
| `test_pages_use_viewer_team_resolver.py` (frame guard, extended 2026-06-05) | Every personalized page calls `resolve_viewer_team_name()` WITH a frame (no zero-arg call) so the emoji/whitespace reconciliation runs. League Standings' no-frame call returned the env-seeded "Team Hickey", which never matched the Yahoo emoji-prefixed name → "Your Position: Team not found in standings" + 9 other broken `== user_team` comparisons (`b7f0567`) |
| `test_build_optimizer_context_user_roster_filter.py` (F5, extended 2026-06-05) | `build_optimizer_context` refuses a multi-team rosters fallback when `user_team_name` is falsy + `roster=None` (leaves roster empty + WARNs) instead of silently using all 12 teams → league-wide nonsense recs. Single-team fetched rosters (unambiguous) are still used |
| `test_standings_close_battles.py` | `standings_utils.close_battle_categories()` skips matchup categories with a missing/empty/whitespace name so the standings banner never renders "Close battles: , , ." (2026-06-05 cosmetic); tolerates malformed rows without raising |
| `test_combustion_lock.py` | Locks the Combustion Index design system (2026-06-09 + finale 2026-06-10, 25 tests): `THEME["primary"] == "#ff6d00"`; Google-Fonts @import loads Archivo + Inter + IBM Plex Mono (Figtree/Bebas banned module-wide); `--font-*` + `--fp-*` tokens emitted; `.sec-head` uses Archivo, never all-caps; navy sidebar rail with desktop-scoped width that applies ONLY to `[aria-expanded="true"]` (forcing width on the collapsed shell left a blank 100px strip) AND the desktop header-hide conditioned on `body:has(sidebar[aria-expanded="true"])` so the » expand control returns when the rail is collapsed (2026-06-10 stranding fix — owner clicked « in prod and had no way back); block container full-width; no emoji in `inject_custom_css()` output or `PAGE_ICONS`; root body-font lock + BaseWeb widget font-lock; `.heater-link`/`.clickable` orange + underline scoped to main. Finale locks: motion tokens; `prefers-reduced-motion` block; `::selection`; `:focus-visible`; grain data-URI on `.stApp`; NO slideUp entrance on the six persistent-card classes; `stDialog` chrome at `min(92vw, 1100px)`; navy tooltip/toast portal chrome; `render_empty_state` emoji-free; `html_cols` cells render unescaped while others stay escaped |
| `test_no_offpalette_hex_in_pages.py` | pages/, `app.py`, and `src/ui_analytics_badge.py` contain ZERO Material/Tailwind-era hex literals (`#9E9E9E`, `#4CAF50`, `#ef4444`, `#84cc16`, `#6b7280`, `#9ca3af`, `#FF9800`, `#f97316`, `#22c55e`, `#666(666)`, `#2c2f36`, `#f5f5f5`, `#e8f5e9`, `#457b9d`, `#9c27b0`, `#999`) — colors come from THEME/`--fp-*` tokens. My Team's news-source brand colors (ESPN/Yahoo/RotoWire/MLB) + `TEAM_BRAND` are intentional and not banned (Combustion Finale, 2026-06-10) |

**Spec-completion guards (Items 1-8, 2026-05-25 — finish the Enhanced Trade Engine report):**

| Test | Guards |
|------|--------|
| `test_trade_secondary_diagnostics.py` | Items 1/3/4/7. `evaluate_trade` surfaces `delta_g_score`+`gscore_detail` (Rosenof G-score delta, report B.5/C.2), `delta_vorp_prp`+`vorp_detail` (league-wide PRP/VORP secondary check, report B.8), `drl_replacement_chain` (named bench/FA fills, report B.7 pseudocode output — pickup entry on 2-for-1, empty on equal trade), and `specialist_cap_penalty`+`specialist_cap_detail` (report H.2 — a single received STARTER's per-category credit capped at `_SPECIALIST_CAP_FRACTION=0.25 × (num_teams−1)`; benched receives skip the cap). Also locks the regression where `picked_up_ids`/`dropped_ids` were branch-local (NameError on equal/roster-grow trades). None of these feed the grade — Phase 1 surplus remains the grade authority. |
| `test_cara_lambda_sweep.py` | Item 6. `simulate_trade_playoff_delta` returns `cara_utility_sweep` keyed by `_LAMBDA_SWEEP=(0.05, 0.15, 0.30)` (report H.10/C.7). Central λ=0.15 matches the standalone `cara_utility`. Utility is non-increasing in λ when `Var[Δchamp] ≥ 0`. |
| `test_kalman_variance_wiring.py` | Item 5. `build_team_kalman_variances(roster_ids, pool, cats, season_weeks, config)` derives per-team per-category true-talent variance (per-week-mean units) from YTD sample sizes (`ytd_pa`/`ytd_ip`) via a stabilization-shrinkage model (`_KALMAN_STABILIZE_PA=200`, `_KALMAN_STABILIZE_IP=70`, `_KALMAN_TALENT_CV=0.40`). Under-sampled rosters carry MORE variance than fully-sampled veterans. `compute_weekly_matrix(enable_kalman=True)` threads it into `_category_win_prob`'s `kalman_var_h/o`, shrinking favorable win-probs toward 0.5; method tag includes `kalman`. Turns the design-Q3 plumbing into a real caller. |
| `test_skellam_win_prob.py` | Item 8. `_category_win_prob_skellam(mu_h, mu_o, inverse)` models each side's weekly count as Poisson and uses the Skellam distribution (exact difference of two Poissons) for the head-to-head win prob — capturing the right-skew of low-count cats (SB/SV) the Gaussian approx misses (report C.5/H.9). `_category_win_prob(..., variance_model="skellam", counting=True)` dispatches counting cats to Skellam while RATE cats keep the normal model. Equal means → ~0.5; inverse flips; mu=0 safe; `compute_weekly_matrix(variance_model="skellam")` method tag = `skellam`. |
| `test_historical_backtest.py` | Item 2. Section G error metrics: `rank_mae` (target <1.5), `cat_win_rmse` (<8), `trade_prediction_spearman` (>0.4), `projection_mae` (HR<5, AVG<0.020). `run_historical_backtest(records_df)` is the data-agnostic ingestion path — computes whichever metrics the supplied columns support and grades each against `_SECTION_G_TARGETS`. No historical data ships with the repo (needs prior-season week-by-week logs), so it operates on caller-supplied records; near-perfect synthetic predictions pass every target. Partial columns + empty frame handled without crash. |
| `test_dcv_hitter_games_fraction.py` | DCV-A1-012 (last open LOW finding from the 2026-05-14 DCV-engine audit, closed 2026-05-25). `_HITTER_GAMES_PER_SEASON=145.0` (cited FanGraphs games-played medians) is the hitter season divisor in BOTH the rate-stat daily fraction and the counting-stat per-game path; hitters scale by 162/145 (~+10%) vs the prior full-season divisor, while pitchers keep `_FULL_SEASON_GAMES=162.0` (their appearance frequency is carried by the role daily fraction). All 28 DCV audit findings (2 HIGH + 14 MED + 12 LOW) are now resolved. |

**FA Engine Overhaul guards (PRs #89-#110, 2026-05-20 → 2026-05-21):**

| Test | Guards |
|------|--------|
| `test_fa_il_stash_unified_protection.py` | FA page UI-layer IL filter applies to BOTH top "Recommended Adds/Drops" and bottom "Recommended Drops" tables. Imports + uses `get_il_stash_names()` not the static set (FA PR #89 + PR #102) |
| `test_roster_category_totals_il_weighted.py` | `_roster_category_totals` no longer zeroes IL players — they keep their projection × `_il_weight_from_status` weight (FA PR #90) |
| `test_fa_page_uses_recommend_fa_moves.py` | FA page imports + calls `recommend_fa_moves`, not legacy `compute_add_drop_recommendations` (FA PR #91) |
| `test_fa_blending_ros_ytd.py` | `_blend_fa_row` uses canonical 0.70/0.20 ROS/YTD blend with sample-size gating (FA PR #92) |
| `test_sustainability_calibrated.py` | `compute_sustainability_score` is a calibrated sigmoid, not a 5-bucket step function (FA PR #93) |
| `test_fa_composite_multiplicative.py` | Composite formula is purely multiplicative (urgency applied per-category inside `_compute_base_value`, not additively outside) (FA PR #94) |
| `test_il_weight_return_date_aware.py` + `test_roster_category_totals_return_date_aware.py` | `_il_weight_from_status(status, expected_return_days)` uses the piecewise-linear curve when ESPN/Yahoo provides return-date data (FA PR #95) |
| `test_drop_cost_positional_scarcity.py` | `compute_drop_cost(player_id, ..., replacement_levels=...)` multiplies base_cost by symmetric scarcity factor. `compute_positional_scarcity_factor` importable from `src.valuation` (FA PR #96) |
| `test_fa_recommender_positional_scarcity.py` | `_compute_base_value` multiplies by positional scarcity factor (top-2 catcher with raw SGP X outranks 25th OF with same raw SGP) (FA PR #91/PR3 + PR #96) |
| `test_fa_roster_construction_guard.py` + `test_fa_position_cap_at_boundary.py` | `_passes_roster_construction_guard` blocks: 3rd catcher (at-cap position), drops below 10 active hitters or 8 active pitchers (FourzynBurn starting-lineup minimums). Boundary semantics: block when count `> cap`, NOT `>= cap` (FA PR #97 + PR #100) |
| `test_build_optimizer_context_user_roster_filter.py` + `test_fa_page_passes_user_roster.py` | `build_optimizer_context` filters multi-team rosters by `user_team_name`; FA page passes `user_roster` not `rosters` (FA PR #98 / FA P3.6) |
| `test_build_optimizer_context_level_filter.py` | `level_filter` param defaults to `"MLB only"`; minor leaguers filtered from `ctx.player_pool` (FA PR #98) |
| `test_compute_base_value_inverse_stats.py` | `_compute_base_value` delegates to `SGPCalculator.marginal_sgp` so inverse stats (ERA/WHIP/L) get negative contribution. Unknown pitchers with default-bad stats score LOWER than real MLB hitters (FA PR #99) |
| `test_fa_playing_time_gate.py` | `_playing_time_multiplier` penalizes low-GP FAs after 30-day grace period; curve 0.30x/0.60x/0.85x/1.0x (FA PR #101) |
| `test_dynamic_il_stash_list.py` + `test_no_hardcoded_il_stash.py` | `IL_STASH_NAMES` derived dynamically from `league_rosters`; no hardcoded set literal allowed in `src/alerts.py` (FA PR #102) |
| `test_fa_page_ui_polish.py` | ECR formatted as integer; position string dedup; "Other categories" reconciliation line in why-expander; "Show all NNN" pagination toggle present (FA PR #103) |
| `test_roster_sync_mlb_id_matching.py` + `test_no_name_only_roster_matching.py` | `match_player_id` emits DNA-collision WARNING when name-only resolves to a different team than caller's `team_abbr`; no new name-only roster matching code in `src/live_stats.py` / `src/yahoo_data_service.py` / `src/league_manager.py` / `src/data_bootstrap.py` (FA PR #104) |
| `test_il_return_date_multi_source.py` | `_parse_return_days_from_text` extracts day count from Yahoo status strings like `"IL10 - 3 days"` (FA PR #105) |
| `test_regression_flag_wiring.py` + `test_rate_stat_volume_in_targeted_adds.py` | `_score_fa_candidates` multiplies composite by `regression_mult` (BUY_LOW=1.05, SELL_HIGH=0.95). `compute_matchup_targeted_adds` does NOT use naive `team_totals["AVG"] +=` rate-stat addition (FA PR #106) |
| `test_constants_registry_coverage.py` | 23 FA-engine constants registered in `CONSTANTS_REGISTRY` with `value`, `citation`, `lower_bound`, `upper_bound`, `sensitivity`. Includes streaming, playing-time, ownership, floor, position-cap thresholds (FA PR #107) |
| `test_sustainability_counting_vs_rate.py` | `compute_sustainability_score` incorporates HR/FB% z-score (hitters) and LOB% z-score (pitchers) alongside existing rate-stat signals. Backward compat: missing counting cols → rate-only (FA PR #108) |
| `test_news_suspension_parser.py` | `parse_suspension_days` recognizes "Suspended through MM/DD" and "Suspended for N games" patterns; returns None for non-suspension text; clamps past-dates to 0 (FA PR #109) |
| `test_l14_blend_wiring.py` | `_blend_fa_row` consumes `l14_*` columns when `l14_pa >= 20` (hitters) or `l14_ip >= 5` (pitchers); below gate, falls back to renormalized ROS+YTD (FA PR #110 / P5b) |
| `test_ros_projection_playing_time_scale.py` | `_scale_ros_by_playing_time` discounts ROS counting stats by `max(0.20, ytd_gp/expected_gp)` for low-volume FAs after 30-day grace period (FA PR #110 / P5c) |
| `test_drop_candidate_diversity.py` | `_deduplicate_and_limit` walks sorted swaps and produces distinct (add, drop) pairs when alternatives exist (FA PR #110 / P5a — regression guard) |
| `test_punt_category_awareness.py` | `_score_fa_candidates` overrides `category_weights` to 0.05× for punted categories (from `ctx.h2h_strategy.get("punt", [])` or per-cat `win_prob < 0.10`) (FA PR #110 / P5f) |
| `test_fa_nan_guard.py` | `_score_fa_candidates` must not raise `ValueError` when FA rows contain NaN `player_id` or NaN `is_hitter` (post-pool-merge left-join artefact). `_is_hitter_safe()` helper guards all 5 int-cast sites; NaN `player_id` rows silently skipped; valid FA rows still score correctly. `compute_sustainability_score` in `waiver_wire.py` also guarded. |
| `test_no_dna_collision_dedup.py` | `upsert_player_bulk` matches by `mlb_id` first (then name+NULL-mlb_id), so two MLB players sharing a name but with distinct mlb_ids (Muncy LAD vs Muncy ATH, Will Smith LAD vs ATL, etc.) each get their own players row. `deduplicate_players` skips groups where distinct non-null mlb_ids > 1 — preserves same-name-different-mlb_id rows. Backward compat: same-mlb_id + NULL-mlb_id duplicates still merge. Replaces the one-shot `scripts/migrate_muncy_dna_2026_05_21.py` — fix is now self-healing on every bootstrap. |

## Audit History

The data + analytics pipeline has been audited across 13 waves (April–May 2026) covering ~105 silent-failure / drift / dead-code / type-design / migration bugs. All HIGH-severity findings are resolved across PRs #7–#23 + the 2026-05-17 deep-audit cleanup (PRs #29–#46) + 2026-05-19 deep-audit completion + 2026-05-19 follow-up cleanups (PRs #47–#52) + 2026-05-20 silent-failure sweep (PRs #59–#72).

The cumulative structural-invariant guard set in `tests/test_no_*.py`, `test_pages_*.py`, `test_wave*.py`, and `test_sf*.py` covers ~75 patterns that were silent-failure-prone, duplication-prone, or schema-evolution-prone before audit. See `docs/archive/2026-05-17-deep-audit-punchlist.md` for the historical deep-audit punchlist (now fully shipped) and `docs/archive/specs/` for shipped design docs from earlier waves.

**2026-05-19 follow-up PRs (post deep-audit completion):**

| PR | Title | Highlights |
|----|-------|------------|
| #47 | Deep-audit punchlist completion + repo hygiene + preflight tooling | 17 commits squashed; Sections 2-8 shipped; 6 new structural-invariant guard tests; preflight env+access verification script |
| #48 | Research.md gap-status reconciliation against PRs #18-#47 | 20 numbered gaps reclassified: 13 shipped, 3 partial, 4 still gap |
| #49 | silent-failure-hunter follow-up: H1 + M1/M2/M3/L1 + H2 regression lock | Found 2 HIGH + 3 MED + 1 LOW post-merge; H1 case-insensitive SQL was a real bug; H2 was a false positive locked with regression test |
| #50 | News_fetcher O(N*M) → 3-tier lookup | Caused CI shard 1 hang in PR #47; now <50ms exact / <5s at-scale |
| #51 | Un-skip 3 bootstrap integration tests via auto-mock fixture | Replaces 8-of-30-phases mock list with introspection-based fixture |
| #52 | Test isolation fixes for 4 local-only failures | DataFreshnessTracker hydration + YahooDataService singleton mocked; tests pass locally and on CI |

Milestone tag `milestone/2026-05-19-deep-audit-complete` marks SHA after PR #49.

**2026-05-20 silent-failure sweep (PRs #59–#72):**

| PR | Title | Highlights |
|----|-------|------------|
| #59 | fix(bootstrap) add 120s timeout to Yahoo FA phase | Prevent indefinite hang on Yahoo rate-limit (480-FA pagination × backoff) |
| #60 | fix(app) session-start bootstrap uses staleness check not force=True | Stops every new tab from re-running the full 33-phase bootstrap |
| #61 | fix SFH HIGH findings (M-1 + H-1 + H-2) | M-1: news_fetcher canonical-name collisions (Muncy DNA — "Will Smith" + "Will Smith Jr." silently overwrote). H-1: PR #59 FA-timeout signal clobbered by empty-fa_df else branch. H-2: standings_engine.py:206 dead `float(hitters["pa"].sum())` |
| #62 | fix SFH MED/LOW polish | M-2 news_fetcher prefix fallback, M-4 lineup_optimizer SIM114 parens, L-1 dual_objective dead branch, L-2 live_stats COLLATE NOCASE on remaining 2 sites, K CLAUDE.md "7" → "6" |
| #63 | fix(logging) isolate bootstrap.log from pytest test runs | RotatingFileHandler gated on `pytest not in sys.modules`. Test mock noise (e.g. `RuntimeError("DB out")`) no longer leaks into production log file |
| #64 | fix(refresh_log) honor partial/cached/skipped statuses on read (M-3) | `check_staleness` force-refreshes only {error, unknown}; partial/no_data/cached/skipped honor TTL. `data_freshness` hydrates from {success, partial, cached, skipped} |
| #65 | fix(refresh_log) record error status on exception path (D) | 9 phases (ecr_consensus, adp_sources, contracts, news, depth_charts, prospects, news_intel, bat_speed, forty_man) silently swallowed exceptions. AST-checked guard prevents regression |
| #66 | fix(bootstrap) surface Yahoo-skipped state in injury_writeback (B) | `lr_count == 0` skip path now appends `[yahoo skipped: rosters empty]` to refresh_log message instead of looking identical to a full run |
| #67 | fix(refresh_log) treat 0 >= 0 as success (idempotent no-op phases) (F) | `update_refresh_log_auto` was labeling `_enrich_pitcher_positions` (writes 0 rows when already enriched) as "no_data" instead of "success" |
| #68 | fix(season_stats) lower expected_min threshold 70% → 25% (G) | Realistic match rate — MLB API returns stats for all players (incl. minor-league callups) but `players` table covers ~1100 |
| #69 | fix(db) bump SQLite busy_timeout 30s → 60s (A) | 7 "database is locked" errors on 2026-05-20 across parallel ThreadPoolExecutor group (umpire + catcher_framing + pvb_splits) |
| #70 | test(perf) relax test_prefix_pruning_speed thresholds (J) | CI "Coverage Floor" was actually a flaky timing test (1.2s > 1s on GitHub Actions); coverage was 70.57% all along |
| #71, #72 | Follow-ups | Updated `test_check_staleness_treats_non_success_as_stale` for M-3's new contract; ruff format on the renamed test |
| #74 | fix SFH MED-1 + LOW-1 + LOW-3 follow-ups | MED-1: `_enrich_pitcher_positions` error path silent-skip (same DNA as #65, missed by the `_bootstrap_*`-prefixed structural guard). LOW-1: CLAUDE.md busy_timeout doc drift after #69. LOW-3: split injured_count into fresh-ESPN vs stale-yahoo when skipped path taken |
| #75 | fix case-insensitive `WHERE name = ?` cleanup (LOW-2) | 12 case-sensitive `WHERE name = ?` queries against `players` across data_bootstrap, data_pipeline, database, league_manager, live_stats. Pre-existing Muncy-DNA gap. Added structural guard `test_players_name_lookup_case_insensitive.py` (no allowlist) |

Milestone tag `milestone/2026-05-20-silent-failure-sweep-complete` marks SHA after PR #75 (re-tagged 3x as the docs/follow-ups landed).

**2026-05-20 afternoon "fix everything" sweep (PRs #77–#86):**

Bootstrap diagnostic surfaced 12 distinct issues across HIGH/MEDIUM/LOW severity. The user directive was "everything must be fixed, no matter how small"; after triage 7 confirmed-bug PRs shipped, 3 investigation PRs shipped (each found a real bug under the LOW category), and 5 items were documented as design choices (see _Known Design Choices_ below) rather than re-flagged on future audits.

| PR | Title | Highlights |
|----|-------|------------|
| #77 | fix(live_stats) TWP routing in save_season_stats_to_db (H1) | Ohtani's 2026 pitching-only row was rejected by the Bellinger is_hitter type guard — fetched as is_hitter=False, stored as is_hitter=1 → type_mismatch → zero 2026 season_stats. Fix: route TWP players through stat-type-specific UPSERTs (hitting cols + games OR pitching cols + games), preserving the off-type half. Bellinger protection preserved verbatim for non-TWP |
| #78 | fix(bootstrap) refresh_log resilience for lock + timeout failures (H2 + H3) | Two silent-failure modes: per-phase except handlers attempted update_refresh_log during a parallel DB lock and the inner `except Exception: pass` swallowed it (H2); _run_with_timeout returned "Timeout after Ns" but never wrote refresh_log (H3). Fix: _try_write_refresh_log helper with 3-attempt 0.5s/1s backoff; _run_with_timeout accepts source= kwarg; end-of-bootstrap _reconcile_results_to_refresh_log walks results dict and overwrites stale-success rows. Added "timeout" to _VALID_REFRESH_STATUSES |
| #79 | fix(bootstrap) pvb_splits commit per-batter (M1) | Root cause of the parallel-write contention PR #69 band-aided. pvb_splits held a single write transaction across 50 batters × 1-3s Statcast fetches → blew through 60s busy_timeout, starving umpire_tendencies + catcher_framing in the same ThreadPoolExecutor group. Move conn.commit() inside the outer batter loop. AST guard ensures it stays inside |
| #80 | fix(bootstrap) drop pybaseball batting_stats pos= kwarg (M2) | pybaseball 2.2.7 rejects pos=. Drop the kwarg, filter by Pos column post-hoc. AST guard prevents regression |
| #81 | fix(db) yahoo_player_key column on league_rosters (M4) | Ohtani-Pitcher and Ohtani-Batter are TWO Yahoo entities (different player_keys, both → HEATER player_id=2). Without yahoo_player_key, queries can't distinguish which Yahoo entity each team rosters. Foundation step: add the nullable column, populate it in Yahoo sync, preserve UNIQUE(team_name, player_id). Downstream stats routing left for a follow-up once consumers need it |
| #82 | test(isolation) fix 4 test failures from production-DB pollution (G) | TestRollingStats::test_compute_total + 3 TestComputeHotColdReport pitcher tests were reading production game_logs (8347 rows mid-season) because setup_method used INSERT OR IGNORE and assumed empty tables. Fix: DELETE-before-INSERT + INSERT OR REPLACE for rolling stats; patch src.player_databank.compute_rolling_stats to empty DataFrame for war_room tests so the mocked get_player_recent_form_cached actually runs |
| #83 | perf(ecr) parallelize 6 source fetches via ThreadPoolExecutor (L1) | ECR consensus routinely hit its 240s budget — 6 sequential HTTP scrapes summed to ~240s. Parallel with max_workers=6 drops to max(individual), roughly 4× speedup. Per-source try/except preserved verbatim |
| #84 | fix(bootstrap) use paginated get_all_free_agents (L6) | Bootstrap called get_free_agents(count=200) but Yahoo's API caps single calls at 25 results per page regardless of `count`. get_all_free_agents already existed and paginates start=0,25,50,... Switched the call. Stored FAs jumps from 25 → ~500 |
| #85 | fix(yahoo) batch-resolve draft pick names via Yahoo Players API (L8) | ~25 rounds 1-3 picks (70%) came back as "Player 469.p.XXXX" placeholders because yfpy's get_league_draft_results doesn't expand the player resource. New resolve_player_names_by_keys helper batch-fetches names (25 per call, Yahoo's cap). Still-unresolved keys keep the legacy placeholder so the NOT NULL constraint doesn't break |
| #86 | docs(claude.md) document not-bugs + audit history append | This PR. Adds the _Known Design Choices_ section so future audits don't re-flag L2/L3/L4/L5/L7/M3 |

**2026-05-20 → 2026-05-21 FA Engine Overhaul (PRs #89-#110)** — 22 PRs landing the most invasive engine rewrite in the codebase's history. Triggered by the Crochet/Kirk live-validation report ("Drop top-30 SP on IL15 for a backup catcher"). The plan in `docs/archive/specs/2026-05-20-fa-engine-overhaul-plan.md` + `docs/archive/specs/2026-05-20-fa-engine-p3.5-plan.md` covers the architectural decisions.

**Phase 1 — Critical defense-in-depth (PRs #89, #90, #91):**

| PR | Title | Highlights |
|----|-------|------------|
| #89 | fix(fa_page) unified IL-stash protection across top + bottom drop tables | UI-layer filter expands IL_STASH_NAMES with news-keyword matches + league_rosters.status. Both the headline "Top pickup" callout and the bottom "Recommended Drops" table now apply the same filter |
| #90 | fix(in_season) IL players keep weighted projection instead of being zeroed | Root cause of Crochet/Kirk class — `_roster_category_totals` was zeroing IL contributions, making drops of IL aces look "free" in the SGP math. Replaced with `_il_weight_from_status(status)` returning a [0, 1] weight scaled by expected return window |
| #91 | fix(fa_page) wire to recommend_fa_moves + positional scarcity | Free Agents page switched from legacy `compute_add_drop_recommendations` to `recommend_fa_moves`. PR1 + PR3 combined |

**Phase 2 — Engine quality (PRs #92, #93, #94):**

| PR | Title | Highlights |
|----|-------|------------|
| #92 | fix(fa_recommender) blend ROS + YTD when scoring FAs | `_blend_fa_row` implements canonical 0.70 ROS + 0.20 YTD blend with sample-size gating (skips when ytd_gp < 30). L14 placeholder noted as future work |
| #93 | fix(waiver_wire) rewrite compute_sustainability_score as calibrated sigmoid | Replaces 5-bucket step function with logistic on xwOBA-wOBA gap (hitters) + SIERA-ERA gap (pitchers). Pitcher inversion bug fixed |
| #94 | fix(fa_recommender) composite formula no longer double-counts urgency | Urgency now applied multiplicatively per-category inside `_compute_base_value` (via `ctx.category_weights`), not additively in the composite line. The additive boost summed weights across 4-6 cats which dwarfed base_value for marginal FAs |

**Phase 3.5 — Live-validation surfaced bugs (PRs #95, #96, #97):**

| PR | Title | Highlights |
|----|-------|------------|
| #95 | fix(in_season) `_il_weight_from_status` uses return-date curve for short suspensions | Framber Valdez (6-day suspension ending today) was getting weight 0.0. New piecewise-linear curve consumes `expected_return_days` from ESPN + Yahoo news |
| #96 | fix(waiver_wire) symmetric positional scarcity on `compute_drop_cost` | PR3 added scarcity to add side; this PR matches on drop side. Hoisted `compute_positional_scarcity_factor` to `src/valuation.py` to break import cycle |
| #97 | fix(fa_recommender) add roster-construction guard | New `_passes_roster_construction_guard` blocks 3rd catcher adds + drops below starting-lineup minimums |

**Phase 3.6/3.7/3.8/3.9 — Root-cause cascade discovered via diagnostic (PRs #98, #99, #100, #101):**

| PR | Title | Highlights |
|----|-------|------------|
| #98 | fix(fa_recommender) pass user roster + filter MLB-only pool | THREE compounding bugs: (a) FA page passed full 12-team rosters to `build_optimizer_context`, making `ctx.user_roster_ids` contain all 317 league players; (b) `load_player_pool()` loaded minor leaguers, polluting top FA candidates; (c) silent legacy fallback masked engine errors. Fix: defensive multi-team filter in builder + `level_filter` param + visible `st.warning` on fallback |
| #99 | fix(fa_recommender) delegate `_compute_base_value` to canonical `marginal_sgp` | The earlier inline formula `value += (stat/denom) * weight` had WRONG SIGN for inverse stats (ERA, WHIP, L). Unknown pitchers with default `era=9.0/whip=2.0` scored base=+38.8 (should be -38.8). Brandon Marsh (real hitter, .325 AVG) scored 18.3 — half the unknown phantom. Delegated to `SGPCalculator.marginal_sgp` which handles signs and rate-stat volume correctly |
| #100 | fix(fa_recommender) position-cap boundary off-by-one | PR #97's check used `cnt >= cap` (block at or above). Cap = max allowed depth; having exactly `cap` is healthy. Changed to `cnt > cap` (block only when EXCEEDING) — was blocking legitimate same-position upgrade swaps |
| #101 | fix(fa_recommender) playing-time gate de-weights IL-stash phantoms | Jordan Westburg (0 YTD GP, post-wrist-surgery IL stash) was top FA pickup because his ROS projection assumed regular playing time. New `_playing_time_multiplier` curve (0 GP → 0.30x) penalizes low-volume players with 30-day grace period |

**Phase 4 — DNA collision fixes (PR #104):**

| PR | Title | Highlights |
|----|-------|------------|
| #104 | fix(roster_sync) DNA collision warning + Muncy migration | Two Max Muncys in MLB (LAD veteran mlb_id=571970 + ATH rookie mlb_id=691777). Yahoo correctly identified user's LAD Muncy via `editorial_team_abbr`, but `match_player_id` did name-only lookup → mapped to ATH Muncy's `player_id=71`. Engine read ATH Muncy's stats (.239/2HR/26GP) for user's LAD Muncy (.263/12HR/47GP). Fix: WARNING when name-only resolves to a different team than caller's `team_abbr`. One-time `migrate_muncy_dna_2026_05_21.py` inserted LAD Muncy as new row + repointed user's league_rosters |

**Phase P3+P4 cleanup (PRs #102, #103, #105, #106, #107):**

| PR | Title | Highlights |
|----|-------|------------|
| #102 | fix(alerts) derive IL_STASH_NAMES dynamically from league_rosters (FA PR7) | Hardcoded {Bieber, Strider} set went stale weekly. New `get_il_stash_names()` queries `league_rosters.status IN (IL10, IL15, IL60, IL, DTD)`. Surfaces ~52 IL players vs 2 hardcoded |
| #103 | FA Page UI Polish (PR11+12+13+14) | ECR shown as integer; position string dedup (`"2B/3B,3B"` → `"2B,3B"`); "Other categories" reconciliation line; "Show all NNN" pagination toggle |
| #105 | fix(in_season) `_il_weight_from_status` parses day-count from status string (FA PR8) | Yahoo status strings like `"IL10 - 3 days"` now extract the day count via 4 regex patterns; falls through to existing string lookup when no pattern matches |
| #106 | fix(fa_recommender, waiver_wire) wire regression_flag + fix rate-stat volume (FA PR10) | `regression_flag` column (BUY_LOW / SELL_HIGH) was loaded but never consumed. Wired as 5% nudge in `_score_fa_candidates`. Rate-stat volume guard test added (no bug found, but locked against future regression) |
| #107 | fix(constants_registry) consolidate FA-engine magic constants (FA PR9) | 23 magic constants migrated from `fa_recommender.py` + `waiver_wire.py` into `CONSTANTS_REGISTRY` with citations, bounds, sensitivity tags. Includes streaming thresholds (Razzball / Pitcher List), playing-time gate (PR21 calibration), ownership boost, floor multipliers |

**Phase P5 — Additional engine quality (PRs #108, #109, #110):**

| PR | Title | Highlights |
|----|-------|------------|
| #108 | feat(waiver_wire) sustainability differentiates counting vs rate regression (FA P5e) | `compute_sustainability_score` adds HR/FB% z-score (hitters) and LOB% strand-rate z-score (pitchers). A 30% HR/FB hitter now gets lower sustainability than 13% peer; high-LOB pitcher rides regression risk |
| #109 | feat(news_sentiment) parse_suspension_days extracts return date from news (FA P5d) | Recognizes "Suspended through MM/DD" (date parse) and "Suspended for N games" (game-count → day estimate). Wired into `expected_return_days` after ESPN merge (ESPN priority) |
| #110 | FA Engine P5 Bundle: L14 + ROS-scale + diversity + punt (P5b+c+a+f) | Four sequential commits in one PR. L14 wired into `_blend_fa_row` (canonical 0.70/0.20/0.10); ROS scaled by playing-time ratio before blending; drop-candidate diversity regression-guard test; punt-category weight override (`0.05x` for categories with win_prob<10% or explicit punt tag) |

Milestone tag `milestone/2026-05-21-fa-engine-overhaul-complete` marks SHA after PR #110.

**2026-05-21 evening — DNA-collision root-cause fix (post-#110):**

The Muncy DNA migration (PR #104) was a one-shot fix that did not survive subsequent bootstraps. Investigation surfaced two compounding upstream bugs:

- `upsert_player_bulk` in `src/database.py` did SELECT-first by **name only** — collapsed LAD Muncy + ATH Muncy into a single row on every MLB Stats API ingest, with last-write winning the team column.
- `deduplicate_players` grouped rows by **name only**, then merged genuinely-different MLB players (different mlb_ids) into a single canonical row — destroying the migration's inserted row on every full bootstrap (phase 12 dedup).

Both functions are now `mlb_id`-aware:

- `upsert_player_bulk` matches by `mlb_id` first; falls back to (name + NULL-mlb_id) only for data enrichment of legacy rows. Never collapses rows with distinct non-null mlb_ids.
- `deduplicate_players` detects same-name groups with >1 distinct non-null mlb_id and skips dedup for those groups (logs an `INFO` line). Same-mlb_id and NULL-mlb_id duplicates still merge (back-compat preserved).

End-to-end behavior on a full bootstrap with this fix:

1. Players phase fetches all MLB players → both LAD Muncy (mlb_id=571970) and ATH Muncy (mlb_id=691777) get separate rows in `players`.
2. Dedup phase finds the name collision, sees distinct mlb_ids, **skips merge**.
3. Yahoo roster sync calls `match_player_id('Max Muncy', editorial_team_abbr='LAD')` → (name + team) precise match hits LAD Muncy → `league_rosters.player_id` updated correctly.
4. FA recommender, lineup optimizer, war room — all downstream consumers reason about the correct player's projections.

Guard test: `tests/test_no_dna_collision_dedup.py` (7 tests; 4 target the bug, 3 lock the back-compat semantics).

`scripts/migrate_muncy_dna_2026_05_21.py` is now redundant — kept for reference but can be removed once verified across fresh installs.

**2026-05-23 → 2026-05-28 — Enhanced Trade Engine + T1.21 OAuth decoupling (PRs #113-#118):**

These PRs landed the Enhanced Trade Engine research report (Sections B-H, plus Section G backtest framework) and the T1.21 OAuth-hang fix. All are squash-merged to master; CI green on each.

| PR | Title | Highlights |
|----|-------|------------|
| #113 | fix+feat: Bugs D/E/F + UI wiring of Features 1/2/3 into Trade Analyzer | Bug D (ghost-team rank filter), Bug E (grade/grade_range consistency post-Bug-A), Bug F (IP-floor `weeks_remaining` ROS horizon). Wired IP-floor penalty, weekly H2H matrix, playoff/championship sim into `pages/11_Trade_Analyzer.py` |
| #114 | fix(trade_engine) empty league_rosters must not silently zero standings (PR #113 hotfix) | Caught by master CI ~22 min after #113 merge: `test_sf21_caller_chain` failed ("category_gap_analysis was never called"). Empty `league_rosters` was silently zeroing standings instead of falling back to DB query |
| #115 | fix+feat: Bug G (LP pitcher fill) + three-horizon UI split | Bug G: LP pitcher-fill edge case. Three-horizon UI split (this-week / ROS / playoff) on the trade output |
| #116 | feat(playoff_sim) full opponent schedule simulation (report B.10 Phase 3) | `simulate_playoff_outcomes` + `simulate_trade_playoff_delta` with full opponent week-by-week schedule, Bradley-Terry bracket, paired-MC discipline |
| #117 | feat(engine) backtest + calibration framework (report Section G) | `run_historical_backtest` data-agnostic ingestion; Section G error-metric targets (rank_mae, cat_win_rmse, trade_prediction_spearman, projection_mae) |
| #118 | T1.21 OAuth decoupling | Lineup Optimizer optimize click no longer force-refreshes Yahoo (closed the 159s OAuth hang); explicit "Refresh Yahoo Data" button; 15s ThreadPoolExecutor timeout in `YahooDataService._get_cached` (timeouts recorded as `error` in cache stats). Guarded by `tests/test_oauth_decoupling.py` |

Milestone tags: `milestone/2026-05-21-fa-engine-overhaul-complete` (PR #110); `milestone/t1.21-oauth-decoupling-complete` (PR #118); `milestone/2026-05-29-v2-admin-dashboard-complete` (Plan 3, current master).

**2026-06-13 Beta design-audit fix pass (merge `88a552f`, DEPLOYED to master/Railway):** consolidated two pre-Beta UI/UX audits (`HEATER Design_Audit_Claude.md` at repo root + `docs/design-audit/HEATER_Design_Audit_Hermes.md`; per-page reports in `docs/design-audit/`) into a 5-phase fix pass — plan in `docs/superpowers/plans/2026-06-13-heater-beta-design-audit-implementation.md`. 37 commits, +19,952/−932, ~680 new tests (suite 5944 green). **Admin pages + the HEATER AI chat (`src/ai/`) were out of scope and untouched.** Behavioral/structural changes future work must respect:

- **In-season nav default is now My Team, not the Draft Tool.** `src/nav.is_in_season()` (→ `league_rules.weeks_remaining() > 0`) drives `build_pages`: in-season → `Season` group with My Team `default=True` + a `Preseason` group (Draft Tool + Draft Simulator); off-season → the old `Home`/`Season` layout. Draft Simulator eyebrow is `SCOUTING` in-season. Guard: `tests/test_nav.py`.
- **New shared `ui_shared.py` components** (locked-system-safe, no emoji/off-palette hex): `render_data_freshness_chip(source=None, *, age_minutes=None)` + pure `humanize_age()` (amber > 24h, reads `get_refresh_log_snapshot()`); `JARGON` dict + `jargon_help(term)` (for `help=` tooltips) + `render_glossary_expander(terms=None)`. Used across the data-dense pages. Guard: `tests/test_ui_shared_components.py`.
- **Trade Analyzer Monte Carlo is OPT-IN** (`enable_mc=False` default + a labeled checkbox) — fixed the 44.8s synchronous-MC WebSocket-drop freeze. Pool loads are `@st.cache_data`-wrapped; the Trade Finder full scan is button-gated + session-cached (no 43.8s cold load). Punt uses `total_sgp_batch` (2.1s→6ms).
- **`tests/test_no_offpalette_hex_in_pages.py` now also scans `src/`** (banned Material/Tailwind-era hexes), with a per-file allowlist for intentional brand colors (`TEAM_BRAND`, news-source colors). `src/trade_value.py`/`schedule_grid.py`/`player_tags.py`/`injury_model.py` were migrated to THEME tokens. The injury Low-Risk dot is now `#1f9d6b` (= `THEME["ok"]`), not `#84cc16`.
- **Splash timer uses `st.html(_timer_html, unsafe_allow_javascript=True)`** — the supported replacement for the deprecated `st.components.v1.html`. NOT `st.iframe` (that takes a URL, not srcdoc HTML) and NOT plain `st.markdown` (strips `<script>`).
- New per-fix guard tests added throughout `tests/` (`test_*_trust_*`, `test_*_audit_*`, `test_trade_analyzer_perf_fixes`, `test_pool_caching`, `test_databank_dead_controls`, `test_fig_numbering_and_title_drift`, etc.).

**2026-06-14 Beta roadmap (Phase 7) — SHIPPED + DEPLOYED to master/Railway (merge `1b7da28`, branch `feat/beta-roadmap-phase7`, 17 commits, full suite 6311 green).** The previously-deferred roadmap is now built (plan `docs/superpowers/plans/2026-06-13-heater-beta-design-audit-implementation.md`, Phase 7 R-1..R-11). New surfaces + invariants future work MUST respect:

- **Per-user state (`src/user_data.py`):** additive, idempotent, MULTI_USER-gated tables `user_watchlist` + `user_saved_views` (created in `init_db()`); helpers `add_to_watchlist`/`toggle_watchlist`/`get_watchlist`, `save_view(kind,name,payload)`/`load_view`/`list_views`/`delete_view` (`kind` ∈ "lineup"/"trade"/"ui"). Falls back to `user_id=0` when the flag is off. Powers **watchlists** (Free Agents, Closer) + **save/load** (Lineup, Trade Analyzer) [R-3] and the **onboarding** dismissal flag [R-8].
- **CSV export [R-4]:** `st.download_button(..., mime="text/csv")` on Free Agents, Leaders, Databank, Standings (alongside existing Excel where present). Guard: `test_csv_export_buttons.py`/`test_r4_csv_export.py`.
- **Mobile [R-7] + a11y [R-9]:** additive `@media (max-width:768px/480px)` block in `inject_custom_css` (scrollable tables, 44px touch targets, hero/card scaling); ARIA/`title`/`role` + `build_health_dot_html()` on the HTML builders + `PAGE_ICONS`. Both keep the combustion lock green. Guards: `test_mobile_responsive.py`, `test_accessibility.py`.
- **Dark mode [R-10]:** default-OFF toggle persisted via `user_data`; `inject_custom_css` emits dark CSS-var overrides ONLY when the user pref is dark, so the light theme stays byte-identical (combustion lock unaffected). Guard: `test_dark_mode.py`.
- **Async optimize [R-6]:** Lineup "Optimize" runs on a module-level `ThreadPoolExecutor`; the threaded `_run_optimize_compute(...)` is PURE (no `st.*`), `_poll_optimize(session_state)` is the idle/running/done/error state machine, results land in `lineup_optimizer_result` then render. Math unchanged. Guard: `test_lineup_optimizer_async.py`. (`test_heavy_action_spinners`/`test_load_park_factors_reader` updated for the new shape.)
- **Yahoo write-back [R-1]:** `YahooFantasyClient.set_lineup(assignments, date)` + `add_drop(add_key, drop_key)` are GRACEFUL (never raise; structured `{"ok",...}`; 401/403 → "re-authorize Yahoo" msg) and confirm-gated in the UI (Lineup "Apply"; Free Agents "Add/Drop"), token-owner only, with a "do it manually" fallback. **REQUIRES the Yahoo app to have write (`fspt-w`) scope — it was registered read-only, so the first live write will likely 401 until re-authorized.** Untestable here (all tests mocked). Guards: `test_yahoo_write_methods.py`, `test_lineup_apply_yahoo.py`, `test_fa_add_drop_yahoo.py`.
- **Monolith split [R-5]:** pure helpers extracted to `src/my_team_helpers.py` + `src/lineup_optimizer_helpers.py` (behavior-identical; extraction limited to helpers NOT pinned by structural guards, which all stay green).
- **Chrome tweaks:** global top bar halved (`.stApp::before` 90→45px, content offset 53px); sidebar rail widened 100→**140px** (the MAX under `test_combustion_lock`'s `<=140` "thin rail" cap — this also un-truncates the nav labels); the **HEATER logo is sized by `!important` CSS in `inject_custom_css`** (`.heater-logo`/`.heater-logo img`), NOT the one-time `components.html` JS inline style (that only runs when the div is first created and can't update on rerun — edit the CSS, not the JS).

**Roadmap status (updated 2026-06-16 — see ★ NORTH STAR in Overview):** **multi-league / multi-tenancy** and the **thin-client / FastAPI-Next.js (or Reflex) re-platform** are NO LONGER "out" — they are now the planned path to the public, monetized consumer product (the ultimate goal). **Progress (2026-06-19):** the **backend API foundation (Sub-project B) is now BUILT — 16 dormant `/api/*` endpoints on `master`** (see "★ Sub-project B status" in the Program structure section above) and the CMO's Next.js frontend (`web/`) is on `master` too. The **deferred infra (Postgres/workers/auth/multi-tenancy, B2–B5)** and the frontend page-cutover (C/D) remain unbuilt; the current private-league Streamlit app stays the working baseline (untouched) until the new stack reaches parity. Do not start the deferred infra (B2–B5) without first agreeing a migration plan with the owner — those touch the LIVE data path. **★ Updated 2026-06-19:** the CMO's sequenced M0–M5 product roadmap (`docs/superpowers/specs/2026-06-19-heater-migration-roadmap.md`) now puts this infra in **M4 — AFTER** the single-league paid beta (owner's revenue-first / lean-infra call). So **B2/Postgres execution is PAUSED** (B2.0/B2.1 shipped but dormant) and the **active backend line is M0 (contract lock)** — slice 1 shipped (see the ★ Sub-project B status block). **★ Updated 2026-06-21 (Platform/M4 track):** the **B4 multi-tenancy + tenant-resolution + per-user team-resolution design is LOCKED** (`docs/superpowers/specs/2026-06-21-heater-b4-multi-tenancy-tenant-resolution-design.md`; owner chose "design full multi-league now" + "admin-assigned teams for the beta") and its first slice **M3-1 per-user team resolution is SHIPPED on `master` — additive + DORMANT** (plan `docs/superpowers/plans/2026-06-21-heater-b4-m3-1-per-user-team-resolution.md`; memory [[project_b4_m3_team_resolution]]). M3-1 replaces the trusted client-supplied `team_name` query param with a viewer-team resolved from the (optional) Clerk identity: NEW api-owned `league` + `user_team` stores in the SEPARATE `api_state.db` (NOT `draft_tool.db`), `api/tenancy.py` (`ViewerContext`/`require_viewer_context`, replicating `src.auth.resolve_viewer_team_name`'s no-fallthrough + emoji/whitespace reconciliation), `api/identity.py::optional_app_user` (open reads stay open when Clerk off → byte-for-byte today's), deny-by-default admin assign endpoints (`api/routers/admin.py`, env `HEATER_ADMIN_CLERK_IDS`), and the 7 personalized routers wired additively (`ctx.effective_team(team_name)`). 340 api tests; both code-reviewer + silent-failure-hunter ran. **Owner-gated remainder (needs owner go + a REAL Postgres):** M4-1 `league_id` tagging + request-scoped repository filter + tenant-isolation tests; M4-2 `LeagueConnector` (Sleeper/Yahoo/ESPN/CBS/manual) + per-user encrypted `league_credential` + per-tenant refresh workers (B3) + per-user Yahoo OAuth; M4-3 multi-league switch-on. The **require-login activation flip is now SHIPPED + dormant** (2026-06-21): `require_viewer_context` 401s personalized reads when `api/auth.clerk_configured()` (the shared predicate that also picks `get_auth_verifier`, so no bypass) is true + no valid identity; `CLERK_ISSUER` unset → reads stay OPEN = byte-for-byte today's. So setting `CLERK_ISSUER` (M6) auto-requires login; silent-failure-hunter verified fail-closed. The **M3 beta launch/deploy plan** is written: `docs/superpowers/plans/2026-06-21-heater-m3-beta-launch-plan.md` (Vercel+Railway on current infra; owner decisions: reads require login, friends FREE via Pro-gate-OFF + Stripe checkout proven separately in TEST mode, default URLs; lane-tagged 🟦CEO/🟧CMO/🔧owner-M6). The remaining M3 step = the 🟧 frontend dropping its hardcoded "Team Hickey" + sending the Clerk token, and the 🔧 M6 owner setup. **★ M6 owner setup (Clerk + Stripe + Railway + Vercel) is DONE — owner-confirmed 2026-06-22; App B FastAPI is LIVE with the Clerk login-gate working ([[project_railway_two_service_arch]]).** **★ LAUNCH-DAY SEQUENCING (owner instruction 2026-06-22): the 12-leaguemate team-linking — set `HEATER_ADMIN_CLERK_IDS`=owner's Clerk user id, then `POST /api/admin/assignments` mapping each login → their team — is the VERY LAST launch-day item. Do EVERYTHING else first (the frontend "Team Hickey" cleanup + wiring `api/routers/schedule.py` to the viewer context, CORS=Vercel URL, Vercel verify); the assignment goes last because it needs each leaguemate's Clerk user id (they must sign in first) and is the final flip that resolves each login to their own team's data — until it runs, an un-linked login resolves to no/empty team.** **★ Updated 2026-06-22 (CEO deploy + audit session): the M3 pre-reqs above are now ALL DONE — only the launch-day assignment remains.** App B's 502 fixed (serviceDomain `targetPort` 8000→8080); App B flipped from a mis-running Streamlit to the real **FastAPI API** (added `railway.api.toml` → builds `api/Dockerfile`/`start-api.sh`; service setting `railwayConfigFile=railway.api.toml`; `STRIPE_*` vars removed = free friends). Vercel `heater-v1-0-1.vercel.app` wired to App B (`HEATER_API_BASE`=App B URL + `NEXT_PUBLIC_HEATER_LIVE=1` + `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` — derivable from `CLERK_ISSUER` as `pk_<env>_`+base64(`<frontend-api-host>$`)). The frontend "Team Hickey" Standings/Trades hardcode is FIXED (`f2decf7`, new `web/src/lib/viewer-team.ts` cache populated by `fetchMyTeam` from the API's identity-resolved team) and `api/routers/schedule.py` probables/hitter-matchups now use `require_viewer_context`/`effective_team` (`0418c89`, were trusting client `team_name`). **CORS is MOOT** — the Next.js frontend proxies `/api/*` SERVER-SIDE (`web/next.config.ts` rewrite → `HEATER_API_BASE`), so there are NO browser cross-origin calls (`HEATER_API_CORS_ORIGINS` need not be set). A full **data/engine/provider AUDIT passed**: App B reads the same `draft_tool.db` via its own boot scheduler; Yahoo (40 KB live rosters) + MLB StatsAPI + ESPN injuries (352) + season/historical stats all live; all 25 `api/services/*` engines wired = the same UNCHANGED `src/` engines as Streamlit (LP optimizer, 6-phase trade, FA, playoff sim, streaming, etc.), full 14-page parity + extras (Probable/Hitter grids, Bubba). **Operate Railway/Vercel headlessly with an owner token** (their CLIs ignore team tokens → drive the GraphQL/REST APIs directly with a browser `User-Agent` to clear Cloudflare; helper scripts `C:\Users\conno\railway_gql.py` / `vercel_api.py`; tokens EPHEMERAL — see [[reference-railway-ops-graphql]], [[reference-heater-live-urls]], [[project_railway_two_service_arch]]).** **After this flip the platform track's buildable-now backlog is EMPTY** until the owner unblocks M4 (real Postgres) / M6 (Clerk/Stripe/Railway/Vercel). **Multi-player trades already work** — the Trade Analyzer evaluates any N-for-M list; only the Trade *Finder's* auto-suggestion generator caps at 2-for-1 (a complex, low-value combinatorial enhancement, not built).

**★ Updated 2026-06-23 — two pre-launch deep-audit campaigns COMPLETE, all on `master`, CI green, both reviewers run per tier.** Two separate audit backlogs are fully shipped: **`Outstanding_June26.md`** (the parallel "round-2 hardening" session) = the 3 HIGH pre-launch fixes + the whole buildable-now MED/LOW backlog (service-logging sweep, Clerk-gated cross-team write guard, structured/`Literal` contracts + mapper coercion, live-never-fabricates empty states, matchup a11y). **`Bugs_June26.md`** (this track, deep-debug; memory [[project_bugs_june26_fix_campaign]]) = **Tier 1** correctness (league W-L records + per-category ranks via `load_league_records`, 5 NaN→crash/ MAX-score guards, matchup-hero `opp_name`), **Tier 2** `require_login` router-gate on the 7 private read routers (roster/manager leak closed; dormant until `clerk_configured()`, 401 when Clerk live), **Tier 3** `GET /api/players/{mlb_id}` player-detail endpoint + live `PlayerDialog` wiring + viewer-team fetchers + **true-sim matchup win/tie/loss** (`estimate_h2h_outcome_probs` over the Gaussian copula), **Tier 4** Low/Minor (leaders logistic `_norm_value` no-saturation, NaN/inf sweep across services + `valuation` scarcity + `kalman` 0/0 + `trade_service._f` latent-NaN-leak fix, **`ruff==0.14.10` pinned to CI** in requirements.txt — was a local-vs-CI format skew, per-page `<title>` via `DocumentTitle`, matchup empty-cell alignment via `cols`). A live-only regression I introduced (streaming-date formatter reused as the `/streaming/analyze` request body) was **caught in code-review and fixed** (`date` stays canonical ISO; format display-only). **Owner-gated remainder:** Clerk DEV→prod keys before public launch; player-detail slice 2 (l7/l14/l30 + game logs — needs a statsapi slice); the `request_refresh` capability-flag refactor (already safe in prod — silent-failure-hunter confirmed the live `MULTI_USER=1` write-gate blocks it for every Bubba user). The **`feat/ARCHIVED-web-closer-monitor`** remote branch is an intentional owner archive (its web build is superseded by the live `/closers` + `/api/closers`; the branch preserves the built foundation + a "Saves Finder" revival spec — **do not merge**, it would regress).

## Known Design Choices & External Limitations

These items have surfaced in audits as "broken" or "flagged" but are NOT bugs. They are either external limitations we can't fix from our side, deliberate design choices, or correctly-handled edge cases. **Do not flag any of these as bugs in future audits without re-reading this section first.**

| Item | Why it's not a bug |
|------|---------------------|
| **FanGraphs `leaders-legacy.aspx` 403** | External anti-scrape policy. We have Tier 2 (browser-headers — also 403) and Tier 3 (neutral defaults + FIP/xFIP K-boost proxy). No code on our side can change FanGraphs' policy. Documented in Data Sources table |
| **`bat_speed` phase skipped** | pybaseball doesn't expose the Statcast bat-tracking endpoint (no `statcast_batter_bat_tracking` in 2.2.7). The phase correctly returns "Skipped: pybaseball bat tracking not available". Cannot fix without upstream library support |
| **`pitcher_positions: 0 rows enriched`** | Idempotent no-op when all pitchers already have SP/RP enrichment from a prior run. PR #67 made this correctly report as "success" (not "no_data"). Adding pitchers to enrich next time would change the count — the 0 is the **correct** result for the cached state |
| **News fetcher canonical-name collisions** | There are genuinely 3 different active MLB players named "Luis Garcia", 2 named "Will Smith", etc. PR #61 made the warning explicit (`canonical-name collision 'luis garcia' maps to player_ids [...] - keeping first`). The fix path is team-based disambiguation at every news caller — a multi-module refactor for a known, accepted limitation. The current keep-first behavior is documented as deliberate |
| **`season_stats: partial 2497/7498`** | MLB Stats API returns ~7500 player-seasons including all 40-man + spring training + minor-league callups. Our `players` table is curated to ~1100 active MLB rosters. The ~5000 unmatched rows are players we don't track by design (not data loss). PR #68 lowered expected_min threshold 70%→25% to reflect this reality. The "partial" status reflects intentional scope, not pipeline failure |
| **pybaseball `statcast_catcher_framing` CSV tokenization** | Upstream pybaseball/Savant integration issue — Savant's response can't be parsed by pybaseball 2.2.7. Our 3-tier waterfall already catches this gracefully: Tier 2 (browser-headers Savant scrape) covers, then Tier 3 seed file. The pybaseball failure is logged as WARNING, not ERROR. No code on our side fixes the upstream parser; removing the call would lose resilience when pybaseball updates |
| **Playing-time gate 30-day grace period** | First 30 days of season, `_playing_time_multiplier` returns 1.0 regardless of GP/IP. Avoids penalizing late-March / early-April call-ups whose YTD samples are too small to interpret. After Day 30, the curve kicks in (PR #101) |
| **ROS playing-time scale floor (0.20x)** | `_scale_ros_by_playing_time` floors at 0.20× even for 0-GP players. Hot rookies just called up shouldn't lose ALL their ROS projection — 0.20 preserves a meaningful baseline while still discounting IL phantoms (PR #110 / P5c) |
| **Punt-category weight = 0.05x (not 0)** | When a category is detected as punted (h2h_strategy or win_prob < 10%), weight overridden to 0.05 not exactly 0. Avoids divide-by-zero edge cases in downstream rate-stat math while effectively zeroing the contribution (PR #110 / P5f) |
| **L14 volume gate (20 PA hitters, 5 IP pitchers)** | Below this, `_blend_fa_row` skips L14 and renormalizes to ROS+YTD. 20 PA / 5 IP is the empirical "signal noise floor" — below this, single hot/cold games dominate (PR #110 / P5b) |
| **Drop candidate dedup collapses to 1 when single drop dominates** | When all top FAs share a single best drop, `_deduplicate_and_limit` correctly surfaces only 1 rec (each drop used at most once). Adding alternative drops requires the engine to see additional viable swap pairs in `_evaluate_swaps`, not a dedup-logic change (PR #110 / P5a) |
| **ATH Max Muncy (player_id=71) preserved post-migration** | The Muncy DNA migration (PR #104 + `scripts/migrate_muncy_dna_2026_05_21.py`) inserted LAD Muncy as player_id=9864 (mlb_id=571970) and repointed Team Hickey's league_rosters to player_id=9864 — it did NOT delete or update the ATH Muncy row (player_id=71, mlb_id=691777). The ATH row remains for any other team in the league that might roster him |
| **Local Windows full-suite parallel run hang ~98% (FIXED 2026-05-29)** | _Root cause:_ a handful of network-touching tests could issue a real blocking C-level `socket.recv()`; on Windows `pytest-timeout` only has the `thread` method (SIGALRM is Unix-only), which DETECTS but cannot INTERRUPT a blocking socket, so the xdist worker died (`node down: Not properly terminated`) near the end of the run and the master hung indefinitely. CI (Linux) was always unaffected. _Fix:_ `tests/conftest.py` installs a session-wide **network guard** at import time — `socket.socket.connect`/`connect_ex` to any non-loopback host raises `NetworkBlockedError` instantly. HEATER's 3-tier fetch waterfalls catch it and fall back to seed/neutral data, so tests pass deterministically with no external dependency. The full suite (incl. `test_data_pipeline.py`, no longer ignored) now completes green in ~155-160s. Opt-outs: `@pytest.mark.allow_network` (per-test) or `HEATER_ALLOW_NETWORK=1` (global). Guarded by `tests/test_network_guard.py` |

When a future audit flags one of these, the correct response is: confirm it matches the entry above, then move on.

## GitHub

- **Repo:** https://github.com/hicklax13/HEATER_v1.0.0
- **CI:** GitHub Actions — ruff lint/format + sharded pytest (4 shards × 2 xdist workers, Python 3.12, ~5–7 min wall time) + 60% coverage floor + structural-invariant guards

## Testing

- **~5361 passing tests** across 448 test files, ~110 skipped (PyMC/XGBoost/WeasyPrint optional + network-gated)
- **CI:** GitHub Actions (Python 3.12, sharded 4 ways)
- **Coverage:** ~71.6% (60% CI floor — the 5-week "Coverage Floor red" was actually a flaky timing test, fixed 2026-05-20 by PR #70)
- **Pre-commit hook:** Enforces `ruff format` + `ruff check` on every commit
- **Pre-push hook:** Runs structural-invariant suite (~222 tests) — catches regressions on the locked design choices before they reach CI
- **Backtesting framework:** Historical replay validates engine recommendations vs actual outcomes
- **FA engine diagnostic toolkit:** 9 `scripts/diag_fa_*.py` scripts preserved as evidence + future debug tools (cross-type trace, score plateau inspection, roster data audit, etc.)

## Resume Checklist (New Session)

1. Confirm shell is in `C:\Users\conno\Code\HEATER_v1.0.1` (NOT the deprecated `OneDrive\Desktop` path; note the local folder is `v1.0.1` while GitHub repo NAME remains `HEATER_v1.0.0`).
2. Read `CLAUDE.md` (this file)
3. Check git status: `git status`, `git log --oneline -10`. Master should be at or beyond SHA marked by tag `milestone/t1.21-oauth-decoupling-complete` (PR #118)
4. Run `python -m pytest --ignore=tests/test_cheat_sheet.py -x -q` to verify tests pass (~4534 tests, ~3-5 min; ~155-160s parallel via `-n auto`). The 2026-05-29 conftest network guard eliminated the prior Windows ~98% hang — full suite now runs green incl. `test_data_pipeline.py`. See the "Local Windows full-suite parallel run hang (FIXED)" row in _Known Design Choices_ for the mechanism
5. Run `streamlit run app.py` and verify Yahoo auto-reconnect
6. Inspect refresh_log: `python -c "from src.database import get_refresh_log_snapshot; import json; print(json.dumps(get_refresh_log_snapshot(), indent=2))"`
7. (If continuing FA engine work) Read `docs/archive/specs/2026-05-20-fa-engine-overhaul-plan.md` + `docs/archive/specs/2026-05-20-fa-engine-p3.5-plan.md` for full design rationale of PRs #89-#110
8. (If debugging FA recommendations) Use `scripts/diag_fa_stage_by_stage.py` to see what the engine sees stage-by-stage. `scripts/diag_roster_data_audit.py` flags DNA collisions across the roster
