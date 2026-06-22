# HEATER — Deep-Debug Audit (`Bugs_June26.md`)

**Date:** 2026-06-22 · **Branch:** `master` (HEAD `7ec4666`) · **Method:** baseline gates + 13-agent finder sweep (read-only) + independent adversarial verification (live probes, code traces, venv repros). No application code or tests were modified.

> **Verification key:** `repro` = reproduced with the venv; `live` = observed on the deployed Railway/Vercel hosts; `code` = traced in source; `finder` = surfaced by a finder agent (confidence as noted), not individually re-reproduced here. Every High and the key Mediums were personally reproduced/traced.

## Summary

| Severity | Count | One-line |
|---|---:|---|
| **Critical** | 0 | No data-loss, RCE, write-auth bypass, secret leak, or cross-tenant data write found. |
| **High** | 4 | Streaming NaN→max-score; **league W-L records never surfaced (standings + my-team show 0-0/rank 0)**; draft NaN crash; canonical player card is 100% mock on live. |
| **Medium** | 9 | Private league data readable unauthenticated; Clerk **dev keys in production**; closers NaN crash; viewer-team/“Team Hickey” not wired (web+Bubba+tenancy); closer adapter drift; FA top-need; my-team matchup hero; Bubba metering; stale generated types. |
| **Low** | 13 | NaN/inf-safety gaps, leaders value saturation, stale UI copy, four-state error/empty mix-ups, CI ruff skew + guard under-coverage, etc. |
| **Minor** | 16 | Sign-corrupting string sanitize, dead no-ops, stale comments, page `<title>`, perf, ruff-format drift, warnings noise. |

### What was checked (green unless noted)
- **Baseline:** `pytest` **6813 passed / 108 skipped / 0 failed** (148 s); `ruff check` clean; `ruff format --check` 1 drift (Minor); `tests/api/test_openapi_contract.py` pass; web `tsc`/`lint`/`build` all exit 0 (19 routes prerender); **CI on master GREEN**.
- **Live Railway API** (`celebrated-respect-production.up.railway.app`): status matrix for ~14 endpoints (nothing 500'd); bodies inspected.
- **Live Vercel web** (`heater-v1-0-1.vercel.app`): all 14 routes rendered + console/network captured.
- **Engines** (`src/`): valuation/trade-engine, optimizer/DCV/streaming, draft/matchup/standings/war-room/backtesting.
- **API** (`api/`): 23 endpoints, contracts, openapi, auth/gating, dual-DB seam, per-tier AI cap.
- **Silent failures** across `src/` + `api/` + `src/ai/`.
- **Security:** Bubba `run_read_only_sql`, Pro gate, write-auth, CORS, secret hygiene.
- **Data:** schema, per-user isolation, viewer-team resolution, ledgers.

---

## Findings by surface

### A. Standings / My-Team / records (HIGH — systemic, live-confirmed)

- **[High]** API never surfaces league W-L records → `/api/standings` **and** `/api/me/team` always show `0-0` / rank `0` — `api/services/standings_service.py:48-99`, `api/services/team_service.py:321-329`, `src/yahoo_data_service.py:858-867`, `src/database.py:2389` — `_sync_standings` writes only per-**category** rows to `league_standings` (with `total`=category total and `rank`=the team's **overall** rank cloned into every category), and writes W-L-T to a **separate `league_records` table**. But `standings_service._build_teams` sources rank+record from a non-existent `OVERALL`/`RECORD` category row, and `team_service._rank_and_record` looks for a non-existent `category=="WINS"` row; `load_league_standings()` is `SELECT * FROM league_standings` and never joins `league_records`. — **Evidence (live):** `/api/standings` returns all 12 real teams at `rank:0, wins:0`, and **every per-category rank == that team's single overall position** (BUBBA=4 across all 18 cats, “Over the Rembow”=1, “Team Hickey”=8); the live Standings page renders exactly this while `/api/leaders/overall` shows real mid-season stats (James Wood 20 HR) → season is underway, so 0-0 is wrong. — **Fix:** read W-L-T from `load_league_records()`; derive top-level rank from records; compute per-category ranks from the `total` values (rank teams within each category) instead of cloning the overall rank. — **Confidence: High** (live + code; found independently by 2 finders + the live probe).
- **[Low]** Standings mapping is untested at the integration layer — `tests/api/test_standings.py` — only asserts contract serialization on hand-built `TeamStanding` objects; `_build_teams(df)` is never exercised against a realistic `league_standings` frame, so the test cannot catch the bug above. — **Fix:** add a `_build_teams` test fed a frame shaped like `_sync_standings`'s real output. — **Confidence: High** (code).
- **[Medium]** `/api/me/team` matchup hero is empty when the matchup isn't cached — `api/services/team_service.py:332-341` (`_matchup`) — opponent/week/win-prob default to `""`/`0`/`0.0`; combined with the records bug, a logged-in user's header is unpopulated. — **Confidence: finder/High.**

### B. Optimizer / streaming / DCV engines (`src/optimizer/`, `src/lineup_optimizer.py`)

- **[High]** Pitcher streaming: a present-but-NaN stat yields the **maximum** Stream-Score SGP component — `src/optimizer/stream_analyzer.py:381` (`comp_sgp = _clamp(net_sgp)`), fed by `src/optimizer/streaming.py` `_get` (`float(obj.get(key, default))`). — **Evidence (repro):** `_clamp(nan)` returns `1.0` because `min(1.0, nan)` returns its first arg (NaN comparisons are False); a probable with NaN era/whip/k/ip gets `components.sgp=1.0`, ranking a data-missing pitcher to the top of the board (finder: stream_score 68.3 vs 51.6 for a genuinely-average arm). — **Fix:** NaN-guard `_get` (treat present-NaN as the league-avg default), and make `_clamp` map NaN→0. — **Confidence: High** (repro).
- **[Medium]** Lineup LP / DCV NaN-unsafe stat extraction — `src/lineup_optimizer.py:247,~844` — `float(x)` on pool stats without a NaN guard can poison the LP objective. — **Confidence: finder/High.**
- **[Low]** SP/RP slot label arbitrary when roster slot-fill bonus ties — `src/lineup_optimizer.py:315-362`. **[Low]** Two parallel optimizer constant systems can drift (`_DEFAULT_TEAM_WEEKLY_IP=55.0` vs the registry `54`) — `src/optimizer/streaming.py:48,217-222`. **[Minor]** stale comment + weather-HR-bonus leakage + Bayesian expected-ER ignores opponent — `src/optimizer/daily_optimizer.py:460-478`, `src/optimizer/streaming.py:505-529`. — **Confidence: finder.**

### C. Draft / matchup / war-room / in-season (`src/`)

- **[High]** `DraftState.get_all_team_roster_totals` raises `ValueError` on a NaN pool stat — `src/draft_state.py:351-367` — `int(p.get("r",0) or 0)` doesn't guard NaN: `nan or 0` is `nan` (NaN is truthy) → `int(nan)` raises. — **Evidence (repro):** `int(float('nan') or 0)` → `ValueError: cannot convert float NaN to integer`. Reachable when any drafted player has a NaN counting stat (sparse minor-leaguer rows). — **Fix:** coerce via a NaN-safe helper (`int(v) if pd.notna(v) else 0`). — **Confidence: High** (repro).
- **[Medium]** Same NaN pattern in `_roster_category_totals` — `src/in_season.py:274-290` — and **[Medium]** breakout-score `is_hitter` NaN cast — `src/leaders.py:289`. **[Low]** `is_hitter` NaN misclassification across `src/in_season.py:405`, `src/matchup_planner.py`. **[Minor]** dead no-op statements (`src/draft_grader.py:196-199`, `src/matchup_planner.py:572`). **[Low]** cross-category raw-stat mixing in `src/simulation.py:876-913`. — **Confidence: finder (High/Med).**

### D. FastAPI surface (`api/routers/`, `api/services/`)

- **[Medium]** `/api/free-agents/pool` “top need” can mis-key — `api/services/fa_pool_service.py:43-48,103` (`_top_need`) — category-key case/lookup fragility (sibling of the documented lever `.upper()` bug). — **Confidence: finder/High.**
- **[Medium]** Closer Monitor crashes on a NaN `mlb_id` — `api/services/closers_service.py:84` — `int(mlb_id) if mlb_id is not None else 0` doesn't catch NaN (`nan is not None` is True). — **Evidence (repro):** `int(float('nan'))` → `ValueError`. Latent (live `/api/closers` currently 200; no per-row guard in `_to_entry`). — **Fix:** `int(mlb_id) if pd.notna(mlb_id) else 0`. — **Confidence: High** (repro; flagged by 2 finders).
- **[Low]** Several services emit raw floats without NaN/inf guards — `api/services/streaming_service.py:55-57`, `api/services/trade_service.py` (multiple). **[Minor]** matchup category parse uses `float(str(x).replace("-","0"))` which **flips signs** (`"-5"`→`"05"`→`5.0`) and corrupts hyphenated values — `api/services/matchup_service.py:647,651` — currently masked because category totals are non-negative. **[Minor]** NaN/inf-guard inconsistency across services; cross-router `team_name` required on some endpoints — `api/routers/{team,free_agents,...}.py`. — **Confidence: finder.**

### E. Auth / billing / multi-tenancy (`api/`)

- **[Medium]** `ViewerContext.effective_team` falls back to the **client-supplied `team_name` query param** for an authenticated-but-unassigned user — `api/tenancy.py:60-63` — `return self.team_name or fallback`. When Clerk is dormant this is the intended byte-for-byte behavior, but Clerk is **live in production**, so a logged-in user with no team assignment can read any team's personalized data by passing `?team_name=…`. — **Fix:** once `clerk_configured()`, drop the query-param fallback (return `None`, i.e. 401/empty) instead of trusting the client value. — **Confidence: code/High** (design tension — mitigated only after all 12 users are assigned).
- **[Low]** Clerk JWT verify options — `api/auth.py:135-143` — confirm `verify_aud`/issuer pinning. **[Low]** Stripe webhook state machine ordering — `api/services/billing_service.py:159-202`. — **Confidence: finder.**

### F. Bubba AI assistant (`src/ai/`, `api/services/chat_service.py`)

- **[Medium]** Bubba doesn't resolve the **viewer's** team — `api/routers/chat.py:37-81` + `chat_service` — `/send`/`/send-stream` are correctly per-user gated (`require_app_user`, identity offset), but no `ViewerContext` is threaded, so team-scoped answers fall back to the global `is_user_team` (Team Hickey). Every user's Bubba would give Team-Hickey-centric advice. — **Fix:** thread the resolved viewer team into the chat tool context. — **Confidence: finder/Med (code-corroborated: no viewer-team passed).**
- **[Medium]** Bubba spend metering / daily cap accuracy — `src/ai/router.py:114-117` (`price_per_token`) — pricing-table gaps could under/over-count managed spend. **[Low]** BYOK key-store observability (`src/ai/keys.py:67-72`); `request_refresh` tool gating in the API context (`src/ai/tools.py:112-134`). — **Confidence: finder.**

### G. Web frontend (`web/`)

- **[High]** The canonical player card is **100% mock on the live product** — `web/src/lib/player-detail.ts:235-292` — `getPlayerDetail` returns `BASE[mlbId] ?? fallback(m)` (hardcoded season lines, `rankOverall:120`, `rosteredBy:"Team Hickey"`, history member “Team Hickey”). Every `PlayerDialog` shows fabricated stats; there is **no player-detail API endpoint** to wire it to. — **Evidence (code + live):** clicking any player on the live site shows pre-baked numbers. — **Fix:** add a player-detail endpoint and replace the mock; until then it's a known not-yet-wired gap with high user-facing impact for a launching product. — **Confidence: High** (code).
- **[Medium]** Viewer identity hardcoded to “Team Hickey” across live paths — `web/src/lib/viewer-team.ts:5`, `web/src/lib/data.ts:102,127`, `matchup-data.ts:225`, `optimizer-data.ts`, `player-detail.ts:281` — the “YOU” marker and mock fallbacks default to Team Hickey (the documented remaining-M3 frontend cleanup; the frontend also still sends `?team_name=Team+Hickey`). — **Confidence: live + finder/High.**
- **[Medium]** Closer adapter discards the live security field — `web/src/lib/api/adapters.ts:638-653` (`apiClosersToData`) — contract drift between adapter and `/api/closers`. **[Low]** Generated API types stale vs `api/openapi.json` (last regen `35c4139`) — `web/src/lib/api/generated.ts`. — **Confidence: finder/High.**
- **[Low]** Four-state machine mixes error vs empty — `web/src/lib/optimizer-data.ts:122-137`; live failure silently renders mock data (`matchup-data.ts:232`, `trades-data.ts:138`, `data.ts`); trade-builder live-eval failure is a silent dead-end (`web/src/components/trades/BuildPanel.tsx:41-51`). **[Minor]** raw ISO date in streaming header (`web/src/app/streaming/page.tsx:37`); matchup hardcoded 7-cell fallback (`web/src/app/matchup/page.tsx:338`); `getPlayerDetail` recomputed every render (`PlayerDialog.tsx:28`); dormant TopBar account menu hardcoded identity (`TopBar.tsx:162-196`). — **Confidence: finder.**

### H. Live deploy (Railway + Vercel) — observed only on the deployed hosts

- **[Medium]** Private league data is readable **unauthenticated** — 7 routers have no auth dependency: `api/routers/{closers,compare,databank,leaders,players,standings,streaming}.py` (`players` also serves `/api/league/rosters` + `/api/players/search`). — **Evidence (live):** `GET /api/league/rosters` (no token) returns **all 12 teams' full rosters + manager names** (e.g. manager “elias”); `GET /api/standings` returns all team names/standings. Contradicts the documented “reads require login” decision (only personalized routers got `require_viewer_context`). — **Fix:** add an auth-only (`require_login`) dependency to the league-wide private reads (standings, league/rosters) at minimum. — **Confidence: High** (live + code).
- **[Medium]** Production frontend loads Clerk with **development keys** — `web` Clerk config — **Evidence (live):** console warns *“Clerk has been loaded with development keys … should not be used when deploying to production”*; JS loads from `meet-firefly-38.clerk.accounts.dev`. Dev instances have strict usage limits. Acceptable for a 12-friend beta; must move to production keys before public/monetized launch. — **Confidence: High** (live).
- **[Low]** `/api/leaders/overall` value saturates the elite tier to 100 — `api/services/leaders_overall_service.py:50` — `round(max(0,min(100,(fval+10)/20*100)))`; any sum-of-z ≥ 10 clamps to 100. **Live:** top 7 overall leaders all show `value=100` (ranking order still correct; display undifferentiated). — **Confidence: High** (live + code).
- **[Low]** Standings page stale copy — “Playoff odds need the season-simulation endpoint (a backend add) — not available on live data yet” — the `/api/playoff-odds` endpoint **exists** (returns 401 when logged out). — **Confidence: High** (live).
- **[Minor]** Document `<title>` is “HEATER — My Team” on every route (Standings/Research/etc.). Cross-page mock inconsistency (Team landing “Week 12, 3-7-1” vs Matchup “Week 13, 4-7-1”) — tied to the known Team-Hickey mock state. — **Confidence: live.**

### I. Tests / CI

- **[Low]** Local ruff (newer) would reformat `tests/test_ui_shared_components.py:310` (a redundant paren); CI is GREEN, so CI's pinned ruff doesn't flag it — latent **version skew** (`.github/workflows/ci.yml`, no ruff pin in `requirements.txt`). **[Low]** Optional-dependency tests (XGBoost/PyMC) `skipif` never execute in CI → those code paths are unverified by CI. **[Low]** A couple structural-invariant guards under-cover (hardcoded page/script allowlists). **[Minor]** `tests/test_ai_sql_tool_readonly.py:60-67` test name/comment over-claims vs the body. — **Confidence: finder/High.**

### J. Engine math / Monte Carlo / valuation

- **[Low]** Antithetic-variate sampling detail — `src/engine/monte_carlo/trade_simulator.py:94-115`. **[Minor]** positional-scarcity NaN robustness (`src/valuation.py:747-751`); degenerate rate-stat sentinel in `_roster_category_totals` (`src/in_season.py:303-309`); Kalman diagnostic (`src/engine/signals/kalman.py:220-222`). — No High/Critical math/sign bugs found (inverse-stat signs, paired seeds, SGP all correct). — **Confidence: finder.**

---

## VERIFIED CLEAN (checked and found correct)

- **Test/build gates:** full `pytest` 6813/0; `ruff check` clean; `openapi` snapshot test passes; web `tsc`+`lint`+`build` all clean (19 routes prerender); CI on master green.
- **Frontend render layer:** all **14 routes** render with no JS crashes / error boundaries (Team, Optimizer, Streaming, Probables, Hitter-Matchups, Closers, Matchup, Standings, Punt, Trades, Players, Research, Databank, Draft). Only console output = the Clerk dev-keys warning + expected 401s on personalized endpoints.
- **Good-path wiring works end-to-end:** Research renders real live leaders; Closers renders real data; `/api/players/search?q=trout` returns real Mike Trout (mlb_id 545361).
- **Auth gates functional:** personalized reads, both write endpoints, and Pro endpoints all return **401** unauthenticated; the Clerk login-gate works; no endpoint 500'd.
- **Security:** `src/ai/sql_tool.run_read_only_sql` is robustly defended — read-only driver (`mode=ro`), SELECT/WITH-only, single-statement, banned-keyword + excluded-table (keys/auth/users) filters, 500-row cap, never-raises, and connects only to `draft_tool.db` (not `api_state.db`). Only a **Minor** over-broad substring filter (false-positive over-blocking, not a bypass).
- **CORS:** explicit env allowlist (never `*` with credentials); the live host **rejects** a forged `Origin: evil.example` (preflight 400, no `Access-Control-Allow-Origin` echoed).
- **Dual-DB seam:** all 5 `api/stores/*` (user, subscription, prompt, membership, league) use `data/api_state.db` (`HEATER_API_DB_PATH`), **never** `draft_tool.db`. The data-correctness finder found **0** issues (per-user isolation, viewer-team resolution algorithm, DNA-collision dedup, ledger owner-scoping all correct).
- **Secret hygiene:** no hardcoded `sk_live`/`pk_live`/`whsec`/AWS/PEM/password literals in `api/` or `src/`.
- **`standings_service.py` mapping logic itself** is faithful + NaN-safe (the records bug is upstream sync + missing `league_records` join, not the mapper).

---

## Method note & residual risk

- A 13-agent finder workflow ran read-only across all surfaces (12/13 surfaces completed before this report; the 13th, type-design/comments, was the lowest-stakes). Each **High** and the load-bearing **Mediums** were independently reproduced (venv repros for the NaN crashes; live probes for the records/auth/Clerk findings; code traces for the rest). Several bugs were found by **multiple independent finders** (standings ×2 + live, closers-NaN ×2, matchup-sign ×2) — strong cross-validation.
- **Pattern, not just instances:** the Low/Minor tier is dominated by two systemic patterns — (1) **NaN/inf-safety gaps** where `or 0` / `is not None` / `int(float(...))` don't guard pandas NaN (the codebase has `_f`/`_is_hitter_safe` helpers that should be applied uniformly), and (2) the **viewer-team / “Team Hickey” not-yet-wired** state across web + Bubba + the tenancy query-param fallback (the documented remaining-M3 cleanup). Fixing the helper-application pattern and finishing the viewer-team wiring resolves most of the long tail.
- Nothing in CLAUDE.md's **Known Design Choices** was re-flagged as a bug. The dormant-when-env-unset behaviors, strangler-fig Streamlit, paused B2–B5 infra, and documented deferrals (matchup live stats, champ odds) were excluded.
