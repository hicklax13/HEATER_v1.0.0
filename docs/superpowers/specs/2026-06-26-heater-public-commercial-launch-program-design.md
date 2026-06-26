# HEATER Public Commercial Launch — Master Program Design

**Date:** 2026-06-26
**Status:** Approved design (brainstorming complete) — pending owner review of this written spec, then per-phase planning.
**Owner:** Connor (solo operator + Claude until revenue funds hires)
**Track:** CEO / Product Officer (this track) reconciled with the CMO design track.
**Supersedes for launch purposes:** the M0–M6 migration roadmap framing and the A/B/C/D sub-project split — both are folded into the 16 phases below. The Streamlit app remains a strangler-fig fallback until Phase 15.

---

## 1. Purpose

Take HEATER from a working private friends-beta (React on Vercel + FastAPI on Railway + Clerk + dormant Stripe over the unchanged Python engines) to a **public, monetized, self-service SaaS product that strangers can sign up for and pay to use** — at a quality bar high enough that we would not be ashamed to charge for it.

This document is the single master program. It folds the **union** of three plan-only audits — `codex-critiques.md` (overall 51/100), `codex-backend-critiques.md` (54/100), `codex-ui-ux-critiques.md` (58/100) — plus the existing migration roadmap and the locked B4 multi-tenancy design, into **one sequenced set of 16 phases**, each of which becomes its own implementation plan file.

This document is plan-level only. It does not authorize implementation by itself. Each phase is built only after its own plan is written and (at owner-gated phases) the owner approves.

---

## 2. Locked decisions (resolved with the owner, 2026-06-26)

1. **Launch bar — Full Codex 100/100 before any public dollar.** No public signup or charge until every phase's internal exit gate passes. The bar is the Codex "100/100" standard for every audit category.
2. **External audits — internal-ready now, paid stamps later.** Build and verify everything to a true-100/100, audit-ready state using the hardest internal/automated verification available. The four paid external certifications (security penetration test, independent statistical review, independent WCAG accessibility audit, lawyer + commercial data licenses) are an **owner-arranged step before the public flip**, not gates on the build. Until arranged, the product is "audit-ready, not yet certified."
3. **Format scope — all major formats and providers.** Support H2H Categories **and** Rotisserie **and** Points scoring, across Yahoo, ESPN, CBS, and Sleeper, plus a validated manual import. This is the widest market and the largest version of the program; the magnitude is accepted, not re-litigated.
4. **Pricing — tiered plus usage-aware.** Free / Pro / Premium subscription tiers **with** usage metering (connected-league count and Bubba AI usage) as tier limits and/or metered add-ons. Exact price points are validated in Phase 15; the billing and entitlement system is built to this model.

---

## 3. Governing standards

### 3.1 Definition of "internal 100/100-ready" (per category)

A category is internally ready when all of the following hold (the external-stamp items are tracked separately in §6):

- Every mandatory capability for that category (per the Codex 100/100 standards) is implemented.
- Automated tests cover success, invalid input, degraded dependency, abusive input, and recovery behavior.
- No fallback fabricates data or presents an operational failure as a valid analytical result.
- Every material limitation that can change a user's decision is surfaced in the product, not buried in code.
- No probability/confidence/"calibrated"/"accurate" claim is shown in the UI without an approved internal validation result behind it.
- The phase's evidence-registry rows (§3.5) are populated and reproducible by re-running the listed verification commands.

### 3.2 The per-phase exit-gate pattern

Every phase exits in two parts:

- **Internal gate (I build + verify):** code complete, tests green, evidence-registry rows populated, the phase's verification commands reproducible by a second person. This is what unlocks the *next* phase.
- **Deferred external certification (owner arranges before public launch):** a checklist of paid third-party stamps this phase contributes to the §6 register. These do **not** block subsequent phases; they block only the final public flip in Phase 15.

### 3.3 Sequencing rules (inherited from Codex, adapted)

1. Legal and data-rights foundations precede commercialization work that depends on them.
2. PostgreSQL precedes full multi-tenancy.
3. Tenant-isolation proof precedes any wider release ring.
4. Per-user provider authorization precedes enabling public league writes.
5. Point-in-time historical datasets precede any calibration or accuracy claim.
6. Prediction validation precedes probability/confidence/accuracy marketing language.
7. Recommendation evidence and constraints precede any future automated execution.
8. Any phase that changes the live data path requires a rollback plan and explicit owner approval.
9. Bias every decision toward managed infrastructure and solo-operator maintainability.
10. The Streamlit app stays available as a fallback until Phase 15 cutover gates pass.

### 3.4 Release rings (the launch funnel)

Local dev → CI → internal staging → owner-only production → existing 12-user league → invite-only external beta → limited public launch → general availability. Each phase plan names the highest ring it unlocks.

### 3.5 Evidence registry

Phase 0 creates a machine-readable registry (one row per requirement) with: requirement ID, category, description, status, owning subsystem, verification command, production metric, evidence path, external-review requirement, blocking release ring, last-verified timestamp, score contribution. Every later phase updates it. This registry is how "are we at 100/100?" is answered objectively rather than by assertion.

### 3.6 Managed-infrastructure default

Vercel (frontend), Railway (FastAPI + workers), managed PostgreSQL, managed Redis, Arq workers, S3-compatible object storage, Sentry (errors/traces), PostHog (privacy-controlled analytics + flags), Clerk (production identity), Stripe (billing/tax/portal), GitHub Actions (gates). Every provider must have a documented export/exit path; core HEATER domain logic stays in the repo.

### 3.7 Preserve the engine

Do not rewrite working analytical logic to change infrastructure. Put typed boundaries around the Python engines, build point-in-time validation around them, and move heavy execution behind durable workers. The Combustion design language and the engine are preserved through the whole program.

---

## 4. Program structure

- **One master spec** (this document) → **one plan file per phase** under `docs/superpowers/plans/`, named `2026-06-..-heater-launch-phaseNN-<slug>.md`.
- Each plan decomposes into small TDD tasks and is executed with parallel agents, pausing at owner-gates.
- **Lanes** tag which competency a phase mostly draws on: Platform, Backend, Engine, Data, Integrations, Frontend, AI, Legal, GTM. Lanes are for skill/agent routing, not separate documents.
- Hierarchy (do not conflate levels): **Program → phase → plan → task.**

---

## 5. The 16 phases

Legend: **🔒 owner-gated** = needs owner money, a real third-party account/app, or touches the live data path (requires explicit owner go + a rollback plan).

### Phase 0 — Rebaseline, evidence registry & API contract foundation
**Lane:** Platform · **Depends:** none · **Unlocks:** all later phases
**Objective:** One authoritative baseline; separate historical findings from current defects; convert the 100/100 standards into machine-checkable gates; and lay the cross-cutting API contracts every later phase builds on.
**Scope:** Freeze the audit target (master SHA, deploy IDs, env-var names, tool versions, OpenAPI export, route inventories, test counts, schema revision, provider scopes, feature flags, model/constant versions). Re-audit current `master` (full Python + API suites, ruff, OpenAPI snapshot, tsc, eslint, Next build, route smoke, prod probes) and classify every historical `Bugs_June26.md` / `Outstanding_June26.md` finding as fixed/open/obsolete/accepted/unverifiable. Build the evidence registry (§3.5) and structural quality guards that fail when a tenant table lacks scope, a private route lacks auth, a recommendation lacks evidence, a "calibrated" model lacks provenance, a live fetcher converts an error into mock data, OpenAPI and generated TS diverge, or docs claim a feature state that differs from generated inventory. **Establish the API contract foundations** (folding the backend audit's contract phase): a single versioned error envelope (`code`/`message`/`request_id`/`retryable`/`degraded`/`dependency`/`details`) so operational failures never masquerade as valid empty data; a correlation-ID middleware bound to logs/traces/jobs/audits; an OpenAPI bearer-security scheme + documented auth/scope/Pro/admin requirements on every operation (so generated clients infer auth correctly); an API versioning + compatibility policy; a mutation-idempotency framework contract; and an async-job contract shape (built out in Phase 9). These are light but cross-cutting, so they land first.
**Internal gate:** current-state report approved; every certification requirement has a registry row; the score is reproducible by a second person; the error envelope + correlation ID + OpenAPI security scheme are live and enforced by tests. Fix the one known failing `test_openapi_snapshot_is_current` on the work branch.
**Deferred external cert:** none.

### Phase 1 — Legal, data rights & privacy 🔒
**Lane:** Legal/Backend · **Depends:** 0 · **Unlocks:** commercialization
**Objective:** Remove the risk of charging for a product without rights to its data; define lawful handling of account, league, billing, and AI data.
**Scope:** Build the source inventory (Yahoo, MLB Stats API/MLBAM, FanGraphs, FantasyPros, Retrosheet, Statcast/Savant, Open-Meteo, Clerk, Stripe, and every AI provider) recording fields consumed, acquisition method, display/redistribution, commercial-use status, attribution, retention, rate limits. Resolve commercial rights per source (license, replace, or disable the dependent feature). Draft public policies (ToS, Privacy, Cookie/analytics, AUP, AI disclosure, subscription/refund terms, no-guarantee disclaimer, data-source attribution, deletion/export instructions, vulnerability disclosure). Classify data and define retention/encryption/access/export/deletion per class. Implement and test privacy operations (export, account deletion, league disconnect, OAuth revoke, AI-conversation deletion, attachment deletion, analytics opt-out, deletion-from-backups-after-retention).
**Internal gate:** every production source is approved/replaced/disabled in our metadata; policy drafts complete; privacy export+deletion workflows pass in staging; no feature relies on an unresolved source.
**Deferred external cert:** lawyer review of the public policies + sign-off on commercial data licensing.
**Owner-gate:** signing licenses / paying for legal review; deciding to drop or replace any unlicensable source.

### Phase 2 — PostgreSQL migration 🔒
**Lane:** Backend/Platform · **Depends:** 0 · **Unlocks:** Phase 3
**Objective:** Complete the dormant B2 work (B2.2–B2.5) so the live data path is PostgreSQL, with no behavior change to the engines.
**Scope:** Provision managed PostgreSQL (local + CI + prod). Port SQLite-specific SQL; route persistence through SQLAlchemy engines / repositories; verify date/timezone/boolean/JSON/case-insensitive-text/autoincrement/conflict/transaction semantics. Move API-store `CREATE TABLE` runtime DDL and `init_db` schema into Alembic as the single owner; add foreign keys, unique/check constraints, indexes, timezone-aware timestamps. Build repositories by bounded context (player, projection, league, roster, matchup, schedule, transaction, user, membership, subscription, conversation, recommendation, job, audit). Migrate data with export+checksum → staging load → parity/shadow reads → rehearsed cutover + rollback. Unify the split `draft_tool.db` / `api_state.db` stores.
**Internal gate:** all production paths use PostgreSQL; no runtime table creation; no personalized path uses direct SQLite; frozen-fixture parity between old and new; backup + restore drill passes; live app behaviorally equivalent.
**Deferred external cert:** none (covered by Phase 10 security + Phase 9 DR).
**Owner-gate:** provisioning managed PG; approving the live cutover window.

### Phase 3 — Tenant-native model & isolation 🔒
**Lane:** Platform/Backend · **Depends:** 2 · **Unlocks:** Phases 4, 5, ring 6
**Objective:** Make cross-tenant exposure structurally difficult and testably impossible; support multiple leagues per user.
**Scope:** Add `tenants`, `tenant_memberships`, `leagues`, `league_memberships`, `fantasy_teams`, `provider_connections`, `provider_credentials`, `provider_capabilities`, audit/privacy/entitlement/AI-usage tables. Put `tenant_id` on every tenant-owned row and `tenant_id`+`league_id` on every league-owned row. Replace name-as-identity with stable team IDs (names become display fields; support renames + duplicate names across leagues). Derive tenant context server-side from verified Clerk identity + membership — never from query/body team names, defaults, or env. Scoped repositories reject missing context. Add PostgreSQL row-level security on sensitive tables; tenant-scoped cache namespaces, object-storage prefixes, and (future) worker payloads. Build the multi-league lifecycle (list, connect, select active, switch without stale-cache leak, reconnect, disconnect, delete imported data). Run the full isolation test matrix (A-cannot-read/mutate-B, unscoped-cache, deleted-tenant, guessable-object-URL, evidence-snapshot cross-reference, billing-entitlement cross-apply).
**Internal gate:** no authenticated path uses a configured default league; cross-tenant adversarial suite passes; RLS enabled; multiple real test leagues coexist safely; existing 12-user league passes full regression; rollback demonstrated.
**Deferred external cert:** independent penetration test (cross-tenant scope) — Phase 10/15.
**Owner-gate:** live data-path change; approve cutover.

### Phase 4 — League-format generalization
**Lane:** Engine · **Depends:** 3 · **Unlocks:** Phases 6, 7
**Objective:** Make every engine accept arbitrary league configurations instead of the hardcoded 12-team, 6×6 H2H-Categories FourzynBurn assumptions, and add Rotisserie and Points scoring paradigms. (This phase is the direct consequence of the "all formats" decision; the Codex docs assume the existing single-format engine.)
**Scope:** Generalize `LeagueConfig` into a canonical per-league rules model (category set, inverse/rate flags, roster slots, team count, transaction limits, playoff structure, **scoring type**) sourced from each league's tenant record. Audit and remove every remaining hardcoded category list, team count, roster assumption, and inverse-stat set across `src/` engines; extend the structural guards (`test_no_hardcoded_categories_*`) to also guard team-count, roster, and scoring-type assumptions. Build the **Rotisserie** path (rank-sum standings, per-roto SGP denominators; optimizer/trade/standings/playoff adapt). Build the **Points** scoring engine (per-stat point weights; lineup optimization maximizes projected points; trade/FA/draft value expressed in points — a valuation path parallel to SGP). Per-format validation harness over synthetic + real league fixtures.
**Internal gate:** engines produce correct results across a corpus spanning all three scoring types and varied category/roster/team configs; no remaining hardcoded format assumption; guards extended and green.
**Deferred external cert:** none (validated quantitatively in Phase 7).

### Phase 5 — Provider connector platform 🔒
**Lane:** Integrations · **Depends:** 3 · **Unlocks:** public onboarding
**Objective:** Turn Yahoo-specific integration into a safe, provider-neutral public connector system across Yahoo/ESPN/CBS/Sleeper + manual import.
**Scope:** Define the canonical `LeagueConnector` interface (capabilities, authorize, refresh, revoke, discover_leagues, import settings/teams/rosters/standings/matchups/transactions, set_lineup, add_drop, health) returning typed canonical records + stable error codes. Implement per-provider connectors; advertise capability flags so the UI hides/explains unsupported actions (a connector may never claim a capability it lacks). Yahoo: per-user OAuth, least-privilege scopes, explicit write-scope consent, encrypted tokens, auto-refresh, revocation handling, reconnect flow, write rate-limit + idempotency, verify the selected team belongs to the authenticated user, read-only option. ESPN/CBS: approved partner API where available, else secure guided import. Sleeper: adapter-ready but baseball disabled until officially supported. Manual import: versioned templates, declared file types, malware scan, isolated-worker parse, normalized preview, validation + row-level correction, confirm-before-commit, artifact+checksum+parser-version, rollback, duplicate-application prevention.
**Internal gate:** public users can connect/reconnect without admin intervention; credentials isolated + revocable; unsupported operations impossible; Yahoo writes user-authorized + audited; at least Yahoo + one guided import path certified by recorded-fixture + sandbox tests.
**Deferred external cert:** provider terms/commercial-rights approval (ties to Phase 1).
**Owner-gate:** registering ESPN/CBS/Sleeper developer apps + Yahoo write-scope re-registration (Yahoo write may remain externally blocked — see Known Limitations).

### Phase 6 — Data quality, identity & lineage
**Lane:** Data · **Depends:** 4, 5 · **Unlocks:** Phase 7
**Objective:** Make every recommendation traceable to reliable, versioned, point-in-time data.
**Scope:** Canonical player identity graph (immutable internal IDs ↔ MLB/Yahoo/provider IDs, name variants, team/position history, batter/pitcher + split identities, effective-date ranges) with deterministic matching, confidence levels, a manual-resolution quarantine queue, and collision protection. (Closes the local-DB gap: ~73% of player rows lack `mlb_id`, ~17% lack a team.) Ingestion quality pipeline on every refresh (schema/type/range/null/referential/duplicate/temporal checks → provider reconciliation → freshness → immutable snapshot → quality score → alerts). Documented reconciliation precedence per field. Attach `observed_at`/`effective_at`/`refreshed_at`/`source`/`source_snapshot_id`/`quality_status`/`freshness_status`/`warnings` to every frontend data payload; the UI visibly distinguishes current/stale/partial/unavailable/conflicting. Immutable, replayable point-in-time snapshots of projections, rosters, settings, standings, matchups, transactions, injuries, probables, model inputs, and recommendation outputs.
**Internal gate:** identity coverage target met for fantasy-relevant active players; quality failures block publication; every recommendation input traceable to a snapshot; data-quality dashboard operational; no unsupported platform advertised.
**Deferred external cert:** none.

### Phase 7 — Prediction validation & calibration
**Lane:** Engine · **Depends:** 6 · **Unlocks:** any accuracy claim
**Objective:** Replace impressive-looking output with empirically defensible output, across all three scoring formats.
**Scope:** Build versioned point-in-time historical datasets (≥3 completed seasons where legally/practically available) with `available_at` proof. Repair the optimizer backtest to **replay the actual optimizer** against feasible baselines (start-all-active, consensus-rank, recent-value, no-transaction, random-legal) joined by canonical ID + effective date (never DataFrame row order). Validate projections (MAE/RMSE/bias/rank-corr/interval coverage, segmented by position/sample/injury/season-phase). Validate probability models (Brier/log-loss/calibration slope+intercept/ECE/sharpness by band). Validate streaming, trade, and draft recommendations against their baselines. Calibrate every one of the 30 registered constants (all currently `calibrated: false`): assign target, training data, method, bounds; fit on train, select on calibration, evaluate once on holdout; record provenance + CIs; mark calibrated only after approval, else label "heuristic." Model registry + model cards (intended use, exclusions, weaknesses, retraining triggers). Internal statistical-methodology review (leakage, baselines, power, calibration, decision utility).
**Internal gate:** no constant mislabeled calibrated; backtests run the real decision systems; holdout metrics meet certification thresholds with CIs; model cards complete; weak-data cases abstain or downgrade confidence.
**Deferred external cert:** independent statistical reviewer sign-off on methodology + any public accuracy claims.

### Phase 8 — Recommendation evidence & trust layer
**Lane:** Backend/Frontend · **Depends:** 7 · **Unlocks:** Phase 12, 14
**Objective:** Make every recommendation inspectable, reproducible, and outcome-measured; prevent overconfident advice.
**Scope:** Canonical recommendation service that all pages **and Bubba** call (frontend/AI may format explanations but never recompute the recommended action). Immutable recommendation records (ID, tenant/league/user, type, created/expiry, input-snapshot IDs, model version, constraints, proposed action, expected category impact, uncertainty, confidence band, alternatives + rejection reasons, assumptions, warnings, explanation, outcome status, feedback). Evidence cards on every recommendation surface. Abstention policy (stale data, unresolved identity, incomplete settings, failed quality rules, statistically indistinguishable alternatives, provider can't execute, excessive projection uncertainty, model out of intended use). Conflict detection + reconciliation across tools (e.g., optimizer starts an unavailable player; Bubba contradicts canonical; transaction exceeds weekly/season limits). Outcome measurement (viewed → expanded → saved → accepted/rejected → executed → observed → realized utility → user rating); acceptance alone is never treated as correctness. Trust dashboard.
**Internal gate:** every recommendation has immutable evidence; replay-equivalence target met; contradiction detection passes; trust dashboard operational; a second person can reproduce sampled recommendations.
**Deferred external cert:** none.

### Phase 9 — Reliability, workers & operations 🔒
**Lane:** Platform · **Depends:** 2 (parallelizable after 3) · **Unlocks:** rings 6–8
**Objective:** Remove expensive/failure-prone work from request paths; establish production operations (completes B3).
**Scope:** Managed Redis + Arq workers + scheduler (scheduler enqueues only). Move to durable jobs: provider refreshes, historical imports, model calibration, Monte Carlo/trade/playoff/draft simulations, full FA ranking, large exports, attachment parsing, AI deep research, snapshot generation, reconciliation. Job API (`POST/GET/cancel /api/v1/jobs`, SSE/poll progress, stable terminal states, idempotency keys, dead-letter queue, replay). Caching policy with tenant/league/season/config/snapshot/model-version-scoped keys + invalidation. Upstream resilience per provider (timeout/retry/backoff/jitter/circuit-breaker/stale-fallback/alert/user-message). Observability: Sentry, OpenTelemetry traces (HTTP/DB/queue/provider/Stripe/Clerk/AI), RED + USE metrics, request/correlation IDs, structured redacted logs, readiness + dependency-health endpoints (not just static `/healthz`). SLOs + error budgets + actionable alerts mapped to runbooks. Encrypted backups, point-in-time recovery, quarterly restore drill, documented failover. Load + chaos testing (2× forecast traffic + Redis/worker/PG/Yahoo/MLB/AI/Stripe failure injection).
**Internal gate:** web requests contain no unbounded compute; failed jobs don't lose work; duplicate jobs don't duplicate side effects; SLO dashboards + alerts live; load/chaos + restore drills pass; 30-day SLO observation begins.
**Deferred external cert:** none.
**Owner-gate:** provisioning Redis + worker services; live infra changes.

### Phase 10 — Security & privacy enforcement
**Lane:** Platform · **Depends:** 1, 3, 5 · **Unlocks:** ring 7
**Objective:** Harden every public attack surface; prove isolation and credential safety to an audit-ready state.
**Scope:** Centralized authorization policy engine (principal, tenant, league role, team ownership, subscription entitlement, provider capability, admin/support, worker, webhook, AI tool) — every permission server-enforced + negatively tested. Production Clerk keys + issuer/audience verification + session expiry/revocation + MFA support + brute-force/suspicious-login protection. Rate limits by IP/user/tenant/route-class/cost-unit; quotas + concurrency limits on expensive analytical/AI endpoints; request-size/upload/pagination/query-complexity bounds; strict CORS + security headers + idempotency. Credential envelope-encryption with a separate key-encryption key, rotation, log/analytics redaction, immediate revocation on disconnect, decryption-failure alerts. Import/attachment security (store outside executable paths, malware scan, archive-bomb limits, isolated-worker parse, treat text as untrusted, never follow embedded instructions). AI security (tenant-scoped read-only tools with table allowlists, prompt-injection tests, output encoding, attachment isolation, no secrets in prompts, configurable retention). Secure SDLC in CI (secret/dependency/SAST/container/IaC/license scanning, SBOM, authorization tests). Internal threat models + an internal pen-test pass that prepares for the external test.
**Internal gate:** zero Critical/High in our internal security suite + threat-model review; production keys/secrets verified; privacy workflows pass; incident-response tabletop passes.
**Deferred external cert:** independent penetration test (auth, authz, tenant isolation, APIs, billing, OAuth, imports/attachments, AI prompt injection, admin, deploy config) — owner-arranged in Phase 15.

### Phase 11 — Billing, entitlements & customer lifecycle 🔒
**Lane:** Backend/Frontend · **Depends:** 3, 10 · **Unlocks:** ring 7
**Objective:** Make payment state reliable and complete the self-service customer lifecycle for the Free/Pro/Premium + metering model.
**Scope:** Subscription state machine (free, trialing, active, past_due, grace, paused, canceled, refunded, disputed, incomplete, expired) with documented Stripe-event transitions. Webhook integrity (signature verify, store event ID, **deduplicate**, creation-time ordering, replay, processing-outcome record, repeated-failure alert, idempotent entitlement updates). Entitlement service of explicit capabilities (feature keys, usage allowance, **connected-league limit**, **AI allowance**, renewal, grace, denial reason) — not just free-vs-Pro. Tiered plans (Free/Pro/Premium) + usage metering (leagues + AI) enforced server-side, fail-closed when metering is unavailable (BYOK AI may continue independently). Customer portal (plan, trial, renewal, invoices, payment method, upgrade/downgrade/cancel/reactivate, support). Daily financial reconciliation (Stripe customers vs users, subscriptions vs entitlements, processed webhooks vs Stripe events, AI usage vs allowance ledger, refunds/disputes, orphans/duplicates). Immutable billing audit ledger.
**Internal gate:** Stripe certification scenarios + webhook-replay tests pass; daily reconciliation clean; portal works; AI cost limits unbypassable; pricing disclosure drafted.
**Deferred external cert:** live-mode end-to-end purchase certification (owner runs a real card pre-launch).
**Owner-gate:** Stripe live-mode keys, tax config, final price points, product/price creation.

### Phase 12 — Self-service onboarding, adoption & support
**Lane:** Frontend/Backend · **Depends:** 3, 5, 8, 11 · **Unlocks:** ring 6 (invite beta)
**Objective:** Remove admin dependency; lead a stranger to a useful first result; make support sustainable solo.
**Scope:** Onboarding flow (sign up → accept terms → choose live-connect / guided-import / labeled-demo → select provider → authenticate/upload → select league → confirm team → validate settings → review freshness/gaps → initial refresh → first personalized recommendation → next actions), resumable. Distinct recovery states with stable reason codes (OAuth denied/expired, league/team not found, team unlinked, unsupported format, missing settings, import failure, provider outage, refresh timeout, insufficient data, subscription required, account conflict). Product education (contextual metric definitions, evidence/confidence/freshness explanations, disclaimer, glossary, help center, status-page link). Lifecycle comms (welcome, connection success/failure, refresh failure, trial-ending, payment failure, cancellation, security notice, export-ready, deletion-complete). Support ops (in-app form, secret-free diagnostic bundle, stable issue categories, visible ticket ID, queue, response targets, known-issues page, incident banners). PostHog product analytics (onboarding step, connector success, first recommendation, acceptance, return, upgrade, churn, dead-end) — never raw private league data to analytics.
**Internal gate:** unassisted-onboarding + activation targets met in testing; recovery states tested; support runbook operational; analytics contain no prohibited data.
**Deferred external cert:** none.

### Phase 13 — Frontend UI/UX & accessibility
**Lane:** Frontend · **Depends:** 8, 12 · **Unlocks:** ring 7
**Objective:** Make the product usable, accessible, honest, and fast across devices while preserving the Combustion identity (this folds the entire `codex-ui-ux-critiques.md` program).
**Scope:** Information architecture + navigation redesign (group 18 routes into 5–7 user-job groups; no clipping ≥1024px; real league switcher + connection/freshness status; grouped mobile nav). Responsive system with zero page-level overflow at 320/360/390/430/768/1024/1280/1440/1920 (fix Optimizer 742px + Players 782px mobile overflow; replace desktop-table-on-mobile with priority columns / card modes / row expansion; 44px touch targets; safe Bubba dock). Semantic type scale (no meaningful text <12px; body 16px on mobile) + accessible color tokens (every normal-text color ≥4.5:1; split brand vs status vs data colors). Documented design-system package + Storybook + visual regression. Page archetypes (dashboard / decision tool / monitoring / research / setup / commercial) each with one primary action. One visualization grammar with uncertainty + accessible alternatives. State/feedback/recovery system (route identity preserved in every state; route-specific skeletons; error taxonomy; recovery actions). Interaction safety (action hierarchy, transaction preview before Add/Drop/Lineup-Set, confirmation/undo, focus management). WCAG 2.2 AA to an audit-ready state (axe component+route tests, keyboard, screen-reader, 200%/400% zoom, reduced motion). Performance (server-rendered metadata — kills the route-title bug; route JS/image/font budgets; lazy-load Bubba + heavy dialogs; Core Web Vitals good at p75). A real frontend test suite (unit, component, Playwright E2E, axe, visual regression, responsive-overflow gate) wired into CI — `web/` currently has none.
**Internal gate:** zero horizontal overflow at supported widths; zero clipped global nav; internal axe + keyboard + zoom passes with no Critical/High; CWV good at p75; frontend CI checks block merges.
**Deferred external cert:** independent accessibility (WCAG 2.2 AA) audit — owner-arranged in Phase 15.

### Phase 14 — AI (Bubba) grounding, safety & evaluation
**Lane:** AI · **Depends:** 8, 10, 13 · **Unlocks:** ring 7
**Objective:** Make Bubba contextual, grounded, tenant-safe, and measurably useful.
**Scope:** Bubba uses the canonical HEATER recommendation/data services (Phase 8) instead of rebuilding domain logic in prompts; tool access tenant-scoped + read-only unless separately authorized. League-specific factual claims grounded in tool results, with citations to source time, model version, and recommendation evidence. Prompt-injection + malicious-attachment red-team test suite (100% of injection fixtures fail to override authorization; 100% of malicious attachments stay non-executable; zero cross-tenant disclosure). Continuous evaluation system: tool-call schema validity, tenant-context correctness, unsupported-claim rate, evidence coverage, recommendation-consistency with canonical engines, judged usefulness, cost/latency/safety observable by provider/model/tenant/feature. Fail-closed daily per-user + global cost limits (reconciled with Phase 11). Frontend Bubba UX redesign per Phase 13 (resizable desktop dock / mobile full-screen sheet, context header, route-aware prompts, typed failure states).
**Internal gate:** internal eval-set thresholds met (schema validity, tenant correctness, evidence coverage, consistency, usefulness); red-team suite passes; cost limits unbypassable.
**Deferred external cert:** none (the AI red-team can be added to the external pen-test scope at owner discretion).

### Phase 15 — Go-to-market & public launch 🔒
**Lane:** GTM/CEO · **Depends:** all (0–14) · **Unlocks:** rings 7–8 (limited public → GA)
**Objective:** Convert a certified product into a public, paying business and retire the Streamlit fallback.
**Scope:** Public marketing landing page (outcome promise, product proof, methodology proof, supported providers, pricing, FAQ, security/privacy, CTA) on a focused commercial shell. SEO. Final pricing/packaging validation + price points for the Free/Pro/Premium + metering model. **Arrange and pass the four deferred external certifications** (security pen-test, statistical review, accessibility audit, legal/licensing sign-off) — the §6 register must be clear. Progress the release rings: owner-prod → existing 12-league regression → invite-only external beta → limited public launch → GA, each with the evidence required to advance. Strangler-fig cutover: dual-run, migrate, then retire the Streamlit app once parity + cutover gates pass.
**Internal gate:** purchase works on desktop + mobile; all prior phases' internal gates green; ring criteria met for each step.
**Deferred external cert → now required:** all four external certifications complete and recorded in the evidence registry. **This is the gate that allows the first public dollar.**
**Owner-gate:** booking/paying external auditors; flipping Stripe live + public signup; the GA decision.

---

## 6. Deferred external-certification register

These are the paid third-party stamps the owner arranges before the Phase 15 public flip. They do not block the build.

| Cert | Contributed by | Arranged in |
|---|---|---|
| Lawyer review of ToS/Privacy/AUP + commercial data licenses | Phase 1 | Phase 1 draft → Phase 15 sign-off |
| Independent statistical-methodology review + accuracy-claim sign-off | Phase 7 | Phase 15 |
| Independent penetration test (incl. tenant isolation, billing, OAuth, AI injection) | Phases 3, 10 | Phase 15 |
| Independent WCAG 2.2 AA accessibility audit | Phase 13 | Phase 15 |
| Stripe live-mode end-to-end purchase certification | Phase 11 | Phase 15 |

Until all are recorded clear, public status is "audit-ready, not yet certified" and no public charge occurs.

---

## 7. Out of scope / non-goals

- Automated roster execution beyond user-confirmed Yahoo writes (recommendations remain advisory; Bubba stays read-only on the league).
- Native mobile apps (responsive web only).
- Sports other than MLB fantasy baseball.
- Real-money DFS / gambling features.
- Replacing or rewriting the proven Python analytical engines (we wrap, validate, and operationalize them).

## 8. Known limitations carried in

- **Yahoo write scope (`fspt-w`) is not self-serve available** — Yahoo's console shows only "Read" for new and existing apps. Phase 5 builds the per-user write path and graceful-confirm UX, but live writes may remain blocked externally; HEATER stays a read-only advisor until/unless Yahoo grants write. This is an external limit, not a build gap.
- **FanGraphs scraping returns 403 from datacenter IPs** — the Marcel local-projection fallback already covers this; Phase 1 must resolve commercial rights regardless of access method.

## 9. Honest magnitude note

This is the largest version of the program the owner could have chosen: full Codex 100/100, all three scoring formats, all four providers, tiered + metered billing. Realistically it is many months of work even with parallel AI agents, and several phases (1, 2, 3, 5, 9, 11, 15) are owner-gated on money, third-party accounts, or live-data cutovers. The phases are ordered so that value and risk-reduction compound: foundations and isolation first, then the engine/data/validation core, then commerce/experience, then launch. Each phase is independently reviewable and leaves the product in a coherent state.

## 10. Immediate next steps

1. Owner reviews this spec.
2. On approval, invoke **writing-plans** to turn **Phase 0** into a detailed TDD plan file, then proceed phase by phase.
3. Execute with parallel agents, pausing at each 🔒 owner-gate for explicit go + a rollback plan.
