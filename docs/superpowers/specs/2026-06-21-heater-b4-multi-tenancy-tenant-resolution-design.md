# HEATER — B4: Multi-tenancy + tenant-resolution auth + per-user team resolution

> **Date:** 2026-06-21 · **Lane:** CEO / Platform (data layer + execution model + multi-tenancy) · **Status:** Draft for owner review
> **Milestone:** M4 (the public prerequisite) — with its **single-league subset pulled forward as an M3 beta prerequisite**.
> **Pairs with / does not re-specify:** the B foundation design (`2026-06-16-heater-backend-api-foundation-design.md`, §4.2/§4.4/§4.5 + the B4 row of §5), the migration roadmap (`2026-06-19-heater-migration-roadmap.md`, M3 hidden-prereq + M4), and the Clerk-auth wiring plan (`2026-06-19-heater-backend-clerk-auth-wiring-plan.md`, which owns the *authentication* half and explicitly defers *this* — the multi-tenancy / who→team half — to a separate B4 plan). This spec is that plan's design.
> **Owner decisions baked in (2026-06-21):** design the **full multi-league** model now (not just the beta subset); the **beta links each of the 12 users to their team by admin assignment** (the proven Streamlit mechanism).

---

## 1. Why this exists

Every personalized API endpoint is hardcoded to one team. The viewer's team arrives as a **client-supplied query parameter** (`GET /api/me/team?team_name=Team%20Hickey`, and the same on `/api/matchup`, `/api/lineup/optimize`, `/api/punt`, `/api/free-agents`, `/api/playoff-odds`, `/api/trade-finder`), and the frontend hardcodes `"Team Hickey"`. Two problems:

1. **No authorization.** The backend trusts whatever team name the client asks for — any caller can read any team's data by changing the string. There is no "this is *my* team, tied to *my* login."
2. **Single-league.** The whole baseball dataset (`draft_tool.db`) is one league (Yahoo `game_key 469`, FourzynBurn). A public product needs every user to connect *their own* league with per-tenant isolation.

This is the functional gate for going public (`CLAUDE.md` ceiling #2: "one hardcoded league → cannot be multi-tenant"). Its **minimal single-league subset** — resolve the viewer's team from their login — is also a hard **M3 beta** gate: until it lands, all 12 beta users see Team Hickey.

### Load-bearing principle (unchanged): keep the brain, replace the plumbing
The Python analytics (`src/engine`, `src/optimizer`, `src/valuation`, ~6,300 tests) are **not rewritten**. Multi-tenancy is added in the **plumbing beneath** them: the identity/membership layer (new, api-owned) and the data-access layer (the B-design's repository that replaces `get_connection()`). The engine entry points stay frozen.

---

## 2. The central architecture decision (ratified with owner 2026-06-21)

**Isolation model: a single shared database, every league-specific row tagged with `league_id` (under a `tenant`), all league reads filtered by it (row-level / shared-schema multi-tenancy).**

- **Chosen** because it is one schema + one migration set to operate (fits the solo-operator / lean-ops North Star), matches the existing single-DB mental model, and is exactly what the B foundation design named (§4.2: "add `tenant_id` (and/or `league_id`) scoping to every league-specific table; row-level isolation").
- **Rejected — schema-per-tenant / database-per-tenant:** stronger isolation, but N schemas/DBs to migrate, back up, and operate; wrong fit for solo ops at the user counts we're targeting. (Revisit only if a future enterprise/whitelabel need appears.)

**The two layers this produces (the spine of the whole design):**

| Layer | What it holds | Where it lives | When it changes |
|---|---|---|---|
| **A — Identity & membership** | who a user is, which leagues they own, which team they are in each league, their per-league connection credentials | **api-owned store** — `data/api_state.db` today (env `HEATER_API_DB_PATH`), Postgres at M4 — the SAME store seam M2 established (`api/stores/`), **never** the live `draft_tool.db` | **NEW now** (additive, dormant) |
| **B — Baseball data** | players, projections, rosters, standings, statcast, refresh log | the engine's DB — `draft_tool.db` (SQLite) → Postgres at B2/M4 | **`league_id` tagging only when Postgres lands (M4)** |

**Why the beta needs only Layer A:** in the beta there is exactly **one** league (the shared FourzynBurn dataset everyone already reads), so Layer B has nothing to disambiguate — resolution is purely "which `team_name` do I filter this one dataset by," which lives entirely in Layer A. **That is what makes the M3 slice buildable on today's SQLite with no engine-DB change.** Layer B's `league_id` tagging is the Postgres-gated M4 work, required only once a *second* league exists.

---

## 3. Data model (Layer A — provider-agnostic, api-owned)

All tables are **new and api-owned**, created idempotently on first authenticated use (the M2 `SqliteUserStore` pattern: WAL + `busy_timeout`, own file, dormant until a Clerk user authenticates). At M4 a Postgres implementation drops in behind the same store Protocols.

### 3.1 `app_user` *(exists — M2)*
`id` (PK), `clerk_user_id` (UNIQUE), `created_at`. The local identity for a verified Clerk caller. Provisioned by the existing `provision_app_user`. The **tenant unit** of the product is the user account (it owns the connected leagues + the billing relationship). No separate `tenant` table unless org/family accounts appear; `app_user.id` *is* the tenant id. (Documented so "tenant_id" in the roadmap reads as `app_user.id` here.)

### 3.2 `league` *(new)*
A connected-league dataset.

| Column | Type | Notes |
|---|---|---|
| `id` | PK | HEATER's internal league id (the `league_id` Layer B will tag with) |
| `provider` | text | `yahoo` \| `espn` \| `sleeper` \| `cbs` \| `manual` |
| `external_league_id` | text | provider's own id (e.g. Yahoo `469.l.<key>`); NULL for `manual` |
| `name` | text | display name |
| `owner_user_id` | FK→app_user | who connected it; **NULL/system** for the shared beta league (it predates per-user connection) |
| `created_at` | text | |

UNIQUE(`provider`, `external_league_id`) where `external_league_id` is non-null. **Beta = exactly one row** (`provider=yahoo`, the FourzynBurn `external_league_id`, `owner_user_id=NULL`), seeded once.

### 3.3 `user_team` *(new — the replacement for the `team_name` query param)*
Maps a user to their team within a league. This is the membership/authorization edge.

| Column | Type | Notes |
|---|---|---|
| `id` | PK | |
| `user_id` | FK→app_user | the member |
| `league_id` | FK→league | which league |
| `team_name` | text | the human team name used to filter rosters today (what services already accept) |
| `team_key` | text | provider team identity (e.g. Yahoo `team_key`); preferred stable id, NULL-able for `manual` |
| `is_primary` | bool | the user's default league when they have several |
| `assigned_by` | FK→app_user | the admin who set it (the beta mechanism); NULL for self/OAuth-derived |
| `created_at` | text | |

UNIQUE(`user_id`, `league_id`). **Beta = 12 rows**, all → the one `league`, each with a member's `team_name`, `assigned_by` = the admin. This table is the sole source of truth for "the viewer's team," exactly as `league_rosters.is_user_team` / the Streamlit per-session assignment is today.

### 3.4 `league_credential` *(new — M4/M5-gated, empty in the beta)*
Each user's own connection secrets for a league (per-user OAuth tokens / API keys), **encrypted at rest**.

| Column | Type | Notes |
|---|---|---|
| `id` | PK | |
| `user_id` | FK→app_user | |
| `league_id` | FK→league | |
| `provider` | text | mirrors `league.provider` |
| `secret_enc` | blob/text | encrypted token bundle (encryption key from env/KMS; never logged) |
| `expires_at` | text | for OAuth refresh scheduling |

**Beta = empty:** all 12 users read the one shared dataset via the existing single Yahoo token (the scheduler's). Per-user credentials are populated only at public signup (M5). Spec'd now so the model is complete; **not built in M3-1.**

---

## 4. Identity → team/league resolution

### 4.1 The resolution chain (composes the existing, dormant auth seam)
```
Authorization: Bearer <clerk-jwt>
   └─ require_principal      → Principal(clerk_user_id)         [BUILT, api/auth.py]
       └─ require_app_user   → AppUser(id, clerk_user_id)       [BUILT, api/identity.py]
           └─ require_viewer_context  → ViewerContext           [NEW, this spec]
                 { user_id, league_id, team_name, team_key }
```

`require_viewer_context` (new, in a new `api/tenancy.py` composing the new `LeagueStore`/`MembershipStore`): given the `AppUser` and the active league, look up the user's `user_team` row and return a typed `ViewerContext`. **Active-league selection:** the user's `is_primary` membership by default; overridable by an explicit `league_id` (a future header/param for the frontend league switcher — single value in the beta). Personalized endpoints depend on `require_viewer_context` and read `ctx.team_name` instead of accepting a `team_name` query parameter.

### 4.2 Modeled on the proven Streamlit resolver
`src/auth.resolve_viewer_team_name` is the reference (the task mandates reuse, not reinvention). Port its two hard-won behaviors:
1. **Identity-first, no fallthrough to a global flag.** A logged-in member with **no** assignment resolves to a **`None` team** (→ a "connect / no team yet" state), never silently to someone else's team. (This is the 2026-06-01 launch-blocker lesson.)
2. **Name reconciliation.** A Yahoo team name often carries a leading emoji/whitespace (`"🏆 Team Hickey"`) that won't exact-match an admin-typed assignment (`"Team Hickey"`). When the assigned `team_name` isn't an exact roster-name match, normalize (the Streamlit `_normalize_team_name`) and return the **exact roster name** so downstream roster filters hit. Exact matches short-circuit.

### 4.3 Dormancy & backward-compatibility (the live app stays byte-for-byte)
The resolver mirrors `resolve_viewer_team_name`'s flag-gated shape — *"identity present → resolved team; absent → legacy behavior."* Concretely:

- **Clerk OFF (today, through M6 activation):** no authenticated identity → `require_viewer_context` returns the **`team_name` query param verbatim**, preserving each endpoint's **current param contract exactly** (required where required → still 422 if absent; optional/`""` where optional). The personalized endpoints behave **byte-for-byte as today**, and the new tables are never created (dormant, like the M2 stores). *(An optional `HEATER_VIEWER_DEFAULT_TEAM` convenience default is out of M3-1 scope — it would change required-param endpoints' 422 contract; deferred unless wanted.)*
- **Clerk ON + membership exists:** the resolved team wins; the query param is ignored (closing the no-authorization hole).
- **Clerk ON + no membership:** `team=None` → endpoints return the typed "no team assigned" state (never another user's data).

This makes the whole M3-1 slice **additive and dormant** — it cannot change live behavior until Clerk is configured *and* memberships are populated.

### 4.4 Admin assignment surface (the beta mechanism)
A small **admin-only** write path to create/update `user_team` rows:
`POST /api/admin/assignments { clerk_user_id | email, league_id, team_name }` → upserts the membership; `assigned_by` = the calling admin.

- **Admin gate:** an allowlist of admin Clerk user ids via env (`HEATER_ADMIN_CLERK_IDS`), checked by a `require_admin` dependency built on `require_app_user` — the api-owned twin of Streamlit's `ADMIN_USERNAME` seed + `require_admin`. Deny-by-default (empty allowlist → no admins → 403), fail-closed.
- **Beta flow:** Connor (admin) assigns each of the 12 users → their team, once. Read-back endpoint (`GET /api/admin/assignments`) lists current mappings for verification.
- Additive, dormant (no effect until Clerk users exist), no live-path touch.

---

## 5. League-scoping reaches the frozen engine without rewriting it *(Layer B — Postgres-gated, M4)*

Required only once a second league exists (i.e. **after** the beta, with Postgres). Design now; build behind the owner gate.

### 5.1 The seam: the repository layer, not the engine
The B foundation design (§4.2) introduces a **repository abstraction that replaces direct `get_connection()`**. League-scoping lives there:
- The API resolves `league_id` from identity (§4) and binds it to the data-access layer **for that one request**.
- **League-specific reads** (`league_rosters`, `league_standings`, `league_teams`, matchup caches, transactions) get an automatic `WHERE league_id = ?` filter.
- **Reference reads** (`players`, projections, `statcast_archive`) are **league-agnostic** (the same baseball facts for everyone) and stay unscoped — they do NOT get a `league_id`.

### 5.2 How `league_id` reaches deep engine code (the real tension)
The engine pervasively calls `get_connection()` across hundreds of sites; threading a `league_id` parameter through every signature would violate "keep the engine frozen."

- **Recommended: a request-scoped active-league context** (a `contextvar` set at the API boundary, read by the data-access/repository layer). Near-zero engine call-site churn. **Tradeoffs to verify under TDD with a real Postgres:** (a) it is implicit ("magic") — mitigated by funnelling ALL scoped reads through the one repository layer and a structural guard; (b) FastAPI runs the sync engine in a threadpool — `contextvars` *do* copy into threadpool workers, but this **must be proven** (set-per-request, reset in `finally`, and an isolation test that a tenant-A request never sees tenant-B rows even under concurrency).
- **Alternative: explicit parameter threading** at the few high-level engine entry points only (e.g. `build_optimizer_context`, `load_player_pool`) — cleaner/more testable there, but doesn't reach the deep call sites. **Hybrid (the actual plan): contextvar + repository scoping for the deep reads, explicit `league_id` at the high-level entry points where it's already clean.**

### 5.3 Isolation tests (new, required before this ships)
Per the B-design §7: a request scoped to tenant/league A **never** returns league-B rows — asserted directly, and under concurrency. This is the gating test for M4-1; **🔒 needs a real Postgres** (the dialect + threadpool behavior can't be proven on SQLite).

---

## 6. Per-user league connection (connectors + credentials) *(M4/M5-gated)*

From the B foundation design §4.5; spec'd here for completeness, built post-beta.

- **One `LeagueConnector` interface**, five implementations: **`SleeperConnector`** (open API, lowest legal risk — lead with it), **`YahooConnector`** (the existing `yahoo_data_service`, generalized to per-user OAuth), **`ESPNConnector`**, **`CBSConnector`**, **`ManualImportConnector`** (always-safe universal fallback). Each maps a provider's league/roster/team data into HEATER's canonical shape and discovers the user's `team_key`/`team_name` (which feeds `user_team` for the self/OAuth-derived path — the public alternative to admin assignment).
- **Per-user credentials** in `league_credential` (§3.4), refreshed by the workers (B3) per-tenant.
- **Rollout order** (adopting the B-design's data posture; owner may re-prioritize): Sleeper + Yahoo + manual first; ESPN/CBS best-effort.

---

## 7. Build sequence & owner-gates

| Slice | Scope | Infra | Gate |
|---|---|---|---|
| **M3-1** *(buildable now)* | Layer-A stores (`league` + `user_team`) in `api_state.db`; `ViewerContext` + `require_viewer_context` resolver (Streamlit-modeled, name-reconciling, dormant fallback); `require_admin` + admin-assignment endpoints; wire the 7 personalized endpoints to resolve team from identity (current query-param contract retained as the dormant fallback); seed the one shared beta `league` row | **Current SQLite** | **None — additive, dormant, no live-path touch, no CEO-file collision** |
| **M4-1** | Postgres (B2 complete) → `league_id` on Layer-B league tables; request-scoped repository filter (§5); tenant-isolation tests | Postgres | 🔒 **Owner go + real Postgres** (live read paths) |
| **M4-2** | `LeagueConnector` interface + `league_credential` + per-tenant refresh jobs; per-user Yahoo OAuth | Postgres + workers (B3) | 🔒 **Owner go** |
| **M4-3** | Turn multi-league on (frontend league switcher, active-league header) → feeds the M5 public launch | full stack | 🔒 **Owner go** |

**Only M3-1 is proposed for build after this spec + its plan are approved.** Everything below the line is a HARD STOP for the owner's explicit go-ahead (and M4-1+ additionally require a real Postgres, which is itself owner-provisioned).

---

## 8. Architecture & code placement (M3-1)

Follows the proven per-slice API pattern (`CLAUDE.md` "Proven per-slice pattern") + the M2 store seam:

- **New stores** (`api/stores/league_store.py`, `api/stores/membership_store.py`): Protocol + `InMemory*` fake + `Sqlite*` default (own tables in `api_state.db`, idempotent create, never `draft_tool.db`). Mirrors `user_store.py` exactly.
- **New tenancy module** (`api/tenancy.py`): `ViewerContext` model + `require_viewer_context` + the resolver helper (pure, DB-free-testable — stores injected). Replicates the **behavior** of the Streamlit `_normalize_team_name` (a ~5-line api-local normalizer — strip leading emoji/whitespace/punctuation, casefold). **It does NOT import from `src/`** — non-service api modules importing `src/` would break the "services are the one place importing `src/`" discipline, and re-homing the helper would be a `src/` edit. Replicating the small, well-understood normalization is the guard-clean, no-`src/`-touch choice (the task's "model on, don't reinvent" = same semantics, not a shared import). A unit test pins the api copy's behavior against the Streamlit cases.
- **New admin** (`api/admin.py` + `api/routers/admin.py`): `require_admin` (env allowlist, deny-by-default) + thin assignment routes + contracts (`api/contracts/admin.py`).
- **Additive edits to shared api files** (all additive, none CEO-active; `git pull --no-rebase` immediately before and after, tiny commits): DI providers in `api/deps.py`; mount the admin router in `api/main.py`; the 7 personalized routers (`team`, `matchup`, `lineup`, `punt`, `free_agents`, `playoff`, `trade_finder`) gain `ctx = Depends(require_viewer_context)` and pass `ctx.team_name`, **keeping the `team_name` query param with its current contract as the dormant fallback** so behavior is unchanged until Clerk is on. (CEO owns `src/ai`, the chat endpoint, `stream_analyzer`/`game_day`, the schedule endpoints — disjoint from these.)
- **No `src/` changes in M3-1.** The services already accept `team_name`. (`src/` changes begin at M4-1.)

---

## 9. Testing

- **DB-free api tests** (the worktree/CI DB is empty — memory `reference_worktree_empty_db`): inject the in-memory store fakes; cover resolver cases (assigned → team; no assignment → None; name reconciliation with a leading-emoji roster name; exact match short-circuit; Clerk-off fallback to query-param/env).
- **Admin gate:** empty allowlist → 403; listed admin → 200; non-admin Clerk user → 403; env-token path → dormant.
- **Dormancy guard:** Clerk-off behaves byte-for-byte like today (query param honored, no table created) — a connection spy proves no `api_state.db` table is created when no Clerk user authenticates.
- **Structural guard:** the personalized routers contain no team-resolution logic (resolution lives in `api/tenancy.py`) — extends the existing `test_no_logic_in_routers` discipline.
- **(M4-1) tenant-isolation tests** — 🔒 require a real Postgres (§5.3).

---

## 10. Risks & open questions

1. **Identity ≠ authorization gap** (from the Clerk plan §6): until this lands, a verified Clerk user must NOT get cross-tenant authority. This spec *is* the closure — the `user_team` membership is the authorization edge. Until M3-1 ships, the single-shared-team assumption is the guardrail.
2. **`contextvar` + threadpool correctness** (§5.2) — the one genuinely uncertain mechanism; gated behind a real-Postgres isolation test. If it doesn't hold, fall back to explicit `league_id` threading at the repository constructor.
3. **Credential encryption** (§3.4) — `league_credential` secrets must be encrypted at rest with an env/KMS key, never logged (the M2 "never log tokens" rule). Detailed in the M4-2 plan.
4. **Two users, one real league (public)** — by design each user connects their *own* copy (their own OAuth → their own `league` row), so two members of the same real-world league are two tenants with two datasets. Simpler + matches per-user OAuth; the (rare) duplicate-refresh cost is acceptable and revisited at scale (B-design §8).
5. **Cross-process provisioning race** (M2 noted) — at multi-replica, get-or-create on a brand-new id can lose the UNIQUE bet → 500 (retry succeeds). Catch `IntegrityError` + re-SELECT when multi-replica lands (M4). Acceptable at `numReplicas=1`.
6. **Open product input still needed for M4-2+ (not M3-1):** the exact per-provider connect UX + which providers ship at public launch (M5) — deferred to the M4-2/M5 plans; does not block M3-1.

---

## 11. What this spec does / does not do

**Does:** define the full multi-league multi-tenant model (Layers A/B), the identity→team/league resolution (Streamlit-modeled, dormant-safe), the admin-assignment beta mechanism, the engine-scoping approach, the connector/credential model, and a gated build sequence whose **first slice (M3-1) is buildable now** on current infra with no live-path or CEO-file collision.

**Does not:** build anything (that's the plan), touch the live data path, change any `src/` engine in M3-1, start Postgres, or implement connectors/credentials. M4-1+ are owner-gated and several require a real Postgres.

---

## 12. Next step

On approval → `writing-plans` produces the **M3-1 TDD plan** (the only buildable-now slice) — stores → resolver → admin gate → router wiring → tests — each step test-first, reviewed (code-reviewer + silent-failure-hunter), verified against the uvicorn API, committed + pushed. M4-1+ spawn their own plans when the owner green-lights them (and provisions a Postgres).
