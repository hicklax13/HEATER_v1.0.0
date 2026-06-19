# Sub-project B2 — SQLite → Postgres Migration Plan

> **STATUS: PLAN ONLY — NOT APPROVED FOR EXECUTION.** This document is the "agreed plan" gate the owner requires before any B2–B5 infra work touches the LIVE data path (per the ★ NORTH STAR / B-status in CLAUDE.md). No code in this slice. Each phase below spawns its own bite-sized TDD plan (`docs/superpowers/plans/`) only when the owner green-lights that phase.

**Goal:** Move HEATER's persistence from a single-writer SQLite file to Postgres, so the app can run multiple replicas + background workers (B3) and become multi-tenant (B4) — **without changing the Python engines** (`src/engine`, `src/optimizer`, `src/valuation`). The live Streamlit app keeps running on SQLite until each phase is proven and the owner cuts over (strangler-fig).

**Non-goals (explicitly out of B2):** Redis/Arq workers (B3), Clerk auth + multi-tenancy (B4), the Streamlit→Next.js page cutover (C/D). B2 stops at "Postgres is a drop-in backend the existing code can run on."

---

## 1. Current state (surveyed 2026-06-19 — the facts the plan is built on)

| Fact | Value | Source |
|------|-------|--------|
| Connection seam | `get_connection()` — `src/database.py:114` — sets WAL + `busy_timeout=60000` + `synchronous=NORMAL`, `row_factory=sqlite3.Row` | survey §1 |
| Non-test call sites | **482** `get_connection()` across **149** files (src/ 424, pages/ 30, scripts/ 25, api/ 3) | survey §1 |
| DB path config | `_resolve_db_path()` honors env `HEATER_DB_PATH`, else `data/draft_tool.db`. **No `DATABASE_URL` anywhere.** | survey §1/§5 |
| Schema | **53 `CREATE TABLE`** in `init_db()` + **62 `_safe_add_column` boot-time migrations**. **No Alembic / version table** — schema evolves by "add column if missing on every boot." | survey §2 |
| Existing PG scaffolding | **NONE** — 0 occurrences of sqlalchemy / psycopg / asyncpg / alembic / DATABASE_URL. Greenfield. | survey §5 |
| Keystone read | `_load_player_pool_impl` — `src/database.py:1856-2232` (~377 lines, dual 70-col SELECT) — consumed by **39 modules** (all 14 pages + 7 api/services + 18 src engines) | survey §4 |
| Sole-writer invariant | `src/scheduler.py` — `HEATER_SCHEDULER_IS_OWNER` gates the one writer thread; Railway `numReplicas=1`; readers rely on WAL + busy_timeout | survey §5 |
| Tests + DB | per-xdist-worker temp file via `HEATER_DB_PATH` (set in `tests/conftest.py` before any src import); **72 test files** hit the DB via `get_connection`; **565** occurrences | survey §6 |
| Network guard | `tests/conftest.py` **allows loopback** (127.0.0.1/localhost) — a local/container Postgres in tests passes unblocked | survey §6 |

### Dialect hotspots (the real work — not the connection swap)

| Idiom | Count | Postgres delta |
|-------|------:|----------------|
| `?` qmark placeholders | **~937 across ~42 files** | psycopg uses `%s` — **largest mechanical surface** |
| `INSERT OR REPLACE` / `REPLACE INTO` | 29 (+22 overlap) | → `ON CONFLICT (...) DO UPDATE` (REPLACE = delete+reinsert; semantics differ — can churn SERIALs/FKs) |
| `INSERT OR IGNORE` | 10 | → `ON CONFLICT DO NOTHING` |
| `ON CONFLICT(...)` already used | 25 | PG-compatible **iff** a matching UNIQUE/PK exists (SQLite was looser) |
| `COLLATE NOCASE` | 25 | **No PG equivalent** → `citext` or `lower()` expression index. Auth username uniqueness + Muncy-DNA player dedup depend on it — **correctness risk** |
| `julianday('now')` age math, `datetime('now')`, SQL `strftime` | ~13 SQL sites | → PG `age()`/`date_part`/`now()` |
| `PRAGMA table_info(...)` introspection | 3 (all inside the keystone read) | → `information_schema.columns` |
| `cursor.lastrowid` | 8 | → `INSERT ... RETURNING id` |
| `pd.read_sql_query` | **91** | works on a raw psycopg conn but officially wants a SQLAlchemy engine |
| `df.to_sql` | **0** | no pandas-write dtype surprises |
| Booleans as 0/1 INTEGER | ~9 flag columns | keep `INTEGER` (zero churn) — do **not** convert to `BOOLEAN` |
| Py-3.14 "SQLite returns bytes" workaround (`coerce_numeric_df`) | — | **disappears** under psycopg; the call stays harmless |
| `src/ai/sql_tool.py:60` raw read-only `file:...?mode=ro` connect | 1 | → PG read-only role/conn; **not covered by the structural guard** |

---

## 2. The architecture decision (the spine)

**Chosen: a SQLAlchemy *engine* connection seam + Alembic for schema + explicit (hand-ported) SQL dialect fixes + a strangler dual-backend transition.**

We adopt SQLAlchemy's **engine/connection layer only** — NOT the ORM and NOT a wholesale rewrite of every query into Core expressions. Raw SQL strings stay; they are made dialect-correct explicitly (not by a runtime regex shim). `get_connection()` remains the single chokepoint but is backed by an engine selected by `DATABASE_URL`.

Why this and not the alternatives:

- **Rejected — thin psycopg swap + runtime dialect "shim"** (regex-rewrite `?`→`%s`, `INSERT OR REPLACE`→`ON CONFLICT` at call time): fragile across ~937 placeholders / 61 upserts, silently wrong on `%` in `LIKE`, literal `?` in strings, and `COLLATE NOCASE`/`julianday`/`PRAGMA` it can't fix. A correctness-bug factory on the live data path. **No.**
- **Rejected — full repository/DAO rewrite** (encapsulate all 482 sites behind data-access objects): cleanest end-state but enormous, and it would change engine behavior — violating the North Star's "keep the Python engine UNCHANGED." Out of scope for B2.
- **Chosen — engine seam + explicit port**: smallest change that (a) gives pandas a real engine (silences the 91 `read_sql` warnings, lets reads run on either backend), (b) lets **Alembic** own schema, (c) keeps raw SQL (engines unchanged) while fixing dialect divergences deliberately and test-covered, and (d) supports running **both backends** during the strangler transition via `DATABASE_URL`/a backend flag.

**Corollary decisions (recommended defaults — open to owner override at review):**
1. **Driver:** `psycopg` (v3) + `SQLAlchemy>=2.0`. Sync first (matches today's blocking code); async (`asyncpg`) deferred to B3 when the MC/optimize paths become Arq jobs.
2. **Paramstyle:** route parameters through the engine so we standardize on one style; where raw `?` SQL remains, convert to `%s` per-module during B2.2 (not globally at once).
3. **Booleans stay `INTEGER` 0/1** (zero churn). **Timestamps stay `TEXT` ISO** for B2 (native `timestamptz` is a later, separate cleanup — porting all readers/writers is its own risk and not required to run on PG).
4. **`COLLATE NOCASE` → `citext`** extension on the two columns that need it (`users.username`, the player-name dedup key), not app-wide `lower()` rewrites.
5. **Tests:** keep the **SQLite** fast unit lane (engine abstraction keeps most queries running on both) **and** add a **Postgres integration lane** (a local/container PG; the network guard already allows loopback). Queries that become PG-only (e.g. `citext`, `information_schema`) get marked to the PG lane.
6. **Cutover:** strangler — a `DATABASE_BACKEND`/`DATABASE_URL` switch; **reads migrate first** behind the flag, then writes; the live SQLite app is untouched until the owner flips prod. Full rollback at every step.

---

## 3. Phased plan (each phase = its own future TDD plan + worktree + review + owner gate)

> Ordering is dependency-driven. **Only B2.0 is backward-compatible enough to merge while still defaulting to SQLite**; B2.1+ are developed/verified against Postgres behind the flag and are NOT cut over in prod until B2.5 (owner-gated).

### Phase B2.0 — Engine seam (backward-compatible; SQLite stays the default)
**What:** Add deps (`sqlalchemy`, `psycopg[binary]`, `alembic`). Introduce `src/db/engine.py` with `get_engine()` selecting the backend from `DATABASE_URL` (absent → today's SQLite file via a SQLite URL). Re-implement `get_connection()` to hand out a DBAPI connection **from the engine**, preserving `row_factory`-equivalent dict rows and the WAL/busy_timeout PRAGMAs **only on the SQLite backend**. Point the 91 `pd.read_sql_query` calls' connection at the engine (or a thin `engine_conn()` helper).
**Why first:** establishes the single swap point with **zero behavior change** (default still SQLite). This is the only phase that can land on master early without touching prod behavior.
**Risk:** low. **Verification:** the entire existing suite stays green on SQLite-via-engine; `get_connection()` contract (dict-row access, PRAGMAs) unchanged; structural guard `test_no_direct_sqlite_connect_in_scripts` updated to recognize the engine path.
**Touches live path?** No (default backend unchanged).

### Phase B2.1 — Alembic baseline + schema port
**What:** Stand up Alembic. Author the **baseline migration** = the 53 tables + the 62 `_safe_add_column` columns + indexes, expressed in **Postgres-correct DDL** (`SERIAL`/identity for the 23 `AUTOINCREMENT`, drop affinity assumptions, `citext` for the 2 NOCASE columns, partial unique indexes on `player_id_map` carried over verbatim — PG supports them, the single-row `CHECK(id=1)` carried over). Keep the boot-time `init_db()`/`_safe_add_column` idempotency as a transition safety net (so SQLite still self-heals), but Postgres schema is owned by Alembic.
**Risk:** medium (53 tables × type decisions). **Verification:** `alembic upgrade head` builds a fresh PG schema; a schema-diff test asserts every table/column the code references exists; `init_db()` on SQLite is unchanged.
**Touches live path?** No (PG only; SQLite untouched).

### Phase B2.2 — SQL dialect port (module-by-module)
**What:** The explicit fixes, done per-module with PG-lane tests: `?`→`%s`; `INSERT OR REPLACE`/`REPLACE INTO`→`ON CONFLICT ... DO UPDATE` (auditing each for the delete+reinsert semantic change); `INSERT OR IGNORE`→`ON CONFLICT DO NOTHING`; confirm all 25 existing `ON CONFLICT` targets have real PG constraints; `cursor.lastrowid`→`RETURNING id` (8 sites); `julianday`/`datetime('now')`/`strftime`→PG date funcs; `COLLATE NOCASE`→`citext` (verify the auth uniqueness + Muncy-DNA dedup behavior on PG). Sequence modules by blast radius, lowest first; `src/database.py` writers last before the keystone.
**Risk:** high (volume + correctness). **Verification:** each module's existing tests run on the **PG lane** and pass; targeted new tests for the COLLATE/upsert/RETURNING behavior changes.
**Touches live path?** No (PG lane only until B2.5).

### Phase B2.3 — Keystone read path (`_load_player_pool_impl`)
**What:** Port the 377-line dual-SELECT: the 3 `PRAGMA table_info` probes → `information_schema.columns`, `julianday` age → PG date math, the correlated `ownership_trends` subquery + 5-way LEFT JOINs verified on PG. This unblocks all 39 consumers (pages + API services).
**Risk:** high (single highest-leverage function). **Verification:** `load_player_pool()` returns the **same columns/shape** on PG as on SQLite for a seeded dataset (golden-frame comparison test); the api/services that call it pass on the PG lane.
**Touches live path?** No (PG lane only).

### Phase B2.4 — Writers + relax the sole-writer invariant
**What:** Run the `bootstrap_all_data` write path + `src/scheduler.py` against PG. With PG MVCC, **retire** the `HEATER_SCHEDULER_IS_OWNER` single-writer lock + WAL/busy_timeout workarounds (keep them on the SQLite backend). This is the conceptual core of the migration and the **boundary with B3** (where writers become out-of-process Arq jobs) — B2.4 only proves concurrent writes are safe on PG; it does NOT add Redis/workers.
**Risk:** high — **this is where multi-writer correctness lives.** **Verification:** concurrent-write stress test on PG (the bootstrap's `ThreadPoolExecutor` bursts that needed `busy_timeout` on SQLite) shows no lock errors / no lost writes; refresh_log integrity preserved.
**Touches live path?** Conceptually yes (the writer model) — **owner gate required before this phase starts.**

### Phase B2.5 — Data migration + strangler cutover
**What:** A one-time SQLite→PG data loader (idempotent, re-runnable). Stand up prod Postgres (Railway PG addon). Run **both** backends behind `DATABASE_URL`: cut **reads** over first (the app reads PG, scheduler still writes SQLite → verify parity), then **writes**, then retire SQLite. Document rollback (flip `DATABASE_URL` back).
**Risk:** high (production data). **Verification:** row-count + checksum parity SQLite vs PG post-load; a staging run of the full app on PG; owner sign-off per sub-step.
**Touches live path?** **YES — this is the cutover. Hard owner gate; runs in a staging/parallel environment first.**

---

## 4. Testing strategy

- **Keep the SQLite unit lane** (fast, per-worker temp file via `HEATER_DB_PATH`) for everything the engine abstraction keeps backend-portable.
- **Add a Postgres integration lane**: a local/container PG reached over loopback (the conftest network guard already permits 127.0.0.1/localhost). Wire a `DATABASE_URL` env for this lane; mark PG-only tests (`citext`, `information_schema`, multi-writer) with a marker so the SQLite lane skips them.
- **Golden-frame parity**: for the keystone read (B2.3) and key writers (B2.2), assert SQLite-result == PG-result on a seeded fixture — the strongest guard that the port preserved behavior.
- **CI**: add a PG service container to a new CI job (the existing sharded SQLite job stays). Pin PG version.

## 5. Risk register (ranked — from the survey)

1. **~937 `?`→`%s` across ~42 files** — volume; missed/over-eager substitution. *Mitigation:* per-module port (B2.2) with PG-lane tests, never a global regex.
2. **Sole-writer → multi-writer** (`scheduler.py` lock + WAL/busy_timeout) — the core architectural shift, touches the live writer. *Mitigation:* B2.4 concurrent-write stress tests; owner gate; overlaps/coordinated with B3.
3. **`_load_player_pool_impl` port** (PRAGMA introspection + `julianday`, 39 consumers) — breaks everything if wrong. *Mitigation:* golden-frame parity test (B2.3).
4. **`INSERT OR REPLACE`→`ON CONFLICT` semantics** (61 sites; REPLACE ≠ upsert) + every `ON CONFLICT` target needs a real PG constraint. *Mitigation:* per-site audit in B2.2; constraint coverage test.
5. **`COLLATE NOCASE`→`citext`** (auth uniqueness + Muncy-DNA dedup) — silent mis-port = duplicate accounts / duplicate players (correctness, not crash). *Mitigation:* dedicated behavior tests on PG (B2.2).

**Honorable mentions:** no Alembic today (the migration *tooling* is itself B2.1 work); `src/ai/sql_tool.py` read-only connect (separate PG read-only-role port, and it escapes the current structural guard); 91 `pd.read_sql` (engine adoption in B2.0 resolves).

## 6. Decisions for owner ratification

Before B2 execution begins, confirm:
- [ ] **The spine** (§2): SQLAlchemy-engine seam + Alembic + explicit per-module dialect port + strangler dual-backend. (Recommended; alternatives rejected with reasons.)
- [ ] **Scope line**: B2 ends at "runs on Postgres." Workers (B3) / auth+multi-tenancy (B4) stay separate. B2.4 only *proves* concurrent writes; B3 makes them out-of-process.
- [ ] **Defaults** (§2 corollaries): psycopg v3 + SQLAlchemy 2.0 sync; booleans stay INTEGER; timestamps stay TEXT for B2; NOCASE→citext; keep both test lanes.
- [ ] **Sequencing & gates**: B2.0 may merge early (backward-compatible). B2.4 and B2.5 require explicit per-phase owner go-ahead because they concern the live writer model and the production cutover.
- [ ] **When to start**: this is the infra long-pole and the owner has deferred execution. Nothing in B2 begins until this checklist is signed off.

---

## 7. What this slice delivered (and did not)

**Delivered:** this plan (grounded in a full survey of the SQLite layer), the architecture decision with rejected alternatives, a 6-phase sequence with per-phase risk/verification/live-path flags, a test strategy, a ranked risk register, and an owner-ratification checklist.

**Did NOT (by design):** write or run any migration code, add any dependency, touch the live data path, or change any engine. Execution remains deferred pending owner sign-off on §6. When green-lit, **Phase B2.0** is the first to spawn a bite-sized TDD plan.
