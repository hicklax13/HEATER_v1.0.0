# Sub-project B3 — Background Workers (Redis + Arq) Migration Plan

> **STATUS: PLAN ONLY — NOT APPROVED FOR EXECUTION.** Part of the owner-gated B2–B5 infra. **Depends on B2 (Postgres)** — workers run in separate processes and need shared state, which SQLite's single-writer model can't provide. No code in this slice; each phase spawns its own bite-sized TDD plan when green-lit. Nothing starts until the owner ratifies §7 (and B2 is underway/done).

**Goal:** Move HEATER's heavy compute — Monte Carlo simulation, LP lineup optimization, trade-finder scans, draft MC — off the API request thread into **Arq background workers backed by Redis**, so the API stays responsive and scales horizontally (many web replicas + a separate worker pool). The Python **engines stay UNCHANGED** — a worker calls the exact same `src/` engine the synchronous service calls today; only *where* it runs moves.

**Non-goals (out of B3):** Postgres itself (B2), Clerk auth (Clerk-auth plan), multi-tenancy (B4), the Streamlit→Next.js cutover (C/D). B3 = "heavy jobs run async on a worker pool."

---

## 1. Why B3 exists — the heavy paths (surveyed from the live API surface)

The synchronous heaviness is already known and partly worked around today:

| Endpoint | Service → engine | Why heavy | Current state |
|----------|------------------|-----------|---------------|
| `POST /api/trade/evaluate` | `TradeService` → `evaluate_trade(enable_mc=...)` | Monte Carlo simulation | **MC is OPT-IN** (`enable_mc=False` default) precisely because synchronous MC caused a **44.8s WebSocket-drop freeze** in Streamlit (CLAUDE.md beta-audit). This is exhibit A for B3. |
| `POST /api/lineup/optimize` | `LineupService` → `LineupOptimizerPipeline` | PuLP LP solve + DCV scoring | Streamlit already moved this to a `ThreadPoolExecutor` (the R-6 async-optimize fix) — a band-aid B3 makes proper. |
| `POST /api/trade-finder` | `TradeFinderService` → `find_trade_opportunities` | cosine → 1v1 → 2v1 combinatorial scan | Streamlit gates it behind a button + session cache (cold scan was ~43.8s). |
| `POST /api/draft/recommend`, `/api/draft/simulate-picks` | `DraftService` → MC draft sim | `n_simulations` × candidates | Stateless; fine for small sims, heavy for large `n_simulations`. |

**The pattern is already in our favor:** each heavy path is a single service method (`api/services/*_service.py`) — the ONE seam that calls `src/`. B3 turns that method body into a **job function** and makes the router **enqueue** instead of call-and-wait. The engine code never moves.

---

## 2. The architecture decision (the spine)

**Chosen: Arq (async Redis task queue) + Redis as broker/result store; API enqueues → returns `202 {job_id}`; client polls `GET /api/jobs/{job_id}`.**

- **Arq** (by the encode/Starlette author) fits a FastAPI app: async, lightweight, Redis-backed, first-class `enqueue_job` + result retrieval, built-in retries/timeouts. Pairs naturally with the async API.
- **Job contract (the API shape):** heavy endpoints return **`202 Accepted` + `{ "job_id": "...", "status": "queued" }`** instead of blocking. A new **`GET /api/jobs/{job_id}`** returns `{ status: queued|running|complete|failed, result?: <the existing response contract>, error?: ... }`. The *result* payload is the SAME Pydantic contract the sync endpoint returns today (e.g. `LineupOptimizeResponse`) — so the frontend's types don't change, only the delivery (poll then read).
- **Result store:** Arq stores results in Redis with a TTL. (Large/durable results can later move to Postgres from B2 — a B3.3 option, not required for v1.)
- **Why not alternatives:**
  - *Celery* — heavier, sync-first, more operational surface than a solo-operator wants (North Star: low ops overhead).
  - *FastAPI `BackgroundTasks`* — runs in the SAME process; doesn't survive restarts, doesn't scale to a separate worker pool, doesn't fix the "one replica" ceiling. Not a real queue.
  - *RQ* — sync/threaded; Arq's async model matches the FastAPI stack better.
- **Delivery — poll first, push later:** `202 + GET /api/jobs/{id}` polling is the simplest correct contract for v1; Server-Sent Events / WebSocket push is a later enhancement (B3.3, optional) once the frontend wants live progress.
- **Keep the sync path during transition:** small inputs (e.g. `n_simulations` below a threshold, `enable_mc=False`) may still answer synchronously; only the genuinely heavy calls enqueue. The strangler principle from B2 applies — both paths coexist behind a flag/threshold until the frontend fully adopts polling.

**New deps:** `arq` (brings `redis`/`hiredis`). One stack addition. **Hard dependency: B2 Postgres** for any worker that writes shared state (the bootstrap/refresh write path); read-only compute jobs (optimize/trade/draft) can run on workers even before B2, but the writer model (`src/scheduler.py`) only relaxes once B2 lands — so B3 is sequenced **after** B2.

---

## 3. Phased plan (each phase = its own future TDD plan + worktree + 2-stage review + owner gate)

### Phase B3.0 — Worker infra + the jobs contract (no heavy path moved yet)
**What:** Add `arq` + a Redis URL config (`REDIS_URL`, localhost default for dev). Create `api/worker.py` (the Arq `WorkerSettings` + a trivial `ping` job). Add a **`GET /api/jobs/{job_id}`** router + a `JobStatus` contract (`status`, `result`, `error`) + a `JobService` seam that wraps Arq's result lookup. Add an `enqueue` DI provider (an Arq pool) so routers can enqueue without importing Arq directly (keeps routers thin).
**Why first:** establishes the queue + the polling contract with a throwaway job, before touching any real compute.
**Risk:** low. **Verification:** with a test Redis (or `fakeredis`/Arq's test harness), enqueue `ping` → `GET /api/jobs/{id}` transitions queued→complete; unknown id → a clean `not found` status (never 500). **Live path:** none.

### Phase B3.1 — Convert the first heavy path: trade Monte Carlo
**What:** Make `POST /api/trade/evaluate` with `enable_mc=True` **enqueue** an `evaluate_trade_mc` job (job body = today's `TradeService` MC call, engine unchanged) and return `202 {job_id}`; `enable_mc=False` stays synchronous. The job result is the existing `TradeEvaluationResponse`.
**Why this one first:** it's exhibit A (the 44.8s freeze) — highest pain, clearest win, and `enable_mc` already isolates the heavy branch.
**Risk:** medium. **Verification:** enqueued job runs the real engine on a seeded fixture and the polled result equals the synchronous result (parity test); `enable_mc=False` path unchanged. **Live path:** none (API dormant).

### Phase B3.2 — Convert lineup/optimize, trade-finder, draft MC
**What:** Same treatment for `POST /api/lineup/optimize`, `POST /api/trade-finder`, and the draft MC endpoints (threshold-gated: small `n_simulations` stays sync). Retire the Streamlit `ThreadPoolExecutor` band-aid only at the C/D cutover, not here.
**Risk:** medium. **Verification:** per-endpoint parity tests (job result == sync result on fixtures); threshold routing tested. **Live path:** none.

### Phase B3.3 — Result durability + (optional) progress push
**What:** Decide result store TTL; optionally persist long-lived results to Postgres (B2). Optionally add SSE/WebSocket progress for long jobs (the MC sim already computes convergence diagnostics that could stream).
**Risk:** low-medium. **Verification:** result survives the configured TTL; SSE (if built) streams then closes. **Live path:** none.

### Phase B3.4 — Deploy the worker pool
**What:** Provision Upstash Redis (Railway addon). Run the Arq worker as a **separate Railway service/process** from the web replicas. This is where the "single replica" ceiling actually breaks: web replicas scale out, workers scale independently. Coordinate with B2's Postgres (shared state) and the B2.4 sole-writer relaxation.
**Risk:** high (production infra + the multi-process writer model). **Live path:** **YES — hard owner gate; runs in staging first.**

---

## 4. Testing strategy
- **Unit/integration (CI):** use Arq's test utilities + `fakeredis` (or a Redis service container — the conftest network guard allows loopback) so jobs run without external infra. **Parity tests** are the core guard: the job result must equal the current synchronous result on a seeded fixture (proves "we only moved *where* it runs").
- **Contract test:** heavy endpoints return `202 {job_id}`; `GET /api/jobs/{id}` returns the `JobStatus` shape; the `result` matches the existing per-endpoint contract.
- **No-logic routers preserved:** enqueue happens via a DI'd pool/`JobService`, not inline logic — the `test_no_logic_in_routers` guard stays green.
- **OpenAPI:** the new `/api/jobs/{job_id}` path + the `202` responses regenerate `api/openapi.json` (snapshot-guarded).

## 5. Risk register
1. **B2 dependency** — workers need shared state; doing B3 on SQLite would reintroduce the single-writer lock. *Mitigation:* sequence B3 after B2; read-only compute jobs can pilot earlier but the writer model waits for B2.4.
2. **Result delivery / lost jobs** — polling must handle queued/running/complete/failed/expired/unknown without 500s; jobs must be idempotent enough to retry. *Mitigation:* explicit `JobStatus` states + Arq retry/timeout config + parity/idempotency tests.
3. **Job timeouts** — MC/trade-finder can be long; Arq job timeout must exceed worst-case or the job must chunk. *Mitigation:* per-job timeout config; convergence-based early stop already exists in the MC engine.
4. **Redis availability** — if Redis is down, enqueue must fail cleanly (503 + "try again"), never silently drop. *Mitigation:* fail-closed enqueue + health check.
5. **Frontend adoption** — the 202+poll contract is a client change; until the frontend polls, keep the sync path for small inputs. *Mitigation:* threshold-gated dual path (strangler).

## 6. Boundary with the rest of the roadmap
- **B2 (Postgres)** is the prerequisite (shared state).
- **Clerk-auth** is orthogonal (auth doesn't depend on workers).
- **B4 multi-tenancy** layers on top (per-tenant jobs/quotas come later).
- **B5 cutover** retires the Streamlit band-aids (its `ThreadPoolExecutor` optimize, the button-gated trade-finder) once the frontend uses the async API.

## 7. Decisions for owner ratification
- [ ] **Sequence:** confirm B3 lands after B2 (Postgres) is underway/done.
- [ ] **Spine** (§2): Arq + Redis; `202 {job_id}` + `GET /api/jobs/{id}` polling (SSE/WebSocket deferred). New dep `arq`.
- [ ] **Scope line:** B3 = async compute only. No auth/multi-tenancy here.
- [ ] **Dual-path threshold:** keep small/`enable_mc=False`/low-`n_simulations` calls synchronous; only heavy calls enqueue.
- [ ] **Infra:** Upstash Redis on Railway; worker as a separate Railway service (B3.4 = hard gate, staging first).

---

## 8. What this slice delivered (and did not)
**Delivered:** a ready-to-dispatch B3 plan — the architecture decision (Arq + Redis, 202+poll) with rejected alternatives, the heavy-path inventory grounded in the live API, a 5-phase additive sequence, a parity-test-centric test strategy, a risk register, and the explicit B2-dependency + roadmap boundaries.

**Did NOT (by design):** add any dependency, write any worker/job code, touch the live data path, or move any engine. Execution is deferred pending B2 progress + owner §7 ratification. When green-lit, **Phase B3.0** is the first to spawn a bite-sized TDD plan.
