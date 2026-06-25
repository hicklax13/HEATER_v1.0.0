# HEATER Backend 100/100 Remediation Program

**Audit date:** June 25, 2026
**Current strict public-SaaS backend grade:** 54/100
**Target:** 100/100 in every backend category
**Scope:** FastAPI, Python engines, persistence, data ingestion, integrations, auth, tenancy, billing, AI backend, workers, observability, tests, release engineering, and production operations
**Operating model:** Connor plus Codex, using managed infrastructure and low-operations architecture
**Document status:** Plan only. Do not implement any remediation solely because it appears in this document.

---

## 1. Executive Assessment

HEATER has a substantial backend for a private fantasy-baseball application:

- A FastAPI transport layer with 44 mounted `/api/*` routes.
- 121 generated API schemas.
- Thin routers, dependency injection, Pydantic contracts, and service seams.
- Clerk JWT verification and server-side team assignment.
- Stripe checkout, portal, webhook, subscription, and Pro-gating foundations.
- A large Python analytics engine covering lineups, free agents, streaming, trades, standings, draft decisions, simulations, and AI.
- Hundreds of API tests and thousands of repository tests.
- A dormant SQLAlchemy and Alembic foundation for PostgreSQL.
- A production deployment that is alive and rejects unauthenticated access to protected routes.

Those are real strengths. They make HEATER a capable private beta backend.

They do not make it a production-grade public SaaS backend.

The main issue is not feature quantity. It is that the backend's public-product control systems are incomplete:

1. The live data plane is still SQLite and direct SQL.
2. Account state and fantasy data are split across two SQLite files.
3. The tenant model still resolves one configured default league.
4. Domain tables are not consistently tenant- and league-scoped.
5. Heavy analytics execute synchronously inside web requests.
6. There is no durable distributed job system.
7. The API has no standard error envelope, request identifier, idempotency framework, or documented authentication scheme.
8. `/healthz` proves only that the process can return a static dictionary.
9. There is no production metrics, tracing, SLO, alerting, or dependency-readiness system visible in the backend.
10. Provider access remains centered on one Yahoo league and shared application credentials.
11. Data quality is materially incomplete in the local canonical player dataset.
12. Analytical confidence exceeds the available empirical validation evidence.
13. All 30 registered model constants remain marked `calibrated: false`.
14. The historical playoff calibration path is explicitly a skeleton.
15. The optimizer backtest evaluates projections but does not replay optimizer decisions against a counterfactual baseline.
16. Broad exception recovery is common enough that partial or empty output can be mistaken for valid output.
17. The local API suite currently has one failing OpenAPI snapshot test.

For a private beta, several of these are acceptable transitional choices. For a public paid SaaS, they are launch blockers.

### Strict conclusion

HEATER's backend is best described as:

> A strong single-league analytical monolith with a credible API facade, early SaaS control-plane seams, and insufficient production evidence for multi-tenant scale, operational reliability, or quantitative trust claims.

The current grade is **54/100**. The grade is intentionally harsh because the target is a paid public product that stores account data, connects to third-party fantasy services, produces consequential recommendations, and may eventually execute roster transactions.

---

## 2. Audit Method and Evidence

### 2.1 Audit standard

The backend was evaluated as if HEATER were preparing for:

- Public self-service signup.
- Multiple users and leagues.
- Paid subscriptions.
- Per-user provider authorization.
- Continuous in-season operation.
- High-trust analytical recommendations.
- Safe roster writes.
- Supportable incidents.
- Repeatable releases.
- Solo-operator maintenance.

The audit did not award full credit for:

- Code that is intentionally dormant.
- Planned architecture that is not on the live data path.
- Unit tests that do not exercise the real production dependency.
- Graceful fallback that hides loss of data or analytical validity.
- A probability or confidence label without historical calibration.
- A schema that can compile for PostgreSQL but has not run against PostgreSQL.
- Authentication code that is secure in isolation while the tenant data model remains shared.
- High test count without contract, load, isolation, recovery, and production evidence.

### 2.2 Evidence inspected

The audit inspected:

- `api/main.py`.
- API routers, services, contracts, dependencies, gateways, and stores.
- Clerk verification and identity provisioning.
- Viewer-context and membership resolution.
- Stripe billing and entitlement logic.
- SQLAlchemy engine and Alembic schema seams.
- The live SQLite connection layer.
- The local `draft_tool.db` dataset.
- Model calibration and backtesting modules.
- CI workflows and pytest configuration.
- The committed OpenAPI document.
- The generated live OpenAPI object.
- Representative production endpoints.
- Existing architecture, migration, and remediation documentation.

### 2.3 Objective repository measurements

| Measurement | Result |
|---|---:|
| Mounted `/api/*` routes | 44 |
| OpenAPI operations including `/healthz` | 45 |
| OpenAPI component schemas | 121 |
| OpenAPI security schemes | 0 |
| API Python files | 90 |
| API Python lines | 9,525 |
| `src` Python files | 175 |
| `src` Python lines | 90,800 |
| API test files | 74 |
| API tests run locally | 587 |
| API tests passing | 586 |
| API tests failing | 1 |
| `get_connection()` references in `api` and `src` | 222 |
| Direct `sqlite3` imports in `api` and `src` | 9 |
| Broad `except Exception` handlers | 687 |
| Broad exception handlers ending in `pass` | 82 |
| Model constants registered | 30 |
| Model constants marked calibrated | 0 |

The line and search counts are not defects by themselves. They are risk indicators. The relevant concern is the combination of direct persistence access, broad fallback behavior, synchronous computation, and a future multi-tenant product.

### 2.4 Local database measurements

The local `data/draft_tool.db` inspected during this audit contained:

| Measurement | Result |
|---|---:|
| Database size | 26,677,248 bytes |
| Tables | 54 |
| Players | 9,888 |
| Players missing `mlb_id` | 7,180 |
| Players missing team | 1,693 |
| Projection rows | 31,620 |
| Projection forecast seasons | 2026 only |
| Season-stat rows | 6,944 |
| Season-stat seasons | 2023-2026 |
| Local league-roster rows | 0 |
| Refresh-log rows | 39 |
| Refresh statuses not `success` | 15 |

Approximately 73% of player rows lack an MLB identifier and approximately 17% lack a team. Some missing identifiers may be legitimate historical, minor-league, duplicate, or non-current records. The backend does not yet provide the lineage and quality reporting needed to distinguish acceptable missingness from resolution failures.

### 2.5 Test result

The local API test command produced:

```text
586 passed, 1 failed
```

The failure was:

```text
tests/api/test_openapi_contract.py::test_openapi_snapshot_is_current
```

The committed OpenAPI snapshot contains `ValidationError.properties.ctx`; the generated schema does not. This is small in behavioral impact but important as release evidence: the repository currently fails its own API contract gate.

### 2.6 Production probes

Representative unauthenticated probes returned:

| Endpoint | Result |
|---|---|
| `/healthz` | `200 {"status":"ok"}` |
| `/api/standings` | `401` |
| `/api/me/team` | `401` |

This proves:

- The production process was reachable during the audit.
- Authentication is active for those routes.

It does not prove:

- Database readiness.
- Provider readiness.
- Stripe readiness.
- Queue readiness.
- Data freshness.
- Tenant isolation.
- Correctness of authenticated results.
- Recovery from dependency failure.
- Capacity under concurrent load.

### 2.7 Evidence limitations

This was a repository-grounded and unauthenticated production audit. It did not:

- Use a production user token.
- Inspect production databases.
- inspect production secrets.
- Execute a real Yahoo write.
- Run a real Stripe purchase.
- Run a real Clerk account lifecycle.
- Conduct a penetration test.
- Conduct a sustained load test.
- Restore a production backup.
- Replay a historical fantasy season end to end.
- Verify provider data licensing.

Those limitations are not assumed to be failures. They are required future audit evidence.

---

## 3. Backend Scorecard

| Category | Weight | Current score | Primary reason |
|---|---:|---:|---|
| API architecture and contract governance | 7 | 6 | Good seams; stale contract, no auth scheme, inconsistent errors |
| Domain boundaries and maintainability | 6 | 4 | Strong service pattern; large monolith and direct data coupling remain |
| Persistence and data model | 9 | 3 | SQLite live path, split stores, incomplete PostgreSQL migration |
| Multi-tenancy and isolation | 9 | 4 | Identity mapping exists; one default league and incomplete scoped schema |
| Authentication, security, and privacy | 8 | 6 | Clerk and write gates are credible; broader controls are incomplete |
| Reliability and failure semantics | 7 | 4 | Defensive fallbacks exist; degraded results are often ambiguous |
| Scalability and background execution | 7 | 2 | Heavy synchronous request work; no durable distributed workers |
| Data ingestion, quality, and lineage | 8 | 5 | Broad ingestion surface; weak canonical identity and evidence controls |
| Analytical correctness and model validation | 10 | 3 | Sophisticated engines; insufficient historical calibration and backtests |
| Provider integrations and portability | 6 | 4 | Deep Yahoo integration; no public connector lifecycle or abstraction proof |
| Billing and entitlement integrity | 5 | 3 | Good foundation; incomplete reconciliation, idempotency, and lifecycle proof |
| Observability and production operations | 6 | 2 | Basic logs and health; no complete telemetry or SLO operating system |
| Performance, caching, and capacity | 5 | 3 | Some bounded work and caches; no measured public capacity envelope |
| Testing, CI, migrations, and release safety | 5 | 4 | Large suite and CI; current contract failure and missing production-class tests |
| Documentation and developer experience | 2 | 1 | Extensive planning; operational source of truth remains fragmented |
| **Total** | **100** | **54/100** | **Strong beta backend, not public-SaaS ready** |

### Grade interpretation

- **90-100:** Public SaaS backend with independently verifiable controls.
- **80-89:** Production-capable with bounded launch exceptions.
- **70-79:** External beta capable with meaningful operational risk.
- **60-69:** Strong private beta.
- **50-59:** Feature-rich prototype or private product with public-product seams.
- **Below 50:** Material correctness or architecture risk prevents dependable operation.

HEATER is at the upper end of the feature-rich private-product band.

---

## 4. Category Critiques and 100/100 Standards

### 4.1 API Architecture and Contract Governance

#### Strengths

- Routers are generally thin.
- Pydantic request and response contracts are explicit.
- Dependency injection makes services testable.
- The committed OpenAPI document is snapshot-tested.
- Write routes advertise key authentication errors.
- Services isolate much of the legacy engine surface.

#### Critique

1. The committed OpenAPI contract is stale on the audited commit.
2. OpenAPI declares no bearer-auth security scheme, so generated clients and external developers cannot infer authentication correctly.
3. Authentication requirements are expressed through dependencies but not consistently represented as operation security requirements.
4. Error responses are not governed by one versioned error envelope.
5. Several services return successful HTTP responses containing `ok=false`, empty arrays, or fallback values for operational failures.
6. The API does not expose stable machine-readable degraded-state metadata across all endpoints.
7. There is no global request ID or correlation ID contract.
8. There is no API version prefix or explicit compatibility policy.
9. Pagination, sorting, filtering, field limits, and maximum response sizes are not standardized.
10. Mutation idempotency is not a platform-level contract.
11. Retry safety is not documented per operation.
12. Long-running compute has no asynchronous job contract.
13. The API title version `0.1.0` is not connected to a compatibility or deprecation process.
14. Contract tests primarily protect shape, not semantic invariants.
15. The router/service rule is strong, but service methods sometimes import persistence helpers directly, making the public contract depend on legacy data behavior.

#### 100/100 standard

- OpenAPI is regenerated deterministically and cannot be stale on a mergeable commit.
- Bearer auth, required scopes, Pro entitlements, admin roles, and write permissions are documented in OpenAPI.
- Every error uses a shared envelope with:
  - `code`.
  - `message`.
  - `request_id`.
  - `retryable`.
  - `details`.
  - `dependency`.
  - `degraded`.
- Status codes distinguish validation, authentication, authorization, conflict, rate limit, unavailable dependency, timeout, and internal failure.
- All list endpoints follow one pagination and limits standard.
- All mutations support or explicitly reject idempotency keys.
- Asynchronous operations return a job resource.
- Contract changes run automated breaking-change analysis.
- The compatibility policy states support windows, deprecation headers, and removal gates.
- Generated TypeScript clients compile in CI.
- Every endpoint has semantic contract tests, examples, and documented degraded behavior.

### 4.2 Domain Boundaries and Maintainability

#### Strengths

- The Python analytics engine is separated from HTTP routers.
- API services provide a useful migration seam.
- Gateways and protocols exist for Stripe and several stores.
- The no-logic router test enforces a valuable boundary.
- Major engines have focused modules rather than one universal service.

#### Critique

1. The backend is still a large in-process monolith with over 100,000 Python lines across `api` and `src`.
2. `src/database.py` and `src/data_bootstrap.py` are each over 4,000 lines.
3. Multiple domain engines exceed 2,000 lines.
4. Several API services exceed 500 or 700 lines.
5. There are 222 references to the legacy connection helper.
6. Domain services often know table shapes and SQL details.
7. Transport services sometimes compensate for inconsistent engine output.
8. Canonical concepts such as player identity, team identity, category configuration, freshness, and recommendation evidence are not represented by one domain model.
9. The API layer and Streamlit layer can still implement similar transformations independently.
10. Shared behavior is protected by conventions and tests more than by module ownership.
11. Error behavior is distributed across broad exception handlers.
12. The architecture does not yet make tenant context a required dependency of every personalized domain operation.
13. There is no clear package-level dependency graph enforced in CI.
14. Read models, write models, analytical models, and persistence models are often conflated in DataFrames or dictionaries.
15. The solo-operator burden grows with each endpoint because cross-cutting controls are not centralized.

#### 100/100 standard

- A documented bounded-context map defines account, league, provider, player, data, recommendation, simulation, billing, AI, and audit domains.
- Each context has owned models, repositories, services, events, and tests.
- Dependency rules are executable and checked in CI.
- Personalized services require an immutable request context containing user, tenant, league, role, plan, locale, and request ID.
- Persistence is accessed only through repositories or explicitly approved analytical read models.
- Engine input and output types are stable, typed, and versioned.
- No API service exceeds the agreed complexity and size threshold without an approved exception.
- Legacy modules have staged extraction plans and characterization tests.
- DataFrames remain allowed inside analytical computation but do not serve as untyped cross-layer contracts.
- Cross-cutting concerns are middleware, dependencies, or shared services rather than repeated endpoint code.

### 4.3 Persistence and Data Model

#### Strengths

- SQLAlchemy and Alembic foundations exist.
- The schema includes the current live table surface.
- SQLite uses WAL, busy timeout, and normal synchronous mode.
- API account stores have protocol-backed in-memory test implementations.
- Database behavior has extensive existing tests.

#### Critique

1. The live data path remains SQLite.
2. Setting a non-SQLite `DATABASE_URL` intentionally raises `NotImplementedError`.
3. The account and billing control plane uses a separate SQLite file.
4. Store classes create tables at runtime rather than relying exclusively on migrations.
5. Domain data still relies heavily on direct SQL and sqlite row behavior.
6. The dormant SQLAlchemy schema intentionally omits foreign keys.
7. The live application historically runs with foreign-key enforcement disabled.
8. Many timestamps and structured values are stored as text.
9. Schema ownership is split between `init_db`, SQLAlchemy metadata, and API-store `CREATE TABLE IF NOT EXISTS` statements.
10. There is no demonstrated production PostgreSQL migration, load, rollback, or restore.
11. There is no connection-pool budget documented for web and worker processes.
12. There is no repeatable zero-downtime migration protocol.
13. There is no row-level tenant isolation defense in the database.
14. There is no formal retention and deletion model.
15. Database backup evidence and restore-time evidence are absent.

#### 100/100 standard

- PostgreSQL is the only production source of truth.
- One migration system owns all production tables.
- Runtime application code never creates or alters production tables.
- All entities use stable primary keys and explicit foreign keys.
- All tenant-owned rows carry enforced tenant and league references.
- PostgreSQL row-level security or equivalent database-enforced isolation protects sensitive tenant tables.
- Timestamps use timezone-aware database types.
- Structured data uses documented JSONB only when relational columns are inappropriate.
- Referential actions are explicit.
- Every high-volume query has an index and query-plan evidence.
- Migrations are tested forward and backward where rollback is safe.
- Backups are encrypted, monitored, retained, and restored in drills.
- A data deletion workflow removes or anonymizes all user-owned records.
- The system meets defined RPO and RTO.

### 4.4 Multi-Tenancy and Isolation

#### Strengths

- Clerk identity can provision a local application user.
- Membership maps a user to a league and team.
- Authenticated unassigned users do not fall back to another client-supplied team.
- Team-required endpoints can return a specific `team_not_linked` conflict.
- Admin assignment is explicitly gated.

#### Critique

1. The league store still creates and resolves exactly one configured default league.
2. The default league configuration still references Yahoo and FourzynBurn assumptions.
3. No canonical `tenant_id` field appears in the main domain schema.
4. Many league tables are keyed by team names, week numbers, or implicit singleton context.
5. Personalized engine functions often receive a team name rather than a tenant-safe team identifier.
6. Cache keys are not universally tenant- and league-scoped.
7. Background-job tenant propagation does not exist because the worker system does not exist.
8. Provider credentials are not modeled per connected account and league.
9. The application has no complete multi-league switch lifecycle.
10. There is no user-owned organization or household boundary.
11. Tenant deletion, export, transfer, and support impersonation are not complete.
12. Cross-tenant adversarial tests cover only the current seams, not every repository and cache.
13. AI conversation and tool access require full tenant-boundary review.
14. Logs may include team or player context without a formal privacy classification.
15. One missed SQL predicate could expose shared league data once multiple leagues use the same database.

#### 100/100 standard

- `tenant_id`, `league_id`, and stable team IDs are mandatory where applicable.
- Every repository method accepts a server-derived scope.
- Every cache key, job payload, object-storage key, event, log, and audit record carries scope.
- The database rejects cross-tenant foreign-key relationships.
- Row-level security blocks unscoped reads and writes.
- Users can connect multiple leagues and switch explicitly.
- Roles include owner, manager, viewer, support, and system worker where needed.
- Provider credentials are encrypted and bound to the correct user and connection.
- Cross-tenant fuzz, integration, and penetration tests show zero leakage.
- Support access is time-limited, reason-coded, approved, and audited.
- Deleting a tenant removes or anonymizes every owned artifact.

### 4.5 Authentication, Security, and Privacy

#### Strengths

- Clerk JWTs are verified locally with RS256 JWKS.
- Issuer and audience validation are supported.
- Authentication fails closed.
- Static-token fallback uses constant-time comparison.
- Write routes are explicitly authenticated.
- Admin routes use a separate authorization gate.
- Stripe webhook signatures fail closed.
- CORS uses an explicit origin allowlist.

#### Critique

1. The API contract does not publish its bearer security scheme.
2. No centralized authorization policy engine governs route, resource, role, league, and action decisions.
3. There is no visible application-wide rate limiting.
4. There is no abuse protection for expensive analytical or AI endpoints.
5. There is no global request-size policy.
6. There is no IP, account, or device anomaly response system.
7. Secrets-management expectations are environment-variable conventions rather than a complete rotation and audit system.
8. Provider tokens and AI keys require a formal encryption-at-rest and access-control audit.
9. No independent penetration test evidence exists.
10. No software-composition or container vulnerability gate was found in the core CI workflow.
11. No SBOM or signed artifact requirement was found.
12. Security headers, TLS policy, and edge protections are outside the backend contract.
13. Account deletion, data export, consent, privacy policy, and retention are not complete backend workflows.
14. Audit logs are not yet the authoritative append-only record for privileged and destructive actions.
15. Safe roster-write authorization is not yet proven with per-user Yahoo write scope.
16. The fallback static-token mode increases configuration risk if enabled outside controlled environments.

#### 100/100 standard

- One authorization service evaluates principal, tenant, league, resource, action, and entitlement.
- OpenAPI accurately documents auth and scopes.
- Rate limits exist per IP, user, tenant, route class, and cost unit.
- Expensive requests use quotas and concurrency limits.
- Request bodies, uploads, pagination, and query complexity are bounded.
- Secrets are stored in managed secret storage, rotated, inventoried, and never logged.
- Provider credentials are envelope-encrypted.
- Dependency, secret, SAST, container, and IaC scanning are required checks.
- Builds produce an SBOM and signed image provenance.
- Annual penetration testing and pre-launch testing are complete.
- Privacy export and deletion are verified end to end.
- Audit logs are immutable and include actor, scope, action, target, result, request ID, and reason.
- Yahoo writes require the user's authorized write scope and an explicit confirmation workflow.

### 4.6 Reliability and Failure Semantics

#### Strengths

- Many integration paths use timeouts and bounded retries.
- Services frequently choose safe empty responses rather than crashing.
- Critical webhook verification fails closed.
- SQLite contention protections are present.
- Some expensive operations are bounded.

#### Critique

1. There are 687 broad exception handlers in `api` and `src`.
2. Eighty-two broad handlers end in `pass`.
3. Empty lists, `None`, zeros, and fallback scores can represent both valid data and failure.
4. Partial result provenance is not standardized.
5. An endpoint can return HTTP 200 while a required dependency failed.
6. There is no unified retry classification.
7. There is no circuit-breaker standard.
8. There is no bulkhead or concurrency-isolation standard.
9. Idempotent replay is not guaranteed for mutations or jobs.
10. Dependency timeouts are not centrally budgeted against a request deadline.
11. There is no application-wide graceful-shutdown contract for active work.
12. There is no dead-letter queue.
13. There is no durable replay mechanism for failed refreshes or provider events.
14. There is no chaos or fault-injection test suite.
15. The health endpoint remains green when databases or providers are unusable.

#### 100/100 standard

- Errors have typed causes and explicit degraded semantics.
- Required dependency failure returns an appropriate 5xx response.
- Optional dependency failure returns data plus a structured warning.
- No user-visible zero or empty value can ambiguously mean system failure.
- Retry policies are centralized, bounded, jittered, and operation-aware.
- Circuit breakers protect unstable providers.
- Web, worker, and scheduler workloads have separate concurrency pools.
- Mutations and jobs are idempotent.
- Failed jobs enter a dead-letter queue with operator replay.
- Request deadlines propagate through all dependency calls.
- Graceful shutdown drains requests and workers safely.
- Fault-injection tests prove expected degraded behavior.
- Recovery metrics and runbooks exist for every critical dependency.

### 4.7 Scalability and Background Execution

#### Strengths

- The code identifies several operations that should move to Arq.
- Some computational input sizes are capped.
- Monte Carlo engines can use deterministic seeds.
- Railway can host separate process types.

#### Critique

1. There is no implemented Redis and Arq production worker system.
2. Lineup, trade, draft, playoff, and other compute can run in web workers.
3. Data refresh and user requests can compete for CPU, memory, database locks, and provider limits.
4. SQLite enforces a practical single-writer architecture.
5. In-process caches are replica-local and disappear on restart.
6. There is no durable job state model.
7. There is no per-tenant fair scheduling.
8. There is no job cancellation, timeout, progress, or retry API.
9. There is no queue latency SLO.
10. There is no worker autoscaling policy.
11. There is no capacity model for concurrent simulations.
12. There is no admission control when compute demand exceeds safe capacity.
13. Scheduled work is not coordinated through distributed locks.
14. Duplicate refresh work can consume provider quotas.
15. A single large computation can still harm request latency.

#### 100/100 standard

- Web requests perform bounded orchestration only.
- Long-running and CPU-heavy work uses durable jobs.
- Redis and workers are independently deployable and observable.
- Jobs have stable types, schemas, tenant scope, idempotency keys, progress, deadlines, attempts, and results.
- Priority queues protect interactive work.
- Per-user and per-tenant quotas prevent noisy-neighbor behavior.
- Distributed locks deduplicate refresh work.
- Worker autoscaling is tied to queue depth, age, CPU, and memory.
- Queue latency and completion SLOs are measured.
- Backpressure returns a clear retry response.
- Capacity tests prove the supported public concurrency envelope.

### 4.8 Data Ingestion, Quality, and Lineage

#### Strengths

- HEATER ingests many useful baseball and fantasy data sources.
- Refresh logging exists.
- Canonical player-reference helpers exist in the API.
- Several refresh paths distinguish cached, partial, skipped, no-data, and error states.
- The database holds multiple seasons of actual statistics.

#### Critique

1. Approximately 73% of local player rows lack `mlb_id`.
2. Approximately 17% of local player rows lack a team.
3. Canonical player identity is not complete enough for a public recommendation product.
4. The local league roster table is empty, so many real paths cannot be validated locally.
5. Projection rows cover only the active forecast season.
6. Refresh state is not a complete lineage record.
7. Source, retrieval time, event time, transformation version, and quality state are not attached to every derived record.
8. Partial provider refreshes can leave mixed-time datasets.
9. Reconciliation across Yahoo IDs, MLB IDs, names, teams, and projection IDs remains probabilistic.
10. There is no quarantine workflow for unresolved identity.
11. There is no immutable raw-data landing zone.
12. There is no deterministic replay from raw input through transformations.
13. Data-quality thresholds are not release gates.
14. Commercial rights and allowed uses are not encoded as source metadata.
15. User-facing outputs do not uniformly expose freshness and source coverage.

#### 100/100 standard

- Every active player has a canonical internal ID and source-ID mappings.
- Identity conflicts are quarantined and never silently merged.
- Every dataset records source, event time, ingestion time, schema version, transformation version, completeness, and quality.
- Raw provider payloads are retained according to legal and operational policy.
- Transformations are deterministic and replayable.
- Batch manifests support atomic publication.
- Readers never observe a half-published dataset.
- Quality rules cover uniqueness, validity, completeness, consistency, freshness, and distribution drift.
- Failed quality checks block publication.
- Data contracts alert on upstream schema changes.
- Source licensing and retention restrictions are documented and enforced.
- Product responses expose meaningful freshness and coverage.

### 4.9 Analytical Correctness and Model Validation

#### Strengths

- HEATER contains sophisticated Monte Carlo, Bayesian, optimization, correlation, and game-theory logic.
- Deterministic seeds and self-consistency checks exist.
- Metrics such as Brier score, rank MAE, RMSE, and Spearman correlation are implemented.
- Backtest scaffolding and target thresholds exist.
- Analytical code has extensive unit tests.

#### Critique

1. All 30 registered constants are marked uncalibrated.
2. The playoff historical calibration module explicitly says the external historical dataset is future work.
3. Self-consistency confirms implementation behavior, not real-world predictive validity.
4. Monte Carlo stability confirms sampling noise, not model correctness.
5. Historical matchup calibration can create schedule rows with category outcomes all set to zero.
6. Trade-history ingestion does not yet reconstruct complete players, teams, counterfactuals, or realized utility.
7. The optimizer backtest evaluates projection accuracy but does not run the optimizer and compare its selected lineup with feasible alternatives.
8. The optimizer backtest can compare values by row position rather than a fully verified point-in-time identity join.
9. Projection validation is not proven against immutable, point-in-time forecasts.
10. Data leakage controls are not documented for each model.
11. There is no model registry with version, training data, parameters, metrics, approval, and rollback.
12. There is no production outcome capture tied to recommendation version.
13. Confidence and probability language can appear more authoritative than the validation evidence supports.
14. Baseline strategies are not consistently defined.
15. Performance is not segmented by season phase, player type, sample size, data freshness, league configuration, or recommendation class.
16. Statistical uncertainty around validation metrics is not consistently reported.
17. There is no abstention policy when evidence is weak or inputs are stale.
18. There is no independent quantitative review.

#### 100/100 standard

- Point-in-time historical datasets exist for at least three completed seasons.
- Every forecast is joined only to information available at prediction time.
- Each recommendation family has explicit baselines.
- Lineup backtests replay real eligibility, schedules, roster locks, and transaction constraints.
- Trade backtests evaluate realized rest-of-season category utility and uncertainty, not only rank change.
- Playoff probabilities are calibrated by time horizon and probability bucket.
- Streaming and free-agent recommendations are evaluated against replacement and consensus baselines.
- Every key constant is empirically calibrated or explicitly labeled as a prior with sensitivity evidence.
- Validation reports include sample size, confidence intervals, subgroup performance, and failure cases.
- A model registry governs promotion and rollback.
- Production recommendations record model version, feature snapshot, uncertainty, and eventual outcome.
- User-facing probability language is allowed only after passing calibration gates.
- Weak-data cases abstain or downgrade confidence.
- An independent reviewer signs off on methodology and claims.

### 4.10 Provider Integrations and Portability

#### Strengths

- Yahoo integration is deep and supports many league operations.
- Provider calls include rate-limit awareness.
- Read and write paths are separated.
- The API can degrade when Yahoo data is unavailable.

#### Critique

1. The product still assumes Yahoo in critical data and identity paths.
2. The configured beta league uses one external league ID.
3. Per-user Yahoo OAuth connection is not a complete public workflow.
4. Yahoo write scope is not enabled and proven.
5. There is no provider-neutral connector contract demonstrated by a second platform.
6. League settings are not normalized through one canonical rules model.
7. Provider capability differences are not a first-class API resource.
8. Token refresh, revocation, reconnect, and degraded-state workflows need production evidence.
9. Provider webhooks are not the primary synchronization model.
10. Polling budgets and backoff are not centrally coordinated per tenant.
11. Manual import is not a complete fallback connector.
12. Provider terms, caching restrictions, and data deletion obligations are not operational controls.
13. Connector fixtures do not replace end-to-end sandbox or real-account certification.
14. Provider outage status is not exposed in one operational dashboard.
15. A provider schema change can still break multiple legacy modules.

#### 100/100 standard

- A connector interface owns authorization, leagues, teams, rosters, settings, matchups, transactions, writes, capabilities, and sync cursors.
- Yahoo is one implementation, not the domain model.
- At least one additional provider or a robust manual-import connector proves portability.
- OAuth lifecycle is self-service.
- Credentials are encrypted and revocable.
- Capability discovery prevents unsupported actions.
- Provider calls use shared rate-limit, retry, circuit-breaker, and telemetry controls.
- Sync is incremental, idempotent, and replayable.
- Provider terms and retention limits are documented.
- Contract tests use recorded and sanitized fixtures.
- Release certification includes real connected-account tests.

### 4.11 Billing and Entitlement Integrity

#### Strengths

- Stripe is isolated behind a gateway.
- Signature verification fails closed.
- Checkout and portal sessions exist.
- Subscription status drives Pro access.
- Out-of-order checkout completion does not blindly reactivate canceled users.
- Billing can remain dormant until configuration is complete.

#### Critique

1. Subscription state lives in SQLite.
2. Webhook event IDs are not visibly stored for deduplication.
3. Event ordering is handled partially but not by a complete monotonic event model.
4. There is no periodic Stripe reconciliation job.
5. Checkout customer creation is not protected by a platform idempotency key.
6. Entitlements are represented mainly as free versus Pro rather than explicit capabilities.
7. The managed AI allowance can over-grant during a subscription-store failure.
8. There is no complete dunning, grace-period, refund, dispute, and chargeback policy.
9. Tax, invoice, coupon, price migration, and grandfathering behavior are not certified.
10. Billing audit records are not an append-only ledger.
11. There is no end-to-end live-mode purchase certification.
12. Billing support tools and operator runbooks are incomplete.
13. A webhook-secret misconfiguration can drop events while returning a non-error response when billing is partially configured.
14. Account deletion and Stripe-customer retention behavior are not fully specified.
15. Revenue recognition and financial export needs are not documented.

#### 100/100 standard

- Stripe event IDs are deduplicated.
- Subscription updates reject stale events.
- A reconciliation job repairs drift.
- Checkout uses idempotency keys.
- Entitlements are explicit capabilities with effective dates.
- Billing failure modes fail safely and alert operators.
- Dunning, grace, cancellation, refund, dispute, and reactivation rules are tested.
- Live-mode end-to-end certification succeeds.
- Billing actions produce immutable audit records.
- Support tools show Stripe and local state with safe repair actions.
- Privacy deletion and required financial retention are reconciled.

### 4.12 Observability and Production Operations

#### Strengths

- Python logging is widely used.
- Important integration failures often log warnings.
- Production exposes a health endpoint.
- Refresh status exists in the data model.
- CI provides broad regression feedback.

#### Critique

1. `/healthz` is a static liveness response.
2. There is no readiness endpoint.
3. There is no dependency health summary.
4. There is no application-wide request ID.
5. There is no structured logging contract.
6. There is no distributed tracing.
7. There is no standard metrics endpoint or managed metric export.
8. There are no documented backend SLOs.
9. There are no error-budget policies.
10. There are no alert thresholds tied to user impact.
11. There is no production dashboard for API latency, errors, queues, databases, providers, data freshness, billing, and model jobs.
12. Release versions are not visibly correlated with errors and traces.
13. There is no incident severity system in the backend plan.
14. There is no demonstrated backup restore drill.
15. There is no status-page integration.
16. Broad fallback behavior can suppress the signal needed for incident detection.

#### 100/100 standard

- Separate liveness, readiness, and detailed authenticated diagnostics exist.
- Every request and job has a correlation ID.
- Logs are structured, redacted, scoped, and centrally searchable.
- OpenTelemetry traces cover HTTP, database, queues, providers, Stripe, Clerk, and AI calls.
- RED metrics cover every route.
- USE metrics cover infrastructure.
- Data freshness and quality metrics cover every source.
- SLOs and error budgets govern releases.
- Alerts are actionable and mapped to runbooks.
- Deploy markers correlate changes with failures.
- Backup restore and disaster-recovery drills produce evidence.
- Public incidents can update a status page.

### 4.13 Performance, Caching, and Capacity

#### Strengths

- Several services cap expensive input sets.
- Some live data has TTL caching.
- Monte Carlo sample counts are configurable.
- CI uses timeouts to detect hangs.
- The system has already fixed known expensive paths.

#### Critique

1. There is no published endpoint latency budget.
2. There is no load-test suite representing public traffic.
3. In-process cache is not coherent across replicas.
4. Cache keys are not universally tenant-, league-, model-, and data-version aware.
5. Cache invalidation is not tied to atomic dataset publication.
6. Expensive operations share the web process.
7. Database query plans are not continuously checked.
8. Response size budgets are not enforced.
9. There is no capacity model for simulations per minute.
10. There is no explicit memory budget per worker.
11. Provider fan-out can make latency unpredictable.
12. Cold-start performance is not an SLO.
13. Backpressure behavior is not defined.
14. Performance regressions are not a required CI gate.
15. Cache hit ratio and stale-hit behavior are not measured.

#### 100/100 standard

- Every route class has p50, p95, p99, timeout, and size budgets.
- Load tests cover normal, peak, burst, and dependency-degraded traffic.
- Shared caches use Redis where coherence matters.
- Cache keys include all correctness dimensions.
- Cache invalidation is event-driven or versioned.
- Compute work is isolated in workers.
- Query plans and slow-query thresholds are monitored.
- Capacity tests define safe web and worker concurrency.
- Admission control protects the service.
- Performance budgets are enforced before release.

### 4.14 Testing, CI, Migrations, and Release Safety

#### Strengths

- The repository has a large automated test suite.
- CI shards tests and enforces a coverage floor.
- Ruff linting and formatting are required.
- API contracts are snapshot-tested.
- Routers are structurally guarded.
- Tests use dependency overrides and fake services.

#### Critique

1. The audited API suite has one failing contract test.
2. The coverage floor is 60% for `src`, which is not enough evidence for critical paths.
3. Coverage is not risk-weighted.
4. There is no full PostgreSQL integration suite on every relevant change.
5. There is no Redis and worker integration suite.
6. There is no cross-tenant adversarial test matrix over all repositories.
7. There is no provider contract certification against real accounts.
8. There is no Stripe live-mode certification workflow.
9. There is no sustained load or soak test gate.
10. There is no fault-injection suite.
11. There is no migration dry-run against a production-like database snapshot.
12. There is no automated breaking-OpenAPI comparison.
13. Security scanning is not a complete required pipeline.
14. Model validation reports are not release artifacts.
15. Rollback is not automatically exercised.
16. Production smoke tests are not a comprehensive post-deploy gate.

#### 100/100 standard

- Main is always green.
- Critical modules meet branch and mutation-testing thresholds.
- PostgreSQL, Redis, worker, Clerk, Stripe, and connector integration tests run in isolated environments.
- Tenant isolation is tested generatively.
- Contract compatibility is automatically classified.
- Migration plans run against realistic data volume.
- Load, soak, and failure tests run before public releases.
- Security scans and SBOM generation are required.
- Model-validation reports are immutable release artifacts.
- Canary deploys and automatic rollback are proven.
- Production smoke tests verify authenticated personalized workflows.

### 4.15 Documentation and Developer Experience

#### Strengths

- The repository contains extensive plans and specifications.
- Architecture decisions are often recorded in code comments.
- API patterns are consistent enough to copy.
- The project has a detailed operating context.

#### Critique

1. Current truth is distributed across long instructions, specs, plans, comments, historical bug files, and code.
2. Planned, dormant, active, and retired architecture can be difficult to distinguish.
3. There is no concise backend architecture map generated from the current system.
4. There is no one-command local environment containing PostgreSQL, Redis, workers, and provider stubs.
5. Operational runbooks are not the primary interface for incidents.
6. Environment variables lack one validated schema and generated reference.
7. API examples and auth requirements are not fully self-documenting.
8. Data contracts and model cards are incomplete.
9. Ownership and escalation remain implicit because the team is one person plus Codex.
10. The volume of historical context raises the risk of following obsolete instructions.

#### 100/100 standard

- A short architecture index links to authoritative current documents.
- Architecture decision records distinguish accepted, superseded, and retired decisions.
- One command starts a production-like local stack.
- Environment configuration is typed and validated at startup.
- API docs include auth, examples, errors, rate limits, and degraded behavior.
- Runbooks cover every alert and critical dependency.
- Data contracts, model cards, connector guides, and migration guides are current.
- Generated diagrams reflect real module and infrastructure dependencies.
- Documentation freshness is checked in release review.

---

## 5. Remediation Principles

### 5.1 Preserve the analytical engine

Do not rewrite working analytical logic merely to change infrastructure. Put typed boundaries around it, build point-in-time validation, and move execution behind durable services.

### 5.2 Fix control planes before feature breadth

Public signup must not precede:

- PostgreSQL.
- Tenant-scoped data.
- Provider credential isolation.
- Rate limiting.
- Readiness monitoring.
- Reliable jobs.
- Restore evidence.

### 5.3 Never hide degraded correctness

Availability is not success when the user is making a decision from incomplete data. Every fallback must identify:

- What failed.
- What data was omitted.
- Whether the result remains safe to use.
- Whether retrying may help.

### 5.4 Treat recommendation evidence as backend data

Every recommendation must be reproducible from:

- Input snapshot.
- Model version.
- Constants version.
- Constraints.
- Seed.
- Data freshness.
- Evidence.
- Uncertainty.

### 5.5 Use managed infrastructure

The target should minimize operator burden:

- Managed PostgreSQL.
- Managed Redis.
- Railway web and worker services.
- Managed object storage.
- Sentry.
- PostHog or equivalent controlled product analytics.
- Clerk.
- Stripe.

### 5.6 Migrate through seams

Use the existing service, store, gateway, and SQLAlchemy seams. Do not combine database, tenancy, workers, auth, provider migration, and model changes in one release.

### 5.7 Make rollback a feature

Every live-data change needs:

- A rollback trigger.
- A rollback procedure.
- Compatibility across the rollback window.
- A production observation period.

---

## 6. Target Backend Architecture

### 6.1 Runtime components

1. **FastAPI web service**
   - Authentication.
   - Authorization.
   - Validation.
   - Bounded orchestration.
   - Read APIs.
   - Job submission.
   - Mutation confirmation.

2. **PostgreSQL**
   - Account, tenant, league, provider, billing, job, data, recommendation, and audit state.
   - Row-level tenant protection.
   - Transactional publication state.

3. **Redis**
   - Durable queue support.
   - Distributed locks.
   - Rate limits.
   - Shared caches.
   - Short-lived job progress.

4. **Arq workers**
   - Simulations.
   - Optimizations.
   - Provider syncs.
   - Historical imports.
   - Data-quality checks.
   - Model evaluations.
   - Billing reconciliation.

5. **Scheduler**
   - Enqueues jobs only.
   - Uses distributed locks.
   - Never performs heavy work inside the scheduler process.

6. **Object storage**
   - Raw provider payloads.
   - Immutable data manifests.
   - Historical feature snapshots.
   - Validation reports.
   - Export archives.

7. **Observability**
   - Sentry errors and release correlation.
   - OpenTelemetry traces.
   - Central metrics and logs.
   - SLO dashboards and alerts.

### 6.2 Request context

Every personalized service receives:

```text
RequestContext
  request_id
  principal_id
  app_user_id
  tenant_id
  league_id
  team_id
  role
  plan
  provider_connection_id
  locale
  deadline
```

No personalized service may infer this context from:

- A global constant.
- A configured default league.
- A client-provided team name.
- A module-level cache.
- An environment variable.

### 6.3 Core domain identifiers

Use stable internal identifiers:

- `tenant_id`.
- `user_id`.
- `league_id`.
- `fantasy_team_id`.
- `provider_connection_id`.
- `player_id`.
- `recommendation_id`.
- `model_version_id`.
- `dataset_version_id`.
- `job_id`.

Provider IDs and names are attributes, not primary authorization identifiers.

---

## 7. Phase 0: Baseline, Freeze, and Evidence Registry

### Objective

Create an authoritative baseline before changing the live backend.

### Work package 0.1: Record the current API

1. Export OpenAPI from a clean environment.
2. Fix the current snapshot mismatch in the future implementation branch.
3. Record every route, method, auth dependency, entitlement, request model, response model, timeout, and service.
4. Classify each route:
   - Public static.
   - Authenticated global.
   - Tenant read.
   - Tenant write.
   - Admin.
   - Billing webhook.
   - Expensive compute.
   - AI.
5. Record documented and actual errors.
6. Record whether the route can return partial data.

### Work package 0.2: Record current persistence

1. Inventory every table and column.
2. Map every direct SQL call to its owning domain.
3. Map every writer and transaction boundary.
4. Identify singleton assumptions.
5. Identify name-keyed relationships.
6. Record retention and sensitivity.
7. Record expected row counts for one, 100, 1,000, and 10,000 leagues.

### Work package 0.3: Record current analytical behavior

1. Freeze representative engine inputs and outputs.
2. Store deterministic fixtures for each recommendation family.
3. Record seeds and constant versions.
4. Record performance on realistic local data.
5. Record all known fallback conditions.

### Work package 0.4: Establish issue severity

- **Critical:** Cross-tenant exposure, unauthorized write, incorrect charge, secret exposure, corrupt publication, or materially fabricated recommendation.
- **High:** Public outage, silent stale data, unrecoverable job loss, incorrect entitlement, or unsupported confidence claim.
- **Medium:** Partial endpoint failure, poor diagnostics, performance regression, or incomplete workflow.
- **Low:** Internal inconsistency without material user impact.

### Exit gate

- The current API, schema, data flows, dependencies, and analytical outputs are reproducible.
- Every later phase links its changes to baseline evidence.

---

## 8. Phase 1: API Contract and Failure Architecture

### Objective

Turn the API from an internal frontend seam into a governed public-product interface.

### Work package 1.1: Standard error model

Create:

```json
{
  "error": {
    "code": "provider_unavailable",
    "message": "League data could not be refreshed.",
    "request_id": "req_...",
    "retryable": true,
    "degraded": false,
    "dependency": "yahoo",
    "details": {}
  }
}
```

Rules:

1. Codes are stable and documented.
2. Messages are safe for users.
3. Details contain no secrets.
4. Operational failures never masquerade as valid empty data.
5. Validation errors are normalized.
6. Provider errors map to stable categories.
7. Timeout and rate-limit responses include retry guidance.

### Work package 1.2: Request identity

1. Accept a valid incoming correlation ID or generate one.
2. Add it to response headers.
3. Bind it to logs, traces, jobs, audits, and provider calls.
4. Return it in all error envelopes.

### Work package 1.3: OpenAPI security

1. Add bearer security schemes.
2. Annotate authenticated operations.
3. Annotate admin and Pro requirements.
4. Document webhook authentication separately.
5. Add examples for success, error, degraded, and async responses.
6. Generate the frontend client in CI.

### Work package 1.4: API versioning

1. Define compatibility rules for `/api`.
2. Decide whether the stable public surface becomes `/api/v1`.
3. Support additive changes without version bumps.
4. Require migration windows for breaking changes.
5. Add deprecation and sunset headers.

### Work package 1.5: Mutation idempotency

1. Accept `Idempotency-Key` on writes.
2. Scope keys to principal, tenant, route, and payload hash.
3. Store status and result.
4. Reject key reuse with a different payload.
5. Replay the original result safely.

### Work package 1.6: Async job API

Create:

- `POST /api/v1/jobs`.
- `GET /api/v1/jobs/{job_id}`.
- `DELETE /api/v1/jobs/{job_id}` where cancellation is safe.

Job states:

- `queued`.
- `running`.
- `succeeded`.
- `failed`.
- `canceled`.
- `expired`.

### Exit gate

- OpenAPI is current.
- Authentication appears in generated clients.
- Every endpoint has standard errors.
- Mutations are retry-safe.
- Long-running endpoints have an approved sync or async classification.

---

## 9. Phase 2: PostgreSQL and Repository Migration

### Objective

Complete the dormant B2 migration without changing analytical behavior.

### Work package 2.1: Production-like PostgreSQL environment

1. Add a local PostgreSQL service.
2. Add a CI PostgreSQL service.
3. Use the pinned production major version.
4. Configure connection pooling.
5. Add database startup validation.
6. Run Alembic to head in local and CI environments.

### Work package 2.2: Unify schema ownership

1. Move API store tables into Alembic.
2. Remove runtime `CREATE TABLE`.
3. Make SQLAlchemy metadata descriptive, not an alternative migration source.
4. Add foreign keys.
5. Add uniqueness and check constraints.
6. Convert timestamps.
7. Inventory and convert structured text.

### Work package 2.3: Repository layer

Create repositories by bounded context:

- Player.
- Projection.
- League.
- Roster.
- Matchup.
- Schedule.
- Transaction.
- User.
- Membership.
- Subscription.
- Conversation.
- Recommendation.
- Job.
- Audit.

Rules:

1. Repositories require scope where data is tenant-owned.
2. Repositories own SQL.
3. Services own business orchestration.
4. Engines receive typed snapshots.
5. Tests use repository interfaces, not global connection monkeypatches.

### Work package 2.4: Port reads

1. Port one endpoint family at a time.
2. Compare SQLite and PostgreSQL outputs against frozen fixtures.
3. Record query plans.
4. Add indexes.
5. Run shadow reads in staging.
6. Detect differences before cutover.

### Work package 2.5: Port writes

1. Identify every transaction boundary.
2. Add row locking where needed.
3. Add optimistic versioning where needed.
4. Make refresh publication atomic.
5. Make provider event application idempotent.
6. Make billing event application transactional.

### Work package 2.6: Data migration

1. Export SQLite with checksums.
2. Transform and load into staging PostgreSQL.
3. Validate row counts, key uniqueness, null rates, totals, and samples.
4. Run application parity tests.
5. Rehearse cutover.
6. Rehearse rollback.
7. Schedule a final sync window.

### Exit gate

- Production uses PostgreSQL.
- No production code creates tables.
- No personalized API path uses direct SQLite.
- Backups and restore drills pass.
- The live application remains behaviorally equivalent.

---

## 10. Phase 3: Tenant-Native Data Model

### Objective

Make cross-tenant exposure structurally difficult and testably impossible.

### Work package 3.1: Tenant schema

Add:

- `tenants`.
- `tenant_users`.
- `leagues`.
- `league_memberships`.
- `fantasy_teams`.
- `provider_connections`.
- `provider_leagues`.
- `provider_team_links`.

### Work package 3.2: Scope existing tables

Classify each table:

- Global reference data.
- Provider-global data.
- Tenant-owned data.
- League-owned data.
- User-owned data.
- Derived analytical data.

Add required scope columns and foreign keys.

### Work package 3.3: Replace names as identity

1. Introduce stable fantasy-team IDs.
2. Preserve names as display fields.
3. Migrate schedules, rosters, standings, matchups, and transactions.
4. Support team renames.
5. Support duplicate names across leagues.

### Work package 3.4: Row-level security

1. Set tenant and user context on each database transaction.
2. Add policies to sensitive tables.
3. Deny unscoped access.
4. Give workers explicit service roles.
5. Test policy bypass attempts.

### Work package 3.5: Scoped caches and jobs

1. Include tenant and league in every cache key.
2. Include tenant and league in every job payload.
3. Reject jobs with missing scope.
4. Add scope to traces and audits.
5. Prevent shared in-memory personalized caches.

### Work package 3.6: Multi-league lifecycle

1. List connected leagues.
2. Connect a league.
3. Select a team.
4. Switch active league.
5. Reconnect.
6. Disconnect.
7. Delete imported data.
8. Transfer ownership.

### Exit gate

- No server path uses a configured default league for authenticated users.
- Cross-tenant integration and penetration tests pass.
- Row-level security is enabled.
- Multiple real test leagues can coexist safely.

---

## 11. Phase 4: Durable Workers and Scheduling

### Objective

Remove expensive and unreliable work from web request processes.

### Work package 4.1: Redis and Arq foundation

1. Add managed Redis.
2. Add worker and scheduler process types.
3. Define queue names and priorities.
4. Add connection health checks.
5. Add job serialization versioning.

### Work package 4.2: Job persistence

Store durable job metadata in PostgreSQL:

- Job type.
- Scope.
- Requester.
- Input reference.
- Input hash.
- Status.
- Progress.
- Attempts.
- Deadline.
- Result reference.
- Error code.
- Created, started, and completed times.

### Work package 4.3: Migrate compute

Priority order:

1. Playoff simulations.
2. Trade Monte Carlo.
3. Draft simulations.
4. Large lineup optimizations.
5. Historical validation.
6. Bulk AI operations.

Keep a synchronous fast path only when:

- p95 remains below the approved budget.
- work is bounded.
- cancellation is not required.
- dependency fan-out is small.

### Work package 4.4: Migrate refreshes

1. Provider sync.
2. Projection ingestion.
3. Stats ingestion.
4. News and injury ingestion.
5. Schedule and probable ingestion.
6. Identity reconciliation.
7. Data-quality checks.
8. Atomic publication.

### Work package 4.5: Reliability controls

1. Idempotency.
2. Exponential backoff with jitter.
3. Maximum attempts.
4. Dead-letter queue.
5. Manual replay.
6. Cancellation.
7. Distributed locks.
8. Per-tenant quotas.

### Exit gate

- Web workers contain no unbounded compute.
- Worker failures do not lose jobs.
- Duplicate jobs do not duplicate side effects.
- Queue SLOs and alerts are active.

---

## 12. Phase 5: Provider Connector Platform

### Objective

Turn Yahoo-specific integration into a safe public connector system.

### Work package 5.1: Canonical connector interface

Define methods for:

- Authorization URL.
- Callback exchange.
- Token refresh.
- Token revocation.
- League discovery.
- Settings.
- Teams.
- Rosters.
- Standings.
- Schedule.
- Transactions.
- Free agents.
- Lineup writes.
- Add/drop writes.
- Capabilities.
- Sync cursor.

### Work package 5.2: Yahoo connector

1. Move Yahoo-specific translation behind the connector.
2. Bind credentials to a user and provider connection.
3. Encrypt tokens.
4. Store scopes.
5. Track expiration and refresh.
6. Detect revoked access.
7. Add reconnect flow.
8. Add write-scope upgrade flow.

### Work package 5.3: Capability model

Expose:

- Read roster.
- Read transactions.
- Read matchups.
- Write lineup.
- Write add/drop.
- Webhook support.
- Historical access.
- League-settings completeness.

### Work package 5.4: Second connector proof

Implement either:

- A second fantasy provider.
- A robust manual CSV/JSON import connector.

The goal is to prove the domain model is not Yahoo-shaped.

### Work package 5.5: Certification

1. Recorded fixture tests.
2. Sandbox or test-account tests.
3. Token expiration tests.
4. Revocation tests.
5. Rate-limit tests.
6. Schema-change tests.
7. Write confirmation tests.

### Exit gate

- Public users can connect and reconnect without administrator intervention.
- Credentials are isolated.
- Unsupported operations are impossible.
- Yahoo writes are user-authorized and audited.

---

## 13. Phase 6: Data Quality, Lineage, and Publication

### Objective

Make every recommendation traceable to reliable, versioned data.

### Work package 6.1: Raw landing zone

1. Save raw payloads to object storage.
2. Assign immutable object keys.
3. Record checksum, source, request, event time, and ingestion time.
4. Redact secrets and prohibited personal data.
5. Apply retention policies.

### Work package 6.2: Dataset versions

Create a dataset manifest:

- Dataset ID.
- Source versions.
- Transformation version.
- Row counts.
- Quality results.
- Publication status.
- Published time.
- Superseded time.

### Work package 6.3: Atomic publication

1. Ingest into staging tables.
2. Run quality checks.
3. Resolve identities.
4. Build derived views.
5. Publish one manifest transactionally.
6. Point readers to the published version.

### Work package 6.4: Player identity

1. Build source-ID mapping tables.
2. Establish deterministic matching rules.
3. Add confidence and match reason.
4. Quarantine ambiguous matches.
5. Build an operator resolution queue.
6. Track merges and splits.
7. Never reuse retired canonical IDs.

### Work package 6.5: Quality rules

Required checks:

- Primary-key uniqueness.
- Source-ID uniqueness.
- Required fields.
- Valid teams.
- Valid positions.
- Projection ranges.
- Rate-stat ranges.
- Cross-source consistency.
- Freshness.
- Row-count drift.
- Null-rate drift.
- Duplicate-name drift.
- Roster completeness.

### Work package 6.6: User-facing provenance

API responses should include:

- `data_version`.
- `generated_at`.
- `source_summary`.
- `freshness`.
- `coverage`.
- `warnings`.

### Exit gate

- No recommendation uses an unpublished dataset.
- Identity ambiguity is explicit.
- Quality failures block publication.
- Any output can be traced to raw inputs and transformations.

---

## 14. Phase 7: Model Validation and Recommendation Evidence

### Objective

Replace impressive-looking output with empirically defensible output.

### Work package 7.1: Point-in-time archive

For each historical date, store:

- Projection snapshot.
- Player status.
- Team and schedule.
- Fantasy roster.
- League settings.
- Standings.
- Free-agent pool.
- Provider availability.
- Recommendation inputs.

Use at least three completed seasons where legally and practically available.

### Work package 7.2: Projection validation

Measure:

- MAE.
- RMSE.
- Bias.
- Rank correlation.
- Calibration of intervals.
- Performance by player type.
- Performance by playing-time tier.
- Performance by season phase.
- Performance against consensus and simple baselines.

### Work package 7.3: Lineup optimizer replay

For each historical decision:

1. Reconstruct the legal roster.
2. Reconstruct eligible positions.
3. Reconstruct locks and games.
4. Use only point-in-time forecasts.
5. Run the optimizer.
6. Record selected starters and bench.
7. Compare realized utility with:
   - Actual manager lineup.
   - Start-all-active baseline.
   - Projection-rank baseline.
   - Random legal baseline.
8. Report regret and win rate.

### Work package 7.4: Streaming validation

Compare recommendations with:

- Replacement starters.
- Consensus ranks.
- Ownership baseline.
- Simple opponent-quality baseline.

Measure:

- Category utility.
- Bust rate.
- Tail risk.
- Calibration by confidence band.
- Home and away segments.
- Probable-confirmation segments.

### Work package 7.5: Trade validation

1. Reconstruct point-in-time rosters and standings.
2. Measure rest-of-season category utility.
3. Include replacement effects.
4. Include injuries and playing time.
5. Compare with no-trade and consensus-value baselines.
6. Report uncertainty and sample size.
7. Avoid causal claims that observational data cannot support.

### Work package 7.6: Playoff calibration

1. Generate weekly playoff probabilities for historical leagues.
2. Store final outcomes.
3. Compute Brier and log loss.
4. Plot reliability by probability bin.
5. Segment by week.
6. Compare with standings-only and naive baselines.
7. Recalibrate when needed.

### Work package 7.7: Constant calibration

For every registered constant:

1. Identify the dependent outputs.
2. Identify a historical objective.
3. Define the search range.
4. Separate train and holdout seasons.
5. Run sensitivity analysis.
6. Estimate uncertainty.
7. Record calibrated value and provenance.
8. Reject calibration if evidence is weak.

### Work package 7.8: Model registry

Store:

- Version.
- Code commit.
- Data versions.
- Parameters.
- Metrics.
- Limitations.
- Approval.
- Activation time.
- Rollback version.

### Work package 7.9: Recommendation record

Every recommendation stores:

- Recommendation ID.
- User and league scope.
- Model version.
- Dataset version.
- Input snapshot.
- Constraints.
- Alternatives.
- Explanation.
- Confidence.
- Warnings.
- User action.
- Eventual outcome.

### Exit gate

- Every major model beats or matches approved baselines.
- Probability outputs pass calibration thresholds.
- Uncalibrated constants are not presented as calibrated evidence.
- Independent quantitative review is complete.

---

## 15. Phase 8: Security, Privacy, and Abuse Controls

### Objective

Make public access safe before increasing exposure.

### Work package 8.1: Authorization policy

Centralize:

- Route permission.
- Tenant membership.
- League role.
- Team ownership.
- Subscription entitlement.
- Provider capability.
- Admin support access.

### Work package 8.2: Rate limits

Define route classes:

- Normal reads.
- Search.
- Expensive compute.
- AI.
- Provider writes.
- Auth callbacks.
- Billing.
- Admin.

Rate limits must use:

- IP.
- Principal.
- Tenant.
- Route.
- Cost.
- Concurrent jobs.

### Work package 8.3: Secret and credential controls

1. Inventory secrets.
2. Move them to managed secret storage.
3. Encrypt provider credentials.
4. Restrict decryption to required services.
5. Rotate keys.
6. Test revocation.
7. Scan logs and traces for leakage.

### Work package 8.4: Privacy lifecycle

Implement:

- Consent records.
- Data inventory.
- Export.
- Correction.
- Disconnect.
- Deletion.
- Retention.
- Legal hold where required.
- Third-party deletion propagation.

### Work package 8.5: Secure supply chain

Require:

- Dependency scanning.
- Secret scanning.
- SAST.
- Container scanning.
- SBOM.
- Signed build provenance.
- Pinned production dependencies.

### Work package 8.6: External review

1. Threat model.
2. Penetration test.
3. Tenant-isolation assessment.
4. OAuth review.
5. Roster-write review.
6. Remediation and retest.

### Exit gate

- No unresolved Critical or High security finding.
- Privacy lifecycle is tested.
- Abuse controls are active.
- External retest passes.

---

## 16. Phase 9: Billing and Entitlement Hardening

### Objective

Make payment state financially and operationally trustworthy.

### Work package 9.1: Event ledger

Store:

- Stripe event ID.
- Type.
- Created time.
- Received time.
- Applied time.
- Payload hash.
- Result.
- Related customer and subscription.

Reject duplicates and stale events.

### Work package 9.2: Reconciliation

Run a scheduled job that:

1. Lists changed Stripe subscriptions.
2. Compares local state.
3. Repairs safe drift.
4. Queues unsafe drift for review.
5. Alerts on material discrepancies.

### Work package 9.3: Entitlement model

Replace broad tier checks with capabilities:

- Advanced lineup optimization.
- Trade simulation.
- Draft simulation.
- AI managed spend.
- Provider writes.
- Historical exports.
- League count.
- Job concurrency.

### Work package 9.4: Lifecycle tests

Test:

- Trial start.
- Trial conversion.
- Payment failure.
- Grace period.
- Cancellation now.
- Cancellation at period end.
- Reactivation.
- Refund.
- Dispute.
- Price change.
- Coupon.
- Webhook replay.
- Out-of-order events.

### Work package 9.5: Live certification

1. Complete live-mode purchase.
2. Confirm entitlement.
3. Open portal.
4. Cancel.
5. Confirm entitlement change.
6. Reconcile.
7. Verify audit records.

### Exit gate

- Local entitlement matches Stripe.
- Duplicate and stale events are harmless.
- Support can diagnose billing without database edits.
- Live-mode certification passes.

---

## 17. Phase 10: Reliability, Resilience, and Recovery

### Objective

Ensure failure is bounded, visible, and recoverable.

### Work package 10.1: Dependency policy

For PostgreSQL, Redis, Yahoo, MLB, projection sources, Clerk, Stripe, and AI providers define:

- Timeout.
- Retry eligibility.
- Attempts.
- Backoff.
- Circuit threshold.
- Recovery threshold.
- Fallback.
- User-visible behavior.

### Work package 10.2: Deadlines

1. Set an overall request deadline.
2. Reserve serialization time.
3. Propagate remaining time.
4. Cancel downstream work when possible.
5. Return timeout errors with retry guidance.

### Work package 10.3: Health endpoints

- `/livez`: process is alive.
- `/readyz`: required dependencies are usable.
- `/health/details`: authenticated operator detail.

Readiness must check:

- PostgreSQL.
- Redis.
- migrations.
- required configuration.
- queue availability.

Provider outages should not necessarily remove readiness, but must appear in diagnostics and feature status.

### Work package 10.4: Backups and restore

1. Define RPO and RTO.
2. Enable point-in-time recovery.
3. Back up object storage metadata.
4. Restore into an isolated environment.
5. Run integrity tests.
6. Record actual restore time.
7. Repeat quarterly.

### Work package 10.5: Chaos tests

Inject:

- Database latency.
- Redis outage.
- Provider 429.
- Provider timeout.
- Malformed provider payload.
- Stripe replay.
- Clerk JWKS outage.
- Worker crash.
- Partial data publication.
- Disk or memory pressure.

### Exit gate

- Required dependencies control readiness.
- Recovery meets RPO and RTO.
- Fault injection produces the intended user and operator behavior.

---

## 18. Phase 11: Observability, SLOs, and Incident Operations

### Objective

Give one operator enough evidence to run the product safely.

### Work package 11.1: Structured telemetry

Every request log includes:

- Timestamp.
- Level.
- Service.
- Release.
- Environment.
- Request ID.
- Route template.
- Status.
- Duration.
- Principal hash.
- Tenant ID.
- League ID.
- Error code.

Never log:

- Tokens.
- Provider secrets.
- Raw authorization headers.
- AI keys.
- Full payment payloads.

### Work package 11.2: Tracing

Trace:

- Request.
- Auth.
- Repository calls.
- Provider calls.
- Queue submission.
- Job execution.
- Model execution.
- Stripe.
- Clerk.
- AI provider.

### Work package 11.3: Metrics

Required metric families:

- Request rate, error rate, duration.
- Database pool and query latency.
- Queue depth, age, attempts, failures.
- Worker CPU and memory.
- Provider success, latency, 429, auth failures.
- Data freshness and quality.
- Recommendation counts and abstentions.
- Billing webhook lag and drift.
- AI spend and denials.
- Cache hit ratio.

### Work package 11.4: SLOs

Initial targets:

- API availability: 99.9% monthly for authenticated read APIs.
- Mutation correctness: 99.99% without duplicate side effects.
- Read p95: under 500 ms for cached standard reads.
- Interactive compute acknowledgement: under 1 second.
- Interactive job completion p95: under 30 seconds unless explicitly classified otherwise.
- Provider sync freshness: within the documented source window.
- Queue oldest interactive job: under 10 seconds.
- Billing webhook application p95: under 60 seconds.
- Cross-tenant incidents: zero.

### Work package 11.5: Runbooks

Create runbooks for:

- API latency.
- Elevated 5xx.
- Database saturation.
- Redis outage.
- Queue backlog.
- Provider outage.
- Stale data.
- Failed publication.
- Billing drift.
- Auth outage.
- AI spend spike.
- Suspected tenant leak.
- Bad model release.

### Exit gate

- Dashboards and alerts are active.
- Every alert links to a runbook.
- A simulated incident is detected and resolved using telemetry.
- Error budgets govern release decisions.

---

## 19. Phase 12: Performance and Capacity Certification

### Objective

Measure and enforce a safe public operating envelope.

### Work package 12.1: Endpoint budgets

Classify:

- Static metadata.
- Cached personalized read.
- Uncached personalized read.
- Search.
- Job submission.
- Mutation.
- Streaming response.

Define for each:

- p50.
- p95.
- p99.
- timeout.
- maximum response bytes.
- maximum database queries.
- maximum provider calls.

### Work package 12.2: Load profiles

Test:

1. Normal daily use.
2. Morning lineup peak.
3. Sunday roster peak.
4. Trade deadline peak.
5. Draft-room burst.
6. AI burst.
7. Provider slowdown.
8. Cold cache.

### Work package 12.3: Cache architecture

1. Identify immutable global data.
2. Identify versioned league data.
3. Identify user-private data.
4. Define TTL and invalidation.
5. Include model and dataset versions.
6. Measure hit ratio.
7. Prevent cross-tenant keys.

### Work package 12.4: Query certification

1. Capture top queries.
2. Add `EXPLAIN ANALYZE` evidence.
3. Add missing indexes.
4. Remove N+1 patterns.
5. Bound result sets.
6. Monitor slow queries.

### Work package 12.5: Capacity policy

Document:

- Maximum web concurrency per replica.
- Maximum worker concurrency.
- Memory per simulation.
- Simulations per minute.
- Provider calls per tenant.
- Safe queue depth.
- Scale-up and scale-down thresholds.

### Exit gate

- Load and soak tests meet budgets.
- No tenant can starve others.
- Capacity limits and backpressure are active.
- Cost per active league is measured.

---

## 20. Phase 13: Testing, CI, and Release Engineering

### Objective

Make every critical backend property a required automated gate.

### Work package 13.1: Test pyramid

Maintain:

- Unit tests for pure analytical logic.
- Contract tests for engine boundaries.
- Repository integration tests with PostgreSQL.
- Queue integration tests with Redis.
- API integration tests.
- Provider connector tests.
- End-to-end authenticated tests.
- Load and chaos tests.

### Work package 13.2: Risk-weighted coverage

Require higher thresholds for:

- Authorization.
- Tenant scope.
- Billing.
- Provider writes.
- Job idempotency.
- Data publication.
- Recommendation record creation.

Use mutation testing selectively to detect weak assertions.

### Work package 13.3: Migration tests

1. Create a production-like fixture database.
2. Run upgrade.
3. Verify application reads and writes.
4. Verify constraints.
5. Measure duration and locks.
6. Run rollback where supported.

### Work package 13.4: Contract checks

1. Regenerate OpenAPI.
2. Fail on uncommitted differences.
3. Run breaking-change detection.
4. Generate TypeScript.
5. Compile the frontend against it.
6. Run examples as tests.

### Work package 13.5: Security checks

Require:

- Secret scan.
- Dependency audit.
- SAST.
- Container scan.
- SBOM.
- License policy.

### Work package 13.6: Release pipeline

1. Build immutable artifact.
2. Deploy to staging.
3. Run migrations.
4. Run authenticated smoke tests.
5. Run selected model checks.
6. Deploy canary.
7. Observe.
8. Promote or roll back.

### Exit gate

- Main is green.
- The OpenAPI snapshot is current.
- All critical integration systems run in CI or scheduled certification.
- Rollback has been tested.
- Production smoke tests cover personalized reads and safe writes.

---

## 21. Phase 14: Endpoint-Family Remediation Matrix

| Family | Current risk | Required remediation |
|---|---|---|
| Team and matchup | Default-league and team-name assumptions | Stable IDs, scoped repositories, freshness and partial-data metadata |
| Free agents and players | Identity incompleteness | Canonical player map, source coverage, pagination, dataset versions |
| Lineup optimize | Synchronous compute and provider-write risk | Durable job, point-in-time evidence, legal constraints, idempotent confirmed write |
| Standings and playoff | Probability validation gap | Historical calibration, model version, uncertainty, async simulation |
| Closers and leaders | Mixed-source freshness | Source lineage, atomic datasets, quality and freshness warnings |
| Streaming and schedule | Provider fan-out and probabilistic trust | Cached versioned schedules, confirmed probable state, calibrated scorecards |
| Punt and trade | Recommendation evidence gap | Stored alternatives, sensitivity, baseline comparison, uncertainty |
| Draft | Burst traffic and synchronous simulation | Priority jobs, session state, replay, capacity certification |
| League rosters | Provider and tenant sensitivity | Scoped connector, stable team IDs, cache isolation, authorization |
| Roster writes | Unauthorized or duplicate side effects | Per-user write OAuth, capability check, confirmation, idempotency, audit |
| Billing | Event drift | Event ledger, dedupe, reconciliation, capability entitlements |
| Admin | Privileged access | Least privilege, support session, reason, expiry, immutable audit |
| Chat and AI | Spend, grounding, and tenant risk | Scoped tools, evidence citations, prompt defense, quotas, evals, audit |
| Health | False green | Liveness, readiness, dependency diagnostics, release version |

Every endpoint must also document:

- Authentication.
- Authorization.
- Tenant scope.
- Entitlement.
- Rate limit.
- Timeout.
- Retry safety.
- Freshness.
- Degraded behavior.
- SLO.

---

## 22. Required Schema Additions

### Identity and tenancy

- `tenants`.
- `tenant_users`.
- `league_memberships`.
- `fantasy_teams`.
- `support_access_sessions`.

### Providers

- `provider_connections`.
- `provider_credentials`.
- `provider_leagues`.
- `provider_team_links`.
- `provider_sync_cursors`.
- `provider_capabilities`.

### Data platform

- `raw_ingestions`.
- `dataset_versions`.
- `dataset_sources`.
- `data_quality_runs`.
- `data_quality_findings`.
- `player_source_ids`.
- `identity_resolution_cases`.

### Jobs

- `jobs`.
- `job_attempts`.
- `job_events`.
- `dead_letter_jobs`.

### Recommendations and models

- `model_versions`.
- `model_metrics`.
- `constant_versions`.
- `recommendations`.
- `recommendation_alternatives`.
- `recommendation_outcomes`.
- `feature_snapshots`.

### Billing

- `billing_events`.
- `entitlements`.
- `billing_reconciliation_runs`.

### Operations

- `audit_events`.
- `idempotency_keys`.
- `data_exports`.
- `deletion_requests`.
- `incident_markers`.

All tenant-owned tables must have:

- Tenant foreign key.
- Created and updated timestamps.
- Appropriate deletion policy.
- Audit or version strategy.
- Row-level security policy.

---

## 23. Mandatory Test Matrix

### API

- Success.
- Validation failure.
- Missing auth.
- Invalid auth.
- Wrong tenant.
- Wrong league.
- Missing entitlement.
- Rate limit.
- Timeout.
- Dependency unavailable.
- Partial optional dependency.
- Idempotent replay.
- Breaking contract detection.

### Database

- Fresh migration.
- Upgrade from previous release.
- Constraint enforcement.
- Transaction rollback.
- Row-level security.
- Cross-tenant foreign-key rejection.
- Backup restore.
- Production-volume query plan.

### Workers

- Success.
- Retry.
- Duplicate delivery.
- Worker crash.
- Timeout.
- Cancellation.
- Dead-letter.
- Replay.
- Tenant quota.
- Queue priority.

### Providers

- OAuth success.
- Expired token.
- Revoked token.
- Refresh failure.
- 429.
- 5xx.
- Malformed response.
- Partial response.
- Schema drift.
- Duplicate event.
- Unsupported write.
- Confirmed write.

### Billing

- Duplicate webhook.
- Stale webhook.
- Missing metadata.
- Reconciliation repair.
- Payment failure.
- Grace period.
- Cancellation.
- Refund.
- Dispute.
- Customer mismatch.

### Analytics

- Determinism.
- Point-in-time isolation.
- Leakage detection.
- Baseline comparison.
- Calibration.
- Sensitivity.
- Missing-data abstention.
- Stale-data warning.
- Cross-tool consistency.
- Version rollback.

### Security

- IDOR.
- Cross-tenant access.
- Privilege escalation.
- Rate-limit bypass.
- Oversized request.
- Prompt injection into AI tools.
- Secret redaction.
- Audit completeness.
- Provider-write replay.

---

## 24. Backend Metrics and SLO Dashboard

### API

- Requests by route and status.
- p50, p95, p99 latency.
- Active requests.
- Timeouts.
- Response bytes.
- Rate-limit denials.

### Database

- Connections.
- Pool wait.
- Query latency.
- Locks.
- Deadlocks.
- Replication lag if used.
- Storage growth.
- Backup age.

### Jobs

- Queue depth.
- Oldest job.
- Completion time.
- Retry rate.
- Failure rate.
- Dead-letter count.
- Cancellation rate.

### Providers

- Success rate.
- 401 and 403.
- 429.
- 5xx.
- Latency.
- Token refresh failures.
- Sync freshness.

### Data

- Dataset publication age.
- Quality pass rate.
- Missing canonical IDs.
- Missing teams.
- Identity quarantine count.
- Row-count drift.

### Models

- Recommendations by type.
- Abstention rate.
- Confidence distribution.
- Acceptance and rejection.
- Outcome capture rate.
- Calibration and baseline deltas.
- Active model version.

### Billing

- Checkout success.
- Webhook lag.
- Webhook failure.
- Drift count.
- Active subscriptions.
- Entitlement denials.

### AI

- Requests.
- Spend.
- Tokens.
- Latency.
- Provider errors.
- Grounding coverage.
- Safety denials.

---

## 25. Release Gates

### Invite-only external beta

Required:

- PostgreSQL.
- Authenticated tenant-scoped data.
- Provider connection lifecycle.
- Standard errors.
- Readiness checks.
- Rate limits.
- Basic metrics and alerts.
- Backup restore.
- No Critical or High security finding.

### Paid beta

Required:

- Billing event dedupe and reconciliation.
- Explicit entitlements.
- Support tooling.
- Privacy export and deletion.
- Worker system for expensive tasks.
- Operational SLO dashboard.
- Incident runbooks.

### Limited public launch

Required:

- Multi-league self-service.
- Row-level tenant protection.
- External penetration test.
- Load and soak certification.
- Provider-write safety if writes are enabled.
- Data licensing approval.
- Historical validation for marketed claims.
- Disaster-recovery drill.

### General availability

Required:

- 30 consecutive days meeting SLOs.
- Zero unresolved Critical or High findings.
- Model registry and production outcome monitoring.
- Demonstrated billing integrity.
- Complete connector and privacy lifecycle.
- Independent security and quantitative review.

---

## 26. Definition of 100/100

The backend receives 100/100 only when:

1. PostgreSQL is the production source of truth.
2. Redis and durable workers handle expensive and scheduled work.
3. Every personalized row, cache, job, and request is tenant-scoped.
4. Database-enforced isolation is active.
5. Cross-tenant external testing passes.
6. Provider authorization is per user and self-service.
7. Writes are capability-checked, confirmed, idempotent, and audited.
8. OpenAPI is current, secured, versioned, and compatibility-tested.
9. Errors and degraded states are unambiguous.
10. Data publication is atomic, versioned, quality-gated, and traceable.
11. Canonical player identity meets approved completeness thresholds.
12. Every analytical claim has point-in-time historical validation.
13. Every probability claim passes calibration gates.
14. Every model and constant has versioned provenance.
15. Billing is deduplicated, reconciled, and live-certified.
16. Rate limits, quotas, abuse controls, and secret controls are active.
17. Liveness, readiness, tracing, metrics, alerts, and runbooks are active.
18. Backup restoration meets RPO and RTO.
19. Load, soak, migration, chaos, and rollback tests pass.
20. Production meets SLOs for 30 consecutive days.
21. No unresolved Critical or High finding remains.
22. Independent security and quantitative reviews are complete.

---

## 27. Recommended Execution Order

1. Phase 0: Baseline and evidence registry.
2. Phase 1: API contract and failure architecture.
3. Phase 2: PostgreSQL and repositories.
4. Phase 3: Tenant-native data model.
5. Phase 4: Durable workers.
6. Phase 5: Provider connector platform.
7. Phase 6: Data quality and lineage.
8. Phase 8: Security, privacy, and abuse controls.
9. Phase 9: Billing hardening.
10. Phase 10: Reliability and recovery.
11. Phase 11: Observability and SLOs.
12. Phase 12: Performance certification.
13. Phase 7: Model validation, running in parallel after historical data foundations exist.
14. Phase 13: Release engineering, expanded continuously through every phase.
15. Public release gates.

The sequencing intentionally places PostgreSQL, tenancy, workers, data provenance, and security ahead of broad public growth. Model-validation work should begin as soon as point-in-time data can be captured, but its final certification depends on historical datasets and production outcome capture.

---

## 28. Final Backend Critique

HEATER's backend is not weak. It is incomplete in the exact places that distinguish an advanced private application from a dependable public SaaS:

- It has an API, but not full API governance.
- It has authentication, but not a tenant-native data plane.
- It has billing, but not complete financial reconciliation.
- It has simulations, but not enough real-world calibration.
- It has fallbacks, but not consistently honest degraded semantics.
- It has a migration seam, but not a migrated production database.
- It has scheduling and refresh logic, but not durable distributed jobs.
- It has logs, but not an operational observability system.
- It has many tests, but not all production-class evidence.

The correct strategy is not a rewrite. The correct strategy is to preserve the engines, complete the platform seams already started, and make every public-product property measurable and enforceable.

This document defines the work required to move the backend from **54/100** to a defensible **100/100**. It does not authorize or implement that work.
