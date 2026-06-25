# HEATER Public SaaS 100/100 Remediation Program

**Planning baseline:** June 24, 2026
**Current strict public-SaaS grade:** 51/100
**Target:** 100/100 in every audit category
**Role:** Master quality and release-gate overlay for the existing M0-M6 roadmap
**Operating model:** Connor plus Codex, using managed services and low-operations architecture
**Document status:** Plan only. This document does not authorize implementation by itself.

---

## 1. Purpose

HEATER already has substantial fantasy-baseball functionality, a large Python test suite, a FastAPI API, a Next.js frontend, Clerk authentication, Stripe foundations, an AI assistant, and deployed beta infrastructure. Those assets make it a capable private or invite-only product. They do not yet make it a trustworthy, scalable, legally cleared, self-service public SaaS product.

The strict audit graded HEATER as follows:

| Category | Current score | Target score |
|---|---:|---:|
| Core analytical value | 90 | 100 |
| Onboarding and adoption | 43 | 100 |
| Multi-tenancy and isolation | 30 | 100 |
| Reliability and scalability | 40 | 100 |
| Security and privacy | 63 | 100 |
| Billing and entitlements | 50 | 100 |
| Platform integrations | 40 | 100 |
| Maintainability and delivery | 75 | 100 |
| Data quality | 50 | 100 |
| Recommendation trustworthiness | 38 | 100 |
| Prediction accuracy | 25 | 100 |
| UI/UX/design | 63 | 100 |
| AI features | 50 | 100 |
| **Weighted total** | **51/100** | **100/100** |

This program turns each critique into implementation instructions, measurable acceptance criteria, operational evidence, and release gates. It does not replace existing plans. It governs how the existing M0-M6 roadmap and all later feature plans must be executed and certified.

The principal gaps are:

1. Single-league assumptions and incomplete tenant isolation.
2. SQLite and synchronous execution paths that cannot safely support public scale.
3. Incomplete self-service provider connection and import workflows.
4. Unresolved commercial rights for external data.
5. Incomplete historical validation of projections, simulations, and recommendations.
6. Default constants presented through authoritative-looking outputs despite lacking calibration evidence.
7. Insufficient data lineage, freshness, completeness, reconciliation, and replay controls.
8. Missing recommendation evidence, uncertainty, outcome measurement, and contradiction handling.
9. Incomplete security, privacy, disaster recovery, and external assurance.
10. Missing end-to-end customer onboarding, support, account lifecycle, and billing operations.
11. Insufficient frontend automated testing, accessibility verification, responsive validation, and performance enforcement.
12. AI features without a comprehensive grounding, factuality, tenant-safety, recommendation-consistency, and usefulness evaluation system.

---

## 2. Governing Rules

### 2.1 Definition of 100/100

A category receives 100/100 only when all of the following are true:

- Every mandatory capability in that category is implemented.
- Automated tests cover successful, invalid, degraded, abusive, and recovery behavior.
- Production telemetry demonstrates the required service targets for at least 30 consecutive days.
- Required documentation, runbooks, and ownership are complete.
- No unresolved Critical or High finding applies to the category.
- Required independent review is complete.
- Evidence is reproducible from committed code, immutable reports, production dashboards, or signed external reports.
- A feature being present does not count as correct unless its behavior is measured.
- A passing synthetic test does not count as production proof when the real data path is different.
- A graceful fallback may not fabricate data or present an error as a valid analytical result.
- A limitation may not be hidden in implementation notes when it can materially affect a user's decision.
- A metric cannot be called calibrated, predictive, accurate, or trustworthy without an approved validation result.

### 2.2 Current-State Authority

`Bugs_June26.md` and `Outstanding_June26.md` are historical audit inputs. The current `master` commit records both associated remediation campaigns as complete. Phase 0 must independently re-audit the current commit and classify every historical finding as fixed, open, obsolete, accepted, or unverifiable.

The order of authority is:

1. Current production behavior and current committed code.
2. Current automated tests and generated contracts.
3. Current `AGENTS.md` project instructions.
4. Approved current specs and plans.
5. Historical audit reports and stale roadmap statements.

No engineer should reimplement a finding solely because an older report lists it. The current implementation must be inspected first.

### 2.3 Sequencing Rules

1. Commercial data rights and privacy foundations precede public commercialization.
2. PostgreSQL precedes full multi-tenancy.
3. Multi-tenancy and tenant-isolation proof precede open public signup.
4. Per-user provider authorization precedes enabling public league writes.
5. Point-in-time historical datasets precede calibration or accuracy claims.
6. Prediction validation precedes probability, confidence, and performance marketing claims.
7. Recommendation evidence and constraints precede automated execution.
8. Security and privacy gates apply before each expansion to a wider release ring.
9. Public launch remains blocked until all required independent audits pass.
10. Streamlit remains available only as a controlled fallback until the Next.js product passes cutover gates.
11. Any phase that changes the live data path requires a rollback plan and an owner approval gate.
12. All work must bias toward managed infrastructure and solo-operator maintainability.

### 2.4 Effort Scale

| Band | Meaning |
|---|---|
| XS | Less than one focused engineering day |
| S | Several focused engineering days |
| M | One coherent engineering slice involving multiple files or components |
| L | A multi-slice subsystem or migration |
| XL | A program-level change requiring phased delivery and production observation |

The plan intentionally avoids speculative calendar dates. Work is ordered by dependency and release risk.

### 2.5 Managed Infrastructure Default

Use the lowest-operational-overhead managed services compatible with the requirements:

- Vercel for the Next.js frontend.
- Railway for FastAPI and background worker processes.
- Managed PostgreSQL.
- Managed Redis.
- Arq-compatible background workers.
- S3-compatible managed object storage.
- Sentry for application errors, traces, and release correlation.
- PostHog for privacy-controlled product analytics, experiments, and feature flags.
- Clerk production instance for identity.
- Stripe for subscriptions, invoices, tax integration, and customer portal.
- GitHub Actions for required checks and release gates.

Every provider must have documented export, backup, and exit procedures. Critical HEATER domain logic must remain in the repository, not only in a vendor workflow.

### 2.6 Release Rings

1. Local development.
2. Automated CI.
3. Internal staging.
4. Owner-only production.
5. Existing 12-user league.
6. Invite-only external beta.
7. Limited public launch.
8. General availability.

Every implementation plan derived from this program must state the highest release ring it unlocks and the exact evidence needed to advance.

---

## 3. Category Certification Standards

### 3.1 Core Analytical Value

HEATER must prove that its tools improve fantasy-baseball decisions rather than merely produce attractive scores.

#### Required capabilities

- Every primary workflow ends in a clear, legal, feasible action or an explicit abstention.
- Optimizer, streaming, trade, draft, standings, matchup, free-agent, and AI systems use canonical player, league, category, roster, and constraint definitions.
- Recommendations account for roster legality, transaction limits, category direction, schedule, opponent, injury state, playing time, league rules, data freshness, and provider capabilities.
- Each major tool has a defined user outcome and a measurable offline or production success metric.
- No canonical live player surface uses fabricated or generic production data.
- Every recommendation-producing surface exposes evidence and uncertainty.
- At least three simple benchmark strategies exist for each major recommendation family.
- HEATER must beat or be statistically non-inferior to approved baselines on predefined holdout metrics.
- Beta users must be able to understand and act on the result without expert intervention.

#### Certification metrics

- At least 90% task completion for the five highest-value workflows.
- At least 80% of users correctly explain the proposed action during usability testing.
- At least 70% of viewed recommendations receive an explicit accept, reject, save, compare, or explain interaction.
- Zero impossible roster actions in the certification corpus.
- Zero known material contradictions between tools using identical inputs.
- All live player details originate from current API data with source and freshness metadata.
- Every major tool has a documented baseline and utility measure.

### 3.2 Onboarding and Adoption

#### Required capabilities

- A new user can register, connect or import a league, identify their team, validate settings, and reach a personalized dashboard without administrator intervention.
- Demo mode is explicitly labeled and cannot be confused with live league data.
- Connection failures provide actionable recovery instructions and stable reason codes.
- Empty, loading, degraded, stale, unauthorized, unlinked, offline, and error states are distinct.
- Users can reconnect providers, switch leagues, disconnect leagues, export data, and delete their account.
- Contextual guidance explains unfamiliar fantasy and statistical concepts.
- Support and feedback channels have explicit ownership and response targets.
- Onboarding progress is resumable.
- The product provides a useful first recommendation as early as safely possible.

#### Certification metrics

- Median signup-to-first-personalized-recommendation under five minutes.
- At least 80% onboarding completion without support.
- At least 70% first-week activation.
- Less than 5% unresolved connector failure rate.
- Less than 2% of sessions encounter an unexplained dead end.
- Support response target under one business day during beta.
- Every onboarding failure event has a stable reason code and recovery path.

### 3.3 Multi-Tenancy and Isolation

#### Required capabilities

- Every user-owned record carries `tenant_id`.
- Every league-owned record carries `tenant_id` and `league_id`.
- Authorization is derived server-side from verified identity and membership.
- No endpoint trusts a client-supplied team name for authorization.
- Personalized services use request-scoped repositories.
- Tenant context propagates through requests, caches, workers, scheduled jobs, exports, AI tools, logs, and object storage.
- Database constraints prevent common cross-tenant mistakes.
- Users can belong to multiple leagues and safely switch active context.
- Administrative support access is explicit, temporary, audited, and revocable.

#### Certification metrics

- Zero cross-tenant exposure in automated adversarial tests.
- Zero cross-tenant exposure in an independent penetration test.
- 100% of tenant-owned tables are covered by schema and repository guards.
- 100% of background jobs contain tenant and league context.
- 100% of privileged impersonation actions generate append-only audit records.
- No production endpoint accepts team identity as a trusted authorization input.

### 3.4 Reliability and Scalability

#### Required capabilities

- Public request paths no longer depend on a single SQLite writer.
- Expensive simulations, imports, provider refreshes, exports, and AI research run asynchronously.
- Upstream failures are bounded by timeout, retry, circuit-breaker, stale-data, and user-notification policies.
- Production has documented SLOs, capacity limits, backups, restore procedures, and incident runbooks.
- Deployments support rollback without data loss.
- Cache keys include tenant, league, season, scoring configuration, data snapshot, and model version where relevant.
- Scheduled jobs are idempotent and observable.

#### Certification metrics

- Frontend availability at least 99.9% monthly.
- API availability at least 99.9% monthly.
- Cached read p95 under 500 ms.
- Standard uncached read p95 under two seconds.
- Async job creation p95 under 500 ms.
- At least 95% of normal analytical jobs finish within 30 seconds.
- Error rate below 0.5%, excluding correct 4xx responses.
- Recovery point objective of five minutes or better.
- Recovery time objective of 60 minutes or better.
- Successful quarterly restore test.
- Load test supports twice forecast launch traffic with 30% headroom.
- No unbounded external call or simulation remains on an interactive request path.

### 3.5 Security and Privacy

#### Required capabilities

- Clerk production keys are used in production.
- All private league data requires authenticated and authorized access.
- Writes require authentication, authorization, tenant context, provider scope, and idempotency.
- Secrets are stored only in approved secret stores.
- Sensitive data is encrypted in transit and at rest.
- Provider credentials use separate envelope encryption and rotation.
- Data retention, export, deletion, and consent policies are implemented.
- Security headers, dependency scanning, static analysis, secret scanning, license scanning, and container scanning run continuously.
- Threat models cover account takeover, cross-tenant access, provider-token theft, billing abuse, malicious imports, prompt injection, and AI data leakage.
- Incident-response and vulnerability-disclosure processes exist.

#### Certification metrics

- Zero Critical or High findings from independent penetration testing.
- Zero committed secrets remaining after history remediation.
- 100% of sensitive-data access is authenticated and auditable.
- Account deletion completes within the published period.
- Critical vulnerability remediation begins within 24 hours.
- High vulnerability remediation begins within 72 hours.
- Annual threat-model review and penetration test.
- Every security-sensitive endpoint has negative authorization tests.
- Applicable OWASP ASVS Level 2 controls are mapped and verified.

### 3.6 Billing and Entitlements

#### Required capabilities

- Stripe events are signature-verified, deduplicated, ordered safely, and replayable.
- Subscription state has explicit source-of-truth rules.
- Entitlement checks cannot activate before checkout is operational.
- Trials, upgrades, downgrades, cancellations, refunds, payment failures, grace periods, disputes, and reactivation are handled.
- Customer portal, invoices, tax, receipts, and support workflows are complete.
- Managed AI usage and paid features have enforceable server-side budgets.
- Financial events reconcile against Stripe daily.

#### Certification metrics

- Zero duplicate entitlement changes during replay tests.
- Zero unauthorized paid-feature grants in adversarial tests.
- Zero valid subscribers incorrectly denied longer than the incident SLO.
- Daily reconciliation identifies all Stripe/application mismatches.
- Billing webhook processing success above 99.99%.
- Refund and cancellation workflows pass end to end.
- Pricing, renewal, trial, and cancellation terms are visible before purchase.
- All paid plans are enforced server-side.

### 3.7 Platform Integrations

The approved capability strategy is:

- Yahoo: native OAuth connection with approved read and write scopes.
- ESPN: approved partner API when available; otherwise secure guided import.
- CBS: approved partner API when available; otherwise secure guided import.
- Sleeper: adapter-ready, but baseball support remains disabled until officially supported.
- Manual import: validated and versioned templates.

#### Required capabilities

- A canonical `LeagueConnector` interface separates provider behavior from domain logic.
- Provider capabilities are explicitly advertised.
- Unsupported features are disabled rather than simulated.
- OAuth state, refresh, revocation, retries, rate limits, and credential rotation are implemented.
- Imports support preview, validation, correction, confirmation, and rollback.
- Connector health and freshness are visible.
- Provider terms and commercial rights are approved.

#### Certification metrics

- Yahoo connection success above 95% for valid credentials.
- Yahoo refresh success above 99%, excluding confirmed upstream outages.
- Import validation catches all malformed certification fixtures.
- Zero unsupported provider capabilities are presented as available.
- All connector errors use stable reason codes.
- Provider disconnect revokes or deletes credentials immediately.
- Public launch includes Yahoo and at least one approved guided-import path.

### 3.8 Maintainability and Delivery

#### Required capabilities

- Backend, frontend, workers, migrations, and contracts have required CI checks.
- Next.js has unit, component, integration, accessibility, and browser E2E tests.
- OpenAPI generation and frontend type generation are enforced.
- Schema migrations are tested against real PostgreSQL.
- Architectural boundaries are mechanically guarded.
- Documentation is generated or verified against code where possible.
- Optional analytical dependencies have scheduled full-path CI coverage.
- Release artifacts are reproducible.

#### Certification metrics

- Required checks block merges.
- Zero known flaky required tests.
- Frontend line and branch coverage floors are established and enforced.
- Backend coverage cannot decline below the approved floor.
- All public endpoints have contract and authorization tests.
- All critical workflows have E2E coverage.
- Dependency and container scans run on each release.
- Mean time to restore a failed deployment is under 30 minutes.
- Documentation-drift checks cover endpoint counts, generated types, schema revisions, environment variables, and routes.
- No unowned production subsystem or undocumented deployment step remains.

### 3.9 Data Quality

#### Required capabilities

- Every displayed value has source, observation time, effective time, transformation version, and freshness state.
- Player identity uses a canonical cross-provider mapping with collision handling.
- Data completeness, range, type, referential-integrity, and temporal-consistency rules run during ingestion.
- Provider disagreements are reconciled under documented precedence rules.
- Stale or partial data is visible to users.
- Historical snapshots support replay and backtesting.
- Commercial rights exist for every production source.
- No default constant is represented as calibrated.

#### Certification metrics

- At least 99.5% canonical MLB identity coverage for fantasy-relevant active players.
- At least 99% team and position completeness.
- At least 99.5% of successful records pass schema and range validation.
- Zero unexplained duplicate canonical players.
- Core live data meets published freshness SLAs at least 99% of the time.
- All failed refreshes generate alerts and user-visible degradation.
- Every model input can be traced to an immutable source snapshot.
- Every calibrated constant stores dataset, method, metrics, reviewer, and timestamp.
- Reconciliation reports run after each provider refresh.

### 3.10 Recommendation Trustworthiness

#### Required capabilities

- Recommendations expose the action, expected benefit, uncertainty, assumptions, freshness, model version, and alternatives.
- Constraints are validated against league rules and provider capabilities.
- Users can inspect why a recommendation changed.
- Recommendations are immutable and replayable.
- Outcomes and feedback connect to the originating recommendation.
- HEATER abstains when data or confidence is insufficient.
- Conflicting recommendations are detected and reconciled.

#### Certification metrics

- 100% of production recommendations have an evidence record.
- 100% identify model, data snapshot, league context, and constraints.
- Zero impossible actions in the certification suite.
- At least 95% deterministic replay equivalence with identical inputs.
- All probability language maps to measured calibration bands.
- All low-confidence recommendations are labeled or withheld.
- Outcome tracking covers at least 90% of recommendations with observable results.
- Independent reviewers can reproduce sampled recommendations.
- Structured beta trust score at least 85/100.

### 3.11 Prediction Accuracy

#### Required capabilities

- Backtests use historical point-in-time information only.
- Training, calibration, validation, and final holdout periods are separate.
- Optimizer evaluation runs the actual optimizer.
- Synthetic fixtures are not used as production accuracy evidence.
- Models are compared with simple published baselines.
- Calibration, discrimination, error, and decision utility are measured separately.
- Statistical uncertainty accompanies claims.
- An independent statistical reviewer approves the methodology.

#### Certification metrics

- Zero look-ahead leakage in independent review.
- Calibration slope between 0.90 and 1.10 for probability models.
- Calibration intercept within plus or minus 0.05.
- Expected calibration error at most 0.03 when sample size permits.
- Brier skill at least 10% better than the approved naive baseline for probability models.
- Point forecasts beat or are statistically non-inferior to the approved industry baseline in every primary metric.
- Recommendation policy improves simulated category or matchup utility over baseline by a predefined statistically supported margin.
- Confidence intervals accompany all headline metrics.
- Model cards document intended use, exclusions, weaknesses, and retraining triggers.
- Drift alerts fire when production error exceeds the approved threshold.

### 3.12 UI/UX/Design

#### Required capabilities

- Desktop, tablet, and mobile navigation fit without clipping.
- Interactive targets are at least 44 by 44 CSS pixels unless an approved exception exists.
- Primary mobile content appears without excessive hero displacement.
- Loading, stale, empty, unauthorized, unlinked, offline, degraded, and error states are distinct.
- No live failure silently substitutes mock data.
- All major workflows are keyboard operable.
- Color is never the only information carrier.
- The product passes WCAG 2.2 AA.
- Core Web Vitals are good at the 75th percentile.
- Design remains consistent with the Combustion identity.

#### Certification metrics

- Zero horizontal page overflow at supported widths.
- Zero clipped global-navigation actions.
- At least 90% task success in moderated testing.
- System Usability Scale score at least 85.
- Automated accessibility checks report zero serious or critical violations.
- Manual keyboard, screen-reader, zoom, contrast, and reduced-motion review passes.
- LCP at most 2.5 seconds, INP at most 200 ms, and CLS at most 0.1 at p75.
- No production page displays fabricated data without an explicit demo label.
- Every primary control has a visible focus state and accessible name.

### 3.13 AI Features

#### Required capabilities

- Bubba uses canonical HEATER services instead of rebuilding domain logic in prompts.
- Tool access is tenant-scoped and read-only unless separately authorized.
- League-specific factual claims are grounded in tool results.
- Answers cite relevant source time, model version, and recommendation evidence.
- Prompt injection and malicious attachment handling are tested.
- User feedback and quality evaluation are continuous.
- Cost, latency, safety, and quality are observable by provider, model, tenant, and feature.
- Bubba abstains when evidence is unavailable.

#### Certification metrics

- At least 99% tool-call schema validity.
- At least 99% tenant-context correctness.
- Zero cross-tenant disclosure in red-team testing.
- Unsupported factual claim rate below 1% on the approved evaluation set.
- At least 95% evidence coverage for factual league-specific answers.
- At least 90% recommendation consistency with canonical engines.
- At least 85% judged usefulness on the expert evaluation set.
- 100% of malicious attachments remain non-executable.
- 100% of prompt-injection fixtures fail to override authorization.
- Daily per-user and global cost limits are fail-closed and reconciliation-tested.

---

## 4. Phase 0: Current-State Rebaseline and Audit Contract

**Effort:** M
**Dependencies:** None
**Primary categories:** All

### Objectives

- Establish one authoritative baseline.
- Separate historical findings from current defects.
- Create machine-readable evidence for every score.
- Convert the 100/100 standards into release gates.

### Work package 0.1: Freeze the audit target

1. Record the exact `master` commit SHA.
2. Record Vercel and Railway deployment identifiers.
3. Record active environment-variable names without exposing values.
4. Record Python, Node, pnpm, FastAPI, Next.js, PostgreSQL/SQLite, Redis, and browser versions.
5. Export the current OpenAPI contract.
6. Export FastAPI and Next.js route inventories.
7. Record current test counts, skipped tests, and skip reasons.
8. Record the database schema revision.
9. Record active provider capabilities and OAuth scopes.
10. Record currently active feature flags.
11. Record currently deployed model and constants versions.
12. Record which historical audit findings are fixed, open, obsolete, accepted, or unverifiable.

### Work package 0.2: Create an evidence registry

Create a machine-readable registry containing:

- Requirement ID.
- Category.
- Description.
- Current status.
- Responsible subsystem.
- Verification command.
- Production metric.
- Evidence path or URL.
- External-review requirement.
- Blocking release ring.
- Last verification timestamp.
- Current score contribution.

Store immutable audit artifacts outside the production application database. Link each artifact from the registry.

### Work package 0.3: Add structural quality guards

Create tests that fail when:

- A tenant-owned table lacks tenant scope.
- A league-owned table lacks league scope.
- A private route lacks authentication.
- A mutation lacks authorization and idempotency.
- A recommendation lacks evidence metadata.
- A model marked calibrated lacks provenance.
- A live frontend fetcher converts an error into mock data.
- OpenAPI and generated TypeScript types diverge.
- A production route lacks a critical-path E2E assignment.
- A critical workflow lacks telemetry.
- A connector advertises an unsupported capability.
- Documentation claims an endpoint or feature state that differs from generated inventory.

### Work package 0.4: Re-audit current `master`

Run:

- Full Python test suite.
- API test suite.
- Ruff lint and format checks.
- OpenAPI snapshot check.
- TypeScript typecheck.
- ESLint.
- Next.js production build.
- Browser smoke across all routes.
- Current production API probes.
- Current tenant/auth matrix.
- Current data-quality report.
- Current calibration report.

Reproduce every remaining High or Critical issue independently.

### Exit gate

- Current-state report approved.
- Every certification requirement exists in the evidence registry.
- No unresolved discrepancy remains between code, deployment, project instructions, and roadmap.
- Historical June audit items are accurately classified.
- Another engineer can reproduce the score.

---

## 5. Phase 1: Commercial Data Rights, Legal, and Privacy

**Effort:** L
**Dependencies:** Phase 0
**Primary categories:** Data quality, security/privacy, integrations, recommendation trust

### Objectives

- Remove the risk of charging for a product that lacks rights to required data.
- Define lawful handling of account, league, billing, and AI data.
- Establish public legal documents and internal governance.

### Work package 1.1: Build the source inventory

For each source, record:

- Provider and product.
- Fields consumed.
- Acquisition method.
- Refresh frequency.
- Raw-data storage duration.
- Derived-data storage duration.
- Whether raw values are displayed or redistributed.
- Whether data trains or calibrates a model.
- Existing agreement or terms.
- Commercial-use status.
- Attribution requirement.
- Deletion requirement.
- Rate limit.
- Technical owner.
- Legal reviewer.

Mandatory sources include Yahoo, MLB Stats API/MLBAM, FanGraphs, FantasyPros, Retrosheet, Clerk, Stripe, OpenAI, Anthropic, Google, and any other AI or data provider present in code or configuration.

### Work package 1.2: Resolve commercial rights

1. Obtain written commercial approval or an appropriate license for each production source.
2. Replace a source when commercial approval is unavailable.
3. Disable any production feature whose required source remains unresolved.
4. Add required attribution.
5. Prevent raw-data export when the license allows only derived use.
6. Encode source restrictions in metadata and export policy.
7. Require legal approval before adding a new source.
8. Store agreements, effective dates, renewal dates, and restrictions in the governance registry.

### Work package 1.3: Publish user-facing policies

Create and obtain legal review for:

- Terms of Service.
- Privacy Policy.
- Cookie and analytics notice.
- Acceptable Use Policy.
- AI disclosure.
- Subscription, trial, renewal, cancellation, and refund terms.
- Fantasy advice and no-guarantee disclaimer.
- Data-source attribution.
- Account deletion and export instructions.
- Vulnerability disclosure process.

### Work package 1.4: Classify data

Classify:

- Public application data.
- Private league data.
- Personal account data.
- Provider OAuth credentials.
- Billing identifiers.
- AI prompts and conversations.
- Uploaded attachments.
- Operational logs.
- Recommendation evidence.
- Model-development data.
- Security audit records.

For each class, define:

- Retention period.
- Encryption requirement.
- Authorized roles.
- Logging restrictions.
- Export behavior.
- Deletion behavior.
- Backup expiration.
- Legal hold behavior.

### Work package 1.5: Implement privacy operations

Plan and test:

- User data export.
- Account deletion.
- League disconnection.
- OAuth revocation.
- AI conversation deletion.
- Attachment deletion.
- Analytics opt-out.
- Legal hold.
- Deletion from active systems.
- Deletion from backups after retention expiry.
- Deletion confirmation.

### Exit gate

- Every production source is approved, replaced, or disabled.
- Legal reviewer approves public policies.
- Privacy data map is complete.
- Export and deletion workflows pass staging tests.
- No public feature relies on unresolved commercial rights.

---

## 6. Phase 2: PostgreSQL, Tenant Model, and Isolation

**Effort:** XL
**Dependencies:** Phases 0-1 and explicit owner approval to resume B2
**Primary categories:** Multi-tenancy, reliability, security, maintainability

### Objectives

- Complete B2.2-B2.5.
- Replace the single-writer public path.
- Introduce enforceable tenant and league boundaries.
- Support multiple leagues per user.

### Required core data model

Introduce or finalize:

- `tenants`
- `users`
- `tenant_memberships`
- `leagues`
- `league_memberships`
- `fantasy_teams`
- `provider_connections`
- `provider_credentials`
- `provider_capabilities`
- `refresh_jobs`
- `data_snapshots`
- `model_versions`
- `model_runs`
- `recommendations`
- `recommendation_evidence`
- `recommendation_outcomes`
- `recommendation_feedback`
- `audit_events`
- `privacy_requests`
- `entitlement_events`
- `ai_usage_events`

Every tenant-owned row must include `tenant_id`. Every league-owned row must include `tenant_id` and `league_id`.

### Work package 2.1: Complete PostgreSQL compatibility

1. Port SQLite-specific SQL.
2. Replace direct connection assumptions with SQLAlchemy engines or tenant-aware repositories.
3. Verify date, time-zone, boolean, JSON, case-insensitive text, autoincrement, conflict, and transaction semantics.
4. Add missing foreign keys in staged migrations.
5. Add explicit unique and check constraints.
6. Add indexes for tenant, league, player, season, timestamp, status, provider, and job filters.
7. Test every migration against a real PostgreSQL instance.
8. Measure representative query plans.
9. Preserve SQLite only for isolated local tests where appropriate.
10. Add migration smoke tests using production-like row counts.

### Work package 2.2: Implement server-derived tenant context

Tenant context derives from:

- Verified Clerk identity.
- Active tenant membership.
- Active league membership.
- Server-selected league context.

It must never derive authorization from:

- Query-string team names.
- Request-body team names.
- Client manager names.
- Mock defaults.
- Provider display names without verified ownership.

### Work package 2.3: Introduce scoped repositories

Repository methods must:

- Require tenant and league identifiers for private league data.
- Apply scope in every query.
- Reject missing context.
- Return typed domain objects.
- Avoid returning broad cross-league DataFrames.
- Emit audit-safe diagnostics without sensitive values.
- Use explicit transaction boundaries for writes.
- Verify affected rows belong to the expected scope.

### Work package 2.4: Add database isolation controls

Use layered protection:

- Application repository filters.
- Composite tenant-aware unique constraints.
- Tenant-aware foreign keys.
- PostgreSQL row-level security where operationally justified.
- Separate credential-encryption keys.
- Tenant-scoped cache namespaces.
- Tenant-scoped object-storage prefixes.
- Tenant-scoped worker payloads.
- Tenant-scoped recommendation and model-run records.

### Work package 2.5: Migrate existing data

1. Export and checksum SQLite data.
2. Create the initial HEATER tenant and existing league.
3. Transform legacy rows into tenant and league records.
4. Backfill user, team, and provider relationships.
5. Validate row counts, checksums, and aggregates.
6. Run read-only shadow comparisons.
7. Run dual reads in staging.
8. Freeze writes for the final migration window.
9. Migrate final deltas.
10. Switch reads.
11. Switch writes.
12. Retain a rollback snapshot.
13. Verify all personalized pages for every existing team.
14. Remove live SQLite writes only after acceptance.

### Work package 2.6: Add multi-league switching

Users must be able to:

- View connected leagues.
- Select the active league.
- Remember active context per session.
- Switch without stale cache leakage.
- See provider and refresh state.
- Disconnect with confirmation.
- Understand plan limits for connected leagues.
- Retain account-level billing while switching league context.

### Isolation test matrix

Test:

- Tenant A cannot read Tenant B.
- Tenant A cannot mutate Tenant B.
- A user without a league receives `league_not_connected`.
- A user without a team receives `team_not_linked`.
- Disabled membership denies access immediately.
- Cached responses cannot cross tenants.
- Worker jobs cannot load another tenant.
- AI tools cannot query another tenant.
- Admin impersonation is audited and expires.
- Exports include only requester-owned data.
- Deleted tenants become inaccessible before physical purge completes.
- Object-storage URLs cannot be guessed across tenants.
- Recommendation evidence cannot reference another tenant's snapshot.
- Billing entitlements cannot be applied to another tenant.

### Exit gate

- All production paths use PostgreSQL.
- Tenant-isolation suite passes.
- Independent penetration test finds no cross-tenant access.
- Existing 12-user league passes full regression.
- Restore and rollback procedures are demonstrated.
- Release ring 6 is unlocked.

---

## 7. Phase 3: Provider Connectors and Data Quality Platform

**Effort:** XL
**Dependencies:** Phases 1-2
**Primary categories:** Integrations, data quality, onboarding, reliability

### Objectives

- Replace hardcoded provider assumptions.
- Create repeatable ingestion with lineage and quality controls.
- Make data health visible to users and operators.

### Work package 3.1: Define `LeagueConnector`

Required operations:

- `capabilities()`
- `authorize()`
- `refresh_credentials()`
- `revoke()`
- `discover_leagues()`
- `import_league_settings()`
- `import_teams()`
- `import_rosters()`
- `import_standings()`
- `import_matchups()`
- `import_transactions()`
- `set_lineup()` when supported
- `add_drop()` when supported
- `health()`

Each connector returns typed canonical records and stable error codes.

### Work package 3.2: Capability flags

Represent:

- Read league settings.
- Read standings.
- Read rosters.
- Read matchups.
- Read transactions.
- Read historical data.
- Near-real-time refresh.
- Set lineup.
- Add/drop.

The UI must hide or explain unavailable actions. A provider adapter may not claim a capability that is unavailable.

### Work package 3.3: Complete Yahoo connection

1. Use per-user OAuth.
2. Request only required scopes.
3. Add explicit consent for write scope.
4. Encrypt tokens.
5. Refresh tokens automatically.
6. Handle revoked access.
7. Provide a reconnect flow.
8. Rate-limit writes.
9. Add mutation idempotency.
10. Verify the selected Yahoo team belongs to the authenticated user.
11. Log mutation metadata without tokens.
12. Add controlled write tests.
13. Add a read-only option for users unwilling to grant write scope.

### Work package 3.4: Add guided imports

For ESPN and CBS without approved APIs:

1. Publish versioned templates.
2. Accept only declared file types.
3. Virus-scan uploads.
4. Parse in an isolated worker.
5. Show a normalized preview.
6. Validate league settings and roster legality.
7. Show row-level correction messages.
8. Require confirmation before commit.
9. Store the artifact, checksum, and parser version.
10. Support rollback.
11. Explain which features need repeated imports.
12. Prevent duplicate import application.

### Work package 3.5: Canonical player identity graph

Use immutable canonical player IDs linked to:

- MLB ID.
- Yahoo player key.
- Other provider IDs.
- Name variants.
- Team history.
- Position history.
- Batter/pitcher identity.
- Split identities.
- Effective date ranges.

Add:

- Deterministic matching.
- Confidence levels.
- Manual resolution queue.
- Collision protection.
- Duplicate detection.
- Identity-history audit.
- Provider-ID uniqueness constraints with effective-date support.

### Work package 3.6: Ingestion quality pipeline

Every ingestion performs:

1. Schema validation.
2. Type validation.
3. Range validation.
4. Null and completeness checks.
5. Referential-integrity checks.
6. Duplicate detection.
7. Temporal-consistency checks.
8. Provider reconciliation.
9. Freshness calculation.
10. Snapshot persistence.
11. Quality-score calculation.
12. Alert evaluation.

### Work package 3.7: Provider reconciliation

Document precedence for:

- Player identity.
- Team.
- Position eligibility.
- Injury status.
- Probable-starter status.
- Roster ownership.
- League standings.
- Projection fields.
- Final results.

When sources disagree:

- Preserve all source values.
- Apply the approved winner.
- Record the rule.
- Lower the quality score when appropriate.
- Surface material conflicts to users.

### Work package 3.8: Freshness and lineage

Each frontend data payload must expose:

- `observed_at`
- `effective_at`
- `refreshed_at`
- `source`
- `source_snapshot_id`
- `quality_status`
- `freshness_status`
- `warnings`

The UI must visibly distinguish current, stale, partial, unavailable, and conflicting data.

### Work package 3.9: Historical snapshots

Persist point-in-time snapshots for:

- Projections.
- Rosters.
- League settings.
- Standings.
- Matchups.
- Transactions.
- Injuries.
- Probable starters.
- Weather and park context where licensed.
- Model inputs.
- Recommendation outputs.

Snapshots must be immutable, timestamped, versioned, and replayable.

### Exit gate

- Yahoo connection passes end-to-end tests.
- At least one guided import path works.
- Identity coverage reaches the target.
- Data-quality dashboard is operational.
- Every recommendation input is traceable to snapshots.
- No unsupported platform is advertised.
- Public launch remains blocked until source rights are approved.

---

## 8. Phase 4: Prediction Validation and Calibration

**Effort:** XL
**Dependencies:** Phase 3 historical snapshots
**Primary categories:** Prediction accuracy, data quality, analytical value

### Objectives

- Replace placeholder or proxy backtesting with real point-in-time validation.
- Calibrate every probability and confidence claim.
- Establish defensible performance evidence.

### Work package 4.1: Historical datasets

Build versioned datasets for:

- Daily player projections and actual results.
- Weekly H2H category matchups.
- Probable starters and realized starts.
- Streaming decisions and realized lines.
- Transactions and post-transaction outcomes.
- Draft ADP, draft state, picks, and season results.
- Injuries and availability.
- League settings and category rules.

Each record requires an `available_at` timestamp that proves the information existed before the decision.

### Work package 4.2: Repair optimizer backtesting

The optimizer backtest must:

1. Reconstruct the historical roster.
2. Reconstruct league settings.
3. Use only information available at decision time.
4. Run the actual optimizer.
5. Apply roster and transaction constraints.
6. Generate selected actions.
7. Compare against feasible baselines.
8. Measure realized category or matchup utility.
9. Record failures and abstentions.
10. Preserve inputs, outputs, and software versions.

Never align values by DataFrame row order. Join by canonical IDs and effective dates.

### Work package 4.3: Establish baselines

Required baselines include:

- Start every projected active player.
- Highest consensus rank.
- Highest recent fantasy or category value.
- No transaction.
- Most rostered available player.
- Simple season-to-date ranking.
- Provider-native recommendation where legally available.
- Random feasible action as a lower bound.

### Work package 4.4: Validate player projections

Measure:

- MAE.
- RMSE.
- Bias.
- Rank correlation.
- Prediction-interval coverage.
- Performance by position.
- Performance by sample size.
- Performance by injury state.
- Performance by season phase.
- Performance for prospects and low-data players.

### Work package 4.5: Validate probability models

For matchup, playoff, draft-survival, and similar probabilities, measure:

- Brier score.
- Log loss.
- Calibration curve.
- Calibration slope.
- Calibration intercept.
- Expected calibration error.
- Sharpness.
- Performance by probability band.
- Performance by league format and season phase.

### Work package 4.6: Validate streaming recommendations

Measure:

- Realized ERA.
- Realized WHIP.
- Strikeouts.
- Wins.
- Innings.
- Net category utility.
- Blow-up rate.
- Utility versus no-stream.
- Utility versus simple projection baselines.
- Performance by confidence label.
- Performance with and without two-start context.

### Work package 4.7: Validate trade recommendations

Measure:

- Projected versus realized roster utility.
- Category-balance improvement.
- Regret against feasible alternatives.
- Sensitivity to projection uncertainty.
- Performance by trade size.
- Performance by season phase.
- Frequency of infeasible or unbalanced proposals.

### Work package 4.8: Validate draft recommendations

Measure:

- Value over ADP.
- Roster-construction quality.
- Category balance.
- Pick-survival calibration.
- Simulated season utility.
- Realized season utility where data permits.
- Performance against consensus-rank and ADP baselines.

### Work package 4.9: Calibrate constants

For every entry in `data/calibrated_constants.json`:

1. Assign a precise target.
2. Identify historical training data.
3. Define the fitting method.
4. Define permitted bounds.
5. Fit on training data.
6. Select on calibration data.
7. Evaluate once on final holdout.
8. Calculate confidence intervals.
9. Review temporal stability.
10. Record dataset and software versions.
11. Mark calibrated only after approval.
12. Preserve previous value and rollback path.

If a constant cannot be calibrated, label it a heuristic and disclose that limitation.

### Work package 4.10: Model registry and model cards

Every production model version stores:

- Code commit.
- Dataset versions.
- Feature definitions.
- Hyperparameters.
- Calibration data.
- Validation data.
- Final metrics.
- Intended uses.
- Prohibited uses.
- Known limitations.
- Approval.
- Deployment timestamp.
- Retirement timestamp.

### Work package 4.11: Independent statistical review

The reviewer must assess:

- Leakage.
- Selection bias.
- Missing-data handling.
- Statistical power.
- Baseline selection.
- Confidence intervals.
- Multiple comparisons.
- Calibration.
- Decision-utility measurement.
- Claims shown in the UI and marketing.

### Exit gate

- No production constant is incorrectly labeled calibrated.
- Backtests execute the actual decision systems.
- Final holdout metrics meet certification standards.
- Model cards are complete.
- Independent statistical review approves the methodology.
- Probability and confidence marketing claims are approved.

---

## 9. Phase 5: Recommendation Evidence and Trust Layer

**Effort:** L
**Dependencies:** Phases 3-4
**Primary categories:** Recommendation trustworthiness, analytical value, AI

### Objectives

- Make every recommendation inspectable and reproducible.
- Prevent overconfident advice.
- Measure outcomes and user trust.

### Required recommendation record

Every recommendation stores:

- Recommendation ID.
- Tenant and league.
- User.
- Recommendation type.
- Creation and expiry time.
- Input snapshot IDs.
- Model version.
- Rules and constraints.
- Proposed action.
- Expected category impact.
- Uncertainty.
- Confidence band.
- Alternatives considered.
- Alternative rejection reasons.
- Assumptions.
- Warnings.
- Explanation.
- Outcome status.
- User feedback.

### Work package 5.1: Canonical recommendation service

All recommendation-producing pages and Bubba must call the same domain services. Frontend and AI layers may format explanations but may not independently recalculate the recommended action.

### Work package 5.2: Evidence cards

Every recommendation surface displays:

- Recommended action.
- Why it helps.
- Categories affected.
- Expected magnitude.
- Key inputs.
- Data freshness.
- Confidence.
- Important risks.
- Alternative action.
- Why-not comparison where useful.

### Work package 5.3: Abstention policy

Abstain when:

- Data is stale beyond the approved threshold.
- Required player identity is unresolved.
- League settings are incomplete.
- Model inputs fail quality rules.
- Alternatives are statistically indistinguishable.
- Provider capability cannot execute the action.
- Projection uncertainty exceeds the approved threshold.
- A model is outside its documented intended use.

### Work package 5.4: Conflict detection

Detect and reconcile cases such as:

- Streaming recommends an add when the drop model recommends no valid drop.
- Optimizer recommends starting an unavailable player.
- Trade recommendation damages a critical category.
- Bubba contradicts the canonical recommendation.
- Two tools use different settings, snapshots, or model versions.
- A suggested transaction exceeds weekly or season limits.

### Work package 5.5: Outcome measurement

Track:

- Viewed.
- Evidence expanded.
- Saved.
- Accepted.
- Rejected.
- Executed.
- Provider execution succeeded.
- Outcome became observable.
- Realized utility.
- User-rated usefulness.
- User-reported error.

User acceptance alone must never be treated as correctness.

### Work package 5.6: Trust dashboard

Report:

- Recommendation volume.
- Abstention rate.
- Acceptance rate.
- Execution rate.
- Realized utility.
- Calibration by confidence band.
- Contradiction rate.
- Complaint rate.
- Performance by model version.
- Performance by provider.
- Performance by league format.

### Exit gate

- Every recommendation has immutable evidence.
- Replay equivalence reaches target.
- Contradiction detection passes.
- Structured feedback is available.
- Trust dashboard is operational.
- Independent reviewers reproduce sampled recommendations.

---

## 10. Phase 6: Reliability, Workers, and Operations

**Effort:** XL
**Dependencies:** Phase 2
**Primary categories:** Reliability, scalability, maintainability

### Objectives

- Complete B3.
- Remove expensive and failure-prone work from request paths.
- Establish production operations.

### Work package 6.1: Background job system

Move to workers:

- Provider refreshes.
- Historical imports.
- Model calibration.
- Monte Carlo simulations.
- Trade simulation.
- Playoff simulations.
- Full free-agent ranking.
- Large exports.
- Attachment parsing.
- AI deep research.
- Snapshot generation.
- Reconciliation.

Every job includes:

- Job ID.
- Tenant and league.
- Idempotency key.
- Priority.
- Timeout.
- Retry policy.
- Progress.
- Cancellation support where safe.
- Result location.
- Error classification.
- Trace context.

### Work package 6.2: Job API

Add:

- `POST /api/v2/jobs`
- `GET /api/v2/jobs/{job_id}`
- `POST /api/v2/jobs/{job_id}/cancel`
- Server-sent events or polling for progress.
- Stable terminal statuses.

### Work package 6.3: Caching policy

For every cached data class, define:

- Key composition.
- TTL.
- Stale-while-revalidate period.
- Tenant and league scope.
- Model-version scope.
- Snapshot scope.
- Invalidation event.
- Maximum stale duration.
- Failure fallback.
- User-visible freshness state.

### Work package 6.4: Upstream resilience

For every provider, define:

- Connect timeout.
- Read timeout.
- Total request budget.
- Retry count.
- Exponential backoff.
- Jitter.
- Rate-limit behavior.
- Circuit breaker.
- Stale-data fallback.
- Alert threshold.
- User-facing message.

### Work package 6.5: Observability

Instrument:

- Request duration and status.
- Worker queue depth.
- Job duration and failure.
- Database query duration.
- Connection-pool saturation.
- Cache hit rate.
- Provider latency and errors.
- Model execution duration.
- Recommendation volume.
- AI token and cost usage.
- Billing webhook outcomes.
- Authentication and authorization failures.

Logs should carry request, tenant, league, user, job, model, and recommendation IDs where appropriate, without secrets.

### Work package 6.6: SLOs and alerting

Define:

- Availability.
- Latency.
- Error rate.
- Queue delay.
- Refresh freshness.
- Billing processing.
- Data quality.
- AI spend.

Every alert includes severity, owner, runbook, and escalation.

### Work package 6.7: Backup and disaster recovery

1. Automated encrypted backups.
2. Point-in-time recovery.
3. Object-storage versioning.
4. Configuration backup.
5. Secret-rotation procedure.
6. Quarterly restore rehearsal.
7. Documented failover.
8. User communication templates.
9. Post-incident review template.

### Work package 6.8: Load and chaos testing

Test:

- Expected launch traffic.
- Twice expected traffic.
- Sudden traffic spike.
- Redis unavailable.
- Worker unavailable.
- PostgreSQL connection exhaustion.
- Yahoo timeout.
- MLB API timeout.
- AI provider outage.
- Stripe webhook replay.
- Vercel-to-Railway network failure.
- Partial deployment rollback.

### Exit gate

- SLO dashboards and alerts are operational.
- Load and chaos targets pass.
- Restore drill passes.
- No unbounded external call remains.
- Incident runbooks are complete.
- 30-day SLO observation begins.

---

## 11. Phase 7: Security and Privacy Enforcement

**Effort:** L
**Dependencies:** Phases 1-3
**Primary categories:** Security, privacy, multi-tenancy, AI

### Objectives

- Harden all public attack surfaces.
- Prove isolation and credential safety.
- Complete external security assurance.

### Work package 7.1: Authentication and sessions

- Production Clerk keys.
- Issuer and audience verification.
- Session expiration.
- Revocation.
- MFA support.
- Secure cookies.
- CSRF protection where applicable.
- Brute-force protection.
- Suspicious-login monitoring.

### Work package 7.2: Authorization matrix

Define explicit permissions for:

- Account owner.
- League member.
- League administrator.
- HEATER support.
- HEATER administrator.
- Worker.
- Webhook.
- AI tool.

Every permission must be server-enforced and negatively tested.

### Work package 7.3: Credential protection

- Envelope encryption for provider credentials.
- Separate key-encryption key.
- Rotation support.
- Least-privilege access.
- Redaction in logs and monitoring.
- No credential values in analytics.
- Immediate revocation on disconnect.
- Alert on decryption failure.

### Work package 7.4: API hardening

- Authentication on private reads.
- Authorization on writes.
- Rate limits by IP, user, tenant, and route class.
- Payload limits.
- File-type and file-size limits.
- Strict CORS.
- Security headers.
- Request IDs.
- Idempotency for mutations.
- Input validation.
- Safe error responses.
- Abuse monitoring.

### Work package 7.5: Import and attachment security

- Store uploads outside executable paths.
- Scan for malware.
- Restrict archive expansion.
- Limit extracted size.
- Parse in isolated workers.
- Treat file text as untrusted.
- Never follow embedded instructions.
- Delete rejected uploads.
- Audit access.

### Work package 7.6: AI security

- Tenant-scoped tools.
- Read-only SQL with table allowlists.
- Prompt-injection testing.
- Tool-call policy enforcement.
- Output encoding.
- Attachment isolation.
- Provider privacy configuration.
- No secrets in prompts.
- Configurable conversation retention.

### Work package 7.7: Secure development lifecycle

CI must run:

- Secret scanning.
- Dependency scanning.
- Static analysis.
- Infrastructure scanning.
- Container scanning.
- License scanning.
- Security unit tests.
- Authorization tests.
- SBOM generation.

### Work package 7.8: Independent penetration test

Scope:

- Authentication.
- Authorization.
- Tenant isolation.
- APIs.
- Billing.
- OAuth.
- Imports and attachments.
- AI prompt injection.
- Admin workflows.
- Deployment configuration.

### Exit gate

- Zero Critical or High penetration-test findings.
- Medium findings have remediation or approved risk acceptance.
- Production keys and secrets are verified.
- Privacy workflows pass.
- Incident-response tabletop passes.
- Release ring 7 is unlocked from a security perspective.

---

## 12. Phase 8: Billing, Entitlements, and Customer Lifecycle

**Effort:** L
**Dependencies:** Phases 2 and 7
**Primary categories:** Billing, adoption, security, AI

### Objectives

- Make payment state reliable.
- Complete self-service customer lifecycle.
- Prevent entitlement and AI-cost leakage.

### Work package 8.1: Subscription state machine

Support:

- Free.
- Trialing.
- Active.
- Past due.
- Grace period.
- Paused.
- Canceled.
- Refunded.
- Disputed.
- Incomplete.
- Expired.

Document which Stripe events may enter or leave each state.

### Work package 8.2: Webhook integrity

- Verify signatures.
- Store event ID.
- Deduplicate.
- Store event creation time.
- Enforce safe ordering.
- Permit replay.
- Record processing outcome.
- Alert on repeated failure.
- Use idempotent entitlement updates.

### Work package 8.3: Entitlement service

Entitlements include:

- Feature key.
- Usage allowance.
- Connected-league limit.
- AI allowance.
- Renewal time.
- Grace status.
- Denial reason.

### Work package 8.4: Customer portal

Provide:

- Current plan.
- Trial status.
- Renewal date.
- Invoice history.
- Payment method.
- Upgrade.
- Downgrade.
- Cancel.
- Reactivate.
- Billing support.

### Work package 8.5: AI metering

Meter:

- Provider.
- Model.
- Input tokens.
- Output tokens.
- Tool usage.
- Managed cost.
- User allowance.
- Tenant allowance.
- Failure adjustments.
- Reconciliation state.

Managed usage must fail closed when metering is unavailable. Permitted BYOK usage may continue independently.

### Work package 8.6: Financial reconciliation

Run daily:

- Stripe customers versus application users.
- Stripe subscriptions versus entitlements.
- Processed webhooks versus Stripe events.
- AI usage versus allowance ledger.
- Refunds and disputes.
- Orphaned customers.
- Duplicate subscriptions.

### Exit gate

- Stripe certification scenarios pass.
- Webhook replay tests pass.
- Daily reconciliation is clean.
- Customer portal works.
- Pricing disclosure is approved.
- AI cost limits cannot be bypassed.

---

## 13. Phase 9: Self-Service Onboarding, Adoption, and Support

**Effort:** L
**Dependencies:** Phases 2-3 and 8
**Primary categories:** Onboarding, analytical value, integrations, UI/UX

### Objectives

- Remove administrator dependency.
- Lead users to a useful first result.
- Make support sustainable for a solo operator.

### Work package 9.1: Onboarding flow

1. Create or sign into an account.
2. Accept terms and privacy choices.
3. Choose live connection, guided import, or labeled demo.
4. Select provider.
5. Authenticate or upload.
6. Select league.
7. Select or confirm team.
8. Validate categories, roster rules, transaction limits, and schedule.
9. Review data freshness and missing data.
10. Run initial refresh.
11. Display the first personalized recommendation.
12. Explain next actions.

Persist progress and allow resumption.

### Work package 9.2: Recovery states

Provide specific recovery for:

- OAuth denied.
- OAuth expired.
- League not found.
- Team not found.
- Team unlinked.
- Unsupported league format.
- Missing scoring settings.
- Import validation failure.
- Provider outage.
- Refresh timeout.
- Insufficient data for a recommendation.
- Subscription required.
- Account conflict.

### Work package 9.3: Product education

Add:

- Contextual metric definitions.
- Evidence-card walkthrough.
- Confidence and uncertainty explanation.
- Data freshness explanation.
- Recommendation disclaimer.
- Provider capability explanation.
- Short onboarding checklist.
- Searchable help center.
- Status-page link.

### Work package 9.4: Lifecycle communication

Support:

- Welcome.
- Connection success.
- Connection failure.
- Refresh failure.
- Trial ending.
- Payment failure.
- Subscription cancellation.
- Major recommendation alert.
- Security notice.
- Data export ready.
- Account deletion complete.

### Work package 9.5: Support operations

Implement:

- In-app support form.
- Diagnostic bundle without secrets.
- Stable issue categories.
- User-visible ticket ID.
- Support queue.
- Response targets.
- Escalation rules.
- Known-issues page.
- Incident banners.
- Feature-request tagging.

### Work package 9.6: Product analytics

Track privacy-controlled events for:

- Onboarding step.
- Connector success.
- First recommendation.
- Evidence expansion.
- Recommendation acceptance.
- Return usage.
- Upgrade.
- Churn.
- Support need.
- Dead end.

Never send raw private league data to analytics.

### Exit gate

- Unassisted onboarding target is met.
- Activation target is met.
- Recovery states are tested.
- Support runbook is operational.
- Analytics contain no prohibited data.
- Invite-only external beta may begin.

---

## 14. Phase 10: UI/UX, Accessibility, and Design Certification

**Effort:** L
**Dependencies:** Phase 9 workflows stable
**Primary categories:** UI/UX/design, onboarding, recommendation trust

### Objectives

- Preserve the distinctive Combustion identity.
- Make the product usable and accessible across devices.
- Eliminate deceptive live/mock behavior.

### Work package 10.1: Information architecture

- Group navigation by user goal.
- Keep critical actions visible.
- Separate research from action workflows.
- Add a reliable league switcher.
- Add account and connection status.
- Ensure mobile navigation exposes all routes.
- Remove clipping at desktop widths.
- Add responsive handling for wide tables.

### Work package 10.2: Responsive system

Test widths:

- 320.
- 360.
- 390.
- 430.
- 768.
- 1024.
- 1280.
- 1440.
- 1920.

Verify:

- No horizontal overflow.
- No clipped navigation.
- Touch targets.
- Sticky controls.
- Table alternatives.
- Dialog sizing.
- Keyboard visibility.
- Safe-area handling.
- Reduced hero height on small screens.

### Work package 10.3: State system

Every asynchronous surface supports:

- Idle.
- Loading.
- Loaded.
- Empty.
- Stale.
- Partial.
- Degraded.
- Unauthorized.
- Team unlinked.
- Subscription locked.
- Offline.
- Error.

Mock data is allowed only in explicitly labeled demo mode.

### Work package 10.4: Accessibility

Implement and verify:

- Semantic landmarks.
- Skip link.
- Logical headings.
- Accessible names.
- Form labels.
- Error associations.
- Keyboard navigation.
- Focus trapping and restoration.
- Escape behavior.
- Screen-reader announcements.
- Table headers and scopes.
- Non-color status cues.
- Contrast.
- 200% and 400% zoom.
- Reduced motion.
- 44 by 44 targets.

### Work package 10.5: Performance

- Server-render stable shells.
- Avoid unnecessary client bundles.
- Lazy-load heavy charts and dialogs.
- Optimize fonts and images.
- Virtualize large tables.
- Prevent repeated player-detail calculations.
- Use route-level loading boundaries.
- Track Web Vitals by route and device.

### Work package 10.6: Usability research

Test:

1. Connect a league.
2. Find the best pickup.
3. Optimize today's lineup.
4. Evaluate a trade.
5. Understand a streaming recommendation.
6. Interpret matchup odds.
7. Ask Bubba for evidence-backed advice.
8. Manage a subscription.
9. Recover a broken provider connection.
10. Delete an account.

Recruit both experienced and casual fantasy-baseball users.

### Work package 10.7: Independent accessibility audit

Require:

- Automated scanning.
- Keyboard-only review.
- Screen-reader review.
- Low-vision and zoom review.
- Color and contrast review.
- Mobile accessibility review.
- Remediation verification.

### Exit gate

- WCAG 2.2 AA audit passes.
- Core Web Vitals meet targets.
- Task-success and SUS targets pass.
- No live failure fabricates valid-looking data.
- Supported device matrix passes.
- Public design certification is complete.

---

## 15. Phase 11: Trusted Bubba AI Copilot

**Effort:** XL
**Dependencies:** Phases 3-5 and 7
**Primary categories:** AI, recommendation trust, security, analytical value

### Objectives

- Make Bubba a trustworthy interface to canonical HEATER intelligence.
- Evaluate quality instead of relying on anecdotal prompts.
- Prevent cross-tenant and unsupported advice.

### Work package 11.1: Canonical tool layer

Expose narrow tools for:

- Current team.
- League settings.
- Player search.
- Player details.
- Matchup state.
- Free-agent recommendations.
- Lineup optimization.
- Streaming analysis.
- Trade evaluation.
- Standings and playoff odds.
- Recommendation evidence.
- Data health.

Tools call canonical services and return typed results.

### Work package 11.2: Grounding contract

League-specific answers include:

- Relevant tool results.
- Data timestamp.
- League context.
- Recommendation ID when applicable.
- Model version.
- Uncertainty.
- Missing-data warning.
- Evidence links.

When evidence does not exist, Bubba must say so.

### Work package 11.3: Evaluation sets

Build versioned sets for:

- Simple factual questions.
- Multi-step analysis.
- Roster recommendations.
- Trade questions.
- Streaming questions.
- Ambiguous requests.
- Stale-data cases.
- Unsupported platform requests.
- Prompt injection.
- Malicious attachments.
- Cross-tenant attempts.
- Subscription and allowance abuse.
- Hallucination traps.
- Contradictory tool results.

### Work package 11.4: AI metrics

Measure:

- Correctness.
- Groundedness.
- Evidence coverage.
- Recommendation consistency.
- Tool selection.
- Tool argument validity.
- Tenant correctness.
- Refusal correctness.
- Abstention correctness.
- Helpfulness.
- Latency.
- Cost.
- User feedback.

### Work package 11.5: Model routing

Select models using:

- Task complexity.
- Required tools.
- Cost ceiling.
- Latency target.
- User plan.
- BYOK provider.
- Safety requirements.

Routing may never weaken authorization or evidence requirements.

### Work package 11.6: Feedback loop

Users may report:

- Helpful.
- Incorrect.
- Outdated.
- Missing context.
- Unsafe.
- Too slow.
- Too expensive.

Feedback links to the exact prompt, tools, model, recommendation, and snapshots, subject to privacy rules.

### Work package 11.7: AI release gate

A model or prompt change cannot ship when it:

- Reduces tenant isolation.
- Increases unsupported claims beyond threshold.
- Reduces recommendation consistency below threshold.
- Exceeds cost limits.
- Breaks evidence coverage.
- Fails prompt-injection tests.

### Exit gate

- AI evaluation suite passes.
- Red-team suite passes.
- Canonical recommendation consistency meets target.
- Unsupported-claim rate meets target.
- Cost and latency SLOs are met.
- Bubba is approved for the next release ring.

---

## 16. Phase 12: Engineering Quality and Delivery

**Effort:** L
**Dependencies:** Runs alongside Phases 2-11
**Primary categories:** Maintainability, reliability, security, UI/UX

### Objectives

- Make quality repeatable.
- Prevent documentation and contract drift.
- Reduce solo-operator risk.

### Work package 12.1: Frontend testing

Add:

- Unit tests for adapters and utilities.
- Component tests for stateful UI.
- Accessibility assertions.
- API integration tests.
- Playwright E2E tests.
- Visual regression tests.
- Responsive viewport tests.
- Authentication tests.
- Billing tests.
- Demo-versus-live tests.

### Work package 12.2: CI expansion

Required checks:

- Python lint and format.
- Python tests.
- Backend coverage.
- API contract snapshot.
- PostgreSQL migration tests.
- TypeScript typecheck.
- ESLint.
- Frontend unit and component tests.
- Next.js production build.
- Playwright critical-path tests.
- Accessibility checks.
- OpenAPI-to-TypeScript drift check.
- Secret scan.
- Dependency scan.
- Container scan.
- License scan.
- SBOM generation.

### Work package 12.3: Optional analytical dependencies

Create scheduled CI with PyMC, XGBoost, and other heavy dependencies. A skipped optional path is not considered verified unless its scheduled job is green.

### Work package 12.4: Documentation enforcement

Automate checks for:

- Endpoint inventory.
- Environment-variable inventory.
- Generated API types.
- Schema revision.
- Feature inventory.
- Route inventory.
- Plan status.
- Provider capabilities.
- Model versions.
- Data sources.
- Runbook links.

### Work package 12.5: Release process

Every release includes:

- Version.
- Commit.
- Migration revision.
- OpenAPI version.
- Frontend build ID.
- Model versions.
- Feature flags.
- Change summary.
- Rollback instructions.
- Verification evidence.

### Work package 12.6: Ownership and architecture

Document ownership and runbooks for:

- Python engine.
- FastAPI.
- Database.
- Workers.
- Next.js.
- Connectors.
- Billing.
- AI.
- Security.
- Operations.

Even with one human owner, this identifies the correct review and recovery procedure.

### Work package 12.7: Technical debt policy

Classify debt as:

- Security.
- Correctness.
- Reliability.
- Scalability.
- Product.
- Maintainability.
- Cosmetic.

Critical security, correctness, and isolation debt cannot cross a public release gate.

### Exit gate

- All required checks block merges.
- Frontend automation covers critical workflows.
- Release and rollback are reproducible.
- Documentation-drift checks pass.
- No required path depends on an unverified optional dependency.

---

## 17. Phase 13: Progressive Release and Independent Certification

**Effort:** L plus observation period
**Dependencies:** All prior phases
**Primary categories:** All

### Release gate A: Existing league

Required:

- Current 12-user regression passes.
- No Critical or High findings.
- Data freshness and provider health are visible.
- Support and rollback procedures exist.
- Existing users can recover accounts and connections.

### Release gate B: Invite-only external beta

Required:

- PostgreSQL and tenant isolation are live.
- Self-service onboarding is live.
- Yahoo per-user OAuth is live.
- Guided import is live.
- Security test passed.
- Privacy policies published.
- Billing lifecycle passed.
- Recommendation evidence live.
- Production monitoring active.

### Release gate C: Limited public launch

Required:

- Independent penetration test passed.
- Independent statistical review passed.
- Independent WCAG audit passed.
- Legal and licensing review passed.
- 30-day SLO observation meets targets.
- Data-quality targets met.
- AI evaluation targets met.
- Disaster-recovery drill passed.
- Customer-support process proven.

Limit signups using feature flags and capacity thresholds.

### Release gate D: General availability

Required:

- All 13 categories independently rescored at 100.
- No unresolved Critical or High finding.
- No overdue Medium finding involving user harm, privacy, billing, or model claims.
- Capacity supports twice forecast traffic.
- Incident and support coverage is sustainable.
- Streamlit retirement criteria pass.
- Public status page and release notes are active.

---

## 18. Public API and Interface Additions

Use `/api/v2` for tenant-aware contracts when safe backward compatibility cannot be maintained.

### Account and privacy

- `GET /api/v2/account`
- `GET /api/v2/account/export`
- `POST /api/v2/account/deletion`
- `GET /api/v2/account/deletion/{request_id}`
- `POST /api/v2/account/analytics-consent`

### Connections and leagues

- `GET /api/v2/connections`
- `POST /api/v2/connections/{provider}/authorize`
- `POST /api/v2/connections/{provider}/callback`
- `POST /api/v2/connections/{connection_id}/refresh`
- `DELETE /api/v2/connections/{connection_id}`
- `GET /api/v2/leagues`
- `POST /api/v2/leagues/{league_id}/activate`
- `GET /api/v2/leagues/{league_id}/data-health`

### Imports

- `POST /api/v2/imports`
- `GET /api/v2/imports/{import_id}`
- `POST /api/v2/imports/{import_id}/confirm`
- `POST /api/v2/imports/{import_id}/rollback`

### Jobs

- `POST /api/v2/jobs`
- `GET /api/v2/jobs/{job_id}`
- `POST /api/v2/jobs/{job_id}/cancel`

### Recommendations

- `GET /api/v2/recommendations`
- `GET /api/v2/recommendations/{recommendation_id}`
- `GET /api/v2/recommendations/{recommendation_id}/evidence`
- `POST /api/v2/recommendations/{recommendation_id}/feedback`
- `GET /api/v2/recommendations/{recommendation_id}/outcome`

### Models and data

- `GET /api/v2/models`
- `GET /api/v2/models/{model_id}/card`
- `GET /api/v2/data/sources`
- `GET /api/v2/data/health`

### Operations

- `GET /health/live`
- `GET /health/ready`
- Protected internal diagnostics endpoint.
- Protected internal metrics endpoint.

### Contract rules

- Private responses include tenant and league context where useful.
- Analytical responses include freshness and model metadata.
- Errors use stable machine-readable codes.
- Responses include request IDs.
- Mutations accept idempotency keys.
- Sensitive identifiers are not exposed unnecessarily.
- OpenAPI is the source of truth.
- Generated frontend types are committed and CI-verified.

---

## 19. Mandatory Test Program

### Identity and isolation

- Signed-out access.
- Signed-in without league.
- Signed-in without team.
- Multiple leagues.
- Disabled membership.
- Cross-tenant read.
- Cross-tenant write.
- Cross-tenant cache.
- Cross-tenant job.
- Cross-tenant AI tool.
- Support impersonation.
- Deleted account.

### Data

- Missing MLB ID.
- Duplicate player.
- Two players with the same name.
- Split batter/pitcher identity.
- Provider team disagreement.
- Stale projection.
- Partial provider response.
- NaN and infinity.
- Invalid dates.
- Late correction.
- Upstream outage.
- Snapshot replay.

### Predictions and recommendations

- Historical point-in-time replay.
- Baseline comparison.
- No-look-ahead assertion.
- Deterministic replay.
- Low-confidence abstention.
- Conflicting recommendations.
- Illegal roster move.
- Transaction-limit breach.
- Unsupported provider write.
- Missing league settings.
- Prediction drift.

### Billing

- Trial start and expiry.
- Successful renewal.
- Failed renewal.
- Grace period.
- Upgrade.
- Downgrade.
- Cancellation.
- Reactivation.
- Refund.
- Dispute.
- Duplicate webhook.
- Out-of-order webhook.
- Webhook replay.
- Subscription-store outage.

### UI

- All supported viewports.
- Keyboard-only use.
- Screen reader.
- Reduced motion.
- 200% and 400% zoom.
- High contrast.
- Slow network.
- Offline.
- Unauthorized.
- Empty.
- Stale.
- Provider outage.
- Payment failure.
- Long player names.
- Large leagues.
- Missing images.
- No horizontal overflow.

### AI

- Correct tool selection.
- Invalid tool arguments.
- Tool timeout.
- Tool contradiction.
- Prompt injection.
- Malicious attachment.
- Cross-tenant request.
- Unsupported factual question.
- Stale-data question.
- Recommendation consistency.
- Evidence coverage.
- Cost-limit exhaustion.
- BYOK provider outage.
- Managed-provider outage.

### Operations

- Failed migration.
- Rollback.
- Database restore.
- Redis failure.
- Worker failure.
- Queue backlog.
- Provider rate limit.
- Secret rotation.
- Frontend rollback.
- API rollback.
- Model rollback.
- Feature-flag disable.

---

## 20. Traceability Matrix

| Critique category | Primary phases | Final evidence |
|---|---|---|
| Core analytical value | 4, 5, 9, 11 | Benchmark report, usability report, outcome dashboard |
| Onboarding and adoption | 3, 8, 9, 10 | Funnel metrics, support metrics, usability tests |
| Multi-tenancy and isolation | 2, 6, 7 | Isolation suite, penetration report |
| Reliability and scalability | 2, 6, 12 | SLO dashboard, load report, restore drill |
| Security and privacy | 1, 2, 7 | Privacy audit, threat model, penetration report |
| Billing and entitlements | 8 | Stripe scenario suite, reconciliation dashboard |
| Platform integrations | 1, 3 | Connector certification, licensing register |
| Maintainability and delivery | 0, 6, 12 | Required CI checks, release evidence |
| Data quality | 1, 3, 4 | Quality dashboard, lineage, reconciliation |
| Recommendation trustworthiness | 4, 5, 11 | Evidence replay, trust dashboard, user study |
| Prediction accuracy | 3, 4 | Holdout report, independent statistical review |
| UI/UX/design | 9, 10, 12 | WCAG audit, usability report, Web Vitals |
| AI features | 5, 7, 11 | AI evaluation report, red-team report, cost dashboard |

---

## 21. Cost and Approval Gates

The following work requires explicit owner approval before spending money or creating a production dependency:

- Commercial data licenses.
- Legal review.
- Managed PostgreSQL production tier.
- Managed Redis production tier.
- Object storage.
- Sentry paid tier if required.
- PostHog paid tier if required.
- Penetration testing.
- Accessibility audit.
- Independent statistical review.
- Production Clerk migration.
- Stripe Tax or equivalent tax service.
- Additional provider partner programs.

For each approval request, provide:

- The exact requirement it satisfies.
- Available options.
- Recurring and one-time cost.
- Operational burden.
- Lock-in risk.
- Security implications.
- Recommended choice.
- Consequence of deferral.

---

## 22. Final Definition of Done

HEATER may be rescored at 100/100 only when:

1. All 13 certification standards pass.
2. Every requirement links to current evidence.
3. All required CI checks pass on `master`.
4. Production completes the required observation period.
5. No Critical or High defect remains.
6. External penetration, accessibility, legal/privacy, and statistical reviews pass.
7. Commercial rights for every production data source are documented.
8. Tenant isolation is independently validated.
9. Model and recommendation claims have historical evidence.
10. Users can onboard, subscribe, connect, recover, export, and delete without administrator intervention.
11. Published SLOs are met.
12. The owner can operate and recover the service using runbooks.
13. A fresh independent audit awards 100 to every category without credit for planned or partial work.

---

## 23. How to Derive Implementation Plans

This document is a program plan, not a single implementation task. Execution must follow the existing HEATER planning hierarchy:

1. Select one phase or coherent work package.
2. Inspect current code and existing plans first.
3. Write or update one design spec when product or architecture decisions are needed.
4. Break the work into independently verifiable slices.
5. Create one detailed plan file per slice in `docs/superpowers/plans/`.
6. Use TDD for behavior and contract changes.
7. Add production telemetry in the same slice as production behavior.
8. Add evidence-registry entries before marking a slice complete.
9. Review security, data, tenancy, billing, and model implications.
10. Merge only after required checks and release gates pass.

No derived plan may claim that a category reaches 100 solely because code was merged. The category must also satisfy production, documentation, and independent-review evidence.
