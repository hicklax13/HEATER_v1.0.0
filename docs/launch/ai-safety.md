# Bubba AI safety posture & gaps (Phase 14)

**Date:** 2026-06-26 · **Status:** posture consolidated from code review; one critical multi-tenant gap flagged for Phase 3.

Bubba (the AI assistant) is read-only on the league and reaches data through a small tool surface (`src/ai/tools.py`). This consolidates its safety properties (the Codex Phase 14 finding asks for a single place documenting AI tool safety) and records what remains.

## What is already guarded (verified by tests)

| Property | Mechanism | Test |
|---|---|---|
| SQL tool is read-only | `file:…?mode=ro` connection — writes impossible at the driver; plus a SELECT/WITH allowlist rejecting INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/PRAGMA/ATTACH/REPLACE | `tests/test_ai_sql_tool_readonly.py` |
| No multi-statement / injection chaining | single-statement only (rejects `;`) | `tests/test_ai_sql_tool_readonly.py` |
| Secret / PII / auth tables unreadable | `schema_card.excluded_tables()` blocks `ai_provider_keys`, `auth_tokens`, `users`, `sessions`, `audit_log`, `app_settings`, chat meta-tables, **and every per-user table** (`user_saved_views`, `user_watchlist`, `feedback`, `usage_events`, `page_visits`) — closes cross-member exfiltration | `tests/test_ai_sql_excludes_user_tables.py` |
| Only one write/side-effect tool, and it's gated | `request_refresh` is gated by `viewer_can_write()` (admin-only under MULTI_USER; the scheduler is the sole writer) | `tests/test_member_write_safety.py` |
| Tool dispatch never crashes the agent loop | `dispatch_tool` wraps everything in try/except → JSON error | `tests/test_ai_tools.py` |
| Read tools are side-effect-free | `query_data`, `get_player`, `compare_players`, `get_my_team`, `get_standings`, `get_free_agents` are pure reads; `web_search`/`deep_research` are opt-in | — |
| Managed AI spend is bounded | per-tier daily caps, fail-closed metering (`src/ai/budget.py`, `api/gating.py`) | (B3 tests) |

## ★ Critical gap — multi-tenant SQL-tool scoping (Phase 3 must-fix)

`run_read_only_sql` queries the **entire** `draft_tool.db`. Today that is one league's data, so a member's Bubba can only read this league (and the per-user/secret tables are excluded). **In the multi-tenant public product (Phase 3), this is a cross-tenant data-exfiltration vector:** a prompt-injected (or merely curious) user could `SELECT … FROM league_rosters` and read *other tenants'* rosters/standings/trades.

**Required fix (Phase 3 / 14):** scope the SQL tool to the caller's `tenant_id`/`league_id` — either by (a) constraining queries to tenant-scoped views, (b) injecting a mandatory `WHERE tenant_id = :ctx` (hard to enforce on free SQL), or (c) replacing the open SQL tool with **parameterized, tenant-scoped query templates** (the safest — drop arbitrary SQL for public tenants and expose only vetted, scoped queries). Tracked: registry `P14-TENANT-SQL-SCOPE` (deferred → Phase 3).

## Remaining Phase 14 follow-ups (need provider budget / later phases)

- **Live red-team eval** — the above proves the *scaffolding* is safe (the tool can't write or read forbidden tables). It does **not** prove the *model* resists prompt injection in conversation. A real red-team (injection fixtures run against each provider, measuring tenant-context correctness + refusal) needs provider keys + budget → run when managed AI is funded.
- **Grounding/evidence** — Bubba should cite HEATER recommendations + source freshness (Phase 8 evidence layer feeds this).
- **Per-provider no-train verification** — confirm + document each provider's data-use posture (see `docs/legal/AI_DISCLOSURE.md`).
- **Tenant-scoped tool context** — once Phase 3 lands, the read tools (`get_my_team` etc.) must resolve identity from the server-side viewer context, not ambient singletons.
