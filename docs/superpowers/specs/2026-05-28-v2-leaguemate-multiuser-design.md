# HEATER v2 — Leaguemate Multi-User Design

**Date:** 2026-05-28
**Status:** Approved (design); pending implementation plan
**Author:** Connor Hickey (Team Hickey) + Claude
**Supersedes nothing** — additive evolution of the single-user HEATER app.

---

## 1. Context & Goal

HEATER today is a **single-tenant** Streamlit fantasy-baseball manager: one user (Connor / Team Hickey), one local machine, one Yahoo league (FourzynBurn, 12-team H2H Categories). `LeagueConfig` (`src/valuation.py:176`) hardcodes `num_teams=12` and the 6×6 category set; the active league comes from the `YAHOO_LEAGUE_ID` env var; a global flag `league_teams.is_user_team = 1` marks "my team" for all personalization.

**v2 goal:** open the app to **the other 11 members of the same FourzynBurn league** for season-long competitive use — each with their own login, their own personalized view pinned to their own team, reliable always-on uptime, and a privacy guarantee that no one can peek at a rival's personalized analysis. Plus an **admin dashboard** (admin = Connor only) to run the whole thing.

### What makes this tractable

All 12 users are in **one shared league**. This collapses the hardest multi-tenant problems:

- **No per-league calibration.** The same league math (`num_teams=12`, the same SGP denominators, the same category set) is correct for all 12 users because it *is* the same league. Nothing in `LeagueConfig` needs to become per-tenant.
- **No per-user Yahoo write tokens.** HEATER is **read-only** against Yahoo — it never sets lineups, makes adds/drops, or posts trades (verified: no `add_drop` / `place_transaction` / `set_lineup` / `post` in `src/yahoo_api.py`). So one **shared read-only Yahoo connection** serves all 12 users. No per-user OAuth.

The mental model is therefore **not** multi-tenant data isolation. It is **"one shared league brain with a per-person nameplate":** shared league data (correct for everyone) + a per-session identity that replaces the global `is_user_team=1` flag.

---

## 2. Identity: self-registration with an admin approval gate

Users create their own accounts, but an admin approval gate stands between signup and access. The combination satisfies both "let them make their own account" and "invoke and revoke account access," and is *more* secure than self-asserted team selection.

### Flow

1. A leaguemate opens the app → **Create account** → picks username + password (bcrypt-hashed via `streamlit-authenticator`).
2. The account lands in **`pending`** status. They see only a "waiting for approval" screen — no analysis.
3. The admin sees the pending signup in the dashboard, **approves it and assigns their Yahoo team** from a dropdown of the 12 FourzynBurn teams.
4. The account flips to **`active`**, pinned to its assigned team; the app personalizes.
5. The admin can flip any account to **`revoked`** (instant lockout) or reassign its team at any time.

### Why an approval gate (not self-selected teams)

In a 12-person competitive league, self-asserted team selection is an **impersonation hole**: a user could select a rival's team to read that rival's private optimizer / FA / alert analysis. Admin-assigned team at approval means identity is vouched-for, not self-claimed.

### Privacy model

- **Personalized surfaces** — My Team / War Room, Lineup Optimizer, FA recommendations, alerts — pin to the logged-in user's assigned team and **cannot be impersonated**.
- **Competitive tools** — Trade Analyzer / Finder, Matchup Planner, League Standings, Leaders — **legitimately show all rosters**, exactly as Yahoo itself does. This is by design, not a leak.
- The guarantee is: *your identity and personal surface can't be impersonated.* It is **not** "opponents' rosters are secret" (they never were — Yahoo shows them).

---

## 3. Feedback capture next to every feature

A single reusable feedback widget (an `st.popover` / expander labeled "Send feedback on this") renders at the foot of every page and major tab.

On submit it writes to a **`feedback`** table and **auto-captures context** the user would otherwise have to describe: the submitting user, the page/tab, the app version, and the data-freshness snapshot at that moment.

The admin dashboard exposes a **feedback inbox**: filterable by page / user / status, each item markable `new → triaged → resolved`, with an admin-notes field.

**Rationale:** auto-captured context (page + user + version + data-state) is what makes feedback actionable. It's cheap to grab at submit time and nearly impossible to reconstruct later.

---

## 4. Admin dashboard

Admin = Connor only. The dashboard is the control plane for the whole deployment.

### The "separate site" decision

The requirement "a separate site only I have access to" is an **access** requirement, satisfied with auth + routing rather than a literally separate deployment.

**Decision: one Railway service; the admin dashboard is a hard-walled, `is_admin`-gated area** — invisible in every non-admin's navigation, and a hard-stop (not merely hidden) if a non-admin reaches its route directly.

**Why not a literally separate service:** HEATER's data layer is deeply coupled to SQLite — `get_connection()` with its WAL / `busy_timeout` PRAGMAs, the 33-phase bootstrap, and structural tests that forbid raw `sqlite3.connect`. Railway attaches a persistent volume to **one** service, so a truly separate admin service would force migrating the entire app onto a networked Postgres — a change that would dwarf the multi-user effort and risk the working app. "Separate access" is what's actually needed; "separate process/database" is not. If a literally separate URL is ever wanted, that is the moment to evaluate Postgres — it does not gate v2.

### Contents (mapped to stated requirements)

| Requirement | Dashboard feature |
|---|---|
| Invoke / revoke account access | **Account management** — pending-signup queue (approve + assign team), active-user list (revoke / reassign team) |
| Hide any page / tab / feature | **Feature-flag panel** backed by `feature_flags`; app reads flags at render time; hidden pages drop from nav **and** hard-stop if hit directly |
| Track usage | **Usage panel** backed by `usage_events`; daily-active-users, per-user activity, most-used pages, last-seen |
| Get feedback messages | **Feedback inbox** (Section 3) |
| View and control in-app data statuses | **Data-status panel** surfacing existing `refresh_log` / `get_refresh_log_snapshot()` / `DataFreshnessTracker` telemetry (per-source status, tier, last-refresh, staleness) **plus** controls: force-refresh a source, re-run bootstrap, tail `bootstrap.log` |

**Note:** the data-status panel is mostly a *surfacing* job — the per-source health telemetry already exists (`refresh_log`, `DataFreshnessTracker`). The admin value-add is a **control plane** layered on existing observability.

### Recommended additional admin controls

- **Maintenance / kill switch** — flip the player app to a "down for maintenance" screen during risky data operations.
- **Broadcast banner** — push a message to all players ("data refresh running, stats may lag").
- **View-as-user** — load the app as a given user to reproduce a reported bug (audit-logged).
- **Audit log** — every admin action recorded (approved X, revoked Y, toggled flag Z, forced refresh) for accountability + history.
- **Disk / DB health** — persistent-volume usage %, SQLite file + WAL size, surfaced before the disk fills.
- **Export** — feedback and usage as CSV.

---

## 5. Railway hosting + the data-freshness inversion

### The container

No Dockerfile exists today; v2 creates one:

- Python **3.12** base (matching CI, not local 3.14, so the deployed environment mirrors what's tested).
- `pip install -r requirements.txt`, then the `--no-deps` install of `yfpy` + `streamlit-oauth` exactly as `requirements.txt` documents.
- Expose Streamlit's port; start with `streamlit run app.py`.
- Railway builds the image, runs it as one always-on service (~$7–15/mo), and provides HTTPS on its domain automatically.

### Persistent volume

Railway attaches a volume mounted at `data/`, holding the three things that must survive restarts:

- `draft_tool.db` — the shared SQLite DB
- `yahoo_token.json` — the shared read-only Yahoo credential
- `logs/bootstrap.log`

Everything else is rebuilt from the image.

### Secrets

Railway env vars (never in the volume, never in the repo): Yahoo client id/secret, `YAHOO_LEAGUE_ID`, the auth cookie signing key, and the bootstrap-admin credentials that seed the admin account on first boot.

### The Yahoo token (the one real wrinkle)

Today `data/yahoo_token.json` is created by the interactive `oob` OAuth flow (a human visits a Yahoo URL, authorizes, pastes a code back). That is awkward on a headless server. Path:

1. Run the OAuth dance **locally, once** (as today), producing `yahoo_token.json` with a long-lived **refresh token**.
2. Get that file onto Railway's volume once — a one-time upload, or an admin-dashboard "paste your Yahoo token" action that writes it to the volume.
3. From then on the refresh token silently mints new access tokens; no human interaction unless Yahoo revokes it.

The durable credential is the **refresh token** (access tokens expire hourly). The admin data-status panel surfaces Yahoo connection health, so a dead refresh token is visible and re-uploadable rather than failing silently into stale data.

### The behavioral inversion (largest change in v2)

CLAUDE.md documents that bootstrap runs with `force=True` on **every new browser session**. Correct for one local user; **pathological for 12 shared users** — every new tab would kick off the 33-phase pipeline and hammer rate-limited upstreams (FanGraphs, Yahoo, Statcast).

**v2 inverts the model:** one **scheduled server-side refresh** (Railway cron — nightly + a lighter intraday pass) writes the shared DB, and **user sessions become read-only consumers** that never trigger bootstrap.

This inversion also *fixes* the write-contention HEATER has band-aided (the `busy_timeout` bumps, the pvb_splits commit-per-batter fix): heavy writes centralize to one cron job; user sessions do only tiny writes (feedback, usage, login). Twelve concurrent readers on WAL SQLite is a non-event; the existing WAL + 60s `busy_timeout` goes from "barely enough" to "comfortably over-provisioned."

---

## 6. Rollout: additive, behind a mode flag, local loop untouched

**Guiding principle:** everything is additive and gated, so the working single-user app is never at risk and the local dev workflow does not change.

- **`MULTI_USER` env flag** switches behavior:
  - **Off** (laptop): no login, bootstrap-on-session, the existing `is_user_team=1` global flag — identical to today.
  - **On** (Railway): login required, no per-session bootstrap, identity from the session.
- **New tables only** — `users`, `feedback`, `feature_flags`, `usage_events`, `audit_log` — added through `init_db()` migrations. No existing column or table changes.
- **The one load-bearing edit:** `_get_user_team_name()` (`src/yahoo_data_service.py:1103`, currently `SELECT team_name FROM league_teams WHERE is_user_team = 1`) learns to read the logged-in user's assigned team from the session when `MULTI_USER` is on, falling back to the old global flag when off. A structural test guards both paths.
- **Admin seeded first** from the bootstrap-admin env credentials; the other 11 are approved as they sign up.
- **New test coverage:** auth gating, feedback capture, feature-flag enforcement (hidden page hard-stops), admin-only access (non-admin stopped cold), per-session identity isolation, and a guard that `MULTI_USER` off reproduces today's single-user behavior exactly.

**Why the flag matters:** the same codebase serves both the untouched local single-user loop and the deployed multi-user app. At no point does v2 work force giving up the working app. Additive-only schema + a behavior flag is the standard pattern for evolving a live system without a risky big-bang cutover.

---

## 7. Data model — new tables

All additive, created in `init_db()` (no changes to existing tables).

### `users`
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `username` | TEXT UNIQUE | login |
| `password_hash` | TEXT | bcrypt |
| `team_name` | TEXT NULL | assigned by admin at approval; NULL while pending |
| `display_name` | TEXT | |
| `status` | TEXT | `pending` \| `active` \| `revoked` |
| `is_admin` | INTEGER | 0/1 |
| `created_at` | TEXT | UTC ISO |
| `last_seen_at` | TEXT NULL | updated by usage logging |

### `feedback`
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `user_id` | INTEGER FK → users | |
| `page` | TEXT | auto-captured |
| `feature_tag` | TEXT NULL | tab/feature within page |
| `message` | TEXT | |
| `app_version` | TEXT | auto-captured |
| `data_state` | TEXT NULL | freshness snapshot JSON at submit time |
| `status` | TEXT | `new` \| `triaged` \| `resolved` |
| `admin_notes` | TEXT NULL | |
| `created_at` | TEXT | UTC ISO |

### `feature_flags`
| Column | Type | Notes |
|---|---|---|
| `key` | TEXT PK | page / tab / feature id |
| `enabled` | INTEGER | 0/1 |
| `updated_by` | INTEGER FK → users | |
| `updated_at` | TEXT | UTC ISO |

### `usage_events`
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `user_id` | INTEGER FK → users | |
| `page` | TEXT | |
| `action` | TEXT | e.g. `view`, `ran_optimizer`, `evaluated_trade` |
| `session_id` | TEXT | |
| `created_at` | TEXT | UTC ISO |

### `audit_log`
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `admin_id` | INTEGER FK → users | |
| `action` | TEXT | e.g. `approve_user`, `revoke_user`, `toggle_flag`, `force_refresh`, `view_as` |
| `target` | TEXT NULL | affected user / flag / source |
| `detail` | TEXT NULL | JSON |
| `created_at` | TEXT | UTC ISO |

---

## 8. Key code change points

| Area | Change |
|---|---|
| `src/yahoo_data_service.py:1103` `_get_user_team_name()` | Session identity when `MULTI_USER` on; fall back to `is_user_team=1` when off |
| `app.py` | Login gate + registration UI when `MULTI_USER` on; suppress per-session bootstrap when on |
| `src/database.py` `init_db()` | Add the 5 new tables (additive migrations) |
| New `src/auth.py` | `streamlit-authenticator` wiring, bcrypt, status checks, admin gate |
| New `src/feedback.py` + widget helper | Reusable feedback widget + `feedback` writes |
| New `src/usage.py` | `usage_events` logging helper called at page top |
| New `src/feature_flags.py` | Read/write flags; render-time enforcement helper |
| New admin pages (gated) | Account mgmt, feature flags, usage, feedback inbox, data-status control, audit log, extras |
| New `Dockerfile` | Python 3.12, deps, `--no-deps` yfpy/streamlit-oauth, Streamlit start |
| New Railway cron entrypoint | Scheduled `bootstrap_all_data(force=True)` server-side |
| `pages/*` | Feature-flag guard + feedback widget at foot; usage logging at top |

---

## 9. Testing

- Auth: login required when `MULTI_USER` on; `pending`/`revoked` blocked; `active` admits.
- Identity isolation: user A's session never resolves to user B's team.
- Feature flags: disabling a page removes it from nav **and** hard-stops direct navigation.
- Admin gate: non-admin hard-stopped from every admin route.
- Feedback: submit writes a row with auto-captured page/user/version/data-state.
- Usage: page view logs an event.
- **Back-compat guard:** `MULTI_USER` off reproduces today's single-user behavior (bootstrap-on-session, `is_user_team=1` path) exactly.

---

## 10. Open items / explicitly deferred

- **Postgres migration** — deferred. Only revisited if a literally separate admin deployment (separate URL/process) is later required.
- **Per-user Yahoo OAuth** — not needed (read-only app; shared connection).
- **Per-league calibration / multi-league** — out of scope (one shared league).
- **Detailed build sequencing** — handled by the implementation plan (`writing-plans`), not this design doc.
