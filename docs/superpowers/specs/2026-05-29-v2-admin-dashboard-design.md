# HEATER v2 — Plan 3: Admin Dashboard (Role-Based Nav + Controls + Analytics)

**Date:** 2026-05-29
**Status:** Approved (design)
**Parent spec:** [2026-05-28-v2-leaguemate-multiuser-design.md](2026-05-28-v2-leaguemate-multiuser-design.md) (§4 admin dashboard, §7 data model, §8 code change points, §9 testing)
**Predecessors:** Plan 1 — multi-user foundation (merged PR #120, master `a82ef45`) · Plan 2 — feedback inbox + usage logging (merged PR #121, master `9311e7f`)

---

## 1. Context & Goal

Plans 1–2 opened HEATER to the other 11 FourzynBurn league-mates and started collecting two
streams of operational data: a `feedback` table (in-context problem reports) and a `usage_events`
table (page views + `last_seen_at`). Both write paths are live; neither has a read surface beyond
the minimal Feedback tab in the Admin Console.

Plan 3 builds the **admin dashboard** — the surface where the admin (Connor) actually *operates*
the league deployment:

1. **Role-based navigation.** Today every `pages/*.py` shows in every user's sidebar (Streamlit's
   automatic directory nav). The admin tooling must be **invisible to the 11 league-mates** and
   visible only to the admin — which requires moving off auto-nav onto `st.navigation()`.
2. **Usage analytics.** Turn the accumulating `usage_events` data into surfaces: daily active
   users, most-used pages, per-user activity, last-seen rollups — plus **session timing**
   (login / exit / total time-on-app) and **per-page dwell time**, both newly required this plan.
3. **Operational controls.** Feature flags (disable a page deployment-wide), a broadcast banner,
   a maintenance/kill switch, and "view as another user" for support — every state-changing
   action recorded in an `audit_log`.
4. **CSV export** of feedback and usage for offline analysis.

**Goal:** Ship a complete admin dashboard — role-based nav, analytics (incl. session/dwell
timing), operational controls, audit logging, CSV export — **additive and gated by `MULTI_USER`**.
When the flag is off, the app is byte-for-byte v1: automatic `pages/` nav, no admin surfaces, no
new writes.

### The decision that shaped this plan: nav model

HEATER's automatic `pages/` discovery renders every file in `pages/` in every sidebar — it cannot
hide a page from a subset of users. To make the admin area invisible to non-admins we must adopt
`st.navigation()`, where `app.py` builds the page list programmatically and only includes admin
pages when `current_user().is_admin`. Crucially, **calling `st.navigation()` disables automatic
`pages/` discovery for that run**, which lets us branch cleanly on the flag:

- `MULTI_USER` **off** → never call `st.navigation()` → Streamlit's auto-nav runs → **v1 exactly**.
- `MULTI_USER` **on** → `app.py` builds pages via `st.navigation()` → role-based menus.

This is the linchpin that keeps the flag-off path untouched while enabling per-role nav when on.

---

## 2. In Scope / Out of Scope

### In scope (Plan 3)

- **Nav refactor:** `st.navigation()` role-based nav in `app.py`, flag-branched; `render_draft_page()`
  extraction; `pages/00_Admin_Console.py` → `pages/_admin_console.py` rename (the `_` prefix hides
  it from auto-nav so the flag-off sidebar is unchanged).
- **Five new tables** in `_init_multiuser_tables`: `feature_flags`, `audit_log`, `app_settings`,
  `sessions`, `page_visits`.
- **New modules:** `src/feature_flags.py`, `src/audit.py`, `src/app_settings.py`, `src/nav.py`.
- **Extensions:** `src/usage.py` (session + dwell tracking, analytics read helpers, `usage_csv()`),
  `src/feedback.py` (`feedback_csv()`), `src/auth.py` (view-as helpers).
- **New admin pages:** `pages/_admin_analytics.py` (usage + timing surfaces + CSV export),
  `pages/_admin_controls.py` (flags, broadcast, maintenance, view-as, audit-log viewer).
- **Two flag-enforcement points** for page-level flags (nav omission + page-top guard).
- **Entrypoint chrome:** view-as banner, broadcast banner, maintenance gate, 60s heartbeat.
- **Version pin:** Streamlit ≥ 1.37 in `requirements.txt`.
- Tests: structural guards, back-compat, pure unit/behavior.

### Out of scope (deferred to Plan 4 unless noted)

- **Data-status control plane** (force-refresh / re-run bootstrap / tail `bootstrap.log`) and
  **disk/DB health** — both are Railway-coupled; they land with hosting in Plan 4.
- **Tab-level / feature-level flags.** Plan 3 ships **page-level** flags only. The `feature_flags.key`
  column stays generic `TEXT` (`"page:<key>"`) so finer-grained keys are a later pure extension,
  no schema change.
- **True tab-close timestamps.** Streamlit has no `beforeunload` server event; "exit" means *last
  heartbeat / last activity* (accurate to ~60s). Accepted limitation (see §7).
- **Editing or migrating any existing table.** We only ADD tables and ADD read/write helpers.
- A separate admin account — Plan 3 uses a **single dual-role account** (see §3.2).

---

## 3. Architecture & Nav Model

### 3.1 Flag-branched entrypoint (`app.py`)

Today `app.py`'s `main()` runs `init_db()` → `require_auth()` → splash/bootstrap → draft page, and
Streamlit auto-discovers `pages/`. Plan 3 extracts the draft body into `render_draft_page()` and
branches `main()`:

```
main():
    init_db()
    if not multi_user_enabled():          # ── v1 path, byte-for-byte ──
        require_auth()                      # no-op when flag off
        render_draft_page()                 # splash + bootstrap + draft
        return                              # auto pages/ nav handles the rest

    # ── v2 path (flag on) ──
    require_auth()                          # login/register gate; sets current_user
    _render_view_as_banner()                # exit-impersonation affordance (FIRST)
    _render_broadcast_banner()              # if app_settings broadcast_enabled
    _enforce_maintenance_gate()             # if maintenance on AND user not admin → st.stop()
    _start_heartbeat()                      # st.fragment(run_every="60s") → bump last_activity_at
    pages = build_pages(current_user())     # src/nav.py → role-based Page list
    st.navigation(pages).run()
```

**Entrypoint ordering rationale.** The view-as banner renders **before** the maintenance gate so an
admin impersonating a user who is locked out by maintenance can still see (and click) "Exit view-as"
— you can't get trapped. The broadcast banner shows to everyone (admins included). The maintenance
gate always lets admins through.

### 3.2 Single dual-role account

The admin (Connor) uses **one account** with `is_admin=1` and `team_name="Team Hickey"` — it is both
his player identity and his admin identity. The Admin section of the nav is rendered only when
`current_user().is_admin`, so the 11 league-mates never see it. The "what does a normal user
experience" need is met by **view-as-user** (§5.5), not by a second login. (Seeded via the existing
`ensure_bootstrap_admin()` from `ADMIN_USERNAME`/`ADMIN_PASSWORD`/`ADMIN_TEAM_NAME`.)

### 3.3 `st.navigation()` page groups + the `_`-prefix trick

`build_pages(user)` returns a grouped dict for `st.navigation()`:

- **"Draft"** → the draft tool (`render_draft_page` as a callable `st.Page`).
- **"In-Season"** → the 13 interactive pages **filtered through `filter_enabled_pages()`** so a
  flag-disabled page is simply absent from the menu.
- **"Admin"** (only if `user["is_admin"]`) → `_admin_console`, `_admin_analytics`, `_admin_controls`.

Admin page files live in `pages/` but are named with a leading underscore (`_admin_console.py`,
`_admin_analytics.py`, `_admin_controls.py`). **Streamlit's automatic discovery ignores `_`-prefixed
files**, so when the flag is off (auto-nav) they never appear; when the flag is on, `st.navigation()`
routes to them by explicit path. This is what lets the admin pages coexist with the flag-off sidebar
without polluting it.

`src/nav.py` owns the canonical `PAGE_REGISTRY` (the 13 in-season pages as `{key, title, path}`,
plus an optional `icon`). Both `build_pages()` and the feature-flag control panel import this one
registry, so the nav and the flag toggles can never disagree about which pages exist.

**Icons:** `st.Page(icon=...)` accepts only an emoji or a Material Symbols token (`":material/…:"`),
**not** raw SVG. HEATER's existing `PAGE_ICONS` are inline-SVG strings used as *in-page* headers and
are left untouched. If we want nav icons we use Material Symbols tokens (consistent with the "no
emoji" convention); otherwise the `icon` field is omitted and the nav renders text-only. The in-page
SVG headers are unaffected either way.

---

## 4. Data Model

All five tables are appended to the existing `conn.executescript(...)` in
`_init_multiuser_tables(conn)` (`src/database.py:798`), after `usage_events`. `CREATE TABLE IF NOT
EXISTS` keeps it idempotent; tables are **always created** (so a flag flip needs no migration) and
**only written when the flag is on**. FK `REFERENCES` clauses are documentary (SQLite does not
enforce FKs without `PRAGMA foreign_keys=ON`, which `get_connection()` does not set) — consistent
with the rest of the schema. The multiuser tables are intentionally **not** added to
`_VALID_TABLE_NAMES` (that frozenset gates a different code path; the existing `users`/`feedback`/
`usage_events` tables are absent from it too — we stay consistent).

```sql
-- Page-level feature flags. Absence of a row = ENABLED (no pre-seeding).
CREATE TABLE IF NOT EXISTS feature_flags (
    key        TEXT PRIMARY KEY,              -- "page:1_My_Team"; generic TEXT for future keys
    enabled    INTEGER NOT NULL DEFAULT 1,
    updated_by INTEGER REFERENCES users(user_id),
    updated_at TEXT
);

-- Append-only record of every admin state change.
CREATE TABLE IF NOT EXISTS audit_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    admin_id   INTEGER NOT NULL REFERENCES users(user_id),
    action     TEXT NOT NULL,                 -- approve_user|toggle_flag|view_as|exit_view_as|
                                              -- set_broadcast|toggle_maintenance|export_csv
    target     TEXT,                          -- nullable (e.g. username, page key)
    detail     TEXT,                          -- nullable JSON
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit_log(created_at);

-- Key/value store for singleton app state (broadcast + maintenance).
CREATE TABLE IF NOT EXISTS app_settings (
    key        TEXT PRIMARY KEY,              -- broadcast_enabled|broadcast_message|
                                              -- maintenance_enabled|maintenance_message
    value      TEXT,
    updated_by INTEGER REFERENCES users(user_id),
    updated_at TEXT
);

-- One row per browser session; reuses the Plan 2 usage session_id.
CREATE TABLE IF NOT EXISTS sessions (
    session_id       TEXT PRIMARY KEY,         -- = st.session_state["_usage_session_id"]
    user_id          INTEGER NOT NULL REFERENCES users(user_id),
    login_at         TEXT NOT NULL,
    last_activity_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- One row per page visit within a session; dwell computed on navigation/idle-close.
CREATE TABLE IF NOT EXISTS page_visits (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    user_id       INTEGER NOT NULL REFERENCES users(user_id),
    page          TEXT NOT NULL,
    enter_at      TEXT NOT NULL,
    exit_at       TEXT,                         -- NULL while open / last page of session
    dwell_seconds REAL                          -- NULL until the visit is closed
);
CREATE INDEX IF NOT EXISTS idx_page_visits_session ON page_visits(session_id);
```

**Derived metrics (no stored aggregates):**
- Total time-on-app for a session = `last_activity_at − login_at`.
- Per-page time = `SUM(dwell_seconds)` over that page's `page_visits` rows.
- A still-open final visit is closed **lazily at query time** using the session's
  `last_activity_at` (no background worker — that's a Plan 4 concern). See §7.

---

## 5. Components / Modules

### 5.1 `src/nav.py` (new)

```python
PAGE_REGISTRY: list[dict]   # 13 entries: {"key": "1_My_Team", "title": "My Team",
                            #              "path": "pages/1_My_Team.py"}  # optional "icon"
                            #              (Material Symbols token, never raw SVG)

def build_pages(user: dict) -> dict[str, list]:
    """Build the grouped st.navigation() page structure for this user (a {section: [st.Page]} dict).
    Draft + flag-filtered In-Season for everyone; Admin group only if user['is_admin']."""

def filter_enabled_pages(keys: list[str], flags: dict[str, bool]) -> list[str]:
    """Pure helper: drop keys whose flag is explicitly False; keep absent keys
    (absence = enabled). Unit-testable with no Streamlit import."""
```

`PAGE_REGISTRY` is the single source of truth for page identity; `filter_enabled_pages` is pure so
the selection logic is tested without a Streamlit harness.

### 5.2 `src/feature_flags.py` (new)

```python
def is_page_enabled(key: str) -> bool:
    """True if no flag row OR row.enabled=1. Returns True when MULTI_USER off
    (defense-in-depth: flags never hide pages in v1)."""

def set_page_flag(key: str, enabled: bool, admin_id: int) -> None:
    """Upsert a feature_flags row + write an audit_log('toggle_flag') entry."""

def list_page_flags() -> dict[str, bool]:
    """All flag rows as {key: enabled} for the control panel + nav builder."""

def require_page_enabled(key: str) -> None:
    """Page-top guard (secondary enforcement). No-op when MULTI_USER off OR user is admin;
    otherwise if disabled → st.error(...) + st.stop()."""
```

Absence-means-enabled keeps the table inert until the admin explicitly disables something.

### 5.3 `src/audit.py` (new)

```python
def log_action(admin_id: int, action: str, target: str | None = None,
               detail: dict | None = None) -> None:
    """Append an audit_log row (detail json.dumps'd). No-op when MULTI_USER off."""

def list_audit(limit: int = 200, action: str | None = None) -> list[dict]:
    """Recent audit rows, newest first, optionally filtered by action,
    joined to users for the admin's display name."""
```

### 5.4 `src/app_settings.py` (new)

```python
def get_setting(key: str, default: str | None = None) -> str | None
def set_setting(key: str, value: str, admin_id: int) -> None     # upsert + audit

# typed convenience wrappers
def get_broadcast() -> dict      # {"enabled": bool, "message": str}
def set_broadcast(enabled: bool, message: str, admin_id: int) -> None
def get_maintenance() -> dict    # {"enabled": bool, "message": str}
def set_maintenance(enabled: bool, message: str, admin_id: int) -> None
```

### 5.5 `src/auth.py` (extend)

View-as-user, layered on the existing per-session identity (`_SESSION_KEY="auth_user"`):

```python
def enter_view_as(target_username: str, admin_id: int) -> None:
    """Stash the real admin under a separate session key, swap current identity to
    the target user, write audit_log('view_as'). Requires caller is admin."""

def exit_view_as() -> None:
    """Restore the stashed admin identity; audit_log('exit_view_as')."""

def is_viewing_as() -> dict | None:
    """Return the real admin dict if a view-as is active, else None
    (drives the entrypoint banner)."""
```

While view-as is active, `current_user()` returns the **target** — so every personalized surface,
nav (no Admin group), and page guard behaves exactly as the impersonated user experiences it.

### 5.6 `src/usage.py` (extend) — the most-changed existing module

Plan 2's `log_page_view(page, action="view")` early-returns on the per-`(page,action)`-per-session
dedup. Plan 3 **restructures** so session + timing updates run on **every** call **before** the
existing deduped `usage_events` insert:

```
log_page_view(page, action="view"):
    if not multi_user_enabled() or current_user() is None: return
    _ensure_session_row()          # INSERT sessions row once (login_at), reuse session_id
    _track_page_visit(page)        # if page != session_state["_current_page"]:
                                   #     close prior open visit (exit_at, dwell_seconds)
                                   #     INSERT new page_visits row (enter_at); update _current_page
    _bump_last_activity()          # UPDATE sessions.last_activity_at = now  (every call)
    # ── existing Plan 2 behavior, unchanged below this line ──
    if (page, action) in _usage_logged: return
    INSERT usage_events; UPDATE users.last_seen_at; _usage_logged.add((page, action))
```

Read side (analytics helpers + export):

```python
def dau_series(days: int = 30) -> list[dict]          # date -> distinct active users
def most_used_pages(days: int = 30) -> list[dict]      # page -> view count
def per_user_activity() -> list[dict]                  # user -> sessions, total time, last seen
def last_seen_summary() -> list[dict]                  # user -> last_seen_at
def session_timeline(user_id: int | None = None) -> list[dict]
                                                       # per-session login/last-activity/duration
def page_dwell_summary(user_id: int | None = None) -> list[dict]
                                                       # page -> total time, visits, avg dwell
def usage_csv() -> str                                 # usage_events as CSV text
```

`session_timeline` / `page_dwell_summary` perform the **lazy idle-close**: any `page_visits` row with
`exit_at IS NULL` is treated as ending at its session's `last_activity_at` for the purpose of the
computed duration (the row is also UPDATEd so repeated queries converge).

### 5.7 `src/feedback.py` (extend)

```python
def feedback_csv() -> str   # all feedback rows (joined to users) as CSV text
```

### 5.8 `pages/_admin_console.py` (rename of `00_Admin_Console.py`)

Identical content (Users + Feedback tabs, `require_admin()` gate). Only the filename changes so the
`_` prefix removes it from auto-nav; `st.navigation()` routes to it explicitly under the Admin group.

### 5.9 `pages/_admin_analytics.py` (new)

`require_admin()`-gated. Renders, from the §5.6 read helpers:
- **Usage:** DAU series (line), most-used pages (bar), per-user activity table, last-seen rollup.
- **Session timing:** per-session timeline (login / last-seen / duration); per-user rollups
  (total time, session count, avg session length, last seen).
- **Page dwell:** per-page total time / visit count / avg dwell — both aggregate and per-user.
- **Export:** two `st.download_button`s wired to `usage_csv()` and `feedback_csv()`; each click
  writes `audit_log('export_csv', target=...)`.

### 5.10 `pages/_admin_controls.py` (new)

`require_admin()`-gated. Five control surfaces, each state change audited:
- **Feature flags:** a toggle per `PAGE_REGISTRY` entry → `set_page_flag(...)`.
- **Broadcast:** message text + enable checkbox → `set_broadcast(...)`.
- **Maintenance:** on/off + optional message → `set_maintenance(...)`.
- **View-as:** selectbox of active users + button → `enter_view_as(...)` + `st.rerun()`.
- **Audit log:** read-only table from `list_audit()` with an optional action filter.

---

## 6. Page Integration & Flag Enforcement

Page-level flags have **two enforcement points** (defense-in-depth):

1. **Primary — nav omission.** `build_pages()` runs the In-Season registry through
   `filter_enabled_pages()`; a disabled page is not in `st.navigation()` and is not routable (the
   URL falls back to the default page).
2. **Secondary — page-top guard.** Each of the 13 interactive pages calls
   `require_page_enabled("page:<key>")` immediately after `require_auth()`. This covers direct
   navigation via `st.page_link` / `st.switch_page` / a stale bookmark. It is a no-op when the flag
   is off or the user is admin. A structural guard test enforces its presence on every page.

This sits alongside the Plan 1 `require_auth()` and Plan 2 `log_page_view()` /
`render_feedback_widget()` calls already wired into all 13 pages — the per-page top-of-file block
grows by one line.

---

## 7. Session & Dwell-Time Tracking

**The constraint:** Streamlit runs server-side; a browser tab close fires no server event we can
catch (`beforeunload` JS is unreliable and out of scope). So "exit time" cannot be the exact instant
a tab closed.

**The model (accurate to ~60s):**
- A `st.fragment(run_every="60s")` heartbeat in the entrypoint calls a tiny "bump last_activity_at"
  function while the tab is open. Exit ≈ last heartbeat.
- `log_page_view()` (already called near the top of every page) does the navigation-aware
  `page_visits` bookkeeping: closing the prior visit and opening a new one when the active page
  changes (diffed against `st.session_state["_current_page"]`).
- The **final** open visit of a session and the session's effective end are resolved **lazily at
  query time** in the analytics helpers, using `last_activity_at`. No background sweeper (Plan 4).

**Accepted limitation (user-confirmed):** displayed "exit" / total-time values reflect last activity
within the heartbeat interval (~60s), not a true tab-close timestamp. The dashboard labels these as
"last activity"–based.

`st.fragment(run_every=...)` requires Streamlit ≥ 1.37; `st.navigation()` requires ≥ 1.36. We pin
**≥ 1.37** in `requirements.txt` to cover both.

---

## 8. Behavior & Edge-Case Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Nav when `MULTI_USER` off | **Never call `st.navigation()`** → auto `pages/` nav | Byte-for-byte v1; the whole refactor is inert when the flag is off. |
| Hiding admin pages | **`_`-prefix filenames** + `st.navigation()` explicit routing | Auto-discovery ignores `_`-prefixed files, so flag-off sidebar is unchanged; nav routes to them when on. |
| Admin account | **Single dual-role account** (`is_admin=1`) | One identity; Admin nav group shown only to admins; view-as covers the "normal user view" need. |
| Feature-flag default | **Absence = enabled** | Table ships empty/inert; no pre-seeding; a page is hidden only by an explicit `enabled=0` row. |
| Flag granularity | **Page-level only**, `key` stays generic `TEXT` | YAGNI; tab/feature keys (`"page:X:tab:Y"`) are a later pure extension, no schema change. |
| Flag enforcement | **Nav omission + page-top guard** | Primary hides it; secondary blocks direct-URL/bookmark access. |
| Broadcast storage | **`app_settings` KV (enabled + message)** | A bool-only table can't carry the message text; KV holds both. |
| Maintenance + admin | **Admin always bypasses; non-admins `st.stop()`** | The admin must be able to operate the app while it's "down" for others. |
| View-as vs maintenance order | **View-as banner before maintenance gate** | Can't get trapped impersonating a locked-out user — the exit affordance always renders first. |
| "Exit" timestamp | **Last activity within ~60s heartbeat** | No server-side tab-close event exists in Streamlit; documented + labeled accordingly. |
| Idle-close of final visit | **Lazy, at query time, via `last_activity_at`** | Avoids a background worker (deferred to Plan 4); queries converge by UPDATEing the row. |
| Audit scope | **Every state-changing admin action** | `toggle_flag`, `view_as`, `exit_view_as`, `set_broadcast`, `toggle_maintenance`, `export_csv`, plus the existing `approve_user`. |
| All new helpers when flag off | **No-op / pass-through** | `is_page_enabled`→True, writers/loggers return early; zero DB writes. |

---

## 9. Testing & Structural Guards

### 9.1 Structural guard tests (new + updated)

- **`test_admin_tables.py`** (new, mirrors `test_feedback_usage_tables.py`) — `init_db()` creates
  all five tables idempotently; asserts key columns + the `audit_log(created_at)` index; confirms
  `feature_flags` absence-means-enabled default behavior.
- **`test_admin_pages_flag_enforced.py`** (new) — every one of the 13 interactive pages calls
  `require_page_enabled("page:<key>")` after `require_auth()` (AST scan, no allowlist).
- **`test_nav_registry_matches_pages.py`** (new) — `src/nav.py::PAGE_REGISTRY` keys/paths map 1:1
  to the real `pages/*.py` files, so the registry can't drift from disk.
- **`test_admin_console_guarded.py`** (update) — repoint `pages/00_Admin_Console.py` →
  `pages/_admin_console.py`; still assert `require_admin()` precedes any lifecycle action.
- **`test_pages_have_feedback_and_usage.py`** (update) — exclude the three `_admin_*.py` pages from
  the feedback+usage sweep (admin pages are exempt, as the console already is).

### 9.2 Back-compat tests (`MULTI_USER` off ⇒ v1)

- **`test_admin_backcompat.py`** (new, mirrors `test_feedback_usage_backcompat.py`) — with the flag
  off: `is_page_enabled()` returns True without DB access; `log_action()` / `set_setting()` are
  no-ops; the session/dwell logger writes zero rows to `sessions` / `page_visits`; `app.py` takes
  the auto-nav branch (no `st.navigation()` call). Asserts zero DB writes via a connection spy.
- Existing `test_auth_backcompat.py` and `test_feedback_usage_backcompat.py` remain green unchanged.

### 9.3 Pure unit / behavior tests (no Streamlit)

- **`filter_enabled_pages`** — disabled key omitted; absent key kept (absence=enabled); empty flags
  → all kept.
- **view-as** — `enter_view_as` makes `current_user()` return the target; `exit_view_as` restores
  the admin; each writes one audit row (against an injected dict session-state seam, the pattern
  `usage.py` already uses for `_session_state()`).
- **dwell computation** — consecutive `page_visits` yield correct `dwell_seconds`; the lazy
  idle-close uses `last_activity_at` to close the final open visit; total-app-time =
  `last_activity_at − login_at`.
- **audit writers** — `set_page_flag` / `set_broadcast` / `set_maintenance` each write exactly one
  audit row with the right `action`.

### 9.4 Version pin

- **`test_streamlit_min_version.py`** (new) — asserts `requirements.txt` pins Streamlit ≥ 1.37, so
  the nav + heartbeat features can't be undercut by an older resolve.

All new tests obey the `tests/conftest.py` network guard (DB-local only) and use the per-worker
SQLite DB, so writing real rows to the five tables is safe and isolated.

---

## 10. Back-Compat / `MULTI_USER` Off

- The five tables are always created by `init_db()` (idempotent); nothing reads or writes them when
  the flag is off.
- `app.py` takes the **auto-nav branch** when the flag is off — `st.navigation()` is never called,
  the `_`-prefixed admin files stay hidden by auto-discovery, and the sidebar is the v1 sidebar.
- `is_page_enabled()` returns True, `require_page_enabled()` is a no-op, and all
  audit/settings/session/dwell writers return early — zero DB writes, zero new UI.
- No existing table, function signature, or page behavior changes for v1 users. The
  `00_Admin_Console.py` → `_admin_console.py` rename is invisible in v1 (the admin only exists under
  `MULTI_USER`, and the console did nothing in v1 anyway).

---

## 11. File Manifest

| File | Action | Responsibility |
|------|--------|----------------|
| `src/nav.py` | create | `PAGE_REGISTRY` + `build_pages(user)` + pure `filter_enabled_pages` |
| `src/feature_flags.py` | create | `is_page_enabled` / `set_page_flag` / `list_page_flags` / `require_page_enabled` |
| `src/audit.py` | create | `log_action` / `list_audit` |
| `src/app_settings.py` | create | KV `get/set_setting` + broadcast/maintenance wrappers |
| `src/usage.py` | modify | session + dwell tracking; analytics read helpers; `usage_csv()` |
| `src/feedback.py` | modify | add `feedback_csv()` |
| `src/auth.py` | modify | `enter_view_as` / `exit_view_as` / `is_viewing_as` |
| `src/database.py` | modify | add 5 tables to `_init_multiuser_tables` |
| `app.py` | modify | `render_draft_page()` extraction; flag-branched `st.navigation()` entrypoint; banners; maintenance gate; heartbeat |
| `pages/00_Admin_Console.py` | rename → `pages/_admin_console.py` | hide from auto-nav; content unchanged |
| `pages/_admin_analytics.py` | create | usage + session/dwell surfaces + CSV export |
| `pages/_admin_controls.py` | create | flags / broadcast / maintenance / view-as / audit viewer |
| `pages/*.py` (13 interactive) | modify | add `require_page_enabled("page:<key>")` after `require_auth()` |
| `requirements.txt` | modify | pin Streamlit ≥ 1.37 |
| `tests/test_admin_tables.py` | create | 5 tables created idempotently |
| `tests/test_admin_pages_flag_enforced.py` | create | every page calls `require_page_enabled` |
| `tests/test_nav_registry_matches_pages.py` | create | registry ↔ disk 1:1 |
| `tests/test_admin_backcompat.py` | create | MULTI_USER-off inert (zero writes, auto-nav) |
| `tests/test_streamlit_min_version.py` | create | Streamlit ≥ 1.37 pin |
| `tests/test_admin_console_guarded.py` | modify | repoint to `_admin_console.py` |
| `tests/test_pages_have_feedback_and_usage.py` | modify | exclude `_admin_*.py` pages |

---

## 12. Deferred to Plan 4 (recorded so it isn't lost)

- **Data-status control plane:** force-refresh data / re-run bootstrap / tail `bootstrap.log` from
  the dashboard (Railway-coupled).
- **Disk / DB health surfaces** (Railway-coupled).
- **Background idle-close sweeper** for `page_visits` (Plan 3 closes lazily at query time instead).
- **Tab-level / feature-level feature flags** (the `key` column already supports them; only UI +
  enforcement granularity remain).
- Railway hosting + scheduled-refresh data model (the rest of Plan 4).
