# HEATER v2 — Plan 2: Feedback Inbox + Usage Logging

**Date:** 2026-05-29
**Status:** Approved (design)
**Parent spec:** [2026-05-28-v2-leaguemate-multiuser-design.md](2026-05-28-v2-leaguemate-multiuser-design.md) (§3 feedback capture, §7 data model, §8 modules, §9 testing)
**Predecessor:** Plan 1 — multi-user foundation (merged PR #120, master `a82ef45`)

---

## 1. Context & Goal

Plan 1 opened HEATER to the other 11 FourzynBurn league-mates: self-registration → admin
approval → per-session identity, all behind the `MULTI_USER` env flag. With real league-mates
now using the app, we need two things the single-user app never required:

1. **A way for users to report problems in context** — a recommendation that looks wrong, a
   page that misbehaves, a number that doesn't add up. Email/Slack loses the context that makes
   a report actionable (which page, which app version, what data state).
2. **A record of who is actually using what** — so the admin (Connor) knows which surfaces are
   used, who's active, and (in a later plan) can chart engagement.

**Goal:** Add an in-context feedback widget on every page that writes to a `feedback` table an
admin can triage, plus lightweight page-view usage logging to a `usage_events` table. Both are
**additive and gated by `MULTI_USER`** — when the flag is off, the app is byte-for-byte v1.

### Scope boundary (the one decision that shaped this plan)

The parent spec groups "feedback inbox" with "usage analytics" and "admin dashboard" under a
later plan. This plan deliberately splits that:

- **Feedback: end-to-end in Plan 2.** Submit widget + a minimal triage inbox added as a new tab
  in the *existing* `pages/00_Admin_Console.py`. No navigation refactor.
- **Usage logging: write-only in Plan 2.** `log_page_view()` collects events now so data
  accumulates from day one. The analytics *panel* (charts, active-user rollups) is deferred to
  Plan 3, where it lands alongside the `st.navigation()` role-based nav refactor.

Rationale: feedback is only useful if users can submit AND an admin can read it, so both halves
must ship together. Usage data, by contrast, is worthless to chart until weeks of it exist — so
we start the write path early and defer the read path to when there's something to show.

---

## 2. In Scope / Out of Scope

### In scope (Plan 2)

- `src/version.py` — single `APP_VERSION` constant (env-overridable).
- `src/feedback.py` — submit + auto-capture, list/triage helpers, and the `st.popover` widget.
- `src/usage.py` — `log_page_view()` write-only logger with per-session dedup.
- Two new tables in `_init_multiuser_tables`: `feedback`, `usage_events`.
- Feedback inbox tab in `pages/00_Admin_Console.py` (read, status transitions, admin notes).
- Wiring on all 13 interactive pages: `log_page_view(...)` near top, `render_feedback_widget(...)` near foot.
- Tests: capture, inbox, usage logging, back-compat, and a structural guard.

### Out of scope (deferred to Plan 3 unless noted)

- Usage **analytics** panel / charts / active-user rollups.
- `feature_flags` and `audit_log` tables.
- `st.navigation()` role-based navigation refactor.
- Broadcast messages / maintenance mode / "view as another user".
- Editing or migrating any existing table (we only ADD tables; `users.last_seen_at` already
  exists from Plan 1 and is merely *written* here, not altered).

---

## 3. Data Model

Both tables are created in `_init_multiuser_tables(conn)` at `src/database.py:790`, appended to
the existing `conn.executescript(...)` block. `CREATE TABLE IF NOT EXISTS` keeps it idempotent;
the table is always created (so a `MULTI_USER` flag flip needs no migration), but only written
when the flag is on. Foreign keys reference the real `users.user_id` PK.

```sql
CREATE TABLE IF NOT EXISTS feedback (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id      INTEGER NOT NULL REFERENCES users(user_id),
    page         TEXT NOT NULL,              -- e.g. "Trade Analyzer"
    feature_tag  TEXT,                        -- optional sub-area, e.g. a tab name; NULL if page-level
    message      TEXT NOT NULL,
    app_version  TEXT NOT NULL,              -- captured from APP_VERSION at submit time
    data_state   TEXT,                        -- JSON: full get_refresh_log_snapshot(); NULL if unavailable
    status       TEXT NOT NULL DEFAULT 'new', -- new | triaged | resolved
    admin_notes  TEXT,                         -- free-text, admin-only
    created_at   TEXT NOT NULL                 -- datetime.now(UTC).isoformat()
);
CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status);

CREATE TABLE IF NOT EXISTS usage_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(user_id),
    page        TEXT NOT NULL,
    action      TEXT NOT NULL DEFAULT 'view', -- 'view' for Plan 2; richer actions deferred
    session_id  TEXT NOT NULL,                -- uuid4 minted once per browser session
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_user_created ON usage_events(user_id, created_at);
```

Notes:
- SQLite does not enforce FKs unless `PRAGMA foreign_keys=ON`; `get_connection()` does not set it,
  so these `REFERENCES` clauses are documentary/forward-compatible, not enforced constraints. We
  rely on the fact that both writers source `user_id` from `current_user()`, which is always a
  real row. This matches how the rest of the schema treats FKs.
- `status` values are a closed set of three: `new` → `triaged` → `resolved` (transitions may skip
  or reverse; the inbox offers all three). No separate enum table — the app validates.
- `data_state` is the JSON-serialized list returned by `get_refresh_log_snapshot()` (7 keys per
  source). Stored as TEXT via `json.dumps`. NULL only if the snapshot call returns falsy.

---

## 4. Components / Modules

### 4.1 `src/version.py` (new)

```python
"""Single source of truth for the running app version."""
import os

APP_VERSION = os.environ.get("HEATER_APP_VERSION") or "1.0.1"
```

One constant. The env override lets Plan 4 (Railway) inject a git SHA / release tag at deploy
time. Local + test runs get the stable literal. (The literal is `"1.0.1"` to match the project
folder and CLAUDE.md; `pyproject.toml` still reads `1.0.0`, a pre-existing inconsistency this
plan does not try to reconcile.)

### 4.2 `src/feedback.py` (new)

Mirrors the structure of `src/auth.py`: thin DB helpers that import `get_connection` lazily, a
pure-ish submit function, and one Streamlit-rendering widget function.

```python
def submit_feedback(user_id: int, page: str, message: str,
                    feature_tag: str | None = None) -> int:
    """Insert a feedback row, auto-capturing app_version + data_state. Returns new id."""
```
- Captures `APP_VERSION` from `src.version`.
- Captures `data_state` by calling `get_refresh_log_snapshot()` and `json.dumps`-ing it; on any
  failure (returns falsy / raises), stores NULL — never blocks a submit.
- `created_at = datetime.now(UTC).isoformat()`.
- `status` defaults to `'new'` at the DB layer.

```python
def render_feedback_widget(page: str, feature_tag: str | None = None) -> None:
    """Render the 'Send feedback on this' popover at the foot of a page. No-op when
    MULTI_USER is off OR there is no current_user (defense-in-depth)."""
```
- Early-returns when `not multi_user_enabled()` or `current_user() is None`.
- Renders an `st.popover("Send feedback on this")` containing an `st.form` with a single
  `st.text_area` and a submit button.
- On submit with non-empty message: calls `submit_feedback(...)`, shows `st.success(...)`.
- Empty message → `st.warning(...)`, no write.

```python
def list_feedback(status: str | None = None) -> list[dict]:
    """All feedback rows (optionally filtered by status), newest first, joined to
    users for username/team_name display."""

def set_feedback_status(feedback_id: int, status: str) -> None:
    """Transition a feedback row's status (new|triaged|resolved)."""

def set_feedback_notes(feedback_id: int, notes: str) -> None:
    """Set/replace the admin_notes free-text on a feedback row."""
```

### 4.3 `src/usage.py` (new)

```python
def log_page_view(page: str, action: str = "view") -> None:
    """Record one usage_events row for the current user + page this session.
    No-op when MULTI_USER is off OR there is no current_user."""
```
- Early-returns when `not multi_user_enabled()` or `current_user() is None`.
- **Per-session dedup:** maintains `st.session_state["_usage_logged"]` as a `set` of
  `(page, action)` tuples. If `(page, action)` is already in the set, return without writing.
  This collapses Streamlit's many top-to-bottom reruns per page into a single event per
  (session, page, action). The set resets naturally on a new browser session.
- **session_id:** minted once per session — `st.session_state["_usage_session_id"] = uuid4().hex`
  on first call, reused thereafter.
- Writes the `usage_events` row, then bumps `users.last_seen_at = datetime.now(UTC).isoformat()`
  for the current user (the column already exists from Plan 1).

### 4.4 `pages/00_Admin_Console.py` (modify)

Add a feedback inbox. The console currently renders pending/active/revoked user sections
directly. Wrap the existing content and the new inbox in `st.tabs(["Users", "Feedback"])`: the
existing user-management sections move under the "Users" tab unchanged, the inbox is the
"Feedback" tab. The Feedback tab:
- A status filter (`st.selectbox`: All / new / triaged / resolved).
- A list of feedback rows (newest first) showing username · team · page · feature_tag ·
  created_at · message · app_version.
- A `data_state` viewer (collapsed `st.expander` rendering the JSON) per row.
- Per-row status control (set new/triaged/resolved) and an admin-notes text input.
- Already behind `require_admin()` (the file's existing top-of-page guard) — no new gating needed.

---

## 5. Page Integration

All **13 interactive pages** (every `pages/*.py` except `00_Admin_Console.py`):

```
10_Punt_Analyzer · 11_Trade_Analyzer · 12_Trade_Finder · 14_Free_Agents ·
16_Player_Compare · 17_Leaders · 19_Player_Databank · 1_My_Team ·
20_Draft_Simulator · 2_Line-up_Optimizer · 3_Closer_Monitor ·
5_Matchup_Planner · 6_League_Standings
```

Each gets two additions, by analogy to the existing per-page `require_auth()` wiring:

1. **Near the top** (after `inject_custom_css()` / `require_auth()`, before the body renders):
   ```python
   from src.usage import log_page_view
   log_page_view("My Team")          # human-readable page label
   ```
2. **Near the foot** (after the page body, before/with the page-timer footer):
   ```python
   from src.feedback import render_feedback_widget
   render_feedback_widget("My Team")
   ```

The page label string is the canonical human name (e.g. `"Lineup Optimizer"`, `"Trade Analyzer"`)
— stable, used as the `page` value in both tables.

`app.py` (the Draft Tool / splash) is **not** in this set — the 13 in-season pages are the
surfaces league-mates use. (If desired later, `app.py` can adopt the same two calls; out of scope
now.)

---

## 6. Behavior & Edge-Case Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Usage granularity | **Page-view only** (`action="view"`) | YAGNI. The helper accepts an `action` arg so richer events (`ran_optimizer`, `evaluated_trade`) can be added later without a signature change, but Plan 2 only emits views. |
| Usage dedup | **One row per (session, page, action)** | Streamlit reruns a page top-to-bottom on every widget interaction; without dedup a single visit logs dozens of rows. Session-scoped `set` collapses to one. |
| `data_state` payload | **Full `get_refresh_log_snapshot()` JSON** | Maximum debugging signal at submit time; the snapshot is small (one row per source) and the function never raises. |
| Widget control | **`st.popover`** at page foot | Unobtrusive; doesn't push page content down; matches parent spec §3 ("Send feedback on this"). |
| Both helpers when `MULTI_USER` off | **No-op (early return)** | Byte-for-byte v1 behavior; no tables touched, no UI rendered. |
| Both helpers when flag on but no `current_user` | **No-op** | Defense-in-depth — pages are already auth-gated, but the helpers shouldn't write a NULL/0 user_id if reached out of order. |
| Inbox placement | **Tab in existing Admin Console** | No nav refactor (deferred to Plan 3); reuses the existing `require_admin()` gate. |
| Feedback validation | Empty message rejected with `st.warning`; non-empty writes | Avoids blank rows; minimal friction. |

---

## 7. Testing & Structural Guards

New test files under `tests/`:

- **`test_feedback_capture.py`** — `submit_feedback(...)` writes a row whose `app_version`
  equals `APP_VERSION` and whose `data_state` is the JSON of `get_refresh_log_snapshot()`;
  `status` defaults to `'new'`; `created_at` is set. Covers the NULL-`data_state` path when the
  snapshot is unavailable.
- **`test_feedback_inbox.py`** — `list_feedback()` returns rows newest-first and filters by
  status; `set_feedback_status` transitions through new→triaged→resolved; `set_feedback_notes`
  persists notes. AppTest smoke: the Feedback tab renders for an admin and is unreachable for a
  non-admin (the existing `require_admin()` already guards this — assert the tab content given an
  admin session).
- **`test_usage_logging.py`** — `log_page_view("X")` writes exactly one `usage_events` row;
  a second call in the same session for the same page writes **no** additional row (dedup);
  `last_seen_at` is bumped; `session_id` is stable across calls within a session.
- **`test_feedback_usage_backcompat.py`** — with `MULTI_USER` unset, `render_feedback_widget`
  and `log_page_view` are no-ops (no DB writes, no Streamlit calls). Parallels
  `test_auth_backcompat.py`.

New structural-invariant guard (parallels `tests/test_pages_have_auth_guard.py`):

- **`test_pages_have_feedback_and_usage.py`** — every interactive `pages/*.py` (those calling
  `inject_custom_css()`, excluding `00_Admin_Console.py`) imports and calls **both**
  `log_page_view(...)` and `render_feedback_widget(...)`. AST/source check, same shape as the
  auth-guard test. This is the regression lock: a new page added without the two calls fails CI.

All new tests must obey the `tests/conftest.py` network guard (no real outbound sockets) and the
session-scoped DB-schema fixture. They use the per-worker SQLite DB, so writing real
`feedback`/`usage_events` rows is safe and isolated.

---

## 8. Back-Compat / `MULTI_USER` Off

- The two tables are always created by `init_db()` (idempotent), but nothing reads or writes them
  when the flag is off.
- `render_feedback_widget` and `log_page_view` return immediately when `multi_user_enabled()` is
  false — no Streamlit elements, no DB access. The 13 pages render exactly as in v1.
- No existing table, function signature, or page behavior changes for v1 users.
- The Admin Console is already a `require_admin()`-gated page that does nothing in v1 (admin only
  exists under `MULTI_USER`); adding a tab to it is invisible to v1.

---

## 9. File Manifest

| File | Action | Responsibility |
|------|--------|----------------|
| `src/version.py` | create | `APP_VERSION` constant (env-overridable) |
| `src/feedback.py` | create | submit + auto-capture, list/triage helpers, popover widget |
| `src/usage.py` | create | `log_page_view()` write-only logger with per-session dedup |
| `src/database.py` | modify | add `feedback` + `usage_events` to `_init_multiuser_tables` |
| `pages/00_Admin_Console.py` | modify | add Feedback inbox tab |
| `pages/*.py` (13 interactive) | modify | add `log_page_view(...)` + `render_feedback_widget(...)` |
| `tests/test_feedback_capture.py` | create | submit/auto-capture behavior |
| `tests/test_feedback_inbox.py` | create | list/filter/status/notes + admin-gate smoke |
| `tests/test_usage_logging.py` | create | single-write, dedup, last_seen_at, session_id |
| `tests/test_feedback_usage_backcompat.py` | create | MULTI_USER-off no-op |
| `tests/test_pages_have_feedback_and_usage.py` | create | structural guard (every interactive page wires both) |

---

## 10. Deferred to Plan 3 (recorded so it isn't lost)

- Usage analytics panel: active-user counts, per-page view charts, last-seen rollups.
- `feature_flags` + `audit_log` tables and their admin UI.
- `st.navigation()` role-based navigation (admin vs league-mate menus).
- Broadcast/maintenance/"view as" admin tooling.
