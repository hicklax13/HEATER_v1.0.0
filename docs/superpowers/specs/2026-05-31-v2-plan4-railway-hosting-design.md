# Plan 4 — Railway Hosting + Scheduled-Refresh Data Model

**Date:** 2026-05-31
**Status:** Approved design (pre-implementation)
**Plan:** Final sub-project of the HEATER v2 multi-user rollout (Plans 1-3 shipped via PRs #120/#121/#122 + #123).
**Predecessor doc:** `docs/superpowers/specs/2026-05-28-v2-leaguemate-multiuser-design.md` (master v2 design — this doc supersedes its Section 5 "Railway hosting" and Section 8 "what remains" for Plan 4 specifics).

## 1. Goal

Deploy the Streamlit app to Railway so all 12 FourzynBurn leaguemates can use it as a hosted, multi-user app (`MULTI_USER=1`). A single server-side scheduled job keeps the data fresh; user sessions become read-only consumers of the shared pool. Target cost ~$7-15/mo (one always-on service + one volume).

**Scope (agreed): "Deploy-ready + runbook."** Claude writes everything — Dockerfile, Railway config, the scheduler/bootstrap wiring, the Yahoo-token admin control, all tests, and a step-by-step deployment runbook — and verifies the image builds and serves locally with `MULTI_USER=1`. The user performs the Railway-side actions (account, env vars, volume, OAuth consent, deploy clicks) by following the runbook, because Claude cannot create a Railway account or complete interactive Yahoo OAuth.

## 2. Chosen Architecture (Approach A)

In-process scheduler thread; SQLite stays; one Railway service + one volume; no Postgres, no separate cron service.

- **One writer, many readers.** A single background thread (`src/scheduler.py`, already built) runs `bootstrap_all_data(force=False)` on a loop. The 12 user sessions only *read* the pool. SQLite WAL already permits concurrent reads during a write, so this is exactly the workload WAL handles well.
- **No Postgres.** Deferred (predecessor doc Section 10). SQLite on a persistent volume is sufficient for 12 users + one writer.
- **No separate cron service.** Railway attaches a persistent volume to exactly **one** service. A separate cron service therefore could not share the SQLite file. The in-process scheduler sidesteps this entirely.

## 3. Findings That Change the Design vs. the Predecessor Doc

Two current-state facts (verified by reading the shipped Plan 1-3 code) revise the original Section 5 sketch:

1. **There is no auth cookie.** `src/auth.py` keeps identity only in `st.session_state["auth_user"]` (auth.py:251) — there is no signing key and no persistent login cookie. Therefore **an "auth cookie signing key" is NOT a required secret.** The tradeoff: a server restart logs everyone out and they re-login. For 12 leaguemates this is acceptable, so **cookie-based persistent login is an explicit non-goal** (YAGNI).
2. **The headless Yahoo reconnect already exists and the token already lives on the volume path.** `try_reconnect_yahoo()` (yahoo_api.py:2099) reads `YAHOO_LEAGUE_ID` from env plus the token from `_AUTH_DIR / "yahoo_token.json"`, where `_AUTH_DIR = Path(__file__).parent.parent / "data"` (yahoo_api.py:133). Because the Railway volume mounts at `/app/data`, the token sits on the volume automatically. The boot-time reconnect needs **zero new plumbing** — only a way to get the token file onto the volume the first time (Section 7).

## 4. Container — `Dockerfile`

- Base `python:3.12-slim`. (CI already tests on 3.12; Python 3.14 still lacks some transitive wheels — the exact reason `requirements.txt` installs `yfpy`/`streamlit-oauth` with `--no-deps`.)
- `WORKDIR /app`.
- Copy `requirements.txt`, then:
  - `pip install --no-cache-dir -r requirements.txt`
  - `pip install --no-cache-dir --no-deps "yfpy>=17.0" "streamlit-oauth>=0.1.14"` (PR #125 pinned their transitive deps in `requirements.txt`, so `--no-deps` is now safe).
- Copy the app source.
- `EXPOSE 8501`.
- Start command (shell form, so Railway's injected `$PORT` expands; falls back to 8501 locally):
  `streamlit run app.py --server.port "${PORT:-8501}" --server.address 0.0.0.0 --server.headless true`
- XSRF protection stays on (it is structurally guarded by `test_streamlit_security_settings.py`); the existing `.streamlit/config.toml` is used as-is.
- System packages: rely on manylinux wheels (pandas/numpy/scipy/bcrypt/lxml all ship them for 3.12-slim). Add `build-essential` only if a wheel turns out to be missing at build time — decided empirically during local build verification, not pre-emptively.

## 5. Railway Config — `railway.toml` + Volume

- One service, Dockerfile build, **single replica** (see invariant below).
- `railway.toml`:
  - `[build] builder = "DOCKERFILE"`, `dockerfilePath = "Dockerfile"`
  - `[deploy] startCommand = "streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true"`
  - `[deploy] healthcheckPath = "/_stcore/health"` (Streamlit's built-in health endpoint)
  - `[deploy] healthcheckTimeout = 300`, `restartPolicyType = "ON_FAILURE"`, `restartPolicyMaxRetries = 3`
- **Port arg — two intentional forms, not a contradiction.** The Dockerfile `CMD` (§4) uses `"${PORT:-8501}"` so a bare local `docker run` (no `$PORT`) still binds 8501. The `railway.toml` `startCommand` uses plain `$PORT` because Railway always injects it, and on Railway the `startCommand` *overrides* the Dockerfile `CMD`. So: local build → Dockerfile CMD with 8501 fallback; Railway → startCommand with injected `$PORT`. Keep both.
- **Volume** mounted at `/app/data` (created in the Railway dashboard; documented in the runbook). Persists `draft_tool.db` (+ `-wal`/`-shm`), `yahoo_token.json`, and `logs/bootstrap.log`.
- **Healthcheck is decoupled from bootstrap.** `/_stcore/health` returns OK the moment Streamlit binds the port — within seconds. The 15-20 min first bootstrap runs in the background thread and never sits on the healthcheck path. (Same lesson as PR #118's OAuth decoupling: never put a slow data op on the "is the app up?" critical path.) `healthcheckTimeout` only needs to cover cold image startup, not bootstrap.

### Critical invariant: single replica only

The in-process scheduler + SQLite assume exactly one writer process. **Railway must run this service at `numReplicas = 1` with no horizontal autoscaling.** Two replicas would mean two scheduler threads writing the same SQLite file on the shared volume → WAL write contention and duplicated refresh work. This is a hard constraint, enforced in config and called out in the runbook.

## 6. Secrets / Env Vars

| Var | Purpose |
|-----|---------|
| `MULTI_USER=1` | Enables the entire v2 surface (auth, admin, scheduler wiring). |
| `ADMIN_USERNAME` / `ADMIN_PASSWORD` / `ADMIN_TEAM_NAME` | Seeds the admin idempotently via `ensure_bootstrap_admin()`. |
| `YAHOO_CLIENT_ID` / `YAHOO_CLIENT_SECRET` / `YAHOO_LEAGUE_ID` | Headless Yahoo reconnect (`try_reconnect_yahoo()`). |
| `HEATER_APP_VERSION` | Optional — stamps feedback (`src/version.py`). |
| `$PORT` | Injected by Railway automatically; not set by hand. |

**Not required:** an auth cookie signing key (see Finding #1).

## 7. Yahoo Token Bootstrap (the one unavoidable manual step)

Yahoo OAuth requires a one-time interactive `oob` consent that cannot run headlessly inside a container. Chosen flow (recommended option, approved):

1. The admin runs the existing OAuth flow **locally once**, producing `data/yahoo_token.json` (contains the refresh token + consumer key/secret).
2. The admin opens a new **"Yahoo Token" control in `pages/_admin_controls.py`**, pastes the JSON contents, and clicks save.
3. The handler validates that the text parses as JSON and carries the expected keys (refresh token + consumer credentials), writes it to `_AUTH_DIR / "yahoo_token.json"` on the volume, and **audit-logs the action** (`log_action(admin_id, "yahoo_token_update")`) — the action only, never the token contents. The pasted text is never echoed back.
4. On the next scheduler tick / reconnect, `try_reconnect_yahoo()` picks it up. yfpy refreshes the access token from the refresh token thereafter and persists the refreshed dict back to the same path on the volume.

Rejected alternative: direct volume upload (clumsy for a non-CLI user; the paste-UI is self-service and audit-logged).

## 8. Refresh Inversion + Per-Session Bootstrap Suppression + Cold Start

This is the behavioral core of Plan 4.

- **Today (v1 / flag-off):** every new session runs `render_splash_screen()` → `bootstrap_all_data(force=False)` (TTL-respecting, per PR #60). Correct for one user; wrong for twelve, where every new session would try to write the shared DB.
- **Flag-on:** exactly **one writer.** `start_background_refresh()` (src/scheduler.py) is started once per *process* from `app.py::main()`. It loops `bootstrap_all_data(force=False)` every 300s; the per-source TTLs gate the real work, so a tick is cheap when data is already fresh. Sessions become read-only consumers.
- **Cold start sequence:** container boots → Streamlit binds `$PORT` (healthcheck green) → `main()` starts the scheduler thread → the first ~15-20 min refresh runs in the background → the Home page shows **"Data warming up — first refresh in progress (~15-20 min); this page populates automatically"** until the first successful refresh row appears in `get_refresh_log_snapshot()` (the §9 authoritative signal), then renders normally. Boot never blocks on bootstrap.

## 9. MULTI_USER-On Wiring (the code deltas)

All deltas are additive and gated on `multi_user_enabled()`; flag-off remains v1 byte-for-byte.

- **`app.py::main()`** — in the flag-on branch, before `st.navigation(...).run()`, call `start_background_refresh()`. It is idempotent (guarded by `_scheduler_lock` + `_scheduler_running` in scheduler.py), so calling it on every session re-run starts the thread only once per process. No extra guard needed.
- **`app.py::render_single_user_app()` (the Home page under nav)** — branch on the flag:
  - flag-off: unchanged — `render_splash_screen()` (per-session, TTL-respecting bootstrap).
  - flag-on: render a lightweight Home that shows the freshness badge / "warming up" state and does **not** call `render_splash_screen()` (no per-session bootstrap). **"Warming up" has a single authoritative signal:** `get_refresh_log_snapshot()` contains no successful refresh row yet (the same snapshot the freshness badge reads). Reusing one data source for both the badge and the warming-up gate means they can never disagree.
- **"Refresh All Data" sidebar button** — under flag-on, shown to **admins only** (`current_user().get("is_admin")`), hidden for non-admins; the admin click still does `bootstrap_all_data(force=True)`. flag-off: unchanged (always visible, v1).
- **Freshness badge** — reuse `get_refresh_log_snapshot()` to compute the most recent `last_refresh` and show "Data last refreshed N min ago" on the Home header / sidebar.

## 10. Testing

New guard tests (all MULTI_USER-gated where behavioral; structural where about files):

| Test file | Guards |
|-----------|--------|
| `tests/test_plan4_dockerfile_railway.py` | `Dockerfile` exists, uses `python:3.12-slim`, performs the `--no-deps yfpy/streamlit-oauth` install, and binds `$PORT`. `railway.toml` exists with `builder=DOCKERFILE`, `healthcheckPath=/_stcore/health`, and a single-replica deploy config. |
| `tests/test_plan4_scheduler_wiring.py` | `app.py::main()` calls `start_background_refresh()` in the flag-on path and **not** in the flag-off path (AST/source check). `start_background_refresh()` is idempotent — two calls start exactly one thread. |
| `tests/test_plan4_bootstrap_suppression.py` | Under flag-on, the Home path does not run a per-session `bootstrap_all_data(force=...)`; under flag-off it still does (v1 byte-for-byte). AST/source structure + an AppTest smoke. |
| `tests/test_plan4_yahoo_token_paste.py` | The admin token-paste control writes valid JSON to `_AUTH_DIR/yahoo_token.json` (monkeypatched to tmp), rejects invalid JSON, audit-logs `yahoo_token_update` (action only, no token contents), and is `require_admin()`-gated. |
| `tests/test_plan4_backcompat.py` | flag-off: `main()` starts no scheduler thread; Home bootstraps per-session; the refresh button logic is unchanged. Connection-spy style, mirroring `test_admin_backcompat.py`. |

Each test also earns a one-line row in the CLAUDE.md structural-invariants table, per project convention.

**Local verification (part of "Deploy-ready"):** `docker build -t heater .` succeeds; `docker run -e MULTI_USER=1 -e ADMIN_USERNAME=… -e ADMIN_PASSWORD=… -p 8501:8501 heater` serves; `GET /_stcore/health` returns OK and the login page renders. Reported with evidence.

## 11. Deployment Runbook — `docs/deployment/railway-runbook.md`

Step-by-step the **user** follows (written during implementation):

1. Create a Railway project; connect the GitHub repo (Dockerfile build).
2. Add a volume mounted at `/app/data`.
3. Set the env vars from Section 6. Confirm the service is pinned to **one replica**.
4. Deploy; watch the build; confirm the healthcheck goes green within seconds.
5. Run the Yahoo OAuth flow locally once to produce `yahoo_token.json`.
6. Sign in as admin → Admin Controls → **Yahoo Token** → paste the JSON → save.
7. Verify: the freshness badge advances, `try_reconnect_yahoo()` connects, the first refresh completes (watch the "warming up" state clear).
8. Invite the 11 leaguemates; approve each via the existing admin approval flow and assign their Yahoo team.

## 12. Non-Goals

- **Persistent login cookie** — session-state only; a restart means re-login (Finding #1).
- **Postgres migration** — deferred (predecessor Section 10); SQLite + volume is sufficient.
- **Separate cron service** — Approach A's in-process scheduler replaces it.
- **Horizontal autoscaling / multiple replicas** — forbidden by the single-writer invariant (Section 5).

## 13. Files Touched (preview for the implementation plan)

- **New:** `Dockerfile`, `railway.toml`, `.dockerignore`, `docs/deployment/railway-runbook.md`, the 5 `tests/test_plan4_*.py` files.
- **Modified:** `app.py` (`main()` scheduler start; `render_single_user_app()` flag-on Home + admin-only refresh button + freshness badge), `pages/_admin_controls.py` (Yahoo Token paste section), `CLAUDE.md` (structural-invariant rows + a Plan 4 deployment note).
- **Reused as-is:** `src/scheduler.py`, `src/yahoo_api.py::try_reconnect_yahoo`, `src/audit.py::log_action`, `src/database.py::get_refresh_log_snapshot`.
