# HEATER on Railway — Deployment Runbook

Operator steps to deploy HEATER as a hosted multi-user app for the 12
FourzynBurn leaguemates. Claude wrote the `Dockerfile`, `railway.toml`,
`.dockerignore`, and all app wiring (v2 Plan 4); the steps below are the
Railway-side actions only YOU can perform (account, env vars, volume, OAuth
consent, deploy clicks).

## Architecture in one paragraph

One Railway **service**, one persistent **volume**, **single replica**. A
background thread inside the app (`src/scheduler.py`) is the only process that
writes the SQLite DB — it runs `bootstrap_all_data(force=False)` every 5 minutes
(refreshing only stale sources). All 12 users are read-only consumers. Because
the writer lives in-process and the DB is SQLite, you must NOT scale past one
replica.

## Prerequisites

- A [Railway](https://railway.app) account (Hobby plan ≈ $5/mo base + usage; this
  app's idle footprint lands around $7-15/mo).
- Your Yahoo OAuth app credentials (client id/secret) and a locally-generated
  `data/yahoo_token.json` (run the app once locally and complete Yahoo OAuth to
  produce it).
- This repo pushed to GitHub (Railway deploys from the repo).

## Step 1 — Create the service

1. Railway dashboard → **New Project** → **Deploy from GitHub repo** → pick
   `hicklax13/HEATER_v1.0.0`.
2. Railway detects `railway.toml` and uses the **DOCKERFILE** builder
   automatically. No build config needed.

## Step 2 — Add the persistent volume

1. Service → **Settings** → **Volumes** → **New Volume**.
2. Mount path: **`/app/data`**. (The app reads/writes `data/draft_tool.db`,
   `data/yahoo_token.json`, and `data/logs/` here. `data/seed/` ships in the
   image and is read-only baseline — the volume mount overlays the writable
   files alongside it.)
3. Size: 1 GB is plenty (the DB is tens of MB).

> **Volume + seed interaction:** the image bakes `data/seed/`. Mounting a volume
> at `/app/data` shadows the *directory*, but the seed files were COPYed into the
> image under `/app/data/seed/`. On Railway the volume is empty on first boot, so
> the app re-creates `draft_tool.db` there; the seed files remain available from
> the image layer because the volume only overlays paths that exist on it. If you
> ever see missing-seed errors, copy `data/seed/` onto the volume once via the
> Railway shell.

## Step 3 — Set environment variables

Service → **Variables** → add:

| Variable | Value | Notes |
|----------|-------|-------|
| `MULTI_USER` | `1` | **Required** — turns on the hosted multi-user surface |
| `ADMIN_USERNAME` | _your choice_ | Seeds the admin account (idempotent) |
| `ADMIN_PASSWORD` | _strong secret_ | Seeds the admin account |
| `ADMIN_TEAM_NAME` | `Team Hickey` | Your Yahoo team name |
| `YAHOO_LEAGUE_ID` | _your league id_ | Read by `try_reconnect_yahoo` for headless reconnect |
| `YAHOO_CLIENT_ID` | _client id_ | Yahoo OAuth app credential (fallback when not embedded in the token JSON) |
| `YAHOO_CLIENT_SECRET` | _client secret_ | Yahoo OAuth app credential (fallback when not embedded in the token JSON) |

> Railway injects `PORT` automatically — do NOT set it. The `railway.toml`
> `startCommand` binds `$PORT`.

## Step 4 — First deploy

1. **Deploy**. The image builds (~3-6 min first time).
2. Healthcheck hits `/_stcore/health`; Streamlit answers as soon as the port
   binds (seconds) — it does NOT wait for the 15-20 min first data refresh.
3. Open the service URL. Log in with the `ADMIN_*` credentials.
4. The Home page shows **"Data warming up — first refresh in progress (~15-20
   min)…"** This is expected: the scheduler thread is running its first
   `bootstrap_all_data`. The page populates automatically once the first
   successful refresh lands.

## Step 5 — Seed the Yahoo token

Headless Yahoo reconnect needs `data/yahoo_token.json` on the volume.

1. Open `data/yahoo_token.json` from your LOCAL machine in a text editor; copy
   its full contents.
2. In the hosted app (as admin): **Admin → Controls → Yahoo token**.
3. Paste the JSON into the text area → **Save Yahoo token**. You'll see "Yahoo
   token saved." (The token is written to the volume and never displayed back.)
4. The next scheduler cycle (or a manual **Refresh All Data**) reconnects Yahoo
   headlessly.

## Step 6 — Invite leaguemates

1. Share the service URL.
2. Each leaguemate **registers** → lands in `pending`.
3. You **approve** them and assign their Yahoo team: **Admin → Console → pending
   users**.

## Operations

- **Manual refresh:** Admin sees a **Refresh All Data** button in the sidebar
  (forces a full `bootstrap_all_data(force=True)`). Regular users do not.
- **Freshness:** Home shows "Data last refreshed N min ago." The scheduler
  re-checks every 5 minutes and refreshes only stale sources.
- **Logs:** Railway → service → **Deployments** → **View Logs**. Bootstrap and
  scheduler activity print here; persistent bootstrap log is on the volume at
  `data/logs/bootstrap.log`.
- **Restart:** safe anytime. On restart the warming gate reappears briefly until
  the first refresh row is re-read from the (persisted) DB; if the DB already has
  fresh rows the page renders immediately.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Stuck on "Data warming up" > 30 min | First bootstrap erroring | Check Deploy logs for tracebacks; verify `YAHOO_*` vars; the 3-tier waterfalls fall back to seed data, so most failures still produce a partial first refresh |
| Yahoo data never loads | Token not seeded / expired | Re-do Step 5 with a fresh local `yahoo_token.json` |
| "database is locked" in logs | More than one replica | Service → Settings → ensure **Replicas = 1** (also pinned in `railway.toml`) |
| Login fails for admin | `ADMIN_*` vars unset/typo'd | Verify the three `ADMIN_*` variables; redeploy |
| Healthcheck failing at deploy | App crashed at import | Check logs for the traceback; usually a missing env var or a dep that didn't install |

## Cost note

Hobby plan base + usage-based compute/volume. A single always-on small service
with a 1 GB volume typically runs **$7-15/month**. Railway bills compute by the
second; the app is lightweight at idle (Streamlit + a 5-min scheduler tick).
