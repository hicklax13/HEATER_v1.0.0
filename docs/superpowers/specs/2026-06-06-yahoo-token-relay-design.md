# Design — Yahoo token relay (fix: Yahoo blocks OAuth refresh from Railway's datacenter IP)

> Status: **approved design, pre-implementation**
> Date: 2026-06-06
> Supersedes the diagnosis in `2026-06-06-yahoo-connection-hang-fix.md` (that handoff's
> "hang/throttle/volume" theory was **disproven** during this session — see Root Cause).

## 1. Problem & confirmed root cause

On the live Railway deployment, Yahoo data goes stale ~1 hour after every token paste.

**Root cause (proven this session):** Yahoo's OAuth login endpoint
(`https://api.login.yahoo.com/oauth2/get_token`) **refuses to refresh the access token when
the request originates from Railway's datacenter IP**, returning a misleading
`HTTP 400 INVALID_CONSUMER_KEY: "Client ID does not exist"`. The credentials are valid; the
**same** client ID + refresh token succeed instantly from a residential IP (the owner's mini PC).

### Mechanism
1. A pasted token's **access token** is a Bearer credential valid for ~1 hour. Using it for the
   Yahoo *data* API needs no client ID, so data flows for that hour.
2. After 1 hour the access token expires. Renewing it is the **only** step that uses the client
   ID, via `api.login.yahoo.com`.
3. From the Railway IP, Yahoo rejects that call → `yahoo_oauth` raises `KeyError('access_token')`
   → `YahooFantasyClient.authenticate()` returns `False` → the scheduler degrades to
   `yahoo_client=None` → no Yahoo sync → data goes stale.

### Evidence (decisive)
| Origin | Client ID | Refresh result |
|--------|-----------|----------------|
| Owner's mini PC (residential) | `dj0yJm…PWI2` | **HTTP 200, has_access_token: True** |
| Railway server (datacenter) | `dj0yJm…PWI2` (identical) | **HTTP 400, INVALID_CONSUMER_KEY** |

- The token file's `consumer_key` equals the env `YAHOO_CLIENT_ID` (no mismatch), with no
  whitespace; a browser `User-Agent` did **not** change the rejection (so it is IP-based, not a
  bot-UA heuristic).
- Volume is ~8 refresh attempts/hour — far under Yahoo's ~2000/hour limit — so it is **not**
  rate-limiting. The Railway scheduler runs fine on a ~7-minute cadence (logs prove it is **not**
  frozen); every reconnect simply fails fast at the refresh.

### What this rules out (do not re-pursue)
Not a hang, not a frozen scheduler, not call volume/throttling, not a credential/whitespace
problem, not the Yahoo app being deleted, not a User-Agent filter. **No code change on Railway can
make Yahoo accept the refresh from that IP.**

## 2. Goals / non-goals

**Goals**
- Keep a valid (< ~35 min old) Yahoo access token present on the Railway server at all times, so
  the server never needs to call Yahoo's blocked refresh endpoint.
- Refresh the token from a residential IP (the owner's ASUS Mini PC) and relay it to the server.
- Surface the *real* Yahoo error when a refresh does fail, and make the scheduler's activity
  visible in the live Railway logs (required to verify the fix on the live server).
- Preserve every hard invariant; keep v1 / relay-unconfigured behavior byte-for-byte unchanged.

**Non-goals**
- Eliminating Railway. The data API works fine from the server; only OAuth *refresh* is blocked.
- Rotating the Yahoo app credentials (optional hygiene; does not affect the fix — a new app's
  client ID would behave identically: residential OK, datacenter blocked).
- Reducing scheduler call volume as a fix (volume was never the cause).

## 3. Architecture

```
 MINI PC (residential IP)              GitHub secret gist            RAILWAY (datacenter IP)
 Task Scheduler every 30 min:          (encrypted blob)              scheduler every ~5 min:
   1. refresh Yahoo token  ✅  ──PATCH (PAT)──►  ciphertext  ──GET (raw URL)──►  1. pull + decrypt
   2. persist locally                                                            2. write token to volume
   3. encrypt + upload                                                           3. existing reconnect
                                                                                    SUCCEEDS w/o refresh
                                                                                 4. bootstrap Yahoo sync
```

Three single-purpose components plus a passive drop-box. Each is independently testable.

## 4. Component detail

### 4.1 Relay (mini PC) — `scripts/yahoo_token_relay.py`
Run by Windows Task Scheduler every 30 minutes. Each run:
1. Load `data/yahoo_token.json` (the seed token; owner already has a valid one).
2. **Refresh** via a direct POST to `api.login.yahoo.com/oauth2/get_token`
   (`grant_type=refresh_token`). New helper `src.yahoo_api.refresh_yahoo_token(token_dict) ->
   dict | None` performs and parses this, returning the merged token dict (new `access_token`,
   `token_time`, plus whatever `refresh_token` Yahoo returns — so rotation, if it ever happens, is
   handled). Returns `None` on failure (logs the actual Yahoo error code).
3. Persist the refreshed token locally via the existing atomic `src.yahoo_api._write_token_file`.
4. **Encrypt** the token JSON with `cryptography.Fernet` (key = env `HEATER_RELAY_KEY`).
5. **Upload** the ciphertext to the gist: `PATCH /gists/{HEATER_GIST_ID}` with
   `Authorization: token {HEATER_GIST_PAT}`, updating one file (`heater_yahoo_token.enc`).
6. Log a one-line success/failure (never the token or secrets).

Idempotent and self-contained; safe to run by hand for testing. A non-zero exit on failure lets
Task Scheduler surface problems.

### 4.2 Drop-box — GitHub secret gist
One secret (unlisted) gist, one file `heater_yahoo_token.enc` holding base64 Fernet ciphertext.
Secret-gist raw content is reachable at the **stable** URL
`https://gist.githubusercontent.com/{user}/{gist_id}/raw/heater_yahoo_token.enc`
(no commit SHA → always serves the latest revision), so the server reads it **without a PAT**.

### 4.3 Server reader — `src/token_relay.py` (new) + scheduler wiring
- `pull_relayed_token() -> bool`:
  - If `HEATER_TOKEN_RELAY_URL` or `HEATER_RELAY_KEY` is unset → **no-op, return False** (dormant;
    v1 / local unchanged).
  - GET the raw URL (15 s timeout) → Fernet-decrypt → `json.loads` → validate it carries
    `access_token` + `refresh_token`.
  - Write to `data/yahoo_token.json` via `_write_token_file` **only if** the relayed token's
    `token_time` is newer than the on-disk one (avoids clobbering with stale CDN copies).
  - Never logs token contents; logs only outcome + token age.
- Scheduler (`src/scheduler.py::_refresh_loop`): call `pull_relayed_token()` **before**
  `try_reconnect_yahoo()` each cycle. Because the on-disk token is then < ~35 min old,
  `yfpy`'s `token_is_valid()` passes and **the blocked refresh endpoint is never called.**

## 5. Robustness upgrades (surfaced by the investigation)

1. **Surface the real Yahoo error.** When a refresh fails, log the actual Yahoo error
   (e.g. `INVALID_CONSUMER_KEY`, `invalid_grant`) and an actionable hint
   ("server IP cannot refresh — is the mini-PC relay running?") instead of the bare
   "authentication returned False." Implemented in `refresh_yahoo_token()` (relay path) and a
   clearer message in `try_reconnect_yahoo` / `authenticate` (server path).
2. **Observability for verification.** Root-cause why `src.*` INFO logs are not reaching Railway's
   stdout on the current deploy (the `e7d7150` logging change regressed it; the on-volume
   `bootstrap.log` last wrote Jun 5 while the Jun 6 deploy ran), and fix it so the scheduler's
   reconnect/persist/pull lines appear on the live console. `data/logs/bootstrap.log` (readable via
   Railway Console) remains the backup verification channel.
3. **No error storms when the relay is down.** Log the healthy→stale transition **once**
   (relayed token stale / relay likely down), not every cycle; fall back to cached data
   (already built). Recovery is logged once too.

## 6. Security
- **End-to-end encryption.** Token is Fernet-encrypted on the mini PC before upload; the key
  (`HEATER_RELAY_KEY`) lives only on the mini PC and in Railway Variables — never in GitHub. The
  gist holds only ciphertext.
- **Low blast radius.** The Yahoo token is **Fantasy Sports → Read** only (no email, no account
  changes, no writes).
- **No secret logging.** Token, secret, PAT, and key are never logged or echoed (AST-guarded where
  practical).
- **Manual paste remains** (Admin Controls → Yahoo token) as a fallback if the relay is down long.
- The Client Secret was exposed in a screenshot this session; rotating the Yahoo app is
  *recommended hygiene* but **out of scope** for this fix (does not affect behavior).

## 7. Configuration

| Env var | Where | Purpose |
|---------|-------|---------|
| `HEATER_RELAY_KEY` | mini PC + Railway | Fernet key (encrypt / decrypt). Generated once. |
| `HEATER_GIST_ID` | mini PC | Target gist for PATCH upload. |
| `HEATER_GIST_PAT` | mini PC | GitHub PAT, **`gist` scope only**, for writing the gist. |
| `HEATER_TOKEN_RELAY_URL` | Railway | Stable raw gist URL the server reads (no PAT needed). |

Relay is **enabled on the server** only when both `HEATER_TOKEN_RELAY_URL` and `HEATER_RELAY_KEY`
are set; otherwise `pull_relayed_token()` is a no-op. New dependency: `cryptography` (Fernet) added
to `requirements.txt`.

## 8. Failure modes & degradation
| Failure | Behavior |
|---------|----------|
| Mini PC off / Task Scheduler missed a run | Token ages; within 60 min still valid (30-min cadence tolerates one miss). Beyond that, server serves **cached** data + logs the stale transition once. |
| Relay refresh fails (Yahoo error) | Relay logs the real error, exits non-zero, leaves the last good gist content; server keeps using the last relayed token until it ages out. |
| Gist unreachable from server | `pull_relayed_token()` returns False, logs once; server uses on-disk token. |
| Decrypt fails (key mismatch) | Logged as a config error; on-disk token untouched. |
| Relay unconfigured | Everything dormant; identical to today's v1 behavior. |

## 9. Testing (TDD)
A failing test precedes each unit. New / changed guard tests (exact names defined in the
implementation plan):
- `refresh_yahoo_token()` — parses a 200 refresh, returns merged dict; returns None + logs the real
  error on a 400 `INVALID_CONSUMER_KEY` (mocked HTTP).
- Fernet round-trip — encrypt on "mini PC", decrypt on "server" yields the original token dict.
- `pull_relayed_token()` — no-op when unconfigured (v1 guard); writes only when `token_time` newer;
  never writes token contents to logs; honors the 15 s timeout.
- Scheduler wiring — `_refresh_loop` calls `pull_relayed_token()` before `try_reconnect_yahoo()`
  (AST/behavior check), gated on config.
- Observability — `src.*` INFO reaches a stdout handler at runtime (a real assertion, not just a
  source-text grep like the current `test_src_logs_to_stdout.py`).
- "No storm" — repeated stale cycles log the transition once, not per cycle.
- Back-compat — relay-unconfigured ⇒ zero new DB/network/file activity (connection/HTTP spy).

Reviews before ship: `pr-review-toolkit:silent-failure-hunter`, `pr-review-toolkit:code-reviewer`,
`coderabbit:code-review`.

## 10. Verification (the real bar — not green tests)
Done **only** when, on the **live Railway server** (read via Railway Console / logs):
1. The scheduler pulls + decrypts the relayed token each cycle without error.
2. The server **never** calls Yahoo's refresh endpoint (no `INVALID_CONSUMER_KEY`, no
   "authentication returned False") in steady state.
3. The relay refreshes ~every 30 min; `token_time` advances each cycle; the gist updates.
4. **All 7 Data Freshness rows stay Live/Cached for ≥ 2 hours, across at least one hour boundary
   (≥ 1 token refresh)** — i.e., it survives the window where it previously died.
5. The owner's app shows live matchup/roster/standings; the "Warming up / No matchup data" state is
   gone.

Green local tests are necessary but **not sufficient**.

## 11. Invariants preserved
- **Single Railway replica**; the in-process scheduler stays the **sole SQLite writer** and the
  sole Yahoo **connector** on the server.
- League members stay **read-only**; only the scheduler + admin act.
- **v1 / flag-off unchanged**: with the relay unconfigured, no new behavior, byte-for-byte.
- The mini PC relay only *writes a gist* and *refreshes a token* — it is **not** a second DB writer
  and does **not** touch Railway's volume directly.

## 12. Assumptions
- Yahoo access tokens expire ~3600 s (observed: a 1.43 h token was expired). Refresh from a
  residential IP works (observed: HTTP 200).
- Yahoo does not rotate the refresh token (handoff doc); the relay persists the full returned token
  regardless, so rotation is handled if it occurs.
- The Yahoo **data** API works from the Railway IP with a valid Bearer token (inferred from "fresh
  for ~1 h after each paste"); to be re-confirmed during live verification.
- A secret-gist raw URL without a commit SHA serves the latest revision (standard GitHub behavior;
  small CDN cache acceptable at a 30-min cadence).

## 13. Rollout (owner vs. me)
- **Owner (~5 min, guided):** create a GitHub PAT with `gist` scope; set `HEATER_TOKEN_RELAY_URL`
  + `HEATER_RELAY_KEY` in Railway Variables; confirm each push/deploy.
- **Me:** generate the Fernet key, create the gist, write `refresh_yahoo_token`, the relay script,
  `src/token_relay.py`, the scheduler wiring, the observability fix, the guard tests; install the
  mini-PC Task Scheduler job; drive live verification.
- Railway auto-deploys on push to `master`; a deploy resets sessions (~3–5 min). Every push/deploy
  is confirmed with the owner first.
