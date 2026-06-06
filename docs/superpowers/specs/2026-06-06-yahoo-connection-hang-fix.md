# Handoff prompt — Fix the live Yahoo "connection hang → stale data" bug (HEATER on Railway)

> Paste this whole file as the opening prompt of a fresh session. It is self-contained.
> Created 2026-06-06 after a long live-debugging session that **diagnosed** the cause but
> paused before implementing the fix (late night; avoid a rushed 3am deploy).

---

## TASK (one line)
On the live Railway deployment, Yahoo data goes stale a few hours after every token paste because the **server's Yahoo API calls HANG** (15s timeouts), which freezes the no-timeout background-refresh scheduler. Confirm the root cause from the now-visible Railway logs, then implement a robust fix: make the scheduler resilient to hangs **and** cut the server's Yahoo call volume so Yahoo stops throttling it.

## USE THESE SKILLS, IN THIS ORDER (this is a requirement, not a suggestion)
1. **superpowers:systematic-debugging** — Do NOT fix blind. The prior session shipped two guess-fixes that didn't hold *because production logs were invisible*. They're visible now (`e7d7150`: `src.*` → stdout). FIRST, have the owner paste fresh Railway logs and CONFIRM the failure mode from the **scheduler's own log lines** before changing any code. Complete Phase 1 (root cause) before proposing fixes.
2. **superpowers:writing-plans** — turn the confirmed diagnosis into a step-by-step plan with checkpoints.
3. **superpowers:test-driven-development** — failing test → watch it fail → minimal fix → green, for every change.
4. **pr-review-toolkit:silent-failure-hunter** then **pr-review-toolkit:code-reviewer** then **coderabbit:code-review** — review every fix. This entire saga WAS a silent failure (a debug-swallowed persist + file-only logging); be paranoid about new ones.
5. **superpowers:verification-before-completion** — the fix is verified ONLY when the LIVE Railway logs show the scheduler refreshing without hanging and data staying Live through ≥2 hours incl. an hourly token refresh. Green local tests are necessary but NOT sufficient.
6. **claude-md-management:revise-claude-md** — lock the new guard tests into CLAUDE.md's structural-invariant table.
- Optional: **feature-dev:code-explorer** to trace the Yahoo data flow; the **Explore** agent for broad searches; **superpowers:subagent-driven-development** or **executing-plans** to execute; the **verify** / **run** skills to exercise the running app.

## CONFIRMED FACTS (2026-06-06 session)
- The Yahoo **token is valid**. The exact token reconnects **instantly** (sub-second, `reconnect: SUCCESS`) from the local dev machine. The **refresh_token is stable** — Yahoo does NOT rotate it (identical string across every paste).
- On the **server**, that same token's Yahoo calls **HANG ~15s and time out**. Confirmed Railway log line:
  `yahoo_data_service._get_cached: Yahoo fetch for 'schedule' timed out after 15s — falling back to SQLite cache. If this persists, the OAuth token may have expired.`
- Therefore it is a **server-side connection hang, NOT a token-validity problem.** Same token, two IPs, opposite results.
- Symptom pattern: Data Freshness rows go Cached → Stale (only the 24h-TTL Settings/Schedule stay "Live"); header reads "Yahoo: Warming up"; "This Week: No matchup data available." Onset ~1–3 hours after each fresh paste.
- `src/scheduler.py::_refresh_loop` reconnects **every 5 minutes** (`_CHECK_INTERVAL_SECONDS=300`) via `try_reconnect_yahoo()` → `bootstrap_all_data(force=False)`. **`try_reconnect_yahoo()` has NO timeout**, so a hanging Yahoo call blocks the loop with no error — the refresher silently freezes.
- The session-side `_get_cached` DOES have a 15s `ThreadPoolExecutor` timeout (the T1.21 work) — that is the line we saw. But sessions hitting Yahoo at all under MULTI_USER is itself suspect (see Part B).

## LEADING HYPOTHESIS — confirm from the logs before fixing
Yahoo is **rate-limiting/throttling the server's egress IP** due to call volume: reconnect+refresh every 5 min + per-page-view session fetches + 480-row free-agent pagination. A throttled Yahoo connection **hangs** rather than returning HTTP 429. With no scheduler timeout, the hang freezes the refresher → stale data.
**Confirm:** read the now-visible scheduler logs. Look for reconnect/refresh firing every ~5 min, hangs/timeouts, the loop stuck mid-cycle, or `refresh_access_token` storms. The logs will distinguish "throttle/hang" from "token rejected" from "scheduler not running."

## THE FIX (two parts)
### Part A — Scheduler resilience (a hang must not freeze the refresher)
Wrap the scheduler's Yahoo reconnect (and ideally each Yahoo bootstrap phase) in a **timeout**. On timeout: log WARNING, skip Yahoo this cycle, retry next cycle. Model on `yahoo_data_service._get_cached`'s 15s `ThreadPoolExecutor` pattern.
**Concurrency gotcha:** a `with ThreadPoolExecutor()` block waits for the worker on exit, so it would itself hang on a stuck thread. Use a non-blocking pattern (persistent executor or per-attempt daemon thread joined with a timeout) so the hung thread leaks harmlessly rather than blocking the loop.

### Part B — Cut the server's Yahoo call volume (so it stops getting throttled)
- **Scheduler cadence:** stop reconnecting+refreshing every 5 minutes.
  **CRITICAL yfpy gotcha:** yfpy builds `OAuth2(..., store_file=False)` and refreshes the access token only inside `_authenticate()` (at client construction), **not per request** — see `.venv/Lib/site-packages/yfpy/query.py::_authenticate` (~lines 291–343). A naive "long-lived client" would let the access token expire after ~1h and start 401-ing. So design a refresh cadence matched to the ~1h token lifetime (reconnect/refresh roughly hourly, reuse the SQLite cache otherwise), NOT every 5 min. `persist_current_token()` (already shipped) keeps the on-disk token fresh so a reconnect only actually refreshes when truly expired.
- **Sessions under MULTI_USER must read the SQLite cache ONLY** — never trigger live Yahoo fetches (the scheduler is the sole connector; see CLAUDE.md). Investigate why `_get_cached` made a live `schedule` fetch from a `yds-fetch` thread (a page render). Every avoided session fetch is one fewer call against the throttle.
- Consider raising the short freshness TTLs and/or the scheduler interval so the data layer doesn't expect minute-fresh data when a single in-process scheduler is the only writer.

## ALREADY DONE — on master, do NOT redo
- `persist_current_token()` + module fn `_write_token_file()` — atomic token persistence reading the LIVE yfpy `oauth` object; refuses incomplete tokens; WARNING on failure. (`95912d4`)
- `connection_status()` honesty — returns "server"/"Live (via server)" only when a CORE real-time source is fresh (not the 24h Settings/Schedule). (`95912d4`)
- `src.*` logs now emit to **stdout** so Railway's console shows scheduler/Yahoo diagnostics (was file-only on the volume → invisible). (`e7d7150`) — **this is why you can finally SEE the cause.**
- Standings emoji-team fix (`b7f0567`), pre-launch cleanup F5 + banner cosmetic (`06ffb1d`).
- A fresh valid token is currently pasted on the server (Admin Controls audit id 14, 2026-06-06 ~02:39 UTC).

## KEY CODE LOCATIONS
- `src/scheduler.py` — `_refresh_loop`, `_CHECK_INTERVAL_SECONDS=300`, the un-timed `try_reconnect_yahoo()` call (Part A target).
- `src/yahoo_api.py` — `try_reconnect_yahoo()`, `authenticate()` (~388–507), `persist_current_token()`/`_extract_live_token()`, `_write_token_file()`, `refresh_token()`.
- `src/yahoo_data_service.py` — `_get_cached` (the 15s timeout pattern to model Part A on), `connection_status()`, `get_data_freshness()`, `_age_freshness_label`, the per-source TTLs.
- `src/data_bootstrap.py` — `bootstrap_all_data`, `_refresh_yahoo_aux`, the stdout logging setup at top (`e7d7150`).
- `.venv/Lib/site-packages/yfpy/query.py::_authenticate` — `store_file=False` + refresh-only-at-construction (the Part B gotcha).

## HARD INVARIANTS / CONSTRAINTS
- **Single Railway replica** (hard invariant: in-process scheduler + SQLite). Do not add a second writer.
- **MULTI_USER:** the scheduler is the SOLE SQLite writer AND the SOLE Yahoo connector; members are read-only.
- Keep the full suite green (~4869 pass) + the pre-push structural suite + CI; add guard tests for the fix.
- **Confirm every push/deploy with the owner.** Railway auto-deploys on push to master; a redeploy resets sessions (~3–5 min).
- **Never log or echo the Yahoo token secret.**
- Do not change v1 (flag-off / single-user) behavior.

## VERIFICATION CRITERIA (verification-before-completion)
Done ONLY when, on the live server (read via Railway logs, now visible):
1. The scheduler reconnects + refreshes **without hanging** (timeout catches any hang).
2. Refreshes happen ~hourly, not a 5-min storm.
3. The token **persists** across refreshes (`Persisted Yahoo token …` with a fresh `token_time` each hour).
4. All 7 Data Freshness rows stay **Live/Cached** through **≥2 hours including at least one hourly token refresh** — i.e., it survives the window where it previously died.
5. No `timed out` / `auto-reconnect failed` WARNINGs in steady state.

## OWNER INTERACTION (hand-holding)
Owner = CLI novice. Explain like to a 16-year-old, plain language, copy-paste boxes, confirm pushes/deploys per action. The owner pastes the Yahoo token via Admin Console → Admin Controls → Yahoo token. To prep a fresh token: locally run `try_reconnect_yahoo()` with `YAHOO_LEAGUE_ID=109662` (it refreshes + persists `data/yahoo_token.json` via the new code), then copy that file to the clipboard — **never echo the token**. League id 109662, game_key 469 (MLB 2026). After the fix deploys + a fresh paste, watch the live logs against the criteria above, THEN proceed to onboarding: revoke the leftover `testuser` → invite the 12 leaguemates → approve/assign each Yahoo team in Admin Console.

## RESUME-CHECK FIRST
Confirm `git status` clean and `origin/master` tip is `e7d7150` (the logging fix). Read `CLAUDE.md` + the `live-onboarding-state` memory. Then start at systematic-debugging Phase 1 (get fresh Railway logs).
