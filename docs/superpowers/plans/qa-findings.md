# Per-Team Pre-Launch QA — Findings Log

> Living log of issues found during the per-team QA program. Created Phase 2.
> Severity: **launch_blocker** > **high** > **medium** > **low**.
> Status: **open** → **fixing** → **fixed** (verified) / **wontfix** (design choice).

## How findings are gathered
- **Smoke catalog** (`tests/qa/test_per_team_smoke.py`): every page × every team — crashes / `st.exception` / `st.error`.
- **Deep assertions** (`tests/qa/test_page_*.py`): per-page value plausibility (NaN/None, out-of-range stats, wrong slot counts).
- **silent-failure-hunter**: wrong/empty data that does NOT raise.
- **Browser walkthrough** (Playwright, desktop 1280 + mobile 390): visual/layout/mobile/odd-value.

---

## Phase 2 — Smoke baseline (crash level)

**Run 2026-06-02** (`tests/qa/test_per_team_smoke.py`, serial): **16 passed in 33:23, exit 0.**
- 13 member pages × 12 teams = 156 member renders + 3 admin pages = **159 renders total**.
- **Zero** crashes (`did-not-run`), **zero** `st.exception`, **zero** `st.error` for any (team × page).
- Conclusion: **no launch-blocking crashes** at the page-load level for any of the 12 teams. The deep-assertion suite (value plausibility) and the browser walkthrough are what remain to surface non-crashing / visual issues.

## Phase 1/2 — Deep + ownership result (run #2, timeout-fixed)

**68 passed, 3 failed** (41 min). All 3 failures were adversarially confirmed (via re-render + DB inspection) as **TEST artifacts, not app bugs** — the app is clean on every value the suite checks:

- **OWNERSHIP: all 4 tests PASSED.** No cross-team roster bleed on My Team or Lineup for any of the 12 teams; the signal is live (calibrated 26/26 on a probe); no wrong-team name shown. **Team isolation is verified for all 12 teams** — the launch-critical property the 2026-06-01 bug violated.
- The 3 failures were INFRA-2 (My Team ERA/WHIP regex grabbed roster-table cells; `BUBBA CROSBY` "ERA 34.00" is actually a hitter's cell — confirmed the team has no >20-ERA pitcher), INFRA-3 (Matchup `\d{1,3}%` matched CSS `saturate(180%)`/`-200%`; real win-probs were `30%`), INFRA-4 (Closer render for `My Precious` hit the 90s AppTest cap from stacked 15s Yahoo-timeouts + concurrent load). None is a real implausible value, crash, or wrong-data.

Fixes applied (INFRA-2/3/4 below) and **verified — run #3: the 3 fixed modules pass 15/15**. Because the fixes were isolated (2 test-logic edits + 2 additive timeout bumps that can only flip fail→pass), the 10 untouched modules + 4 ownership tests from run #2 remain valid. **Full per-team gate is GREEN: smoke 16/16, deep + ownership 71/71.** Committed locally as `9226e86`.

## Phase 3 — Browser walkthrough (live app, Railway)

**Login page** (unauthenticated) validated at desktop (1280) and mobile (390): clean Heater theme, Sign-in / Create-account tabs, username + password (with show-password toggle), Sign-in button. No overflow or layout breakage at either width; mobile fully usable. Screenshots: `qa-login-desktop-1280.png`, `qa-login-mobile-390.png`.

**Authenticated walkthrough — DONE (owner chose "you log in, I drive"):** owner logged into the live app as a member (`testuser` → team HUMAN INTELLIGENCE) in their Chrome; assistant drove that authenticated session via the Claude-in-Chrome extension (never handling the password). Walked all 13 member pages at desktop (1280).

- **Team isolation confirmed IN PRODUCTION:** every page showed HUMAN INTELLIGENCE's data (roster 25, 6th in standings, "vs Team Hickey" matchup) — NOT the admin's "Team Hickey." Member nav only (no admin pages). The 2026-06-01 bug class is verified fixed live.
- **10 of 13 pages clean** (My Team, Lineup, Closer, Punt, Standings, Trade Analyzer, Player Compare, Leaders, Player Databank, Draft Simulator). 3 findings (F-VIS-1/2, F-PERF-1 below).
- **Mobile:** login page validated at 390 (this session) + mobile nav verified live 2026-06-01 (`7a99277`). Per-page *authenticated* 390px layout could NOT be auto-captured — the Claude-in-Chrome `resize_window` didn't reflect in screenshots (window resized but render stayed desktop-width). Recommend the owner spot-check on an actual phone; the layout uses shared responsive CSS.
- **Operational note:** live freshness panel shows **Yahoo Offline** → app is serving CACHED data. Owner should re-paste the Yahoo token (Admin Controls) before inviting so members get live data.

## Findings

| # | Severity | Area | Problem | Source | Status |
|---|----------|------|---------|--------|--------|
| INFRA-1 | infra | `tests/qa` deep modules | Module-scoped `results` fixture renders all 12 teams (~6 min on a heavy page); that batch wall-time blew the repo-global `timeout = 120` (pytest-timeout, `thread` method on Windows) → killed the worker mid-render, producing **zero** deep results on run #1. | deep run #1 | **fixed** — `conftest.pytest_collection_modifyitems` gives qa items lacking their own marker an 1800s ceiling (path-filtered to `tests/qa`); per-render AppTest `default_timeout=90` still bounds a true hang. Re-verifying in deep run #2. |
| F1 | **high** | the QA gate itself | Per-team suite asserted plausibility (NaN/range/slot counts) but never **ownership** — a page silently rendering another team's roster (the 2026-06-01 launch-blocker) would pass green because the values are still plausible. | silent-failure-hunter | **FIXED + VERIFIED** — `test_team_ownership.py` (cross-team-bleed arg-max on My Team + Lineup + calibration guard + wrong-team-name check). All 4 ownership tests **PASSED in deep run #2** → team isolation confirmed for all 12 teams. |
| F2 | medium | `src/standings_utils.py` | Module-global caches (`_cached_team_totals` / `_cached_fa_pool`) are not reset between teams in the serial harness. **Production-safe** (the caches hold league-global, viewer-independent data) but a QA-integrity risk: a future per-team value test could validate team 1's data for all 12. | silent-failure-hunter | **fixed** — harness calls `standings_utils.clear_cache()` before each render. Re-verifying in deep run #2. |
| F3 | medium | `src/yahoo_data_service.py::_get_cached` | 15s Yahoo timeout → SQLite fallback → if the DB fallback is *also* empty, returns a bare `pd.DataFrame()` rendered as the generic "No league data loaded" with no distinct "live data timed out / temporarily unavailable" signal. | silent-failure-hunter | **open** (app-side UX fix; triage → Phase 4). Probe **confirmed the fallback serves CORRECT per-team data when the DB is warm** (timeout fired, My Team still showed 26/26 own roster) → the empty case is the rare risk (single replica + warm DB). |
| F4 | low | `src/waiver_wire.py::compute_drop_cost` (~L417) | Reads `st.session_state["_cached_team_totals"]`, a key nothing writes (the cache is a module global in `standings_utils`), so it silently falls back to hardcoded `.250/.320` league averages for the rate-stat drop-cost drag. Same slightly-off value for everyone. | silent-failure-hunter | **open** (app-side; triage → Phase 4). |
| F5 | low (latent) | `src/optimizer/shared_data_layer.py::build_optimizer_context` (~L258) | Falsy `user_team_name` + `roster=None` → `ctx.roster = rosters` (all 12 teams) → league-wide nonsense recommendations with no error. **Unreachable** from the sole production caller (Free Agents passes a resolved team + hard-stops on miss). | silent-failure-hunter | **open (latent)** — defense-in-depth; not a current member-facing bug. |
| F-UI-1 | medium | Beta banner (login + every page) | "Share feedback" link points to `https://forms.gle/PLACEHOLDER` — a dead placeholder. Members are explicitly invited to "share feedback" but the link goes nowhere. (An in-app feedback popover also exists, so the banner link is redundant-but-broken.) | browser (live) | **open** — set a real Google Form URL or drop the banner link; triage → Phase 4. |
| INFRA-2 | infra (test calib) | `test_my_team_rate_stats_in_range` | ERA/WHIP upper-bound arm false-positived: the keyword-anchored regex grabbed generic roster-table cells (`BUBBA CROSBY` "ERA 34.00" is a hitter's cell — the team has NO >20-ERA pitcher per DB query). ERA/WHIP are also genuinely unbounded for small samples (Estévez 162.0 on 0.33 IP is real). Only AVG/OBP≤1 is a true ceiling. | deep run #2 | **fixed** — dropped the ERA/WHIP arm, kept AVG/OBP∈[0,1]; other modules' ERA/WHIP arms are keyword-anchored and didn't false-positive. Re-verifying in run #3. |
| INFRA-3 | infra (test calib) | `test_matchup_win_prob_in_range` | Bare `\d{1,3}%` regex matched CSS inside surviving `<style>` blocks (`saturate(180%)`, `background-position:-200%`) — `_strip_html` removed tags but not `<style>` CONTENT. Real win-probs (`30% chance to win`) are in range. | deep run #2 | **fixed** — `_strip_html` now strips `<style>`/`<script>` block content before scanning (My Team + Matchup). Re-verifying in run #3. |
| INFRA-4 | infra (test env) | Closer Monitor per-render timeout | `My Precious` Closer render hit the 90s AppTest cap (1 of 12 teams). Under the network guard each Yahoo fetch burns ~15s (backoff → 15s executor cap); Closer makes several, and concurrent browser load during the run pushed it over 90s. Not a crash — a `did-not-run` timeout. | deep run #2 | **fixed** — harness per-render `default_timeout` 90→180s; conftest ceiling 1800→3000s. Re-verifying in run #3. Non-blocking prod note: members triggering several live Yahoo calls per page — fine in prod (Yahoo fast, scheduler keeps SQLite warm), but a latency risk if Yahoo degrades. |
| F-VIS-1 | high (visual) | Matchup Planner — category bars | `_build_category_prob_html` built each bar as a multi-line, 4-space-indented triple-quoted f-string joined with `\n`. Streamlit markdown treats the blank line between rows as a block terminator and the next indented row as a CODE BLOCK → the 2nd+ category bars rendered as escaped raw HTML to members. | browser walkthrough | **FIXED (local)** — single-line HTML joined with `""`. Guard: `test_walkthrough_fixes_2026_06_02.py`. Pending redeploy + live re-verify. |
| F-VIS-2 | medium | Free Agents — streaming | Streaming header showed "(vs HUMAN INTELLIGENCE)" — the viewer's OWN team. `get_opponent_context()` was called with no `opponent_name` → global single-user path (`get_current_opponent`) mis-resolves under MULTI_USER. Same class as the 2026-06-01 launch-blocker, in an un-migrated spot. | browser walkthrough | **FIXED (local)** — resolve the member's own opponent (`find_user_opponent` + live-matchup fallback) and pass `opponent_name=`, mirroring Matchup Planner; defensive self-guard. Guard added. Pending redeploy + live re-verify. |
| F-PERF-1 | medium (UX) | Trade Finder | "Re-rank by title odds" toggle defaulted **ON** → a ~68s title-odds Monte-Carlo sim ran on every page load (a 68s spinner reads as broken). Pre-existing. | browser walkthrough | **FIXED (local)** — toggle defaults **OFF** + honest "~1 min" label; users opt into the slow re-rank. Guard added. Pending redeploy. |

---

## Phase 4 — Fixes applied (owner-approved 2026-06-02)

QA-suite artifacts (INFRA-1..4) + the launch-critical ownership gap (F1) + harness cache-reset (F2) are fixed + verified (commits `70adbdc`, `9226e86`). Owner then approved fixing the member-facing items:

- **F-UI-1 (done, local):** removed the dead `forms.gle/PLACEHOLDER` banner link in `app.py`; members use the in-app feedback popover. Guard: `test_no_placeholder_feedback_url_in_app`.
- **F3 (done, local):** added `YahooDataService.data_unavailable_reason()` + `ui_shared.no_league_data_message(reason)`; wired the 4 personalized pages to show a timeout/error-aware message instead of the generic "connect your Yahoo league" string. Tests: `test_no_league_data_message.py` (8).
- **F4 (done, local):** `compute_drop_cost` now reads league averages from `get_all_team_totals()` instead of the never-written `st.session_state["_cached_team_totals"]` key (it had silently used the .250/.320 defaults forever). Tests: `test_drop_cost_dynamic_league_avg.py` (2).
- **F5:** left as-is (latent, unreachable from the sole production caller).

Verification: F3+F4+F-UI-1 unit/guard tests green; full main-suite regression gate (excl. `qa/`) running; then owner-confirmed push → Railway redeploy → live re-verify + the authenticated browser walkthrough.

## Triage notes
_(adversarial confirmation that each finding is real, not a harness artifact)_

- **Smoke baseline (run #1):** 16/16 pass, 159 real per-team renders (as real local QA users), zero crash/exception/error. Conclusion holds adversarially: **no launch-blocking crashes** at page-load level.
- **F1 (the headline):** confirmed real by reading the suite — not one test cross-checked the rendered roster against the team's *expected* roster, so the green status did not certify team isolation. The fix is meaningful (probe = 26/26 own-overlap), not vacuous (Test B calibration guard enforces this).
- **F2:** confirmed production-safe by the hunter (caches are league-global / viewer-independent). Reset added purely for QA integrity.
- **F3:** confirmed benign in the warm-DB case by the calibration probe (timeout fired; page still showed the correct 26/26 roster). Real risk only in the rare empty-DB-during-scheduler-write window.
- **F5:** confirmed unreachable from the sole production caller (FA page passes a resolved team + hard-stops on an unresolved team).
- **INFRA-2:** confirmed a TEST artifact, not an app bug — ERA 162.00 / WHIP 18.00 for Carlos Estévez (0.333 IP) is real small-sample data, displayed exactly as Yahoo/FanGraphs would. **Optional polish (NOT launch-blocking, owner's call):** suppress or cap rate-stat display below a minimum IP so a 1-out blowup doesn't show "162.00".

## Resolved / clean-bill (positive launch signals)
_(paths the silent-failure audit verified correct — not findings)_

- `src/auth.py::resolve_viewer_team_name` returns ONLY the session user's admin-assigned team (or `None`) under `MULTI_USER` — never the global `is_user_team` flag for a logged-in member. Empty/whitespace team → `None`.
- All 8 personalized pages route through the resolver and guard the empty-team case with a visible `st.warning + st.stop()` — a teamless member sees "no team", never another team's data.
- View-as: `enter_view_as` refuses admin targets; `exit_view_as` fully restores the stashed admin; the exit banner renders above every routed page under `st.navigation`.
- `get_team_roster` is strictly per-team (`WHERE lr.team_name = ?`); matchup DB fallback keyed by team; YDS session cache is per-session.
- `viewer_can_write` gates member writes (admins only under `MULTI_USER`); the 4 member refresh/sync buttons are wrapped in it.
