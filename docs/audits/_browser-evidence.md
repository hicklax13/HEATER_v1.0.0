# Phase A — Browser sweep evidence (2026-06-07)

Live read-only sweep of https://heaterv100-production.up.railway.app — admin session (Team Hickey),
Windows Chrome. **Read-only:** only analysis controls exercised (tabs, Simulate Season); never clicked
Sync/Refresh/add-drop/admin-write. Findings prefixed **BR-** (browser). Severities pre-verification.

## Per-surface status
| Surface | L1 works | Notes |
|---|---|---|
| Home / Draft Tool | ⚠️ | Renders, but shows "Yahoo Not Connected" + "PLAYER POOL Loading..." despite "Data last refreshed 6 min ago" + live data elsewhere (BR-5, cosmetic; handoff task #13). Connect-Yahoo expander, risk slider, engine-mode, Authorize-with-Yahoo all present (inspected only). |
| My Team | ✅ | Roster 27 (13H/14P); Today's Actions, hot/cold streaks, category-gap panel, roster table (2026 Live + Projected/2025/24/23 tabs), lineup validation "1 issue". Data sane. BR-2 (losing-cats banner). |
| Matchup Planner | ✅ | Win 29% / Tie 23% / Loss 48% (=100%); cats ordered K 99%→WHIP 14%; inverse cats "lower is better"; **Gaussian copula, 10k sims**; loaded 0.43s. BR-3 (label vs My Team). |
| League Standings | ⚠️ | Current Standings + Category Standings render (12 teams, no ghost; Team Hickey 9th — consistent w/ My Team). Data freshness = **"Yahoo: Live (via server)"** (relay working). **Playoff Odds tab CRASHES → BR-4.** |
| Closer Monitor | _pending_ | |
| Lineup Optimizer | _pending_ | |
| Punt Analyzer | _pending_ | |
| Trade Analyzer | _pending_ | |
| Trade Finder | _pending_ | |
| Free Agents | _pending_ | |
| Player Compare | _pending_ | |
| Leaders | _pending_ | |
| Player Databank | _pending_ | |
| Draft Simulator | _pending_ | |
| Admin (Console/Analytics/Controls) | _pending_ | inspect only |

## Browser findings
- **BR-4 (HIGH — live functional crash).** Playoff Odds tab throws `KeyError: 'accent'` to screen after "Simulate Season". `pages/6_League_Standings.py:528` — `else T.get("primary", T["accent"])`. The `.get` default `T["accent"]` is evaluated EAGERLY (Python evals args before the call), so it raises whenever THEME lacks `"accent"` (it does — theme uses short keys per CLAUDE.md). Deterministic crash on a flagship feature; raw traceback exposed to users. The playoff-odds value itself computes fine (card shows 27%); only the tab-detail render dies. **Fix:** `T.get("primary", T.get("accent", T["ac"]))` or use the correct existing key. Not a Known Design Choice.
- **BR-1 (Medium — UX/auth).** Auth is per-`st.session_state` (not cookie/token-backed) → hard-navigating to a page URL, refreshing, or opening a bookmarked deep link drops to the login screen even when logged in elsewhere. Every user who bookmarks `/My_Team` or refreshes hits re-login. (Workaround for the sweep: navigate via in-app sidebar clicks only.) Verify intent; likely a real friction for 12 users.
- **BR-2 (Medium? — consistency).** My Team red banner "LOSING CATEGORIES: R · SB" contradicts (a) its own per-category WEEK-11 detail and (b) League Standings "trailing 3-7-2 in categories" — both of which show ~7 losing cats (R, SB, AVG, OBP, L, SV, WHIP). Either the banner is mislabeled (perhaps a "flippable losing cats" subset) or it under-counts. Needs code/semantics check (`pages/1_My_Team.py`).
- **BR-3 (Low/Medium — label clarity).** "Week 11 vs HUMAN INTELLIGENCE" shows week-to-date ACTUALS on My Team (R 29, K 61, RBI 28) but a 7-day forward PROJECTION on Matchup Planner (R 13.9, K 30.5, RBI 13.4) under the same label, no on-screen cue. Reads as contradictory. Likely different-metric-same-label (not a math bug); add disambiguation.
- **BR-5 (Low — cosmetic; handoff #13).** Draft/Home shows "Yahoo Not Connected" + "PLAYER POOL Loading..." under MULTI_USER even though data is live via the scheduler. Wording/state misleading on the draft-home surface.

## Cross-page consistency confirmed GOOD
- Standings position (Team Hickey 9th, 3-6-1) consistent: My Team ↔ League Standings.
- Week-11 category win/loss tally consistent: My Team per-cat detail (3 win / 2 tie / 7 lose) == League Standings "3-7-2".
- 12 teams, no ghost 13th (Bug-D clean on the standings H2H table).
- Yahoo relay live in production ("Live (via server)", all core sources LIVE).

## Checkpoint — surfaces 5-7 (Closer / Lineup / Free Agents)
- **Closer Monitor** ⚠️ renders 30 teams but confirms **DB-C2** vividly: prints the false caption "populated from the FanGraphs depth chart data loaded at app launch"; every team "Setup: —" (no committee); confidence = preseason-projected-SV ÷ ~33, contradicting 2026 actuals — HOU Josh Hader 67% conf w/ "SV: 22" but **1 actual SV**; LAD Edwin Díaz 51% w/ 10.50 actual ERA; ARI Sewald 13% despite 15 actual SV. Real-world output materially misleading (strengthens DB-C2; advisory page so still Medium). Iglesias 13 SV/0.87 ERA matches My Team ✓.
- **Lineup Optimizer** ✅ Optimize works, full lineup, no NaN/crash. LEAVE EMPTY logic correct (1 SP DCV≤0 + 3 P no-eligible; 6 SPs benched DCV 0.00 — correct for a daily view). Muncy = LAD (DNA fix holds) ✓. Yahoo-mismatch banner works. **BR-6 (confirms MS-E1):** win prob 8.5% here vs 29% on Matchup Planner — same matchup, ~3.4× gap. **BR-7 (candidate Low/Med):** post-Optimize "Today" IP budget flips to 16.5/20 forfeit WARNING (down from 27.2) — post-LP IP counts only today's pitchers, so a daily lineup triggers a weekly-forfeit warning (likely misleading). Latency: page 28s, Optimize 33s (task #10).
- **Free Agents** ✅ renders (14.6s). **FA-C2 confirmed:** L14 columns blank for all 7,954 FAs. **BR-8 (candidate Med):** "No add/drop recommendations found" for a 9th-place losing team while the FA list shows large IMPACTs (Kirk +19.76) and streaming produces recs — recommender yields nothing. **BR-9 (candidate Low):** ownership Heat Index all zeros / HEAT column blank (percent_owned not feeding heat). Streaming recs are real FAs ✓. Minor: some ECR inflated (Correa 1521).
- **MS-E1 confirmed live** (BR-6); **DB-C2 confirmed live**; **FA-C2 confirmed live**.

## Checkpoint — surfaces 8-12 (Trade Analyzer / Trade Finder / Player Databank / Leaders / Admin Console)
- **Trade Analyzer** ✅ builder renders (17.4s); RECENT TRANSACTIONS shows live Yahoo data; give/receive multiselects + Analyze present. Freshness panel shows **Matchup: STALE** here (was LIVE on Standings min earlier) → **confirms DB-C3 / task #12** live (matchup TTL flicker). Full multi-trade build NOT exercised (trade engine is the most test-guarded component; B1 found 0 Critical/High) — recommend a light follow-up build to confirm grade↔surplus live.
- **Trade Finder** ✅ completes but **SLOW: scan time 74.00s** (Trades found 50, Teams scanned 11) — quantifies/confirms task #10; plus a "Re-rank by title odds (~1 min)" even slower. Category Needs (SB/AVG/L/Runs) consistent w/ losing cats ✓. **Soft obs (BR-10b, low conf):** top rec = "Send Swanson + Giménez (2 everyday IF) for Riley O'Brien (marginal RP) +0.81 SGP" looks unappealing + O'Brien (pitcher) doesn't serve the hitting needs; possibly defensible surplus-shed, worth a look.
- **Player Databank** ✅ Search returns 1–25 of 4,235; live stats (refreshed 6/7 1:52 AM); AVG/OBP 3-dp formatted. **% Ros populated here** (36/42%...) → **strengthens BR-9** (percent_owned exists but FA Heat Index ignores it). DNA fix confirmed live elsewhere (Muncy=LAD). Minor: inflated "Current" rank for Luis García Jr. (3750) — matches the documented 3-Luis-Garcias collision.
- **Leaders** ✅ (8.0s). Runs leaderboard descending (James Wood 57→41) ✓; **"Rostered By" matches Yahoo** (Olson/Trout/Yordan/Reynolds → Team Hickey) ✓ L3. 7 tabs present. Inverse-cat (ERA asc) not toggled — code-verified.
- **Admin Console** ✅ Users + Feedback tabs render; pending approvals none; active users testuser·HUMAN INTELLIGENCE (+ Reassign/REVOKE) and conlaxer13·admin·Team Hickey. Inspect-only (no controls actuated). Onboarding flow functional.

## NOT exercised this pass (low priority — recommend a quick follow-up render check)
Punt Analyzer, Player Compare, Draft Simulator (preseason), Usage Analytics, Admin Controls. No reason to expect issues; not exercised due to scope/time. (BR-4 shows crashes are possible, so a fast render pass is worth doing later.)

## Browser findings summary (BR-1..BR-10)
| ID | Sev | Finding |
|---|---|---|
| BR-4 | **HIGH** | Playoff Odds tab crashes (`KeyError: 'accent'`, League_Standings.py:528) after Simulate Season — live traceback to users. |
| BR-6 | (= MS-E1) | Win prob 8.5% (Lineup) vs 29% (Matchup Planner) for same matchup — divergent variance models confirmed live. |
| BR-8 | Med | FA "No add/drop recommendations" for a 9th-place losing team while list shows large IMPACTs + streaming works. |
| BR-1 | Med | Per-session auth: deep-link/refresh/bookmark → forced re-login. |
| BR-2 | Med | My Team "LOSING CATEGORIES: R·SB" contradicts its own detail + Standings "3-7-2" (7 losing). |
| BR-10 | Med (UX) | Trade Finder scan 74s (task #10). |
| BR-3 | Low/Med | "Week 11" = actuals (My Team) vs 7-day projection (Matchup Planner), same label. |
| BR-7 | Low/Med | Optimizer "Today" IP budget shows weekly-forfeit WARNING from daily pitcher set (16.5/20). |
| BR-9 | Low | FA Ownership Heat Index all zeros (percent_owned exists in Databank, unused by FA). |
| BR-5 | Low | Draft/Home "Yahoo Not Connected" + "Player Pool Loading" under MULTI_USER (task #13). |
| BR-10b | Low | Trade Finder top rec looks lopsided (2 IF for 1 marginal RP). |

## Live confirmations of code findings
DB-C1 (park factors frozen — timestamp fresh but tier=emergency), **DB-C2** (closer heuristic, vivid), **DB-C3/#12** (Matchup STALE), **MS-E1** (8.5% vs 29% win prob), **FA-C2** (L14 columns blank). Yahoo relay LIVE in production. DNA fix (Muncy=LAD) holds. Standings consistent across pages; 12 teams no ghost.
