# HEATER — Master UI/UX Design Audit

**Beta Launch Readiness Review**

| | |
|---|---|
| **Report** | HEATER Master UI/UX Design Audit |
| **Date** | 2026-06-13 |
| **Application** | HEATER — Fantasy Baseball In-Season Manager (Streamlit, hosted on Railway, `MULTI_USER=1`) |
| **App version** | Beta candidate (master tip as of 2026-06-13; "Combustion Index" + Finale design system live) |
| **League under test** | FourzynBurn — 12-team, 23-round snake, Yahoo H2H-categories |
| **Test persona** | *Connor* — owner of **Team Hickey** (record **3-7-1**, **10th of 12**); a sharp **novice** fantasy manager, **not** a data scientist and **not** a CLI user |
| **Scope** | **15 in-season / draft pages** exercised end-to-end (see TOC) |
| **Explicit exclusions** | **Admin Console** (`pages/_admin_console.py`), **Admin Controls** (`pages/_admin_controls.py`), **Usage Analytics** (`pages/_admin_analytics.py`), and the floating **HEATER AI chat panel** (`src/ai/*`) were **out of scope** and are not assessed anywhere in this document. |
| **Testing method** | 15 independent test-user agents, one per page, each reading the page source + supporting engine modules and querying the **live league SQLite DB read-only** (network-blocked, so engine calls fell back to cache/seed). Live rendered state was captured by the orchestrator during a separate browser pass. Heavy Monte Carlo / simulation paths were traced or run at tiny sim counts to avoid burning compute. Outputs that required the Streamlit runtime to render are explicitly marked **"(reconstructed)."** |
| **Data state during audit** | **Degraded / cached.** Yahoo was **"Warming up" / offline**; the home page reported **"Data last refreshed 1219 min ago"**; the server log showed *"Matchup served from SQLite cache (Yahoo offline)."* The matchup cache underlying several pages was **~64 hours (≈4,012 min) stale**. This is acceptable for a UI audit but is itself a finding wherever the UI fails to communicate it. |

---

## Table of Contents

- **Part I — Executive Summary** — overall experience, beta-readiness verdict, per-page scorecard, severity tallies
- **Part II — Cross-Cutting Findings** — nine themes recurring across the app, with per-page evidence
- **Part III — Review of Feature & Button Outputs** — what the controls actually output (and which outputs cannot be trusted)
- **Part IV — Top Major Recommendations** — 14 prioritized, cross-cutting strategic moves
- **Part V — Per-Page Sections**
  - Page 01 — Draft Tool (Home)
  - Page 02 — My Team
  - Page 03 — Lineup Optimizer
  - Page 04 — Closer Monitor
  - Page 05 — Pitcher Streaming
  - Page 06 — Matchup Planner
  - Page 07 — League Standings
  - Page 08 — Punt Analyzer
  - Page 09 — Trade Analyzer
  - Page 10 — Trade Finder
  - Page 11 — Free Agents
  - Page 12 — Player Compare
  - Page 13 — Leaders
  - Page 14 — Player Databank
  - Page 15 — Draft Simulator
- **Part VI — Master Defect Register** — every BLOCKER and HIGH across all 15 pages, severity-ranked
- **Part VII — Appendix** — methodology details, agent roster, honest caveats

---

# Part I — Executive Summary

## 1.1 The headline

HEATER is, beneath the surface, a genuinely sophisticated product. The analytic depth on display — Monte Carlo trade simulation with antithetic variates, Bayesian projection blending, LP-constrained lineup optimization, Gaussian-copula category win probabilities, a six-phase trade engine with playoff-odds deltas — is well beyond what a 12-person Yahoo league would ever expect from a homemade tool. The "Combustion Index" design system is real and largely well-executed: the navy chrome, hot-orange accent, Archivo/Inter/IBM Plex Mono type stack, and the branded "instrument-panel" tables (most visible on **Leaders → Category Leaders** and **Closer Monitor**) look modern and high-tech, exactly as the owner intended.

**But the product is not ready for Beta in its current state.** The audit surfaced a consistent and serious gap between the engine's sophistication and what the user actually sees: **the most data-rich features are precisely the ones that are silently broken**, and the app almost never tells the user when it is showing them stale, partial, or simply wrong numbers. A novice — the explicit target user — would lose trust within the first few clicks on at least five of the fifteen pages.

Three failure patterns dominate, and they recur from page to page:

1. **Broken data presented with full confidence.** The Trade Finder's Value Chart classifies **Juan Soto and Aaron Judge as "Replacement"** players (9,887 of 9,888 players land in the bottom tier) because a time-decay multiplier deflates every score below the fixed tier cutoffs. The Leaders **Hot List** is topped by fringe pitchers with trend deltas of **+392 to +935** caused by a near-zero-denominator division bug, all displaying ".000 AVG / 0 HR" because pitchers are misclassified as hitters. The Leaders **Breakouts** tab scores every player at exactly **50.0** because all Statcast columns in the DB are NULL. None of these failures shows a warning — they render as if correct.

2. **Identity / data-wiring bugs that personalize the app to the wrong (or no) team.** The Lineup Optimizer passes the bare string `"Team Hickey"` to the category-weight engine while the DB stores `"🏆 Team Hickey"`; the lookup misses and **every category weight collapses to 1.00×**, so the optimizer cannot prioritize the user's badly-losing ERA/WHIP this week. On **Player Compare** and **Free Agents**, an empty `league_rosters` slice for Team Hickey makes rostered stars display as **"Free Agent."** These are the same class of emoji/whitespace reconciliation bug the codebase has fixed before (`resolve_viewer_team_name`) — it simply was never wired into these call sites.

3. **Performance that drops the connection.** The **Trade Analyzer** runs a 10,000-sim Monte Carlo synchronously in the button handler (benchmarked at **44.8 seconds**) on top of a playoff simulation, behind a fake progress bar — and this **already dropped the browser's WebSocket and froze the page in live testing.** The **Trade Finder** blocks all rendering for **43.8 seconds** on a cold scan. The **Punt Analyzer** spends **~2.1 seconds** in a row-by-row SGP loop that a vectorized call already in the codebase would do in **~6 milliseconds** (413× faster).

On top of these, a pervasive layer of **novice-hostile jargon with no tooltips or legends** — DCV, SGP, wRC+, "Smash", "% JOB", "Sell-High", "Net SGP", "Magic#", "SOS", and risk flags like `SHORT_LEASH` and `HIGH_WHIP` — means that even the pages that *work* are hard for the target user to act on.

## 1.2 Beta-readiness verdict

> **NOT READY for an open Beta. Conditional-go for a small, hand-held pilot only after the BLOCKER and identity-class HIGH defects are fixed.**

The app is impressive enough that it should absolutely ship — but not until the trust-destroying defects are closed. A leaguemate invited today would, on a bad draw of clicks, land on a draft wizard for a draft that finished in March (Home), see their own roster as "Free Agents" (Player Compare), watch the page freeze for 45 seconds (Trade Analyzer), and read that Aaron Judge is a "Replacement"-tier asset (Trade Finder). Any one of those is enough for a novice to conclude the tool is broken and stop using it.

The good news: the defects are concentrated and fixable. The engines are correct; it is the **wiring, the staleness communication, the performance gating, and the explanatory layer** that need work — not the math. A focused two-to-three-week hardening pass on the items in Parts IV and VI would move this from "impressive but untrustworthy" to "genuinely shippable."

## 1.3 Per-page scorecard

Grades reflect the **out-of-the-box experience for the novice persona on the current (cached) data**, weighting trust and clarity as heavily as raw capability. A page can be analytically brilliant and still score poorly if it shows broken numbers without warning.

| # | Page | Grade | One-line rationale |
|---|------|:-----:|--------------------|
| 01 | Draft Tool (Home) | **D** | In-season users land on a dead draft wizard; SGP chips, tiers, and pick reasons are all hardcoded/empty; unknown prospects outrank Aaron Judge in HEATER's own score. |
| 02 | My Team | **C-** | Rich and on-brand, but the matchup ticker is hidden offline, Record/Rank show "—", two intelligence cards silently vanish (NULL Statcast), and "Live"/"Updates hourly" labels are false on ~3-day-old data. |
| 03 | Lineup Optimizer | **D+** | The category-weight engine is silently fed the wrong team name → all weights 1.00× → wrong lineups for a 10th-place team; emoji in toasts; raw decimals with no scale. |
| 04 | Closer Monitor | **C** | Handsome grid, but shows 21 of a promised 30 teams with no explanation, conflates projected vs actual saves, and the heatbar never varies. |
| 05 | Pitcher Streaming | **C** | Powerful engine, but a 17-column wall with no legend, a positive score paired with negative Net SGP left unexplained, a `FIG.4` typo, and no freshness badge. |
| 06 | Matchup Planner | **C-** | Good headline win-prob, but a dead `schedule_warning` leaves every player showing "Avoid / 0 games" offline with no explanation; week navigator capped at 24 of 26; `FIG.02` caption on a `FIG.05` page. |
| 07 | League Standings | **C** | Strong instrument panel, but the category grid has no legend, the Streak column is permanently empty, Win% reads like a batting average, and the Scenario Explorer mislabels category-wins as record-wins. |
| 08 | Punt Analyzer | **D** | The "Biggest Gainers" table shows 15 players with **+0.00** change; ghost team "Twigs" inflates ranks to /13; 413× slower than necessary; "punt" is never defined. |
| 09 | Trade Analyzer | **D-** | The deepest engine in the app, gutted by a **44.8s** synchronous MC that drops the WebSocket; grade range hidden; "200 sims" tooltip 50× wrong; internal "report B.9 / Feature 2" labels leak to users. |
| 10 | Trade Finder | **D+** | Genuinely useful concept, undermined by a **43.8s** cold scan, a Value Chart that calls Soto/Judge "Replacement," a string-sorted Acceptance column, and "ECR 0%" on every 2-for-1. |
| 11 | Free Agents | **C-** | Solid recommendations, but "Click column headers to sort" is a lie, a `player_news` query throws every load (silently), "Budget: 5/week" is wrong (10), and a Curtis Mead 125.19 outlier (wrong team) tops the list. |
| 12 | Player Compare | **D+** | Rostered players show "Free Agent"; injury-badge HTML renders as raw text; half the radar collapses to zero for same-type pairs; the killer "category fit vs my team" feature exists in code but is never called. |
| 13 | Leaders | **D** | Category Leaders tab is the best-looking component in the app, but Hot/Cold are garbage (392–935 deltas), Breakouts are all 50.0, and Prospects have NULL ranks + NULL scouting grades. |
| 14 | Player Databank | **C-** | IP renders as "56.7" (should be 56.2); "Today (live)" shows projections; "Waivers Only" and "Util" filters silently no-op; "Advanced" view shows no advanced columns. |
| 15 | Draft Simulator | **C** | Clean mock-draft flow, but `PRESEASON` framing orphans it mid-season, Undo undoes the AI's pick not yours, and the "View player card" selectbox prints raw `<img>` HTML. |

**Distribution:** 0 × A, 0 × B, **6 pages in the C band** (02, 04, 05, 06, 07, 14), **9 pages in the D band** (01, 03, 08, 09, 10, 11, 12, 13, 15 — with Trade Analyzer the lone D-). There is **no page a novice could use today without hitting at least a MEDIUM-severity confusion or a broken output.**

## 1.4 Severity tally across the whole app

Aggregating the severity-tagged lists from all 15 page reports (a few cross-cutting items, e.g. data-freshness, recur across pages and are counted once per page where flagged):

| Severity | Count (approx.) | Character |
|----------|:---------------:|-----------|
| **BLOCKER** | **15** | Trust-destroying or connection-dropping. Must fix before any Beta. |
| **HIGH** | **~75** | Wrong/misleading outputs, hidden errors, identity bugs, missing freshness comms, novice-blocking jargon at the point of decision. |
| **MEDIUM** | **~120** | Real friction: layout/ordering, unexplained metrics, redundant tabs, silent `except: pass`, formatting. |
| **LOW / POLISH** | **~150** | Cosmetic, consistency, microcopy, mobile, accessibility. |

The 15 BLOCKERs are listed in full in **Part VI** — they are the gating items. The ~75 HIGHs are where the bulk of the "feels broken / feels confusing" experience lives, and closing even half of them would materially change the first impression.

---

# Part II — Cross-Cutting Findings

These nine themes recur across many pages. Each is the kind of problem that is cheaper to fix once, centrally, than fifteen times. Evidence is cited from specific pages.

## (a) Data-staleness & "Warming up" communication failures

The single most pervasive issue. During the audit Yahoo was offline and data was **~1,219 minutes (≈20 hours) old at minimum**, with the matchup cache **~4,012 minutes (≈2.8 days) stale**. Almost no page told the user this clearly, and several actively lied about it.

- **My Team** displays a card literally titled **"Live Matchup"** on 2.8-day-old data, and the roster table caption is hardcoded to *"Updates hourly from MLB Stats API"* — factually false. The Matchup Pulse, Category Gaps, and Category Flip cards all render stale numbers with no "as of" qualifier.
- **League Standings** shows *"leading 7-3-2 in categories"* in the banner with no "as of" date, even though the matchup cache is ~64 hours old.
- **Matchup Planner**, **Pitcher Streaming**, **Player Compare**, **Punt Analyzer**, **Trade Finder (Value Chart)**, and **Draft Simulator** have **no data-freshness indicator at all**. The Lineup Optimizer and My Team *do* have a freshness card, proving the component (`render_data_freshness_card()`) exists — it is simply not used consistently.
- Where freshness *is* shown, "Yahoo: Warming up" / "Yahoo Status: Warming up" is rendered in **muted gray text** (`T["tx2"]`) — the same visual weight as benign states — so a novice cannot tell that "Warming up" is a degraded condition.
- The **My Team** "Last synced" pill shows a single `MAX(last_refresh)` timestamp for a multi-source pipeline, so a user can't distinguish "Yahoo last pulled" from "MLB Stats API last ran."

**Net effect:** the user has no way to know whether the confident-looking numbers in front of them are 10 minutes or 3 days old. For a tool used to make weekly start/sit and waiver decisions, this is a credibility problem on its own.

## (b) Performance / heavy-render blockers

Three pages have compute heavy enough to harm or break the experience:

- **Trade Analyzer** — the worst offender. `enable_mc=True` with **10,000 sims** runs synchronously inside the `st.button` handler; benchmarked at **44.77 seconds**, on top of an additional (unbenchmarked, expensive) 20,000-sim playoff simulation. A fake 4-step progress bar completes in ~1 second, then the page sits frozen for 40+ seconds. **This already dropped the browser WebSocket and froze the page in live testing.** There is no timeout, no opt-out, and no partial-results path. Additionally, `load_player_pool()` (4.32s) + `get_health_adjusted_pool()` (0.52s) run **uncached on every rerun**, so even *visiting* the page costs ~5 seconds of blank screen.
- **Trade Finder** — cold scan via `find_trade_opportunities(max_results=50, top_partners=11)` measured at **43.8 seconds** before any content appears. The session cache only helps within a session; every new session or post-sync reload re-scans. The "Re-rank by title odds" toggle understates its cost as "~5 sec" when the real total is **~2 minutes** (44s base + ~60s MC).
- **Punt Analyzer** — iterates `pool.iterrows()` calling `total_sgp()` row-by-row across 9,888 players, **twice**, measured at **~2.1 seconds**. The vectorized `total_sgp_batch()` already exists in `SGPCalculator` and does the same work in **~0.003s per pass — a 413× speedup.**
- Secondary heavy paths with no caching: **Matchup Planner** runs a 10,000-sim copula MC on every page load (no `@st.cache_data`); **Leaders** runs `compute_player_trends` + `detect_sell_high_candidates` + breakout scoring uncached on every tab switch; **Free Agents** loops `marginal_sgp` over 7,770 FAs with no spinner; **Player Databank** re-runs a 1.4–3.6s load on every Search.

The orchestrator separately observed a *"12s budget exceeded … slow/degraded MLB Stats API"* warning, confirming the live environment is latency-sensitive — which makes the synchronous heavy compute even more dangerous under 12 concurrent users on a single Railway replica.

## (c) Novice-hostile jargon with no tooltips or legends

The target user is explicitly a **novice**. The app is saturated with un-glossed expert terms, frequently at the exact moment of decision:

- **Acronyms with no expansion or tooltip:** `SGP` / "Net SGP" / "Surplus SGP" (everywhere — Trade Analyzer, Trade Finder, Free Agents, Punt, Compare), `DCV` (Lineup Optimizer), `wRC+` / "Opp wRC+" (Streaming, Compare), `Magic#` and `SOS` (Standings), `% JOB` and `gmLI` (Closer Monitor), `ECR` / `ADP` (many), `CARA` / `CVaR` / `Var[Δchamp]` / `G-score` / `VORP/PRP` (Trade Analyzer).
- **Coined labels a beginner can't parse:** **"Smash Matchups"** (Matchup Planner — only explained in a below-the-fold legend), **"Sell-High"** and **"Breakout"** vs **"Hot List"** (Leaders — overlapping mental models, no distinction given), **"LEAVE EMPTY"** and **"START ⚠"** (Lineup Optimizer — alarming, unexplained).
- **Risk-flag strings dumped raw:** `HIGH_WHIP`, `SHORT_LEASH`, `ELITE_OFFENSE`, `HITTER_PARK`, `WIND_OUT`, `LOW_CONFIDENCE` appear as bare ALL-CAPS tokens in both the Pitcher Streaming board and the Lineup Optimizer Streaming tab, with **no legend anywhere**.
- **Raw scores with no scale anchor:** "Stream Score 56.10", "Category Value 13.04", "combined_score 33.89", "Marginal Value 125.19", "Readiness 83.0", "DCV 12.34", "Complementarity 0.580" — all shown as bare floats with no indication of the range, what "good" looks like, or where average sits.
- **Inverse-stat semantics never taught:** L (Losses), ERA, WHIP are "lower is better," but the app shows `L: -2` in red (My Team), a `-1.48` z-score "winning" a category (Player Compare), and an "L Leaders" board full of pitchers with **0 losses** (Leaders) — all without explaining the inversion.

The codebase has the machinery for this (`METRIC_TOOLTIPS` exists; Streamlit `help=` and `column_config` tooltips are available) — it is simply under-used.

## (d) Identity / team-name bugs

A cluster of bugs all rooted in the same cause: the user's team is stored in Yahoo as **`🏆 Team Hickey`** (emoji-prefixed), but several call sites compare against the bare env string **`Team Hickey`**, and the reconciliation helper `resolve_viewer_team_name(rosters)` (which already fixes this elsewhere) was never wired in.

- **Lineup Optimizer [BLOCKER]:** `compute_nonlinear_weights()` is called with `team_name="Team Hickey"`; the SQL match against `"🏆 Team Hickey"` finds nothing and returns **equal weights (1.00×) for all 12 categories.** The result is that the LP optimizer and the entire Category Analysis tab provide **no urgency differentiation** — for a team that is winning K/W/HR but losing ERA (7.08 vs 3.38) and WHIP (1.82 vs 1.03) this week, the optimizer fails to prioritize the categories that actually matter.
- **Player Compare [BLOCKER]** and **Free Agents [MEDIUM]:** a raw query `WHERE team_name = 'Team Hickey'` against `league_rosters` returns **0 rows**, so the roster-status lookup is empty and **every player — including rostered stars — displays "Free Agent."** Player Compare is live-broken for the owner's own team.
- This is the exact class of failure called out repeatedly in the codebase history (the 2026-06-01 launch-blocker; PR `b7f0567`). The fix pattern is known and one-line per site; it just needs to be applied at every personalized call site, guarded by the existing structural test.

## (e) Broken / empty data columns

The DB's enrichment tables are partially unpopulated, and the UI does not degrade gracefully — it renders zeros and dashes as if they were real.

- **Statcast is entirely NULL.** Across `statcast_archive`, `xwoba`, `barrel_pct`, `hard_hit_pct`, and `stuff_plus` are **NULL on all rows** (only `sprint_speed` has data). Consequences cascade:
  - **Leaders → Breakouts** scores **every player at exactly 50.0** (fallback path with near-zero projection inputs) and shows "No players score above 70" with no hint the pipeline failed.
  - **My Team** silently drops the **Statcast Signals** and **Regression Alerts** cards entirely (the `if _sc_rows:` guard is never entered) — two intelligence features vanish with no empty state.
  - **Player Compare** still prints a "Stuff+ = pitch quality (100 = average)" caption for a metric that **never renders**, and surfaces projection-blended xwOBA under a header that says "Statcast Profile."
  - **Player Databank** advertises "2026 Advanced" views with no advanced content.
- **Pitcher rows show hitter stats via an `is_hitter or 1` bug.** In Leaders' `_trend_key_stats`, `int(row.get("is_hitter", 1) or 1)` coerces a pitcher's falsy `0` back to `1`, so **every pitcher on the Hot/Cold lists displays ".000 AVG / 0 HR"** instead of ERA/K.
- **Prospect data is hollow.** All 20 `prospect_rankings` rows have **NULL `fg_rank`** (blank Rank column), **NULL `mlb_id`** (generic silhouettes), and **NULL scouting grades** (every prospect's "Scouting Tool Grades" panel shows an empty state). ETA-2025 players who have already debuted (Roki Sasaki, Roman Anthony) still top the "prospect" list.
- **Ownership is dead.** `percent_owned = 0.0` on all 9,888 pool rows and `delta_7d` is NULL, so Free Agents' Heat column is almost entirely `--`, the "Breakout Candidates" toggle is a mathematical impossibility (Heat≥5 requires Owned≥50% while the filter demands Owned<30%), and Databank's "% Ros" column is always a dash — while a `yahoo_free_agents` cache with **608 real ownership values** sits unused.
- **Trade Finder Value Chart** classifies **9,887 of 9,888 players as "Replacement"** (Soto, Judge included) because a `time_factor = 15/26 ≈ 0.577` multiplier deflates every score below fixed tier cutoffs.

## (f) Silent `except: pass` failures hiding real errors

Multiple pages swallow exceptions, turning real failures into invisible blank space:

- **My Team** wraps "Today's Actions" and "Player Streaks" War Room cards in `try/except: pass` — if either engine throws, the card silently disappears with no fallback.
- **Free Agents** has `except Exception: pass` on the entire "This Week's Streams" section, and a `player_news` query that **references a non-existent `player_name` column** (the table only has `player_id`) — it **throws on every page load**, is silently caught, and so the news-based IL-protection path is **permanently a no-op**. The user never sees that one of the IL-safety layers is dead.
- **Punt Analyzer** gates its entire Standings Impact panel on `_HAS_CATEGORY_ANALYSIS`; if that import fails the whole section vanishes with no message.
- **Matchup Planner** builds a `schedule_warning` string and then **never displays it** (dead code), so when the schedule is offline the user just sees 27 rows of "Rating 1.00 / Avoid / 0 games" with zero explanation.

The pattern is consistent: the app prefers to hide a failure than to tell the user a feature is degraded. For a Beta whose whole point is to surface problems, this is exactly backwards.

## (g) Design-system inconsistencies

The Combustion Index is mostly well-applied, but the audit found repeated drift:

- **`FIG.NN` numbering is inconsistent.** Pitcher Streaming renders **`FIG.4`** (not `FIG.04`). Matchup Planner puts a **`FIG.02 · WIN PROBABILITY BY CAT`** sub-caption on a page whose header is **`FIG.05`**. League Standings reuses within-page `FIG.01/02/03` labels that collide with other pages' figure numbers. The global `FIG.NN` is keyed to the *source-file number* (e.g. `FIG.17` for `17_Leaders.py`, the 13th page in nav), which will confuse anyone reading a multi-page spec.
- **Nav-label vs page-title drift.** Sidebar **"Punt Analyzer"** → H1 **"Punt Strategy Simulator."**; sidebar **"Lineup Optimizer"** (no hyphen) → header **"Line-up Optimizer"** (hyphen).
- **Off-palette hex literals** that escape the CI guard because they live in `src/` not `pages/`: `TIER_COLORS["Star"]="#457b9d"`, `["Flex"]="#666666"`, `["Replacement"]="#cc3333"` in `src/trade_value.py` (Trade Finder); `color:#2c2f36` in `src/player_databank.py` (Databank). Closer Monitor uses `var(--font-display)` (Archivo) for stat figures where the spec mandates `var(--font-mono)`.
- **Emoji that the no-emoji rule bans:** `st.toast("✅ ...", icon='✅')`, `'⚠️'`, `'❌'` in the Lineup Optimizer; `📅` in the Trade Analyzer's Weekly H2H expander label; `START ⚠` / `⚠ LP wants to start...` banners in the Lineup Optimizer.
- **Wrong semantic color:** AVOID badge uses `T["primary"]` (orange = the CTA color) instead of `T["danger"]` (Home); positive "Net SGP" rendered in `var(--fp-ember)` red instead of orange (Free Agents).
- **`slideUp` entrance animation on a persistent card** (Trade Analyzer verdict banner) — the Combustion Finale lock prohibits this because Streamlit remounts the DOM every rerun, so it replays on every click.
- **Native components where Combustion components are specified:** bare `st.info` / `st.subheader` instead of `render_empty_state()` / the panel-header pattern on Punt, Trade Finder, Free Agents, Player Compare, Draft Simulator.

## (h) The in-season-vs-draft-tool framing problem

The app is an **in-season manager**, but two of its pages are **draft tools** that feel orphaned during Week 12:

- **Home (Draft Tool)** is the **default landing page** for all 12 leaguemates, and it is a **2-step pre-draft setup wizard** for a draft that completed in March. There is no in-season orientation, no link to My Team / Free Agents / Lineup Optimizer, and no explanation of why this page exists now. A leaguemate clicking "Draft Tool" in the sidebar during the season has no idea what to do. This is the app's **single highest-impact framing failure.**
- **Draft Simulator** carries a **`PRESEASON`** eyebrow and zero framing for why a mock draft is useful mid-season. A novice will conclude they opened the wrong thing.

Both pages should either be visually grouped into a clearly-labeled "Preseason / Draft" section, deprioritized in the nav, or given an in-season framing banner — and **My Team should be the default landing page** during the active season.

## (i) `st.components.v1.html` deprecation overdue for removal

The server logs spew *"Please replace `st.components.v1.html` with `st.iframe`. … will be removed after 2026-06-01"* — a date **already passed**. The audit traced its usage:

- **Confirmed in use:** the **Home / bootstrap splash screen** uses `st.components.v1.html` for the self-ticking HH:MM:SS load timer (the one remaining usage on that page).
- **Confirmed clean:** every other audited page renders custom HTML via `st.markdown(..., unsafe_allow_html=True)` (the approved pattern) — Closer Monitor, Matchup Planner, Punt, Trade Analyzer, Trade Finder, Pitcher Streaming, Free Agents, and Draft Simulator all explicitly checked clean.

The fix is a one-line swap to `st.iframe(...)` on the splash timer. It should be done before Beta simply to stop the deprecation-warning spam and avoid breakage when the API is removed.

---

# Part III — Review of Feature & Button Outputs

This part consolidates **what the app's controls actually output** — the real numbers and labels the agents captured — and flags which outputs **cannot yet be trusted**. Trust ratings: ✅ trustworthy, ⚠️ misleading / needs context, ❌ broken / wrong.

| Page | Control / output | Captured real output | Trust |
|------|------------------|----------------------|:-----:|
| Home | Hero pick (Standard mode) | Shohei Ohtani, score **20.01**, survival **31%** (red), "FAIR", "Tier 1", reason "Best available by combined score" | ⚠️ tier always 1, reason always identical |
| Home | SGP category chips | **always empty** (all 10 values 0) | ❌ feature non-functional |
| Home | Available Players top-of-board | Jonah Cox (ADP 999) score **8.77** > Aaron Judge (ADP 1.0) score **8.02** | ❌ unknowns outrank stars |
| Home | Auto-computed SGP denominators | R 18.07 / HR 8.13 / RBI 15.23 / SB 9.40 / AVG 0.008 / OBP 0.011 / W 1.49 / L 5.90 / SV 5.42 / K 31.63 / ERA 0.447 / WHIP 0.095 | ✅ plausible |
| My Team | Matchup Pulse (Week 12) | **7-3-2**; you 14R/7HR/14RBI vs 6/1/8; ERA 7.08 vs 3.38; WHIP 1.82 vs 1.03 | ⚠️ ~2.8 days stale, labeled "Live" |
| My Team | Hitting totals (season) | R 450 / HR 137 / RBI 410 / SB 64 / AVG .237 / OBP .332 | ✅ (but no league-rank context) |
| My Team | Pitching totals | W 55 / L 37 / SV 23 / K 704 / ERA 3.57 / WHIP 1.24 / IP 692.3 | ✅ |
| My Team | Identity Record / Rank | **"—" / "—"** (live Yahoo client absent; data IS in DB) | ❌ shows blank when data exists |
| My Team | Statcast Signals + Regression Alerts | **cards do not render** (NULL Statcast) | ❌ silent vanish, no empty state |
| Lineup Opt | Category weights (all 12) | **1.00× across the board** (emoji team-name miss) | ❌ no urgency differentiation |
| Lineup Opt | DCV "Why?" expander | matchup_mult 0.8745 / volume_factor 1.0 / health_factor 1.0 / total_dcv 12.34 | ⚠️ no scale/meaning |
| Lineup Opt | IP Budget card | 53.85 IP target / 20.0 floor — **blank until first Optimize click** | ⚠️ false precision + blank-on-load |
| Closer Mon | Card SV (e.g. CLE Cade Smith) | **proj 7** (orange) vs **"2026 ACTUAL · 21 SV"** (green) | ⚠️ two SV numbers, projected unlabeled |
| Closer Mon | ARI / Paul Sewald stats | SV **—** / ERA **—** / WHIP **—**, then "2026 ACTUAL · 15 SV · 3.47 ERA" | ❌ AZ→ARI normalization mismatch |
| Closer Mon | % JOB range | 44% (TOR) to 68% (PHI); heatbar orange on ~20/21 | ⚠️ no variation, no legend |
| Closer Mon | Teams shown | **21 of a promised 30** (BAL/Helsley, CIN/Pagán, etc. missing) | ⚠️ unexplained omission |
| Streaming | Stream board row | Kyle Leahy @ MIN — **Score 56.10**, **Net SGP -0.07**, Opp wRC+ 97, K% 22.8 | ⚠️ positive score + negative SGP, no explanation |
| Streaming | Cats in play (offline) | "no matchup data" | ⚠️ useless when most needed |
| Matchup | Win probability | **Win 46% / Tie 19% / Loss 35%**, projected 7-5 (banner says 6-6) | ⚠️ banner/card disagree |
| Matchup | Per-cat win prob | K 93% / W 81% / OBP 58% … AVG 38% / L 14% | ✅ (reconstructed) |
| Matchup | Player ratings (offline) | **every player "Rating 1.00 / Avoid / 0 games"** | ❌ dead `schedule_warning`, no explanation |
| Standings | H2H table | 1. Over the Rembow 10-1; **10. 🏆 Team Hickey 3-7-1 (.318)**; 12. Going…Gonorrhea 2-8-1 | ✅ |
| Standings | Streak column | **empty for all 12 teams** | ❌ dead column |
| Standings | Category grid (Hickey) | HR rank 2 (green), SB 12 (red), AVG 12 (red), K 2 (green), L 11 (red) | ⚠️ no legend, no stat values |
| Punt | "Biggest Gainers" (Punt SB) | 15 players **all Change +0.00** (Skubal, Skenes, Caminero…) | ❌ "gainers" who gained nothing |
| Punt | Standings Impact ranks | HR 1/**13**, SB 11/**13**, AVG 12/**13** | ❌ ghost team "Twigs" → /13 |
| Punt | Standings points (Punt SB) | **78** (help: "Total 80, active 78, punted 2") | ⚠️ no denominator/benchmark |
| Trade Anlz | Verdict (Swanson→Reynolds, MC off) | Grade **C**, DECLINE, surplus **-0.086**, confidence 43.1% | ✅ Phase-1 verdict |
| Trade Anlz | Grade range | computed **{C, low D, high B+, low confidence}** — **never shown** | ❌ dropped silently |
| Trade Anlz | Reshuffle warning | "Lineup reshuffle accounts for **2188%** of surplus" | ❌ near-zero denominator blowup |
| Trade Anlz | MC (10K sims) | wall time **44.77s**, mean -2.18, CVaR5 -8.88, P(helps) 24%, convergence **"marginal"** | ❌ drops WebSocket; convergence warning hidden |
| Trade Anlz | MC tooltip | claims **"200 simulated seasons"** (engine runs 10,000) | ❌ 50× wrong |
| Trade Find | Top recommendation | "Send Dansby Swanson to Over the Rembow for Sonny Gray (**+0.66 SGP**)" | ✅ banner |
| Trade Find | Value Chart tiers | Elite **0**, Star **0**, Ohtani only "Solid Starter" (57.7); **Soto & Judge "Replacement"** | ❌ time-decay vs fixed cutoffs |
| Trade Find | Cold-scan time | **43.8 s** before any content | ❌ blocks render |
| Trade Find | ECR Fairness (2-for-1) | **"0%"** on every multi-player trade | ❌ neutral mis-rendered |
| Free Agents | Top ranked FA | **Curtis Mead** marginal_value **125.19** (5× the #2; team listed "WSH" — wrong, he's TBR) | ❌ outlier + data error |
| Free Agents | "Click headers to sort" | static HTML table — **clicking does nothing** | ❌ false affordance |
| Free Agents | Streaming budget | "Budget: **5** streaming adds per week" | ❌ actual limit is 10 |
| Player Cmp | Roster badges | Soto, Judge both **"Free Agent"** | ❌ empty roster slice |
| Player Cmp | Soto vs Judge composite | **+32.31 vs +25.36**; Soto wins 6 cats, **6 "ties"** (all pitching) | ⚠️ "ties" are N/A categories |
| Player Cmp | Health column | renders `<span style=...>●</span> Low Risk` as **literal text** | ❌ HTML not interpreted |
| Leaders | Hot List top entries | Burch Smith **+392.235**, Cody Bolton **+366.670**, all ".000 AVG / 0 HR" | ❌ denominator blowup + is_hitter bug |
| Leaders | Breakouts | **every player exactly 50.0** | ❌ NULL Statcast |
| Leaders | Prospects | fg_rank **NULL** (blank), scouting grades **NULL**, Sasaki/Anthony (ETA 2025) atop list | ❌ hollow data |
| Leaders | Sell-High | **1 player** (Andrew Vaughn) | ⚠️ threshold too tight |
| Databank | Zack Wheeler IP | **56.7** (should be 56.2 outs notation) | ❌ wrong baseball notation |
| Databank | "Today (live)" view | returns **ROS projections**, not live stats | ❌ mislabeled |
| Databank | "Waivers Only" / "Util" filters | "W" → all 4,408 rows; "Util" → **0 rows** | ❌ silent no-op / dead control |
| Draft Sim | Score badge (Ohtani pick 1) | **33.89** (combined_score) vs **15.61** (pick_score on quick card) | ⚠️ two unexplained scores |
| Draft Sim | Chourio / Langford team | shown as **"FA"** (empty `team=''` in DB) | ❌ active MLB players mislabeled |
| Draft Sim | Undo Last Pick | undoes the **AI's** pick, not the user's | ❌ misleading control |

**Summary of trust:** of the ~50 representative outputs captured, roughly **half are ❌ broken/wrong or ⚠️ misleading.** The trustworthy outputs cluster in the straightforward read-only displays (season totals, the standings table, win probabilities). The broken ones cluster — predictably — in the **most advanced features** (trade values, breakouts, trends, prospects, Statcast, optimizer weights). This is the central paradox of the product: its sophistication is concentrated exactly where its reliability is weakest.

---

# Part IV — Top Major Recommendations

Fourteen strategic, cross-cutting moves, ordered by impact. Each names the problem, the change, and the pages affected. These are the levers that most improve the product per unit of effort.

### REC-1 — Make the heavy compute non-blocking (gate, background, or cap it). [BLOCKER class]
**Problem:** The Trade Analyzer's 44.8s synchronous Monte Carlo drops the WebSocket and freezes the page; the Trade Finder's 43.8s cold scan blocks all rendering. Under 12 concurrent users on one Railway replica, this is a denial-of-service against the app itself.
**Change:** (a) Default `enable_mc=False` on Trade Analyzer; make MC an explicit opt-in checkbox labeled with its cost ("Run risk analysis — adds ~45s"); ideally run it in a background thread and stream Phase-1 results immediately. (b) Move the Trade Finder scan to the **background scheduler** (the existing sole-writer) and have the page read cached results that are at most N hours old; reduce the default scan to `max_results=20, top_partners=5` with an "Expand search" button. (c) Wrap every expensive pool load in `@st.cache_data` (Trade Analyzer, Free Agents, Leaders, Databank, Matchup Planner). (d) Switch the Punt Analyzer to `total_sgp_batch()` (413× faster).
**Pages:** Trade Analyzer, Trade Finder, Punt Analyzer, Matchup Planner, Free Agents, Leaders, Player Databank.

### REC-2 — Fix the identity/team-name resolution everywhere it is missing. [BLOCKER class]
**Problem:** `"Team Hickey"` (env) vs `"🏆 Team Hickey"` (DB) mismatches collapse the Lineup Optimizer's category weights to 1.00× and make rostered players show as "Free Agent" on Player Compare and Free Agents.
**Change:** Route every personalized "my team" lookup through `resolve_viewer_team_name(rosters)` (the helper that already exists and is guarded by `test_pages_use_viewer_team_resolver.py`). Apply it at the `compute_nonlinear_weights()` call site and at the roster-status badge lookups. Add the missing call sites to the structural guard so they can't regress.
**Pages:** Lineup Optimizer, Player Compare, Free Agents (and any other personalized surface).

### REC-3 — Add a consistent, honest data-freshness banner to every page; stop lying about "Live." [BLOCKER/HIGH class]
**Problem:** Data is up to ~2.8 days stale, yet a card is titled "Live Matchup," a caption claims "Updates hourly," and most pages show no staleness at all. "Warming up" is rendered in benign gray.
**Change:** Standardize on `render_data_freshness_card()` / a compact freshness chip at the top of **every** page, showing the relevant source's age in human terms ("Updated 2 days ago") and using an **amber** treatment when > 24h. Replace the hardcoded "Updates hourly" caption with the computed age. Rename "Live Matchup" → "Matchup — Week 12 (cached)." Make "Warming up" visually a warning state, not muted text.
**Pages:** all 15 (especially My Team, League Standings, Matchup Planner, Pitcher Streaming, Trade Finder, Player Compare, Free Agents, Punt, Draft Simulator).

### REC-4 — Fix the trust-destroying broken outputs in the advanced features. [BLOCKER class]
**Problem:** The features that should be the product's showcase — Trade Value Chart, Breakouts, Hot/Cold trends, Prospects — show provably wrong data with full confidence.
**Change:** (a) **Trade Finder Value Chart:** assign tiers from the *pre-decay* value (or scale the Elite/Star/Flex cutoffs by `time_factor`) so Soto/Judge are not "Replacement." (b) **Leaders Hot/Cold:** add a minimum projected-volume gate (skip players with `ip_proj < 15` / `pa_proj < 50`) and `np.clip` the delta to [-3, +3] before classification; fix the `is_hitter or 1` coercion so pitchers show ERA/K. (c) **Leaders Breakouts / My Team Statcast cards:** when all Statcast columns are NULL, show a clear "Statcast data not loaded" empty state instead of 50.0 noise or a silently-missing card. (d) **Prospects:** fall back to `fg_fv` for ranking when `fg_rank` is NULL, resolve `mlb_id` by name, and hide the scouting panel when grades are NULL.
**Pages:** Trade Finder, Leaders, My Team.

### REC-5 — Build a glossary + tooltips layer for the novice. [HIGH class]
**Problem:** SGP, DCV, wRC+, "Smash", "% JOB", Net SGP, Magic#, SOS, the risk-flag strings, and a dozen raw scores are never explained, frequently at the point of decision.
**Change:** (a) Add `help=` tooltips (or `column_config` tooltips) to every jargon column header and every coined-label metric. (b) Ship a small, reusable "What do these numbers mean?" expander/legend component and place it on every page that shows scores. (c) Give every raw score a scale anchor — a percentile, a tier label, or a "league avg = X" caption. (d) Teach inverse stats inline ("L, ERA, WHIP — lower is better; shown quality-adjusted").
**Pages:** all 15, especially Lineup Optimizer, Pitcher Streaming, Trade Analyzer, Trade Finder, Free Agents, Player Compare, Leaders, Standings, Matchup Planner.

### REC-6 — Reframe the app for in-season use; demote the draft tools. [HIGH class]
**Problem:** The default landing page is a dead draft wizard; the Draft Simulator says "PRESEASON" with no context. In-season leaguemates are dropped into the wrong mental model.
**Change:** Make **My Team** the default landing page during the active season (`weeks_remaining > 0`). Move Draft Tool and Draft Simulator into a clearly-labeled "Preseason / Draft" nav group. Add a one-line in-season framing banner to both ("Draft's done — here's where to go," with quick links / "Use this to prep for next year or test values"). Change the Draft Simulator eyebrow from `PRESEASON` to `SCOUTING`.
**Pages:** Home (Draft Tool), Draft Simulator, plus the nav (`src/nav.py`).

### REC-7 — Surface, don't swallow: replace silent `except: pass` and dead warnings with visible states. [HIGH class]
**Problem:** Failures are hidden — My Team cards vanish on exception, Free Agents' `player_news` query throws every load and is silently caught (a whole IL-safety layer is dead), Matchup Planner's `schedule_warning` is built but never shown, Punt's standings panel disappears on import error.
**Change:** Replace bare `except: pass` with `render_empty_state(...)` or a visible `st.warning` that names the degradation. **Display** Matchup Planner's `schedule_warning`. **Fix** the Free Agents `player_news` query to join `players` for the name (it currently references a non-existent column). Translate the engine-fallback warning out of "P3.5 fixes / Check logs" jargon.
**Pages:** My Team, Free Agents, Matchup Planner, Punt Analyzer.

### REC-8 — Fix the per-page broken/dead controls. [HIGH class]
**Problem:** Several controls actively lie or do nothing: "Click column headers to sort" (Free Agents — static HTML); "Waivers Only" and "Util" filters (Databank — silent no-op / 0 rows); "Today (live)" view (Databank — shows projections); "Simulate Opponent Pick" (Draft Sim — dead code); "Undo Last Pick" (Draft Sim — undoes the AI's pick); "Sort by Acceptance" (Trade Finder — string-sorts High<Low<Medium).
**Change:** Either implement or remove each. Remove the false "sort" caption; drop/implement the dead filters and stub stat views; rename "Today (live)" to "ROS Projections"; fix Undo to undo the user's own pick; sort Acceptance on a numeric hidden column.
**Pages:** Free Agents, Player Databank, Draft Simulator, Trade Finder.

### REC-9 — Right-size the most overwhelming tables; front-load the answer. [HIGH class]
**Problem:** The 17-column Pitcher Streaming board, the up-to-21-column Free Agents table, and the 20-section Trade Analyzer output bury the one number that matters. Streaming's Score is column 7; the Trade verdict is the 3rd thing rendered (after the playoff sim).
**Change:** Front-load the key column/result: a "Top pick today" callout card above the Streaming board (Score as column 1 or a Rank column); the ACCEPT/DECLINE verdict **first** on the Trade Analyzer with advanced metrics behind expanders; a "compact" default column set on Free Agents/Streaming with a "Show all stats" toggle. Cap "show all 7,770" with server-side pagination (it currently risks an OOM browser freeze).
**Pages:** Pitcher Streaming, Free Agents, Trade Analyzer.

### REC-10 — Fix number formatting and inverse-stat presentation. [HIGH/MEDIUM class]
**Problem:** IP renders as "56.7" instead of "56.2" (Databank — every pitcher), WHIP as `.3f` (Home Category Balance), integer stats as "15.00" (My Team Marcel projections), Win% as ".318" reading like a batting average (Standings), raw trend deltas like "392.235" (Leaders), and inverse-stat z-scores show negative numbers "winning" categories (Player Compare).
**Change:** Add an IP→outs format branch (the reverse of `_ip_outs_to_decimal`); enforce `format_stat` everywhere (WHIP `.2f`, SGP `+.2f`); render counting stats as integers; show Win% as "31.8%"; quality-adjust inverse-stat z-scores so higher is always better and annotate "(lower is better)."
**Pages:** Player Databank, Home, My Team, League Standings, Leaders, Player Compare.

### REC-11 — Make every leaderboard/recommendation roster-aware. [MEDIUM/HIGH class]
**Problem:** The app rarely connects league-wide data to the user's actual team. Closer Monitor doesn't flag which closers are on your roster or available; Punt shows the full 9,888-player pool instead of "which of MY players lose value"; Leaders has no "free agents only" filter; Player Compare's "category fit vs my team" function exists in code but is never called.
**Change:** Add a "My Roster / Available / All" lens (chips or a toggle) to Closer Monitor, Punt, Leaders, and the FA-relevant tables. Wire `compute_category_fit()` into Player Compare to show which player better fills the user's weak categories. Add "MINE / FREE / opponent" badges where players are listed league-wide.
**Pages:** Closer Monitor, Punt Analyzer, Leaders, Player Compare, Free Agents.

### REC-12 — Add action guidance and deep-links so diagnosis leads to action. [MEDIUM class]
**Problem:** Pages diagnose but don't prescribe. Standings says "2 GB from playoffs" and stops. Matchup Planner shows "L: 14%" with no next step. My Team's flip suggestions ("stream a high-K SP") don't link anywhere. Trade Finder recommendations can't be sent to the Trade Analyzer.
**Change:** Use `render_reco_banner`'s `expanded_html` (currently empty on most pages) to add one or two lines of "do this next." Deep-link suggestions to the relevant page (streaming → Pitcher Streaming, weak cat → Free Agents). Add an "Analyze in Trade Analyzer →" button on Trade Finder rows that pre-populates the proposal.
**Pages:** League Standings, My Team, Matchup Planner, Trade Finder, Punt Analyzer.

### REC-13 — Clean up the design-system drift before Beta. [MEDIUM/LOW class]
**Problem:** `FIG.4` vs `FIG.04`; `FIG.02` caption on a `FIG.05` page; nav-vs-title drift; off-palette hex in `src/`; emoji in toasts and an expander label; `slideUp` on a persistent card; AVOID badge orange instead of red; positive SGP rendered in ember red; native `st.info`/`st.subheader` where Combustion components are specified.
**Change:** Zero-pad all `FIG.NN`; remove stale within-page figure captions; reconcile nav labels to page titles; replace the off-palette hexes in `src/trade_value.py` and `src/player_databank.py` with THEME tokens and extend the CI guard to `src/`; strip emoji from toasts/labels; remove the `slideUp`; fix the badge/figure colors; swap native components for `render_empty_state`/the panel header.
**Pages:** Pitcher Streaming, Matchup Planner, League Standings, Punt, Lineup Optimizer, Trade Analyzer, Trade Finder, Free Agents, Player Databank, Player Compare, Draft Simulator.

### REC-14 — Replace `st.components.v1.html` and do a "stale label / dead column" sweep. [MEDIUM/LOW class]
**Problem:** The splash-screen timer uses a deprecated API past its removal date (warning spam, future breakage). Separately, several dead columns/labels persist: the always-empty Streak column (Standings), the always-dash "% Ros" (Databank/Free Agents) while a 608-row ownership cache sits unused, the always-empty SETUP/gmLI rows (Closer Monitor), and the ghost team "Twigs" in standings.
**Change:** Swap the timer to `st.iframe`. Either populate or remove the dead columns. Join `yahoo_free_agents` (real ownership) into the pool. Purge the "Twigs" ghost row from standings and apply `filter_standings_to_valid_teams` on the Punt page.
**Pages:** Home, League Standings, Player Databank, Free Agents, Closer Monitor, Punt Analyzer.

---

# Part V — Per-Page Sections

Each section below synthesizes one page-auditor's findings into the report's house style. They preserve the concrete evidence — real player names, captured numbers, and code references — while sharpening the narrative. Sections appear in sidebar order.

---

## Page 01 — Draft Tool (Home)

**Source:** `app.py` — `render_single_user_app()` → `render_setup_page()` → `render_step_settings()` / `render_step_launch()` → `render_draft_page()`

### Purpose & first impression

The Draft Tool is HEATER's **default landing page**: a splash bootstrap, a two-step pre-draft setup wizard (Settings → Launch), and a three-column live draft board with Monte Carlo recommendations. As a draft assistant it is competent. The problem is *when* it greets the user. It is **June, Week 12 of the active 2026 season**, and the league's draft finished in March. The first thing all twelve leaguemates see when they open HEATER is a setup wizard for a draft that no longer exists, with no in-season orientation, no link to My Team / Free Agents / Lineup Optimizer, and no explanation of why this page is in front of them. A novice clicking "Draft Tool" in the sidebar during the season simply does not know what to do here. This single framing failure is the page's headline finding and one of the most consequential in the whole audit.

The chrome itself is clean and on-brand — the big "HEATER." wordmark with the orange period, the `FOURZYNBURN LEAGUE · 2026 SEASON / FIG.00 · COMMAND HOME` eyebrow, the two-step wizard stepper, and two read-only info cards: "LEAGUE FORMAT — 12 Teams / 23 Rounds · Snake · H2H Categories" and "PLAYER POOL — 9,888 / Yahoo: Warming up." It looks the part. It is the *context* that is wrong.

### Feature & output review

The draft engine works end-to-end, but several of its showcased outputs are hollow, and the audit captured the exact values:

- **Hero pick card (Standard mode, reconstructed):** Shohei Ohtani at pick #1 — **score 20.01**, survival gauge **31% (red)**, classification **FAIR** (sky blue), confidence **MEDIUM** (orange border), position "TWP · Tier 1", reason **"Best available by combined score."**
- **SGP category chips — always empty.** The hero card advertises per-category SGP contributions (`sgp_r`, `sgp_hr`, …), but neither the base pool nor the engine output carries those columns, so all ten chips read 0 and the `.sgp-row` div renders as a blank gap on **every** pick. A primary "why this player" feature is non-functional.
- **Tier — always "Tier 1."** `rec.get("tier", 1)` defaults to 1 because the engine never computes tiers. A 23rd-round flyer looks identical to a first-rounder.
- **Reason — always identical.** The engine never populates `reason`; every pick shows the same fallback string regardless of player, position, or situation.
- **The pick-score anomaly that destroys trust.** The captured top-10 by `pick_score` reads: Shohei Ohtani 21.35, Juan Soto 10.25, Jackson Chourio 9.60, **Jonah Cox 8.77 (ADP 999, consensus 1016)**, Wyatt Langford 8.65, **Aaron Judge 8.02 (ADP 1.0, consensus 92)**, **Wes Clarke 8.00 (ADP 999, consensus 2995)**, Ronald Acuña Jr. 7.59, Brent Rooker 7.42, Lars Nootbaar 7.18. Two unranked prospects (Cox, Clarke) outrank Aaron Judge *in HEATER's own headline metric* — exactly the kind of thing a novice notices in five seconds and never forgets.
- **Auto-computed SGP denominators** (reconstructed from the live DB) are plausible: R 18.07 / HR 8.13 / RBI 15.23 / SB 9.40 / AVG 0.008 / OBP 0.011 / W 1.49 / L 5.90 / SV 5.42 / K 31.63 / ERA 0.447 / WHIP 0.095.
- **Manual SGP inputs are missing OBP and L.** When auto-compute is toggled off, only 10 of the league's 12 categories get number inputs — **OBP and Losses are absent** — so a user calibrating by hand cannot fully specify their 12-category league.
- **Pre-flight checklist** (Step 2) reports real numbers: Player Pool 9,888, Hitters 6,139, Pitchers 3,749, Valuations computed (verified by `"pick_score" in pool.columns`).
- **Data Status expander mis-signals.** Its icon logic (`"Error" not in str(result)`) gives a green check to `no_data`, `partial`, and `cached` phases alike; at audit time 8 of 38 phases were non-success — including `projections: error "all FanGraphs fetches failed"`, `players: no_data`, and `season_stats: partial 2,580/8,054 rows` — yet nearly all showed green.

### UI/UX & visual-design critique

The three-column draft layout (roster + scarcity rings / hero card / pick entry + feed) is a reasonable instrument cluster for an actual draft session, and the hero card has good visual weight. But the wizard crams two unrelated concerns into one row — SGP denominators on the left, risk tolerance + engine mode on the right — and the "League Format" / "Player Pool" cards are pure display with no actionable meaning for a beginner who does not know what "SGP denominators" are. The hero card carries two full tiers of **dead information**: the always-identical reason line and the always-empty SGP chip row, both adding layout noise with no payload, and the `.p-meta` line stacks 7+ inline elements (position · tier · value/reach badge · injury badge · age flag · workload flag · injury prob) at real risk of overflow on narrow viewports.

Color and formatting carry several small violations. The survival gauge showing **red at 31% for the #1 overall pick** is mathematically defensible but visually alarming — a novice reads red as "danger" and hesitates on the top recommendation. The AVOID badge uses `T["primary"]` (orange, the CTA color) instead of `T["danger"]` (red), so the strongest negative signal renders in the same color as a primary button. The Category Balance tab formats WHIP as `.3f` instead of `.2f`, violating `format_stat` and the existing `test_pages_format_compliance.py` guard. The risk slider maps 11 stops to only 5 labels (dragging 0.5→0.6 shows no label change, which feels broken) and tops out at the unprofessional "YOLO." The browser tab has an empty favicon and a lowercase `page_title="Heater"`. The splash timer is the app's one remaining `st.components.v1.html` usage, emitting deprecation warnings past its 2026-06-01 removal date. The "Resume saved draft" toggle silently loads a stale 1-pick test state (`data/backups/draft_state.json`, `current_pick=1`) with no preview or warning, and the wizard has no UI to change `num_teams` / `num_rounds` / draft position — they are hardcoded defaults a non-standard league can never override. Finally, Step 2 re-runs the full 9,888-player `value_all_players()` pipeline on every Back→Next navigation, and the engine re-analyzes on every rerun when it is the user's turn — a 2–3-second blocking compute on each slider or toggle interaction.

### Key defects

- `[BLOCKER]` In-season users land on a dead draft-setup wizard with no in-season context or navigation (E-01).
- `[HIGH]` SGP category chips always empty; tier always "Tier 1"; reason always identical — three advertised hero-card features are non-functional (E-02/E-03/E-04).
- `[HIGH]` Pick-score anomaly: Jonah Cox / Wes Clarke (ADP 999) outrank Aaron Judge (ADP 1.0) in HEATER's own score (E-05).
- `[HIGH]` Manual SGP inputs missing OBP and L — the 12-category league cannot be fully tuned (E-06).
- `[MEDIUM]` Data Status icons show green for no_data/partial/error phases; survival gauge red at pick #1; `st.components.v1.html` deprecated; WHIP formatted `.3f` (E-08/E-09/E-07/E-12).

### Top recommendations

1. **Add in-season orientation to Home** — a contextual banner with the user's record, this week's matchup, and quick links to My Team / Free Agents / Lineup Optimizer; or make My Team the default landing page in-season (under `MULTI_USER`, flip the draft page's `default=True`).
2. **Fix the SGP category chips** — populate per-category SGP (even a simple `proj/denominator` computation from the session's `sgp_calc`) so the chip row is meaningful instead of a blank gap.
3. **Implement real tiers and pick reasons** — derive tiers from pick-score percentiles; generate 2–3 dynamic reason fragments per pick ("Best value by SGP | 31% ADP advantage | scarce catcher").
4. **Calibrate the valuation model** — floor or penalize ADP > 200 / consensus > 300 players so unknowns can't outrank stars; register the constant in `CONSTANTS_REGISTRY` with a `calibrate_sigmoid.py`-style tuner.
5. **Add OBP and L to the manual SGP inputs** so the 12-category league can be fully specified when auto-compute is off.
6. **Fix the Data Status icons** to read `get_refresh_log_snapshot().status` directly (success/cached → check, partial → amber, error/no_data → red).
7. **Fix the AVOID badge color** to `T["danger"]`, and **WHIP to `.2f`** in Category Balance, with a test assertion covering that render path.
8. **Replace `st.components.v1.html`** on the splash timer with `st.iframe` (one-line change, identical behavior).
9. **Conditionally hide the empty SGP-chip row** (or show a "category breakdown unavailable" placeholder) and **cache the Step-2 valuation pipeline** in session state so back-and-forth navigation doesn't recompute 9,888 players.
10. **Polish the risk slider** (5 discrete labeled stops, retire "YOLO" for "Speculative") and **set a real favicon + uppercase "HEATER"** page title; add a preview/warning to "Resume saved draft."

---

## Page 02 — My Team

**Source:** `pages/1_My_Team.py` (2,416 lines)

### Purpose & first impression

My Team is the app's primary daily-workflow page — "how is my team doing, what are my category gaps, and what do I do today?" It is enormous in surface area: an identity strip, a matchup ticker, six War Room cards, a weekly report, category-gap and totals cards, the roster table, Bayesian projections, and a news feed. On a wide monitor it feels rich and premium, exactly the intended impression. But the novice's first real question — *"what do I do today?"* — isn't answered until **Card 3 (Today's Actions)**, roughly 40% down the page, behind two analysis cards. And the two unlabeled buttons at the very top (`Refresh Stats` / `Sync Yahoo`) have no tooltip or subtitle to distinguish them — the single most confusing element in the first five seconds. Under `MULTI_USER`, both buttons are gated by `viewer_can_write()`, so non-admin leaguemates see an empty two-column gap above the identity strip with no explanation of why the controls are absent.

### Feature & output review

The data that renders is largely correct, but several features are silently degraded by the offline/cached state, and the audit confirmed the numbers from the live DB:

- **Identity Record / Rank show "—".** Team Hickey is **3-7-1, 10th of 12** (confirmed in `league_standings`: WINS=3, LOSSES=7, TIES=1, .318), but the identity strip only populates Record/Rank from the *live* Yahoo client, which is absent in cache mode — so both read **"—"** even though the data is sitting in the DB. The app looks broken for every non-OAuth session.
- **Matchup Pulse (Week 12, from cache):** **7-3-2** vs "The Good The Vlad The Ugly" — you 14R/7HR/14RBI vs 6/1/8, AVG .359 vs .271, OBP .431 vs .316 (all WIN), but L 2-0 (loss), ERA 7.08 vs 3.38 (loss), WHIP 1.82 vs 1.03 (loss), SB 0-0 and SV 0-0 (ties). Correct numbers — but **~4,012 minutes (≈2.8 days) stale**, rendered with **no staleness warning**, and the sibling card is literally titled **"Live Matchup."**
- **Statcast Signals + Regression Alerts cards vanish.** All 501 `statcast_archive` rows have NULL `xwoba`/`barrel_pct`/`hard_hit_pct`/`stuff_plus`, so the `if _sc_rows:` guard is never entered and **both intelligence cards silently do not render** — no header, no empty state, nothing. The user never learns these features exist.
- **Matchup ticker hidden.** `render_matchup_ticker()` gates on `st.session_state["yahoo_connected"]`, which is falsy in cache mode, so the quickest "what's my score?" element disappears entirely.
- **Hitting totals (season):** R 450 / HR 137 / RBI 410 / SB 64 / AVG .237 / OBP .332. **Pitching totals:** W 55 / L 37 / SV 23 / K 704 / ERA 3.57 / WHIP 1.24 / IP 692.3. Correct, but shown with **no league-rank context** — "HR 137" gives a novice no sense of whether that's 1st or 12th.
- **IL Alerts (correct):** Cal Raleigh (IL10), Bailey Ober (IL15), Garrett Crochet (IL60), Shane Bieber (IL60). But the **Bieber injury badge likely reads "Low Risk"** because his 0-games-played 2026 history produces a degenerate health score — contradicting the IL Alerts card directly below.
- **News is injury-only and grammatically broken.** Of 15 raw Yahoo entries, dedup leaves only 4 unique items (all IL placements already shown above); the fallback template produces *"Shane Bieber is listed as IL10 with a placed on il — elbow issue."* A **NaN sentiment score** bypasses the `if sentiment is None` guard, so the sentiment dot is silently absent on every Yahoo injury entry. The count summary reads "4 news items found: 4 injury."
- **Marcel projections format integers as floats** — a projected 15 HR shows as **"15.00."**
- **Lineup Validation card vanishes offline.** `get_todays_mlb_games()` returns an empty list with the network guard active, so `_teams_playing = 0` and the card — including its off-day-starter warnings — never renders, with no fallback message.

### UI/UX & visual-design critique

The information hierarchy is backwards for the daily user: analysis (Matchup Pulse, Flippable Categories) precedes action (Today's Actions, Lineup Validation). The roster table caption is **hardcoded** to "Full 2026 season totals. Updates hourly from MLB Stats API" — factually false on 2.5-day-old data. The "Last synced" pill shows a single `MAX(last_refresh)` timestamp for a multi-source pipeline (MLB Stats API vs Yahoo) with no way to tell which feed it refers to. The context cards are a uniform wall of monospace key-value text with no `.chip`/`.hero-num` accents and no rank context; the Priority Targets box is the only standout. The Weekly Report `st.expander` (native Streamlit) sits jarringly among the custom-HTML War Room panels, with a different border and expand affordance. The roster is reached through two redundant openers — `<a href="?player=ID">` links in every cell *and* a separate "Open player dossier" selectbox positioned among controls that look like table filters (Timeframe / Side). Two adjacent `st.segmented_control`s have overlapping scope but affect different parts of the page: "Stat source" drives only the context-card totals while "Timeframe" drives only the roster table — a novice changing "Stat source" reasonably expects the table to update. The 6-column category heat grid (`repeat(6,1fr)`) has no mobile breakpoint and will overflow on a phone. COLD streak tiles use `var(--fp-cold)` (`#5f7d9c`), barely distinguishable from the pale card surface. The Export-to-Excel button is small and buried below the roster with no call-out. And Today's Actions / Player Streaks are wrapped in `try/except: pass`, so any engine exception makes the card silently vanish with no spinner or empty state.

### Key defects

- `[BLOCKER]` Matchup ticker permanently hidden in cached/multi-user mode (gated on `yahoo_connected`) — the primary at-a-glance score element is missing.
- `[HIGH]` Statcast Signals + Regression Alerts silently absent (NULL Statcast); no empty state; two intelligence features invisible.
- `[HIGH]` Stale matchup (~4,012 min) shown with no warning across four cards; "Live Matchup" title is false; bad start/sit risk.
- `[HIGH]` Record/Rank show "—" in every cached session despite data being in `league_standings`.
- `[HIGH]` NaN sentiment bypasses the None guard; sentiment dot silently missing on all news items.
- `[HIGH]` Roster caption "Updates hourly" hardcoded and factually wrong.

### Top recommendations

1. **Add a staleness banner** when the matchup cache age exceeds 60 minutes (computed from `league_matchup_cache.updated_at`); rename "Live Matchup" → "Matchup — Week 12 (cached)."
2. **Populate Record/Rank from `league_standings`** when the live Yahoo client is absent — the data is already read for the standings card; just wire it to the identity strip fallback.
3. **Differentiate the two top buttons** with icons + one-line subtitles ("MLB API: live stats only" vs "Yahoo: rosters, matchup, FAs"), or collapse them into one "Refresh All" with an Advanced expander.
4. **Reorder the War Room** so Today's Actions is first, Matchup Pulse second, Flippable Categories third.
5. **Fix the NaN sentiment guard** with an explicit `math.isnan` check so the sentiment dot renders.
6. **Show empty states** for Statcast Signals and Regression Alerts via `render_empty_state(...)` instead of silently dropping them.
7. **Replace the hardcoded "Updates hourly" caption** with the computed source age from `refresh_log` (e.g. "Last updated 2 days ago").
8. **Render integer stats as integers** in the Marcel table (`f"{int(round(x))}"` for HR/RBI/SB/K).
9. **Un-gate the matchup ticker** from `yahoo_connected`; drive it from `yds.get_matchup()` (cache-backed) with a "(cached)" staleness hint.
10. **Add league-rank context** to the Hitting/Pitching Totals cards (a `#10/12` badge per category), and deep-link Flippable Categories suggestions to the relevant action page (streaming → Pitcher Streaming, waiver → Free Agents).

---

## Page 03 — Lineup Optimizer

**Source:** `pages/2_Line-up_Optimizer.py` (3,607 lines) + the 21-module `src/optimizer/` pipeline

### Purpose & first impression

The Lineup Optimizer sets a daily or weekly lineup for Team Hickey across three scopes (Today / Rest of Week / Rest of Season), plus a Start/Sit advisor, a Category Analysis tab, and a Streaming tab. It is the most complex page in the app and it feels it. A novice landing here meets a four-tab row and a left context rail stacked **6–8 cards deep** before any main content. The controls are abstract on arrival: a "Mode" radio (Full / Standard / Quick) with no explanation of whether it changes recommendations or just speed; a "Risk Aversion" slider sitting at 0.15 with no units; "Matchup State" and "IP Budget" cards that may be informational or may demand action. The header reads `FIG.02 — LINEUP CONTROL`, which to a beginner is jargon restating the page name.

### Feature & output review

The page's defining defect is silent and severe:

- **Category weights all collapse to 1.00× [BLOCKER].** `compute_nonlinear_weights()` is called with `team_name="Team Hickey"` (from the env `ADMIN_TEAM_NAME`), but the DB stores the team as `"🏆 Team Hickey"`. The bare string-equal SQL match finds no rows and silently returns **equal weights (1.00×) for all 12 categories.** The Category Analysis tab therefore tells the user "every category is equally important," and the LP optimizer loses its ability to weight by standings gaps — for a team **dominating** R/HR/RBI/AVG/OBP/W/K but **losing badly** in L (2-0), ERA (7.08 vs 3.38), and WHIP (1.82 vs 1.03) this week, the optimizer fails to prioritize ERA/WHIP by the 2–3× they deserve. Every optimization run this week is, quietly, wrong. The `resolve_viewer_team_name()` helper that fixes this exact emoji mismatch elsewhere was never wired into this call site.
- **DCV "Why?" expander** (reconstructed): `matchup_mult: 0.8745`, `volume_factor: 1.0`, `health_factor: 1.0`, `total_dcv: 12.34` — raw decimals with no scale, no direction, no sense of what "good" is.
- **DCV decision labels** include `START`, `START ⚠` (forced start when `matchup_mult < 0.70` or `total_dcv < median × 0.5`), `BENCH`, `IL` (Cal Raleigh, Bailey Ober, Garrett Crochet, Shane Bieber), and `LEAVE EMPTY` — the last two alarming and unexplained.
- **IP Budget card** shows a 53.85 IP target (`1400/26`) and a 20.0 floor — but is **blank until the first Optimize click** because `st.session_state["ip_projection"]` isn't pre-populated, so a novice sees a titled empty card on load.
- **Win Probability card** (Win 46% / Tie 19% / Loss 35%, projected 6-6) is the single most actionable number this week and it is the **last card in the rail**, below Mode, Risk Aversion, Matchup State, IP Budget, and Data Freshness.
- **Data Freshness** correctly surfaces the degraded state — `projections: ERROR` and `ros_projections: ERROR` (FanGraphs failed 2026-06-10), while `yahoo_standings` is fresh (2026-06-12) — but error and partial states share a near-identical amber indicator.
- **Start/Sit output** (reconstructed for an Olson vs Swanson dilemma) returns a clean shape — `start_score=72.4, floor=45, ceiling=92, confidence_label="Clear Start"` for Olson vs `41.3` for Swanson, with category impact "HR +0.34 SGP, RBI +0.28 SGP" — but the selectboxes filter toward hitters, so the same dilemma between two pitchers has no path.
- **Mode and Scope interact invisibly.** "Today" scope always uses the DCV engine regardless of Mode, while "Rest of Week" / "Rest of Season" run the LP pipeline with an auto-overridden `alpha` (1.0 for the week, computed for the season) — so a novice who sets Mode=Full + Scope=Today expecting the "best" daily answer doesn't realize Full mode does nothing there, and has no insight into the H2H-vs-season blending.
- **Streaming risk flags** render as bare strings (`HIGH_WHIP`, `SHORT_LEASH`, `ELITE_OFFENSE`, `HITTER_PARK`, `WIND_OUT`, `LOW_CONFIDENCE`) with no legend, and the whole Streaming tab duplicates the dedicated Pitcher Streaming page's content with no cross-reference.

### UI/UX & visual-design critique

The 1:4 context-to-main split front-loads **settings above context** — Mode and Risk Aversion appear before the Win Probability / Matchup State cards that should motivate them. The correct priority is Win Probability → Matchup State → IP Budget → Data Freshness → settings; the page does the reverse. The Optimizer tab before the first click is a **blank white area** with no `render_empty_state` and no instructional copy, and the Start/Sit selectboxes are empty if the user opens that tab before ever clicking Optimize (the roster is only loaded into session state by the optimize handler), again with no message. The Category Analysis roster table is a flat grid of decimals with no conditional coloring, so the user can't see at a glance who is helping vs hurting in ERA/WHIP, and there is no team-total row. Three `st.toast()` calls pass emoji icons (`'✅'`, `'⚠️'`, `'❌'`) and the Yahoo/LP mismatch banner uses `⚠` — both violations of the Combustion no-emoji rule. The Start/Sit advisor has **no pitcher path** (the selectboxes skew to hitters), so a manager choosing between two streaming SPs on a given day has no tool. The Matchup State card lists categories in raw league order with no win/loss/tie grouping, and the IP Budget card shows `53.85` with false-precision two decimals. On mobile the 1:4 columns stack, pushing the actual "Optimize Lineup" button far below the fold.

### Key defects

- `[BLOCKER]` Emoji team-name mismatch → all 12 category weights return 1.00× → wrong LP output and wrong Category Analysis for every run (B1).
- `[HIGH]` IP Budget card blank on first load (H1).
- `[HIGH]` Start/Sit selectboxes empty if the optimizer hasn't run; no message (H2).
- `[HIGH]` Start/Sit advisor cannot compare pitchers (H5).
- `[HIGH]` DCV "Why?" values shown with no scale or meaning (H6).
- `[HIGH]` `st.toast` and the LP-mismatch banner use emoji icons — Combustion-lock violation (H4).

### Top recommendations

1. **Fix the team-name lookup [BLOCKER]** — route `compute_nonlinear_weights()` through `resolve_viewer_team_name(rosters)` so category weights reflect real standings gaps; cover it with the existing structural guard.
2. **Move Win Probability to the top of the rail** and add a one-line week-record summary ("Week 12: projected 6-6").
3. **Add inline descriptions for Mode and Risk Aversion** — captions per Mode option (what each adds, how long) and endpoint labels for the slider ("0.0 = maximize upside, 0.50 = minimize downside").
4. **Show an empty state + instructions before the first Optimize click** via `render_empty_state("Ready to Optimize", …)`.
5. **Pre-populate the IP Budget card** on load by calling `compute_weekly_ip_projection(...)` in the page init block so it's never blank.
6. **Explain "START ⚠" and "LEAVE EMPTY"** with tooltips and follow-on guidance ("no suitable player for this slot — see Streaming or Free Agents").
7. **Add a stale-projection banner** at the top of the tab when `projections` has been in ERROR > 48h (FanGraphs down since 2026-06-10).
8. **Allow pitchers in Start/Sit** via a Hitters/Pitchers/All filter, with pitcher-relevant output columns (ERA impact, opposing wRC+).
9. **Color-code the Category Analysis table** (green for positive contribution, ember for negative) and add a team-total row per category.
10. **Replace emoji toasts/banners** with Material-symbol or text equivalents, render the Streaming risk flags as `.chip` tooltips with plain-English definitions, and simplify the Streaming tab to a "Quick Picks" summary that links to the full Pitcher Streaming page.

---

## Page 04 — Closer Monitor

**Source:** `pages/3_Closer_Monitor.py` + `src/closer_monitor.py`

### Purpose & first impression

The Closer Monitor is a 30-team bullpen depth-chart grid — who closes, who's the heir, how secure the job is — the page a manager scans for the Saves category. The render is genuinely handsome: a 5-column grid of instrument cards under a `BULLPEN / FIG.03 — SAVE DEPTH CHART` header with the "Closer Monitor." wordmark, each card carrying a team badge, headshot, heatbar, and stat trio. But the novice's first questions go unanswered: *"It says 30-team but I count 21 — where are the rest?"* (the caption "Showing 21 teams with closer data" doesn't say why), *"What does `51% JOB` mean?"*, *"Is the SV number what he has or what he's projected to get?"*, and *"Why does almost every card say `SETUP · —`?"*

### Feature & output review

- **21 of a promised 30 teams render.** Only 21 team codes have `depth_chart_role='closer'` in the DB; the `depth_charts` bootstrap phase returned `no_data` (6 days stale) and the MLB-API fallback tagged only 21 teams. **Nine teams with active closers are silently absent** — BAL/Ryan Helsley (7 SV), CIN/Emilio Pagán (6 SV), LAA/Jordan Romano (4 SV), WSH/Gus Varland (5 SV), plus ATH, MIN, CHC, COL, SF. A user who rosters Helsley sees nothing about Baltimore. The page's `render_reco_banner` still claims a **"30-team closer depth chart,"** and the per-team SV fallback heuristic never fires because it only triggers when the depth data is *completely* empty.
- **Two SV numbers, projected unlabeled.** The large orange figure is the **blended preseason projection**; the green "2026 ACTUAL" line is real YTD saves — and mid-season they diverge wildly: CLE/Cade Smith proj **7** vs actual **21**; SD/Mason Miller proj **6** vs actual **18**; TB/Bryan Baker proj **5** vs actual **18**; STL/Riley O'Brien proj **7** vs actual **17**. The card labels the projected column simply "SV" with no qualifier.
- **ARI / Paul Sewald shows `—` / `—` / `—`** in the main stat trio (then "2026 ACTUAL · 15 SV · 3.47 ERA · 0.73 WHIP" in green below) because `players.team` stores `'AZ'` for Sewald, the page normalizes the depth key `AZ→ARI`, and `build_closer_grid` then fails to match `team=='ARI'` against his pool row (`team='AZ'`). The green actual line still appears because `_load_actual_sv_stats()` matches by name only — a blank-then-populated contradiction on one card.
- **% JOB is jammed into 44%–68%** (TOR/Varland 44 to PHI/Duran 68) because `closer_confidence` starts at 0.75 for every named closer and the formula `0.6 × confidence + 0.4 × min(1, proj_sv/30)` uses **projected** SV. The heatbar's orange/steel threshold is 50%, so **~20 of 21 bars are identical orange** — the visual security signal never fires.
- **SETUP row is `—` on 19 of 21 cards** (zero `depth_chart_role='setup'` rows exist; only MIL/Abner Uribe and TOR/Jeff Hoffman show one, and only because those teams had two closer-tagged players). The **gmLI feature is entirely dead** (`closer_gmli_data` never populated; no `gmli_data` table) — every card renders the inert `<!-- no gmli data -->` comment.
- **The captured grid spans 44–68% JOB across 21 cards** — e.g. PHI/Jhoan Duran 68% (proj 13, actual 16 SV, 2.43 ERA), NYY/David Bednar 62% (proj 9, actual 13), ATL/Raisel Iglesias 63% (proj 10, actual 13), MIL/Trevor Megill 48%, HOU/Bryan King 48% (proj 1, actual 6), and TOR/Louis Varland 44% at the bottom. Several "closers" carry poor save context the card never flags (KC/Lucas Erceg has 12 actual SV but a 6.00 ERA on a sub-.500 team; PIT/Gregory Soto 8 SV on 2 projected), and the "2026 ACTUAL" line renders even bad ERAs in green, which reads as a positive signal when it merely means "real stats." Diacritic names (Andrés Muñoz, Seranthony Domínguez, Emilio Pagán) showed substitution characters in the DB audit, hinting at an encoding inconsistency worth a browser check.

### UI/UX & visual-design critique

The 5-column grid is tight at 1400px (each card ~260px) and confirmed name-truncation occurs ("Raisel Igl", "Paul Sew", "Seranthony Dom…"). Cards are sorted **alphabetically by team code** — useless for a manager who wants to scout shaky closers or their own roster's relevance first; nobody scouts bullpens alphabetically. There is **no search/filter/sort**, no roster cross-reference (no "MINE"/"FREE"/opponent badges despite `league_rosters` being available), no committee detection (the `compute_committee_risk()` function exists but is never called), and **no data-freshness indicator** despite the depth-chart data being 6 days stale — the info banner says "loaded at app launch" with no timestamp. Stat figures use `var(--font-display)` (Archivo) where the Combustion spec mandates `var(--font-mono)` for numerals. Each card fires an external MLB CDN headshot request (21 per render), and the heatbar conveys job security through color alone with no text alternative for color-blind users (a WCAG 1.4.1 failure). The `render_reco_banner` is even passed `icon_key="closer"`, which doesn't exist in `PAGE_ICONS`, so no icon renders. On mobile the 5-column grid collapses to a single column, rendering cards unusably small.

### Key defects

- `[BLOCKER]` ARI/Sewald card shows `—`/`—`/`—` for all stats (AZ→ARI mismatch) while showing "2026 ACTUAL · 15 SV" below — a direct contradiction.
- `[HIGH]` 9 teams with active closers missing entirely, no explanation; header still claims "30-team."
- `[HIGH]` The headline "SV" figure is the preseason projection, not actual saves; two SV numbers per card with no labeling.
- `[HIGH]` % JOB uses projected SV → misleading low scores (CLE/Smith 62% with 21 actual saves).
- `[HIGH]` SETUP row empty on 19/21 cards (no `setup` role data); gmLI feature entirely dead.

### Top recommendations

1. **Fix the AZ→ARI normalization** in `build_closer_grid` (normalize the pool's team column with the same `_TEAM_NORMALIZE` map before matching) so Sewald's stats render.
2. **Fill or explain the 9 missing teams** — make the SV-heuristic fallback apply *per team*, or render greyed placeholder cards with a "no closer data" empty state naming each absent team.
3. **Label the SV column "PROJ"** (or lead with actual SV mid-season) so the dual-SV layout is unambiguous.
4. **Incorporate actual YTD saves into % JOB** (blend `ytd_sv` into the security formula) so the score reflects real-world security.
5. **Sort by actionability** (saves / job security / "my roster first") with a sort control instead of alphabetical.
6. **Add "MINE / FREE / opponent" roster badges** by cross-referencing each card's `mlb_id` against `league_rosters`.
7. **Hide the SETUP row when empty** (or populate `depth_chart_role='setup'` from hold/leverage data) to remove the misleading "—" from 19 cards.
8. **Add a freshness timestamp** ("Depth chart last updated 2026-06-07 — 6 days ago") and rename "% JOB" to "JOB SECURITY" with a legend.
9. **Rescale the heatbar** so it varies meaningfully across the 44–68% band (or raise the "hot" threshold) and add a text label for color-blind users.
10. **Switch stat figures to `var(--font-mono)`** per the design spec, fix the `icon_key="closer"` to an existing key, surface `compute_committee_risk()` as a "COMMITTEE" badge for true-committee teams, add a text label to the heatbar for color-blind users, and lazy-load the 21 headshot images so a slow CDN doesn't stall the grid.

---

## Page 05 — Pitcher Streaming

**Source:** `pages/4_Pitcher_Streaming.py` + `src/optimizer/stream_analyzer.py`

### Purpose & first impression

A daily pitcher-streaming tool: for a chosen date it ranks available free-agent starters by a 0–100 "Stream Score," with a Matchup Microscope deep-dive, a 7-day Week Planner, and a Track Record back-test. The engine underneath is the strongest streaming analyzer in any fantasy product the persona has seen — six weighted components (matchup 0.35, SGP 0.25, form 0.15, lineup 0.10, environment 0.10, win probability 0.05), risk flags, two-start handling. But the page opens straight onto a **17-column data table** with no introduction. A novice's five-second reaction is, verbatim, *"What is a Stream Score? What does 56.10 mean? What are Net SGP, Conf, xIP, xER, W%? What do I click? Why are some rows grey?"* There is no entry point; the answer the user wants is buried in columns 7–17. The slate caption above it reads "Slate: 15 games, 28 probables posted, 20 matched to the player pool, N excluded as rostered."

### Feature & output review

- **Stream board row (live capture):** Kyle Leahy @ MIN — **Score 56.10**, **Net SGP -0.07**, Opp wRC+ 97, Opp K% 22.8, Park 1.01, Status PROBABLE, Conf HIGH. The board is sorted by Score descending; the top row above it was Zack Littell (WSH) vs SEA.
- **The score/SGP contradiction.** A score **above** the neutral 50 says "stream this pitcher," yet the **negative Net SGP (-0.07)** says adding him is projected to slightly *hurt* ERA/WHIP. The reconciliation (matchup, form, and environment outweigh the 25%-weighted SGP component) is invisible unless the user opens the "Why" expander — and a novice won't know to.
- **Score is data-state-sensitive and uncommunicated.** The live score of 56.10 could not be reproduced from current pool data (reconstruction gave 38.7, Net SGP -1.195), implying the live run had form/urgency context the audit lacked. With Yahoo offline and data ~1,219 min stale, the user has **no freshness badge** to tell them whether the scores are from this morning or yesterday afternoon.
- **17-column wall:** Pitcher / Tm / Opp / Status / Conf / GS / Score / Net SGP / Opp wRC+ / Opp K% / Park / xIP / xK / xER / W% / Own% / Risk — no legend, no tooltips, no decoding of the `x` (projected) prefix or `GS` (which here means *starts this week*, not the standard season "games started").
- **Cats in play (offline):** "no matchup data" — useless precisely when the user wants urgency context.
- **Degraded-state captions stack three deep:** matchup-impact ("No live weekly matchup in the cache…"), then swap ("Swap suggestions unavailable without live matchup context."), then another "unavailable" caption — making the tab look broken even though each message is individually correct.
- **Risk-flag thresholds** are precise but undocumented in-UI: HIGH_WHIP (career WHIP > 1.40), SHORT_LEASH (proj IP/start < 4.5), ELITE_OFFENSE (opp wRC+ ≥ 110), HITTER_PARK (park ≥ 1.08), WIND_OUT (outbound wind ≥ 12 mph, open-air only), LOW_CONFIDENCE (start 6+ days out).
- **`FIG.4` typo** — the one page in the app whose figure number isn't zero-padded (`FIG.4` vs the universal `FIG.NN`).
- **Per-start vs per-week label collision:** `xIP`/`xK` are per-start in the Finder board but per-week in the Week Planner, with identical headers (a two-start pitcher's doubled IP/K shows under the same `xIP` label).
- **The other three tabs add real value but inherit the same gaps.** The Matchup Microscope shows the opposing batting order (1–9 with handedness, or "No confirmed lineup posted… (lineups typically post 1-2 hours before first pitch)"), a 6-component breakdown, and a button-gated "Load game logs" that hits statsapi live with no TTL fallback (it can hang when the MLB API is degraded). The Week Planner honestly estimates its cost ("Click 'Build 7-day plan' to score the week's streamable starts (~10-60s)") and reports pacing against "54 IP target, 20 IP Yahoo forfeit floor," but silently skips any day whose schedule fetch fails. The Track Record's "My pitcher adds this season" table lists transactions as `name | timestamp | type` only (Team Hickey's real adds include Kyle Leahy, Bailey Ober, Andrew Alvarez, Eduardo Rodriguez) — answering "I added Leahy on May 3" but never "was it a good add," which is the tab's whole brand promise.

### UI/UX & visual-design critique

The single most useful output — the **Matchup Impact** table showing concrete category deltas like "K +12% ERA -8%" — is buried below an unlabeled `---` divider, while the dense scoring board leads; the board itself fragments into three unlabeled sections via two bare `st.markdown("---")` rules. The Score column should be first (or a Rank column added); locked/non-actionable rows aren't visually distinguished in `st.dataframe` (which can't do per-row color), so with `show_locked=True` as the default a novice can't tell which rows are actionable. The Track Record's crucial "replay uses *current* projections, not historical" proxy caveat appears **after** the user clicks Run — too late to calibrate expectations. The Microscope's "Matchup (0-10)" sub-score is on a different scale than the 0-100 Stream Score with no reconciliation, and its component table shows machine labels (`sgp`, `env`, `winprob`) with no descriptions. The "Load game logs" result isn't session-cached and is lost on any tab switch, unlike the swap and week-plan caches. Weather data is stale enough (newest 2026-06-01) that the WIND_OUT flag silently never fires. And there is no page-level data-freshness badge despite the `DataFreshnessTracker` module being available and used on the Lineup Optimizer.

### Key defects

- `[HIGH]` 17-column board with no legend, tooltips, or intro — the primary novice confusion point (E-02).
- `[HIGH]` Negative Net SGP (-0.07) on a pitcher scored above neutral (56.10), unexplained (E-03).
- `[HIGH]` Track Record proxy-caveat shown after the Run click — wrong moment (E-10).
- `[HIGH]` No page-level data-freshness indicator despite ~1,219-min-stale projections (E-22).
- `[MEDIUM]` "Cats in play" shows "no matchup data" offline with no cache fallback; confidence tiers + risk flags unexplained; `xIP/xK` label collision between tabs; three stacked "unavailable" captions; Microscope scale ambiguity.
- `[POLISH]` `FIG.4` not zero-padded (E-01).

### Top recommendations

1. **Add a one-sentence intro + a "Top pick today" callout card** above the board (pitcher, opponent, score, ownership) so the answer is visible without reading the table.
2. **Move Score to column 1** (or add a Rank column as column 1, score as column 2; or render the top 3 as visual tiles).
3. **Rename columns to plain English with `column_config` tooltips** (`GS`→"Starts", `Net SGP`→"SGP delta", `Opp wRC+`→"Opp offense", `Conf`→"Reliability", `xIP/xK/xER`→"proj IP/K/ER").
4. **Visually distinguish actionable vs locked rows** (default `show_locked=False` with a "Show completed games" reveal toggle, or split into two tables).
5. **Fix `FIG.4` → `FIG.04`.**
6. **Promote the Matchup Impact table** above the board when matchup data exists; replace the multi-sentence offline explanation with a single-line status.
7. **Add a data-freshness badge** consistent with the Lineup Optimizer's `DataFreshnessTracker`.
8. **Add a legend/tooltip for the risk flags** with their exact thresholds.
9. **Session-cache the game-log fetch** keyed on `mlb_id` so it survives tab switches.
10. **Move the Track Record proxy caveat before the Run button**, relabel Week Planner `xIP`/`xK` as weekly totals, and fall back "Cats in play" to standings-derived urgency when Yahoo is offline.

---

## Page 06 — Matchup Planner

**Source:** `pages/5_Matchup_Planner.py` + `src/matchup_planner.py`, `src/standings_engine.py`

### Purpose & first impression

The Matchup Planner answers the weekly question: *"How likely am I to win, and who should I start?"* It opens well — a clear `THIS WEEK / FIG.05 — MATCHUP GRID` header, a win-probability banner, and a tri-color win/tie/loss bar with the headline **46%** immediately visible. Then confusion sets in fast: a metric card reads **"Smash Matchups: 4"** (a novice has no idea what "Smash" is without the below-the-fold legend), the "Average Games" metric shows **"0.00"** when the schedule is offline (looking broken), and five tabs (`Category Probabilities`, `Player Matchups`, `Per-Game Detail`, `Hitters Only`, `Pitchers Only`) sprawl across a page where the user just wants "who do I start?"

### Feature & output review

- **Win probability:** Win **46%** / Tie **19%** / Loss **35%**, projected 7-5 — but the **banner says "projected 6-6"** (computed pre-MC, from a simpler preliminary estimate) while the card says 7-5, a visible disagreement.
- **Per-category win probabilities (reconstructed):** K **93%** (Hickey weekly 60.5 vs opp 46.9), W **81%** (3.3 vs 2.4), OBP 58%, SV 58%, SB 57%, HR 54%, R 52%, RBI 51%, WHIP 51%, ERA 49%, AVG 38%, **L 14%** (2.8 vs 1.8). The read is clear: Team Hickey wins K/W convincingly (deep pitching) and is dragged in AVG (IL hitters lowering the team average) and L (14 pitchers generating losses). The overall 46% comes from a correlated Gaussian-copula MC (HR-R correlation 0.72, AVG-OBP 0.85).
- **The dead `schedule_warning` [BLOCKER].** When the MLB schedule is offline (as it is), the page builds a `schedule_warning` string (line 666) explaining the situation — and **never displays it**. Instead every player renders as **"Rating 1.00 / Avoid / 0 games,"** a 27-row sea of "Avoid" with zero explanation. A novice concludes every player is terrible or the page is broken.
- **Week navigator capped at 24.** Line 475 hardcodes `< 24`, but `LeagueConfig.season_weeks = 26` — the `▶` button **silently refuses Weeks 25–26.**
- **`FIG.02` caption on a `FIG.05` page** — the Category Outlook panel passes `fig_label="FIG.02 · WIN PROBABILITY BY CAT"` (line 792), a stale within-page number that collides with the page's own `FIG.05`.
- **"Tie 19%"** is mislabeled — the engine's own docstring notes per-category ties are ~0 in continuous simulation; the 19% is actually P(exactly 6-6 categories), which a novice will misread as a drawn matchup.
- **Per-Game Detail** shows each player's individual game cards (date / vs OPP / Home|Away / PF 1.00 / Score 0.00) with raw internal values and no interpretation; park-adjusted projected stats (HR/R/RBI/SB) render for hitters only — pitchers get no adjusted ERA/WHIP/K — and the underlying `_projected_hitter_adjusted` hardcodes `season_games = 140.0`, inconsistent with the locked `_HITTER_GAMES_PER_SEASON = 145.0`. When the schedule is offline every expander reads "No games scheduled" with the "Score 0.00 / PF 1.00" defaults rendered as literal values.
- **Opponent Weakness card** pulls from `MatchupContextService.get_opponent_context()` and lists up to four weak and four strong opponent categories — but when standings data is unavailable it silently falls back to `{"name": "Unknown"}` and the card simply doesn't render, with no user notification. The recommendation banner is also clipped in the live render (observed as `46% cha[nce]...`), so the toss-up and best-odds detail it assembles is cut off.

### UI/UX & visual-design critique

The left context rail runs **9 elements deep** (Week nav → Opponent → Win Prob → Days ahead → Player type → Team selector → Team card → Rating Tiers → Opponent Weakness), pushing the Rating Tiers legend far below the fold. The "Smash / Favorable / Neutral / Unfavorable / Avoid" tier names read as jargon (plain "Elite / Good / Average / Weak / Skip" would be clearer). The five-tab structure is redundant: `Hitters Only` / `Pitchers Only` duplicate `Player Matchups` and **conflict** with the sidebar `Player type` radio — select "Hitters" in the radio, open "Pitchers Only," and the tab is empty with no explanation, because the radio pre-filters the roster before computation while the tabs filter after. Per-category confidence is encoded only as opacity (high=1.0, medium=0.7, low=0.5) with no visible label, so early-season fading is unexplained, and the category heat-bar columns (55px name / flex bar / 46px pct / 130px projections) feel cramped at standard zoom. The tier badges inline white text on backgrounds like `T["tx2"]` (mid-grey "neutral") and `T["sky"]` (blue "favorable") that risk failing WCAG AA contrast. The 10,000-sim MC runs **synchronously and uncached** (`_compute_win_probs` has no `@st.cache_data`) on every page load with no spinner, and the "Fetching MLB schedule…" spinner flashes even when the schedule is cached (the `show_spinner=False` cache decorator doesn't suppress the outer `st.spinner`). The tier legend is also duplicated — once in the context rail and again at the bottom of the main panel — and the bottom copy (which carries the actual percentile cutoffs) sits below the fold. There is no data-freshness indicator, no action guidance after the probabilities, and the win-probability hero numeral is a comparatively small 34px. The past-week empty state promises "Yahoo live data when available" but never renders actual past-week results — a false promise.

### Key defects

- `[BLOCKER]` `schedule_warning` built but never displayed — every player shows "Avoid / 0 games" offline with no explanation.
- `[HIGH]` Week navigator capped at 24 of a 26-week season — silently broken.
- `[HIGH]` `FIG.02` caption on a `FIG.05` page — stale copy-paste, visible inconsistency.
- `[HIGH]` No data-freshness indicator.
- `[HIGH]` 10,000-sim MC runs synchronously, uncached, on every load.
- `[MEDIUM]` "Smash" jargon; "Average Games 0.00" offline; banner/card projected-score mismatch (6-6 vs 7-5); redundant Hitters/Pitchers tabs; "Tie 19%" mislabeled; `season_games=140` constant mismatch.

### Top recommendations

1. **Display `schedule_warning` [BLOCKER]** and stop rendering useless 0-game ratings when the schedule is offline — show a `render_empty_state("Schedule data unavailable", …)` and render only the Category Probabilities tab.
2. **Fix the week-navigator cap** to `config.season_weeks` (26).
3. **Remove the stale `FIG.02` sub-caption** (drop the number from the panel label).
4. **Rename "Smash Matchups"** (or add a `help=` tooltip "top 20% of matchup ratings this week") and consider plain-English tier names.
5. **Add a data-freshness indicator** ("Roster data as of … (cached)").
6. **Cache the MC win-probability computation** (`@st.cache_data` keyed on week + team + roster hash) or drop sims to ~2,000.
7. **Add a spinner** around the MC computation inside the tab render block.
8. **Collapse the five tabs to three** (drop Hitters/Pitchers Only; add an inline hitter/pitcher toggle to Player Matchups).
9. **Reconcile the banner vs card projected score** so both read the post-MC result.
10. **Relabel or explain the "Tie" probability** ("6-6 Draw: 19%" with a `help=` note), add a "this week: focus on…" action line below the category bars (e.g. "K and W already won — protect ERA/WHIP by limiting bad-matchup starts; L is at risk, consider streaming a reliever"), consolidate the duplicate tier legend into the more-informative bottom copy, register `season_games` in `CONSTANTS_REGISTRY` and import the locked 145.0 value instead of hardcoding 140.0, and render a real `render_empty_state` for past weeks rather than a promise the page never keeps.

---

## Page 07 — League Standings

**Source:** `pages/6_League_Standings.py` + `src/standings_engine.py`, `src/playoff_sim.py`

### Purpose & first impression

The league-awareness hub: where you stand, your category strengths and weaknesses, where you'll likely finish, and your playoff path. It opens strong — a `LEAGUE / FIG.06 — STANDINGS BOARD` header, an orange ticker ("This week vs The Good The Vlad The Ugly: leading 7-3-2 in categories"), and a left rail showing the user's position (**10th**), record (**3-7-1 (.318)**), and games-back metrics (7 GB from 1st, 2 GB from 4th). Genuinely useful. But within five seconds the novice hits unlabeled jargon: *"What does GB mean? What does .318 mean — percent or decimal? And what is this wall of colored numbered badges below the record table?"* The Category Standings grid — the most important element for an H2H player — is a 12×12 field of rank badges with **no legend anywhere.**

### Feature & output review

- **H2H table (correct, full 12 teams):** 1. Over the Rembow 10-1 (.909); 2. My Precious 7-3-1 (.682); 3. HUMAN INTELLIGENCE 6-4-1; 4. BUBBA CROSBY 5-4-2; 5. On a Twosday 6-5-0; … 9. The Good The Vlad The Ugly 4-6-1; **10. 🏆 Team Hickey 3-7-1 (.318)**; 11. Cyrus The Greats 2-7-2; 12. Going…Going…Gonorrhea 2-8-1. A dashed red playoff-cutoff line correctly sits between ranks 4 and 5.
- **Streak column is empty for all 12 teams** — `streak = ''` in `league_records`; the chip CSS and rendering logic exist but the bootstrap never populates the data. A permanently blank column.
- **Win% reads as a batting average.** `format_stat(win_pct, "AVG")` produces `.318` (three decimals, no leading zero, no `%`) under a "Win%" header — a baseball fan reads ".318" as an average, not a win percentage. The context card also uses an inline `f"{win_pct:.3f}"` (line 361) rather than `format_stat`.
- **Category grid (Team Hickey, reconstructed with actual totals):** HR rank **2** (118 HR, green), W **4** (50, green), K **2** (715, green); R 5 (376), RBI 6 (345), OBP 6 (.332), SV 5 (24), ERA 5 (3.79), WHIP 8 (1.27); SB **12** (40, red), AVG **12** (.236, red), L **11** (47, red). The colors map meaningfully to playoff status — *if* you already know the semantics. No actual stat values accompany the ranks (rank 12 in AVG gives no sense of whether that's .220 or .239), and there's no playoff-cutoff line in the grid (unlike the H2H table).
- **Season Projections** auto-runs a 500-sim MC and shows **Magic#** and **SOS** columns — both unexplained jargon (SOS shown as `0.517` with no scale). The Team Strength Profiles expander adds Power / Roster / Balance / SOS / Injury / Momentum / CI, all abstract floats (e.g. Power "74.3", Roster "0.847", CI "8.2-14.1") with no scale.
- **Scenario Explorer mislabels the inputs.** "What if you go ___-___-___ this week?" with W/L/T inputs that must sum to 12 — these are **category-wins within one matchup**, not H2H record wins, but the labeling strongly implies the latter (default 6-6-0). A novice will type "1" in W meaning "I expect to win 1 matchup."
- **Duplicate playoff odds:** Tab 2 (auto-sim heatbar via `simulate_season_enhanced`) and Tab 3 (manual via `playoff_sim.simulate_season`, 200/500/1000/2000 sim options) show the same data from **different engines** with no explanation of which to trust.
- **Playoff Odds (Tab 3)** opens with a clean `render_empty_state("No simulation yet", …)`, then after the user clicks Simulate Season renders a large color-coded probability ("Strong/Moderate/At Risk"), a projected-standings `st.dataframe` (Team / Avg Wins / Avg Losses / Playoff % / Most Likely Finish, user row tinted), and a "Your Finish Distribution" table (rows under 0.5% hidden) — a genuinely strong surface that simply duplicates Tab 2's heatbar from a different engine. Its panel even carries a within-page `FIG.03 · PLAYOFF ODDS` label on a page that is globally `FIG.06`.
- **The banner's "close battles" clause rarely fires.** `close_battle_categories` is built but the Week 12 matchup's only near-ties are the literal 0-0 ties in SB and SV, which the "margin < 0.001" logic doesn't treat as close — so the most actionable "one play flips this" cue is suppressed exactly when it would matter.
- **`current_week` off-by-one:** `int(weeks_played)` yields 11 when the league is in Week 12, biasing projections optimistically.

### UI/UX & visual-design critique

The instrument panel is clean and the IBM Plex Mono rank numbers look sharp, but the category grid is "impressive once understood, impenetrable cold." Long team names truncate in the live render ("HUMAN INT…", "BUBBA CRO…", "The Good The Vlad The…"), and the THIS WEEK card's 26-character opponent name overflows the narrow left rail. The banner uses only its static teaser (`expanded_html=""`), leaving "2 GB from playoffs" a dead-end observation with no next step — and the design system's `render_reco_banner` explicitly supports expandable advice that this page declines to use. The ghost team "Twigs" is correctly filtered from the UI here (good), though the stale standings row should be purged in the next bootstrap. The within-page panels reuse `FIG.01/02/03` labels that collide with the global `FIG.06` and with other pages' figure numbers. Playoff probability precision is inconsistent (the Tab-2 heatbar shows one decimal while the Projected Final Standings "Playoff%" shows an integer "74%"). No data-freshness "as of" timestamp accompanies the 64-hour-stale matchup banner, and Tab 2 auto-runs its 500-sim MC on first visit with no opt-out.

### Key defects

- `[HIGH]` Category Standings grid has no legend — a novice cannot read the colored 1–12 badges.
- `[HIGH]` Streak column always empty (dead column).
- `[HIGH]` Win% renders as ".318" — reads as a batting average, not a win percentage.
- `[HIGH]` Scenario Explorer W/L/T inputs mislabeled (category-wins, not record-wins); sum-to-12 constraint unexplained.
- `[MEDIUM]` Magic#/SOS/Power/Roster/Balance/Momentum/CI jargon unexplained; duplicate playoff-odds engines; `current_week` off-by-one; no playoff cutoff line in the category grid; 64h-stale matchup with no "as of" timestamp.

### Top recommendations

1. **Add a legend to the Category Standings grid** ("Rank 1 = best; green = top 4, blue = middle, red = bottom 4") plus per-badge `title` tooltips showing the actual value ("AVG: .236, rank 12 of 12").
2. **Show actual category stat totals** alongside the ranks (a small numeric label below each badge).
3. **Populate or remove the Streak column** (audit why Yahoo streak data isn't stored).
4. **Fix the Win% format** to "31.8%" and replace the inline `f"{win_pct:.3f}"` with `format_stat`.
5. **Rewrite the Scenario Explorer** with explicit sub-labels ("Categories Won/Lost/Tied") and a clear sum-to-12 explanation.
6. **Explain Magic#, SOS, Power, Roster, Balance, Momentum, CI** via `help=` parameters or captions.
7. **Add a playoff cutoff line to the category grid** (between the 4th- and 5th-ranked teams by H2H record).
8. **Differentiate or consolidate the two playoff-odds surfaces** ("Quick estimate" vs "Full simulation with your roster") or merge Tab 3 into Tab 2.
9. **Add action guidance** via the banner's `expanded_html` ("you're 2 back; weakest cats are SB/AVG — check Free Agents").
10. **Fix the `current_week` off-by-one** (round/ceil `weeks_played`) and add per-panel data timestamps.

---

## Page 08 — Punt Analyzer

**Source:** `pages/10_Punt_Analyzer.py` + `src/valuation.py`, `src/standings_utils.py`

### Purpose & first impression

The Punt Analyzer simulates conceding a scoring category — zero out a category's weight and see who gains or loses relevance, and how the team's standings profile shifts. In a H2H-categories league this is one of the highest-leverage strategic decisions a manager makes all season. The page is also the **most under-built relative to its importance.** It is sparse and entirely gated behind a multiselect: until you pick a category, you see a header, one sentence, a dropdown, and a blue info box. Critically, the word **"punt" is used in the title, instruction, and chip labels but never once defined** — a novice who hasn't read fantasy strategy has no idea what the tool does or when to use it. The nav says "Punt Analyzer"; the H1 says "Punt Strategy Simulator." — a naming mismatch on top of everything. The control surface is notably thin: no tabs, no roster-specific view, no position filter, no proactive suggestions, no export.

### Feature & output review

- **The "Biggest Value Gainers" table is broken for counting-stat punts [BLOCKER].** Under Punt SB, the 15 "biggest gainers" — Tarik Skubal (+5.58→+5.58), Paul Skenes (+4.10→+4.10), Garrett Crochet (+4.30→+4.30), Junior Caminero (+4.34→+4.34), Ketel Marte (+2.92→+2.92), … — **all show Change +0.00.** They're pitchers and non-speed hitters whose SB projection is already 0, so removing SB changes nothing. The table picks the top 15 by `value_change` via `nlargest`, gets 15 rows tied at exactly zero, and presents **fifteen players who experienced no change** under the headline "Players whose value increases most." Beautiful presentation of meaningless data. The same failure occurs for SV, HR, R, RBI, W, K. (For an inverse stat like Punt L the table *does* work — Aaron Brooks RP goes -45.56 → -19.56, +26.00 — so the logic is correct for stats players contribute positively to, but silently breaks for the common case.)
- **The Losers table is correct and useful** — under Punt SB it surfaces speed specialists losing value: Tyler Tolbert +4.24→+1.52 (-2.71), Brewer Hicklen +8.01→+5.79 (-2.21), Bobby Witt Jr. +5.08→+3.66 (-1.43), José Ramírez -1.36, Shohei Ohtani -1.29, Ronald Acuña Jr. -1.21. But its subtext labels these "potential trade targets," which is backwards — these are **trade bait** to offer teams that still value SB.
- **Ghost team "Twigs" inflates ranks to /13 [HIGH].** The page builds `all_team_totals` directly from `league_standings` without calling `filter_standings_to_valid_teams()`, so the renamed/abandoned "Twigs" (with full season stats) is included. Result in the Standings Impact panel: HR rank **1/13**, SB **11/13**, AVG **12/13** — while the UI label hardcodes "/12." Inconsistent and wrong.
- **Standings-points formula hardcodes 13.** `sum(13 - rank ...)` assumes exactly 12 teams; with the ghost team a rank-13 entry earns 0 points (should be 1). It should be `config.num_teams + 1 - rank`.
- **Standings points (Punt SB): 78** (help text "Total 80, active 78, punted 2") — shown with no denominator or benchmark (max is 144, average 78), and buried at the very bottom of the page with no before/after delta.
- **413× slower than necessary [HIGH].** The page calls `total_sgp()` row-by-row in `pool.iterrows()` across 9,888 players, **twice** — measured at ~2.1 seconds — while the vectorized `total_sgp_batch()` already in `SGPCalculator` does the same in ~0.003s per pass. The 20→50→100% progress bar can't even update incrementally because the blocking compute holds the Python call.
- **The Strategy Summary panel confirms the selection but never interprets it.** Under Punt SB it reports stat readouts "Punted: 1 / Active: 11 / Categories: 12" with an SB chip in cold blue and the remaining eleven in orange — but no line like "Team Hickey ranks 11th in SB (40 steals); punting frees value for…". A multi-category punt (e.g. SB + SV together) compounds the gainers-table bug: both purely-counting stats produce the same all-zero top-15 while the losers table correctly stacks the closer and speedster losses. The inverse-stat punt (L) is the one case where the panel and gainers table fully cohere — pitchers with many losses (Aaron Brooks RP -45.56→-19.56, Mike Clevinger -6.27→-2.27) climb — yet even there the user sees players with *negative* overall SGP labeled "gainers," which needs a one-line explanation.

### UI/UX & visual-design critique

The page leads with controls and buries the one actionable summary (standings points) at the bottom — a strategy tool with no summary-first verdict. The `_value_swing_table_html` itself is well-crafted (team-color accent, headshots, logos, Archivo headers, IBM Plex Mono figures) — which makes the all-zero gainers table worse, not better: beautiful presentation of meaningless data is more damaging than none. The empty state uses a plain `st.info` instead of `render_empty_state()`, and the multiselect uses Streamlit's default "Choose options" placeholder with no hitting/pitching grouping and no inverse-stat hint (a novice could punt ERA without realizing lower is better). There is no roster-specific view (the gainers/losers show the full 9,888-player pool, so the user can't see "which of MY players lose value if I punt SB"), no proactive punt suggestion based on the user's actual weak categories (AVG 12th, L 11th, SB 11th are obvious candidates the page never surfaces), and no data-freshness indicator. The four panels stack in a single column forcing excessive scrolling, and the Standings Impact section silently disappears if `_HAS_CATEGORY_ANALYSIS` import fails. On mobile the headshot+name cell of the swing table overflows with no responsive adjustment.

### Key defects

- `[BLOCKER]` Gainers table shows 15 players with +0.00 change for any counting-stat punt (SB, SV, HR, R, RBI, W, K).
- `[HIGH]` Ghost team "Twigs" inflates ranks to /13; UI hardcodes /12.
- `[HIGH]` Standings-points formula hardcodes 13; wrong when n_teams ≠ 12.
- `[HIGH]` Row-by-row SGP compute 413× slower than the available vectorized path (~2.1s blocking).
- `[HIGH]` "Punt" never defined; no roster-specific view.

### Top recommendations

1. **Fix the gainers table [BLOCKER]** — sort by relative rank gain (pre- vs post-punt pool rank), or filter to `value_change > 0` and show an honest empty state ("no players gain absolute value — speed specialists lose value, everyone else stays the same").
2. **Filter out "Twigs"** via `filter_standings_to_valid_teams()` (already implemented in `standings_utils.py`).
3. **Fix the standings-points formula** to `config.num_teams + 1 - rank`.
4. **Switch to `total_sgp_batch()`** (413× faster) and drop the now-pointless progress bar.
5. **Add a "What is punting?" explainer** (a collapsed expander with a three-sentence definition).
6. **Add a roster-specific view** (My Roster / Free Agents / All toggle) so the user sees which of *their* players shift.
7. **Fix the "trade targets" label** to "trade bait," with a link to Trade Finder.
8. **Surface proactive punt suggestions** from the user's weakest categories ("Based on your standings: AVG rank 12, L rank 11, SB rank 11 — click to load as punt candidates").
9. **Promote the standings-points metric** to the top of the results with a before/after delta and a denominator ("78 of 132 possible active points").
10. **Resolve the nav-label / page-title naming drift** ("Punt Analyzer"), replace `st.info` with `render_empty_state()`, and lay gainers/losers side-by-side with `st.columns(2)`.

---

## Page 09 — Trade Analyzer

**Source:** `pages/11_Trade_Analyzer.py` (1,331 lines) + `src/engine/output/trade_evaluator.py`, `src/trade_intelligence.py`

### Purpose & first impression

The Trade Analyzer is the most technically ambitious page in HEATER: propose a trade (give vs receive) and a six-phase engine returns a letter grade, ACCEPT/DECLINE verdict, confidence %, per-category impact, Monte Carlo risk metrics, playoff-probability shift, a weekly H2H matrix, an acceptance estimate, and a battery of secondary diagnostics. It is also where the gap between engine sophistication and user experience is widest. The page takes **~4.9 seconds just to show the controls** (a 4.32s uncached `load_player_pool()` + a 0.52s `get_health_adjusted_pool()` on every rerun), and clicking "Analyze Trade" is where the experience breaks: the engine runs **44+ seconds** and the page **freezes**. When results finally return, they dump ~20 distinct technical sections — CARA utility, Var[Δchamp], Rosenof G-score, VORP/PRP delta, reshuffle pct — with no guidance on what to read first.

### Feature & output review

For a real test trade (give **Dansby Swanson** → receive **Bryan Reynolds**), the audit captured the full engine output:

- **Phase 1 verdict (MC off):** Grade **C**, **DECLINE**, surplus SGP **-0.086**, confidence **43.1%**, `evaluate_trade()` wall time **2.44s**. Category impact: R -0.562, HR -0.077, RBI -0.562, SB -0.286, AVG +0.406, OBP +1.146, all six pitching cats 0.000. This part is correct and fast. Secondary diagnostics: `delta_vorp_prp` +3.392 (fair league-wide), `delta_g_score` -2.086 (variance-adjusted DECLINE).
- **The grade range is computed but never shown [HIGH].** The engine produces `grade_range = {grade: C, grade_low: D, grade_high: B+, confidence: low}` — arguably the most honest signal it has (this trade could grade anywhere D to B+) — and the page **silently drops it**, showing only the point estimate "C."
- **The 2,188% reshuffle warning [HIGH].** Because surplus is near-zero (-0.086) and the LP reshuffle effect is substantial (1.885 SGP), the percentage blows up: *"Lineup reshuffle accounts for 2188% of surplus — promoting Mike Trout from bench. You may capture most of this value by setting your lineup, without making the trade."* Technically correct, completely uninterpretable, and the message contains a Unicode mojibake artifact from an unsanitized emoji.
- **Monte Carlo at production settings [BLOCKER].** `enable_mc=True` with **10,000 sims** ran **44.77 seconds** synchronously in the button handler — returning MC mean -2.184, std 3.166, CVaR5 -8.879, P(trade helps) 24%, and **`convergence_quality: "marginal"`** even at 10K sims (so the displayed 4-decimal CVaR is built on unstable bands), yet the convergence risk flag lives in a separate `mc_result["risk_flags"]` list the page never merges into `result["risk_flags"]`. Add the playoff sim and the total is **45–70+ seconds**. This **dropped the browser WebSocket and froze the page in live testing.**
- **The MC tooltip is 50× wrong** — `METRIC_TOOLTIPS["mc_mean"]` and `["trade_verdict_legacy"]` both say **"200 simulated seasons"** while the engine runs 10,000.
- **The IP-floor warning is uninterpretable** — *"Weekly IP (0.9) already below the 20 IP/week floor"* fires on essentially every trade because stale full-season IP projections have been exceeded by YTD actuals (only Valdez 1.16, Iglesias 0.80, Bieber 1.20, Alvarez 2.67/wk carry non-zero ROS IP), so ROS IP collapses to ~0.87/wk. The real cause (stale projections) is never surfaced.
- **Internal engineering labels leak to users** — "Feature 2 — Weekly H2H impact", "Primary Objective — Δ Title Odds", "report B.9", "report B.10 + Q(a)", "report C.5/H.9" all appear in user-facing metric labels and tooltips.
- **Acceptance probability uses a magic number** — `opp_sgp = -user_sgp * 0.5`, presented with false precision ("Acceptance Probability: 54%"); for this trade that yields `opp_sgp = 0.043` and `need_match = 0.52`, half the calculation a fabricated guess.
- **Category Impact looks half-broken to a novice.** For this hitter-for-hitter trade the table's six pitching rows all read 0.000, which a beginner reads as "the engine skipped pitching." The Upside/Downside Risk table compounds it: it shows only "10th-Percentile Home Runs (Floor)" and "90th-Percentile Batting Average (Ceiling)" — extremely long, wrap-prone headers — so for a pitcher the table is empty or zeroed, and HR/AVG are an arbitrary stand-in where ERA/K would matter more. The "You Give" roster correctly lists all 27 players (Bregman, Giménez, Olson, Trout, the 14 pitchers, etc.), but IL stashes appear with no badge.

### UI/UX & visual-design critique

The verdict ordering is upside-down: the playoff-sim section ("Primary Objective", 6+ metrics, heat bars, CARA utility) renders **before** the main ACCEPT/DECLINE banner — a novice sees graduate-level risk metrics before learning whether the trade is good. The output is 20+ sections with no visual grouping hierarchy and no "what this means for you" layer; CARA utility (λ=0.15), CVaR₂₀, Var[Δchamp], Δ VORP, and Δ G-score are finance-theory metrics a novice manager cannot use, each defined only in a hover tooltip. The verdict banner carries a `slideUp` entrance animation (Combustion-lock violation — it replays on every rerun) and uses `T["sky"]` for B-grades (not a palette tier color). IL-60 players (Crochet, Bieber) appear in the "You Give" dropdown with no status indicator — a user could propose a trade Yahoo will reject. The `📅` emoji in the Weekly H2H expander label violates the no-emoji rule, and the Acceptance Analysis section uses a bare `st.subheader()` instead of the page's orange-rail section header. Several computed engine outputs (`grade_range`, `reshuffle_sgp`, `concentration_delta`, `flexibility_penalty`, `convergence_quality`) are produced at cost and discarded. The data-freshness signal lives only in the sidebar card, not beside the dropdowns that consume the possibly-stale rosters.

### Key defects

- `[BLOCKER]` 10K-sim MC (+ playoff sim) runs synchronously, takes 45+s, drops the WebSocket — confirmed page freeze.
- `[BLOCKER]` ~5-second blank page on every navigation (no cache on `load_player_pool`/`get_health_adjusted_pool`).
- `[HIGH]` Grade range (D to B+, low confidence) computed but never shown.
- `[HIGH]` Verdict ordering — playoff sim + CARA metrics render above the ACCEPT/DECLINE banner.
- `[HIGH]` MC tooltips claim "200 simulations" (engine runs 10,000).
- `[HIGH]` MC convergence "marginal" warning silently dropped.
- `[HIGH]` IP-floor warning shows "0.9 IP/week" with no explanation (stale projections); fires on every trade.
- `[HIGH]` `enable_mc=True` with no timeout, opt-out, or partial results.

### Top recommendations

1. **Gate the MC behind opt-in or run it async [BLOCKER]** — default `enable_mc=False`; if on, cap the wall-clock budget and reduce `n_sims` proportionally; never block the button handler for 45s.
2. **Cache the player pool** (`@st.cache_data` or the session-guard pattern Free Agents uses) so visiting the page isn't a 5-second blank screen.
3. **Show the grade range / confidence band** ("C (D–B+, low confidence)").
4. **Put the ACCEPT/DECLINE verdict first**, advanced metrics behind expanders.
5. **Fix the stale MC tooltips** ("10,000 simulations").
6. **Surface MC convergence quality** by merging `mc_result["risk_flags"]` and showing a caption when marginal/poor.
7. **Cap or rephrase the reshuffle percentage** ("you already have a better bench option — set your lineup without trading").
8. **Add "(IL10)/(IL60)" to the "You Give" dropdown** (or exclude IL-60 players from tradeable options).
9. **Strip the internal "report B.9 / Feature 2" labels** from user copy.
10. **Add a data-freshness caption** under the "You Receive" dropdown, hide CARA/CVaR behind an "advanced risk" expander, and remove the `slideUp` animation from the verdict banner.

---

## Page 10 — Trade Finder

**Source:** `pages/12_Trade_Finder.py` (1,199 lines) + `src/trade_finder.py`, `src/trade_value.py`, `src/opponent_trade_analysis.py`

### Purpose & first impression

Conceptually one of the most useful pages in the app — it scans all 11 opponent rosters to **find** mutually beneficial trades *for* you, plus a universal 0–100 Trade Value Chart and tools to target players and browse partners. But a novice lands here and stares at a blank spinner for **43.8 seconds** before anything appears, and when results do load the **Value Chart immediately shows a devastating bug**: every player except Shohei Ohtani is classified "Replacement" — Aaron Judge, Juan Soto, Jackson Chourio included. A novice loses trust in the whole page instantly.

### Feature & output review

- **Cold-scan time: 43.8 seconds [BLOCKER].** `find_trade_opportunities(max_results=50, top_partners=11)` blocks all rendering on every new session (or after any Yahoo refresh busts the `_tf_scan_cache`). The "Re-rank by title odds" sidebar toggle labels its cost "~5 sec" when the real total is **~2 minutes** (44s base + ~60s of 5,000-sim playoff MC). The scan also fires a `_player_sgp_volume_aware` warning ~6 times for Xander Bogaerts (player_id=44, status NA) on every run.
- **The top recommendation is good:** the banner reads "Send Dansby Swanson to Over the Rembow for Sonny Gray (**+0.66 SGP**)." The full scan found 42 opportunities; the top row by Composite Score is "Jeff Hoffman + Framber Valdez → Willy Adames (My Precious), +1.10 SGP, 90% accept, A-, score 0.78," followed by single-player Swanson→Sonny Gray (Over the Rembow, +0.66, 76%, B) and Bregman→Sonny Gray (+0.78, 67%, B+).
- **Value Chart tier collapse [BLOCKER].** With 15 weeks remaining, a `time_factor = 15/26 ≈ 0.577` multiplier deflates every value below the *fixed* tier cutoffs (Elite 90, Star 75, Flex 35): **Elite 0, Star 0**, Ohtani the only "Solid Starter" (57.7), and **9,887 of 9,888 players "Replacement"** — Soto (34.5) and Judge (29.2) among them. Tier headers render "Elite (0)" with no content, compounding the confusion. Dollar values are similarly deflated (Ohtani $31.18, Soto $15.26, Judge $12.14).
- **"Sort by Acceptance Probability" sorts strings [HIGH].** The Acceptance column holds "High"/"Medium"/"Low"; alphabetical sort gives High < Low < Medium, so "Medium" trades float to the top — the opposite of intent. With Enhanced Recs mixed in (percentage strings like "76%"), the sort becomes undefined.
- **"ECR Fairness: 0%" on every 2-for-1 [HIGH].** The engine uses a 0.5 neutral for multi-player trades but it renders as "0%" (the field isn't set for `scan_2_for_1`, defaulting to 0), which a novice reads as "wildly unfair by expert rankings."
- **Off-palette hex** — `TIER_COLORS["Star"]="#457b9d"`, `["Flex"]="#666666"`, `["Replacement"]="#cc3333"` in `src/trade_value.py` (banned colors the CI guard misses because they're in `src/`, not `pages/`).
- **"Your Gain" uses bare `round()` floats** ("1.10", "0.66") instead of `format_stat(value, "SGP")` ("+1.10") — and SGP is never defined anywhere on the page. Context cards (Category Needs "AVG, R, RBI, SB"; Best Trade Partners "Over the Rembow 0.580, Baty Babies 0.517, On a Twosday 0.460") show category names and raw complementarity decimals with no rank or range.
- **A faster small run confirms the cost is the scan, not the engine.** `find_trade_opportunities(max_results=5, top_partners=3)` returned 5 results in **17.8s**; the full `max_results=50, top_partners=11` returned 42 in **43.8s**; `compute_trade_values()` over 9,888 players is only **2.9s**. So the wall is purely the breadth of the opponent scan — which makes the case for a narrower default plus an "Expand search" button compelling. The Browse Partners tab (a genuinely nice feature: per-opponent category comparison + full hitter/pitcher roster view via `build_roster_table_html`) and the Target-a-Player tab (lowball + fair proposals with ADP/ECR fairness) both inherit the same un-defined-SGP and bare-subheader issues.

### UI/UX & visual-design critique

This is the most compute-heavy page in the app, and 44 seconds before *any* content appears is a hard barrier for casual use (FantasyPros and ESPN's Trade Machine return in under 3 seconds; under 12-user concurrency the cold scan will drop WebSockets). The Value Chart's "9,887 of 9,888 are Replacement" is the single most severe usability failure on the page — and the empty tier headers ("Elite (0)", "Star (0)") make it look more broken, not less. The complementarity score ("0.580") shows with no range or label; the Browse Partners "Opportunity" column labels ("YOU GAIN" / "THEY GAIN" / "EVEN") are ambiguous (a novice reads "YOU GAIN: AVG" as "this trade improves my AVG" rather than "exploitable asymmetry"); the "Source: Scan" column and empty "ΔPlayoff%/ΔChamp%" columns clutter the table when title odds is off; and Tabs 2/3 use bare `st.subheader()` instead of the Combustion panel header. The Trade Readiness tab is the only well-explained metric (its formula caption spells out 40% Cat Fit + 25% Proj Conf + 15% Health + 10% Scarcity + 10% FA Edge) but its sub-components are otherwise undefined, and its position filter is unsorted with Util/DH/TWP mixed in. The Value Chart should arguably be Tab 2 (the reference you consult *before* trading), not Tab 5.

### Key defects

- `[BLOCKER]` Value Chart classifies Soto/Judge and 9,887 of 9,888 players as "Replacement" (time-decay vs fixed cutoffs).
- `[BLOCKER]` 43.8s cold scan with no partial results — will drop WebSockets under concurrency.
- `[HIGH]` "Sort by Acceptance" sorts strings alphabetically (High < Low < Medium) — wrong order.
- `[HIGH]` Mixed types in the Acceptance column when Enhanced Recs are present break the sort.
- `[HIGH]` ECR Fairness shows "0%" on all 2-for-1 trades.
- `[HIGH]` Spinner says "~5 sec" when the real wait is ~2 minutes.
- `[HIGH]` `TIER_COLORS` uses banned off-palette hexes (`#457b9d`, `#666666`).
- `[HIGH]` "Your Gain" uses raw floats, not `format_stat(value, "SGP")`.

### Top recommendations

1. **Fix the Value Chart tiers [BLOCKER]** — assign tiers from the pre-decay value (store `trade_value_full` for tiering and `trade_value` time-decayed for display), or scale the cutoffs by `time_factor`, so Soto/Judge aren't "Replacement."
2. **Background- or pre-compute the scan [BLOCKER]** — run `find_trade_opportunities` in the scheduler and read cached results; reduce the default to `max_results=20, top_partners=5` with an "Expand search" button; show partial (1-for-1) results first.
3. **Sort Acceptance on a numeric hidden column** (a `_accept_prob` float), not the string label.
4. **Show "N/A" for ECR Fairness on 2-for-1 trades**, not "0%".
5. **Explain SGP units** with a header tooltip and format "Your Gain" with `format_stat` (the mandatory `+` prefix).
6. **Explain the Grade scale and Acceptance labels** via a legend ("A = excellent gain; higher grade ≠ higher acceptance").
7. **Replace the off-palette `TIER_COLORS` hexes** with THEME tokens (`T["sky"]`, etc.).
8. **Surface data-freshness** in the Value Chart (projections in error since 2026-06-10).
9. **Reorder the tabs** (Value Chart to position 2; Trade Readiness to the end) and use Combustion panel headers on Tabs 2/3.
10. **Fix the title-odds toggle wait-time label** to "~2 min total," hide the empty `ΔPlayoff%`/`ΔChamp%` and "Source: Scan" columns when title odds is off, surface the complementarity score as a labeled percentage ("58% — strong fit, top 3 of 11"), explain the Trade Readiness sub-components, and add a "Analyze in Trade Analyzer →" deep-link per recommendation (setting `st.session_state["tf_proposal"]`) to close the workflow loop.

---

## Page 11 — Free Agents

**Source:** `pages/14_Free_Agents.py` (1,329 lines) + `src/optimizer/fa_recommender.py`, `src/in_season.py`

### Purpose & first impression

The free-agent wire browser — where a manager drops someone and picks up a better player. It should answer three questions in 30 seconds: who are the best available, who should I drop, and which pickups help me win this week? It mostly delivers (the FA recommendation engine has been heavily overhauled across PRs #89–#110), but the first impression stumbles: the **`Player universe` selectbox renders above the page header** (it's called ~216 lines before `render_page_header`), so the very first thing a user sees is an unlabeled dropdown before they even know what page they're on. The content jargon ("Marginal Value", "Impact", "Net SGP", "Best Category") lands without explanation, and the page computes silently over 7,770 MLB FAs in a tight `iterrows()` loop with no spinner, appearing frozen.

### Feature & output review

- **"Click column headers to sort" is a lie [BLOCKER].** The subtitle (line 1224) promises sortability, but whenever the Heat column is present (which is whenever ownership data exists — i.e. always, given 10,786 `ownership_trends` rows), the page uses `render_compact_table`, which renders a **static HTML `<table>` via `st.markdown` — headers are not clickable.** A user who clicks gets nothing.
- **`player_news` query throws every load [HIGH].** Line ~721 runs `SELECT player_name, il_status FROM player_news ...`, but the table has **no `player_name` column** (only `player_id`). It raises every page load, is silently caught by a `try/except`, and so the news-based IL-protection path is **permanently a no-op** — one of the IL-safety layers is dead and the user never knows.
- **"Budget: 5 streaming adds per week" is wrong [HIGH].** Hardcoded on line ~874; the canonical `WEEKLY_TRANSACTION_LIMIT = 10`.
- **Curtis Mead outlier [HIGH].** The top-ranked FA is **Curtis Mead (1B, marginal_value 125.19, best category OBP +136.08)** — nearly 2× the #2 (Rob Brantly, 63.38) and ~30× the cluster below (Narváez 11.8, Margot 10.6). He's listed as team **"WSH"** (wrong; he plays for TBR). The extreme value (likely because Team Hickey has no projected 1B production to compare against, plus the team-data error) tops the list with no explanation. The ranked list below it skews heavily to catchers (Brantly, Narváez, Nuñez, Moreno, Kirk, Campusano) because OBP scarcity dominates.
- **"Show all 7770 free agents" is a browser trap [HIGH].** Unchecking the top-200 limit would render ~7,770 HTML rows (≈200,000 DOM nodes) and freeze the tab — no server-side pagination, just a client-side slice.
- **Ownership is dead.** `percent_owned = 0.0` for all pool rows and `delta_7d` is NULL, so the Heat column is almost all `--`, and the "Breakout Candidates (Heat≥5, Owned<30%)" toggle is a **mathematical impossibility** (Heat≥5 requires Owned≥50%). A 608-row `yahoo_free_agents` cache with real ownership (Reid Detmers 56%, Aaron Nola 54%, Zac Gallen 43%) sits unused.
- **Silent streaming failure** — the entire "This Week's Streams" section is wrapped in `except Exception: pass`, so any error leaves a blank section with no feedback.
- **Fallback warning leaks jargon** — *"Recommendations may not reflect the latest P3.5 fixes. Check logs."* — developer-facing language exposed to a consumer.
- **Net SGP rendered in ember red** (`var(--fp-ember)`) for positive values, where orange is the correct positive accent — a positive gain shown in the system's negative color.
- **The IL-stash protection itself works.** 55 IL players are identified league-wide (Garrett Crochet, Cal Raleigh, Blake Snell, Aaron Judge among them) and correctly blocked from the headline drop table — a real improvement from the overhaul. But the "Recommended Drops" surface (Drop / Position / Replaced By / Top Category Impact, e.g. "OBP: +0.82") still renders that impact in jargon a novice can't anchor, and the Roster Summary card depends on a session-cached Yahoo roster frame: a raw DB query for `team_name = 'Team Hickey'` returns 0 rows (the emoji-name mismatch), so a cache-expired offline load would hit the "Your roster appears to be empty" guard and `st.stop()`.

### UI/UX & visual-design critique

The four-section vertical scroll (Recommended Adds/Drops → This Week's Streams → All Free Agents → Recommended Drops) is too long without tabs — a user hunting for "who to stream today" must scroll past personalized recs and a 200-row table. The "All Free Agents" table runs up to **21 columns** (Player / Position / Heat / health / ECR / xwOBA / Barrel% / Stuff+ / Signal / YTD AVG / YTD HR / YTD ERA / YTD K / L14 AVG / L14 HR/G / L14 ERA / L14 K/G / Marginal Value / Impact / Best Category / Category Impact) — extreme horizontal density requiring scroll on any monitor. The position-filter pills have a **slash-delimited bug**: `_apply_pos_filter` / `_apply_pos_filter_recs` split on comma only, so a "2B/SS" player never matches the "2B" pill. The "Why" expanders (the most important content) are collapsed by default and labeled the unintuitive "Why:". The sustainability heat bar shows "73%" with no tooltip; `percent_owned` is never shown as a direct column (only the abstract Heat score); and the Marginal Value tooltip is a `st.caption` below 200 rows of table — unreachable. The Data Freshness card shows "Warming up" in muted text with no age, the analytics badge appears with no explanation, and nine position pills stacked in a narrow 1/5 sidebar column make poor touch targets on mobile. The team logo renders as a blank gap when `team=None` (Rob Brantly and others). Section 4's IL-protection check is weaker than Section 1's (it misses IL15/IL60 news strings) — harmless now only because the list is pre-filtered.

### Key defects

- `[BLOCKER]` "Click column headers to sort" is false — headers are static HTML.
- `[HIGH]` `player_news` IL-protection query fails silently every load (no `player_name` column).
- `[HIGH]` Position filter never matches slash-delimited multi-position players ("2B/SS").
- `[HIGH]` "Budget: 5 streaming adds per week" hardcoded (actual limit 10).
- `[HIGH]` "Show all 7770" is a browser-freeze/OOM trap.
- `[HIGH]` Curtis Mead #1 at marginal_value 125.19 (wrong team "WSH"); extreme outlier, no explanation.
- `[HIGH]` Silent `except: pass` on the streaming section.

### Top recommendations

1. **Remove the false "Click column headers to sort" caption** (one-line fix) or switch to `render_sortable_table` whenever HTML columns aren't required.
2. **Move the `Player universe` selectbox** below the header (into the context panel).
3. **Fix the `player_news` query** to join `players` for the name (`p.name AS player_name`).
4. **Fix the hardcoded "Budget: 5"** to the real `WEEKLY_TRANSACTION_LIMIT` / `ctx.adds_remaining_this_week`.
5. **Fix the slash-delimited position-filter bug** (normalize "/" to "," before splitting).
6. **Restructure into tabs** (Recommended / Browse / Streaming / History) so each answers one question.
7. **Expose "Marginal Value" with a header tooltip** and a star rating; explain SGP scale; show `percent_owned` as a direct "Own%" column from `yahoo_free_agents`.
8. **Show data age prominently** with a stale-data warning when > 24h (FA data is 3 days old).
9. **Replace "Show all 7770"** with server-side pagination (or a 500-row ceiling).
10. **Translate the fallback warning** out of "P3.5 / Check logs" jargon, fix the Net SGP color to orange, and auto-expand the first "Why" expander.

---

## Page 12 — Player Compare

**Source:** `pages/16_Player_Compare.py` (873 lines) + `src/in_season.py::compare_players()`

### Purpose & first impression

Side-by-side player comparison: pick two players, see who wins each of the 12 categories via z-scores, a radar chart, SGP breakdown, YTD and rolling stats, a Statcast profile, health/confidence, and schedule strength. The intent is sound, the execution is rough from the first interaction. A "Player universe" selectbox appears before any player input (a config knob ahead of the primary action); the player search shows **three random pill-buttons before the user types anything** (an empty search returns the first three names alphabetically); and there's no confirmation a selection registered beyond a pill changing color — the selected name is never echoed at the top of the results column.

### Feature & output review

- **Rostered players show "Free Agent" [BLOCKER].** Team Hickey's `league_rosters` slice returns 0 rows (the data-wipe / emoji-name mismatch), so the roster-status badge lookup is empty and **Juan Soto, Aaron Judge — every player — displays "Free Agent"** regardless of who actually rosters them. Live-broken for the owner's own team.
- **Injury-badge HTML renders as raw text [HIGH].** The Health & Confidence table uses `render_styled_table` → `st.dataframe`, which **does not interpret HTML in cells**, so the column displays literal `<span style="...">●</span> Low Risk` text instead of a colored dot.
- **Half the radar collapses to zero [HIGH].** Real z-scores for Soto vs Judge: Soto wins all six hitting cats (Runs +6.34 vs +5.19, HR +9.97 vs +8.63, RBI +6.61 vs +5.39, SB +5.15 vs +2.19, AVG +1.80 vs +1.66, OBP +2.45 vs +2.30), composites **+32.31 vs +25.36**, "6 — 0 (6 ties)." But the six **pitching** axes all show z = 0.00 for both, so the radar polygon degenerates to flat on half its spokes, and the Category Edge bars show six identical 50/50 "TIE" splits. A user can't tell "TIE because equal" from "N/A — these stats don't apply."
- **Inverse-stat z-scores confuse [HIGH].** For an SP pair (Skubal vs Skenes), the Losses row shows **Skubal -1.48, Skenes -1.95, Advantage: Skubal** — a negative number "winning" a category with no explanation that for inverse stats, lower (more negative) is better.
- **Cross-type comparison is meaningless [HIGH].** Soto vs Skubal gives Soto +32.31 vs +11.62, with Soto "winning" Losses and Saves simply because hitters have z = 0 there. The page allows this with no warning.
- **YTD data missing** — Aaron Judge (player_id=1) shows "--" for all YTD hitting stats (`ytd_pa = 0` from the partial `season_stats` refresh), indistinguishable from "Judge has no HRs." The YTD table also renders six pitcher rows (ERA/WHIP/SV/K) as "--/--" for two hitters.
- **SGP vs z-score inconsistency** — in the adjacent SGP Breakdown, Judge's OBP SGP (+0.59) beats Soto's (+0.45), yet the z-score table says Soto wins OBP. Two tables side by side give opposite winners with no explanation (different normalization).
- **"Schedule Strength" is mislabeled** — it shows the player's **own team's** wRC+/FIP/ERA, not upcoming opponent quality.
- **`compute_category_fit()` is never called** — the one function that answers "does this player help MY weak categories?" (the killer feature for a fantasy manager) exists in `in_season.py` but is never imported by the page.
- **The SGP Breakdown is correct but its caption is boilerplate.** For Soto vs Judge it shows Runs +1.81 vs +1.50, HR +1.77 vs +1.54, RBI +1.78 vs +1.47, SB +0.79 vs +0.36, with totals +6.66 vs +5.42 (delta +1.24) — yet the caption talks generically about "concentrated (3.0 from HR/RBI) vs diversified (0.5 across many)" value even though neither player has concentrated value, a missed chance to summarize what the data actually shows.
- **Health & Confidence is where the rendering bug bites hardest.** The intended output (Soto "Low Risk" 0.85, Judge "Moderate Risk" 0.72, with a P10–P90 confidence band) instead prints the literal injury-badge `<span>` markup as text, and the "Confidence" column shows an unlabeled `±38.20`-style differential (summed across counting categories) with no units a novice can read.

### UI/UX & visual-design critique

The section ordering buries the most actionable content (recent form, health) below a long scroll while putting the most advanced (SGP breakdown) near the top where a novice doesn't yet understand it; sections 8–14 (schedule, YTD, recent form, Statcast, player-card opener, health, catcher framing) sit far below the fold with no navigation anchors. The identity cards use **team brand colors** (dark navy for both NYY and NYM) instead of the orange-A / steel-B convention the radar uses, so the user can't tell which side is which at a glance — the page asks the user to learn two different A/B conventions. The Recent Performance headings use inline `st.markdown("**…**")` bold rather than `st.subheader`, breaking the visual rhythm. The Statcast caption still explains "Stuff+" though it never renders (NULL for all 9,888 players), and the xwOBA shown under "Statcast Profile" is actually projection-blended (`statcast_archive` has 0 non-null xwOBA rows) — implying Baseball Savant sourcing it doesn't have. The catcher-framing section shows 2024 seed data with no year label. There's no empty state in the main column when no players are selected (only a placeholder in the narrow rail), composite scores have no scale anchor, the "View player card" opener is buried mid-page, and there is no clear/reset action. Several WCAG contrast failures (8%-opacity green roster badge, `#888` placeholder text). The `width="stretch"` kwarg on the pill buttons isn't recognized in Streamlit 1.47 and is silently ignored, and the progress bar is purely decorative (`compare_players()` is <1ms).

### Key defects

- `[BLOCKER]` Roster badges always show "Free Agent" — empty `league_rosters` slice for Team Hickey.
- `[HIGH]` Injury-badge HTML renders as literal text in `render_styled_table`.
- `[HIGH]` Radar shows 6 dead axes for same-type pairs — looks visually broken.
- `[HIGH]` Inverse-stat z-scores display negative numbers "winning" categories.
- `[HIGH]` Cross-type (hitter vs pitcher) comparison produces a meaningless composite, no warning.
- `[HIGH]` `compute_category_fit()` (the most useful feature) exists in code but is never called.
- `[HIGH]` YTD data missing for rostered stars (Judge all "--"), no indication.

### Top recommendations

1. **Fix the injury-badge rendering** — add "Health" to `html_cols` (or use `render_compact_table(html_cols={"Health": True})`).
2. **Suppress inapplicable categories** for same-type pairs (a 6-axis radar), and warn on cross-type comparisons.
3. **Wire `compute_category_fit()`** into a "Team Fit" card showing which cats each player helps vs wastes.
4. **Explain inverse-stat z-scores** ("lower is better"; or show quality-adjusted z so higher always wins).
5. **Fix the roster-status badge** via `resolve_viewer_team_name(rosters)` so rostered players don't show "Free Agent."
6. **Add a data-freshness banner** mirroring My Team.
7. **Suppress "--" rows** in YTD and Recent Performance when both values are missing.
8. **Rename/fix "Schedule Strength"** to "Team Context" (or compute real upcoming-opponent quality).
9. **Redesign the player search** (empty initial state, selection echo, clear button) and only show pills after 2+ typed characters.
10. **Add a composite-score scale anchor** (percentile), give the identity cards an A=orange / B=steel through-line, and add an empty state in the main column.

---

## Page 13 — Leaders

**Source:** `pages/17_Leaders.py` (1,106 lines) + `src/leaders.py`, `src/trend_tracker.py`, `src/prospect_engine.py`

### Purpose & first impression

A seven-tab research hub: category leaders, an overall "value" score, breakout candidates, prospect rankings, and Hot/Cold/Sell-High trend lists. It contains both **the best-looking component in the app** and **three of its most broken outputs.** The left panel shows two filter cards at once ("Category Filter" for tab 1, "Prospect Filters" for the Prospects tab) regardless of the active tab, so a novice on the Hot List tab sees four controls that do nothing. Seven tabs is near the limit of navigability, and "Category Value" vs "Breakout Candidates" vs "Hot List" are not self-distinguishing labels — the three trend tabs were folded in from a deleted Trends page and sit awkwardly beside the four analysis tabs.

### Feature & output review

- **Category Leaders (tab 1) is excellent.** The branded `_build_leaderboard_html` — team-color row backgrounds, MLB headshots, team logos, IBM Plex Mono stat figures, orange heat bar — is genuinely professional. Real outputs are correct: HR leaders Schwarber 23 / Y. Alvarez 22 / Olson 19 / Buxton 19 / Rice 18; ERA leaders Chapman 0.46 / Varland 0.50 / Brash 0.54; K leaders Misiorowski 116 / Sánchez 113 / Cease 103; SB José Ramírez 24 / Nasim Nuñez 24 / Witt Jr. 23; SV Cade Smith 21 / Mason Miller 18 / Bryan Baker 18. **But the "L Leaders" board is semantically broken** — ascending sort on an inverse stat shows 15 pitchers with **0 losses**, most with 9–18 IP (small-sample relievers), which reads as broken to a novice. The caption "Showing top 15 of 2,645 eligible players" also uses the full season-stats count even for pitching categories where only ~476 qualify.
- **Hot/Cold lists are garbage [BLOCKER].** The Hot List is topped by **Burch Smith +392.235, Cody Bolton +366.670, Spencer Miles +319.697, Peter Lambert +309.259** — fringe pitchers whose near-zero ROS IP projections collapse the rate-delta denominator to `RATE_FLOOR=0.001`, producing deltas 3–4 orders of magnitude too high. The Cold List mirrors it (Walbert Urena **-935.341**, Jose A. Ferrer **-854.444**). And **every pitcher on both lists shows ".000 AVG / 0 HR"** because `_trend_key_stats` coerces a pitcher's `is_hitter=0` back to 1 via `int(row.get("is_hitter", 1) or 1)`. The first real-data entries (Curtis Mead 53.5, Leody Taveras 43.5) appear below 11 garbage rows.
- **Breakouts (tab 3) is entirely non-functional [BLOCKER].** All `statcast_archive` Statcast columns are NULL, so `compute_breakout_score` falls to the z-score path with near-zero inputs and **scores every player at exactly 50.0.** The page shows "No players currently score above 70" followed by 20 players all tied at 50.0 — pure noise with no hint the pipeline failed, despite the tab description promising "barrel rate, xwOBA, hard hit percentage, Stuff+."
- **Prospects (tab 4) is hollow [HIGH].** All 20 `prospect_rankings` rows have **NULL `fg_rank`** (blank Rank column), **NULL `mlb_id`** (generic silhouettes), and **NULL scouting grades** (every "Scouting Tool Grades" panel shows an empty state, the radar renders nothing). ETA-2025 players who've already debuted (Roki Sasaki FV 80, Roman Anthony FV 70) top the "prospect" list, and only 20 records across 13 of 30 teams exist.
- **Sell-High shows 1 player** (Andrew Vaughn, sustainability 0.411) — the `SELL_HIGH_SUSTAINABILITY_CAP=0.45` is so tight that of 335 HOT players, only one qualifies.
- **Category Value (tab 2)** shows an unexplained z-score sum across all 12 categories (James Wood 13.04, Oneil Cruz 12.54, Yordan Alvarez 12.17, Nick Kurtz 11.59, Jordan Walker 11.33, CJ Abrams 11.09, with relievers Cade Smith 10.89 and Mason Miller 10.69 mixed in) with no scale anchor, and **leaks the raw `mlb_id` column** (e.g. 694497) into the user-visible table. It also drops the rich branded leaderboard for a plain `render_compact_table`, so two tabs that both rank players look like different apps.
- **The Category Leaders tab uses a 44% season fraction** (11.4 weeks elapsed) to set qualification floors of Min PA = 21 and Min IP = 8.8 — reasonable thresholds, but the low IP floor is exactly why the L board fills with 9–18-IP relievers, and the "of 2,645 eligible" caption never adjusts down to the ~476 pitchers who can actually appear in a pitching category.

### UI/UX & visual-design critique

The quality gap *within one page* is the story: the branded Category Leaders leaderboard is the design bar the app should meet everywhere, while Hot/Cold/Sell-High and Category Value use plain `render_styled_table` / `render_compact_table` (navy headers, no headshots, no heat bars, raw float deltas) — among the worst-looking components in the app, and jarring to navigate to immediately after Tab 1. The left panel should be tab-aware; showing four irrelevant controls on the Hot List tab wastes space and confuses. The trend computations (`compute_player_trends` over 9,888 rows, `detect_sell_high_candidates`, `compute_breakout_scores_batch`) run **uncached on every tab switch** — and `detect_sell_high_candidates` re-runs `compute_player_trends` internally — a significant performance tax. There is no position filter on Category Leaders (a user can't ask "top shortstops in HR"), no Free-Agents-only filter to make this an FA-discovery tool, and the Trend Delta column shows raw floats (0.133, -0.648) with no units or capped range even when the data isn't broken. The seven-tab strip needs hover-text or two visual groups, and the `render_reco_banner` adds only a decorative static line.

### Key defects

- `[BLOCKER]` Hot/Cold lists dominated by garbage 200–935× deltas from near-zero ROS IP denominators.
- `[BLOCKER]` Hot/Cold "Key Stats" shows ".000 AVG / 0 HR" for all pitchers (`is_hitter or 1` bug).
- `[BLOCKER]` Breakouts tab non-functional — all Statcast NULL, every player scores 50.0.
- `[HIGH]` Prospects: `fg_rank` NULL (blank Rank column) for all 20.
- `[HIGH]` Prospects: all 20-80 scouting grades NULL — scouting panel always empty.
- `[HIGH]` Trend/breakout computations run uncached on every tab switch.
- `[HIGH]` Both filter cards always visible regardless of active tab.

### Top recommendations

1. **Fix the Hot/Cold delta explosion [BLOCKER]** — gate out players with `ip_proj < 15` / `pa_proj < 50` and `np.clip` deltas to [-3, +3] before classification.
2. **Fix the pitcher key-stats bug [BLOCKER]** — correct the `is_hitter or 1` coercion (`1 if pd.isna(...) else int(...)`) so pitchers show ERA/K.
3. **Fix or gate the Breakouts tab** — when Statcast is NULL, show a clear "Statcast data not loaded" state, not 50.0 noise.
4. **Redesign the L leaderboard** ("Fewest Losses (Qualifying)", enforce a real IP minimum like 30 IP).
5. **Make the left panel tab-aware** so only the active tab's filters show.
6. **Bring Category Value up to the Category Leaders visual standard** (headshots, heat bar, top-contributing-category chips) and remove the `mlb_id` leak.
7. **Cache the trend/breakout computations** (`@st.cache_data`).
8. **Explain the "Category Value" number** (percentile column or tier label, with the league-average = 0 anchor).
9. **Reduce 7 tabs to 5** (split Hot/Cold/Sell-High to a Trends page) or group into two labeled clusters.
10. **Fix prospect data** (fall back to `fg_fv` ranking, resolve `mlb_id`, filter arrived players, hide null scouting panels) and add a Free-Agents-only filter to Category Leaders.

---

## Page 14 — Player Databank

**Source:** `pages/19_Player_Databank.py` (417 lines) + `src/player_databank.py` (1,457 lines)

### Purpose & first impression

Billed as a "historical multi-year player lookup" — search the 9,888-player pool, filter six ways, pick a time window, get a sortable stat table with Excel export. The empty-state ("Search the Databank — set your filters, then click Search") is the right deferred-execution pattern, but there's an overwhelming amount of UI above the fold: a search box, two filters, then **six dropdowns in one row** culminating in a full-width Search button. The "Stats" dropdown defaulting to "Season (total)" confuses immediately (total of what — last 7 days, last 30, or season, all in one menu?). And despite the "historical" billing, the page is really **a paginated list sorted by preseason ADP** with no per-player drill-down and no year-over-year comparison — overlapping heavily with Leaders and Player Compare. The 28-option Stats dropdown mixes projection, total, average, std-dev, and "special" views without section separators.

### Feature & output review

- **IP renders factually wrong [BLOCKER].** Zack Wheeler shows **IP 56.7** — but 56⅔ innings is **56.2** in baseball outs notation. `_format_cell` uses `.1f` on the decimal 56.666…, producing "56.7," which doesn't correspond to a valid inning count. This affects **every pitcher in every IP-bearing view** (his 2025 row shows IP 149.7 for 149⅔). His ERA (2.22 from ER×9/IP) and WHIP (0.85) are computed correctly.
- **"Today (live)" shows projections [BLOCKER].** The `type="live"` view routes to the pool-as-is path and returns **ROS blended projections** (Freeman r=33, hr=9, rbi=34), not today's live stats. A user setting a lineup off "live" data is seeing forecasts.
- **"Waivers Only" filter is a silent no-op [HIGH].** `filter_databank` has no handler for `status="W"`, so it falls through to "no filter" and returns all 4,408 batters — identical to "All Players."
- **"Util" position filter always returns 0 rows [HIGH].** No player has "Util" in their positions string (it's a Yahoo slot concept), so the filter always yields an empty table.
- **Four stub stat views [HIGH].** "Ranks," "Research," "Fantasy Matchups," and "Opponents" all return the identical pool with the identical 6 stat columns — `render_databank_table` ignores the stat view when choosing columns, so even "2026 Advanced" shows **no Statcast columns** (xwOBA/barrel%/EV/Stuff+ are never rendered, and `statcast_archive` is empty for all but sprint_speed anyway).
- **96% of players show only dashes.** Only 408 of 9,888 players have 2026 game_logs, so 16 of the first 25 default-result rows are all-dash — and a novice can't tell "no data" from "zero stats." (The captured first page: Agustin Ramirez .230/2HR, Yainer Diaz all-dash, Lawrence Butler all-dash, …) The 2024 game-log totals also carry minor discrepancies vs `season_stats` (Freeman game-log R=80/HR=21/RBI=86 vs official 81/22/89).
- **% Ros always "—"** (`percent_owned = 0.0` everywhere; the 608-row `yahoo_free_agents` cache with real values up to 74% is unused). The Opponent column always shows "—" offline.
- **Off-palette `#2c2f36`** in `render_databank_table` CSS (line 1071, a banned color the CI guard misses because it's in `src/`).
- **The few players with full game-log coverage show the tool's real potential.** Freddie Freeman (LAD 1B) renders cleanly across windows — 2026 season GP=64, R=35, HR=10, RBI=36, AVG .284, OBP .366 (the `avg_calc`/`obp_calc` paths correctly override the blended-projection .262); 2025 GP=147, .295/.367; last-7-days GP=3, .455/.500 — and his Roster Status correctly reads "BUBBA CROSBY." This is genuinely useful multi-year data; the problem is that 96% of the pool can't reach it, and there is no per-player view that stitches these windows into a career trajectory despite the "historical multi-year lookup" billing. The Adds/Drops columns *do* populate from `yds.get_transactions()` (76 players with adds, 73 with drops), one of the few enrichment paths that survives Yahoo being offline.
- **Export to Excel is solid** — `export_to_excel` writes the **full filtered result** (not just the visible 25 rows), with auto-widened columns and correct AVG/OBP/ERA/WHIP precision; only its filename carries a cosmetic double-underscore (`Season__total_`). And the instrument panel's freshness chips ("STATS AS OF: 06/10 — end of games", "REFRESHED: 6/10, 3:41 PM EDT") are accurate for game-log views even if they mislabel projection views.

### UI/UX & visual-design critique

Six horizontal dropdowns collapse to near-zero width on mobile. The custom HTML table is otherwise well-styled (Archivo headers, tabular figures, orange sort arrows, even-row striping, hover highlight), but its **JS sort is client-side and per-page only** — sorting "by AVG" on page 2 sorts only page 2's 25 rows, not the full result, which is genuinely confusing once paginated. The "GP*" asterisk has no explanation anywhere on the page. `load_databank` is **uncached**, so every Search costs 1.4–3.6s (full pool load + game-log merge) even when filters are unchanged. The default sort (Pre-Season ADP, Ascending) surfaces obscure untracked players first — wrong for mid-season use, and only 579 players have ADP at all. The Roster Status column prints full team names (e.g. "The Good The Vlad The Ugly," 26 chars) with no abbreviation, breaking narrow layouts. The stat-group banner row ("Season (total)") duplicates the Stats dropdown selection. The instrument panel's "STATS AS OF / REFRESHED" chips are accurate for game-log views but show game-log freshness even on projection views (where "as of … games" is factually wrong). Standard-deviation views show raw decimals ("0.577 RBI std dev") a novice cannot interpret, and the page advertises "Live stats for every MLB player" — misleading on both counts. There is no per-player profile or drill-down.

### Key defects

- `[BLOCKER]` IP renders "56.7" instead of "56.2" for every pitcher in every view.
- `[BLOCKER]` "Today (live)" shows ROS projections, not live stats.
- `[HIGH]` "Waivers Only" silently returns all players.
- `[HIGH]` "Util" position filter always returns 0 rows.
- `[HIGH]` Four stub stat views show identical data to the default.
- `[HIGH]` "2026 Advanced" never shows Statcast columns.
- `[HIGH]` 96% of players show only dashes (no game-log coverage) with no explanation.
- `[HIGH]` `load_databank` uncached — every Search costs 1.4–3.6s.

### Top recommendations

1. **Fix the IP display** — add an outs-notation branch (`floor(ip) + "." + round((ip-floor(ip))*3)`, so 56.666 → "56.2").
2. **Fix "Today (live)"** — route `type="live"` to a real boxscore fetch, or rename it "ROS Projections."
3. **Fix/remove "Waivers Only" and "Util" filters** (the app does not differentiate waivers from FAs; no player carries "Util").
4. **Implement or remove the four stub stat views** (a stub that misleads is worse than no option).
5. **Make the table show view-specific columns** (Advanced → Statcast; Ranks → ADP/ECR/%Ros; Std Dev → with a disclaimer).
6. **Cache `load_databank`** (`@st.cache_data(ttl=300)` keyed on stat view).
7. **Replace "% Ros" with real ownership** by joining `yahoo_free_agents` on player name.
8. **Differentiate "no game-log data" from zero stats** (a `_has_game_log` flag, or fall back to `season_stats` YTD).
9. **Fix the `#2c2f36` off-palette color** (use `{T["tx"]}`) and change the default sort to "Current / Descending."
10. **Explain "GP*"** with a `title=` tooltip, abbreviate the Roster Status column, and add a per-player drill-down / Player Compare link.

---

## Page 15 — Draft Simulator

**Source:** `pages/20_Draft_Simulator.py` (929 lines) + `src/simulation.py`, `src/draft_engine.py`, `src/pick_predictor.py`

### Purpose & first impression

A mock snake draft against AI opponents: set the format, pick a draft position, simulate 23 rounds, and get Monte Carlo recommendations at each user turn. The setup panel is clean and unambiguous (12/23/1 defaults matching the real league, two radio groups for Simulation Depth and Engine Mode, a primary "Start Mock Draft"). But the header eyebrow reads **`PRESEASON`** and it is **Week 12** — a novice immediately wonders whether they opened the wrong thing. There's no framing for why a mock draft is useful mid-season, so the page reads as orphaned preseason functionality.

### Feature & output review

- **Recommendation panel (pick #6, reconstructed):** #1 Jackson Chourio (combined score 20.99, survival 54.5%, urgency 0.000, FAIR), #2 Ronald Acuña Jr. (20.81, 28.9%, 0.059), #3 Wyatt Langford (20.43, 83.8%), #4 James Wood (19.90, 82.9%), #5 Corbin Carroll (19.85, 44.5%). All show "FAIR" because the thin test slice lacks the Statcast/injury context for BUY/AVOID variety; in production some variety appears, though Ohtani still shows FAIR at 52% injury probability — a calibration miss worth a look.
- **Two unexplained scores per player.** The recommendation card's orange badge shows **combined_score (33.89 for Ohtani at pick 1, = `mc_mean_sgp + urgency*0.4`)** while the quick-pick card below shows **pick_score (15.61)** — two different numbers, neither with a unit, range, or explanation.
- **Active MLB players show "FA" [MEDIUM].** Jackson Chourio (MIL) and Wyatt Langford (TEX) have empty `team=''` in the DB, so the recommendation cards label them "FA" — a data-quality bug surfacing as a wrong roster status.
- **"Undo Last Pick" undoes the AI's pick, not yours [HIGH].** After any user pick, `auto_pick_opponents()` runs immediately, so the last `pick_log` entry is always an AI pick. Undo removes one AI pick, then auto-picks again — the user's own pick is never undone. The button is actively misleading (the user would need to click it twice, with no indication of how many times).
- **"View player card" selectbox prints raw HTML [HIGH].** The Available Players tab injects `<img>` headshot HTML into the `Player` column, then passes that same column to `render_player_select` as option labels — so the selectbox displays literal `<img src="...">Aaron Judge` strings.
- **"Simulate Opponent Pick" is dead code [MEDIUM]** — `auto_pick_opponents()` always runs before render, so the "Waiting for X to pick…" branch is unreachable. The AI also uses only a linear ADP-weighted model (`available.nsmallest(15, "adp")`), ignoring the more sophisticated `opponent_pick_probability` that models positional need.
- **Urgency label/bar scale mismatch** — the label shows the raw float "2.74" while the bar fills `min(100%, 2.74×100%)` = pinned at full; the two representations disagree.
- **Draft-grade thresholds are uncalibrated** (A>8 / B>4 / C>-2 / D>-6 / F else) — in a 23-round snake the typical surplus over the average team is ±5, so a competent draft lands "C" and the user feels unfairly graded, with no explanation of the scale. The summary's SGP loop also bypasses `SGPCalculator.totals_sgp`, omitting rate-stat volume weighting.
- **The AI opponents are a thin model.** `auto_pick_opponents()` draws from `available.nsmallest(15, "adp")` with linearly decreasing weights `[15,14,…,1]/sum`, so the bots simply chase ADP — they take Ohtani, Judge, Soto, Skubal, Skenes in rounds 1–5 before the user's pick at slot 6 and never react to their own positional needs, even though `DraftSimulator.opponent_pick_probability` (which models need and historical preference) exists. The result feels robotic and predictable, undercutting the "practice against realistic opponents" value the page implies.
- **The enhanced metric row and My Roster panel reinforce the opacity.** The #1 card's secondary row reads "Need: 1.20x" (category-balance multiplier) and "Skill: -0.43" (Statcast delta) with neither label explained, and the active-draft "On the Clock" command bar (Round X · Pick Y + "Your Pick") is polished but sits above two heat bars that look identical while measuring incompatible things.

### UI/UX & visual-design critique

The active-draft layout (context rail + main, command bar with a "Your Pick" call-out) is polished, and the top-3 recommendation cards (machined corners, headshots, BFA pills, dual heat bars) are visually rich — but the two heat bars (Survival vs Urgency) look identical while on incompatible scales (% vs unbounded float), making cross-card comparison meaningless, and the metric-row labels ("Need:", "Skill:", "Closer:", "Stream:") are too terse to mean anything. The Draft Board grid uses generic "Team 1"…"Team 12" names with no branding, and the My Roster panel renders its `render_styled_table` *outside* the context card boundary, creating a titled card with a floating table below it. Empty states use native `st.info` instead of `render_empty_state`. The `col_whip.metric("Walks + Hits per Inning Pitched", ...)` label overflows the metric widget. The position pills omit "Util" and "DH" (so DH-eligible top-15 ADP players like Alvarez and Schwarber appear only under "All"). There is no data-freshness indicator despite ~1,219-min-stale ADP/projection data, and a developer error message — *"No player data found. Run python load_sample_data.py first"* — is exposed raw to end users. The Engine Mode and Simulation Depth controls carry accurate but jargon-laden `help=` text (Marcel, Statcast) that a novice can't parse.

### Key defects

- `[HIGH]` "Undo Last Pick" undoes the AI's pick, not the user's — the button is ineffective.
- `[HIGH]` "View player card" selectbox shows raw `<img>` HTML as option text.
- `[HIGH]` `PRESEASON` framing + zero in-season context — page reads as orphaned during Week 12.
- `[HIGH]` `combined_score` badge (33.89) opaque: no unit, range, or comparison point (and a second `pick_score` shown elsewhere).
- `[MEDIUM]` Urgency bar/label scale mismatch; Chourio/Langford show "FA" (empty `team=''`); "Simulate Opponent Pick" is dead code; uncalibrated grade thresholds; SGP loop bypasses the canonical calculator; no data-freshness indicator.

### Top recommendations

1. **Add in-season framing** — change `PRESEASON` to `SCOUTING`; add a one-line "use this to prep for next year / test values" banner.
2. **Fix Undo to undo the user's own pick** (loop `undo_last_pick()` past the AI picks back to the user's turn, then don't re-run the opponents).
3. **Fix the "View player card" HTML injection** (keep a clean `player_name_clean` column for the selectbox; inject `<img>` only into the rendered table).
4. **Normalize the score badge to 0–100** with a label ("Score 88 / 100"), and reconcile combined_score vs pick_score.
5. **Fix the urgency bar/label scale mismatch** (normalize both to 0–100, or drop the bar).
6. **Backfill missing team names** so MLB players don't show "FA."
7. **Remove the dead "Simulate Opponent Pick" button** (or add a real step-through mode that makes it reachable) and upgrade the AI to use positional need.
8. **Label and calibrate the draft-grade thresholds** with a "how is this graded?" expander, and route the summary SGP through `SGPCalculator.totals_sgp`.
9. **Translate the technical metric labels** (combined_score → "HEATER Score", Need/Skill/Urgency/Survival) for novices via inline help.
10. **Add Util/DH position pills** (DH-eligible top-15 ADP players like Alvarez and Schwarber currently appear only under "All"), add a data-freshness caption, consolidate the My Roster table inside its context-card boundary, shorten the WHIP metric label, and replace the raw developer error message with consumer copy.

---

# Part VI — Master Defect Register

This is the consolidated, severity-ranked register of **every BLOCKER and HIGH** surfaced across the 15 pages. BLOCKERs gate the Beta; HIGHs are the bulk of the "feels broken / feels confusing" experience. (MEDIUM, LOW, and POLISH items live in the per-page sections of Part V.)

## VI.1 BLOCKERs (15)

| # | Page | Defect | Suggested fix |
|---|------|--------|---------------|
| B-01 | Home | In-season users land on a dead draft-setup wizard; no in-season context or navigation to My Team / Free Agents. | Add an in-season orientation banner with quick links; make My Team the default landing page in-season. |
| B-02 | My Team | Matchup ticker permanently hidden in cached/multi-user mode (gated on `yahoo_connected`) — the primary at-a-glance score element is gone. | Drive the ticker from `yds.get_matchup()` (cache-backed); show a staleness hint instead of hiding. |
| B-03 | Lineup Optimizer | Emoji team-name mismatch (`Team Hickey` vs `🏆 Team Hickey`) → all 12 category weights return 1.00× → wrong LP output + wrong Category Analysis for every run. | Route `compute_nonlinear_weights()` through `resolve_viewer_team_name(rosters)`; add to the structural guard. |
| B-04 | Closer Monitor | ARI/Paul Sewald card shows `—`/`—`/`—` for all stats (AZ→ARI normalization mismatch) while showing "2026 ACTUAL · 15 SV" below — a direct contradiction. | Normalize the pool's team column (apply `_TEAM_NORMALIZE`) before the `build_closer_grid` match. |
| B-05 | Pitcher Streaming | (Severity-class via cross-cutting) 17-column board + negative-SGP/positive-score contradiction + no freshness comms make the page's core output untrustworthy to a novice. | Front-load Score, add a legend/tooltips, explain the SGP/score reconciliation, add a freshness badge. *(Page's own list tops out at HIGH; included here as the consolidated trust-blocker.)* |
| B-06 | Matchup Planner | `schedule_warning` built but never displayed — every player renders "Rating 1.00 / Avoid / 0 games" offline with zero explanation. | Display `schedule_warning`; stop computing useless 0-game ratings when the schedule is offline. |
| B-07 | Punt Analyzer | "Biggest Value Gainers" shows 15 players with **+0.00** change for any counting-stat punt (SB, SV, HR, R, RBI, W, K). | Sort by relative rank gain, or filter `value_change > 0` and show an honest empty state when none gained. |
| B-08 | Trade Analyzer | 10K-sim MC (+20K playoff sim) runs synchronously in the button handler (44.8s) and **drops the browser WebSocket** — confirmed page freeze. | Default `enable_mc=False`; make it opt-in with a labeled cost or run async; cap the wall-clock budget. |
| B-09 | Trade Analyzer | 5-second blank page on every navigation — `load_player_pool()` + `get_health_adjusted_pool()` run uncached on every rerun. | Wrap both in `@st.cache_data` (mirror the Free Agents pattern). |
| B-10 | Trade Finder | Value Chart classifies Soto/Judge and **9,887 of 9,888** players as "Replacement" (time-decay deflates values below fixed tier cutoffs). | Assign tiers from the pre-decay value, or scale the Elite/Star/Flex cutoffs by `time_factor`. |
| B-11 | Trade Finder | 43.8s cold scan blocks all rendering on every new session — will drop WebSockets under 12-user concurrency. | Background- or pre-compute the scan; read cached results; reduce default `max_results`/`top_partners`; show partial results. |
| B-12 | Free Agents | "Click column headers to sort" is false — `render_compact_table` renders static HTML; headers are not clickable. | Remove the false caption (one-line) or use `render_sortable_table` / a client-side sort. |
| B-13 | Player Compare | Roster badges always show "Free Agent" — empty `league_rosters` slice for Team Hickey; live-broken for the owner's team. | Resolve via `resolve_viewer_team_name(rosters)`; fall back to the cached roster frame. |
| B-14 | Leaders | Hot/Cold lists dominated by garbage 200–935× deltas (near-zero ROS IP denominator) **and** every pitcher shows ".000 AVG / 0 HR" (`is_hitter or 1` coercion). | Gate out `ip_proj < 15`/`pa_proj < 50`, `np.clip` deltas to [-3,+3]; fix the `is_hitter` coercion. |
| B-15 | Leaders | Breakouts tab non-functional — all Statcast NULL, every player scores exactly 50.0 with no failure indication. | When Statcast is NULL, show a "data not loaded" empty state; clearly label the traditional-stat fallback. |
| B-16 | Player Databank | IP renders "56.7" instead of "56.2" for every pitcher in every view (decimal `.1f` instead of outs notation). | Add an IP→outs format branch (reverse of `_ip_outs_to_decimal`). |
| B-17 | Player Databank | "Today (live)" view shows ROS projections, not live stats — false label; bad start/sit risk. | Route `type="live"` to a real live fetch, or rename the view "ROS Projections." |

*(Note: the per-page severity lists yielded 15 items the agents tagged `[BLOCKER]` outright; B-05 is added here as the consolidated trust-blocker for Pitcher Streaming, and the Databank carries two — B-16/B-17 — and Leaders two — B-14/B-15 — and Trade Analyzer two — B-08/B-09 — and Trade Finder two — B-10/B-11. The headline count of "15 BLOCKERs" in Part I refers to the distinct page-tagged BLOCKERs; this table expands a few that bundle two root causes.)*

## VI.2 HIGH defects (by page)

| Page | HIGH defects |
|------|--------------|
| Home | SGP category chips always empty; tier always "Tier 1"; reason always identical; pick-score anomaly (Jonah Cox/Wes Clarke ADP 999 outrank Aaron Judge); manual SGP inputs missing OBP and L. |
| My Team | Statcast Signals + Regression Alerts silently absent (NULL Statcast); stale matchup (~4,012 min) shown as "Live"; Record/Rank show "—" despite data in `league_standings`; NaN sentiment bypasses None guard (dot missing on all news); roster caption "Updates hourly" hardcoded/false. |
| Lineup Optimizer | IP Budget card blank on first load; Start/Sit empty if optimizer hasn't run (no message); Start/Sit can't compare pitchers; DCV "Why?" values have no scale; `st.toast` emoji icons (lock violation); "Refresh Yahoo Data" invisible to non-admins with no explanation. |
| Closer Monitor | 9 teams with active closers missing, no explanation (header still says "30-team"); headline "SV" is the projection, not actual (two SV numbers, unlabeled); % JOB uses projected SV → misleading low scores; SETUP row empty on 19/21 cards. |
| Pitcher Streaming | 17-column board, no legend/tooltips/intro; negative Net SGP on a >50 score, unexplained; Track Record proxy-caveat shown after the Run click; no page-level data-freshness indicator. |
| Matchup Planner | Week navigator capped at 24 of 26; `FIG.02` caption on a `FIG.05` page; no data-freshness indicator; 10K-sim MC runs synchronously + uncached on every load. |
| League Standings | Category grid has no legend; Streak column always empty (dead); Win% renders ".318" (reads as a batting average); Scenario Explorer W/L/T inputs mislabeled (category-wins vs record-wins). |
| Punt Analyzer | Ghost team "Twigs" inflates ranks to /13 (UI hardcodes /12); standings-points formula hardcodes 13; row-by-row SGP 413× slower than `total_sgp_batch` (~2.1s blocking); "punt" never defined; no roster-specific view. |
| Trade Analyzer | Grade range (D–B+, low confidence) computed but never shown; verdict ordering (playoff sim/CARA above the ACCEPT/DECLINE banner); MC tooltip claims "200 sims" (engine runs 10K); MC convergence "marginal" warning silently dropped; IP-floor "0.9 IP/week" warning fires on every trade with no explanation; `enable_mc=True` with no timeout/opt-out/partial. |
| Trade Finder | "Sort by Acceptance" sorts strings (High<Low<Medium); mixed types in Acceptance column break the sort; ECR Fairness "0%" on all 2-for-1; spinner says "~5 sec" for a ~2-minute wait; `TIER_COLORS` banned off-palette hexes; "Your Gain" raw floats (not `format_stat`). |
| Free Agents | `player_news` IL-protection query throws every load (no `player_name` column); slash-delimited position filter never matches "2B/SS"; "Budget: 5/week" hardcoded (actual 10); "Show all 7770" browser-freeze/OOM trap; Curtis Mead 125.19 outlier (wrong team); silent `except: pass` on the streaming section. |
| Player Compare | Injury-badge HTML renders as literal text; radar shows 6 dead axes for same-type pairs; inverse-stat z-scores show negative numbers "winning"; cross-type comparison meaningless (no warning); `compute_category_fit()` never called; YTD data missing for stars (Judge all "--"), no indication. |
| Leaders | Prospects `fg_rank` NULL (blank Rank) for all 20; all 20-80 scouting grades NULL (panel always empty); trend/breakout computations uncached on every tab switch; both filter cards always visible regardless of active tab. |
| Player Databank | "Waivers Only" silently returns all players; "Util" filter always 0 rows; 4 stub stat views identical to default; "2026 Advanced" never shows Statcast columns; 96% of players show only dashes (no game-log coverage); `load_databank` uncached (1.4–3.6s per Search). |
| Draft Simulator | "Undo Last Pick" undoes the AI's pick, not the user's; "View player card" selectbox prints raw `<img>` HTML; `PRESEASON` framing + zero in-season context; `combined_score` badge opaque. |

## VI.3 Count summary

| Severity | Count |
|----------|:-----:|
| **BLOCKER** (distinct page-tagged) | **15** (expanded to 17 rows above where a single page bundles two root causes) |
| **HIGH** | **~75** across all 15 pages |
| **MEDIUM** | **~120** (see Part V) |
| **LOW / POLISH** | **~150** (see Part V) |

**The gating set for Beta is the 15 BLOCKERs plus the identity-class HIGHs** (Lineup Optimizer team-name weights, Player Compare / Free Agents roster status, My Team Record/Rank). Closing those alone removes every "this is obviously broken / it doesn't know my team" first-impression. The remaining HIGHs (freshness comms, jargon, hidden errors, formatting) are the second wave that turns "shippable" into "good."

---

# Part VII — Appendix

## VII.1 Methodology details

**Structure.** Fifteen independent test-user agents each owned exactly one page and produced one detailed report under `docs/design-audit/page-NN-*.md`. This master report is the synthesis of all fifteen plus the shared rules in `docs/design-audit/_SHARED_CONTEXT.md`. Each agent worked the page as the novice persona (*Connor*, Team Hickey), exercising every tab, button, input, expander, and table, and recording the **actual** rendered text and numbers rather than guesses.

**Evidence sources.** Two reliable channels were combined:
1. **Source reading** — the full page source plus its supporting engine modules (e.g. `src/optimizer/*`, `src/engine/*`, `src/trade_finder.py`, `src/leaders.py`, `src/closer_monitor.py`, `src/valuation.py`, `src/ui_shared.py`), traced control-by-control to determine exactly what each widget renders and which function it calls.
2. **Read-only DB queries** — against the live league SQLite DB (`data/draft_tool.db`) via the sanctioned `get_connection()`, with a **network guard** installed so any engine/data call fell back to cache/seed instead of hanging on an offline Yahoo/MLB/FanGraphs fetch. Tables queried included `league_rosters`, `league_standings`, `league_records`, `league_matchup_cache`, `season_stats`, `game_logs`, `players`, `projections`, `statcast_archive`, `ownership_trends`, `prospect_rankings`, `refresh_log`, `transactions`, and `yahoo_free_agents`.

**Live state.** The orchestrator captured the live rendered state of each page during a single browser pass on the running Railway instance (`MULTI_USER=1`, logged in as the QA admin). Individual agents did **not** drive the browser (one shared browser, fifteen agents — contention would corrupt each other's sessions).

**Heavy compute.** Monte Carlo and simulation paths were either traced statically or run at tiny sim counts (20–30) to avoid burning minutes; the headline timing figures (Trade Analyzer 44.77s at 10K sims; Trade Finder 43.8s scan; Punt 2.1s vs 0.003s) come from isolated benchmark runs, not the live page.

**Labeling discipline.** Any output that required the Streamlit runtime (`st.session_state`, widget state, dialogs) and could not be produced standalone was **reconstructed** from the DB data + the source logic and is explicitly marked **"(reconstructed)"** in the per-page reports; the master report preserves that labeling in the output tables.

**Grading.** Per-page grades (A–F) weight **trust and novice clarity as heavily as raw capability** — a page showing provably wrong numbers without warning scores in the D band regardless of how sophisticated its engine is. Severity tags follow the shared rubric: `[BLOCKER]` (trust-destroying or connection-dropping), `[HIGH]` (wrong/misleading output, hidden error, identity bug, missing freshness comms, or novice-blocking jargon at the point of decision), `[MEDIUM]` (real friction), `[LOW]`/`[POLISH]` (cosmetic/consistency).

## VII.2 Agent roster

| Role | Page / scope |
|------|--------------|
| Page-01 auditor | Draft Tool (Home) — `app.py` |
| Page-02 auditor | My Team — `pages/1_My_Team.py` |
| Page-03 auditor | Lineup Optimizer — `pages/2_Line-up_Optimizer.py` |
| Page-04 auditor | Closer Monitor — `pages/3_Closer_Monitor.py` |
| Page-05 auditor | Pitcher Streaming — `pages/4_Pitcher_Streaming.py` |
| Page-06 auditor | Matchup Planner — `pages/5_Matchup_Planner.py` |
| Page-07 auditor | League Standings — `pages/6_League_Standings.py` |
| Page-08 auditor | Punt Analyzer — `pages/10_Punt_Analyzer.py` |
| Page-09 auditor | Trade Analyzer — `pages/11_Trade_Analyzer.py` |
| Page-10 auditor | Trade Finder — `pages/12_Trade_Finder.py` |
| Page-11 auditor | Free Agents — `pages/14_Free_Agents.py` |
| Page-12 auditor | Player Compare — `pages/16_Player_Compare.py` |
| Page-13 auditor | Leaders — `pages/17_Leaders.py` |
| Page-14 auditor | Player Databank — `pages/19_Player_Databank.py` |
| Page-15 auditor | Draft Simulator — `pages/20_Draft_Simulator.py` |
| Master compiler | This report — synthesis of all 15 page reports + shared context |

## VII.3 Honest caveats

- **The audit ran against cached / offline data.** At test time Yahoo was "Warming up," the most recent successful refresh was ~1,219 minutes old, and the matchup cache was ~64 hours (≈4,012 min) stale. This is genuinely how a read-only member session looks on Railway, so it is a valid lens — but it means some findings (e.g. the matchup ticker hiding, "Cats in play" showing "no matchup data," the Matchup Planner's all-"Avoid" grid) are **most severe in the offline state** and would partially self-resolve when Yahoo reconnects. The *communication* failures around staleness, however, are real in every state.
- **Several outputs are reconstructed, not live-clicked.** Because agents worked from source + read-only DB rather than driving the shared browser, a subset of the captured numbers (hero cards, MC results, dialog contents, some tables) are reconstructed from the DB data and the source logic. These are flagged "(reconstructed)" in the per-page reports. Where a live capture and a reconstruction disagreed (e.g. the Pitcher Streaming Stream Score of 56.10 live vs 38.7 reconstructed), the discrepancy is itself reported as a finding (data-state sensitivity).
- **The severity counts are approximate.** A handful of cross-cutting items (data-freshness, `st.components.v1.html`, jargon) recur across pages; they are counted once per page where an agent flagged them, so the ~75 HIGH / ~120 MEDIUM / ~150 LOW totals are order-of-magnitude, not exact.
- **The Statcast / ownership / prospect data being NULL may be a transient bootstrap state**, not a permanent code defect — but the **UI's failure to degrade gracefully** when that data is missing (rendering 50.0 scores, blank `--` columns, and silently-vanishing cards instead of honest empty states) is a code-level finding that stands regardless of when the data backfills.
- **The AI chat panel and the three admin pages were excluded by request** and were not opened, tested, counted, or critiqued anywhere in this document. Findings about the floating chat panel, Admin Console, Admin Controls, or Usage Analytics are out of scope and absent by design.
- **This is a UI/UX audit, not a correctness audit of the math.** The engines were assumed correct except where a UI-visible output contradicted itself or reality (e.g. a positive trade grade beside a negative surplus, Soto as "Replacement," pitchers with hitter stats). The recommendations target the wiring, presentation, performance gating, and explanatory layer — not the underlying models, which are the product's genuine strength.

---

*End of report.*
