# Page 09 — Trade Analyzer — Test-User Report

**Auditor persona:** Connor, novice fantasy manager, Team Hickey (3-7-1, 10th of 12).
**Date of audit:** 2026-06-13.
**Sources consulted:** `pages/11_Trade_Analyzer.py` (1,331 lines), `src/engine/output/trade_evaluator.py` (2,251 lines), `src/trade_intelligence.py` (2,001 lines), `src/ui_shared.py` (METRIC_TOOLTIPS, render helpers), live SQLite DB (read-only queries), timed benchmark runs.

---

## 2. Page Purpose & First Impression

The Trade Analyzer is the most technically complex page in HEATER. Its purpose: let the user propose a trade (players they give vs. players they receive), then run a 6-phase engine pipeline to produce a letter grade, verdict (ACCEPT/DECLINE), confidence %, per-category impact, Monte Carlo risk metrics, playoff probability shift, weekly H2H matrix, acceptance probability estimate, and secondary diagnostics.

**First impression for a novice:**
The page takes roughly 5 seconds to fully load before you can interact at all — that alone raises a red flag. You see two big multiselect boxes ("You Give" / "You Receive") and a blue "Analyze Trade" button. That part is clean and self-evident. Clicking the button and waiting is where the experience breaks down. When the page finally returns results, it dumps ~20 distinct sections of technical output — CARA utility, Var[Δchamp], Rosenof G-score, VORP/PRP delta, reshuffle pct — with no guidance on what to look at first or what to do. A novice manager has no reference frame for most of these metrics.

---

## 3. Methodology

1. Read all 1,331 lines of `pages/11_Trade_Analyzer.py` in full.
2. Read key sections of `src/engine/output/trade_evaluator.py` (the full `evaluate_trade` function and all helpers).
3. Read `src/trade_intelligence.py` (health adjustment, scarcity flags, acceptance probability).
4. Queried live SQLite DB: Team Hickey roster (27 players), all 12 team names, standings category values, projection system row counts, refresh log.
5. Timed all upfront page loads (load_player_pool, get_health_adjusted_pool, apply_scarcity_flags, rosters) via isolated Python benchmark.
6. Ran `evaluate_trade()` with `enable_mc=False` for a real 1-for-1 trade (Dansby Swanson → Bryan Reynolds) and captured full output.
7. Ran `evaluate_trade()` with `enable_mc=True`, `n_sims=10_000` on the same trade to benchmark MC cost.
8. Inspected all output keys vs. what the page actually renders to identify silently-dropped fields.
9. Checked METRIC_TOOLTIPS for accuracy vs. current engine behavior.
10. Identified emoji usage, FIG numbering, and design system compliance.

---

## 4. Feature & Control Inventory

| Control | Type | What It Does | Tested? |
|---------|------|-------------|---------|
| Page header (TRADES / FIG.11 — TRADE EVALUATION ENGINE) | Static display | Eyebrow + orange-period wordmark | Yes |
| Recommendation banner ("Analyze a trade below") | Static banner | Static text only (no expanded content) | Yes |
| Matchup ticker | Dynamic bar | Shows this-week matchup (Yahoo-connected); falls back to nothing if offline | Yes (offline) |
| "You Give" multiselect | Multi-select input | Picks players from Team Hickey's roster to give away | Yes |
| "You Receive" multiselect | Multi-select input | Picks players from other teams' rosters | Yes |
| "Analyze Trade" button | Primary button | Triggers the full 6-phase engine | Yes (real trade) |
| Progress bar (0→100%) | Progress indicator | 4-step fake progress during evaluation | Yes |
| Trade Verdict context card (sidebar) | Dynamic card | Shows ACCEPT/DECLINE, grade letter, confidence % | Yes (reconstructed) |
| Punted Categories context card | Conditional sidebar card | Lists categories zero-weighted in eval | Yes (reconstructed) |
| Surplus SGP context card | Dynamic sidebar card | Shows net SGP gain/loss with color | Yes (reconstructed) |
| Verdict banner | HTML card | ACCEPT/DECLINE icon + grade + confidence heatbar + hero-num SGP | Yes (reconstructed) |
| IP floor error/warning | st.error / st.warning | Shows if post-trade weekly IP < 20 IP/wk | Yes (real: triggered) |
| Secondary diagnostics expander | st.expander | VORP delta, G-score delta, specialist cap, DRL chain | Yes (reconstructed) |
| Standings quality warning | st.warning / st.caption | Warns when standings are missing or zero | Yes (reconstructed) |
| Analytics badge | Component | Pipeline transparency (data quality, module status) | Yes (reconstructed) |
| 5-column metrics row | st.metric | Trade Grade, Surplus SGP, Roster Move, Replacement Penalty, Punted Cats count | Yes (real values) |
| Punted categories info | st.info | Lists punted categories by name | Conditional |
| MC Risk section | st.metric × 4 | MC mean, MC std, CVaR5, P(trade helps) + 95% CI caption | Yes (benchmarked) |
| Three-Horizon Impact | st.metric × 3 | Pre-deadline, Full regular season, Playoff window Δ cat-wins | Yes (reconstructed) |
| Weekly H2H impact expander | st.expander + Plotly heatmap | Per-week × per-category win-prob delta matrix | Yes (reconstructed) |
| Acceptance Analysis section | st.metric × 4 | Acceptance Probability, ADP Fairness, ECR Fairness, Tier label | Yes (reconstructed) |
| ADP & ECR Detail expander | st.expander + DataFrame | Per-pair ADP fairness, ECR ranks for ≤3×3 trades | Yes (reconstructed) |
| Category Impact table | render_compact_table | Per-category SGP change, rank, gap-to-next | Yes (real values) |
| Category Replacement Difficulty table | render_styled_table | Raw loss, best FA, unrecoverable gap, SGP penalty | Conditional |
| Risk Flags section | st.warning × N | One banner per risk flag | Yes (real: 3 flags fired) |
| Player cards (Giving/Receiving) | HTML rows | Headshot + team logo + injury badge + name | Yes (reconstructed) |
| Player card selector | render_player_select | Selectbox to open detailed dossier dialog | Yes |
| Upside/Downside Risk table | render_styled_table | P10/P90 HR and AVG from cross-system volatility | Conditional |
| Playoff Sim section | st.metric × 3 + bars | P(playoffs), P(championship), E[wins], heat bars | Conditional |
| CARA / CVaR utility section | st.metric × 3 + caption | CARA utility, CVaR20, Var[Δchamp], λ-sweep caption | Conditional |
| Page timer footer | Static | Load time in milliseconds | Yes |
| Feedback widget | Popover | Per-feature feedback (MULTI_USER mode) | Yes |

---

## 5. Feature-by-Feature Test Log with Real Outputs

### 5.1 Upfront Page Load (before any trade is built)

**Measured times** (live DB, network blocked, no Yahoo live calls):

| Step | Time |
|------|------|
| `load_player_pool()` | **4.32 s** |
| `get_health_adjusted_pool(pool)` | 0.52 s |
| `apply_scarcity_flags(pool)` | 0.00 s |
| `LeagueConfig()` | 0.000 s |
| `yds.get_rosters()` (SQLite cache) | 0.01 s |
| **Total upfront before user sees controls** | **~4.9 s** |

Every page visit — even just visiting the page to look at the controls — loads 9,888 player rows and runs `get_health_adjusted_pool` (which iterates the entire pool twice: once for status adjustment, once for rate-stat recalculation). This happens unconditionally on every rerun/navigation. The user sees a blank page with a spinner for nearly 5 seconds before the two multiselect boxes appear.

### 5.2 "Analyze Trade" button — real trade execution

**Trade tested:** Give Dansby Swanson (SS, Team Hickey, player_id=47) → Receive Bryan Reynolds (LF, PIT, player_id=63).

**Phase 1 only (MC=False):**

| Metric | Value |
|--------|-------|
| `evaluate_trade()` wall time | **2.44 s** |
| Grade | **C** |
| Verdict | **DECLINE** |
| Surplus SGP | **-0.086** |
| Confidence | **43.1%** |
| Punt categories | `[]` (none) |
| Lineup constrained (LP used) | `True` |

**Category impact (real outputs):**
| Category | Δ SGP |
|----------|-------|
| R | -0.562 |
| HR | -0.077 |
| RBI | -0.562 |
| SB | -0.286 |
| AVG | +0.406 |
| OBP | +1.146 |
| W | 0.000 |
| L | 0.000 |
| SV | 0.000 |
| K | 0.000 |
| ERA | 0.000 |
| WHIP | 0.000 |

**Risk flags generated (3):**
1. "Weekly IP (0.9) already below 20 IP/week Yahoo floor; trade doesn't worsen it but pitching cats are at risk"
2. "Caution: Trade worsens 3 categories (R, RBI, SB)."
3. "Lineup reshuffle accounts for 2188% of surplus — promoting Mike Trout from bench. You may capture most of this value by setting your lineup, without making the trade."

**Secondary diagnostics:**
- `delta_vorp_prp`: +3.392 (secondary check says trade is fair league-wide)
- `delta_g_score`: -2.086 (variance-adjusted says DECLINE)
- `grade_range`: `{grade: 'C', grade_low: 'D', grade_high: 'B+', confidence: 'low'}` — **NOT rendered in page**
- `ip_floor_detail`: before/after both 0.87 IP/week (far below 20 floor — see §6 for why)
- `reshuffle_pct`: **2,188%** (the LP is promoting Mike Trout who was benched; nearly all measured SGP is from lineup correction, not the actual player swap)

**With MC=True (10,000 sims, page's production setting):**

| Metric | Value |
|--------|-------|
| `evaluate_trade()` wall time | **44.77 s** |
| MC mean | -2.184 |
| MC std | 3.166 |
| CVaR5 | -8.879 |
| P(trade helps) | 24% |
| Convergence quality | **marginal** |

**Total page time after "Analyze Trade" click (estimated):**
- Phase 1: ~2.5 s
- Phase 2 (MC 10K): ~42 s
- Schedule/roster DB queries: ~0.1 s
- Playoff sim (20K sims): not benchmarked (additional unknown seconds)
- **Conservative floor: 45+ seconds before results appear**

### 5.3 Player selection dropdowns

- "You Give": Populated from Team Hickey roster (13 hitters, 14 pitchers = 27 entries). Real names confirm: Alex Bregman, Andres Gimenez, Angel Martinez, Bryan Reynolds, Cal Raleigh (IL10), Dansby Swanson, Dillon Dingler, Jackson Merrill, Jakob Marsee, Matt Olson, Max Muncy (LAD), Mike Trout, Yordan Alvarez, plus 14 pitchers.
- "You Receive": Populated from all other 11 teams' rosters (approx. 270 players, de-duped vs pool). Names come from the pool's `player_name` column, filtered to `other_team_pids`.

**Issue:** The "You Give" list contains IL-60 players (Garrett Crochet, Shane Bieber) — you cannot realistically trade an IL-60 player until they return. No IL badge or "(IL60)" suffix in the dropdown. A novice would try to trade Crochet and only discover the problem if/when Yahoo rejects it.

### 5.4 IP floor warning (triggered in real test)

The page shows: `"⚠ Weekly IP (0.9) already below the 20 IP/week Yahoo floor..."`

**Root cause of the 0.9 IP/week figure:** ROS IP for the starter corps is nearly zero because most projected full-season IP have been exceeded mid-season (ytd_ip > projected_ip → ROS IP = max(0, ip − ytd_ip) = 0). The pitchers with non-zero ROS IP are: Framber Valdez (1.16/wk), Raisel Iglesias (0.80/wk), Shane Bieber (1.20/wk), Andrew Alvarez (2.67/wk). The LP assigns starters from these and the total is 0.87/wk. This is a **real data problem** — the blended projection system's full-season IP values are stale vs. YTD actuals — not an engine bug. But the user sees a cryptic "0.9 IP/week" warning with no explanation of what it means for them.

### 5.5 Acceptance Analysis (reconstructed)

The page computes `estimate_acceptance_probability` from `src/trade_finder.py` and renders:
- **Acceptance Probability** (e.g., ~30-60% for typical trades)
- **ADP Fairness** (how draft-position-matched are the players)
- **ECR Fairness** (expert-rank alignment)
- **Acceptance Tier** ("High" / "Medium" / "Low")

The opponent SGP estimate uses a hard-coded approximation: `opp_sgp = -user_sgp * 0.5`. This is a rough guess, not a principled calculation. It directly affects the `need_match` variable and therefore the acceptance probability output. Users who see "Medium (47%)" don't know that half the calculation is `opp_sgp = -(-0.086) * 0.5 = 0.043`.

### 5.6 Category Impact table (real outputs)

Rendered with `render_compact_table()` showing per-category SGP change, your standings rank, and gap-to-next. The pitching categories (W, L, SV, K, ERA, WHIP) all show **0.000** for this hitter-for-hitter trade, which is technically correct but looks like a rendering bug to a novice — "did the engine skip all pitching categories?"

### 5.7 Risk flags (real)

Risk flag 3 contains a Unicode encoding issue in the terminal output (`\U0001f` mojibake from the emoji in the page source at line 887). The "2188%" reshuffle warning is mathematically correct but extraordinarily alarming — the actual surplus is -0.086 SGP so 1.885 SGP of reshuffle effect is 2,188× larger. The denominator is essentially zero, making this percentage meaningless and confusing.

---

## 6. Errors, Issues & Difficulties

### P0 — PERFORMANCE BLOCKER: Analyze Trade button hangs 45+ seconds

The page runs `enable_mc=True` with 10,000 sims by default (line 343). The MC simulation took **44.77 seconds** in benchmark. Add 20,000-sim playoff simulation (not measured but materially expensive) and the total time for a single trade evaluation is likely **50-70+ seconds**. During this entire window, the user sees a fake 4-step progress bar that jumps from 20% → 40% → 60% → 100% in about 1 second (the progress updates are synchronous on the Python side before the actual compute starts), then nothing for 40+ seconds. The orchestrator confirmed this already caused the browser WebSocket to drop in live testing (page froze). A beta user gets a white/frozen page with no feedback. This is a beta-launch blocker.

The convergence diagnostic returned **`convergence_quality: "marginal"`** even for 10,000 sims, which means the uncertainty bands in the MC output are not stable — yet the results display with 4 decimal places of precision (e.g., `CVaR5: -8.879`). No UI warning about marginal convergence.

### P1 — 5-second blank page on every navigation

`load_player_pool()` (4.32 s) + `get_health_adjusted_pool()` (0.52 s) run unconditionally on every Streamlit rerun, including page navigation, widget interaction, and any state change. There is no `@st.cache_data` decorator protecting these expensive calls on this page. The Free Agents page caches its pool; this page does not.

### HIGH — Reshuffle percentage overflow (2,188% with encoding corruption)

When `total_surplus` is near-zero (e.g., -0.086 SGP) and `reshuffle_sgp` is substantial (1.885 SGP), the reshuffle percentage blows up: `1.885 / abs(-0.086) × 100 = 2,188%`. The risk flag text "accounts for 2188% of surplus" is technically correct per the code but completely uninterpretable by a user. Additionally, the flag message contains a Unicode replacement character (`?`) in certain terminal/display environments — an emoji in the expander label (line 887: `"📅 Weekly H2H impact..."`) is not sanitized consistently.

### HIGH — Grade range never shown to user

The engine computes `grade_range = {grade: 'C', grade_low: 'D', grade_high: 'B+', confidence: 'low'}` for the Swanson-Reynolds trade. This tells the user the trade could be graded anywhere from D to B+, and the confidence band is **low**. This is arguably the most honest signal from the engine. The page silently drops it — only the point estimate grade `'C'` appears. A user confidently declining a C-grade trade doesn't know the true range spans D to B+.

### HIGH — MC tooltips still say "200 simulated seasons"

`METRIC_TOOLTIPS["mc_mean"]` reads: *"...average Standings Gained Points change across **200** simulated seasons."* `METRIC_TOOLTIPS["trade_verdict_legacy"]` reads: *"...based on **200 Monte Carlo simulations**."* The engine runs **10,000 sims** (single-trade path, page line 349 uses `n_sims=10_000` default). The displayed tooltip is 50× wrong.

### HIGH — IP floor warning uses completely implausible value (0.9 IP/week)

The stale projection data causes all but 4 pitchers to show zero ROS IP (projected season IP already exceeded by YTD actuals). The page warns about "0.9 IP/week" — a figure no user can interpret as meaningful. A novice will not understand why 14 pitchers on their roster contribute less than 1 inning per week in the model. The real issue (stale blended projections) is never surfaced; only the symptom (low IP) appears.

### HIGH — Enable MC=True with no streaming or timeout guard

The page unconditionally sets `enable_mc=True` (line 343). There is no time cap, no timeout, no ability for the user to abort. A 45-second synchronous Python computation on the main thread during a Streamlit `st.button()` handler will eventually hit the server's WebSocket keep-alive timeout and disconnect. This is the confirmed cause of the live-testing browser freeze. The MC run should be either: (a) opt-in via a checkbox the user controls, or (b) run with a time budget cap that falls back to a faster sims count if the budget is exceeded.

### MEDIUM — Emoji in Weekly H2H expander label (design system violation)

Line 887: `f"📅 Weekly H2H impact (Feature 2 — {len(_wm['summary'])} weeks)"`. The Combustion design system prohibits emoji (`tests/test_combustion_lock.py` checks `inject_custom_css()` output for emoji-free content, and the CLAUDE.md says "No emoji — all icons are inline SVGs from `PAGE_ICONS`"). This is an off-system emoji in an expander label.

### MEDIUM — "Feature 2" and "Feature 1" / "report Q(b)" labels exposed to users

Multiple places in the output reference internal engineering labels: "Feature 2 — Weekly H2H impact", "Primary Objective — Δ Title Odds", "Feature 1 (2026-05-23)", "IP-floor status indicator", "report B.9", "report B.10 + Q(a)", "report C.5/H.9". These are spec cross-references for the development team, not consumer UI copy. A novice user sees "report B.9" in a metric tooltip and is completely lost.

### MEDIUM — Confusing output ordering (playoff sim FIRST, verdict SECOND)

The playlist sim section ("Primary Objective — Δ Title Odds") renders before the main verdict banner ("ACCEPT/DECLINE + grade"). For a novice: you see complex probability metrics for a specific future scenario (playoffs, championships) before you even know if the engine says the trade is good or bad. The verdict should come first, then the supporting analysis.

### MEDIUM — Convergence quality never surfaced in UI

The MC result includes `convergence_quality: "marginal"` for this trade's 10,000 sims. The engine even adds a risk flag when quality is marginal/poor (`risk_flags.append(...)`), but this risk flag lives in `mc_result.setdefault("risk_flags", [])` — a separate list from the main `result["risk_flags"]`. The main page only renders `result["risk_flags"]`, so MC convergence warnings are silently swallowed.

### MEDIUM — Acceptance probability formula uses hardcoded `opp_sgp = -user_sgp * 0.5`

Line 1072: `opp_sgp = -user_sgp * 0.5  # Rough estimate: opponent loses ~half what user gains`. Then `need_match = min(1.0, max(0.0, (opp_sgp + 1.0) / 2.0))`. For a trade with surplus_sgp = -0.086, this gives `opp_sgp = 0.043` and `need_match = 0.52`. This magic number directly determines the acceptance probability metric. The approximation is documented as "rough" in a comment but is presented with false precision to the user as "Acceptance Probability: 54%". No uncertainty range is shown.

### MEDIUM — Data freshness not communicated on this page

Yahoo is "Warming up" (1,219 minutes stale per the shared context). The Trade Analyzer fetches `yds.get_rosters()` (served from SQLite cache) and uses this roster data to populate the "You Receive" dropdown. There is no banner, caption, or freshness indicator telling the user "these player lists are from data last refreshed 20 hours ago." A player recently traded between real-MLB teams could appear on the wrong team in the dropdown. The `render_data_freshness_card()` in the sidebar context panel is the only signal — and it's buried in the sidebar, not in the main content where the dropdowns live.

### MEDIUM — slideUp CSS animation on persistent card (Combustion lock violation)

Line 592: `"animation:slideUp 0.4s ease-out both;"` on the verdict banner `<div class="glass">`. The Combustion Finale lock (`test_combustion_lock.py`) explicitly prohibits entrance keyframes on persistent cards because Streamlit remounts the DOM on every rerun (the animation replays on every click). `test_combustion_lock.py` checks that these six persistent-card class names do NOT have a `slideUp` animation. However this is an inline style, not a class — it may escape the guard depending on the exact test implementation. The behavior is wrong either way: clicking any other button causes the verdict banner to slide in again.

### LOW — Upside/Downside Risk table uses HR and AVG only

The risk table (lines 1309-1325) shows "10th Percentile Home Runs (Floor)" and "90th Percentile Batting Average (Ceiling)". For pitchers who receive no HR/AVG entries, the table is empty or shows zero. The column names are also extremely long and wrap awkwardly. Choosing only HR and AVG as representative stats is arbitrary — ERA/WHIP would matter more for pitchers.

### LOW — "You Give" dropdown includes IL-60 players without warning

Crochet and Bieber (both IL60 per the live DB) appear in the "You Give" multiselect with no indicator. Yahoo will likely block a trade of an IL-60 player. The UI should at minimum add "(IL60)" to the display name or exclude them from tradeable options.

### LOW — FIG numbering is non-zero-padded

Page header shows `FIG.11` (correct). The internal playoff sim label shows `FIG.11.1`. No zero-padding inconsistency here, but compare to other pages that use `FIG.04`, `FIG.05`. These are different formats across pages but `FIG.11` on this page is consistent with "no zero padding" which conflicts with the pattern established on the first 9 pages.

### LOW — `render_reco_banner("Analyze a trade below", "", "trade_analyzer")` is empty

The banner renders as a flat non-interactive line with just "Analyze a trade below" — no expanded content, no icons, no links. On pages like Free Agents, the banner at least recommends specific actions. Here it adds no value.

### LOW — `st.components.v1.html` — NOT used on this page

Unlike some sibling pages, Trade Analyzer does not use the deprecated `st.components.v1.html` API. No finding here.

---

## 7. UI/UX & Visual Design Critique

### Layout

The two-column layout (narrow context sidebar + wide main) is structurally correct. Before the trade is analyzed, the main column contains the two multiselect boxes and the Analyze button — clean and well-proportioned. Post-analysis, the main column explodes into 20+ sections stacked vertically without any visual grouping hierarchy. There is no "section break," card boundary, or progressive disclosure structure separating major findings (grade + verdict) from supporting evidence (secondary diagnostics, CARA utility) from reference material (weekly H2H matrix).

### Information density vs. novice readability

The output is built for a quant analyst, not a fantasy baseball manager who just learned what SGP means. In a single scroll, the user encounters: Surplus SGP, CARA utility (λ=0.15), CVaR₂₀ (worst-20% Δchamp), Var[Δchamp], Δ VORP (league-wide), Δ G-score (variance-aware), MC mean/std, CVaR₅ (worst-5%), P(trade helps), and a 3-column metric for reshuffle. No metric is labeled "What This Means For You." Every metric is defined in a help tooltip but tooltips require discovery (hover or click the `?` icon).

### Color usage

Color is used consistently for verdict direction (green = ACCEPT, red = DECLINE) across the sidebar card, the verdict banner, the heatbars, and the metric deltas. The hero-number SGP in the verdict banner correctly uses the `.hero-num` class for the gradient fill. However: the grade letter color on the verdict banner uses `T["sky"]` for B-grades (a teal/blue), which does not appear in the Combustion palette as a primary tier color. The design system specifies orange for positive signals, ember for negative, and navy for structural chrome.

### Typography & formatting

- Rate stats (AVG = 0.236, OBP = 0.332) are shown with `.3f` via `format_stat()` — correct.
- SGP values shown as `+.2f` via `format_stat()` — correct.
- ERA/WHIP shown as `.2f` — correct where they appear.
- The "2188%" reshuffle percentage in a risk flag is shown as a string from an f-string (`f"{reshuffle_pct:.0%}"`), which formats as `2188%`. This is monstrously large and should be capped at a display maximum with a note.

### Hierarchy and visual weight

The most important output — the verdict (ACCEPT / DECLINE) — is the **third** thing rendered after the playoff sim section and an `hr-fade` divider. The output ordering should be: verdict first → supporting metrics → advanced diagnostics.

### Empty/partial states

If the "You Give" or "You Receive" multiselect is submitted empty, the user gets `st.error("Select at least one player on each side.")` — simple and correct. The error clears on next interaction. No issues here.

### Mobile

Not audited (desktop-only scope per shared context).

---

## 8. High-Level Recommendations (≥10)

### 1. Fix the WebSocket-killing MC computation — gate it behind opt-in or async [BLOCKER]

**Problem:** `enable_mc=True` with 10,000 sims takes 44+ seconds synchronously in the Streamlit button handler, causing WebSocket disconnects in production.
**Fix:** Change `enable_mc` default to `False`. Add a checkbox control: *"Run Monte Carlo risk analysis (adds ~45 seconds)"*. Alternatively, run the MC in a background thread and return Phase 1 results immediately, appending MC results when ready. At minimum, set a hard wall-clock budget (e.g., 15 seconds) and reduce `n_sims` proportionally.

### 2. Cache the player pool on this page [HIGH]

**Problem:** `load_player_pool()` (4.32 s) + `get_health_adjusted_pool()` (0.52 s) run on every Streamlit rerun, including widget interactions. No `@st.cache_data` decorator.
**Fix:** Wrap both in `@st.cache_data(ttl=300)` or use a session-state guard identical to what `pages/14_Free_Agents.py` uses. The pool is read-only here; caching is safe.

### 3. Show the grade range / confidence band visually [HIGH]

**Problem:** The engine computes `grade_range = {grade: 'C', grade_low: 'D', grade_high: 'B+', confidence: 'low'}` but the page silently drops it. A user sees "C" when the trade ranges from D to B+.
**Fix:** Render the range inline with the grade: e.g., "C (D – B+, low confidence)" or a visual grade band indicator. This is the single most decision-relevant nuance the engine computes.

### 4. Fix the verdict ordering — ACCEPT/DECLINE must come first [HIGH]

**Problem:** The playoff sim section (6+ metrics, heat bars, CARA utility) renders before the main verdict banner. A user sees complex playoff probability analysis before knowing if the trade is good or bad.
**Fix:** Reorder: (1) verdict banner + grade, (2) surplus SGP + category impact, (3) playoff sim / MC risk / weekly matrix as collapsible sections. The "Primary Objective" headline should live in an expander titled "Playoff Impact" below the verdict, not above it.

### 5. Fix stale MC tooltips (200 vs. 10,000 sims) [HIGH]

**Problem:** `METRIC_TOOLTIPS["mc_mean"]` says "200 simulated seasons." The engine uses 10,000. `METRIC_TOOLTIPS["trade_verdict_legacy"]` also says 200.
**Fix:** Update both tooltips to "10,000 simulations" (or "10,000 paired Monte Carlo simulations with Gaussian copula"). While editing, also add a note that the legacy path uses 200.

### 6. Surface MC convergence quality as a visible warning [HIGH]

**Problem:** `convergence_quality: "marginal"` was returned for a 10,000-sim run. The engine adds a risk flag to `mc_result["risk_flags"]` but the page only renders `result["risk_flags"]` (the main list). The MC convergence warning is silently dropped.
**Fix:** After `result.update(mc_result)`, merge `mc_result.get("risk_flags", [])` into `result["risk_flags"]`. Add a `st.caption("⚠ MC convergence is marginal — uncertainty bands may be unreliable")` when `convergence_quality in ("marginal", "poor")`.

### 7. Cap or contextualize the reshuffle percentage [MEDIUM]

**Problem:** "Lineup reshuffle accounts for 2188% of surplus" when surplus is near-zero. The math is correct but completely uninterpretable.
**Fix:** Cap display at 999%, or better: rewrite the message as "Your lineup already has a better option for this slot (Mike Trout on bench). You can gain most of this benefit by adjusting your lineup without making the trade." The absolute SGP of the reshuffle (1.885 SGP) is more meaningful than the percentage.

### 8. Add "(IL10)" / "(IL60)" suffix to player names in the "You Give" dropdown [MEDIUM]

**Problem:** Injured players appear in the Give dropdown without any status indicator. A user might propose a trade of an IL-60 player and only discover the error when Yahoo rejects it.
**Fix:** When populating `give_options`, append the status: e.g., `f"{name} (IL60)"` for IL players. Alternatively, grey them out or move them to the bottom of the list.

### 9. Remove internal engineering labels from user-facing copy [MEDIUM]

**Problem:** "Feature 2", "report Q(b)", "report B.9", "report B.10 + Q(a)", "report C.5/H.9" appear in metric labels, captions, and tooltips visible to users.
**Fix:** Replace all engineering-facing spec references with consumer language. "Feature 2 — Weekly H2H impact" → "Win Probability by Week". "Per the Enhanced Trade Engine report Q(b)" → remove entirely. "CARA utility (risk-adjusted)" can stay but the tooltip should explain it in baseball terms, not finance theory.

### 10. Add a data freshness banner in the main content area (not just sidebar) [MEDIUM]

**Problem:** The "You Receive" dropdown is populated from data that may be 20+ hours stale. Players could appear on the wrong team. The only staleness signal is the sidebar data freshness card, which many users won't read.
**Fix:** Add a small `st.caption` below the "You Receive" multiselect: *"Player rosters last updated [N hours ago]. Recent real-world trades may not be reflected."*

### 11. Expose the grade range as the headline, not a buried secondary metric [MEDIUM]

**Problem:** The engine knows this is a low-confidence grade (could be D to B+). The user sees "C" with full certainty.
**Fix:** Add a subtitle under the grade: "Low confidence — range D to B+" using the `confidence` and `grade_low`/`grade_high` fields from `result["grade_range"]`. This is already computed — it just needs to be rendered.

### 12. Remove or replace the CARA/CVaR metrics section for novice users [LOW]

**Problem:** CARA utility (λ=0.15), CVaR₂₀ (worst-20% Δchamp), Var[Δchamp], and the λ-sensitivity sweep are graduate-level risk metrics. A novice manager cannot use them. They add visual clutter without decision value for the intended user.
**Fix:** Hide this section behind a "Show advanced risk metrics" expander, or add a one-sentence plain-English summary ("This trade carries above-average variance — your championship odds could swing significantly in either direction") as the visible output, with the quant metrics available to expand.

### 13. Fix the Upside/Downside Risk table for pitchers [LOW]

**Problem:** The table shows HR and AVG only. Pitchers show zeros or are excluded. For a mixed hitter-pitcher trade, the table is incomplete or misleading.
**Fix:** Show HR/AVG for hitters, ERA/K for pitchers. Use `is_hitter` to branch display stats per player.

---

## 9. Severity-Tagged Issue List

- `[BLOCKER]` MC computation (10K sims + 20K playoff sims) runs synchronously in the button handler, taking 45+ seconds and dropping the browser WebSocket. Confirmed to have frozen the page in live testing.
- `[BLOCKER]` 5-second blank page on every navigation — no `@st.cache_data` on `load_player_pool()` or `get_health_adjusted_pool()`.
- `[HIGH]` Grade range (`grade_low`, `grade_high`, `confidence`) computed by engine but never shown to user. For the Swanson-Reynolds test trade, the actual range is D to B+.
- `[HIGH]` Verdict ordering: playoff sim and CARA metrics appear above the main ACCEPT/DECLINE verdict banner.
- `[HIGH]` METRIC_TOOLTIPS for `mc_mean` and `trade_verdict_legacy` claim "200 simulations" — the engine runs 10,000.
- `[HIGH]` MC convergence risk flag (`convergence_quality: "marginal"`) generated by engine but silently dropped — never reaches the user.
- `[HIGH]` IP floor warning shows `0.9 IP/week` with no explanation of why (stale projection vs. actual YTD data). The warning fires on every trade evaluation regardless, creating alert fatigue.
- `[HIGH]` `enable_mc=True` with no timeout guard, no user opt-out, and no partial-results display. Must be made opt-in or given a hard time budget.
- `[MEDIUM]` Reshuffle percentage of "2188%" is meaningless and alarming. The risk flag message needs capping or rephrasing.
- `[MEDIUM]` Opponent SGP estimate uses `opp_sgp = -user_sgp * 0.5` magic number, creating a false-precision acceptance probability metric.
- `[MEDIUM]` "Feature 2", "report Q(b)", "report B.9" etc. exposed in user-facing labels and tooltips — internal development language.
- `[MEDIUM]` Emoji in Weekly H2H expander label (`📅`) violates Combustion design system icon policy.
- `[MEDIUM]` `slideUp` animation on verdict banner replay on every rerun (violates Combustion lock prohibition on entrance keyframes for persistent cards).
- `[MEDIUM]` Data freshness not communicated in main content — only in sidebar data freshness card.
- `[MEDIUM]` IL-60/IL-10 players appear in "You Give" dropdown with no status indicator.
- `[MEDIUM]` Overall output is 20+ sections with no visual grouping hierarchy — novice has no path through the information.
- `[LOW]` Upside/Downside Risk table shows HR/AVG only — zeroes for pitchers.
- `[LOW]` `render_reco_banner("Analyze a trade below", "", ...)` contributes nothing (empty expanded content).
- `[LOW]` Several computed engine outputs silently dropped: `grade_range`, `reshuffle_sgp`, `markov_fa_detail`, `concentration_delta`, `flexibility_penalty`, `convergence_quality`. These are computed at cost and discarded.
- `[POLISH]` Context sidebar card label "Surplus Standings Gained Points" is too long for the narrow panel — wraps awkwardly at typical sidebar widths.
- `[POLISH]` The `st.subheader("Acceptance Analysis")` section uses a native Streamlit subheader rather than the page's established section header pattern (Archivo bold + orange left-border rail).
