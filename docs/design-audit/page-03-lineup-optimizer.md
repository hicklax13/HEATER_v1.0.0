# Page 03 — Lineup Optimizer — Test-User Report

**Auditor persona:** Connor — novice fantasy manager, Team Hickey (3-7-1, 10th of 12), FourzynBurn H2H-categories league.
**Audit date:** 2026-06-13
**Source file:** `pages/2_Line-up_Optimizer.py` (3,607 lines)
**Supporting engine:** `src/optimizer/` (21 modules), `src/lineup_optimizer.py` (PuLP LP), `src/start_sit.py`, `src/ip_tracker.py`

---

## 1. Page Purpose & First Impression

**What it's for:** Set a daily or weekly lineup for Team Hickey by running one of three optimization scopes (Today, Rest of Week, Rest of Season). The page also has a Start/Sit advisor for head-to-head dilemmas, a Category Analysis tab showing where each stat comes from, and a Streaming tab with pitcher pickup recommendations.

**First 5 seconds as a novice:**

I land on a page titled "Line-up Optimizer" with a four-tab row ("Optimizer / Start/Sit / Category Analysis / Streaming") and a left sidebar context panel. Immediately I see several widgets whose purpose is unclear:

- A "Mode" radio with "Full / Standard / Quick" — what do these modes mean? Do they change my lineup recommendations or just compute speed?
- A "Risk Aversion" slider at 0.15 — no units, no explanation what 0.15 means.
- A "Matchup State" card and an "IP Budget" card sitting in the sidebar — I'm not sure if these are informational or if I need to act on them.
- The main area for the "Optimizer" tab shows a "Scope" radio with "Today / Rest of Week / Rest of Season" and a big orange "Optimize Lineup" button. That part is clear.

The page feels busy. The left column has 6–8 stacked cards before you even reach the main content. The header says "FIG.02 — LINEUP CONTROL" which to a novice reads as jargon with no meaning.

---

## 2. Methodology

**Source read:**
- `pages/2_Line-up_Optimizer.py` — full 3,607-line read; traced all tabs, widgets, render functions, and session-state keys.
- `src/optimizer/pipeline.py` — pipeline modes, OptimizerResult shape, timing.
- `src/optimizer/daily_optimizer.py` — DCV engine, `build_daily_dcv_table()`, column outputs.
- `src/optimizer/sgp_theory.py` — `compute_nonlinear_weights()`, team name lookup.
- `src/start_sit.py` — `start_sit_recommendation()`, output shape.
- `src/ip_tracker.py` — `WEEKLY_TARGET`, `MIN_IP`, `compute_weekly_ip_projection()`.
- `src/ui_shared.py` — `THEME`, `WEEKLY_IN_SEASON`, `render_context_columns()`, `render_compact_table()`, `render_styled_table()`.
- `src/nav.py` — `PAGE_REGISTRY` to check nav-label vs header drift.

**DB queries (read-only, network-blocked):**

Verified team name stored in DB as `'🏆 Team Hickey'`; confirmed `compute_nonlinear_weights()` receives plain `'Team Hickey'` from the env config — which does NOT match any key in the standings query, so it returns equal weights (1.00x) for all 12 categories. Full roster, matchup, refresh log, and season stats also queried.

---

## 3. Feature & Control Inventory

| Control | Type | What it does | Tested? |
|---------|------|-------------|---------|
| Tab: Optimizer | Tab | Main optimization panel — scope selector + "Optimize Lineup" | Yes (source+DB) |
| Tab: Start/Sit | Tab | Head-to-head advisor for 2-player dilemmas | Yes (source trace) |
| Tab: Category Analysis | Tab | Shows per-category SGP weights and roster contributions | Yes (source+DB) |
| Tab: Streaming | Tab | Pitcher streaming recommendations | Yes (source trace) |
| Mode radio (sidebar) | Radio | Quick / Standard / Full pipeline mode | Yes (traced) |
| Risk Aversion slider | Slider | 0.0–0.50, step=0.05, default=0.15 | Yes (traced) |
| Matchup State card | Display card | Shows current H2H win/loss/tie status per category | Yes (DB) |
| IP Budget card | Display card | Projected weekly IP vs 53.85 target and 20.0 floor | Yes (traced) |
| Win Probability card | Conditional display | Shown only when matchup data available; shows 6-6 projection | Yes (DB) |
| Data Freshness card | Display card | Per-source staleness badge | Yes (DB) |
| Refresh Yahoo Data button | Button (writer-gated) | Triggers `force_refresh_all()` on YDS | Yes (traced) |
| Scope radio (main area) | Radio | Today / Rest of Week / Rest of Season | Yes (traced) |
| Optimize Lineup button | Primary button | Runs DCV or LP pipeline; writes results to session state | Yes (traced) |
| DCV results table | Table (compact) | Per-player DCV Score, Matchup multiplier, Decision | Yes (reconstructed) |
| "Why?" expander per player | Expander | Breakdown of DCV components | Yes (source trace) |
| Start/Sit player 1 selectbox | Selectbox | Pick player A for head-to-head compare | Yes (source trace) |
| Start/Sit player 2 selectbox | Selectbox | Pick player B | Yes (source trace) |
| Start/Sit "Get Recommendation" button | Button | Calls `start_sit_recommendation()` | Yes (source trace) |
| Category Analysis weight table | Table | 12-category SGP weights and urgency scores | Yes (reconstructed) |
| Category Analysis roster table | Table | Per-player per-category projected contribution | Yes (source trace) |
| Streaming recommendation table | Table | Top FA pitcher pickups for streaming | Yes (source trace) |

---

## 4. Feature-by-Feature Test Log

### 4.1 Optimizer Tab — Context Sidebar

**Mode radio (Full / Standard / Quick)**
- Default: "Full" (best mode).
- "Full" runs `LineupOptimizerPipeline(mode='full')` — the source maps this to 5–10s compute time; generates `h2h_analysis`, `streaming_suggestions`, `risk_metrics`, and `maximin_comparison`.
- "Quick" maps to <1s — skips MC and just runs LP.
- **No microcopy anywhere explains what Full vs Quick means for a novice.**

**Risk Aversion slider**
- Default: 0.15; range 0.0–0.50, step=0.05.
- Wired into `pipeline._preset["risk_aversion"]`. At 0.0 the pipeline maximizes expected value; at 0.50 it maximizes the minimum-worst-case outcome (maximin). This is a meaningful strategic choice, but the slider label is bare — no tooltip, no description, no indication this affects anything meaningful.
- Source: `risk_aversion = st.slider("Risk Aversion", 0.0, 0.50, 0.15, step=0.05)`

**Matchup State card (reconstructed from DB)**
Week 12 vs "The Good The Vlad The Ugly". Live category snapshot:
```
Winning:  R (14-6), HR (7-1), RBI (14-8), AVG (.359-.271), OBP (.431-.316), W (1-0), K (21-13)
Losing:   L (2-0), ERA (7.08-3.38), WHIP (1.82-1.03)
Tied:     SB (0-0), SV (0-0)
```
The ERA gap (7.08 vs 3.38) and WHIP gap (1.82 vs 1.03) are severe — pitching performance this week is a notable problem that the optimizer should highlight more prominently.

**IP Budget card (reconstructed)**
- `WEEKLY_TARGET = 1400/26 ≈ 53.85 IP` (from `ip_tracker.py`).
- `MIN_IP = 20.0` (Yahoo forfeit threshold).
- The card shows projected IP from current pitchers, pacing status (safe/warning/danger), and whether streaming is needed.
- **Bug:** the card is populated from `st.session_state["ip_projection"]` which only exists AFTER the first "Optimize Lineup" click. Before that, the card renders empty or with a placeholder. A novice sees an empty "IP Budget" card and does not understand why.

**Win Probability card (reconstructed)**
Shown when matchup data available:
- "Win: 46% / Tie: 19% / Loss: 35%"
- Projected record: 6-6.
- **This is buried in the sidebar below the IP card, below the Data Freshness card.** It is arguably the most important piece of information on the page this week and it is the LAST card the user scrolls to.

**Data Freshness card (from refresh_log)**
Real staleness as queried:
```
projections       ERROR    (all FanGraphs failed 2026-06-10)
ros_projections   ERROR    (2026-06-10)
yahoo_standings   SUCCESS  2026-06-12  (18 min ago)
player_pool       SUCCESS  2026-06-11  (~1 day old)
season_stats      SUCCESS  2026-06-10  (~3 days old)
statcast          SUCCESS  2026-06-10  (~3 days old)
```
The page correctly shows a freshness badge. The degraded state (projection errors) is shown but the visual distinction between "error" and "partial" statuses is not clear in the card UI — both show a yellow/amber indicator that looks similar.

**Refresh Yahoo Data button**
- Gated by `viewer_can_write()` — admin only in MULTI_USER mode.
- Non-admins (future league-mates) see nothing in its place — no label, no explanation why the button is absent.

---

### 4.2 Optimizer Tab — Main Area

**Scope radio: Today / Rest of Week / Rest of Season**

- "Today" → `build_daily_dcv_table()` (DCV engine, <2s).
- "Rest of Week" → `LineupOptimizerPipeline(mode=..., alpha=1.0)` (LP).
- "Rest of Season" → LP with `alpha` auto-computed from `weeks_remaining`.

**Optimize Lineup button — DCV output (reconstructed, "Today" scope)**

The DCV engine (`build_daily_dcv_table`) outputs a table with columns:
`Slot | Player | Position Eligibility | Team | DCV Score | Matchup | Decision`

For Team Hickey's 23 active players on 2026-06-13, Decision values are:
- `START` — optimal start
- `START ⚠` — forced start: matchup multiplier < 0.70 OR total_dcv < median × 0.50
- `BENCH` — recommended bench
- `IL` — player on injured list (Cal Raleigh, Bailey Ober, Garrett Crochet, Shane Bieber)
- `LEAVE EMPTY` — no viable player for slot

Real players expected to show:
- **Matt Olson (1B/OF)** — starter; season: .245/17HR/57RBI
- **Angel Martínez (2B/3B/SS/OF)** — multi-position
- **Yordan Alvarez (OF/1B/Util)** — starter
- **Chris Sale (SP)** — K=86, ERA=2.23 (excellent)
- **Framber Valdez (SP)** — likely returning from suspension
- **Cal Raleigh (C)** — IL10, shows Decision=IL
- **Garrett Crochet (SP)** — IL60, shows Decision=IL

**LP output (reconstructed, "Rest of Week" scope)**

The LP pipeline returns `result.lineup` — a dict of slot → player_id. The page then assembles a similar table but shows projected weekly totals instead of DCV scores. Columns include `proj_r/hr/rbi/sb/avg/obp/w/l/sv/k/era/whip` aggregated across the LP-selected starters.

**Yahoo/LP mismatch banner**

If LP recommends starting a player Yahoo has on BN, an orange banner appears:
> "⚠ LP wants to start [Player] from BN — check Yahoo"

This banner uses ⚠ emoji which violates the no-emoji design system rule.

**"Why?" expander per player (DCV mode)**

Each player row has an expander showing DCV component breakdown:
- `matchup_mult`: Park factor × platoon × wRC+ adjustment
- `volume_factor`: Whether games are scheduled today (0.0 if team is locked/final)
- `health_factor`: IL weight from `_il_weight_from_status()`
- `total_dcv`: Weighted sum

This is well-designed but uses raw decimal numbers with no labels. A novice sees:
```
matchup_mult: 0.8745
volume_factor: 1.0
health_factor: 1.0
total_dcv: 12.34
```
There is no explanation of scale (is 12.34 good? bad? what's the max?).

**`st.toast` calls (design-system violation)**

The page uses:
```python
st.toast("✅ Lineup optimized!", icon='✅')
st.toast("⚠️ Partial results", icon='⚠️')
st.toast("❌ Optimization failed", icon='❌')
```
These use emoji icons — violating the no-emoji rule locked in the Combustion design system. The icon parameter should use a Material Symbol name or be omitted.

---

### 4.3 Start/Sit Tab

**Player 1 / Player 2 selectboxes**

Populated from `st.session_state["optimizer_roster"]` (set after optimize runs) OR from a manual load of the roster. If the user navigates directly to Start/Sit without first running the optimizer, the selectboxes may be empty.

**"Get Recommendation" button**

Calls `start_sit_recommendation(player_ids, player_pool, config, ...)` — source-traced output shape:
```python
{
    "players": [
        {
            "player_id": ...,
            "start_score": float,      # 0-100
            "floor": float,
            "ceiling": float,
            "category_impact": dict,   # per-cat contribution
            "reasoning": str
        },
        ...
    ],
    "recommendation": player_id,       # winner
    "confidence": float,               # 0-1
    "confidence_label": str            # "Clear Start" / "Lean Start" / "Toss-up"
}
```

Example output for a hypothetical Olson vs Swanson Start/Sit (reconstructed):
```
Matt Olson:    start_score=72.4, floor=45, ceiling=92, confidence_label="Clear Start"
Dansby Swanson: start_score=41.3, floor=22, ceiling=58
Recommendation: Matt Olson | Confidence: Clear Start
Category impact: HR (+0.34 SGP), RBI (+0.28 SGP)
```

**Missing feature:** There is no way to compare pitchers in Start/Sit — the selectbox population logic filters for hitters only when the roster is hitter-heavy. A novice trying to decide between two streaming SPs finds no way to use this tab for that decision.

---

### 4.4 Category Analysis Tab

**SGP Weight table (reconstructed from DB bug)**

The `compute_nonlinear_weights()` function is called with `team_name="Team Hickey"` but the DB stores the team as `'🏆 Team Hickey'`. The lookup fails silently and returns equal weights (1.00x) for all 12 categories:

```
R=1.00  HR=1.00  RBI=1.00  SB=1.00  AVG=1.00  OBP=1.00
W=1.00  L=1.00   SV=1.00   K=1.00   ERA=1.00  WHIP=1.00
```

**This means the Category Analysis tab is providing WRONG urgency guidance.** The page shows "all categories equally important" when in reality Team Hickey is:
- **Dominating:** R, HR, RBI, AVG, OBP, W, K
- **Losing badly:** L (2-0 leads opponent), ERA (7.08 vs 3.38), WHIP (1.82 vs 1.03)

The optimizer should be prioritizing ERA and WHIP heavily this week, and it is NOT doing so because the team-name lookup fails.

**Roster contribution table**

Shows each player's per-category projected contribution (SGP units). Uses `render_styled_table()` with `df.to_html()`. This table has no visual hierarchy — all numbers look the same regardless of magnitude. There is no color-coding to show which players are contributing vs dragging in each category.

---

### 4.5 Streaming Tab

The Streaming tab delegates entirely to `src/optimizer/stream_analyzer.build_stream_board()` (canonical engine). The page contains no inline score arithmetic (verified by structural guard `test_stream_page_no_inline_scoring.py`).

**Stream board columns (source-traced):**
`Player | Team | Opp | Date | Stream Score (0-100) | Risk Flags | Actionable`

**Risk flags rendered as text strings:** `HIGH_WHIP / SHORT_LEASH / ELITE_OFFENSE / HITTER_PARK / WIND_OUT / LOW_CONFIDENCE`

These are meaningless to a novice — there is no legend. "SHORT_LEASH" especially is fantasy jargon that needs a tooltip.

**Overlap with Pitcher Streaming page (Page 05):** The Streaming tab here and the dedicated Pitcher Streaming page (which has 4 full tabs including Finder, Microscope, Week Planner, Track Record) overlap considerably. A novice does not know which one to use. The Streaming tab here appears to show a simpler board, while the dedicated page has deeper analysis. This is confusing navigation.

---

## 5. Errors, Issues & Difficulties

### BLOCKER

**B1 — Category weights always return 1.00x (emoji mismatch)**
`compute_nonlinear_weights()` is called with `team_name = "Team Hickey"` (from env config `ADMIN_TEAM_NAME`). The DB stores the team as `'🏆 Team Hickey'`. The SQL query in `sgp_theory.py` does a bare string-equal match that finds no rows and returns equal weights. Result: the Category Analysis tab shows no useful urgency differentiation, and the LP optimizer loses its ability to weight categories by current standing gaps.

Root cause: the same emoji-prefix reconciliation bug fixed in `test_pages_use_viewer_team_resolver.py` was NOT applied to the `compute_nonlinear_weights()` lookup. The `resolve_viewer_team_name()` helper exists but is not called here.

**Impact:** Every optimizer run this week is wrong — ERA/WHIP should be 2–3× weighted vs SB/SV (which Team Hickey is already tying), but the LP treats them equally.

---

### HIGH

**H1 — IP Budget card blank on first load**
`st.session_state["ip_projection"]` is set only after the first "Optimize Lineup" click. On fresh page load the IP Budget card in the sidebar renders as an empty `.context-card` div with no message. A novice sees a titled but empty card and doesn't know what happened.

**H2 — Start/Sit does not work before running the optimizer**
If a user navigates directly to the Start/Sit tab without clicking "Optimize Lineup" first, the player selectboxes are empty (roster not loaded into session state). There is no fallback to load the roster directly, and no message explaining why the selectboxes are empty.

**H3 — "Refresh Yahoo Data" invisible to non-admins (no explanation)**
The button is gated by `viewer_can_write()`. Non-admins see nothing where the button was — no "refresh managed by server" message, no staleness explanation. When a league-mate sees stale data and can't find a refresh button, they will be confused.

**H4 — `st.toast` uses emoji icons (design-system violation)**
Three `st.toast()` calls pass `icon='✅'`, `icon='⚠️'`, `icon='❌'`. The Combustion design system bans emoji everywhere. These should use Material Symbol names or be plain text toasts.

**H5 — Pitcher filter missing from Start/Sit**
`start_sit_recommendation()` accepts any player IDs but the tab's player selectboxes appear to filter primarily by position data that skews toward hitters in the roster. Users wanting to compare two starting pitchers (e.g., Sale vs. Harrison on a given day) have no clean way to do so. The tab should allow filtering by player type (hitter/pitcher).

**H6 — "Why?" expander numbers lack scale context**
DCV component values (`matchup_mult=0.8745`, `total_dcv=12.34`) are shown with no explanation of scale, direction, or what "good" means. A novice cannot interpret these values.

---

### MEDIUM

**M1 — No explanation of Mode (Full/Standard/Quick)**
The Mode radio has no help text. "Full" sounds like "better" but a user on a slow connection who hits Full mode and waits 10 seconds does not know why. Each mode needs a one-liner: "Full (5-10s): uses all data sources and LP optimization" etc.

**M2 — Risk Aversion slider has no tooltip or description**
0.15 is the default. A novice has no idea what risk aversion means in the context of lineup setting. No tooltip, no label units, no "safe/balanced/aggressive" descriptor. This reads as a math parameter, not a UI control.

**M3 — Scope radio vs. Mode radio: which to change first?**
The sidebar has "Mode" (Quick/Standard/Full). The main area has "Scope" (Today/Rest of Week/Rest of Season). These interact — Full mode with "Today" scope runs a different pipeline than Full mode with "Rest of Season" scope — but the UI presents them at different locations with no explanation of how they relate. A novice might set Mode=Full and Scope=Today expecting the best possible daily recommendation, not realizing "Today" always uses the DCV engine regardless of mode.

**M4 — Decision label "LEAVE EMPTY" is alarming but unexplained**
"LEAVE EMPTY" appears when no viable player can fill a slot. A novice sees this and panics — am I not setting a lineup? What happens to my team? There is no tooltip or explanation.

**M5 — Streaming tab duplicates Pitcher Streaming page (page 05)**
The Streaming tab in the Optimizer and the dedicated Pitcher Streaming page serve overlapping purposes. There is no cross-reference telling a novice "for deeper streaming analysis, go to Pitcher Streaming." A novice may not discover page 05 at all.

**M6 — "START ⚠" forced-start flag: no explanation of what "⚠" means**
The forced-start flag fires when `matchup_mult < 0.70 OR total_dcv < median × 0.50`. The table shows "START ⚠" but clicking it does nothing — there is no expander specific to the warning condition. A novice sees the warning and does not know whether to heed it.

**M7 — DCV Score column: no scale indicator or benchmark**
DCV scores appear as raw numbers (e.g., 12.34, 7.81). There is no indication of what's high or low. A percentile rank, color gradient, or "above/below median" label would make these actionable.

**M8 — Category Analysis table: no color-coding or visual hierarchy**
The roster contribution table is a flat grid of decimal numbers in `render_styled_table()`. There is no conditional formatting to show which players are positive vs negative contributors in each category. Especially important for the pitching categories where Framber Valdez's L=2 this week is actively hurting the ERA/WHIP totals.

**M9 — Tab count discrepancy with old documentation**
The CLAUDE.md file references "6 tabs" for the Optimizer (Start/Sit, Optimize, Manual, Streaming, Daily, Roster). The actual page has 4 tabs: Optimizer, Start/Sit, Category Analysis, Streaming. The Manual and Roster tabs were removed but some internal comments in the source still reference them. Not a user-facing bug but causes developer confusion.

**M10 — Alpha parameter hidden and auto-overridden**
`alpha = 0.5` is initialized but immediately overridden: Rest of Week → 1.0, Rest of Season → computed, Today → unused. This is correct behavior but the user has no way to influence the blending between H2H and season-total objectives. For a user in 10th place who needs to win *now*, the hard-coded alpha for "Rest of Week = 1.0 (pure H2H)" is actually the right call, but the user doesn't know that's what they're getting.

---

### LOW

**L1 — Nav-label vs header-title drift**
`src/nav.py` PAGE_REGISTRY entry: `title = "Lineup Optimizer"` (no hyphen).
Page header: `"Line-up Optimizer"` (with hyphen).
Sidebar nav and the page title will differ. Minor but visible.

**L2 — FIG.02 eyebrow on the wrong FIG number**
The page header renders eyebrow `"FIG.02 — LINEUP CONTROL"`. The orchestrator notes `FIG.04` format on Pitcher Streaming (two-digit, no leading zero). This page has two-digit leading-zero format which is correct, but the `LINEUP CONTROL` subtitle is not a particularly informative subtitle — it just restates the page function.

**L3 — Risk flags in Streaming tab are bare strings without a legend**
`HIGH_WHIP`, `SHORT_LEASH`, `ELITE_OFFENSE`, `HITTER_PARK`, `WIND_OUT`, `LOW_CONFIDENCE` — no tooltip, no glossary. Fantasy newcomers need these explained.

**L4 — Projection error not highlighted enough in context panel**
The Data Freshness card shows `projections: ERROR` in amber/yellow. Given that FanGraphs projections have been failing since 2026-06-10 (3+ days), this deserves a more prominent warning — ideally a banner at the top of the page saying "Optimization is running on stale projections — results may be less accurate."

---

### POLISH

**P1 — IP Budget card shows "53.85 IP" with 2 decimal places**
54 or "~54 IP" would be more readable than the `1400/26 = 53.846...` precision. Displaying "53.85" signals false precision and reads as a calculation artifact.

**P2 — Win Probability card is the last card in the sidebar**
The 46%/19%/35% win probability is the most actionable number for a user deciding how aggressively to optimize. It should appear first or near the top of the context panel, not buried below IP Budget and Data Freshness.

**P3 — Matchup State card uses raw category names with no sorting**
Categories are shown in source order (R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP) with no visual grouping (winning/losing/tied). A quick-scan format grouping winning cats green, losing cats red, and tied cats neutral would be far more readable.

**P4 — Empty state before first optimize click is a blank white area**
The main content area of the Optimizer tab before clicking "Optimize Lineup" is just the scope radio and the button. No instructional text, no demo of what the output will look like, no "click to see your optimized lineup" CTA copy. The `render_empty_state()` helper from `ui_shared.py` should be used here.

**P5 — "Refresh Yahoo Data" button label is generic**
The button should say "Refresh Player & Matchup Data" or similar to make it clear what data it refreshes.

---

## 6. UI/UX & Visual Design Critique

### Layout & Hierarchy

The `render_context_columns(context_width=1)` creates a 1:4 column split. The left column is packed with 6–8 stacked cards before any main content appears. On a standard 1440px wide monitor this is fine, but on a 1280px monitor the left column becomes very narrow relative to its content density.

The context panel card order is suboptimal. Priority should be:
1. Win Probability (most actionable)
2. Matchup State (current week context)
3. IP Budget (daily constraint)
4. Data Freshness (system health)
5. Settings (Mode, Risk Aversion)

Currently Mode/Risk Aversion appear first (settings above context) which is backwards — settings should follow the context that motivates them.

### Color Use

- The "Optimize Lineup" button correctly uses the orange `#ff6d00` primary. Good.
- `st.toast` icons use emoji (violation).
- The DCV table uses `render_compact_table()` which applies row classes. The `Decision=START` rows get a visual treatment (likely a green/positive class) and `BENCH` rows a muted class — but `START ⚠` rows should have a distinct orange warning class. Source review suggests they receive the same class as `START`.
- Category Analysis weights at 1.00x for all categories (due to the emoji bug) means the weight visualization is a flat horizontal line — it would look broken even if the bug didn't exist, because equal weights look like "nothing computed."

### Typography & Number Formatting

- DCV scores displayed with 2 decimal places (`12.34`). IBM Plex Mono is correctly applied via `--font-mono` in the compact table. ✓
- SGP values in category analysis: source uses `format_stat(value, stat_type)` which enforces SGP `+.2f`. ✓
- AVG/OBP in matchup cards: displayed as `.359`/`.431` — correct 3-decimal format. ✓
- IP Budget: `53.85` — excessive precision (see P1).

### Density

The 4-tab layout with a dense left panel makes this the most complex page in the app. That density is appropriate given the page's function, but the information hierarchy within each panel needs more breathing room. The context cards stack with minimal vertical spacing.

### Consistency with Sibling Pages

- Page header pattern (eyebrow + title with orange period + underline rule): ✓ present.
- `render_empty_state()` for empty data contexts: NOT used before first optimize click (blank white instead).
- No deprecated `st.components.v1.html` usage found in this page. ✓

### Accessibility

- Slider (Risk Aversion) has no `aria-label` beyond its default Streamlit label — acceptable.
- The compact table HTML uses inline styles for row color — not ideal for screen readers but consistent with the rest of the app.
- "LEAVE EMPTY" in the Decision column has no ARIA description of what it means for the slot.

### Mobile

The 1:4 column layout collapses on mobile (Streamlit renders columns vertically). The left context panel would render above the main optimizer content, pushing the actual "Optimize Lineup" button far below the fold on a phone. For a page that a manager might quickly check on their phone at game time, this is a problem.

---

## 7. High-Level Recommendations (≥10, priority-ordered)

### R1. FIX THE TEAM-NAME LOOKUP (BLOCKER) — Apply `resolve_viewer_team_name()` to `compute_nonlinear_weights()`

**Problem:** The emoji mismatch between `'🏆 Team Hickey'` (DB) and `'Team Hickey'` (env config) silently returns equal weights (1.00x) for all 12 categories. Every LP optimization run and the entire Category Analysis tab are producing wrong outputs.

**Fix:** In `pages/2_Line-up_Optimizer.py`, replace the bare `team_name` string passed to `compute_nonlinear_weights()` with the result of `resolve_viewer_team_name(rosters)` — the same helper called by the other 7 personalized pages per the structural invariant. One-line change, guarded by an existing structural test.

**Impact:** The optimizer correctly weights ERA/WHIP 2–3× vs SB/SV this week, which directly improves lineup recommendations for a team that is 10th of 12.

---

### R2. SURFACE WIN PROBABILITY AT THE TOP OF THE SIDEBAR

**Problem:** The 46%/19%/35% win probability card is the last item in the context panel, below Mode, Risk Aversion, Matchup State, IP Budget, and Data Freshness. A novice scrolls past six cards to find the most important contextual number.

**Fix:** Move the Win Probability card to the top of the context panel, before the settings. Add a week-record line: "Week 12: projected 6-6 | Win 46% | Tie 19% | Loss 35%". This is the one number that should drive all optimizer decisions.

---

### R3. ADD INLINE DESCRIPTIONS FOR MODE AND RISK AVERSION

**Problem:** "Full / Standard / Quick" and the 0.15 risk aversion slider are abstract to a novice. Users cannot make an informed choice.

**Fix:**
- Mode: add caption below each radio option: "Full (5-10s): maximum accuracy, uses all data" / "Standard (2-3s): balanced speed" / "Quick (<1s): skip projections, fast answer".
- Risk Aversion: add descriptive labels at endpoints: "0.0 = Maximize upside (aggressive)" / "0.50 = Minimize downside (safe)". Add a tooltip: "Higher values protect against bad weeks at the cost of expected value."

---

### R4. SHOW EMPTY STATE AND INSTRUCTIONS BEFORE FIRST OPTIMIZE CLICK

**Problem:** The main Optimizer tab before first click is a blank white area with just a radio and a button. No guidance, no example of what will appear.

**Fix:** Use `render_empty_state(title="Ready to Optimize", body="Select a scope and click Optimize Lineup to see your start/sit recommendations.", icon_key="lineup_optimizer")` in the empty state. Optionally show a "last optimized" timestamp from session state if a prior run exists.

---

### R5. FIX THE IP BUDGET CARD TO SHOW ON FIRST LOAD

**Problem:** The IP Budget card is empty until the first optimize click because `st.session_state["ip_projection"]` is not pre-populated.

**Fix:** On page load, call `compute_weekly_ip_projection(roster_pitchers, days_remaining)` directly from the page's initialization block (before the sidebar card renders) so the card always shows real projected IP. This adds ~50ms of compute but removes a confusing blank card that a novice might interpret as broken.

---

### R6. EXPLAIN "START ⚠" AND "LEAVE EMPTY" DECISIONS

**Problem:** Forced-start warnings and "LEAVE EMPTY" decisions confuse novices. No explanation of what action to take.

**Fix:**
- "START ⚠": expand the per-player "Why?" expander with a dedicated warning section: "This player's matchup score is below 0.70 — consider streaming a different pitcher or leaving the slot empty."
- "LEAVE EMPTY": add a card below the table when any slot shows LEAVE EMPTY: "No suitable player found for your [Slot] slot. Visit the Streaming tab or Free Agents page for pickup recommendations."

---

### R7. ADD A DOMINANT BANNER WHEN PROJECTION DATA IS STALE

**Problem:** `projections: ERROR` (FanGraphs down since 2026-06-10, 3+ days) is shown as a small amber badge in the Data Freshness card. A user running the optimizer doesn't know the recommendations may be based on stale data.

**Fix:** When the refresh_log shows `projections` as ERROR and the staleness exceeds 48 hours, render an `st.warning()` banner at the top of the Optimizer tab: "⚠ Projection data is 3+ days old (FanGraphs API failed 2026-06-10). Recommendations use cached projections — accuracy may be reduced." This is consistent with how the FA page shows legacy fallback banners.

---

### R8. ALLOW PITCHERS IN THE START/SIT ADVISOR

**Problem:** The Start/Sit tab's player selectboxes do not clearly allow pitcher comparisons. A manager deciding between Sale and Harrison for a streaming slot has no tool here.

**Fix:** Add a "Hitters / Pitchers / All" filter above the player selectboxes. When "Pitchers" is selected, filter the roster to pitchers only and display pitcher-relevant output columns (ERA impact, K/9, matchup score, opposing lineup wRC+).

---

### R9. ADD COLOR CODING AND MAGNITUDE SIGNALS TO THE CATEGORY ANALYSIS TABLE

**Problem:** The roster contribution table is a flat grid of decimal numbers with no visual hierarchy. A novice cannot quickly see who is helping vs hurting in ERA/WHIP.

**Fix:** Apply conditional background coloring to the category contribution table using CSS classes from the Combustion palette:
- Positive contribution (> 0.10 SGP): `--fp-green-faint` background.
- Negative contribution (< -0.10 SGP): `--fp-ember-faint` background.
- Near-zero (within ±0.10): no color.

Also add a "Team Total" row at the bottom of the table showing current week's aggregated SGP estimate per category.

---

### R10. UNIFY STREAMING ENTRY POINT OR ADD CROSS-REFERENCE

**Problem:** The Streaming tab in the Optimizer and the dedicated Pitcher Streaming page (page 05) overlap. A novice doesn't know which one to use or what the difference is.

**Fix:** Simplify the Streaming tab in the Optimizer to a "Quick Picks" summary (top 3 recommendations + a link to the full analysis), with a prominent "→ For detailed streaming analysis, see Pitcher Streaming" link. This DRYs the content and creates a clear mental model: Optimizer Streaming = quick daily check, Page 05 = deep research.

---

### R11. RE-ORDER MATCHUP STATE CARD TO SHOW WIN/LOSS GROUPED BY CATEGORY

**Problem:** Categories displayed in league order (R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP) — no grouping by outcome. A novice cannot quickly scan for which categories to focus on.

**Fix:** Group categories into: Winning (R, HR, RBI, AVG, OBP, W, K) | Losing (L, ERA, WHIP) | Tied (SB, SV). Show each group with a header chip: green "Winning 7", red "Losing 3", neutral "Tied 2". This makes the weekly situation scannable in under 2 seconds.

---

### R12. REPLACE STREAMING RISK FLAG STRINGS WITH ICON + TOOLTIP CHIPS

**Problem:** Risk flags (`HIGH_WHIP`, `SHORT_LEASH`, `HITTER_PARK`) render as bare ALL_CAPS strings. Meaningless to a novice.

**Fix:** Render each flag as a `.chip` component (available in the Combustion system) with a colored dot and a tooltip on hover explaining the flag: SHORT_LEASH = "Pitcher has a history of early hooks — may not reach 5 IP", HITTER_PARK = "This park inflates offense — ERA projection adjusted down", etc. This adds context without adding visual bulk.

---

## 8. Severity-Tagged Issue List

- `[BLOCKER]` B1 — `compute_nonlinear_weights()` receives bare `'Team Hickey'` but DB stores `'🏆 Team Hickey'` — all category weights return 1.00x; Category Analysis and LP optimizer produce wrong results for every run.
- `[HIGH]` H1 — IP Budget card blank on first page load (session state not pre-populated).
- `[HIGH]` H2 — Start/Sit player selectboxes empty if optimizer has not run first; no error message.
- `[HIGH]` H3 — "Refresh Yahoo Data" button invisible to non-admin users with no explanatory message in its place.
- `[HIGH]` H4 — `st.toast()` calls use emoji icons (`'✅'`, `'⚠️'`, `'❌'`) in violation of the no-emoji design system rule (Combustion lock).
- `[HIGH]` H5 — Start/Sit advisor does not support pitcher comparisons — selectboxes skew to hitters.
- `[HIGH]` H6 — DCV component values in "Why?" expander shown without scale or meaning (novice cannot interpret `matchup_mult: 0.8745`).
- `[MEDIUM]` M1 — Mode radio (Full/Standard/Quick) has no description of what each mode does or how long it takes.
- `[MEDIUM]` M2 — Risk Aversion slider has no tooltip, no units, no endpoint labels.
- `[MEDIUM]` M3 — Scope radio and Mode radio interact but are placed in different UI locations with no explanation of their relationship.
- `[MEDIUM]` M4 — "LEAVE EMPTY" decision label has no explanatory tooltip or follow-on guidance.
- `[MEDIUM]` M5 — Streaming tab in Optimizer overlaps with dedicated Pitcher Streaming page (page 05) — no cross-reference or delineation of purpose.
- `[MEDIUM]` M6 — "START ⚠" forced-start flag has no explanation in the UI; a novice does not know what action to take.
- `[MEDIUM]` M7 — DCV Score column has no scale context — 12.34 is not interpretable without knowing the range.
- `[MEDIUM]` M8 — Category Analysis roster table is a flat unformatted grid — no conditional coloring, no team-total row, no magnitude signals.
- `[MEDIUM]` M9 — Internal comment references 6 tabs; actual tab count is 4 — developer confusion risk if undocumented.
- `[MEDIUM]` M10 — `alpha` parameter silently overridden per scope; user has no insight into H2H vs season-total blending objective.
- `[LOW]` L1 — Nav label "Lineup Optimizer" (no hyphen) vs page header "Line-up Optimizer" (with hyphen).
- `[LOW]` L2 — FIG.02 eyebrow subtitle "LINEUP CONTROL" is non-informative; could be "DAILY LINEUP DECISIONS."
- `[LOW]` L3 — Streaming risk flag strings (`HIGH_WHIP`, `SHORT_LEASH`, etc.) have no glossary or tooltips.
- `[LOW]` L4 — `projections: ERROR` shown as small amber badge in Data Freshness card — insufficient prominence for 3-day-old stale projections actively affecting optimizer accuracy.
- `[POLISH]` P1 — IP Budget card shows `53.85` (2 decimal places); should display `~54 IP` for readability.
- `[POLISH]` P2 — Win Probability card is the last item in the sidebar — should appear first.
- `[POLISH]` P3 — Matchup State categories displayed in league order with no visual grouping by win/loss/tie.
- `[POLISH]` P4 — Optimizer tab before first click shows blank white area instead of `render_empty_state()`.
- `[POLISH]` P5 — "Refresh Yahoo Data" button label does not describe what data it refreshes.
