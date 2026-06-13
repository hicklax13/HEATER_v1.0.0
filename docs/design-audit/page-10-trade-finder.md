# Page 10 — Trade Finder — Test-User Report

**Auditor persona:** Connor, novice fantasy-baseball manager, Team Hickey (FourzynBurn, 10th place, 3-7-1)  
**Source file:** `pages/12_Trade_Finder.py`  
**Engine files:** `src/trade_finder.py`, `src/trade_value.py`, `src/opponent_trade_analysis.py`, `src/trade_intelligence.py`  
**Date audited:** 2026-06-13  
**Data state:** Yahoo data ~1,219 min stale (cached); standings current as of 2026-06-12T17:43

---

## 2. Page Purpose & First Impression

**Purpose:** Proactively scan all 11 opponent rosters to find mutually beneficial trade proposals for Team Hickey. Also offers a universal "Trade Value Chart" (0-100) and tools to target specific players, browse individual opponents, and assess each opponent player's trade desirability.

**First impression (novice):** The page is conceptually one of the most useful in the app — it does the work of finding trades FOR you rather than requiring you to think them up. But a novice lands on this page and immediately stares at a blank spinner for **43+ seconds** before seeing anything. When results appear, the top banner says something meaningful ("Send Dansby Swanson to Over the Rembow for Sonny Gray (+0.66 SGP)"), which is good. The 5-tab layout is clear. However, the "Value Chart" tab immediately shows a devastating bug: every player except Shohei Ohtani is classified as "Replacement," including Aaron Judge, Juan Soto, and Jackson Chourio. A novice would instantly lose trust in the entire page. The "Generate Enhanced Recommendations" button is hidden in the main tab with no explanation of how it differs from the automatic scan. The "Re-rank by title odds" sidebar toggle is visible but its ~1-minute wait time is buried in tooltip text.

---

## 3. Methodology

- Read full source: `pages/12_Trade_Finder.py` (1,199 lines), `src/trade_finder.py` (~800+ lines), `src/trade_value.py` (~350 lines), `src/opponent_trade_analysis.py`
- Read shared context (`docs/design-audit/_SHARED_CONTEXT.md`) in full
- Executed `find_trade_opportunities()` live against the real DB with:
  - Small run: `max_results=5, top_partners=3` → **17.8 seconds**, 5 results
  - Full production run: `max_results=50, top_partners=11` → **43.8 seconds**, 42 results
- Executed `compute_trade_values()` live → **2.9 seconds**, 9,888 players
- Queried league rosters, standings, refresh_log, players table directly
- Traced all 5 tab implementations via source code
- Cross-referenced design system (`src/ui_shared.py` THEME tokens, PAGE_ICONS, banned hex list)

---

## 4. Feature & Control Inventory

| Control | Type | What It Does | Tested? |
|---------|------|--------------|---------|
| **Sidebar: "Re-rank by title odds"** | Toggle | Enables playoff/championship MC re-rank (~1 min extra) | Source read |
| **Page header** | Static | "FIG.12 — OPPORTUNITY SCANNER" / "TRADES" eyebrow | Source read |
| **Top banner (reco_banner)** | Static card | Shows top trade opportunity on load | Reconstructed |
| **Matchup ticker** | Widget | Week 12 matchup info strip | Source read |
| **Context panel: Category Needs** | Static card | Team Hickey's 4 weakest cats (reconstructed: AVG, R, RBI, SB) | Reconstructed |
| **Context panel: Best Trade Partners** | Static card | Top 5 complementary teams with heat bar | Live data |
| **Context panel: Scan Summary** | Static card | Trades found / Teams scanned / Scan time | Live data |
| **Context panel: Trade Urgency** | Static card | Schedule urgency multiplier | Source read |
| **Tab 1: Trade Recommendations** | Tab | Full trade scan results table | Live scan |
| **Sort dropdown** | Selectbox | Sort by Composite Score / Need Efficiency / Acceptance / SGP Gain | Source read + bug found |
| **Generate Enhanced Recommendations** | Button (primary) | Runs `recommend_trades_by_need` scan | Source read |
| **Trade table** | Sortable table | You Give / You Receive / Type / Partner / Your Gain / Grade / Acceptance / Score | Live data |
| **Tab 2: Target a Player** | Tab | Select any opponent player, get lowball+fair proposal | Source read |
| **Target player selectbox** | Selectbox | Dropdown of all 275+ opponent players (team \| player format) | Source read |
| **Lowball proposal card** | Panel | Proposed give package + category chips | Source read |
| **Fair Value proposal card** | Panel | Proposed give package + category chips | Source read |
| **Grade / Efficiency / Accept metrics** | st.metric | Per-proposal metrics (3 columns) | Source read |
| **ADP/ECR fairness caption** | Static text | "ADP Fairness: X% \| ECR Fairness: X%" | Source read |
| **Tab 3: Browse Partners** | Tab | Opponent picker with category comparison + roster view | Source read |
| **Opponent team selectbox** | Selectbox | Select from 11 opponent teams | Source read |
| **Complementarity score** | Caption | Decimal score + "higher = better trade fit" | Live data |
| **Category Comparison table** | Sortable table | 12 cats: Your Rank / Their Rank / Gap / Opportunity | Source read |
| **Suggested Trades table** | Sortable table | Up to 10 trades with this partner from main scan | Live data |
| **"Generate Trade Proposals" button** | Button (secondary) | Generates lowball+fair proposals for top 3 opponent players | Source read |
| **Opponent roster tables** | HTML tables | Hitter and pitcher split roster view | Source read |
| **Tab 4: Trade Readiness** | Tab | 0-100 composite scores for all opponent players | Source read |
| **Position filter** | Selectbox | Filter by position | Source read |
| **Readiness table** | Sortable table | Player / Pos / ADP Tier / Readiness / Cat Fit / Proj Conf / Health / Scarcity / FA Edge / Best FA | Source read |
| **Readiness formula caption** | Caption | "Readiness = 40% Cat Fit + 25% Proj Conf + 15% Health..." | Source read |
| **Tab 5: Value Chart** | Tab | Universal 0-100 trade values for all players | Live data |
| **Search players** | Text input | Filter by player name | Source read |
| **"My Team Needs" toggle** | Toggle | Switch to contextual values adjusted for team needs | Source read |
| **Position pills** | Pills | Filter by position | Source read |
| **Metrics row (4)** | st.metric | Total Players / Elite Tier / Star Tier / Weeks Remaining | Live data |
| **Tier sections (5)** | Colored headers | Elite / Star / Solid Starter / Flex / Replacement player tables | Live data |
| **Trade value formula captions** | Caption | Explains SGP + G-Score methodology | Source read |

---

## 5. Feature-by-Feature Test Log (Real Outputs)

### Page load

**Scan timing (CRITICAL):** The page runs `find_trade_opportunities()` on load with `max_results=50, top_partners=11`. Measured time: **43.8 seconds** on the test machine running standalone (without Streamlit overhead). In the browser, this is a visible spinner during which the entire main content area is blank. The session-cache mechanism (`_tf_scan_cache`) helps on subsequent interactions, but cold load on every new session is a blocker.

**Cache key:** `trade_scan_signature(...)` hashes user_team + roster IDs + team totals + title_odds toggle. A Yahoo data refresh changes team totals and busts the cache, triggering a full rescan.

**Sidebar toggle:** "Re-rank by title odds (slower, ~1 min)" defaults to OFF. When ON, `simulate_trade_playoff_delta` is called for top 15 candidates at `n_sims=5,000`, adding estimated ~60+ seconds on top of the 43.8s base scan. The spinner message reads "Scanning league + computing title-odds (~5 sec)..." — **severely understates the actual wait time** (43.8s base + ~60s MC = ~2 minutes total).

### Context panel (sidebar-style cards, left column)

**Category Needs (reconstructed):** Team Hickey's bottom 4 categories:
- AVG: rank 12/12 (value=0.240) — dead last  
- R: rank 11/12 (value=359.85)  
- RBI: rank 10/12 (value=355.80)  
- SB: rank 9/12 (value=48.55)

Context card renders these as a `<ul>` list in the Combustion style. No ranks shown, no values — just the category names. Novice cannot tell HOW weak they are.

**Best Trade Partners (reconstructed from live data):**
1. Over the Rembow: 0.580
2. Baty Babies: 0.517
3. On a Twosday: 0.460
4. Cyrus The Greats: 0.402
5. BUBBA CROSBY: 0.390

The heat bar renders, but the decimal "0.580" has no label explaining what this number means.

**Scan Summary (live):** "Trades: 42 | Teams: 11 | Scan: 43.8s" — notably the "43.8s" is surfaced to the user, which is good for transparency, but 44 seconds for a data-loading step is extremely slow.

**Trade Urgency (reconstructed):** Calls `compute_schedule_urgency(weeks_ahead=3, yds=yds)` and shows "NORMAL" / "Urgency multiplier: 1.00x" (Yahoo offline, likely falls back to neutral 1.0x).

### Tab 1: Trade Recommendations

**Banner:** "Top opportunity: Send Dansby Swanson to Over the Rembow for Sonny Gray (+0.66 Standings Gained Points)" — clear and useful.

**Full scan results (live, 42 opportunities found):**

Top 10 (Composite Score sort):

| Give | Receive | Partner | SGP Gain | Opp SGP | Accept | Grade | Score |
|------|---------|---------|----------|---------|--------|-------|-------|
| Jeff Hoffman, Framber Valdez | Willy Adames | My Precious | +1.10 | +0.91 | 90% | A- | 0.78 |
| Dansby Swanson | Sonny Gray | Over the Rembow | +0.66 | +0.94 | 76% | B | 0.74 |
| Alex Bregman | Sonny Gray | Over the Rembow | +0.78 | +0.41 | 67% | B+ | 0.73 |
| Dansby Swanson + Dillon Dingler | Sonny Gray | Over the Rembow | +0.62 | +0.65 | 91% | B | 0.72 |
| Jeff Hoffman + Eduardo Rodriguez | Willy Adames | My Precious | +0.65 | +0.71 | 91% | B | 0.70 |
| Angel Martínez + Eduardo Rodriguez | Nick Lodolo | The Good The Vlad The Ugly | +0.85 | +0.71 | 89% | B+ | 0.69 |
| Dansby Swanson + Jeff Hoffman | Sonny Gray | Over the Rembow | +0.45 | +0.20 | 78% | B | 0.69 |
| Angel Martínez + Jeff Hoffman | Nick Lodolo | The Good The Vlad The Ugly | +0.43 | +1.12 | 81% | B | 0.69 |
| Angel Martínez + Dillon Dingler | Nick Lodolo | The Good The Vlad The Ugly | +0.58 | +1.46 | 77% | B | 0.68 |
| Jakob Marsee | Sonny Gray | Over the Rembow | +0.34 | +0.96 | 69% | B- | 0.68 |

**Sort dropdown bug (CONFIRMED):** The "Sort by Acceptance Probability" option sorts the `Acceptance` column which contains string labels "High", "Medium", "Low". String sort order is alphabetical: High < Low < Medium — so descending "sort by acceptance" incorrectly puts "Medium" trades at the top. Additionally, if "Generate Enhanced Recommendations" is clicked, the enhanced rows store acceptance as a percentage string (e.g., "76%") while scan rows store "High"/"Medium"/"Low" — mixed types crash the sort silently.

**"Generate Enhanced Recommendations" button:** Calls `recommend_trades_by_need`. A separate, longer scan focused on category-need alignment. Its purpose is not explained to the user ("Generate Enhanced Recommendations" vs the automatic scan — what makes these "enhanced"?). No result count is shown until after the scan completes.

**"Need Efficiency" column:** Only appears in the table when enhanced recs are present. When shown alongside scan rows, scan rows show 0.00 for Need Efficiency, making enhanced recs appear artificially superior.

### Tab 2: Target a Player

**Selectbox:** "Select target player (Team | Player)" with ~275+ options sorted. The dropdown is populated from all 11 opponent teams. Format is "Team Name | Player Name" which is functional but long.

**Player selection flow (reconstructed):** Selecting a player triggers `generate_targeted_proposals()` which returns:
- `target`: player info dict
- `lowball`: proposal dict with `giving_names`, `grade`, `acceptance_probability`, `efficiency`, `category_impact`, `adp_fairness`, `ecr_fairness`
- `fair_value`: same structure

**Proposal cards:** Uses Combustion `render_panel()` with "left" accent rail. Category impact chips use `.chip.hot` (orange) for SGP gains and `.chip.cold` (steel) for losses. Metrics row: `Grade` / `Efficiency` (ratio like "2.1x") / `Accept` (%).

**ECR Fairness display bug (2-for-1 trades):** In the main trade table, `ECR Fair` shows "0%" for ALL 2-for-1 trades (hardcoded `0.5` neutral but formatted as "0%"). Specifically, `COMPOSITE_W_ECR * 0.5` is used for multi-player trades in the engine. This displays as "ECR: 0%" in the table, which a user would interpret as "this trade is extremely unfair by ECR" — the opposite of the intended neutral.

### Tab 3: Browse Partners

**Complementarity scores (live):**
- Over the Rembow: 0.580 (renders as highest)
- Baty Babies: 0.517

**Category comparison table (reconstructed for Over the Rembow):** Shows 12 rows — "Category / Your Rank / Their Rank / Rank Gap / Opportunity." The "Opportunity" column shows "YOU GAIN" when rank_gap >= 3 (opponent is worse by 3+ places), "THEY GAIN" when <= -3, and "EVEN" otherwise.

**Suggested trades per partner (live):** Over the Rembow shows 6 trades. My Precious shows 2 trades.

**Opponent roster view (reconstructed):** Renders team logo (via `team_logo_url()`), then hitter table and pitcher table via `build_roster_table_html()`. This is a nice feature — seeing the full roster helps contextualize trade proposals.

**"Generate Trade Proposals" button:** Evaluates top 3 opponent players by a rough value heuristic (HR + RBI/3 + K/3 + SV). Opens expanders for each player with lowball/fair panels. No loading progress per-player.

### Tab 4: Trade Readiness

Computed with `compute_trade_readiness_batch()` for all ~275 opponent player IDs with `max_players=100`. Has a separate spinner. Returns DataFrame with columns:
- Player / Pos / ADP Tier / Readiness / Cat Fit / Proj Conf / Health / Scarcity / FA Edge / Best FA

ADP tiers: Elite (ADP ≤ 36), Core (≤ 96), Depth (≤ 180), Filler (> 180).

Position filter selectbox is present.

**Readiness formula caption:** "Readiness = 40% Category Fit + 25% Projection Confidence + 15% Health + 10% Scarcity + 10% FA Edge. Higher = better trade target for your team." — The most clearly explained metric on the page.

### Tab 5: Value Chart

**Metrics row (live):**
- Total Players: 9,888 (no position filter applied)
- Elite Tier: **0**
- Star Tier: **0**
- Weeks Remaining: 15

**Tier breakdown (live — CRITICAL BUG):**
- Elite (90-100): **0 players**
- Star (75-89): **0 players**
- Solid Starter (55-74): **1 player** (Shohei Ohtani — trade value 57.7)
- Flex (35-54): **0 players**
- Replacement (0-34): **9,887 players**

**Root cause confirmed:** `compute_trade_values()` applies a `time_factor = weeks_remaining / 26.0 = 15/26 = 0.577`. The pre-scaling maximum trade value is ~100 for Ohtani (post G-score normalization). After the 0.577 multiplier, Ohtani scores only **57.7**. The tier thresholds (Elite=90, Star=75, Flex=35) are fixed absolute cutoffs that assume 26/26 of the season remains. With only 15 weeks left, NO player can reach Elite or Star tier; only Ohtani squeaks into Solid Starter. Juan Soto (TV=34.5) is "Replacement." Aaron Judge (TV=29.2) is "Replacement."

**Dollar values (live):**
- Ohtani: $31.18
- Soto: $15.26
- Judge: $12.14
- 9th-ranked Brent Rooker: $11.13

The dollar values are also deflated by `time_factor`.

**"My Team Needs" toggle:** Switches to `contextual_value` / `contextual_tier` columns from `compute_contextual_values()`. This should boost AVG/R/RBI/SB contributors for Team Hickey. Toggle text "My Team Needs" is functional but sparse.

**Position pills:** Filter functionality works as designed.

**Off-palette color in tier headers (CONFIRMED):** `TIER_COLORS["Star"] = "#457b9d"` — this hex is in the banned list defined in `tests/test_no_offpalette_hex_in_pages.py`. However, the color is defined in `src/trade_value.py` (not in a pages/ file), so the CI guard does not catch it. It renders as a colored div header via `st.markdown`. Should use `T["sky"]` (#5f7d9c) or `T["cold"]` instead.

**Data freshness:** The Value Chart has no staleness indicator. With projections data showing `error` status as of 2026-06-10 in the refresh log, the SGP values powering trade values may be stale. No user-visible warning.

---

## 6. Errors, Issues & Difficulties

### Critical Errors

1. **Trade Value tier system is completely broken mid-season.** With 15 weeks remaining, the time-decay multiplier (0.577) deflates all values below the Elite/Star tier thresholds. The chart shows 0 Elite players, 0 Star players, Shohei Ohtani as the only "Solid Starter," and Juan Soto/Aaron Judge/virtually every player in the game as "Replacement." A novice user will either: (a) think the data is broken, (b) make terrible trades based on false value information, or (c) lose trust in the entire app. **The tier thresholds must scale with time_factor, or tiers must be assigned before time decay is applied.**

2. **Cold-load scan takes 43.8 seconds.** Every new browser session (or after a Yahoo refresh) triggers the full scan before the page shows any content. Streamlit's WebSocket connection can drop during this wait (the Trade Analyzer page already dropped the connection according to the orchestrator notes). No progress indicator beyond the generic "Scanning league for trade opportunities..." spinner. No partial-result display.

3. **"Sort by Acceptance Probability" sorts strings, not numbers.** The Acceptance column contains "High"/"Medium"/"Low" (categorical labels). Alphabetical sort puts High < Low < Medium. The intent is to surface highest-acceptance trades first, but the implementation does the opposite. When enhanced recs are also in the table, Acceptance contains mixed types ("High"/"Medium"/"Low" from scan and "76%"/"45%" strings from enhanced recs), making sort behavior undefined.

4. **ECR Fairness always shows "0%" for 2-for-1 trades.** The engine hardcodes `COMPOSITE_W_ECR * 0.5` for multi-player trades (ECR neutral). This renders as "ECR: 0%" in the table. A novice reading this table sees "ECR Fair: 0%" and concludes these 2-for-1 deals are wildly unfair by expert rankings — the wrong conclusion.

### High Severity

5. **Spinner text misquotes the actual wait time.** When the Title Odds toggle is ON, the spinner says "Scanning league + computing title-odds (~5 sec)..." The base scan already takes 43+ seconds; the title-odds add-on is ~60 more seconds. The actual wait is 1.5-2 minutes, not "~5 sec."

6. **Missing player pool warning fires during every full scan (player_id=44, Xander Bogaerts).** The log shows `_player_sgp_volume_aware: player_id=44 missing from player_pool` ~6 times during the 11-team scan. Bogaerts IS in the player pool but his roster status is "NA" (non-active). The warning fires when LP-constrained totals for the opponent team tries to optimize using an NA-status player. This is a silent data quality issue — Bogaerts is being excluded from opponent SGP calculations, which could understate that team's total.

7. **TIER_COLORS["Star"] = "#457b9d" is a banned off-palette hex.** This is the exact color in the `_BANNED` set of `test_no_offpalette_hex_in_pages.py`. The CI guard only scans `pages/*.py` files, not `src/trade_value.py`, so this slips through. It renders as the "Star" tier header background color in the Value Chart.

8. **Tab code comments are mislabeled in source.** The source shows "# ── Tab 5: Trade Readiness" (it's tab 4) and "# ── Tab 3: Target a Player" (it's tab 2). Two separate sections are labeled "Tab 5." While the visual tab order is correct (the `with` blocks use the right variable names), this makes code maintenance fragile.

9. **`render_reco_banner()` is called with `"trade_analyzer"` icon key, not `"trade_finder"`.** There is no `trade_finder` key in `PAGE_ICONS`, so reusing the trade_analyzer icon is technically correct (avoids a KeyError), but the conceptual mismatch is a minor inconsistency. A dedicated Trade Finder icon would be proper.

### Medium Severity

10. **No explanation of what "Your Gain" (SGP) means to a novice.** The trade table shows "Your Gain: 1.10" and "Their Gain: 0.91" with no units, no tooltip, no caption. A beginner has no idea whether 1.10 is amazing or terrible. The banner appends "(Standings Gained Points)" on the best trade, but the table itself has no label. Format is bare float (e.g., "1.10"), not the `+1.10` format prescribed by `format_stat(value, 'SGP')`.

11. **No data-staleness warning in the Value Chart.** The refresh log shows `projections` status=`error` as of 2026-06-10 (3 days ago). The SGP values are computed from these stale projections. The page shows no warning. Users making trade decisions based on this chart may be using outdated player values.

12. **Complementarity score (0.580) is displayed without explanation.** The context card shows "1. Over the Rembow: 0.580" with a heat bar. A novice cannot decode "0.580" — is this good? The score is in [0, 1] where higher = better fit. No legend, no tooltip, no range label.

13. **Browse Partners: "Opportunity" column labels are confusing.** "YOU GAIN" means "they have an excess in this category" (they'd want your weak players), not that you literally gain that stat. "EVEN" means both teams are similarly ranked. A novice reads "YOU GAIN: AVG" and thinks "this trade improves my AVG" — which is only inferentially true, not directly stated.

14. **The "Source" column in the main table (showing "Scan" or "Enhanced") appears without explanation.** When no enhanced recs exist, the table still renders a "Source: Scan" column that conveys nothing meaningful to the user. Should be hidden when all rows are from the same source.

15. **Grade labels (A-, B+, B, B-) lack explanation.** A trade graded "A-" sounds great, but the novice doesn't know if A = you're giving away a superstar for peanuts or A = this is genuinely excellent for you.

### Low Severity / Polish

16. **"ΔPlayoff%" and "ΔChamp%" columns silently invisible unless title odds enabled.** These columns exist in the full DataFrame but render as empty strings ("") when `ranked_by != "title_odds_blend"`. The table shows empty cells in named columns, which looks broken. These columns should be hidden entirely when title odds aren't enabled.

17. **"Need Efficiency" column shows 0.00 for all scan trades** when enhanced recs are mixed in. This misleads users into thinking scan trades have zero need efficiency.

18. **Tab 2 and Tab 3 use bare `st.subheader()` instead of the Combustion panel header pattern.** "Target a Player" and "Browse Trade Partners" appear as plain Streamlit subheadings, breaking visual consistency with the card-panel pattern used in other pages.

19. **Position filter in Trade Readiness tab shows ALL positions including edge cases** (e.g., "Util", "DH", "TWP"). The filter is not sorted by primary positions first, making the list hard to scan.

20. **`st.caption()` appears in Browse Partners for complementarity score** but uses raw `**bold**` Markdown in `st.caption()` context, which Streamlit renders differently across themes. The complementarity score should use the design system's `render_stat_readout_html` component.

---

## 7. UI/UX & Visual-Design Critique

### Load time / perceived performance

This is the most compute-heavy page in the app. 43.8 seconds before ANY content appears is a hard barrier for casual use. The session-state cache (Issue #10 in the CLAUDE.md history) was supposed to fix this, but it only helps WITHIN a session. Every new tab, every page refresh, every post-Yahoo-sync reload triggers the full rescan. Other AI-powered tools like FantasyPros and ESPN's Trade Machine return results in under 3 seconds. HEATER's 44-second cold load will cause most users to abandon the page.

### Value Chart tier breakdown (catastrophic usability failure)

The Value Chart's purpose is to answer "Who is worth what in a trade?" The actual output at 15 weeks remaining: **9,887 of 9,888 players are "Replacement."** This is factually absurd and would immediately destroy any novice user's confidence in the tool. The Elite/Star/Flex tier labels exist but have zero players. The tier headers appear (because the code iterates tier_order and shows "Elite (0)"), which makes the display even more confusing — headers with no content. This is the most severe UX failure on the page.

### Combustion design system compliance

**Well-executed:**
- Page header uses correct `render_page_header("Trade Finder", eyebrow="TRADES", fig="FIG.12 — OPPORTUNITY SCANNER")` with the orange period pattern. FIG.12 numbering is consistent with page order.
- The reco_banner at the top is appropriately visible.
- Proposal cards in Target/Browse use `render_panel()` with "left" accent rail.
- Category impact chips use `.chip.hot`/`.chip.cold` correctly.
- Context cards with heat bars follow the sidebar card pattern.

**Violations / gaps:**
- `TIER_COLORS["Star"] = "#457b9d"` is a banned hex (should be `T["sky"]` #5f7d9c).
- `TIER_COLORS["Flex"] = "#666666"` is also in the banned set.
- `TIER_COLORS["Replacement"] = "#cc3333"` is not in the banned list but is not a THEME token either; should use `T["danger"]` (#e0492f) or `T["ember"]`.
- Tab 2 and Tab 3 use `st.subheader()` — not the Combustion panel system.
- "Your Gain" in the table is a raw float, not formatted with `format_stat(value, "SGP")` which would give "+1.10" with the mandatory `+` prefix.

### Number formatting

The trade table's "Your Gain" column shows bare floats ("1.10", "0.66") without the `+` prefix required by `format_stat(value, 'SGP')`. The structural guard `test_pages_format_compliance.py` checks for inline `f"{x:.3f}"` near ERA/WHIP, but the SGP formatting is done in `_build_trade_df()` via `round(trade.get("user_sgp_gain", 0), 2)` — a `round()` call, not `format_stat` — so it passes the CI guard but still violates the design contract.

### Accessibility / mobile

The Readiness tab spinner has no progress feedback beyond a generic "Computing Trade Readiness scores..." message covering what could be 5-10 seconds of computation. The position filter dropdown for the Readiness tab surfaces all positions in unsorted order including minor-league designations.

### Information hierarchy

The 5-tab layout is appropriate. However, the tab order could be improved:
- **Tab 1 (Recommendations)** — correct: primary use case
- **Tab 2 (Target)** — fine
- **Tab 3 (Browse Partners)** — fine
- **Tab 4 (Trade Readiness)** — this is the most analytical tab; a novice will click it and be overwhelmed by 100-row tables with 10 columns
- **Tab 5 (Value Chart)** — should probably be Tab 2 (it's the reference chart you'd consult BEFORE making trades)

---

## 8. High-Level Recommendations (≥10)

### R-1: Fix the Value Chart tier thresholds — time-decay the cutoffs, not the values [BLOCKER]

**Problem:** Applying the time-decay factor (15/26 = 0.577) to trade values means NO player can ever reach Elite (90+) or Star (75+) tier after week 11 of 26. Shohei Ohtani is "Solid Starter." Juan Soto is "Replacement."

**Fix (option A — preferred):** Assign tiers BEFORE time decay. Apply time decay only to the final displayed values, not to tier assignment. Store two columns: `trade_value_full` (for tier assignment) and `trade_value` (time-decayed, for display).

**Fix (option B):** Scale tier thresholds proportionally: `Elite_threshold = 90 * time_factor`, etc. This preserves the relative meaning of tier names.

**Impact:** Without this fix, the Value Chart is actively misleading. A user deciding whether to trade Aaron Judge ("Replacement" per the chart) has no usable reference.

---

### R-2: Pre-compute or background-compute the trade scan — eliminate the 44-second cold-load [BLOCKER]

**Problem:** The page blocks all rendering for 43.8+ seconds on first load (or after cache invalidation). This will cause WebSocket timeouts (Trade Analyzer already crashed this way).

**Fixes:**
1. **Run `find_trade_opportunities()` in the background scheduler** (like bootstrap phases) and store results in the DB. The page reads from cache immediately, showing results that are at most N hours old.
2. **Display results progressively** as the scan finds them: show 1-for-1 results immediately (~10s), then expand to 2-for-1 results while the user reads.
3. **Reduce default `max_results` from 50 to 20** and `top_partners` from 11 to 5 for the default (fast) view; add an "Expand search" button for the full 11-team scan.

---

### R-3: Fix the Acceptance Probability sort — sort by numeric probability, not string label [HIGH]

**Problem:** Sorting "by Acceptance Probability" sorts strings ("High", "Medium", "Low") alphabetically. Result: "Medium" sorts highest. Additionally, when Enhanced Recs are present, mixed types ("High" vs "76%") make sort behavior undefined.

**Fix:** Store numeric acceptance probability as a hidden column (`_accept_prob` float 0-1). Sort on that column. The display column remains "High"/"Medium"/"Low" for readability. Alternatively, convert labels to ordinal integers (High=3, Medium=2, Low=1) for sort.

---

### R-4: Show "ECR Fair" correctly for 2-for-1 trades — hide or label the neutral placeholder [HIGH]

**Problem:** All 2-for-1 trades show "ECR Fair: 0%" because the engine uses a 0.5 neutral (shown as `f"{0.5:.0%}"` = "50%"? Actually checking: `0.5 * 100 = 50%`). Wait — looking again at the code: `COMPOSITE_W_ECR * 0.5` is a weight contribution, not formatted as percent. In `_build_trade_df`, `ecr_fairness` comes from `trade.get("ecr_fairness", 0)`. For 2-for-1 trades in `scan_2_for_1`, the ECR field is not set, so it defaults to 0. `f"{0:.0%}"` = "0%". Fix: show "N/A" for multi-player trades where ECR cannot be computed, not "0%".

---

### R-5: Explain SGP units throughout the page [HIGH]

**Problem:** A novice sees "Your Gain: 1.10" and "Their Gain: 0.91" with no context. Nowhere on the page (outside the top banner's one-time mention) is "Standings Gained Points" defined. SGP values are in a vague abstract unit with no intuitive scale.

**Fixes:**
- Add a `?` tooltip to "Your Gain" column header: "Standings Gained Points (SGP): estimated improvement in your league ranking. +1.0 SGP = move up ~1 place in standings."
- Format with `format_stat(value, "SGP")` to get the mandatory `+` prefix: "+1.10" instead of "1.10"
- Add a one-line legend below the table: "SGP gain of +1.0 ≈ winning one additional category across the season"

---

### R-6: Explain the "Grade" letter scale and the Acceptance labels [MEDIUM]

**Problem:** Trade grades (A-, B+, B, B-) and acceptance labels ("High"/"Medium"/"Low") are shown without any scale reference. A novice doesn't know: Does "B" mean "good"? Is acceptance "High" at 50%? At 75%? The grade and the engine's primary objective (maximize YOUR SGP) are not always aligned — a "Grade: A-" trade could still be accepted at "Acceptance: Low."

**Fix:** Add a collapsible legend or caption:
- "Grade: A = excellent SGP gain for you; B = good; C = marginal. Higher grade ≠ higher acceptance."
- "Acceptance: High (≥60%), Medium (30–60%), Low (<30%) — estimated probability the opponent accepts."

---

### R-7: Fix the Value Chart tier color palette — replace off-palette hexes [MEDIUM]

**Problem:** `TIER_COLORS` in `src/trade_value.py` contains banned hex values:
- `"Star": "#457b9d"` → use `T["sky"]` (`#5f7d9c`)
- `"Flex": "#666666"` → use `T["tx_muted"]` (`#646a78`) or `T["tx_subtle"]`
- `"Replacement": "#cc3333"` → use `T["danger"]` (`#e0492f`) or `T["ember"]`

These render in the page HTML via `st.markdown()` and violate the Combustion design system. The CI guard misses them because they're defined in `src/`, not `pages/`.

---

### R-8: Add a "What does this score mean?" explainer to the Trade Readiness tab [MEDIUM]

**Problem:** The Readiness tab shows a 0-100 score with 5 sub-components: Category Fit, Projection Confidence, Health, Scarcity, FA Edge. The caption explains the formula weights (40%/25%/15%/10%/10%), which is a good start, but a novice still doesn't know:
- What is a "good" readiness score? (80? 60?)
- What does "Projection Confidence" mean? (data coverage? consistency with YTD?)
- What does "FA Edge" mean? (this player vs available FAs at the same position?)

**Fix:** Add a score-range legend (e.g., "80-100: Strong target; 60-79: Good target; below 60: Marginal") and 1-sentence definitions for each sub-component.

---

### R-9: Surface the data freshness state in the Value Chart [MEDIUM]

**Problem:** The Value Chart computes trade values from SGP denominators which depend on projections. The refresh log shows `projections` status=`error` as of 2026-06-10 (3 days ago). The chart has no freshness badge or warning.

**Fix:** Query `get_refresh_log_snapshot()` and render a `DataFreshnessTracker` badge (or `st.warning()`) when projections are in error/stale state. Something like: "Trade values based on projections last updated 3 days ago."

---

### R-10: Reorder Value Chart tab and reorder tabs overall [MEDIUM]

**Problem:** The Value Chart (Tab 5) is the least discoverable tab but arguably the most foundational — before you make any trade, you want to know what players are worth. The Trade Readiness tab (Tab 4) is positioned before it but is more analytical.

**Proposed tab order:**
1. Trade Recommendations (same — primary use case)
2. Value Chart (moved up — reference tool)
3. Target a Player (same)
4. Browse Partners (same)
5. Trade Readiness (moved to end — most analytical)

---

### R-11: Display the complementarity score range and add a tooltip [MEDIUM]

**Problem:** The Best Trade Partners card shows scores like "0.580" with a heat bar but no explanation. Users don't know if 0.58 means "barely compatible" or "highly complementary."

**Fix:** Add a label beneath each score: "0.580 — Strong fit (top 3 of 11 teams)" or a tooltip: "Complementarity score (0-1): how much this team's strengths match your needs. Higher = better trade partner." The heat bar is already scaled to [0, 100] percent — could also just show the percent form ("58%").

---

### R-12: Warn users clearly before Title Odds toggle triggers a 2-minute wait [LOW]

**Problem:** The Title Odds toggle in the sidebar says "slower, ~1 min" in the label, but the actual wait is 44s (base scan) + ~60s (MC) ≈ 2 minutes. The label underestimates, and it's in the sidebar where users may not read the tooltip carefully.

**Fix:** When the toggle is flipped ON, show an `st.info()` or `st.warning()` inline: "Title odds re-ranking adds ~60 seconds to the scan (total ~2 minutes). Toggle OFF for fast results." Alternatively, update the label to "Re-rank by title odds (~2 min total)".

---

### R-13: Hide empty columns (ΔPlayoff%, ΔChamp%) when title odds is disabled [LOW]

**Problem:** The trade table always has `ΔPlayoff%` and `ΔChamp%` columns. When title odds is disabled (the default), these columns are empty strings for ALL rows. The table shows blank columns with meaningful-looking headers that contain nothing.

**Fix:** Only add these columns to the display DataFrame when `_title_odds_enabled=True` and at least one trade has `ranked_by == "title_odds_blend"`.

---

### R-14: Add a "Send to Trade Analyzer" action button on each trade [LOW]

**Problem:** The trade recommendations are read-only — a user can see "Give Dansby Swanson for Sonny Gray" but has to manually navigate to the Trade Analyzer page and re-enter the same trade to get detailed analysis (phases 1-6 of the full trade engine, including LP reshuffle, IP floor, playoff delta).

**Fix:** Add a "Analyze in Trade Analyzer →" button per trade row (or in an expanded detail view). Clicking it sets `st.session_state["tf_proposal"]` with the give/receive IDs and navigates to Trade Analyzer, which reads from that state to pre-populate the proposal. This closes the natural user workflow loop.

---

## 9. Severity-Tagged Issue List

- `[BLOCKER]` Trade Value Chart tier thresholds are incompatible with time-decay: 0 Elite, 0 Star, 0 Flex players; Juan Soto and Aaron Judge classified as "Replacement"
- `[BLOCKER]` Cold-load scan takes 43.8 seconds with no partial-result display; at 12 users concurrent this will drop WebSocket connections
- `[HIGH]` "Sort by Acceptance Probability" sorts string labels alphabetically (High < Low < Medium) — wrong order
- `[HIGH]` Mixed types in Acceptance column when Enhanced Recs present ("High"/"Medium" vs "76%") breaks sort and display
- `[HIGH]` ECR Fairness shows "0%" for all 2-for-1 trades (should be "N/A" — value is not computed for multi-player trades)
- `[HIGH]` Spinner text "~5 sec" when title odds toggle is ON; actual wait is ~2 minutes (44s base + 60s MC)
- `[HIGH]` TIER_COLORS["Star"] = "#457b9d" is a banned off-palette hex (should use THEME sky token)
- `[HIGH]` TIER_COLORS["Flex"] = "#666666" is also a banned hex; TIER_COLORS["Replacement"] = "#cc3333" is not a THEME token
- `[HIGH]` "Your Gain" / "Their Gain" values use raw `round()` float, not `format_stat(value, "SGP")` — misses the mandatory `+` prefix and SGP formatting contract
- `[MEDIUM]` No explanation of SGP units anywhere in the trade table; novice cannot interpret "1.10" or "0.66"
- `[MEDIUM]` Grade letters (A-, B+, B) and Acceptance labels (High/Medium/Low) have no legend or tooltip
- `[MEDIUM]` Complementarity score (0.580) displayed without range context or label explaining what "higher" means
- `[MEDIUM]` No data freshness warning in Value Chart despite projections being in `error` state since 2026-06-10
- `[MEDIUM]` Tab 2 (Target) and Tab 3 (Browse) use bare `st.subheader()` instead of Combustion panel headers — visual inconsistency
- `[MEDIUM]` "Source: Scan" column shows when no enhanced recs exist, meaninglessly taking up table width
- `[MEDIUM]` "Need Efficiency: 0.00" for all scan trades when enhanced recs are mixed in, making scan trades look artificially worse
- `[MEDIUM]` Category Needs context card shows only category names, not ranks or values — novice can't tell if "AVG" is rank 9 or rank 12
- `[MEDIUM]` Browse Partners: "YOU GAIN" / "THEY GAIN" labels in Opportunity column are ambiguous — "YOU GAIN" doesn't mean the trade improves that category, it means there's an asymmetry to exploit
- `[MEDIUM]` Trade Readiness tab sub-components (Projection Confidence, FA Edge) are undefined in the UI
- `[MEDIUM]` Value Chart tab is Tab 5 (last) but serves as the foundational reference; should be Tab 2
- `[LOW]` Tab code comments in source are mislabeled (Tab 5/Tab 3/two Tab 5 comments) — maintenance hazard
- `[LOW]` `render_reco_banner()` uses `"trade_analyzer"` icon key on the Trade Finder page — no dedicated Trade Finder icon
- `[LOW]` ΔPlayoff% and ΔChamp% columns appear as blank columns when title odds is disabled
- `[LOW]` Position filter in Trade Readiness tab is unsorted and includes Util/DH/TWP without primary-position grouping
- `[LOW]` Sidebar "Re-rank by title odds" tooltip says "~5 sec" for the add-on but the base scan itself is 44s
- `[LOW]` `st.caption()` uses `**bold**` markdown in Browse Partners for complementarity — renders inconsistently
- `[POLISH]` "Generate Enhanced Recommendations" button label does not explain how enhanced recs differ from the automatic scan
- `[POLISH]` No "Send to Trade Analyzer" deep-link from recommendation rows to close the analysis loop
- `[POLISH]` Xander Bogaerts (player_id=44, status=NA) fires `_player_sgp_volume_aware` warnings during every full scan — suppressed in logs but indicates an NA-player included in LP roster optimization
