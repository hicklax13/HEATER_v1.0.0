# Page 08 — Punt Analyzer — Test-User Report

---

## 1. Page Purpose & First Impression

**Purpose:** The Punt Analyzer lets a fantasy manager simulate the effect of "punting" (intentionally conceding) one or more scoring categories. By zeroing out a category's weight, the tool reshuffles player values to show who gains or loses relevance — and how Team Hickey's standings profile looks if they stop competing in that category.

**First impression (novice, 5 seconds):**

The page is sparse and completely blocked behind a multiselect. Until the user picks a category, the page shows only a header, one sentence of instruction, a dropdown, and a blue info box — nothing else. A newcomer sees:

> "Select categories to punt (ignore). The tool will recompute player values and show how your roster and available players change in value."

The phrase "punt" is used in the label and instruction without any definition. A novice manager who does not already know what a punt strategy is receives no explanation. There is no "What is punting?" tooltip, no example, no guidance on _when_ you would want to use this feature, and no indicator of where the user stands before they start experimenting.

**Nav-label vs page-title drift (confirmed):** The sidebar label reads "Punt Analyzer" but the H1 wordmark renders "Punt Strategy Simulator." This is a naming inconsistency that creates confusion — a user who bookmarks the tool or reads about it will encounter two different names for the same page.

---

## 2. Methodology

- Read `pages/10_Punt_Analyzer.py` in full (341 lines).
- Read `src/valuation.py` (LeagueConfig, SGPCalculator — both `total_sgp` row-by-row and `total_sgp_batch` vectorized methods).
- Read `src/standings_utils.py` (standings aggregation, ghost-team filtering).
- Read `src/ui_shared.py` (design system: THEME, tokens, render_panel, build_heatbar_html, render_empty_state, etc.).
- Ran three read-only DB query scripts (deleted after use) to capture real values. All network calls blocked.
- Reconstructed full runtime output for Punt SB and Punt SV scenarios.
- Timed the SGP computation method used by the page vs the available vectorized alternative.

---

## 3. Feature & Control Inventory

| Control | Type | What it does | Tested? |
|---|---|---|---|
| Page header (eyebrow + wordmark) | Static UI | Renders "STRATEGY / FIG.10 — CATEGORY PUNT MODEL" + "Punt Strategy Simulator." wordmark | Yes |
| Intro paragraph | Static text | One-sentence description of the tool | Yes |
| **Categories to punt:** multiselect | Input | Select which cats to zero out; drives all downstream output | Yes (SB, SV, L, ERA, multi-cat) |
| Info box (empty state) | st.info | Shows when no categories selected; page stops | Yes |
| Progress bar | st.progress | Shows 20%→50%→100% while computing; auto-hides after 0.3s | Yes (traced) |
| Strategy summary panel (FIG.10.1) | render_panel + chips | Shows count of punted/active cats as stat-readouts; chips for each cat | Yes |
| "Biggest Value Gainers" panel (FIG.10.2) | render_panel + custom table | 15-row branded table: Player (headshot+team logo), Pos, Original SGP, Punt SGP, Change | Yes |
| "Biggest Value Losers" panel (FIG.10.3) | render_panel + custom table | 15-row branded table with delta shown in cold color | Yes |
| "Standings Impact" panel (FIG.10.4) | render_panel + custom table | Category | Your Total | Rank | Strength bar | Status | Yes (reconstructed) |
| "Standings Points from Active Categories" metric | st.metric | Integer count; help text shows total/active/punted breakdown | Yes (reconstructed) |
| Feedback widget | render_feedback_widget | Per-feature feedback popover (MULTI_USER gated) | Out-of-scope (excluded) |

**Note:** There are no tabs, no user-roster-specific controls (e.g., no "show only my players"), no filter by position/hitting/pitching, no "suggested punt strategies" shortcuts, no trade/waiver call-to-action, and no export. This is a notably thin control surface for a strategy page.

---

## 4. Feature-by-Feature Test Log with Real Outputs

### 4.1 Empty state (no category selected)

**Output:** `st.info("Select one or more categories above to see the punt analysis.")` then `st.stop()`. Page renders nothing below the multiselect.

**Issue:** `st.info` is a plain blue box — the design system provides `render_empty_state()` for data-empty contexts. The usage of `st.info` here is inconsistent with Combustion Index conventions.

---

### 4.2 Multiselect — "Categories to punt:"

Available options (from `config.all_categories`):
`R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP`

The dropdown reads "Choose options" (Streamlit default placeholder). No custom placeholder text, no pre-selection, no grouping of hitting vs pitching categories. The user must know which categories are hitting vs pitching. A novice could accidentally punt ERA (a pitching inverse stat that they want to be _low_) without understanding that the help text for ERA only says "lower is better" in other pages — it says nothing here.

---

### 4.3 Progress bar

**Sequence (traced from source):**
1. `progress.progress(20, text="Computing baseline values...")`
2. `progress.progress(50, text="Computing punt-adjusted values...")`
3. `progress.progress(100, text="Analysis complete!")`
4. `time.sleep(0.3)` then `progress.empty()`

**Performance issue (measured):** The page iterates `pool.iterrows()` calling `sgp_orig.total_sgp(player)` and `sgp_punt.total_sgp(player)` one row at a time across 9,888 players. Measured timing:
- **Row-by-row (method used):** 1.04s per pass × 2 passes = **~2.1 seconds total compute** before anything renders.
- **Vectorized `total_sgp_batch` (available but unused):** 0.003s per pass.
- **Speedup available:** **413×** — the entire computation could complete in ~6ms instead of ~2 seconds.

The 2-second blocking compute happens inside `try:` with a progress bar, which means Streamlit cannot re-render the progress updates incrementally (Streamlit re-renders only after a Python call returns). The user sees a frozen progress bar, not incremental progress.

---

### 4.4 Strategy Summary Panel (FIG.10.1) — Punt SB

**Actual output (reconstructed):**

Stat readouts:
- `Punted: 1` (accent orange)
- `Active: 11`
- `Categories: 12`

Chip rows:
- **Punted:** `SB` (cold/blue chip)
- **Active:** `R  HR  RBI  AVG  OBP  W  L  SV  K  ERA  WHIP` (hot/orange chips)

Panel title: `"Punting SB"`, fig_label: `"FIG.10.1 — STRATEGY"`

**Novice confusion:** The panel confirms the selection but gives no interpretation. "You're punting SB. Here's what that means for your team" — absent entirely. No statement like "Team Hickey currently ranks 11th/13th in SB (40 stolen bases). Punting this category frees up value for..."

---

### 4.5 Biggest Value Gainers Panel (FIG.10.2) — Punt SB

**Actual output (reconstructed from DB + SGP engine):**

| Player | Pos | Original | Punt | Change |
|---|---|---|---|---|
| Tarik Skubal | SP | +5.58 | +5.58 | +0.00 |
| Paul Skenes | SP | +4.10 | +4.10 | +0.00 |
| Garrett Crochet | SP | +4.30 | +4.30 | +0.00 |
| Junior Caminero | 3B | +4.34 | +4.34 | +0.00 |
| Ketel Marte | 2B | +2.92 | +2.92 | +0.00 |
| ... (all 15 rows show Change = +0.00) | | | | |

**CRITICAL BUG — Gainers table is meaningless:** Under Punt SB, the 15 "biggest gainers" all show Change = +0.000. This is because they are pitchers and non-speed hitters whose SB projection is already 0 — they literally gain nothing. The page picks the top 15 by `value_change = punt_sgp - original_sgp`, but for any player whose SB projection = 0, `value_change` = 0.000 exactly. The `nlargest(15, "value_change")` call returns 15 tied-at-zero rows, sorted by pandas' internal order (roughly by original rank position).

The "Biggest Value Gainers" panel therefore displays FIFTEEN PLAYERS WHO EXPERIENCED NO CHANGE WHATSOEVER. This is the opposite of meaningful. The label says "Players whose value increases most when the punted categories are removed" — but none of the displayed players' values increased at all.

The correct interpretation is that punting SB hurts speed players (losers) and is neutral for everyone else — there are genuinely zero gainers in the absolute-delta sense. The page should either:
(a) Explain that "gainers" means relatively re-ranked (relative to the pool after speed specialists fell), or
(b) Show the top 15 by **post-punt rank** vs **pre-punt rank**, surfacing who climbs in relative rankings, or
(c) Display a note: "No players gained absolute value — these categories were not positively scored for anyone."

For Punt L (inverse stat), the gainers table does work correctly: `Aaron Brooks RP: orig=-45.56, punt=-19.56, delta=+26.00` — players with many losses gain value when L is removed from scoring. So the logic is correct for stats players contribute positively to, but breaks silently for zero-SB players.

---

### 4.6 Biggest Value Losers Panel (FIG.10.3) — Punt SB

**Actual output (reconstructed):**

| Player | Pos | Original | Punt | Change |
|---|---|---|---|---|
| Tyler Tolbert | 2B | +4.24 | +1.52 | -2.71 |
| Brewer Hicklen | OF | +8.01 | +5.79 | -2.21 |
| Justin Dean | CF | +1.74 | -0.33 | -2.07 |
| Bobby Witt Jr. | SS | +5.08 | +3.66 | -1.43 |
| Brenton Doyle | CF | +3.78 | +2.42 | -1.36 |
| José Ramírez | 3B | +4.51 | +3.15 | -1.36 |
| Dairon Blanco | CF | +1.54 | +0.18 | -1.36 |
| Andrew Stevenson | OF | +2.05 | +0.69 | -1.36 |
| Shohei Ohtani | TWP | +13.37 | +12.08 | -1.29 |
| Jonah Cox | CF | +7.40 | +6.11 | -1.29 |
| Esteury Ruiz | RF | +2.00 | +0.79 | -1.21 |
| Ronald Acuña Jr. | RF | +5.42 | +4.21 | -1.21 |
| Luisangel Acuña | CF | +2.24 | +1.03 | -1.21 |
| Jazz Chisholm Jr. | 2B | +3.47 | +2.33 | -1.14 |
| Randy Arozarena | LF | +3.86 | +2.72 | -1.14 |

This table correctly identifies speed specialists who lose value under Punt SB. The interpretation comment "potential trade targets" is a missed opportunity — these are not trade _targets_, these are players whose value is now _lower on your team_, making them **trade bait** (players you want to trade away to a team that still cares about SB). The label is backwards from a strategic standpoint.

---

### 4.7 Standings Impact Panel (FIG.10.4) — Punt SB

**Actual output (reconstructed from live DB):**

User team: `🏆 Team Hickey`

| Category | Your Total | Rank | Strength | Status |
|---|---|---|---|---|
| R | 376 | 5/13 | 69.2% | ACTIVE |
| HR | 118 | 1/13 | 100.0% | ACTIVE |
| RBI | 345 | 6/13 | 61.5% | ACTIVE |
| SB | 40 | 11/13 | 23.1% | PUNT |
| AVG | 0.236 | 12/13 | 15.4% | ACTIVE |
| OBP | 0.332 | 6/13 | 61.5% | ACTIVE |
| W | 50 | 3/13 | 84.6% | ACTIVE |
| L | 47 | 12/13 | 15.4% | ACTIVE |
| SV | 24 | 5/13 | 69.2% | ACTIVE |
| K | 715 | 2/13 | 92.3% | ACTIVE |
| ERA | 3.790 | 6/13 | 61.5% | ACTIVE |
| WHIP | 1.270 | 7/13 | 53.8% | ACTIVE |

**Standings Points from Active Categories: `78`**
Help text: `Total points: 80 (active: 78, punted: 2)`

**BUGS found:**

**Bug 1 — Ghost team "Twigs" inflates n_teams to 13:** The `league_standings` table contains a team named "Twigs" (with full season stats: .229 AVG, 53 HR, etc.) that does not appear in `league_rosters`. This is a renamed/abandoned team (the "ghost team" documented in the CLAUDE.md Bug D context). The page does NOT apply `filter_standings_to_valid_teams()` from `standings_utils.py` — it builds `all_team_totals` directly from the standings DataFrame without filtering. As a result:
- `n_teams = 13` instead of 12
- All ranks and strength bars display `/13` in the denominator (the UI hardcodes `/12` in the label text but displays the actual rank which may be out of 13)
- Team Hickey's HR rank shows 1/13 (correct with Twigs), but the page header displays "/12" — this is inconsistent

**Bug 2 — Standings Points formula hardcodes constant 13:** The code `sum(13 - int(r["Rank"].split("/")[0]) for r in impact_rows...)` assumes a 12-team league (correct formula: `n_teams + 1 - rank`). With the ghost team included (n_teams=13), a team ranked 13th earns 0 points (13-13=0), which is wrong — the minimum should be 1 point. For Team Hickey at rank 12/13 in AVG, the formula gives 13-12=1 point, which happens to be correct here, but a rank-13 team (which can occur) would get 0. The constant `13` should be replaced with `n_teams + 1` or taken from `config.num_teams + 1`.

**Bug 3 — "Your Total" for L shows 47.0 but ERA shows 3.790:** The page uses `format_stat(my_val, cat)` for rate stats and `f"{my_val:.0f}"` for others — correct formatting. However, L (Losses) is an inverse counting stat. The page shows rank 12/13 for L (worst), which is correct (47 losses = most losses = worst rank). But a novice sees "rank 12" for both AVG (.236, genuinely bad) and L (47 losses) without understanding that L rank 12 is also bad. No visual differentiation between "high rank means you are strong" vs "low rank means you are in a category you have many of" is provided.

---

### 4.8 "Standings Points from Active Categories" Metric

**Actual output (Punt SB):** `78`
**Help text:** `Total points: 80 (active: 78, punted: 2)`

**Issues:**
- A novice has no frame of reference for what 78 means. Maximum possible = 12×12 = 144 points (rank 1 in all 12 cats), average expected = 78 (rank 6.5 in each cat). The metric needs a denominator or benchmark ("78 out of 144 possible").
- The metric appears at the bottom of the page below the Standings Impact panel, not highlighted prominently.
- No comparison between the punted scenario and the baseline — there is no "Before punt: 80 points / After punt (active only): 78 points" delta display.

---

### 4.9 Punt L (Inverse Stat) — Gainers correctness check

**Actual output (Punt L, reconstructed):**

| Player | Pos | Original | Punt L | Change |
|---|---|---|---|---|
| Aaron Brooks | RP | -45.56 | -19.56 | +26.00 |
| Braxton Garrett | RP | -13.63 | -7.63 | +6.00 |
| Robby Snelling | RP | -5.36 | -1.02 | +4.33 |
| Mike Clevinger | SP | -6.27 | -2.27 | +4.00 |
| Brett Sears | P | +0.46 | +3.79 | +3.33 |

This works correctly — pitchers with many losses gain value when L is punted. However, the user still sees players with negative overall SGP values as "gainers," which is confusing without explanation.

---

### 4.10 Punt SV — Gainers all zero

Same problem as SB: the 15 "gainers" under Punt SV all show delta = +0.000 (hitters with no saves). The losers table correctly shows closers (Díaz, Miller, etc.) losing value.

---

### 4.11 Multi-category punt (SB + SV simultaneously)

**Actual output (Punt SB+SV, reconstructed):**

Same as Punt SB alone for gainers — all delta=0 for the top 15. The additive zeroing of two purely-counting stats with many zero-valued players still produces gainers with zero change. The losers table correctly stacks the effects.

---

## 5. Errors, Issues & Difficulties

| # | Description |
|---|---|
| E-01 | **Gainers table is broken for counting-stat punts (SB, SV, HR, R, RBI, K, W):** Any category where non-contributors score 0 produces 15 "gainers" with delta=0. The table title is factually wrong — these players did not "gain" anything. |
| E-02 | **Ghost team "Twigs" included in standings calculations:** The page builds `all_team_totals` from standings without calling `filter_standings_to_valid_teams()`. Result: ranks display as `/13` for a 12-team league, and all calculations are off. |
| E-03 | **Standings Points formula hardcodes 13:** `sum(13 - rank)` is only correct for a 12-team league with no ghost teams. With Twigs: 13-team denominator but constant 13 gives wrong points for rank=13. |
| E-04 | **Nav label "Punt Analyzer" vs page title "Punt Strategy Simulator" mismatch** (confirmed by orchestrator; confirmed in code: `render_page_header("Punt Strategy Simulator", ...)`). |
| E-05 | **2× row-by-row SGP compute (413× slower than available vectorized path):** The page calls `sgp_orig.total_sgp(player)` and `sgp_punt.total_sgp(player)` in `iterrows()` loops. `total_sgp_batch()` exists and is 413× faster. Adds ~2.1 seconds of blocking compute before any output. |
| E-06 | **"Losers" description says "potential trade targets" — backwards:** Players who *lose* value when you punt are players on your roster you should *trade away* (they are trade bait), not players you want to acquire. |
| E-07 | **No definition of "punt":** The word is used in title, label, instruction, and chip labels without ever being defined. A novice manager does not know what punting a category means or why they would do it. |
| E-08 | **Page has no roster-specific focus:** The gainers/losers tables show the full 9,888-player pool — not filtered to the user's roster or even to rostered + free agents. A novice cannot see "which of MY players lose value if I punt SB?" without scrolling through a 9,888-row dataset. |
| E-09 | **Empty state uses `st.info` instead of `render_empty_state()`:** Inconsistent with Combustion conventions on other pages. |
| E-10 | **No recommended punt strategies or contextual suggestion:** Given Team Hickey is 10th/12th overall with rank-12 AVG and rank-12 L (Losses), the page never says "Based on your standings, SB and AVG look like punt candidates." |
| E-11 | **The FIG numbering uses "FIG.10" (no zero-padding) but sub-panels use "FIG.10.1", "FIG.10.2", etc.** — style inconsistency with other pages that may use "FIG.04" format. |
| E-12 | **Standings impact panel is gated on `_HAS_CATEGORY_ANALYSIS`** — if that import fails, the entire standings impact section (FIG.10.4 + metric) silently disappears with no user-facing message. |
| E-13 | **No data-staleness indicator:** Data is ~1,219 minutes old (Yahoo last refreshed 2026-06-12). The page shows no freshness badge, no "data as of" timestamp, nothing. |
| E-14 | **The single `st.metric` for standings points is at the bottom of the page, below the long standings table** — it is the most important output but visually buried. |
| E-15 | **No user-team roster shown alongside the standings impact.** A user who sees "AVG rank 12/13" cannot immediately tell which of their players are dragging down their AVG. |

---

## 6. UI/UX & Visual Design Critique

### Layout & hierarchy

The page follows the top-down pattern: header → control → output panels. However, the layout is entirely sequential with no summary-first approach. The most actionable insight — "punting SB saves you 2 standings points from a category where you rank 11th" — is buried at the very bottom of the page as a single `st.metric`. A modern strategy tool should lead with the summary verdict.

The four `render_panel` blocks stack vertically in a single column. The Strategy Summary (FIG.10.1) could be a 2-column layout with the Gainers/Losers side by side; the current linear stack forces excessive scrolling.

### Gainers table design

The `_value_swing_table_html` function is well-crafted visually — team color accent on the left border, headshot images, team logos, IBM Plex Mono for numbers, column headers in Archivo uppercase. However, this visual polish is completely wasted when all 15 rows show `+0.00` change. Beautiful presentation of meaningless data is worse than no presentation.

### Color semantics

- Gainers show delta in `var(--fp-ember)` (orange/warm) — but orange in the Combustion system is the primary action/positive color, which is appropriate here.
- Losers show delta in `var(--fp-cold)` — cold blue is used to denote negative/decreasing values. Correct.
- PUNT chips use the "cold" chip style (blue), ACTIVE chips use "hot" (orange). This is semantically reasonable — punted categories are "cold" categories you've given up.

### Heatbar rendering (FIG.10.4)

The strength bar computation: `strength = (n_teams - rank + 1) / n_teams * 100`. With Twigs included (n_teams=13), rank 1 gives 100%, rank 12 gives 15.4%, rank 13 gives 7.7%. The bar correctly reflects the wrong data.

### Typography

The custom `_value_swing_table_html` function uses Archivo for column headers and IBM Plex Mono for numbers — consistent with Combustion token specs. The position chip uses IBM Plex Mono at 9.5px — correct for a compact monospaced badge. The player name is Inter at 13.5px semi-bold — matches `--font-body`.

### Empty state

`st.info("Select one or more categories above to see the punt analysis.")` is a plain Streamlit blue info box. The Combustion system provides `render_empty_state(title, body, icon_key)` for data-empty contexts. The empty state also misses the opportunity to show what the page _can_ do — a preview illustration or list of what outputs will appear would orient the user.

### Combo of novice-hostile elements

1. No definition of "punt"
2. No preselected suggestions
3. No connection to the user's actual team position
4. Gainers table shows zero-delta rows without explanation
5. The single most useful number (standings points active) is at the bottom
6. No call to action after the analysis (no "Trade these losers" button, no "Add these gainers" link)

This page is the most underbuilt page relative to its strategic importance. In a Head-to-Head categories league, punt strategy is one of the highest-leverage decisions a manager makes all season. The tool as built is a value re-ranking engine with no interpretation layer.

### Consistency with sibling pages

- **FIG numbering:** The eyebrow reads `FIG.10 — CATEGORY PUNT MODEL`. Other pages reviewed by the orchestrator used zero-padded `FIG.04` format. This page uses no zero-padding, consistent with the `FIG.4` issue flagged on the Pitcher Streaming page. All pages should use two-digit figure numbers (`FIG.10` is already two digits but the pattern is inconsistently applied across the suite).
- **`st.components.v1.html`:** NOT used on this page (renders tables via `st.markdown(unsafe_allow_html=True)` via `render_panel`). Not subject to the deprecation warning found on other pages. 

### Mobile

Full-width layout. The value-swing table has 5 columns including a headshot+name flex-layout cell. On narrow screens the headshot+name column will truncate or overflow. No `@media` responsive adjustments are present in the custom table HTML.

---

## 7. Recommendations (≥10, ordered by impact)

### Rec 1 — Fix the Gainers Table Logic (BLOCKER)

**Problem:** "Biggest Value Gainers" shows 15 players with Change = +0.00 when punting counting stats like SB or SV. These players did not gain anything.

**Fix:** Change the gainers table to show players sorted by **relative rank gain** (pre-punt rank vs post-punt rank in the pool), not absolute SGP delta. Alternatively, filter to `value_change > 0` before calling `nlargest` — if the result is empty, display `render_empty_state("No absolute gainers", "When punting SB, no players gain absolute value — speed specialists lose value, and everyone else stays the same.", "baseball")`. This gives an honest, informative result.

---

### Rec 2 — Filter Out Ghost Team "Twigs" from Standings (HIGH)

**Problem:** `all_team_totals` is built directly from `league_standings` without applying `filter_standings_to_valid_teams()`. "Twigs" (a renamed/abandoned team) is included, inflating n_teams to 13, making ranks show as `/13`, and distorting the strength bar.

**Fix:** Call `standings_utils.filter_standings_to_valid_teams(standings_df, valid_teams)` where `valid_teams` is the set of team names in `league_rosters`. Already implemented in `standings_utils.py` — just not called here.

---

### Rec 3 — Fix the Standings Points Formula (HIGH)

**Problem:** `sum(13 - rank)` assumes exactly 12 teams. With 13 teams (ghost included), a rank-13 team gets 0 points (should be 1). Even after fixing the ghost team issue, hardcoding `13` is fragile.

**Fix:** Replace `13` with `config.num_teams + 1` (or `_n_teams + 1` using the local `_n_teams` variable already computed). The correct formula is `n_teams + 1 - rank`, giving rank-1 = n_teams points, rank-n_teams = 1 point.

---

### Rec 4 — Switch to Vectorized SGP Computation (HIGH)

**Problem:** The page calls `sgp_orig.total_sgp(player)` and `sgp_punt.total_sgp(player)` in `iterrows()` loops across 9,888 players — measured at ~2.1 seconds for both passes.

**Fix:** Replace both loops with `total_sgp_batch(pool)` calls (already implemented in `SGPCalculator`). Reduces compute from ~2.1s to ~6ms — a 413× speedup. With this fix, the progress bar becomes unnecessary (computation is instant); it can be removed or compressed to a brief `st.spinner`.

```python
# Current (slow):
for _, player in pool.iterrows():
    orig_sgps.append(sgp_orig.total_sgp(player))

# Fix (413x faster):
pool["original_sgp"] = sgp_orig.total_sgp_batch(pool)
pool["punt_sgp"] = sgp_punt.total_sgp_batch(pool)
```

---

### Rec 5 — Add a "What is punting?" Explainer (HIGH)

**Problem:** The word "punt" is used 15+ times on this page without being defined. A novice manager who has not read about fantasy strategy will not understand what punting means, when to do it, or why this tool matters.

**Fix:** Add a collapsible expander ("What is a punt strategy?") directly below the intro paragraph with a 3-sentence explanation: "Punting a category means intentionally giving up on competing in that category for the season. Instead of trying to be competitive in all 12 categories, you focus your roster resources on the 11 (or 10, or 9) you can win. The tool below shows how player values shift when a category is removed from your strategy." Keep it collapsed by default so it doesn't bloat the page for experienced users.

---

### Rec 6 — Add a Roster-Specific View (HIGH)

**Problem:** The gainers/losers tables show all 9,888 pool players. A manager using this page wants to know specifically: "Which of MY 27 rostered players lose value if I punt SB? And which free agents become more attractive?"

**Fix:** Add a "My Roster" vs "All Players" toggle (radio or tabs: "Full Pool | My Roster | Free Agents") to filter the gainers/losers tables. The roster data is available via `yds.get_rosters()` already fetched for the standings section. Show a 3-panel view: roster players who gain, roster players who lose, and FA pickups who gain.

---

### Rec 7 — Fix the "Losers = Potential Trade Targets" Label (MEDIUM)

**Problem:** The subtext reads: "Players whose value decreases when the punted categories are removed — potential trade targets." But players who lose value under your punt strategy are players you should **trade away** (offer to teams who still value that category), not players you should acquire.

**Fix:** Change to: "Players who lose the most value under your punt strategy — strong trade bait for teams that still need these categories." Optionally add a link to the Trade Finder page.

---

### Rec 8 — Surface Smart Punt Suggestions Based on Current Standings (MEDIUM)

**Problem:** The page is purely reactive — the user must know which category to punt before using the tool. A user ranked 12th/13th in both AVG and L (Losses) is a prime punt candidate in both, but the page offers no guidance.

**Fix:** Before the multiselect, add a "Suggested punt categories" auto-recommendation section that identifies Team Hickey's weakest 1-3 categories (by rank, after filtering ghost teams) and pre-suggests them with one click: "Based on your standings, these are your weakest categories. Click to load as punt candidates: `AVG (rank 12)` `L (rank 12)` `SB (rank 11)`." This alone transforms the page from a blank canvas into an actionable advisor.

---

### Rec 9 — Promote the Standings Points Metric and Add a Before/After Delta (MEDIUM)

**Problem:** The most actionable summary — "if you punt SB, your active category standings points are 78" — is buried at the very bottom of the page as a small `st.metric`.

**Fix:** Move the metric to the top of the results area (between the strategy summary panel and the gainers/losers panels). Add a **before/after comparison**: "Before punt: 80 points (avg rank 6.7/12). After punting SB: 78 active points (avg rank 5.9/11 in active cats)." This shows whether the punt actually improves your competitive standing in the remaining categories. Also add a denominator: "78 of 132 possible active points" (11 cats × 12 max points each).

---

### Rec 10 — Resolve the Nav Label / Page Title Naming Drift (MEDIUM)

**Problem:** Sidebar label says "Punt Analyzer," page H1 wordmark says "Punt Strategy Simulator." Users encounter two different names.

**Fix:** Pick one. "Punt Analyzer" is more concise and matches the navigation label. Update `render_page_header("Punt Strategy Simulator", ...)` to `render_page_header("Punt Analyzer", ...)`. The eyebrow can keep "CATEGORY PUNT MODEL" as the descriptor.

---

### Rec 11 — Replace Empty State `st.info` with `render_empty_state()` (LOW)

**Problem:** The empty state (no category selected) uses `st.info(...)` — a plain Streamlit blue box — instead of `render_empty_state()` from `src/ui_shared.py` as specified by the Combustion design system.

**Fix:** Replace:
```python
st.info("Select one or more categories above to see the punt analysis.")
```
with:
```python
render_empty_state("No category selected", "Choose one or more categories to punt above. The tool will show you which players gain or lose value and how your team's standings profile changes.", "baseball")
```

---

### Rec 12 — Add Data Freshness Indicator (LOW)

**Problem:** The page shows no indication of how old its data is. The DB data is currently ~1,219 minutes stale (Yahoo last synced 2026-06-12). Player SGP values and standings ranks could be out of date.

**Fix:** Display a small freshness chip or caption below the header (consistent with how other pages show data recency). Can use the `refresh_log` entry for `yahoo_standings` (last refresh: `2026-06-12T17:43:09`) to compute and display "Standings data as of [date]."

---

### Rec 13 — Show Category Splits (Hitting vs Pitching) in the Multiselect (LOW)

**Problem:** The multiselect shows all 12 categories in a flat list — no grouping. A novice cannot distinguish hitting from pitching categories without prior knowledge. ERA and WHIP look like "good stats to improve" not "inverse stats (lower is better)."

**Fix:** Optionally render two separate multiselects ("Punt hitting categories" / "Punt pitching categories") or add a hint in the multiselect label. At minimum, add a tooltip on ERA/WHIP/L noting they are inverse stats (lower is better in this league).

---

## 8. Severity-Tagged Issue List

- [BLOCKER] E-01 — Gainers table shows 15 players with Change = +0.00 for all counting-stat punts (SB, SV, HR, R, RBI, W, K). Label says "biggest value gainers" but none gained any value.
- [HIGH] E-02 — Ghost team "Twigs" included in standings; n_teams inflated to 13; ranks and strength bars show `/13` instead of `/12`.
- [HIGH] E-03 — Standings Points formula hardcodes constant 13; fails when n_teams ≠ 12 or ghost team present.
- [HIGH] E-05 — Row-by-row SGP computation 413× slower than available vectorized method; adds ~2.1 seconds of blocking compute on first interaction.
- [HIGH] E-07 — "Punt" not defined anywhere on the page; novice users have no context for what this tool does or why to use it.
- [HIGH] E-08 — No roster-specific view; all 9,888 pool players shown; user cannot identify which of their own players lose value under a punt strategy.
- [MEDIUM] E-04 — Nav label "Punt Analyzer" vs page title "Punt Strategy Simulator" naming drift.
- [MEDIUM] E-06 — "Potential trade targets" label on losers panel is semantically backwards; these are trade bait, not targets.
- [MEDIUM] E-10 — No proactive punt recommendations based on user's actual standings profile.
- [MEDIUM] E-14 — Standings points metric buried at bottom of page; needs to be prominent and paired with a before/after delta.
- [MEDIUM] E-12 — Standings impact section silently disappears if `_HAS_CATEGORY_ANALYSIS` import fails; no user-facing error or fallback.
- [MEDIUM] E-13 — No data-staleness indicator; standings data is ~1,219 minutes old with no visual cue.
- [LOW] E-09 — Empty state uses `st.info` instead of design-system `render_empty_state()`.
- [LOW] E-11 — FIG numbering style inconsistent with sibling pages (no zero-padding convention enforced).
- [LOW] E-15 — No connection between standings rank display and the user's roster (which players are dragging down each category).
- [POLISH] — The heatbar displays `/13` denominator data (ghost team) even when the rank label in the table hardcodes `/12`, creating a silent visual inconsistency in the bar's fill vs its textual rank.
- [POLISH] — The progress bar runs at 20→50→100% with `time.sleep(0.3)` before hiding; with a vectorized fix (Rec 4) the compute is instant and the progress bar becomes a distraction that flashes and vanishes.
- [POLISH] — Single-column vertical layout for four panels forces excessive scrolling; gainers/losers could render side-by-side with `st.columns(2)`.
- [POLISH] — Multiselect placeholder text is Streamlit default "Choose options"; a custom `placeholder="Pick categories to punt..."` would be cleaner.
- [POLISH] — No loading/spinner fallback if `_has_category_analysis=True` but Yahoo standings are empty; the FIG.10.4 panel silently does not render with zero user feedback.

---

*Report written from: source code (`pages/10_Punt_Analyzer.py`, `src/valuation.py`, `src/standings_utils.py`, `src/ui_shared.py`) + three read-only DB query sessions capturing live data from `league_standings`, `league_rosters`, `refresh_log` tables. No browser tools used. All temp scripts deleted.*
