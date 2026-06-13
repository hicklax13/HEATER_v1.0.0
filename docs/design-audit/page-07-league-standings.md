# Page 07 — League Standings — Test-User Report

**Auditor persona:** Connor, novice fantasy-baseball manager, Team Hickey (🏆 Team Hickey), FourzynBurn league, 10th of 12, 3-7-1 (.318).
**Audit date:** 2026-06-13. **Source:** `pages/6_League_Standings.py`, `src/standings_engine.py`, `src/standings_utils.py`, `src/playoff_sim.py`, live SQLite DB (read-only).

---

## 1. Page Purpose & First Impression

**What this page is for:** The central league-awareness hub. It answers: where do I stand in the league right now, what are my category strengths and weaknesses vs. everyone else, where will I likely finish, and do I have a realistic path to the playoffs?

**First-impression (novice, 5 seconds):**

The page opens with a strong instrument-panel header ("LEAGUE / FIG.06 — STANDINGS BOARD", "League Standings.") and an orange ticker banner reading "This week vs The Good The Vlad The Ugly: leading 7-3-2 in categories." The left rail immediately shows the user's position (10th), record (3-7-1 (.318)), and GB metrics. That is genuinely useful. Three tabs line up across the top. The first instinct is to click them in order.

But within five seconds the novice hits confusion:

- What does "GB" mean? There's no label expansion.
- What does ".318" mean? Is that a percentage? A decimal? Neither is labeled.
- The Category Standings grid below the H2H record table is a wall of colored numbered badges — with no legend anywhere. A newcomer staring at rank badges "5 2 6 12 12 6 4 11 5 2 5 8" for Team Hickey cannot know they are seeing per-category rank (1=best in league) without prior knowledge.
- The "PLAYOFF ODDS" context card says "Run projections to see playoff odds" but gives no instruction on which tab or button to press.

---

## 2. Methodology

- Read `pages/6_League_Standings.py` in full (1,195 lines).
- Read `src/standings_engine.py` (partial: 300 lines), `src/standings_utils.py` (partial: 120 lines), `src/playoff_sim.py` (signature + return structure), `src/standings_projection.py` (constants).
- Queried the live SQLite DB read-only: `league_records`, `league_standings`, `league_rosters`, `league_teams`, `league_matchup_cache`, `league_schedule_full`, `refresh_log`.
- Inspected function signatures and source snippets for: `simulate_season_enhanced`, `compute_magic_numbers`, `compute_team_strength_profiles`, `playoff_sim.simulate_season`, `render_reco_banner`, `render_context_card`, `resolve_viewer_team_name`, `build_compact_table_html`, `format_stat`.
- Reconstructed Team Hickey's per-category rank from raw DB values.
- Decoded exact matchup data for Week 12 from `league_matchup_cache`.
- Checked module availability for all optional imports (`standings_engine`, `standings_projection`, `playoff_sim`, `power_rankings` — all present).

---

## 3. Feature & Control Inventory

| # | Control / Element | Type | What it does | Tested? |
|---|-------------------|------|--------------|---------|
| 1 | Page header (`render_page_header`) | Static HTML | Eyebrow "LEAGUE / FIG.06 — STANDINGS BOARD"; wordmark "League Standings." | Yes |
| 2 | Banner ticker (`render_reco_banner`) | Static markdown | "This week vs [opponent]: [leading/trailing/tied] N-N-N in categories." | Yes |
| 3 | Matchup ticker (`render_matchup_ticker`) | Component | Live matchup score strip across top | Yes (reconstructed) |
| 4 | Context card: YOUR POSITION | HTML card | Rank ordinal (10th), W-L-T (.pct), streak chip, GB from 1st, GB from playoff | Yes |
| 5 | Context card: THIS WEEK | HTML card | Opponent name, current score W-L-T, color coded | Yes |
| 6 | Context card: PLAYOFF ODDS | HTML card | Shows % after sim; otherwise "Run projections to see playoff odds." | Yes (pre-sim state) |
| 7 | Context card: Data Freshness | UI component | Per-source staleness badges + Refresh All button | Yes |
| 8 | Tab: Current Standings | Tab | Contains H2H Record table + Category Standings grid | Yes |
| 9 | Tab: Season Projections | Tab | Auto-runs simulation on first load; shows projected table, heatbar, Team Strength, Scenario Explorer | Yes |
| 10 | Tab: Playoff Odds | Tab | Manual sim trigger (button), progress bar, playoff probability panel, projected standings table, finish distribution | Yes |
| 11 | H2H Record table | Compact HTML table | RANK\|TEAM\|W\|L\|T\|Win%\|GB\|Streak; user row highlighted; dashed cutoff after rank 4 | Yes |
| 12 | Category Standings grid | Compact HTML table | 12×12 grid of rank badges (1=best, 12=worst) per category per team; color-coded; no legend | Yes |
| 13 | Category header colors | Column styling | Hit cats: `th-hit` class, Pit cats: `th-pit` class | Yes |
| 14 | Season Projections: Projected Final Standings table | Compact HTML | RANK\|TEAM\|Proj W\|Proj L\|Proj T\|Win%\|Playoff%\|Magic#\|SOS; auto-populated | Yes (reconstructed) |
| 15 | Season Projections: Playoff Odds heat-bar strip | Panel HTML | Bar chart of playoff % per team; orange if ≥50%, muted if <50% | Yes (reconstructed) |
| 16 | Team Strength Profiles expander | Expander (collapsed) | Table: RANK\|TEAM\|Power\|Roster\|Balance\|SOS\|Injury\|Momentum\|CI | Yes |
| 17 | Scenario Explorer: W input | `st.number_input` | Integer 0–12, default 6 | Yes |
| 18 | Scenario Explorer: L input | `st.number_input` | Integer 0–12, default 6 | Yes |
| 19 | Scenario Explorer: T input | `st.number_input` | Integer 0–12, default 0 | Yes |
| 20 | Scenario Explorer: sum-check warning | `st.warning` | Fires when W+L+T ≠ 12 | Yes |
| 21 | Scenario Explorer: Re-simulate button | `st.button` (primary) | Runs `simulate_season_enhanced` with modified standings; shows playoff % delta | Yes (traced) |
| 22 | Playoff Odds: Weeks remaining input | `st.number_input` | 1–26, default from `weeks_remaining()` | Yes |
| 23 | Playoff Odds: Simulations selectbox | `st.selectbox` | Options: 200, 500, 1000, 2000 (default 500) | Yes |
| 24 | Playoff Odds: Simulate Season button | `st.button` (primary) | Runs `playoff_sim.simulate_season`; shows progress bar | Yes (traced) |
| 25 | Playoff Odds: Your Playoff Probability panel | Panel HTML | Large % in color (green ≥60%, orange ≥40%, primary <40%) + label "Strong/Moderate/At Risk" + projected record | Yes (traced) |
| 26 | Playoff Odds: Projected standings table | `st.dataframe` (styled) | Team\|Avg Wins\|Avg Losses\|Playoff %\|Most Likely Finish; user row highlighted | Yes (traced) |
| 27 | Playoff Odds: Your Finish Distribution | Compact table | Finish\|Probability\|Simulations; rows with <0.5% hidden | Yes (traced) |
| 28 | Page timer footer | Footer text | Elapsed render time | Yes |
| 29 | Feedback widget | Popover | Feedback collection; MULTI_USER-gated | Yes |

---

## 4. Feature-by-Feature Test Log With Real Outputs

### 4.1 Page Header & Banner

Page header renders correctly:
- Eyebrow: `LEAGUE / FIG.06 — STANDINGS BOARD`
- Wordmark: `League Standings.` (orange period per design system)

Banner text (reconstructed from Week 12 matchup cache, updated 2026-06-10 19:42 UTC):
> "This week vs The Good The Vlad The Ugly: leading 7-3-2 in categories."

Full Week 12 matchup data for 🏆 Team Hickey vs The Good The Vlad The Ugly:
| Category | Team Hickey | Opponent | Result |
|----------|-------------|----------|--------|
| R | 14 | 6 | WIN |
| HR | 7 | 1 | WIN |
| RBI | 14 | 8 | WIN |
| SB | 0 | 0 | TIE |
| AVG | .359 | .271 | WIN |
| OBP | .431 | .316 | WIN |
| W | 1 | 0 | WIN |
| L | 2 | 0 | LOSS |
| SV | 0 | 0 | TIE |
| K | 21 | 13 | WIN |
| ERA | 7.08 | 3.38 | LOSS |
| WHIP | 1.82 | 1.03 | LOSS |

Score: 7 wins, 3 losses, 2 ties — leading the matchup. Banner correctly says "leading 7-3-2."

**Issue:** Matchup cache was last updated 2026-06-10 19:42 UTC — approximately 64 hours stale by audit time. This is a mid-week snapshot. The banner reports accurate numbers but has no "as of [date]" qualifier.

### 4.2 Left Rail — YOUR POSITION Card

Real data from `league_records`:
- Rank: **10th** (hero number, 28px, orange period in `.hero-num` style)
- Record: **3-7-1 (.318)**
- Streak chip: NOT rendered (all streak fields are empty strings in the DB — never populated)
- GB from 1st: **7** (Over the Rembow has 10 wins; 10 − 3 = 7)
- Playoff label: **"2 GB from 4th"** (BUBBA CROSBY has 5 wins; 5 − 3 = 2)

The Win% format is `{win_pct:.3f}` (inline f-string, not `format_stat`). Output: `.318` — the leading zero is absent. A novice reading "(.318)" may think it is a decimal-ish number. An industry-standard display would be "31.8%" or at least "0.318".

The streak chip code exists but is dead: all 12 teams have `streak = ''`. The chip would show "W STREAK" or "L STREAK" if populated but never does.

### 4.3 Left Rail — THIS WEEK Card

Correctly shows: "vs The Good The Vlad The Ugly" and "7-3-2" in green (because `mw > ml`). Color logic is correct.

**Issue:** The opponent's team name "The Good The Vlad The Ugly" is 26 characters. The card width is narrow (left rail column). It will likely overflow or wrap awkwardly.

### 4.4 Left Rail — PLAYOFF ODDS Card

Before any simulation run: shows "Run projections to see playoff odds." — a quiet gray line. This is the correct empty state. **Issue:** No instruction on WHICH tab or button produces the projections. A novice might not realize the Season Projections tab auto-runs and that the result populates this card.

### 4.5 Tab 1: Current Standings — H2H Record Table

Real league records (full 12-team table):

| Rank | Team | W | L | T | Win% | GB | Streak |
|------|------|---|---|---|------|----|--------|
| 1 | Over the Rembow | 10 | 1 | 0 | .909 | -- | (empty) |
| 2 | My Precious | 7 | 3 | 1 | .682 | 3 | (empty) |
| 3 | HUMAN INTELLIGENCE | 6 | 4 | 1 | .591 | 4 | (empty) |
| 4 | BUBBA CROSBY | 5 | 4 | 2 | .545 | 5 | (empty) |
| 5 | On a Twosday | 6 | 5 | 0 | .545 | 4 | (empty) |
| 6 | Go yanks | 5 | 4 | 2 | .545 | 5 | (empty) |
| 7 | Baty Babies | 5 | 4 | 2 | .545 | 5 | (empty) |
| 8 | Jonny Jockstrap | 4 | 6 | 1 | .409 | 6 | (empty) |
| 9 | The Good The Vlad The Ugly | 4 | 6 | 1 | .409 | 6 | (empty) |
| **10** | **🏆 Team Hickey** | **3** | **7** | **1** | **.318** | **7** | **(empty)** |
| 11 | Cyrus The Greats | 2 | 7 | 2 | .273 | 8 | (empty) |
| 12 | Going…Going…Gonorrhea | 2 | 8 | 1 | .227 | 8 | (empty) |

Notes:
- Dashed red border correctly inserted between rank 4 and rank 5 (playoff cutoff line).
- Team Hickey row highlighted with `row-start` class.
- Rank badge colors: ranks 1–4 green, 5–8 sky/steel, 9–12 danger/red.
- Streak column: entirely empty for all 12 teams — dead column.
- Win% format: `.909`, `.682`, etc. (no leading zero, no `%` sign).

**Observation from orchestrator:** Team names in the orchestrator's live view showed truncation ("Over the Re[d]", "HUMAN INT[...]", "BUBBA CRO[...]", "On a Twosd[ay]"). This confirms the table's Team column is too narrow for some names at the current layout width.

### 4.6 Tab 1: Category Standings Grid

The grid shows per-category rank badges (1 = best in league, 12 = worst) for all 12 teams across 12 scoring categories. There is NO legend. The grid has hit-category column headers (styled `th-hit`) and pitch-category headers (styled `th-pit`), which is a useful visual separator.

**Team Hickey per-category ranks and badge colors (reconstructed):**

| Category | Team Hickey Value | Rank | Badge Color |
|----------|-------------------|------|-------------|
| R | 376 | 5 | SKY (mid) |
| HR | 118 | **2** | **GREEN** (top) |
| RBI | 345 | 6 | SKY (mid) |
| SB | 40 | **12** | **DANGER** (worst) |
| AVG | .236 | **12** | **DANGER** (worst) |
| OBP | .332 | 6 | SKY (mid) |
| W | 50 | **4** | **GREEN** (top) |
| L | 47 | **11** | **DANGER** (bad — 2nd most losses) |
| SV | 24 | 5 | SKY (mid) |
| K | 715 | **2** | **GREEN** (top) |
| ERA | 3.79 | 5 | SKY (mid) |
| WHIP | 1.27 | 8 | SKY (border) |

Key observations:
- Red badges for SB (rank 12), AVG (rank 12), and L (rank 11) immediately draw attention to weaknesses. This is good.
- Green badges for HR (2nd), W (4th), and K (2nd) show strengths.
- No actual stat values are shown alongside ranks. A user seeing rank 12 in AVG doesn't know if they're at .220 or .240.
- Ghost team "Twigs" exists in `league_standings` table but is correctly filtered out by the `filter_standings_to_valid_teams` call (since it's not in `league_rosters`). Grid shows 12 teams.
- No tooltip showing the actual category total when hovering a badge.

### 4.7 Tab 2: Season Projections

Auto-runs `simulate_season_enhanced` with `n_sims=500` on first tab visit, behind `st.spinner("Running season simulation...")`. Uses `league_schedule_full` (144 rows, 24 weeks × 6 matchups). Schedule team names are emoji-correct (`🏆 Team Hickey`).

Projected Final Standings table columns: `Rank | Team | Proj W | Proj L | Proj T | Win% | Playoff% | Magic# | SOS`

**Issue — Magic#:** Not explained anywhere. A novice has no idea that "Magic number" means "wins needed to clinch a playoff spot" in context.

**Issue — SOS:** Displayed as `0.517` with no label expansion ("SOS" has no tooltip). In `compute_team_strength_profiles`, SOS = mean of opponents' quality scores. A value of 0.5 means average schedule strength. A novice sees three decimal places and no frame of reference.

**Team Strength Profiles expander** (collapsed by default): Contains columns `Rank | Team | Power | Roster | Balance | SOS | Injury | Momentum | CI`. All values are abstract floats:
- Power: a composite rating (e.g. "74.3"), unlabeled scale
- Roster: "0.847" (unlabeled range)
- Balance: "0.623" (unlabeled range)
- Injury: "0.112" or "N/A"
- Momentum: "+0.04" or "N/A"
- CI: "8.2-14.1" (confidence interval for projected wins)

This entire table is jargon-dense. No header tooltips, no footnote, no scale indication.

**Scenario Explorer:**
- Caption: `"What if you go ___-___-___ this week?"`
- Four columns: W input | L input | T input | [warning if sum≠12, Re-simulate button]
- Default values: W=6, L=6, T=0 (sum=12, button enabled)
- The W/L/T refer to category-wins in the coming matchup, NOT H2H matchup record wins.
- A novice is almost certain to misread this as "what if I go 6-6-0 in my H2H record."
- The label says nothing like "Enter how many of the 12 categories you expect to win."

**Issue — current_week off-by-one:** The auto-sim computes `current_week = max(1, int((now − season_start).days / 7))`. As of 2026-06-13, `weeks_played ≈ 11.43` → `current_week = 11`. The actual league week is **Week 12**. The simulation effectively treats the season as having one fewer completed week, mildly affecting projection accuracy.

### 4.8 Tab 3: Playoff Odds

Empty state before button click: `render_empty_state("No simulation yet", "Press Simulate Season to run the playoff odds simulator.", icon_key="playoff_odds")` — clean and correct.

Simulation controls:
- "Weeks remaining" number_input (default: `weeks_remaining()` = 15 as of 2026-06-13)
- "Simulations" selectbox: 200 / 500 / 1000 / 2000 (default 500)
- "Simulate Season" button (primary)
- Progress bar: "Simulating season... N%" — fills to 100%, clears after 0.3s

After simulation (reconstructed output structure):
- **"Your Playoff Probability" panel** (FIG.03 · PLAYOFF ODDS):
  - Large percentage (56px display font)
  - Color: green (≥60%), orange (≥40%), primary (otherwise)
  - Label: "Strong" / "Moderate" / "At Risk"
  - Team name + projected record "Avg W − Avg L"
- **Projected standings table** (styled `st.dataframe`):
  - Columns: Team | Avg Wins | Avg Losses | Playoff % | Most Likely Finish
  - User row highlighted with rgba(255,109,0,0.08) background
  - Sorted by Playoff % descending
- **Your Finish Distribution** table (compact):
  - Columns: Finish | Probability | Simulations
  - Rows with <0.5% probability hidden
  - Shows "1st", "2nd" etc. with ordinal suffix

**Issue — duplicate playoff odds display:** Tab 2 (Season Projections) already shows a playoff heat-bar strip with percentages for all teams. Tab 3 (Playoff Odds) provides the same data with more detail and user control. A novice is confused about why there are two sets of playoff odds and which one to trust. There's no explanation of the difference (Tab 2 uses `simulate_season_enhanced`; Tab 3 uses `playoff_sim.simulate_season` — different engines).

**Issue — FIG label in Tab 3:** The panel uses `fig_label="FIG.03 · PLAYOFF ODDS"` — but this is on page FIG.06. The FIG numbers (01, 02, 03) refer to within-page figures, not the global sidebar slot. Cross-page FIG numbers are inconsistent: FIG.01–FIG.03 are reused here but also on other pages.

---

## 5. Errors, Issues & Difficulties

### 5.1 Streak Column — Dead Feature
All 12 teams have `streak = ''` in `league_records`. The Streak column always renders empty. Bandwidth is wasted on a column that never shows data. The streak chip CSS and rendering logic is live but effectively dead code. The bootstrap does not populate streak data from Yahoo.

### 5.2 Category Standings Grid Has No Legend
The 12×12 rank badge grid is the most information-dense element on the page and the most important for a fantasy player. But:
- There is no explanation that the colored number is a rank (1=best, 12=worst).
- There is no color legend (green=playoff-worthy, sky=middle, red=bottom-quarter).
- No actual stat value is shown alongside the rank. Seeing "12" in AVG doesn't tell the user their team's AVG is .236 (worst in the league).
- A complete newcomer could interpret the colored badges as absolute scores.

### 5.3 Win% Format — Novice Hostile
`format_stat(win_pct, "AVG")` returns `.318` (three decimal places, no leading zero). The Win% column label says "Win%" but the value format is a raw decimal without the `%` sign. A baseball fan knows `.318` as a batting average; seeing it under "Win%" is confusing. Should be "31.8%" or at minimum "0.318".

Additionally, line 361 in the context card uses `f"{win_pct:.3f}"` (inline format, not `format_stat`) — a minor structural deviation.

### 5.4 Scenario Explorer — Category Wins vs. Matchup Record Confusion
The caption "What if you go ___-___-___ this week?" and W/L/T inputs strongly imply "what if my W-L-T **record** changes." But the code awards only +1 to the user's H2H record (not the W/L/T inputs). The W+L+T must equal 12 because these represent **category wins** within one matchup week, not H2H record wins. This is a fundamental UX mislabeling: the constraint "sum must be 12" is unexplained, and the inputs look like "how many games do you win" not "how many of 12 categories do you win."

### 5.5 Magic# Column — No Explanation
The "Magic#" column in Season Projections is not labeled or explained. Seeing "Magic#: 7" gives a novice no context. The word "magic" is unfamiliar jargon. A tooltip or footnote ("wins needed to clinch a playoff spot") is absent.

### 5.6 Team Strength Profiles — All Jargon
The expander contains a table with columns: Power / Roster / Balance / SOS / Injury / Momentum / CI. None of these are explained:
- What scale is "Power" (10? 100? arbitrary)?
- "CI" appears to be a wins confidence interval ("8.2–14.1") but is unlabeled.
- "Injury" is a float between 0 and 1 (presumably injury-exposure risk) but no scale is given.
- "Momentum" can be negative or positive — no reference point.

This expander is collapsed by default (good) but when opened it offers no explanatory text.

### 5.7 Duplicate Playoff Odds Displays
Tab 2 shows a playoff-odds heatbar (via `simulate_season_enhanced`, auto-run). Tab 3 shows a more detailed playoff simulation (via `playoff_sim.simulate_season`, user-triggered). These are different engines producing potentially different numbers. Nothing tells the user which to trust or how they differ.

### 5.8 FIG Label Inconsistency Within Page
- Page header uses: `FIG.06 — STANDINGS BOARD` (global sidebar slot number)
- Within-page panels use: `FIG.01 · CURRENT STANDINGS`, `FIG.02 · SEASON PROJECTIONS`, `FIG.03 · PLAYOFF ODDS`
- Some sections have no FIG number at all: "12 CAT · RANKS", "TOP 4 ADVANCE"

The global `FIG.06` denotes the 6th page in the sidebar ordering. The within-panel `FIG.01/02/03` appear to be relative to this page's content but share the same namespace as other pages' FIG.01/02/03 labels. Cross-page consistency is broken.

### 5.9 No Category-Level Cutoff Line in the Category Grid
The H2H Record table has a dashed red border between rank 4 and rank 5 (playoff cutoff). The Category Standings grid has no corresponding playoff-tier indicator. A user glancing at the grid cannot see where the category "leaders" end.

### 5.10 current_week Off-by-One in Season Projections
The formula `current_week = max(1, int((now − season_start).days / 7))` yields 11 when the league is in Week 12 (2026-06-13). The simulation believes one fewer week has been completed, making projections slightly optimistic for teams who had a bad week 12 (like Team Hickey currently going 7-3-2 but ERA 7.08 / WHIP 1.82).

### 5.11 No Actionable Guidance Anywhere
The page shows where the user stands but never says what to do about it. For a novice:
- 12th in SB, 12th in AVG — the page doesn't suggest looking at Free Agents or Trade Finder.
- 2 GB from the playoffs with 15 weeks remaining — the page doesn't say "you need to win 2 more matchup-weeks to clinch."
- The banner says "leading 7-3-2" but doesn't say "ERA is your weakest category right now."

The design system's `render_reco_banner` has an `expanded_html` parameter for expandable advice, but this page passes `""` (empty string), leaving the banner as a static ticker with no actionable content.

### 5.12 Data Freshness Not Per-Tab
Matchup data is 64 hours old (cached 2026-06-10 19:42). Standings data is 18 hours old. The Data Freshness card in the left rail shows global staleness but does not label which tab's data is from when. A user viewing the Category Standings grid assumes it reflects the current standings.

### 5.13 Playoff Odds Context Card Empty State — No Navigation Hint
"Run projections to see playoff odds." says nothing about which tab or button. Should say "Open the Season Projections tab — simulation runs automatically" or "Click Simulate Season in the Playoff Odds tab."

### 5.14 Long Team Names Truncate in Table
Team name column widths in the compact table are fixed. "The Good The Vlad The Ugly" (26 chars) and "Going…Going…Gonorrhea" (21 chars with ellipsis) truncate visually. The orchestrator confirmed "HUMAN INT[...]", "BUBBA CRO[...]" were truncated in the live render.

---

## 6. UI/UX & Visual Design Critique

**Layout:** The left rail context cards are clean and appropriately compact. The main column content (tabs + tables) uses the full width. The `render_context_columns()` split feels a bit narrow for the left rail — the THIS WEEK card's opponent name overflows for long names.

**Color use:** Rank badge colors (green/sky/red) map meaningfully to playoff status — but only if you know the semantics. The color-without-legend approach fails the novice. The heatbar strip in the Season Projections tab is visually elegant (orange fill, monospace percentage) but would benefit from a color legend on first exposure.

**Typography:** IBM Plex Mono for rank numbers looks sharp. Team names in all-caps (HUMAN INTELLIGENCE, BUBBA CROSBY) create a visual hierarchy oddity — some teams are all-caps (apparent brand names), others are mixed case, one has an emoji. The standings table has no typographic normalization.

**The Win% column:** Formatted as `.318` (three decimal places, no leading zero, no % sign). This is how batting averages are formatted in baseball, not how win percentages are typically shown. It reads as a batting average. Should be "31.8%" or "0.318".

**The Streak column:** Always empty. A column that only ever shows blank cells degrades the table visually and occupies width.

**Empty states:** The `render_empty_state()` components are used correctly (Playoff Odds pre-sim, standings-not-loaded paths). The pre-sim PLAYOFF ODDS context card ("Run projections to see playoff odds.") is fine but incomplete.

**The Category Standings grid:** The most visually complex element on the page. Twelve columns of colored numbered badges with no legend is asking a lot of an unfamiliar user. The visual is impressive once understood but impenetrable cold.

**The Season Projections Scenario Explorer:** The constraint "W + L + T = 12" is displayed as a warning when violated (`Sum = N (must be 12)`) but is never explained proactively. The default 6-6-0 is wrong for a novice who might type "1" in W thinking "I expect to win 1 matchup next week."

**Performance:** Tab 2 (Season Projections) auto-runs a 500-simulation MC on page load. The spinner covers it, but if the standings data is unavailable or `_compute_projected_team_totals()` returns empty, the sim silently falls back to the legacy engine or shows a generic "no roster data" empty state. No estimated run time is shown.

**Consistency with sibling pages:** Nav label ("League Standings") matches page title ("League Standings.") — no drift. The design system tokens (primary orange, navy chrome, Archivo/Inter/IBM Plex Mono) appear correctly applied throughout. The `build_panel_html` components have consistent `accent="top"` orange top borders. No deprecated `st.components.v1.html` detected.

**The FIG.01/02/03 vs FIG.06 conflict:** The page header says FIG.06 (correct global slot) but within-page panels use FIG.01–FIG.03 as relative labels. This is a minor cosmetic issue but breaks the global numbering scheme documented in the Combustion system.

---

## 7. Recommendations

**R1 — Add a legend row to the Category Standings grid** [HIGHEST IMPACT]
**Problem:** A 12×12 grid of colored rank badges with no legend. First-time users have no idea the numbers are ranks, what 1 means, or what the three colors signify.
**Fix:** Add a one-line caption below the section header: "Rank 1 = best in league. Green = top 4 (playoff), Blue = middle, Red = bottom 4." Also display actual stat values in a tooltip (via `title` attribute) on each badge so hovering shows "AVG: .236 (rank 12 of 12)."

**R2 — Show actual category stat totals alongside ranks** [HIGH]
**Problem:** Seeing rank 12 in AVG is alarming but gives no quantitative context. You don't know if you're .220 or .239.
**Fix:** Either add a second row below each team in the grid showing the raw value ("Avg .236") or make the rank badge a tooltip-enabled element that reveals the value on hover. A small numeric label below the badge would suffice.

**R3 — Populate or remove the Streak column** [HIGH]
**Problem:** The Streak column is entirely empty for all 12 teams. It occupies table width and creates a visual gap in every row.
**Fix:** Audit the bootstrap to determine why Yahoo's streak data is not being stored. If the data isn't reliably available, drop the column entirely. An empty column signals "the data is broken" to a savvy user and confuses a novice.

**R4 — Fix the Win% display format** [HIGH]
**Problem:** `.318` looks like a batting average. Under a "Win%" column heading it is confusing and non-standard.
**Fix:** Change to `31.8%` (multiply by 100, format as one decimal place + `%`). Update both the H2H Record table and the Season Projections projected-standings table. Also replace the inline `f"{win_pct:.3f}"` on line 361 with `format_stat` for structural compliance.

**R5 — Rewrite the Scenario Explorer inputs and caption** [HIGH]
**Problem:** The W/L/T inputs look like "how many H2H games do I win" but they mean "how many of the 12 scoring categories." The constraint `sum must = 12` is unexplained.
**Fix:** Relabel the section: "Category Scenario: If you win N of the 12 categories this week..." with explicit sub-labels under each input ("Categories Won", "Categories Lost", "Categories Tied"). Change the error message from "Sum = N (must be 12)" to "Total must equal 12 categories (the number of scoring categories in one matchup)."

**R6 — Explain Magic#, SOS, Power, Roster, Balance, Momentum, CI** [MEDIUM]
**Problem:** Six columns in Season Projections and seven columns in Team Strength Profiles are jargon abbreviations with no explanation.
**Fix:** Add `help=` parameter to column headers in `st.dataframe` where possible, or add an `st.caption()` below each table: "Magic# = wins needed to clinch a playoff spot. SOS = average opponent quality (0.5 = average schedule)." For Team Strength Profiles, add a brief one-sentence glossary in the expander header.

**R7 — Add a playoff cutoff line to the Category Standings grid** [MEDIUM]
**Problem:** The H2H Record table has a dashed red border between rank 4 and rank 5 to mark the playoff cut. The Category Standings grid has no such marker.
**Fix:** Add an equivalent styling rule to the Category Standings grid: inject a class that places the dashed border between the row belonging to the 4th-ranked team and the 5th-ranked team (ranked by H2H record, not category rank). This requires passing the H2H rank order into the grid row ordering.

**R8 — Differentiate or consolidate the two Playoff Odds surfaces** [MEDIUM]
**Problem:** Tab 2 (auto-run) and Tab 3 (manual sim button) both show playoff odds but use different engines and are not labeled as such. A novice will not know why the numbers might differ.
**Fix:** Either (a) consolidate them — show the detailed playoff probability panel inside Tab 2 and remove Tab 3, or (b) explicitly label the difference: "Quick estimate (Season Projections tab)" vs. "Full simulation with your roster projections (Playoff Odds tab — press Simulate Season)."

**R9 — Add actionable guidance to the banner and context cards** [MEDIUM]
**Problem:** The banner is a static ticker. The context cards show position data but suggest no next action. A novice at 10th place with weak AVG and SB sees the data but not what to do.
**Fix:** Use `render_reco_banner`'s `expanded_html` parameter to add one or two sentences of guidance: "You are 2 games back from the playoffs. Your weakest categories are SB (12th) and AVG (12th) — consider checking the Free Agents page for upgrades." This is exactly what the banner widget supports.

**R10 — Fix the current_week off-by-one in Season Projections** [MEDIUM]
**Problem:** `current_week` is computed as `int(weeks_played)` which gives 11 when the league is in Week 12. The simulation projects from the wrong starting week.
**Fix:** Change the formula to `current_week = max(1, round(weeks_played))` or use `ceil`, which gives 12 for 11.43. Alternatively, query the maximum week number from `league_matchup_cache` as the canonical current week.

**R11 — Add per-tab or per-panel data timestamp** [MEDIUM]
**Problem:** Standings data is 18 hours old; matchup data is 64 hours old. The Data Freshness card in the left rail covers global freshness but each tab's data age is not shown inline.
**Fix:** Below the H2H Record table header, add a small `st.caption("Standings as of [last_refresh timestamp]")`. Below the THIS WEEK context card, add "as of [matchup_updated_at]." This gives users confidence (or appropriate skepticism) about the numbers they see.

**R12 — Add playoff cutoff context card indicator** [MEDIUM]
**Problem:** The "PLAYOFF ODDS" context card says "Run projections to see playoff odds" with no navigation hint. The "2 GB from 4th" metric in the YOUR POSITION card is the most actionable number, but its significance (4 = playoff cutoff) is not labeled.
**Fix:** Add a sentence to the playoff label: "2 GB from 4th (last playoff spot)." Change the PLAYOFF ODDS card empty state to: "Open Season Projections tab — simulation runs automatically on first visit. Or click Simulate Season in the Playoff Odds tab."

**R13 (bonus) — Remove or populate the "Close battles" banner extension** [LOW]
**Problem:** The banner extension logic `close_battle_categories(cats)` generates "Close battles: [cat1], [cat2]." but the Week 12 matchup has only TIEs (SB 0-0, SV 0-0) and those are filtered by the "margin < 0.001" logic which doesn't detect the zero-zero ties as close. In practice, the "Close battles" clause rarely appears.
**Fix:** Include tied categories (user_val == opp_val) explicitly as "Close" since a tie by definition requires one more at-bat or strikeout to flip. Display as: "Tied: SB, SV — one play can flip these."

---

## 8. Severity-Tagged Issue List

- `[HIGH]` Category Standings grid has no legend for rank badges — novice cannot understand what the colored numbers 1–12 mean
- `[HIGH]` Streak column is always empty (never populated from Yahoo bootstrap) — dead column degrading table quality
- `[HIGH]` Win% column displays `.318` (no leading zero, no %)  — reads as a batting average, not a win percentage
- `[HIGH]` Scenario Explorer W/L/T inputs mislabeled — inputs represent category-wins within one matchup, not H2H record wins; constraint "sum = 12" unexplained
- `[MEDIUM]` Category Standings grid missing playoff cutoff line between rank 4 and rank 5 (H2H table has it)
- `[MEDIUM]` Duplicate playoff odds between Tab 2 (auto-sim heatbar) and Tab 3 (manual Simulate Season) — different engines, no differentiation
- `[MEDIUM]` Magic# column in Season Projections is unexplained jargon (no tooltip or footnote)
- `[MEDIUM]` SOS, Power, Roster, Balance, Injury, Momentum, CI columns in Team Strength Profiles are all jargon with no scale or explanation
- `[MEDIUM]` current_week off-by-one: formula yields 11 when league is in Week 12, biasing season projections
- `[MEDIUM]` No actionable guidance: banner uses static teaser only (expanded_html = ""), leaving "2 GB from playoffs" as a dead-end observation
- `[MEDIUM]` PLAYOFF ODDS context card empty state gives no navigation hint on how to populate it
- `[MEDIUM]` Category stats not shown alongside rank badges — seeing rank 12 in AVG provides no quantitative grounding
- `[MEDIUM]` Long team names truncate in compact table Team column ("HUMAN INT[...]", "BUBBA CRO[...]", "The Good The Vlad The...")
- `[MEDIUM]` FIG.01/FIG.02/FIG.03 within-page panel labels conflict with global FIG.06 page numbering system
- `[MEDIUM]` Matchup data in banner/context card is 64 hours stale with no "as of" timestamp
- `[LOW]` Inline `f"{win_pct:.3f}"` at line 361 does not use `format_stat` (structural deviation)
- `[LOW]` Rank badge color legend missing (green=playoff/sky=bubble/red=bottom-4) — requires prior knowledge
- `[LOW]` Playoff Odds tab simulation count column ("Simulations": 250) is confusing — "250 of 500 sims finished there" is not self-explanatory
- `[LOW]` Scenario Explorer "Re-simulate" button label ambiguous — should say "What are my playoff odds?"
- `[LOW]` Team Strength Profiles expander is collapsed by default with no summary preview of top/bottom teams
- `[LOW]` `Playoff%` in Projected Final Standings table formatted as "74%" (integer) while the Season Projections heatbar uses one decimal — inconsistent precision between the two playoff probability displays
- `[POLISH]` The THIS WEEK context card opponent name ("The Good The Vlad The Ugly") overflows the narrow left rail card width
- `[POLISH]` Ghost team "Twigs" appears in `league_standings` DB table but is correctly filtered from the UI — no action needed, but the stale standings row should be purged in the next bootstrap
- `[POLISH]` The banner "leading 7-3-2 in categories" doesn't distinguish which categories are won/lost/tied — a one-line breakdown ("Winning: R, HR, RBI, AVG, OBP, W, K") would be far more useful
- `[POLISH]` Season Projections auto-runs on first tab visit with no opt-out or skip button — heavy users may want to skip the 2-3 second spinner
