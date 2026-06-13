# Page 15 — Draft Simulator — Test-User Report

**Source file:** `pages/20_Draft_Simulator.py`
**Supporting engine:** `src/simulation.py` (DraftSimulator), `src/draft_engine.py`
(DraftRecommendationEngine), `src/pick_predictor.py`, `src/draft_state.py`
**Date audited:** 2026-06-13

---

## 1. Page Purpose & First Impression

The Draft Simulator lets the user run a mock snake draft: set league format,
pick a draft position, and simulate all 23 rounds against AI opponents. At each
user turn, a Monte Carlo / enhanced-engine recommendation panel shows the top
picks with scores, survival probability, and urgency heat bars.

**First 5 seconds for a novice:** The page opens to a settings panel labeled
"Draft Settings" beneath a hero header that reads
`PRESEASON · / FIG.20 — MOCK DRAFT` with eyebrow `PRESEASON`. The user sees
three number inputs (teams, rounds, draft position), two radio groups
(Simulation Depth, Engine Mode), and a primary "Start Mock Draft" button.
It's clear what to do — but the eyebrow `PRESEASON` is immediately confusing
because the league is in Week 12 of the active 2026 season. A novice will
wonder: "Is this page even useful right now? Did I open the wrong thing?"

The page is a preseason-only utility running live inside an in-season app, with
no explanation of why or how it's still useful, and the hero header actively
reinforces the confusion.

---

## 2. Methodology

- Full read of `pages/20_Draft_Simulator.py` (929 lines).
- Read of `src/simulation.py` (350 lines reviewed), `src/draft_engine.py`
  (230 lines reviewed), `src/draft_state.py` (270 lines reviewed),
  `src/pick_predictor.py` (80 lines reviewed).
- Read of `src/ui_shared.py` for CSS class definitions, POSITIONS constant,
  and `render_position_pills` / `render_player_select` signatures.
- Read-only DB queries: player pool (9,888 rows), ADP table (608 rows), ECR
  consensus (9,338 rows), projections (9,840 blended rows), league_draft_picks
  (276 rows = 12 teams × 23 rounds confirmed), league_settings.
- Reconstructed pick recommendations by running `value_all_players` + both
  `DraftSimulator.evaluate_candidates` and `DraftRecommendationEngine.recommend`
  on a 50-player ADP slice with 20–30 Monte Carlo sims (tiny to stay fast).
- Traced `auto_pick_opponents`, `render_recommendations`, `render_draft_summary`,
  `render_tabs`, and `render_user_roster` in full.

---

## 3. Feature & Control Inventory

| # | Control | Type | What it does | Tested? |
|---|---------|------|--------------|---------|
| 1 | "Number of teams" | `st.number_input` (6–20, default 12) | Sets team count | Yes (source) |
| 2 | "Rounds" | `st.number_input` (10–30, default 23) | Sets round count | Yes (source) |
| 3 | "Your draft position" | `st.number_input` (1–num_teams, default 1) | Sets user slot | Yes (source) |
| 4 | "Simulation Depth" radio | Radio: 50 / 100 / 200, default 50 | MC sim count | Yes (source) |
| 5 | "Engine Mode" radio | Radio: Quick / Standard / Full | Engine pipeline depth | Yes (both modes) |
| 6 | "Start Mock Draft" button | `st.button` (primary) | Builds pool, inits draft, starts AI picks | Yes (source) |
| 7 | "Reset Draft" button | `st.button` | Clears mock_* session state | Yes (source) |
| 8 | "Undo Last Pick" button | `st.button` | Removes last pick in pick_log | Yes (source + logic) |
| 9 | Recommendation top-3 cards | Custom HTML | Renders top 3 picks with score, bars, badges | Yes (reconstructed) |
| 10 | Score badge (orange, top-right of card) | HTML span | Shows `combined_score` as float (e.g. "33.89") | Yes |
| 11 | "Survival" bar + label | Heatbar + label | Probability player survives to next pick (0–1 displayed as %) | Yes |
| 12 | "Urgency" bar + label | Heatbar + label | Raw urgency float (2.74) vs bar clamped to 100% | Yes (BUG) |
| 13 | BUY / FAIR / AVOID pill | HTML pill badge | Buy/Fair/Avoid classification from enhanced engine | Yes |
| 14 | Injury risk text | Inline span | `{inj_prob:.0%} injury risk` shown if > 1% | Yes |
| 15 | Enhanced metric row (Need/Skill/Closer/Stream) | HTML spans | Secondary metrics for top rec only | Yes |
| 16 | Engine timing caption | `st.caption` | Shows "Engine: X.Xs (standard mode)" | Yes |
| 17 | "Search players..." text input | `st.text_input` | Filters available players list | Yes (source) |
| 18 | Top-5 player cards (draft pick cards) | HTML player-card | Shows headshot, name, position, pick_score | Yes |
| 19 | "Draft" buttons (×5) | `st.button` (primary, per card) | Drafts player, triggers AI opponents, reruns | Yes |
| 20 | "Available Players" tab | `st.tabs` | Sortable available-player table with position pills | Yes |
| 21 | Position pills (All/C/1B/2B/3B/SS/OF/SP/RP) | `render_position_pills` | Filters available player table | Yes (source) |
| 22 | "Sort by" selectbox | `st.selectbox` (Pick Score / ADP / ECR) | Sorts available table | Yes (source) |
| 23 | Available players table | `render_styled_table` | Shows rank, Player+headshot, Position, Team, ADP, Pick Score | Yes |
| 24 | "View player card" selectbox | `render_player_select` | Opens player card dialog | Yes (bug found) |
| 25 | "Draft Board" tab | `st.tabs` | Grid: rounds vs teams, shows who drafted whom | Yes (source) |
| 26 | "Pick Log" tab | `st.tabs` | Chronological log of all picks | Yes (source) |
| 27 | Draft Settings context card | HTML card | Shows teams/rounds/position/engine/sims | Yes |
| 28 | My Roster context card + table | HTML + `render_styled_table` | Shows user's current roster slots | Yes |
| 29 | Recent Picks feed | HTML divs | Shows last 10 picks in reverse order | Yes (source) |
| 30 | "On the Clock" command bar | `.cmd-bar` HTML | Shows "Round X · Pick Y" + current team | Yes |
| 31 | "Your Pick" indicator | `.your-turn` CSS span | Orange indicator when user is on clock | Yes |
| 32 | "Simulate Opponent Pick" button | `st.button` | Triggers AI picks when user is NOT on clock | Yes (DEAD CODE) |
| 33 | Draft grade card (A/B/C/D/F) | HTML card | Graded by SGP diff vs league average at end of draft | Yes |
| 34 | 12× `st.metric` cards (6 hitting + 6 pitching) | `st.metric` | Shows projected category totals at draft end | Yes |
| 35 | "### Final Roster" table | `render_styled_table` | Final slot assignments at draft end | Yes |
| 36 | "Start New Mock Draft" button | `st.button` (primary) | Clears mock state and reruns | Yes |
| 37 | `render_feedback_widget` | Feedback popover | Page-level feedback | Yes |
| 38 | Progress bar (pool load) | `st.progress` | Loading indicator during pool build | Yes (source) |
| 39 | Analytics badge | `render_analytics_badge` | Data quality badge from engine context | Yes (source) |

---

## 4. Feature-by-Feature Test Log With Real Outputs

### 4.1 Pre-Start Settings Panel

The settings panel shows three number inputs (default 12/23/1) and two radio
groups. All defaults match the real league (12 teams, 23 rounds). Draft position
defaults to 1 (first pick).

**Reconstructed first impression:** user fills in pos=6, clicks Start Mock Draft.

The Start button calls `refresh_pool()` first (force-rebuilds the valued pool),
then `init_mock_draft()`, then `auto_pick_opponents()`. Since pool build +
`value_all_players` takes roughly 1–2 seconds on the full 9,888-row pool, plus
any progress bar delays, the user waits before the draft begins.

### 4.2 AI Opponent Logic (`auto_pick_opponents`)

The AI picks from `available.nsmallest(min(15, len), "adp")` — top 15 by lowest
ADP — with linearly decreasing probability weights `[15, 14, ..., 1] / sum`.
This is a pure ADP-weighted random model. The more sophisticated
`DraftSimulator.opponent_pick_probability` (which models positional need and
historical preferences) is NOT called by `auto_pick_opponents` — only the
simple ADP-weighted path is used.

**Result:** AI opponents always draft near-ADP. They draft Ohtani, Judge, Soto,
Skubal, Skenes in rounds 1–5 before the user picks at position 6.

### 4.3 Recommendation Panel (Reconstructed — Standard Mode, Pick #6)

With 5 picks gone (Ohtani, Judge, Soto, Skubal, Skenes taken), top 5 at
position 6 (reconstructed from 50-player ADP slice, 30 sims, Standard mode):

| Rank | Player | Pos | Team | ADP | Combined Score | P(Survive) | Urgency | BFA |
|------|--------|-----|------|-----|---------------|-----------|---------|-----|
| #1 | Jackson Chourio | LF | (blank) | 19.4 | 20.99 | 54.5% | 0.000 | FAIR |
| #2 | Ronald Acuña Jr. | RF | ATL | 5.3 | 20.81 | 28.9% | 0.059 | FAIR |
| #3 | Wyatt Langford | LF | (blank) | 37.8 | 20.43 | 83.8% | 0.000 | FAIR |
| #4 | James Wood | RF | WSH | 37.0 | 19.90 | 82.9% | 0.000 | FAIR |
| #5 | Corbin Carroll | RF | AZ | 14.1 | 19.85 | 44.5% | 0.000 | FAIR |

**Issues found:**
- "Combined Score" = 20.99 for Chourio at pick 6. No unit, no range. A novice
  has no idea if this is good or not.
- Jackson Chourio and Wyatt Langford have empty `team` column in the DB
  (`team = ''`). The recommendation card shows `FA` (because `_team_txt`
  falls through to `"FA"` when `_team == ""`). These are active MLB players
  (Chourio: MIL, Langford: TEX) displayed as Free Agents — data quality bug.
- All recommendations are `BFA = FAIR`. In the full session the enhanced engine
  with injury/Statcast data would assign BUY/AVOID; with the small pool slice
  (50 players) these signals are too thin. In production (full 9,888-pool)
  some variety appears, but Ohtani still shows `FAIR` at 52% injury probability
  which seems like a calibration miss.

### 4.4 Recommendation Card Display

**Score badge:** Shows `combined_score` (e.g. `33.89` for Ohtani at pick 1).
This is `mc_mean_sgp + urgency * 0.4` — a raw float with no unit, no
explanation, no denominator, no range. A novice reads "33.89" and has no
reference point.

**Survival bar:** Label shows `{surv:.0%}` (e.g. "31%"), bar fills
`surv * 100.0`. Consistent and readable.

**Urgency bar/label:** MISMATCH. The label shows the raw urgency float
`{urg:.2f}` (e.g. "2.74"). The bar fills `min(100.0, urg * 100.0)` — so
urgency of 2.74 produces a bar value of 100% (clamped). The label says
"2.74" while the bar is pinned at full. These are on different scales.

**Enhanced metric row (top rec only):** Shows "Need: 1.20x", "Skill: -0.43"
(for Ohtani in standard mode). "Need" means category balance multiplier;
"Skill" means Statcast delta. Neither label is explained in-UI.

### 4.5 Player Pick Cards (Top 5, Quick-Draft Row)

Five horizontal card-buttons with headshot, name, position, and `pick_score`
(e.g. "15.61") and a primary "Draft" button beneath each. This is the main
pick mechanism. Works correctly in source.

**Issue:** `pick_score` and `combined_score` are different numbers shown in
different places. The recommendation card top badge shows `combined_score`
(33.89 for Ohtani); the quick-pick card row below shows `pick_score` (15.61).
A novice is tracking two different scores with no explanation of either.

### 4.6 Available Players Tab

Displays all un-drafted players with position pills, sort selectbox, and
`render_styled_table`. Player names have headshot `<img>` HTML injected inline
(via `_add_headshot`). The injected `disp_sorted["Player"]` column (now
containing `<img...>Aaron Judge`) is then passed directly to
`render_player_select(disp_sorted["Player"].tolist(), ...)` as selectbox option
labels. The selectbox will display raw HTML strings like
`<img src="https://img.mlb..." width="22"...>Aaron Judge` as plain text.

Positions filter uses `POSITIONS = ["All", "C", "1B", "2B", "3B", "SS", "OF",
"SP", "RP"]`. "Util" and "DH" are absent. A player like Yordan Alvarez
(positions="DH") and Util-eligible players are hidden when clicking positional
filters, but also not captured by any pill — they only appear under "All".

### 4.7 Draft Board Tab

Renders a round × team grid. AI team names are generic: "Team 1", "Team 2", …,
"Team 12" (one is "My Team"). The board is functional but stark — no team
branding, no ability to identify which "Team 5" corresponds to an opposing
manager.

### 4.8 Pick Log Tab

Chronological reverse log of all picks with pick number, round, pick-in-round,
team, player, and position. Clean, readable. No issues.

### 4.9 Command Bar (Active Draft)

When user is on the clock: `.your-turn` renders "Your Pick" in orange. The
`Round X · Pick Y` counter and "Mock Draft" label are in the `.cmd-bar`.
Visual presentation is polished.

When user is NOT on the clock: shows "Waiting for Team 3 to pick…" text and
a "Simulate Opponent Pick" button. However, this state is effectively
unreachable: `auto_pick_opponents()` runs immediately after every user pick
(and at draft start), so by the time the page renders, the user is always on
the clock. The "Waiting…" state and the "Simulate Opponent Pick" button are
dead code under normal flow.

### 4.10 Undo Last Pick

When the user presses "Undo Last Pick", `ds.undo_last_pick()` removes the
last entry in `pick_log` — which is the last AI pick that ran after the user's
most recent selection, not the user's own pick. Then `auto_pick_opponents()`
runs again and re-picks for the AI. The net effect is: the AI re-selects its
last pick (possibly differently due to randomness), while the user's own pick
remains. The button effectively undoes nothing useful for the user.

To undo the user's own pick the user would need to click Undo twice in a row
(once for each batch of AI picks since the user's last turn). There is no
indication of how many times to press or what happened.

### 4.11 Draft Summary (End of Draft)

After all 276 picks (12 × 23), the summary shows:
- Two rows of 6 `st.metric` cards for all 12 H2H categories.
- A grade card (A/B/C/D/F) with "Standings Gained Points" vs average.
- A "Final Roster" table.
- A "Start New Mock Draft" button.

**Metric label issue:** `col_whip.metric("Walks + Hits per Inning Pitched", ...)`
is an extremely long label that overflows or gets truncated in Streamlit's metric
widget. "col_l.metric("Losses", ...)` uses full name; other labels are fine.

**SGP calculation bypasses `SGPCalculator.totals_sgp`:** The summary computes
user SGP inline with a manual `for cat` loop. This replicates canonical SGP
logic rather than using `sgp_calc.totals_sgp(totals)`. It handles inverse stats
correctly (via `if cat in lc.inverse_stats: user_sgp -= val / denom`) but
bypasses rate-stat weighting — AVG/OBP/ERA/WHIP are added as raw ratios, not
volume-weighted. This matches the pattern that `tests/test_no_hardcoded_
categories_in_src.py` guards against in `src/` but the guard does not cover
`pages/`.

**Grade thresholds:** A=diff>8, B=diff>4, C=diff>-2, D=diff>-6, F=else. These
are arbitrary and uncalibrated. In a 23-round snake draft the typical SGP
surplus over the average team is modest (±5 points). A grade of "C" (diff in
[-2, +4]) is the likely outcome for any competent draft — the threshold to get
a "B" (diff > 4) is steep. No explanation of how grades are computed or what
"Standings Gained Points" means.

### 4.12 Pool Loading & Performance

`build_mock_pool` → `load_player_pool()` (9,888 rows, ~0.3s) →
`value_all_players()` (~0.8s estimated). Progress bar displays 3 stages. On
"Start Mock Draft" click, `refresh_pool()` force-rebuilds. Total cold start
estimated ~1–2s plus Streamlit overhead. Acceptable but not noted to the user.

---

## 5. Errors, Issues & Difficulties

### Functional Bugs

**B1. "Simulate Opponent Pick" button is dead code.** `auto_pick_opponents()`
always runs before the page renders, so `is_user` is always True at render
time. The "Waiting for X to pick…" branch never executes under normal flow.

**B2. "Undo Last Pick" undoes AI picks, not the user's pick.** The last
`pick_log` entry after any user pick is always an AI pick. Pressing Undo
removes one AI pick, then `auto_pick_opponents()` immediately re-picks. The
user's own pick is unaffected. The button is misleading.

**B3. Player name HTML injection breaks the "View player card" selectbox.**
In the Available Players tab, `_add_headshot` injects `<img>` HTML into the
`player_name` column of `disp_sorted`. That column is renamed "Player" then
passed to `render_player_select(disp_sorted["Player"].tolist(), ...)`. The
`st.selectbox` displays raw HTML strings as option text.

**B4. Missing team names for active MLB players.** Jackson Chourio (MIL),
Wyatt Langford (TEX), and other players show empty `team=''` in the DB
(level='AAA' or '40man' without a team assignment). They appear as "FA" in
recommendation cards despite being active MLB players.

**B5. Urgency label/bar scale mismatch.** Label shows raw urgency float
(e.g. "2.74"); bar fills `min(100%, urgency * 100%)` — clamped at 100%. A
value of 2.74 shows label "2.74" and a full bar, which is internally
inconsistent. The bar conveys "maximum urgency" while the label is
unintelligible.

**B6. SGP calc in draft summary bypasses `SGPCalculator.totals_sgp`.**
Inline loop in `render_draft_summary` does not apply rate-stat volume weighting
(AVG/OBP computed as raw fractions, not weighted by AB/PA). Grade may be
inaccurate, and the logic duplicates the canonical calculator.

### UX / Comprehension Issues

**U1. "PRESEASON" eyebrow on an in-season app.** The header `PRESEASON · /
FIG.20 — MOCK DRAFT` signals to the user that this page is for preseason prep.
No context explains why it's useful mid-season (2026 season, Week 12). A novice
visiting in June will feel this is orphaned.

**U2. `combined_score` is shown without unit or range.** The orange badge
shows "33.89" — a raw `mc_mean_sgp + urgency * 0.4` value. No scale, no
context. Is 33.89 good? What's the range? A novice cannot interpret it.

**U3. Two different scores shown for the same player.** Recommendation card
shows `combined_score` (33.89) in the top badge; quick-pick card row shows
`pick_score` (15.61). No explanation of the difference.

**U4. "Standings Gained Points" in the grade card is unexplained jargon.**
Even "SGP" in the expanded label isn't defined anywhere on the page.

**U5. Generic AI team names ("Team 1" … "Team 12").** The draft board looks
sparse and impersonal. No real opponent team names or any flavor.

**U6. "Util" and "DH" missing from position pills.** DH (Yordan Alvarez, Kyle
Schwarber) and Util players can only be found under "All" — no dedicated pill.

**U7. Draft grade thresholds are uncalibrated and unexplained.** A "C" grade
covers diff ∈ (-2, +4) which is the realistic outcome for most users. No tooltip
or explanation. Typical users will see a "C" and feel demotivated without
understanding the scale.

**U8. No in-season context or use-case framing.** The page offers zero
explanation of what someone gains from mock-drafting mid-season (e.g., "practice
for next year", "see how your current roster would have drafted", "test trade
values"). The use case is left entirely to the user's imagination.

**U9. "No player data found. Run python load_sample_data.py first" is a
developer error message.** If pool loading fails, this raw CLI command appears
in the UI. A real user has no idea what this means.

**U10. No explanation of Engine Mode options.** "Quick (< 1 second)", "Standard
(2-3 seconds)", "Full (5-10 seconds)" labels are accurate but do not explain
WHAT the modes add or when to use each. The `help=` tooltip for Engine Mode
reads "Quick: base analysis. Standard: Marcel regression + injury + Statcast.
Full: all contextual factors" — this is a fine start but is hidden and uses
technical jargon (Marcel, Statcast).

**U11. No explanation of Simulation Depth.** 50/100/200 sims — a novice does
not know what simulations are or how many to use.

**U12. FIG.20 label vs sidebar position 15.** The page is listed 15th in the
sidebar but the `fig` parameter says `FIG.20`, matching the file name
`20_Draft_Simulator.py`. There is inconsistency between sidebar rank and FIG
number compared to sibling pages.

---

## 6. UI/UX & Visual-Design Critique

**Hierarchy:** The setup panel is clean and unambiguous. The active-draft
layout (context rail left, main right) is the standard pattern and works well.
The command bar (cmd-bar) with the "Your Pick" call-out is a smart design.

**Recommendation cards:** The top-3 card layout with machined corners on the
#1 rec, headshot, BFA pill, injury risk span, and dual heat bars is visually
rich. However:
- The score badge (`33.89`, floating orange top-right) is the most prominent
  number on the card but is the least interpretable.
- The two heat bars (Survival vs Urgency) look identical but have incompatible
  scales (% vs float), making cross-card comparison meaningless.
- The metric row labels ("Need:", "Skill:", "Closer:", "Stream:") are
  too terse to convey meaning without documentation.

**Quick-pick card row:** Five horizontal `player-card` tiles work well
spatially. On smaller viewports the five-column row may be very tight. The
"Draft" primary buttons are visually correct (orange).

**Draft Board grid:** Functional but visually dated — plain text in a table,
no color-coding for positions or teams, no indication of which team is the
user.

**My Roster sidebar panel:** Shows slot names ("C", "1B", etc.) and player
names. Clean. But the "X of 23 slots filled" counter in the card body and
a separate `render_styled_table` outside the card creates a double-rendering
of the roster label — the `render_context_card` title says "My Roster" and
the inline counter says "X of 23 slots filled", but then `render_styled_table`
renders the actual table immediately below, outside the card. The visual
separation is awkward.

**Typography / numbers:** `pick_score` and `combined_score` are rendered with
`.2f` precision. That's fine. However `urgency` label shows `.2f` (e.g. "2.74")
which is a raw float on an implicit unbounded scale — not a meaningful display
number without context. The ADP is formatted `{adp:.0f}` (integer) in cards
and `{x:.0f}` in the table, which is correct.

**Empty states:** The empty state for no picks, no available players, and engine
failure all use `st.info("No recommendations available.")` / `st.info("No picks
yet.")` — plain Streamlit native, not the Combustion `render_empty_state`
component called for in the design system. Minor inconsistency.

**Data freshness:** The page makes no mention of data staleness. All ADP and
projection data are cached (1,219 minutes old per the shared-context note). The
header has no freshness badge, no caption, no warning.

**Deprecated component check:** The page does NOT use `st.components.v1.html`.
Clean.

**Off-palette hex:** None found. Palette is fully CSS-variable-based.

**Emoji:** None found. Correct.

---

## 7. High-Level Recommendations

### Rec 1 — Add an in-season framing section [IMPACT: HIGH]
**Problem:** The `PRESEASON` eyebrow and the complete absence of context makes
the page feel orphaned in June. Users don't know why this exists right now.
**Fix:** Change eyebrow from `PRESEASON` to `SCOUTING` (or `PREP`). Add a
brief context banner below the header: "Use the Mock Draft Simulator to prepare
for next year's draft, explore player values, or practice strategy. The AI
opponents use 2026 projection data." This one paragraph removes the confusion
entirely.

### Rec 2 — Fix Undo to undo the user's last pick [IMPACT: HIGH]
**Problem:** Undo removes the last AI pick (not the user's own). The user
cannot take back a bad pick.
**Fix:** Track the user's own most-recent pick number. On Undo, loop
`undo_last_pick()` until the removed entry's `team_index != user_team_index`,
then continue undoing until the user's pick is removed. Alternatively, on each
Undo click undo all picks back to and including the user's last pick, then
do NOT call `auto_pick_opponents()` — leave the user back on their prior turn.

### Rec 3 — Normalize the score badge to 0–100 [IMPACT: HIGH]
**Problem:** The orange badge shows `combined_score = 33.89` — an opaque
number on an unbounded scale. A novice cannot tell if this is great or terrible.
**Fix:** Normalize `combined_score` to a 0–100 relative scale within the
current available pool (100 = best available, 0 = worst). Show the normalized
score in the badge with a label ("Score 88 / 100"). Keep the raw values in
a tooltip for advanced users.

### Rec 4 — Fix the urgency bar/label scale mismatch [IMPACT: MEDIUM]
**Problem:** Urgency label shows raw float (2.74); bar is clamped to 100%. The
two representations are on different scales, making the bar misleading.
**Fix:** Either (a) show urgency as a 0–100 normalized value in both bar and
label (normalize by the max urgency in the candidate set), or (b) show the raw
float as a standalone metric without a bar. If using (a): `urg_norm = min(100,
urg / max_urgency * 100)`, display `{urg_norm:.0f}` in label and `urg_norm` in
bar.

### Rec 5 — Fix the "View player card" selectbox HTML injection [IMPACT: HIGH]
**Problem:** After injecting `<img>` HTML into `disp_sorted["Player"]`, those
strings (containing raw `<img>` tags) are passed to `render_player_select` as
option labels. The selectbox shows raw HTML text.
**Fix:** Maintain a parallel `player_name_clean` column (without HTML injection)
for passing to `render_player_select`. Only inject HTML into the display-table
column that is rendered via `render_styled_table` (which accepts HTML).

### Rec 6 — Fix missing team names for MLB players [IMPACT: MEDIUM]
**Problem:** Jackson Chourio (MIL), Wyatt Langford (TEX), and others have
`team=''` in the `players` table. They appear as "FA" in recommendation cards.
**Fix:** The bootstrap phase that enriches player data should not store empty
strings for `team` — it should retain the last known team or leave NULL so
the UI can show "Unknown" instead of "FA". A data migration or
`_build_player_pool` enrichment pass should backfill these from `level` or
roster context.

### Rec 7 — Replace dead "Simulate Opponent Pick" button with step-through mode [IMPACT: MEDIUM]
**Problem:** The "Waiting for X to pick…" branch never renders in normal flow.
**Fix:** Remove the button. Alternatively, add a "Step-by-Step" mode option
on the setup screen: when enabled, each AI pick is shown one-at-a-time with a
"Next Pick" button so the user can watch the draft unfold round by round. This
would also make the button reachable and purposeful.

### Rec 8 — Label the draft grade thresholds and calibrate them [IMPACT: MEDIUM]
**Problem:** A/B/C/D/F thresholds (>8/>4/>-2/>-6/else) are arbitrary, not
calibrated against real draft outcomes, and not explained in the UI.
**Fix:** Add a `st.expander("How is the grade calculated?")` below the grade
card explaining: "Standings Gained Points (SGP) measure how many standings
places your projected stats would move you in a 12-team league. Your grade
compares your draft haul to the simulated average. A 'B' means your picks
scored 4+ SGP above average." Also consider calibrating the thresholds
empirically from real draft data or extending the C/B spread.

### Rec 9 — Add "Util" and "DH" to position pills [IMPACT: LOW]
**Problem:** The POSITIONS pill list is `["All", "C", "1B", "2B", "3B", "SS",
"OF", "SP", "RP"]`. DH and Util slots exist in the real league (2 Util slots,
and DH-eligible players like Alvarez and Schwarber rank in the top 15 by ADP).
**Fix:** Add "DH" and "Util" to the POSITIONS constant in `ui_shared.py`, or
derive pills dynamically from the available pool's position list. The current
list silently hides these filter options.

### Rec 10 — Translate all technical metric labels for novices [IMPACT: MEDIUM]
**Problem:** Labels like "Need: 1.20x", "Skill: -0.43", "Urgency: 2.74",
"combined_score: 33.89", "pick_score: 15.61", "SGP", "Statcast delta" are
incomprehensible to a novice.
**Fix:** Add a collapsible "Glossary / How these scores work" section or inline
`help=` tooltips on each card element. Minimum viable version: rename
"combined_score" badge to "HEATER Score", add one-line explanations:
- "Survival: Probability this player is still available at your next pick."
- "Urgency: How much value you lose by waiting."
- "BUY / FAIR / AVOID: Based on ADP vs projected value."

### Rec 11 — Consolidate the My Roster panel layout [IMPACT: LOW]
**Problem:** "My Roster" renders as a `render_context_card` title with a
fill-count span inside, followed by a `render_styled_table` OUTSIDE the card
boundary. The visual result is a titled card with a subtitle, then an
uncontained table floating below it.
**Fix:** Either pass the table HTML inside the card body, or use
`render_compact_table` inline within the context card. The roster table
should live inside the card's visual boundary.

### Rec 12 — Fix SGP calculation in draft summary to use `SGPCalculator.totals_sgp` [IMPACT: LOW]
**Problem:** `render_draft_summary` manually iterates categories for SGP without
rate-stat volume weighting. This is a canonical SGP bypass.
**Fix:** Replace the inline loop with `sgp_calc.totals_sgp(totals)` from the
session-cached `SGPCalculator`. This ensures ERA/WHIP/AVG/OBP are weighted
by IP/AB, matching the rest of the engine.

---

## 8. Severity-Tagged Issue List

- `[HIGH]` Undo Last Pick does not undo the user's pick — it undoes an AI pick
  and immediately re-picks, making the button ineffective (B2).
- `[HIGH]` "View player card" selectbox shows raw HTML `<img>` tags as option
  text due to headshot injection into the Player column before passing to
  `render_player_select` (B3).
- `[HIGH]` PRESEASON eyebrow and zero in-season context — the page reads as
  abandoned preseason functionality during an active Week 12 season (U1, U8).
- `[HIGH]` `combined_score` badge (e.g. "33.89") is opaque to a novice: no
  unit, no range, no comparison point (U2, U3).
- `[MEDIUM]` Urgency bar and label are on incompatible scales: label shows raw
  float "2.74", bar is clamped percentage at 100% (B5).
- `[MEDIUM]` Jackson Chourio, Wyatt Langford, and other MLB players show as
  "FA" due to empty `team=''` in the DB (B4).
- `[MEDIUM]` "Simulate Opponent Pick" button is dead code / unreachable under
  normal draft flow (B1).
- `[MEDIUM]` SGP grade in draft summary uses inline loop, bypassing
  `SGPCalculator.totals_sgp` and omitting rate-stat volume weighting (B6).
- `[MEDIUM]` Draft grade thresholds (A>8/B>4) are uncalibrated and unexplained;
  most realistic drafts land "C"; user feels unfairly graded (U7).
- `[MEDIUM]` No Util or DH position pills — DH-eligible players (top-15 ADP)
  are invisible in position-filtered views (U6).
- `[MEDIUM]` "Standings Gained Points", "Urgency", "Survival", "Need", "Skill"
  metrics are undefined in-page; jargon wall for a novice (U4, U10, U11).
- `[MEDIUM]` No data-freshness indicator — projection data is ~1,219 min old;
  the page shows nothing about when ADP/projections were last updated (cross-
  cutting finding).
- `[LOW]` Generic AI team names ("Team 1" … "Team 12") in draft board; no
  personality, no real opponent names (U5).
- `[LOW]` `col_whip.metric("Walks + Hits per Inning Pitched", ...)` label is
  too long for Streamlit metric widget (truncates or overflows).
- `[LOW]` My Roster table rendered outside the context card boundary —
  float-below visual gap (U11).
- `[LOW]` FIG.20 label matches the file name (20th file) but not the sidebar
  position (15th page). FIG numbering convention is file-based, not nav-based.
- `[LOW]` "No player data found. Run python load_sample_data.py first" is a
  developer-facing error message exposed to end users (U9).
- `[LOW]` Empty states (no recommendations, no picks) use native `st.info()`
  instead of the Combustion `render_empty_state` component (design system
  inconsistency).
- `[POLISH]` The AI opponent model (`auto_pick_opponents`) ignores positional
  need — it uses linear ADP weighting only, not the more sophisticated
  `opponent_pick_probability` in `DraftSimulator`. AI feels robotic (always
  drafts top-ADP players).
- `[POLISH]` All recommendations in the 50-player test slice return
  `buy_fair_avoid = FAIR` because the BUY/AVOID signal requires richer Statcast
  and injury context that thin pool slices lack. In production (full pool) the
  distribution is better but Ohtani still shows FAIR at 52% injury probability
  — BUY/AVOID calibration worth reviewing.
