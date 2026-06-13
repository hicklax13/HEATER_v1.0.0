# Page 05 — Pitcher Streaming — Test-User Report

**Audit date:** 2026-06-13  
**Auditor persona:** Connor (novice fantasy manager, Team Hickey, FourzynBurn, 12-team H2H)  
**Source:** `pages/4_Pitcher_Streaming.py`  
**Supporting engines:** `src/optimizer/stream_analyzer.py`, `src/optimizer/streaming.py`, `src/two_start.py`, `src/optimizer/constants_registry.py`

---

## 1. Page Purpose & First Impression

**What it is:** A daily pitcher-streaming tool. For each game day, it lists available free-agent starting pitchers (and optionally your own), ranks them by a 0–100 "Stream Score," and helps you decide which pitcher to add for a single start. Secondary features: a deep-dive into one candidate (Matchup Microscope), a greedy 7-day streaming plan (Week Planner), and a historical accuracy check-back (Track Record).

**First impression (novice eye):** The page loads, shows a date picker, and immediately fires an expensive call to `build_stream_board`. What I see first is a 3-metric strip (Adds remaining, Weekly IP target, Cats in play) followed by a caption about the slate — `"Slate: 15 games, 28 probables posted, 20 matched to the player pool, N excluded as rostered"` — and then a dense 17-column table that a novice will not know how to read.

**Five-second reaction:** "What is a Stream Score? What does 56.10 mean — is that good or bad? What are Net SGP, Conf, xIP, xER, W%? What do I click? Why are some rows grey?" A novice has no entry point. The table is technically correct but completely opaque without an introduction.

---

## 2. Methodology

**Sources consulted:**
- Full read of `pages/4_Pitcher_Streaming.py` (772 lines)
- Full read of `src/optimizer/stream_analyzer.py` (950+ lines)
- `src/optimizer/streaming.py` (constants and compute functions)
- `src/two_start.py` (confidence tier, matchup score)
- `src/optimizer/constants_registry.py` (all `stream_*` entries)
- `src/ui_shared.py` (design system, `render_page_header`, `render_sortable_table`)
- `src/ip_tracker.py` (MIN_IP, WEEKLY_TARGET)
- Live DB queries (read-only via `get_connection()`) against the production SQLite
- Reconstructed score simulation using `score_stream_candidate` with real pool data

**Data state at audit time:**  
Data is cached and stale (~1,219 min old, Yahoo offline). All team_strength data for 30 teams is present in DB. Park factors are present for all 30 teams. Weather rows exist but are from prior dates (newest: 2026-06-01). PvB splits: 2,698 rows. Game logs: 58,411 rows. Team Hickey name in DB: `"? Team Hickey"` (emoji prefix included from Yahoo).

---

## 3. Feature & Control Inventory

| Control | Type | What it does | Tested? |
|---------|------|--------------|---------|
| Stream date selectbox | Dropdown | Picks date from today through +7 days, triggers board rebuild | Yes (traced) |
| Include my SPs checkbox | Checkbox | When checked, adds the user's own probable starters to the board | Yes (traced) |
| Show locked starts checkbox | Checkbox (default ON) | Keeps in-progress/final game rows visible (non-actionable, greyed) | Yes (traced) |
| Adds remaining metric | Metric tile | Shows `{ctx.adds_remaining_this_week} / 10` | Yes (reconstructed) |
| Weekly IP target metric | Metric tile | Shows `54 IP` (from `CONSTANTS_REGISTRY["stream_ip_target"]`) | Yes |
| Cats in play metric | Metric tile | Shows losing/tied categories from matchup urgency | Yes (traced) |
| Stream Score board | Sortable data table | 17-column sortable table of all streamable starters, sorted by Score descending | Yes (reconstructed) |
| "Why these scores" expanders | Expandable rows | Shows 6-component breakdown table + risk flags + expected line for top 5 actionable picks | Yes (traced) |
| Matchup impact section | Data table (under `---` divider) | Shows projected category win delta for top-5 streams vs this week's opponent | Yes (traced) |
| Suggested swap section | Data table | Shows recommended add/drop pairs from `recommend_streaming_moves` for today only | Yes (traced) |
| **Tab 2: Matchup Microscope** | Tab | Deep-dive on one selected pitcher | Yes (traced) |
| Candidate selectbox | Dropdown | Picks a pitcher from the board | Yes (traced) |
| 4 metric tiles (Score / Net SGP / Matchup / Status) | Metric row | Key stats for the selected pitcher | Yes |
| Component breakdown table | Data table | 6-row table of score components and weights | Yes |
| Opposing lineup section | Data table | Confirmed batting order (1-9) with handedness | Yes (traced) |
| Load game logs button | Button (lazy) | Fetches last 10 game logs and aggregates career/vs-opp summary | Yes (traced) |
| **Tab 3: Week Planner** | Tab | 7-day greedy streaming sequence | Yes (traced) |
| "Build 7-day plan" / "Rebuild plan" button | Button (session-cached) | Scores all FA starts for next 7 days, selects greedy sequence | Yes (traced) |
| 4 summary metric tiles | Metric row | Planned streams / IP / K / Net SGP | Yes (traced) |
| IP pacing caption | Text | Shows `ip_target` and `ip_floor` | Yes |
| Plan data table | Data table | One row per planned stream (Date / Add / Tm / Opp / GS / Conf / Score / Net SGP / xIP / xK / Risk) | Yes (traced) |
| Sequence caption | Text | `"06-13: Pitcher A → 06-15: Pitcher B → ..."` | Yes (traced) |
| **Tab 4: Track Record** | Tab | Historical replay of past dates | Yes (traced) |
| Replay date selectbox | Dropdown | Picks a date from 1–14 days ago | Yes (traced) |
| Run replay button | Button | Scores the historical board + fetches actual box scores | Yes (traced) |
| 4 summary metric tiles (Track Record) | Metric row | Starts graded / ERA+WHIP / QS rate / K vs expected | Yes (traced) |
| Actuals data table | Data table | Per-pitcher predicted vs actual line | Yes (traced) |
| My pitcher adds section | Data table | Transaction history filtered to pitcher adds for user's team | Yes (traced) |
| Page timer footer | Text | Render time | Yes |
| Feedback widget | Popover | `render_feedback_widget` call | Yes (traced) |

**Total controls inventoried: 32 distinct controls/sections**

---

## 4. Feature-by-Feature Test Log with Real Outputs

### Tab 1: Stream Finder

**Stream date dropdown:**  
- Default: `get_target_game_date()` — today if games not all final, else tomorrow.  
- Options: today through +7 days (8 entries), formatted as `"Fri Jun 13 (today)"`.  
- When a non-today date is selected, the board re-runs via `_fetch_schedule_cached(target_date)` (15-min TTL).  
- Selecting today uses `ctx.todays_schedule` from the optimizer context (no extra fetch).

**Metric strip (reconstructed):**
- `Adds remaining this week: 10 / 10` — confirmed from `WEEKLY_TRANSACTION_LIMIT=10`; `ctx.adds_remaining_this_week` would be 10 at week start.
- `Weekly IP target: 54 IP` — hardcoded from `CONSTANTS_REGISTRY["stream_ip_target"].value = 54`. Note this is also independently defined as `WEEKLY_TARGET = 53.846...` in `ip_tracker.py` — the two are not identical, though they display the same integer.
- `Cats in play:` — when `ctx.urgency_weights` is populated with matchup data, shows comma-joined losing/tied categories (e.g. `"K, ERA, WHIP"`). With Yahoo offline, shows `"no matchup data"`.

**Stream board census caption (real value from orchestrator):**  
`"Slate: 15 games, 28 probables posted, 20 matched to the player pool, N excluded as rostered."`

**Stream board column decoding (17 columns):**

| Column | Label | Data type | What it means |
|--------|-------|-----------|---------------|
| 1 | Pitcher | string | Player name |
| 2 | Tm | 3-char code | Team abbreviation |
| 3 | Opp | string | `"vs SEA"` or `"@ MIN"` |
| 4 | Status | string | `PROBABLE` / `LOCKED` / `FINAL` |
| 5 | Conf | string | `HIGH` (≤2 days) / `MED` (3–5 days) / `LOW` (6+ days) |
| 6 | GS | integer | Number of starts this week (1 or 2 for two-start weeks) |
| 7 | Score | float (X.X) | Stream Score 0–100 (sigmoid-blended composite; 50=neutral) |
| 8 | Net SGP | `+X.XX` string | Marginal standings points this start adds (can be negative) |
| 9 | Opp wRC+ | integer | Opponent offense quality: 100=avg, 110+=elite flag fires |
| 10 | Opp K% | float (XX.X) | Opponent team strikeout rate as a percent (e.g. 22.8%) |
| 11 | Park | float (X.XX) | Park factor for the venue: 1.00=neutral, ≥1.08=HITTER_PARK flag |
| 12 | xIP | float (X.X) | Expected innings pitched this start |
| 13 | xK | float (X.X) | Expected strikeouts |
| 14 | xER | float (X.X) | Expected earned runs |
| 15 | W% | integer | Win probability × 100, as an integer percent |
| 16 | Own% | integer | Yahoo ownership percentage (0=unowned, 100=universal) |
| 17 | Risk | string | Comma-joined risk flags (HIGH_WHIP, SHORT_LEASH, ELITE_OFFENSE, HITTER_PARK, WIND_OUT, LOW_CONFIDENCE) |

**Observed real rows (from orchestrator live capture):**

Row 1: `Zack Littell | WSH | vs SEA | PROBABLE | HIGH` (score and rest of columns cut off by table width)
Row 2: `Kyle Leahy | STL | @ MIN | PROBABLE | HIGH | 1 | 56.10 | -0.07 | 97 | 22.8 | 1.01`

**Simulation of Kyle Leahy @ MIN (reconstructed):**  
Using pool data (era=4.44, whip=1.48, k=37, ip=45.3) the reconstructed score engine gives:
- `stream_score = 38.7` (vs orchestrator's 56.10)
- `net_sgp = -1.195` (vs orchestrator's -0.07)
- Components: matchup=0.29, sgp=-1.00, form=0.00, lineup=0.00, env=-0.018, winprob=-0.057
- Risk flags: `[HIGH_WHIP]` (WHIP=1.48 > threshold 1.40)

Using blended projections (era=4.35, whip=1.36, k=90, ip=104.3):
- `stream_score = 37.7`, `net_sgp = -0.627`
- Risk flags: `[]` (WHIP=1.36 just under 1.40 threshold)

**Score discrepancy:** Our reconstruction cannot reproduce the live score of 56.10. Two explanations: (a) The live page may have had `ctx.recent_form` or `ctx.category_weights` data that pushed the form/lineup components positive; (b) The pool data in the live server session may differ slightly. The Net SGP=-0.07 is particularly hard to explain — that implies the projection system saw Leahy's ERA/WHIP slightly beating the team baseline. This is a signal that the score is **sensitive to data state** and the UI never communicates which projection snapshot is in use.

**Simulation of Zack Littell vs SEA (reconstructed from pool):**  
Pool data: era=4.78, whip=1.33, k=44, ip=65.9, percent_owned=18%
- `net_sgp` would be negative given ERA 4.78 is above league average
- `risk_flags` would likely be empty (WHIP=1.33 under 1.40)
- SEA team_strength: wRC+=99.7, K%=23.1

**"Why these scores" expanders (reconstructed):**  
For the top 5 actionable rows, each expander title is `"Pitcher (Tm) vs/@ Opp — score X.X"`. Inside: a 3-column dataframe with `Component | Value (-1 to +1) | Weight`. The 6 components are:

| Component | Weight | Meaning |
|-----------|--------|---------|
| matchup | 0.35 | Pitcher quality vs. opponent offense (0-10 raw mapped to -1 to +1) |
| sgp | 0.25 | Marginal standings points, clamped to [-1, 1] |
| form | 0.15 | L14 vs season ERA/K9 delta, neutral (0) if < 5 L14 IP |
| lineup | 0.10 | PvB lineup wOBA exposure, neutral if lineups unposted |
| env | 0.10 | Park × wind environment |
| winprob | 0.05 | Win probability adjusted for 50% neutral |

Followed by any risk flags as a caption and `"Expected line: X.X IP, X.X K, X.X ER, XX% win"`.

**Matchup Impact section (reconstructed):**  
With Yahoo offline and no `ctx.my_totals` / `ctx.opp_totals`, this section shows:  
`"No live weekly matchup in the cache (yds.get_matchup returned nothing) — if My Team also shows no matchup, the server's Yahoo sync/token needs attention; impact projection needs the live category scores."`

This is a graceful degradation message, but it fills a large section with explanation when the user's primary need is the board above.

**Suggested swap section:**  
With Yahoo offline and `ctx.user_roster_ids` likely empty, shows: `"Swap suggestions unavailable without live matchup context."` (falls through the `getattr(ctx, 'scope', '') != 'today'` guard).

### Tab 2: Matchup Microscope

**Candidate selectbox:** Formatted as `"Pitcher (Tm) vs/@ Opp"` — matches the board ordering.

**4 Metric tiles:**
- `Stream Score: X.X` (0-100)
- `Net SGP: +X.XX` (can be negative)
- `Matchup (0-10): X.XX` — raw sub-score from `compute_pitcher_matchup_score`; a different scale than the overall Stream Score. The label `"(0-10)"` is the only hint.
- `Status: PROBABLE / HIGH` — concatenated with ` / ` separator.

**Environment caption:** `"Venue WSH (park 0.98) | Opp wRC+ 97, K% 22.8"` plus risk flags if any.

**Component breakdown table:** 3 columns: `Component | Value (-1 to +1) | Weight`. Same data as the Why expander in Tab 1. No descriptions of what each component measures.

**Opposing lineup section:**  
- When confirmed lineup unavailable: `"No confirmed lineup posted for {team} yet (lineups typically post 1-2 hours before first pitch)."`  
- When available: DataFrame with Order (1–9) / Batter / Bats columns. A "Load game logs" button appears below.  
- `compute_lineup_exposure` can return a "Regressed lineup wOBA vs league average: +X.XXX" caption when PvB data is available.

**Recent starts section (Load game logs button):**  
Button-gated. When clicked:
- Fetches last 10 game log entries from statsapi gameLog endpoint (network-gated in audit)
- Displays table: `date | opponent | is_home | ip | k | er | bb | h`
- Two summary metric tiles: `"Last N starts: X.XX ERA, X.XX WHIP, XX K"` and `"vs OPP (M starts): X.XX ERA, X.XX WHIP, XX K"` (or `"no recent starts"` if filtered to zero rows)

**Staleness issue:** The game log button hits statsapi live (no TTL-based fallback). With the server's MLB API degraded (12s budget exceeded warning), this button can hang.

### Tab 3: Week Planner

**Build 7-day plan button:** Session-cached per day (key `stream_week_plan`). First load scores all FA starts for 7 days — estimated 10–60 seconds. The button label flips to "Rebuild plan" once built.

**Info state before building:** `"Click 'Build 7-day plan' to score the week's streamable starts (~10-60s)."` — good honest estimate.

**Summary metrics (reconstructed):**
- `Planned streams: N / M adds` — N greedy picks, M is `ctx.adds_remaining_this_week`
- `IP added: XX.X`
- `K added: XX`
- `Net SGP: +X.XX` (total across all planned streams)
- Caption: `"Weekly pacing: 54 IP target, 20 IP Yahoo forfeit floor."`

**Plan data table:** 11 columns: Date / Add / Tm / Opp / GS / Conf / Score / Net SGP / xIP / xK / Risk. Note: `xIP` and `xK` are per-week values (multiplied by `num_starts`), unlike the Stream Finder board where they are per-start. This inconsistency is not labeled.

**Sequence caption:** `"06-13: Pitcher A → 06-15: Pitcher B → ..."` — shows date and name only.

**Gorilla in the room:** The plan can include pitchers on non-Today dates, but only today's date has live schedule data in `ctx.todays_schedule`. Future dates re-fetch via statsapi at build time. If the fetch fails for any day, that day is silently skipped with no user notification.

### Tab 4: Track Record

**Replay date selectbox:** Last 14 calendar days, formatted `"Thu Jun 12"` etc.

**Run replay button (reconstructed with proxy caveat):**  
- Caption: `"Replay scores use CURRENT projections and form as a proxy — HEATER does not store point-in-time projections. Matchup facts (opponent, park, schedule) are historically exact; treat the form component as approximate."`
- Shows 4 metric tiles: `Top picks graded: N starts` / `Actual line: X.XX ERA / X.XX WHIP` / `Quality-start rate: XX%` / `K vs expected: +X.X`
- Actuals table: Pitcher / Tm / Opp / Score / xK / IP / K / ER / W / QS

**My pitcher adds this season:** Reads `yds.get_transactions()`, filters to user's team and pitcher adds. Real data found in DB: Team Hickey pitcher adds include Kyle Leahy, Kyle Harrison, Bailey Ober, Matt Brash, Caleb Thielbar, Shane Drohan, Erik Sabrowski, Andrew Alvarez, Eduardo Rodriguez, Andrew Kittredge (10 unique pitcher adds via transactions table). Displayed columns: `name | timestamp | type`.

---

## 5. Errors, Issues & Difficulties

### E-01 FIG number not zero-padded — CONFIRMED BUG
`pages/4_Pitcher_Streaming.py` line 184:
```python
render_page_header("Pitcher Streaming", eyebrow="DAILY", fig="FIG.4 — PITCHER STREAMING")
```
Every other page uses `FIG.NN` (two digits, zero-padded): FIG.01, FIG.02, FIG.03, FIG.05, FIG.06, FIG.10, FIG.11, FIG.12, FIG.14, FIG.16, FIG.17, FIG.19, FIG.20. Pitcher Streaming is the sole outlier with `FIG.4`. Minor but inconsistent — looks like an oversight from when the page was slotted in as a reused slot (4_Bullpen was removed; 4_Pitcher_Streaming replaced it).

### E-02 Stream board: 17 columns is overwhelming with zero explanation
A novice opening this table sees: `Pitcher | Tm | Opp | Status | Conf | GS | Score | Net SGP | Opp wRC+ | Opp K% | Park | xIP | xK | xER | W% | Own% | Risk`. There is no legend, tooltip, or introduction explaining what Score means, what Net SGP means, what the "x" prefix on xIP/xK/xER means (projected), or how to read "Opp wRC+". A smart beginner will know ERA and strikeouts but not wRC+, SGP, or confidence tiers.

### E-03 Net SGP displayed as negative in score that says "stream this pitcher"
Observed: `Kyle Leahy | Score 56.10 | Net SGP -0.07`. A score above 50 (neutral) signals a streamable pitcher but the Net SGP is negative — meaning adding this pitcher is projected to _hurt_ ERA/WHIP slightly. This contradiction is never explained. A novice will wonder: if Net SGP is negative, why does the tool recommend this start? (The answer is that matchup, form, and env components outweigh the SGP component at the 25%/35% weight split — but that is invisible without reading the why-expander.)

### E-04 "Matchup (0-10)" label in Microscope is ambiguous and scale-inconsistent
The overall Stream Score is 0–100. The "Matchup (0-10)" sub-score is on a different scale (0–10 raw from `compute_pitcher_matchup_score`). A user looking at `Stream Score: 56.1` and `Matchup (0-10): 6.5` has no mental model to reconcile these two numbers. The label should be something like "Matchup quality" with a tooltip explaining it feeds into the Stream Score.

### E-05 "Why these scores" expander shows component names as machine labels
The expander table shows component names: `matchup | sgp | form | lineup | env | winprob`. A novice sees `sgp`, `env`, `winprob` — none of these are self-explanatory. No descriptions are shown. Compare to Free Agents page where explanatory text accompanies composite scores.

### E-06 Score discrepancy between live observation and reconstructed simulation
The orchestrator saw `Kyle Leahy Score=56.10, Net SGP=-0.07`. Reconstruction using current pool data gives `Score=38.7, Net SGP=-1.195`. The 17-point gap (56 vs 39) suggests the score is sensitive to data freshness — `ctx.recent_form` (L14 form component), `ctx.category_weights` (urgency), and whether the pool's blended projections differ between server session and current DB state. The page shows no data freshness indicator specific to the stream board. A user cannot know if they are seeing scores based on 5-minute-old or 1,200-minute-old projections.

### E-07 Tab 1 has two separate HR dividers adding visual noise
Lines 334 and 413 both call `st.markdown("---")` creating hard rules that cut the tab into three sections (board, matchup impact, swap). These sections have no explicit headers — only `**bold markdown**` labels. The visual rhythm is: giant table → divider → bold text → another section → divider → another bold text. This looks like unstructured content, not a designed page.

### E-08 "Suggested swap (today)" is unavailable when Yahoo is offline
When `ctx.scope != "today"` or `ctx.user_roster_ids` is empty (both true when Yahoo offline), the section shows: `"Swap suggestions unavailable without live matchup context."` This is the correct degraded-state message, but it appears below two other sections that also show degraded-state messages (matchup impact = no data, swap = no data). Three consecutive "unavailable" captions in a single tab feels broken even if each is technically correct.

### E-09 Week Planner "Build 7-day plan" up-front cost is 10–60 seconds with no partial feedback
The button triggers scoring all FA starts across 7 days. A full-week slate has ~105 games × 2 sides = potentially 210 probable starters to score. The `st.spinner("Scoring all streamable starts in the next 7 days...")` is the only feedback. No progress bar, no per-day status, no streaming results. The user sees a blank spinner for potentially 60 seconds.

### E-10 Track Record replay uses current projections, not historical — caveat is buried
The proxy caveat is shown as a `st.caption` (small grey text) AFTER the replay is triggered. A novice who relies on the Track Record to validate their streaming choices may not read this fine print and will incorrectly attribute score accuracy to historical projections. The caveat belongs BEFORE the button, not after.

### E-11 "Load game logs" is a stateful button in a page that re-runs on every interaction
Streamlit's execution model means every widget interaction re-runs the page script. The "Load game logs" button inside Tab 2 will lose its state when the user switches tabs or changes the Candidate dropdown. The game logs will need to be re-fetched. There is no `st.session_state` caching for the game log result — unlike the swap cache and week plan cache which ARE session-cached.

### E-12 My pitcher adds section uses a fragile pitcher detection method
```python
pitcher_names = set(pool.loc[pool.get("is_hitter", 1) == 0, "player_name"].astype(str))
```
`pool.get("is_hitter", 1)` is not the correct pandas idiom — `pool.get()` on a DataFrame returns a Series. The intent is `pool["is_hitter"]` but `pool.get("is_hitter", 1)` actually works because pandas `DataFrame.get()` returns the column Series when found. However, if `is_hitter` is missing from the pool, the fallback `1` is a scalar, not a series, causing the entire filter to silently pass all rows (everyone would be classified as pitcher). This is a latent bug.

### E-13 "Cats in play" metric is misleading when Yahoo is offline
Shows `"no matchup data"` when offline. But a user expecting urgency-aware streaming advice ("I'm losing K and ERA this week — help me find a K strikeout guy") gets nothing here. The metric strip becomes useless precisely when the user most needs matchup context to make streaming decisions.

### E-14 Confidence tier labels are unexplained
`HIGH` means the start is ≤2 days out. `MED` means 3–5 days. `LOW` means 6+ days (probable may change). None of this is explained anywhere on the page. A novice will not know whether HIGH confidence means "the score is trustworthy" or "the pitcher is likely to pitch a lot."

### E-15 "GS" column header is cryptic
`GS` appears to mean "Games Started this week" (1 or 2 for two-start pitchers). But `GS` is a standard fantasy abbreviation for "Games Started" across the full season. Here it means "this start is their N-th start this week." A user might think it is the pitcher's season games-started total. Should be `Starts` or `2-start?`.

### E-16 xIP vs xK vs xER are projected per-start in Tab 1 but per-week in Tab 3 without any label change
The Stream Finder board columns are labeled `xIP | xK | xER` and represent single-start projections. The Week Planner table also uses `xIP | xK` but these are multiplied by `num_starts`. A two-start pitcher in the Week Planner shows doubled IP/K values — but the column header is still `xIP`. The inconsistency will confuse anyone comparing both tabs.

### E-17 No empty-state for when the board shows all LOCKED rows with show_locked=true
When all games on the selected date are in progress or final (e.g., reviewing yesterday's late games), the board shows all grey LOCKED/FINAL rows. The `actionable=False` logic is correct, but a novice will not understand why all rows are greyed and why there are no swap recommendations. There is no explanatory banner — just a dense grey table.

### E-18 Staleness: weather data is from prior dates in the DB
The DB's `game_day_weather` table has entries for WSH from 2026-04-08 through 2026-06-01. When the stream engine calculates `comp_env` (park × wind), it reads `ctx.weather` from `build_optimizer_context`. With Yahoo offline and data stale, the weather map is likely empty or stale. An empty weather map triggers `wind_term = 0.0` (neutral), meaning the WIND_OUT flag never fires even for Wrigley Field on a stiff south wind day. The page shows the `split_source` footnote for wRC+ but has no analogous note for stale weather data.

### E-19 The "Sequence" caption in Week Planner truncates dates
`"Sequence: 06-13: Pitcher A → 06-15: Pitcher B → ..."` — month and day only, no year. On a mobile display this is all one long string that wraps unpredictably. More critically, the dates are `game_date[5:]` (stripping the year prefix `YYYY-`) which works but gives an unusual `MM-DD` format inconsistent with the `Mon Jun 13` format used in the date selectbox.

### E-20 "My pitcher adds this season" table shows no meaningful stats
The table in Track Record tab shows `name | timestamp | type`. The `timestamp` is in raw datetime format (e.g., `2026-05-03T18:22:00`). There are no stats, no scores — just a transaction log. A user wants to know: "Was adding Kyle Leahy a good decision?" This table answers "I added Leahy on May 3" but not whether it was smart. The Track Record page brand promise is accountability, but this section does not deliver on it.

---

## 6. UI/UX & Visual Design Critique

### Hierarchy and information density
The page has **no onboarding sentence** telling a novice what to do. Compare the Matchup Planner which opens with clear category win-probability context. Pitcher Streaming opens with a controls row and immediately a 17-column table. For a novice trying to figure out "should I add Kyle Leahy or Zack Littell today?", the answer is buried inside columns 7–17.

### The Stream Score is not the first column
The primary output — the score — is column 7, not column 1. The user has to scan right past Tm / Opp / Status / Conf / GS before reaching the number that matters most. Tables should front-load the most important column (Score or Rank).

### No explicit "recommended action" call-to-action
After the board, a novice has no guidance about what to DO. The "Suggested swap" section (below two dividers) is the actionable output, but it is visually deprioritized. The board itself shows a sort-by-Score ordering, which implies "pick the top one", but the UI never says that.

### "Why these scores" expander UX
The expanders appear only for the top 5 actionable rows. Their titles duplicate information from the board (`"Pitcher (Tm) vs/@ Opp — score X.X"`). A novice will not know to click these. The component table inside is too technical — `"form: -0.02, env: -0.018, winprob: -0.036"` means nothing without documentation.

### Matchup impact table design
The table has columns: `Pitcher | vs | Exp cat wins | Matchup win% | Biggest movers`. The `Biggest movers` column shows `K +12% ERA -8%` style text — this is the most useful output on the whole page. But it is buried below a `---` divider with only `**Matchup impact (with vs without the stream)**` as a header.

### Color and visual emphasis
The board uses `render_sortable_table` which renders via `st.dataframe()` — Glide Data Grid with navy-on-white headers. Locked (non-actionable) rows are not visually distinguished from actionable rows because `st.dataframe` does not support per-row color styling. The user cannot see at a glance which rows are actionable. The `show_locked` checkbox changes this, but the default is ON with no warning that some rows are not actionable.

### Mobile density
17 columns in a fixed-height 520px `st.dataframe` on mobile will be a horizontal scroll nightmare. `Opp K%`, `Park`, `xIP`, `xK`, `xER`, `W%` are all secondary data that many expert users would accept as expandable — novices definitely would.

### Typography
Score values in the board are plain text in Glide Data Grid's Archivo 700 cells. The Combustion design system calls for IBM Plex Mono on numeric figures (`--font-mono`), but `st.dataframe` canvas rendering does not honor CSS font variables. Numbers in the table will render in Archivo (the gdg-base-font-style), not IBM Plex Mono. Minor inconsistency.

### No page-level data freshness badge
The shared `DataFreshnessTracker` / `data_freshness.py` module is not used on this page. With Yahoo offline and data 1,219 minutes stale, the user sees scores and projections from a 20-hour-old bootstrap with no warning. The Lineup Optimizer shows a data freshness badge — Pitcher Streaming should too.

### Tab 3 Week Planner — aggressive use of spinner vs lazy build
The `10-60s` estimate in the `st.info` is reasonable, but the page uses a full-page spinner (`st.spinner`) for the rebuild. A partial-progress implementation (score day 1, show it, score day 2...) is impossible with Streamlit's synchronous script model, but at minimum a progress bar with day-by-day status would be better UX than a blank spinner.

---

## 7. Recommendations (≥10, ordered by impact)

### R-01 Add a 1-sentence page intro and a "Top pick today" callout card
**Problem:** Novice has no entry point. The board is immediately intimidating.  
**Fix:** Insert a brief sentence above the controls: *"Pick a date to see today's best free-agent starters ranked by Stream Score (0–100). Start with the top green card."* Add a callout chip for the #1 actionable pick (pitcher name, opponent, Stream Score, ownership %) so the answer is visible without reading the table.

### R-02 Flip Stream Score to the first column (or add a separate rank column)
**Problem:** Score is column 7; users must scroll right before seeing the key output.  
**Fix:** Either move Score to column 1, or add a `Rank` column (1, 2, 3…) as column 1 and score as column 2. Alternatively, render the top 3 as visual "card" tiles above the full table so the answer is immediate.

### R-03 Rename columns to plain English with tooltips
**Problem:** `GS`, `Net SGP`, `Opp wRC+`, `Conf`, `xIP/xK/xER`, `W%`, `Own%` are expert jargon.  
**Fix:** Rename in the display DataFrame: `GS` → `Starts`, `Net SGP` → `SGP delta`, `Opp wRC+` → `Opp offense`, `Conf` → `Reliability`, `xIP/xK/xER` → `proj IP/K/ER`, `W%` → `Win%`. Use `st.dataframe(column_config=...)` tooltip entries for each. This is achievable by passing a rich `column_config` dict to `render_sortable_table`.

### R-04 Visually distinguish actionable vs locked rows
**Problem:** Non-actionable LOCKED/FINAL rows are not visually distinct from actionable rows in `st.dataframe`.  
**Fix:** Options: (a) default `show_locked=False` so locked rows are hidden, with a "Show completed games" toggle to reveal them. (b) Use `render_sortable_table`'s `hide_columns` to expose an `actionable` boolean column and configure it as a Streamlit checkbox-style column. (c) Render two separate tables — "Available" and "Locked" — with the latter collapsed by default in a `st.expander`.

### R-05 Fix FIG.4 → FIG.04 in the page header
**Problem:** `fig="FIG.4 — PITCHER STREAMING"` does not match the `FIG.NN` zero-padded format used by all 13 other pages.  
**Fix:** One-line change in `pages/4_Pitcher_Streaming.py` line 184: `fig="FIG.04 — PITCHER STREAMING"`.

### R-06 Surface the most actionable output — Matchup Impact — more prominently
**Problem:** The "Matchup impact" section is the most useful table (shows concrete `K +12% ERA -8%` category deltas) but is buried below a divider after the dense scoring board.  
**Fix:** When matchup data is available, promote the impact table to a highlighted panel directly below the metric strip, before the board. When offline, replace with a clear single-line status (`"No matchup data — Yahoo offline"`) rather than a multi-sentence explanation.

### R-07 Add data-freshness badge to the page (consistent with Lineup Optimizer)
**Problem:** Scores are based on projection data that may be 1,200+ minutes old. No staleness indicator on this page.  
**Fix:** Import and call `DataFreshnessTracker.render_badge()` (or `ui_analytics_badge.render_freshness_badge`) in the metric strip. At minimum, read `ctx.scope` and show a timestamp of the last successful bootstrap. Consistent with the `DataFreshnessTracker` pattern already used in the Lineup Optimizer.

### R-08 Add tooltip or legend for the Risk flags column
**Problem:** `HIGH_WHIP`, `ELITE_OFFENSE`, `SHORT_LEASH` are visible in the table but never explained.  
**Fix:** Add a short legend below the board (or use `st.dataframe(column_config={"Risk": st.column_config.TextColumn(help="...")})`). Example: *"Risk flags: HIGH_WHIP = career WHIP > 1.40; SHORT_LEASH = projected IP/start < 4.5; ELITE_OFFENSE = opponent wRC+ ≥ 110; HITTER_PARK = park factor ≥ 1.08; WIND_OUT = outbound wind ≥ 12 mph; LOW_CONFIDENCE = probable may change (5+ days out)."*

### R-09 Cache the game-log fetch in Tab 2 to survive tab switches
**Problem:** The "Load game logs" button loses its result on any widget interaction because the result is not persisted to `st.session_state`.  
**Fix:** Add `if (cached := st.session_state.get(f"game_log_{mlb_id}")) is not None: hist = cached else: hist = _history_cached(...); st.session_state[f"game_log_{mlb_id}"] = hist`. Same pattern as the swap cache and week-plan cache already in the page.

### R-10 Week Planner: explain the xIP/xK difference from Tab 1
**Problem:** `xIP` / `xK` in Week Planner = per-week values (num_starts × per-start projection). In Tab 1, same labels = per-start. This inconsistency will confuse users comparing the tabs.  
**Fix:** Either rename to `Wk IP` / `Wk K` in the Week Planner table, or add a caption: *"xIP/xK are weekly totals (single start × 2 for two-start pitchers)."*

### R-11 Track Record: proxy-caveat must appear BEFORE the "Run replay" button
**Problem:** The warning that replay scores use current projections is shown AFTER the user clicks "Run replay" — too late to calibrate expectations.  
**Fix:** Move the caveat text to a `st.caption` positioned before the button: *"Note: replay scores are approximate — HEATER uses current projections, not historical. Schedule facts are exact; form/projection inputs are a proxy."*

### R-12 Reduce mobile column count — drop Park, Own%, W% behind an expander
**Problem:** 17 columns on a phone is unusable.  
**Fix:** Use a responsive layout: on narrow viewports (or always), show a "compact" 8-column board (Pitcher / Tm / Opp / Status / Score / Net SGP / Risk / xIP) and put the remaining 9 columns behind a "Show all stats" toggle.

### R-13 "Cats in play" metric: when offline, show last-known categories instead of "no matchup data"
**Problem:** "no matchup data" is useless when Yahoo is offline but the user still wants urgency context.  
**Fix:** Fall back to reading `ctx.my_totals` and computing which categories the user is losing by rank vs. the cached standings. This requires reading `league_standings` or `standings_utils.get_team_totals()` from cache — both should be available even when Yahoo is offline.

---

## 8. Severity-Tagged Issue List

- `[POLISH]` **E-01** FIG.4 → FIG.04: header eyebrow missing zero-pad. One-line fix.
- `[HIGH]` **E-02** Stream board 17 columns with no legend, tooltips, or intro. Primary novice confusion point.
- `[HIGH]` **E-03** Negative Net SGP (-0.07) on a pitcher scored above neutral (56.10) with no explanation. Contradicts the user's expectation that a "good" pick has positive SGP.
- `[MEDIUM]` **E-04** "Matchup (0-10)" metric in Microscope is on a different scale than Stream Score (0-100). Label ambiguity.
- `[MEDIUM]` **E-05** Component table shows machine-readable labels (sgp, env, winprob) without descriptions. Expert only.
- `[MEDIUM]` **E-06** Score discrepancy between live observation (56.10) and reconstructed simulation (38.7) exposes sensitivity to data staleness, with no UI communication.
- `[MEDIUM]` **E-07** Two unmarked `---` dividers fragment Tab 1 into three unlabeled sections. Visual structure lost.
- `[MEDIUM]` **E-08** Three consecutive "unavailable" captions (matchup impact, swap) when Yahoo offline. Looks broken.
- `[MEDIUM]` **E-09** Week Planner build is 10–60s with no partial feedback. Blank spinner only.
- `[HIGH]` **E-10** Track Record proxy-caveat shown AFTER clicking "Run replay" — wrong moment; users have already formed incorrect expectations.
- `[MEDIUM]` **E-11** "Load game logs" button result not session-cached; lost on any widget interaction.
- `[LOW]` **E-12** `pool.get("is_hitter", 1)` fragile pattern in pitcher-add filter — silent all-pass if column missing.
- `[MEDIUM]` **E-13** "Cats in play" shows "no matchup data" when offline; no cache fallback to standings-derived urgency.
- `[MEDIUM]` **E-14** Confidence tier labels (HIGH/MED/LOW) never explained on the page.
- `[MEDIUM]` **E-15** `GS` column header misread as season-games-started (standard fantasy stat) vs this-week starts.
- `[MEDIUM]` **E-16** `xIP`/`xK` are per-start in Tab 1 but per-week in Tab 3 with identical labels.
- `[LOW]` **E-17** All-locked board (e.g., reviewing a past date's early games) shows grey rows with no explanatory banner.
- `[LOW]` **E-18** Weather data stale (newest: 2026-06-01); WIND_OUT flag silently never fires. No UI warning.
- `[LOW]` **E-19** Week Planner sequence dates formatted as `MM-DD` inconsistent with selectbox `Mon Jun 13` format.
- `[LOW]` **E-20** "My pitcher adds" table shows timestamp+type only; no stat comparison to judge add quality.
- `[MEDIUM]` **E-21** (General) No `st.components.v1.html` usage found on this page — CLEAR. No deprecated HTML component risk here.
- `[HIGH]` **E-22** No page-level data freshness indicator. With Yahoo offline and data 1,219 minutes old, user cannot know if stream scores are from this morning or yesterday afternoon.

---

## 9. Summary of Stream Score Weights (reference)

| Component | Weight | Source |
|-----------|--------|--------|
| Matchup quality | 0.35 | `compute_pitcher_matchup_score` (0-10 scale mapped to -1 to +1) |
| Marginal SGP | 0.25 | `compute_streaming_value` (inverse-stat-aware) |
| Recent form (L14) | 0.15 | ERA/K9 delta vs season baseline; neutral if < 5 L14 IP |
| Lineup exposure (PvB) | 0.10 | Regressed lineup wOBA; neutral if lineups not posted |
| Environment (park × wind) | 0.10 | `_env_component(park_factor, weather, venue)` |
| Win probability | 0.05 | `compute_bayesian_stream_score` |
| **Total** | **1.00** | Convex blend, verified by registry test |

Risk flags and their thresholds (from `CONSTANTS_REGISTRY`):
- `HIGH_WHIP`: career WHIP > 1.40
- `SHORT_LEASH`: projected IP/start < 4.5
- `ELITE_OFFENSE`: opponent wRC+ ≥ 110
- `HITTER_PARK`: park factor ≥ 1.08
- `WIND_OUT`: outbound wind ≥ 12 mph (open-air only)
- `LOW_CONFIDENCE`: start 6+ days out
