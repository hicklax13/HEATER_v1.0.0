# Page 13 — Leaders — Test-User Report

**Audit date:** 2026-06-13  
**Auditor persona:** Connor (novice fantasy-baseball manager, Team Hickey, FourzynBurn league)  
**Source file:** `pages/17_Leaders.py`  
**Supporting engines:** `src/leaders.py`, `src/trend_tracker.py`, `src/prospect_engine.py`

---

## 1. Page Purpose & First Impression

**What this page is for:** A research hub offering seven views of the player population: raw category leaders, an overall "value" score, breakout candidates, prospect rankings, and hot/cold/sell-high trend lists.

**First impression (5 seconds):** The page has a clean header ("Leaders." with the orange period) and a familiar Combustion layout. But then the left-panel shows two separate filter cards simultaneously — "Category Filter" (for the Category Leaders tab) and "Prospect Filters" (for the Prospects tab) — regardless of which tab is active. A novice looking at the sidebar sees a Category dropdown AND Position/Organization/ETA dropdowns, with no indication that each filter only works on one specific tab. This is immediately confusing: "Which filter goes with what I'm looking at?"

The seven-tab strip is wide. On a typical laptop screen all seven labels are probably visible, but the names are not all self-explanatory to a novice: "Category Value" and "Sell-High" need explanation, and the distinction between "Breakout Candidates" and "Hot List" is not obvious.

---

## 2. Methodology

**Source read:** Full read of `pages/17_Leaders.py` (1,106 lines), `src/leaders.py` (551 lines), `src/trend_tracker.py` (404 lines), `src/prospect_engine.py` (imports verified via `get_prospect_rankings`).

**DB queries (read-only):** Queried `season_stats` (2,645 rows for 2026), `players`, `league_rosters`, `prospect_rankings`, `statcast_archive`, `projections`, and `refresh_log` via the network-guarded pattern from the shared context. Computed category leaders, category value, breakout scores, and trend data programmatically from the live DB, matching the page's exact logic.

**Reconstruction labels:** Applied where Streamlit rendering (dialog, widget state) could not be invoked standalone.

---

## 3. Feature & Control Inventory

| Control | Type | What it does | Tested? |
|---------|------|--------------|---------|
| Category selectbox (ctx panel) | Selectbox | Picks one of 12 stats (R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP) for Category Leaders | Yes |
| Prospect Position filter (ctx panel) | Selectbox | Filters Prospects tab by position (All, SS, OF, SP, 2B, 3B, 1B, C, RP) | Yes |
| Prospect Organization filter (ctx panel) | Selectbox | Filters Prospects tab by MLB org | Yes |
| Prospect ETA Year filter (ctx panel) | Selectbox | Filters Prospects tab by ETA (All, 2025–2029+) | Yes |
| Refresh Data button (ctx panel, admin-gated) | Button | Forces re-fetch of prospect rankings via `refresh_prospect_rankings(force=True)` | Yes (logic traced) |
| "Category Leaders" tab | Tab | Branded leaderboard table (headshot, logo, heat bar) for selected category | Yes |
| "View player card" selectbox (tab 1) | Selectbox + dialog | Opens player dossier for a top-15 leader | Yes (reconstructed) |
| "Category Value" tab | Tab | Top 20 players by z-score composite across all 12 H2H categories | Yes |
| "View player card" selectbox (tab 2) | Selectbox + dialog | Opens player dossier for a category-value leader | Yes (reconstructed) |
| "Breakout Candidates" tab | Tab | Two sub-tables: Breakout Candidates (score ≥70) and Watch List (40–69) | Yes |
| "View player card" selectbox (tab 4) | Selectbox + dialog | Opens player dossier for a breakout candidate | Yes (reconstructed) |
| "Prospects" tab | Tab | Top 100 prospect table with scouting details + radar chart | Yes |
| Prospect detail selectbox (tab 3) | Selectbox | Picks a prospect; shows scouting grades + radar chart + report | Yes (reconstructed) |
| "Full Scouting Report" expander (tab 3) | Expander | Shows long-form scouting text for selected prospect | Yes (reconstructed) |
| "Hot List" tab | Tab | Top 15 HOT players by trend delta (vs projection) | Yes |
| "Cold List" tab | Tab | Top 15 COLD players by trend delta (vs projection) | Yes |
| "Sell-High" tab | Tab | Hot players with sustainability score < 0.45 | Yes |
| `render_player_select` (all three leader tabs) | Selectbox + st.dialog | Opens full player dossier card (reconstructed) | Yes (reconstructed) |
| `render_feedback_widget` | Popover | Feedback collection widget at bottom | Yes (code verified) |
| `page_timer_footer` | UI element | Render-time stamp at page bottom | Yes |

---

## 4. Feature-by-Feature Test Log with Real Outputs

### Tab 1 — Category Leaders

**Setup:** Category selectbox defaults to "R" (Runs). Season fraction = 44% (11.4 weeks elapsed). Min PA threshold = 21; Min IP = 8.8 IP.

**Real outputs from live DB (top 5 per category):**

| Category | #1 | #2 | #3 | #4 | #5 |
|----------|----|----|----|----|-----|
| HR | Kyle Schwarber 23 | Yordan Alvarez 22 | Matt Olson 19 | Byron Buxton 19 | Ben Rice 18 |
| R | James Wood 60 | Brice Turang 52 | Mike Trout 49 | Ben Rice 49 | Matt Olson 49 |
| RBI | Andy Pages 56 | CJ Abrams 51 | Matt Olson 50 | Nick Kurtz 49 | Christian Walker 48 |
| SB | José Ramírez 24 | Nasim Nuñez 24 | Bobby Witt Jr. 23 | Oneil Cruz 21 | Randy Arozarena 18 |
| AVG | Andrew Vaughn .360 | Nick Madrigal .346 | Otto Lopez .341 | Wade Meckler .340 | Jung Hoo Lee .335 |
| OBP | Max Schuemann .483 | Nick Madrigal .452 | Nick Kurtz .437 | Yordan Alvarez .432 | Jacob Gonzalez .429 |
| W | Gavin Williams 9 | Aaron Ashby 9 | Cristopher Sánchez 8 | Michael Soroka 8 | Davis Martin 8 |
| SV | Cade Smith 21 | Mason Miller 18 | Bryan Baker 18 | Riley O'Brien 17 | Jhoan Duran 16 |
| K | Jacob Misiorowski 116 | Cristopher Sánchez 113 | Dylan Cease 103 | Gavin Williams 99 | Cam Schlittler 89 |
| ERA | Aroldis Chapman 0.46 | Louis Varland 0.50 | Matt Brash 0.54 | Robert Suarez 0.61 | Shohei Ohtani 0.74 |
| WHIP | Dylan Lee 0.62 | Jacob Latz 0.62 | Rico Garcia 0.64 | John King 0.66 | Paul Sewald 0.73 |
| L | Brandan Bidois 0 | Orlando Ribalta 0 | Jayden Murray 0 | Cameron Foster 0 | Jared Jones 0 |

**Finding — L leaderboard is semantically broken:** The "L" (Losses) leaderboard sorts ascending (fewest losses = #1), so it shows the top 15 pitchers with ZERO losses. Three of the top 5 have between 9 and 18.3 IP — these are relievers and fringe starters with tiny sample sizes who just haven't been charged with a loss yet. This is not a useful leaderboard. The stat is inversely scored in H2H (lower = better), but the leaderboard framing of "here are the loss LEADERS" on a list of people with 0 losses is deeply unintuitive. A novice will think this tab is broken. The ERA/WHIP leaderboards have the same ascending-sort issue but are more interpretable because "lowest ERA" is a sensible headline. Losses at 0 is noise.

**Player card ("View player card" selectbox):** Appears below the leaderboard table. Selecting a name opens a `st.dialog` player dossier (reconstructed — same component used on My Team page). Label is generic "View player card" — no visual cue linking it to the rows above.

**Caption text:** Shows "Showing top 15 of 2,645 eligible players" — but 2,645 includes ALL season_stats rows, not just qualifying players for that category. A hitter with 50 SB but only 15 IP is not "eligible" for the ERA leaderboard. The denominator is misleading.

---

### Tab 2 — Category Value

**What it computes:** Z-score sum across all 12 H2H categories. Hitters scored on hitting cats only; pitchers on pitching only. Min PA=21, Min IP=8.8.

**Real outputs (top 10):**

| Rank | Player | Position | Category Value |
|------|--------|----------|----------------|
| 1 | James Wood | RF | 13.04 |
| 2 | Oneil Cruz | CF | 12.54 |
| 3 | Yordan Alvarez | DH | 12.17 |
| 4 | Nick Kurtz | 1B | 11.59 |
| 5 | Jordan Walker | RF | 11.33 |
| 6 | CJ Abrams | SS | 11.09 |
| 7 | Cade Smith | RP,P | 10.89 |
| 8 | Mason Miller | RP,P | 10.69 |
| 9 | José Ramírez | 3B | 10.60 |
| 10 | Randy Arozarena | LF | 10.59 |

**Finding — "Category Value" number is unexplained:** The value 13.04 is a z-score sum. A novice has no idea what this means. No scale anchor (is 10 good? is 20 possible? what's average?). The tab description says "Higher values mean the player contributes more to winning categories" — which is helpful, but the actual number shown in the table is never explained. No percentage-of-league, no tier label, no color-coded bar like the Category Leaders tab has.

**Finding — No heat bar:** The Category Leaders tab has a beautiful branded leaderboard with team logos, headshots, and a heat bar. The Category Value tab uses `render_compact_table` (plain text table) — visually inconsistent and much less polished.

**Finding — mlb_id column leaks into user-visible table:** The code builds `_cv_show = ["name","team","positions","category_value"]` but also appends `mlb_id` if present: `if "mlb_id" in cv_df.columns: _cv_show.append("mlb_id")`. The `mlb_id` (a raw integer like 694497) appears as a visible column for users with no label mapping. This is a data-engineering artifact visible in the UI.

---

### Tab 3 (var: `tab4`) — Breakout Candidates

**Finding — COMPLETE FAILURE: all breakout scores are 50.0 (reconstructed)**

Queried the live DB directly. The `statcast_archive` table has 501 rows but **every Statcast metric is NULL**:
- `barrel_pct`: 501 null / 501 total
- `xwoba`: 501 null / 501 total  
- `hard_hit_pct`: 501 null / 501 total
- `stuff_plus`: 0 non-null (column doesn't even exist in pool)

When `compute_breakout_score` is called with a pool where all Statcast columns are null/zero (they come back as 0.000 from the DB join), it falls into the fallback z-score path (`_fallback_hitter_score` / `_fallback_pitcher_score`) using hr/rbi/sb/avg/obp or k/era/whip/w/sv. But the pool's counting stat projection columns (hr, rbi, etc.) are also near-zero ROS projections for most players, causing the percentile rank to be ~50% for everyone.

**Actual output from live DB:** 0 players score ≥70. 1,441 players score in the 40–70 range. The top 10 are all pitchers at exactly 50.0. The page will show the fallback banner "No players currently score above 70. Showing top scorers instead." followed by 20 players all tied at 50.0 — with no indication that the Statcast data pipeline failed.

The "Breakout Candidates" tab is functionally dead because Statcast data is not populated in the DB. The page description says "Statcast barrel rate, xwOBA, hard hit percentage, Stuff+" — but none of these are available. The failure is silent to the user.

---

### Tab 4 (var: `tab3`) — Prospects

**Real data from `prospect_rankings` table:** 20 records, all fetched 2026-06-09. `fg_rank` is NULL for all 20.

**Top 20 prospects (reconstructed — as user would see):**

| Rank | Player | Org | Position | FV | ETA | Risk | Readiness |
|------|--------|-----|----------|----|----|------|-----------|
| (null) | Roki Sasaki | LAD | SP | 80 | 2025 | -- | 83.0 |
| (null) | Roman Anthony | BOS | OF | 70 | 2025 | -- | 76.3 |
| (null) | Travis Bazzana | CLE | 2B | 65 | 2026 | -- | 73.0 |
| (null) | Charlie Condon | COL | 3B | 65 | 2027 | -- | 68.0 |
| (null) | Jac Caglianone | KC | 1B/SP | 65 | 2027 | -- | 68.0 |
| (null) | Sebastian Walcott | TEX | SS | 65 | 2027 | -- | 68.0 |
| ... | | | | | | | |

**Finding — `fg_rank` is NULL for all 20 prospects:** The "Rank" column in the display table will be blank for every row, because `fg_rank IS NULL` in the DB. The display code prefers `fg_rank` over `rank` — so users see an empty rank column.

**Finding — ALL mlb_id values are NULL:** 20/20 prospects have `mlb_id = NULL`. The headshots in `render_compact_table(show_avatars=True)` will all fall back to the generic silhouette (since headshot URL requires mlb_id). The scouting detail panel's select box references `prospects_df["name"]` — this will work. But the radar chart and scouting tool grades table will show empty results because ALL scouting columns (hit_present, hit_future, game_present, etc.) are NULL across all 20 prospects. This means the "Scouting Tool Grades (20-80 Scale)" table will always render `render_empty_state("No scouting grades", ...)` for every prospect selected.

**Finding — `fg_risk` is NULL for all:** The Risk column in the display table will show "--" for all 20 prospects.

**Finding — Only 20 prospects total vs the filter claiming up to 100:** `get_prospect_rankings(top_n=100)` returns only 20. The organization filter dynamically builds its list from actual data — so only 13 of the 30 MLB teams are shown as options. A user filtering by "NYY" or "ATL" gets an empty result with no explanation of why most teams are missing.

**Finding — ETA 2025 prospects shown:** Roki Sasaki (ETA 2025) and Roman Anthony (ETA 2025) are #1 and #2 on the list. Their ETA has already passed — they are MLB players already. The prospect pipeline didn't filter or flag them as "already arrived." A novice doesn't know whether to see these as "prospects" or as a data error.

---

### Tab 5 — Hot List

**Real outputs (top 15 by trend_delta):**

| Player | Position | Trend Delta | Key Stats |
|--------|----------|-------------|-----------|
| Burch Smith | RP,P | **392.235** | .000 AVG / 0 HR |
| Cody Bolton | RP,P | **366.670** | .000 AVG / 0 HR |
| Spencer Miles | RP,P | **319.697** | .000 AVG / 0 HR |
| Peter Lambert | SP,P | **309.259** | .000 AVG / 0 HR |
| Joe Mantiply | RP,P | **302.189** | .000 AVG / 0 HR |
| Martín Pérez | SP,P | **276.622** | .000 AVG / 0 HR |
| Ryan Watson | RP,P | **267.239** | .000 AVG / 0 HR |
| Jesse Scholtens | RP,P | **265.548** | .000 AVG / 0 HR |
| Brandon Young | SP,P | **250.051** | .000 AVG / 0 HR |
| Joel Kuhnel | RP,P | **223.535** | .000 AVG / 0 HR |
| Keider Montero | SP,P | **222.288** | .000 AVG / 0 HR |
| Curtis Mead | 1B | 53.467 | .247 AVG / 0 HR |
| Ildemaro Vargas | 1B | 49.592 | .248 AVG / 0 HR |
| Brandon Valenzuela | C | 44.348 | .232 AVG / 0 HR |
| Leody Taveras | CF | 43.541 | .248 AVG / 0 HR |

**Finding — CRITICAL BUG: Hot List is garbage data.** The top 11 entries all show "Key Stats: .000 AVG / 0 HR" — the `_trend_key_stats` function reads `avg` and `hr` from the trended pool rows, but these players are pitchers (`is_hitter=0`), yet the code reads `avg` and `hr` as if they're hitters (`is_h = int(row.get("is_hitter", 1) or 1)` — the `or 1` default treats pitchers as hitters when `is_hitter` is 0, which is falsy in Python).

**Root cause of delta=392:** The pool for Burch Smith shows `ip=0.4` (a near-zero ROS projection). The k_rate_proj = 0/0.4 = 0 in the projection data (k=0). The actual k_rate = 20/17 = 1.176. Delta formula: `(1.176 - 0) / max(0, RATE_FLOOR=0.001) = 1.176 / 0.001 = 1176`. Averaged with ERA delta (~0.30) and WHIP delta (~-0.04): the k_rate term dominates completely and creates astronomically inflated deltas. This happens for ALL pitchers whose ROS projections have near-zero IP in the pool — the denominator collapses to RATE_FLOOR and the output is nonsense.

The "Hot List" is showing entirely broken data. The top 11 entries are fringe/minor-league pitchers with meaningless delta values in the hundreds. The first 4 hitters on the list (Curtis Mead etc.) have real data but appear below 11 garbage entries.

---

### Tab 6 — Cold List

**Real outputs (top 15 by trend_delta ascending):**

| Player | Position | Trend Delta | Key Stats |
|--------|----------|-------------|-----------|
| Walbert Urena | SP,P | **-935.341** | .000 AVG / 0 HR |
| Jose A. Ferrer | RP,P | **-854.444** | .000 AVG / 0 HR |
| Carson Williams | SS | -0.648 | .177 AVG / 9 HR |
| Austin Wynns | C | -0.640 | .198 AVG / 1 HR |
| Wilber Dotel | RP,P | -0.617 | .000 AVG / 0 HR |
| Jorge Polanco | 1B | -0.615 | .233 AVG / 12 HR |
| Ha-Seong Kim | SS | -0.539 | .222 AVG / 4 HR |
| Jear Encarnacion | RF | -0.533 | .203 AVG / 1 HR |
| Bo Naylor | C | -0.497 | .209 AVG / 8 HR |
| Christopher Morel | 1B | -0.486 | .219 AVG / 6 HR |
| Sam Haggerty | LF | -0.447 | .230 AVG / 1 HR |
| Jose Trevino | C | -0.416 | .220 AVG / 2 HR |
| Luis García | RP,P | -0.404 | .000 AVG / 0 HR |
| Angel Zerpa | RP,P | -0.397 | .000 AVG / 0 HR |
| Jared Triolo | SS | -0.360 | .240 AVG / 2 HR |

**Same ROOT CAUSE BUG:** Walbert Urena (-935) and Jose A. Ferrer (-854) are pitchers with near-zero ROS IP projections, generating wildly negative deltas. The Cold List has the same garbage-data problem as the Hot List. The top 2 entries are broken.

**Key Stats bug for pitchers persists here too:** Wilber Dotel, Luis García, Angel Zerpa show ".000 AVG / 0 HR" — they are pitchers misclassified as hitters by `_trend_key_stats`.

---

### Tab 7 — Sell-High

**Real output:** `detect_sell_high_candidates` returns **1 player**: Andrew Vaughn (sustainability=0.411, trend_delta=0.133).

Andrew Vaughn's stats: .360 AVG, Top AVG leader (real data) with sustainability below 0.45 threshold — this is a legitimate sell-high call. However, the page shows only 1 entry. A novice expecting a useful sell-high radar showing 5-10 trade targets sees a table with a single row. This is technically correct (only 1 player meets both criteria — HOT trend AND sustainability < 0.45) but feels empty and unhelpful.

**Finding — sell-high threshold is too tight:** `SELL_HIGH_SUSTAINABILITY_CAP = 0.45`. With `_HAS_SUSTAINABILITY = True` (waiver_wire module available) and 335 HOT players, only 1 qualifies. Either the sustainability scores are miscalibrated or the cap is too aggressive.

---

## 5. Errors, Issues & Difficulties

### BLOCKERS / HIGH

**[B1] Hot/Cold lists show garbage data with deltas of 392 to 935:** The top 10-11 entries on both Hot and Cold lists are fringe pitchers with near-zero ROS IP projections, causing `RATE_FLOOR` division that produces deltas 3–4 orders of magnitude higher than real player deltas. This appears to EVERY user as the first thing they see on the Hot List. The list is unusable.

**[B2] Hot/Cold "Key Stats" column is wrong for pitchers:** `_trend_key_stats` reads `avg` and `hr` for all players because `int(row.get("is_hitter", 1) or 1)` coerces `0` (pitcher) to `1` (hitter) via the `or 1` fallback. Every pitcher on the Hot/Cold list shows ".000 AVG / 0 HR" instead of ERA/K.

**[B3] Breakout Candidates tab is entirely non-functional:** All 9,888 statcast_archive values for barrel_pct, xwoba, hard_hit_pct, and stuff_plus are NULL (501 rows in the table, all null). The entire Breakout Candidates and Watch List are placeholders at 50.0. The tab description promises "Statcast barrel rate, xwOBA, hard hit percentage, Stuff+" but delivers none of them. Users see "No players currently score above 70. Showing top scorers instead." followed by 20 players all identical at score 50.0 — no signal, pure noise.

**[H4] Prospects tab: fg_rank is NULL for all 20 prospects:** The displayed "Rank" column is blank for every row. The page is supposed to show "top MLB prospects" but cannot rank them.

**[H5] Prospects tab: ALL scouting data (20-80 tool grades) is NULL:** Every scouting detail panel ("Scouting Tool Grades (20-80 Scale)") will show `render_empty_state("No scouting grades")`. The radar chart renders nothing. The prospect scouting detail section is entirely non-functional.

### MEDIUM

**[M1] L leaderboard shows 15 pitchers with 0 losses:** Ascending sort for an inverse stat means "fewest losses" shows pitchers who simply haven't been charged with a loss yet due to small sample sizes. This is accurate but misleading and unhelpful as a fantasy tool. Pitchers with 9-18 IP having 0 L is unsurprising, not a useful fantasy signal.

**[M2] Trend computation runs UNCACHED on every page load:** `load_player_pool()` (9,888 rows) + `compute_player_trends()` (iterates all 9,888 pool rows) + `detect_sell_high_candidates()` are called inside `with main:` with no `@st.cache_data` decorator — they run on every tab switch, every interaction, every rerun. `compute_player_trends` is O(N) on the pool. `detect_sell_high_candidates` runs `compute_player_trends` again internally. This is a significant performance tax on every page interaction.

**[M3] Both context-panel filter cards always visible regardless of active tab:** The Category selectbox (used only in tab 1) and the Prospect Filters (Position/Org/ETA, used only in the Prospects tab) are both always visible in the left panel. When a user is on the Hot List tab, both filter sections appear and do nothing. A novice will try all four controls and be confused when nothing happens.

**[M4] "Category Value" number has no scale anchor:** The value 13.04 (z-score sum) is shown with no explanation of the range, what "good" looks like, or how to interpret it. The tab description says "Higher values mean the player contributes more to winning categories" — but doesn't tell the user what the typical range is, what score an elite player gets, or what a replacement-level player gets.

**[M5] Sell-High shows only 1 player:** With 335 HOT-trending players and `SELL_HIGH_SUSTAINABILITY_CAP=0.45`, only Andrew Vaughn qualifies. The tab appears nearly empty. The threshold is calibrated so tightly that the feature is practically useless.

**[M6] ETA 2025 prospects shown as current prospects:** Roki Sasaki and Roman Anthony (ETA: 2025) are the #1 and #2 ranked prospects. Both have already arrived in the majors. The prospect list conflates active rookies with true prospects.

**[M7] mlb_id column visible in Category Value table:** Raw integer mlb_id (e.g. 694497) appears as a user-visible column in the Category Value tab. This is a DB key that means nothing to a user.

**[M8] Prospect data coverage is thin (only 20 records, 13 teams):** The Position filter offers 9 position options, but filtering by most positions returns 1-2 results or nothing. A user selecting "C" gets 2 prospects (Samuel Basallo, Ethan Salas). There is no indication how many total prospects should be in the system or why most MLB teams are missing.

**[M9] No position filter on Category Leaders tab:** The leaderboard for R (Runs) mixes hitters and designated hitters with no filter. A user looking for "top shortstops in HR" cannot filter. The Prospects tab has a position filter but Category Leaders does not.

### LOW / POLISH

**[L1] Tab variable naming is reversed (tab4 before tab3):** The code assigns `tab1, tab2, tab4, tab3, tab_hot, tab_cold, tab_sell = st.tabs(...)` — the variables are out of order (tab4 maps to "Breakout Candidates", tab3 maps to "Prospects"). This is a maintenance footgun but doesn't affect users.

**[L2] Season fraction caption shows denominator of 2,645 regardless of category:** "Showing top 15 of 2,645 eligible players" is shown for ERA/WHIP/W leaders — but only pitchers can earn ERA/WHIP, and only ~476 pitchers qualify. The 2,645 denominator is misleading for pitching categories.

**[L3] "Trend Delta" values displayed raw (like 392.235) with no cap or normalization in the UI:** If the computation bug is fixed, the Trend Delta column still displays raw floating-point values like `0.133` or `0.301`. A novice has no idea if +0.133 is good, great, or barely meaningful. No percentage-of-players indicator, no "HOT" badge coloring in a more visible column.

**[L4] `render_reco_banner` shows empty text with empty expanded_html:** The banner is called as `render_reco_banner("Category leaders and breakout detection", "", "leaders")` — with empty `expanded_html`, it renders as a static line with icon and text. No action suggested; purely decorative.

**[L5] "Refresh Data" button is admin-gated but appears in global context panel:** The button uses `viewer_can_write() and st.button(...)` — for non-admin users it evaluates to False and the button is silently absent. No explanation of why the control isn't there. A regular user might notice the "Prospect Filters" section has no way to refresh data.

**[L6] `fig` param inconsistency:** Header is `fig="FIG.17 — CATEGORY LEADERS"` but the page is the 13th seasonal page in the audit. The FIG.17 reflects the source filename `17_Leaders.py`. Other pages (e.g. Pitcher Streaming shows `FIG.4` without zero-padding per the shared context). This is self-consistent within the page but may not match the design system's intended numbering.

**[L7] Hot/Cold tables use `render_styled_table` (plain HTML) while the Category Leaders tab uses a richly branded `_build_leaderboard_html`:** Significant visual inconsistency — a user navigating from the Category Leaders tab (headshots, team logos, heat bars, team color accents) to the Hot List tab (plain `<table>` rows) will notice the quality drop.

**[L8] Player name encoding renders correctly in the DB but confirm in display:** Names like José Ramírez, Nasim Nuñez, Martín Pérez are stored as `Jos? Ram?rez` in cp1252 output but `José Ramírez` in UTF-8. The page uses `_ldr_html.escape(str(r.get("name","")))` which should preserve UTF-8 correctly in the browser. No display bug expected but worth verifying in the live browser.

**[L9] 7-tab strip is dense; tab labels are not self-explanatory:** "Category Value" and "Sell-High" need hover-text or sub-labels explaining what they mean. "Hot List" and "Breakout Candidates" overlap in user mental model — both signal "this player is good right now" without a clear distinction.

---

## 6. UI/UX & Visual Design Critique

**Tab count (7 is too many):** Seven tabs on a single page is near the upper limit of what's navigable before users start getting "tab fatigue." The tabs span two conceptually different domains: (1) league-wide statistical lookups (tabs 1-4) and (2) trend/momentum analysis (tabs 5-7, inherited from the deleted Trends page). These could be two separate pages or two sub-sections with distinct visual treatments. As it stands, a user opening the page for the first time has no mental model of why "Category Leaders" and "Sell-High" live together.

**Category Leaders tab (Tab 1) is the standout:** The branded leaderboard (`_build_leaderboard_html`) with team-color row backgrounds, MLB headshots, team logos, monospaced stat values in Archivo Display, and the orange heat bar is genuinely excellent — professional, high-tech, and information-dense. This is the design bar the other tabs should meet.

**Category Value tab (Tab 2) is markedly weaker:** Uses `render_compact_table` — a plain text table with no heat bar, no headshots, no color-coding. Given that both tabs involve ranking players, Tab 2 should use the same branded leaderboard component. Instead it looks like a different app.

**Breakout Candidates (Tab 3) has good copy but no data:** The description text is crisp and technical ("elite process metrics... signal a breakout before traditional stats catch up"). But the actual output is a table of 20 players all at 50.0 with no differentiation. The disconnect between the description and the output is severe.

**Hot/Cold/Sell-High (Tabs 5-7) have the worst visual quality:** `render_styled_table` outputs a `<table>` with navy headers and no other Combustion theming. No headshots, no team colors, no heat bars. The Trend Delta column shows raw floats (`392.235`, `0.133`) with no context. The colored HOT/COLD/NEUTRAL labels are a nice touch but surrounded by plain grey cells.

**Left panel overload:** Both Category Filter and Prospect Filters are always displayed in the left panel regardless of active tab. For a 7-tab page, the left panel should be tab-aware. Showing 4 irrelevant controls on the Hot List tab wastes space and creates confusion.

**Empty states are Combustion-compliant:** `render_empty_state` is used correctly in most places. The empty state on Prospects (No scouting grades) is appropriate but will fire for EVERY prospect due to data gaps.

**Number formatting on Category Leaders:** Counting stats are formatted as integers (✓). Rate stats use `format_stat` (✓). The breakout score rounds to 1 decimal (✓). The Trend Delta raw values have no formatting standard — `392.235` should never reach the user.

**Consistency with sibling pages:** The Category Leaders leaderboard is among the best-looking components in the app. The Hot/Cold tables are among the worst. The gap within a single page undermines the overall quality impression.

---

## 7. Recommendations (≥10, ordered by impact)

### Rec 1 — Fix the Hot/Cold delta explosion (CRITICAL)
**Problem:** Near-zero ROS IP projections collapse to `RATE_FLOOR=0.001` denominator, producing deltas of 200–900. The top 11 entries on the Hot List are broken.  
**Fix:** In `trend_tracker._compute_rate_delta`, add a minimum projected volume gate: if `ip_proj < 15` or `pa_proj < 50` (reasonable minimum sample thresholds), skip the player entirely (return 0.0 / NEUTRAL). Also normalize the delta output to a capped range [–3.0, +3.0] by using `np.clip` before classification so that even extreme outperformers don't pollute the display.

### Rec 2 — Fix pitcher key stats in Hot/Cold tables (CRITICAL)
**Problem:** `_trend_key_stats` reads `avg`/`hr` for all players; `is_hitter = int(row.get("is_hitter", 1) or 1)` coerces `0` (pitcher) to `1` via `or 1`.  
**Fix:** Replace the default: `is_h = 1 if pd.isna(row.get("is_hitter")) else int(row.get("is_hitter", 1))`. Then pitchers correctly display ERA/K instead of 0 AVG / 0 HR.

### Rec 3 — Fix the Breakout Candidates tab (make it show real data or hide it)
**Problem:** All Statcast values are NULL. The tab is functionally dead. Users see 20 players all at 50.0.  
**Fix (short-term):** If `pool["barrel_pct"].fillna(0).max() == 0`, show a data-quality warning: "Statcast data is not yet loaded. Run a full data refresh to enable breakout detection." Hide the tables behind this gate rather than showing 50.0 noise. **Fix (long-term):** Prioritize populating statcast_archive — the fallback z-score path (using hr/rbi/avg) at least produces differentiated scores; enable it when Statcast is missing but clearly label it "Using traditional stats (Statcast unavailable)."

### Rec 4 — Redesign the L leaderboard to be meaningful
**Problem:** "L Leaders" sorted ascending shows 15 pitchers with 0 losses. Not useful; appears broken to a novice.  
**Fix:** Rename the L leaderboard to "Fewest Losses (Qualifying)" and enforce a meaningful IP minimum (e.g., 30 IP rather than the current ~9 IP threshold). Show the top-quality pitchers keeping their loss count low, not fringe relievers with tiny samples. Alternatively, relabel the category as "Best Loss Rate" and sort by L/9 or L/GS.

### Rec 5 — Make the left panel tab-aware
**Problem:** Category Filter (for tab 1 only) and Prospect Filters (for tab 3 only) are always visible together regardless of the active tab.  
**Fix:** Use `st.session_state` to track the active tab. Show only the relevant filter controls. When Hot/Cold/Sell-High tabs are active, the left panel can show a date-range or position filter for those views. If tab-awareness is too complex in Streamlit's current architecture, collapse the unused filter sections by default (using an expander).

### Rec 6 — Bring Category Value tab up to Category Leaders visual standard
**Problem:** Tab 2 uses a plain `render_compact_table`; Tab 1 uses the rich branded `_build_leaderboard_html`. The quality gap within a single page is jarring.  
**Fix:** Adapt `_build_leaderboard_html` for Category Value, or create a `_build_value_leaderboard_html` that: (a) shows team headshots, (b) adds a heat bar normalized to the z-score range, (c) shows the top 3 contributing categories as chips below the player name. Remove the raw mlb_id column leak.

### Rec 7 — Cache the trend computations and breakout scores
**Problem:** `compute_player_trends` (9,888 rows), `load_player_pool`, `detect_sell_high_candidates`, and `compute_breakout_scores_batch` (O(N×M)) run uncached on every page load.  
**Fix:** Wrap the trend computation block in a `@st.cache_data(ttl=600)` function. At minimum, the `_trended` DataFrame should be cached — it's pure computation on already-cached data. Similarly, wrap `compute_breakout_scores_batch(stats_df_bo)` in a cached function call. This could cut page load times for re-renders from seconds to milliseconds.

### Rec 8 — Explain the "Category Value" number to users
**Problem:** The value 13.04 is an unexplained z-score sum. Novice users have no reference.  
**Fix:** Add a percentile rank column ("League %ile") alongside the Category Value number. Or bin into tiers: "Elite (>10)", "Strong (5–10)", "Average (0–5)", "Below (< 0)". Add a short explanatory caption: "Score reflects sum of per-category z-scores vs all qualifying players. League average = 0; positive scores indicate above-average fantasy value."

### Rec 9 — Reduce from 7 to 5 tabs (or reorganize into sections)
**Problem:** 7 tabs is cognitively overwhelming. The three trend tabs (Hot/Cold/Sell-High) were "folded in" from a deleted page and feel structurally mismatched with the four analysis tabs.  
**Fix option A:** Split the page into two: a "Leaders & Research" page (tabs 1-4) and a "Trends" page (tabs 5-7). The Trends tabs belong together conceptually and could be a richer standalone tool with date-range filtering.  
**Fix option B:** Keep 7 tabs but restructure the tab strip into two visual groups with a divider label: `ANALYSIS | Category Leaders · Category Value · Breakouts · Prospects` and `TRENDS | Hot · Cold · Sell-High`.

### Rec 10 — Fix prospect data quality (rank nulls, scouting nulls, stale ETAs)
**Problem:** `fg_rank` is NULL for all 20 prospects, all `mlb_id` are NULL, all scouting tool grades are NULL, and ETA 2025 players appear as current prospects.  
**Fixes:** (a) Use FG's `fg_fv` (Future Value) as the display rank if `fg_rank` is null, and sort the table by `fg_fv` descending. (b) Run `mlb_id` resolution for prospects by matching on name+team against the `players` table. (c) Filter out prospects with `fg_eta <= current_year` where the player has MLB at-bats (they "arrived"). (d) For scouting grades: if all are null, don't show the scouting detail section; show `render_empty_state("Scouting grades not yet available")`.

### Rec 11 — Add a "Rostered by / Available" filter to Category Leaders
**Problem:** The leaderboard shows a "Rostered" column but there's no way to filter to Free Agents only. A user wanting to know "who is available to pick up?" has to scan the table manually.  
**Fix:** Add a radio button / toggle: "All players | Free agents only | My team only." Filtering the leaderboard to Free Agents would make this the primary FA discovery tool and reduce the need to cross-reference the Free Agents page.

### Rec 12 — Cap Trend Delta display and add visual normalization
**Problem:** Even after fixing the computation bug, raw deltas like 0.301 or -0.648 are hard to interpret. The column header says "Trend Delta" with no units or scale.  
**Fix:** Display the delta as a percentage change ("outperforming projection by 33%") or as a normalized -100 to +100 score. Add a heat bar (same as Category Leaders) scaled to the observed range of non-broken deltas. Replace the column label with something a novice understands: "vs. Projection".

---

## 8. Severity-Tagged Issue List

- `[BLOCKER]` Hot/Cold lists dominated by garbage delta values (200–900×) from near-zero ROS IP projection denominators — top 10-11 entries are meaningless fringe pitchers
- `[BLOCKER]` Hot/Cold "Key Stats" column shows ".000 AVG / 0 HR" for all pitchers due to `or 1` coercion bug in `_trend_key_stats`
- `[BLOCKER]` Breakout Candidates tab entirely non-functional — all Statcast data is NULL, all 2,645+ players score exactly 50.0 with no differentiation
- `[HIGH]` Prospects tab: `fg_rank` NULL for all 20 records — the Rank column is blank for every row
- `[HIGH]` Prospects tab: all scouting tool grades (20-80 scale) are NULL — scouting detail section always shows empty state for every prospect selected
- `[HIGH]` `compute_player_trends` + `detect_sell_high_candidates` + `compute_breakout_scores_batch` run uncached on every page load/tab switch — significant performance penalty
- `[HIGH]` Both context-panel filter cards (Category Filter + Prospect Filters) always visible regardless of active tab — irrelevant controls for 5 of 7 tabs
- `[MEDIUM]` L leaderboard shows 15 pitchers with 0 losses as "leaders" — small sample size noise, not a useful fantasy signal
- `[MEDIUM]` Category Value tab uses plain compact table while Category Leaders uses rich branded leaderboard — jarring inconsistency within one page
- `[MEDIUM]` Category Value score (z-score sum) has no scale anchor, explanation, or percentile — unintelligible to novice users
- `[MEDIUM]` `mlb_id` column exposed in Category Value table as user-visible data
- `[MEDIUM]` Sell-High tab shows only 1 player (SELL_HIGH_SUSTAINABILITY_CAP=0.45 too tight) — practically empty feature
- `[MEDIUM]` ETA 2025 prospects (Roki Sasaki, Roman Anthony) appear as top-ranked "prospects" after their MLB arrival
- `[MEDIUM]` Only 20 prospect records in DB, 13 of 30 teams represented — prospect tab fundamentally underserved
- `[MEDIUM]` Caption "Showing top 15 of 2,645 eligible players" uses full stats row count as denominator regardless of category — misleading for pitching categories
- `[LOW]` Tab variable names are out of order in code (`tab4` before `tab3`) — maintenance confusion
- `[LOW]` "View player card" selectbox label is generic; no visual connection to the leaderboard rows above
- `[LOW]` Hot/Cold/Sell-High use `render_styled_table` (plain HTML) vs Category Leaders branded design — low-quality visual for 3 of 7 tabs
- `[LOW]` Trend Delta column shows raw floats with no formatting, no units, no capped range
- `[LOW]` `render_reco_banner` shows a static label only (no expanded content); purely decorative
- `[LOW]` 7-tab strip is dense; "Category Value" and "Sell-High" labels are not self-explanatory to a novice without hover text or sub-labels
- `[POLISH]` `FIG.17 — CATEGORY LEADERS` in eyebrow is the page file number (17th file) not the page sequence number (13th in the audit) — minor but could confuse in the context of a multi-page design spec
- `[POLISH]` No date-range or time filter on Hot/Cold/Sell-High tabs — no way to see "last 14 days" vs "season" trend data

---

*Report written: 2026-06-13 | Source: `pages/17_Leaders.py`, `src/leaders.py`, `src/trend_tracker.py`, `src/prospect_engine.py` | DB queried read-only via sanctioned `get_connection()` pattern*
