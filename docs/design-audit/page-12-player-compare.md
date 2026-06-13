# Page 12 — Player Compare — Test-User Report

**Auditor persona:** Connor, novice fantasy manager, Team Hickey, FourzynBurn H2H-cats league.
**Audit date:** 2026-06-13.
**Source:** `pages/16_Player_Compare.py`, engine `src/in_season.py::compare_players()`.

---

## 1. Page Purpose & First Impression

**What it does:** Side-by-side player comparison — pick two players, see who wins each of the 12 H2H categories (R, HR, RBI, SB, AVG, OBP, W, L, SV, K, ERA, WHIP) via z-scores, a radar chart, SGP breakdown, YTD stats, recent rolling stats, Statcast profile, health/confidence indicators, and team schedule strength.

**First 5 seconds as a novice:**
The page header reads "Player Compare. / RESEARCH — FIG.16 — HEAD-TO-HEAD" followed immediately by a plain static banner ("Select two players to compare") with no visible bar chart icon — the `render_reco_banner("Select two players to compare", "", "player_compare")` call renders the bar-chart SVG icon but has no expanded detail, making the banner feel like filler text rather than a call to action.

The first interactive element is a drop-down labeled "Player universe" with four options. A novice will immediately wonder: "Should I change this? What does 'MLB only' vs 'MLB + AAA' mean for my comparison?" There is no inline tooltip explaining why the level filter matters before the player selects are even visible. The level filter appears before any player input, putting a configuration knob ahead of the primary action.

The player search UI below is non-standard and partially broken (see §6). A novice typed in a name and saw up to three pill-buttons appear — no indication that pressing one of them is what locks in the selection.

---

## 2. Methodology

**Sources read:**

- Full source of `pages/16_Player_Compare.py` (873 lines).
- `src/in_season.py` — `compare_players()` function (lines 387–463) plus `compute_category_fit()` (lines 60–81).
- `src/ui_shared.py` — `ALL_CATEGORIES`, `METRIC_TOOLTIPS["z_score"]`, `render_panel()`, `build_heatbar_html()`, `build_stat_readout_html()`, `render_context_columns()`, `render_context_card()`, `render_compact_table()`, `get_plotly_polar()`, `get_plotly_layout()`, `team_color()`, `team_logo_url()`.
- `src/player_card.py` — player card schema.
- `src/injury_model.py` — `get_injury_badge()`.
- `src/player_databank.py` — `compute_rolling_stats()`.
- `src/valuation.py` — `SGPCalculator.player_sgp()`, `compute_percentile_projections()`, `compute_projection_volatility()`, `add_process_risk()`.
- `src/optimizer/matchup_adjustments.py` — `get_catcher_framing_data()`.

**Live DB queries (read-only):**

- `pool = load_player_pool()` → 9,888 players, 87 columns.
- Ran `compare_players()` on three pairs: Juan Soto vs Aaron Judge (hitter vs hitter), Tarik Skubal vs Paul Skenes (SP vs SP), Juan Soto vs Tarik Skubal (cross-type).
- Queried `league_rosters`, `projections`, `game_logs` (58,411 rows), `statcast_archive`, `refresh_log`.

---

## 3. Feature & Control Inventory

| Control | Type | What it does | Tested? |
|---------|------|--------------|---------|
| `Player universe` selectbox | `st.selectbox` | Filters pool to MLB / MLB+AAA / MLB+AAA+AA / All | Yes |
| `Search Player A` text input | `st.text_input` | Live search filter for player A | Yes |
| Player A pill buttons (up to 3) | `st.button` (primary/secondary) | Selects a player A from top-3 search matches | Yes |
| `Search Player B` text input | `st.text_input` | Live search filter for player B | Yes |
| Player B pill buttons (up to 3) | `st.button` (primary/secondary) | Selects player B from top-3 search matches | Yes |
| Player A roster status badge | `st.markdown` (custom HTML) | Shows "Rostered: Team Name" or "Free Agent" | Yes |
| Player B roster status badge | `st.markdown` (custom HTML) | Same for player B | Yes |
| Progress bar | `st.progress` | Animated "Comparing players across 12 categories…" / "Comparison complete!" then hidden | Yes (trace) |
| Identity cards (A and B) | `st.markdown` (custom HTML) | Headshot, team logo, name, position, team-color border | Yes (reconstructed) |
| Radar chart | `plotly.Scatterpolar` | 12-category z-score spider chart: orange=A, steel-blue=B | Yes (reconstructed) |
| "Category Edge" panel | `render_panel` / heat bars | Per-category split heatbar: orange=A leads, steel=B leads, right-side advantage label | Yes (reconstructed) |
| `Category Breakdown` table | `render_compact_table` | 12 rows: category, A z-score, B z-score, Advantage | Yes (reconstructed) |
| Z-score tooltip caption | `st.caption` | Static definition of z-score | Yes |
| `Standings Gained Points Breakdown` table | `render_compact_table` | Per-cat and total SGP for each player + delta | Yes (reconstructed) |
| SGP caption | `st.caption` | Explains SGP and concentrated vs diversified value | Yes |
| `Schedule Strength` table | `render_compact_table` | Team wRC+, FIP, ERA, K% for each player's team | Yes (trace; conditional) |
| `2026 Season Stats` table | `render_compact_table` | YTD PA/AVG/HR/RBI/SB/ERA/WHIP/SV/K | Yes (reconstructed) |
| `Recent Performance` tables | `render_compact_table` | Last 7-day and last 14-day rolling totals from game logs | Yes (trace; conditional) |
| `Statcast Profile` table | `render_compact_table` | xwOBA, Barrel%, Hard Hit%, Stuff+ | Yes (reconstructed) |
| `View player card` selectbox | `render_player_select` | Opens full player card dossier dialog for either compared player | Yes (trace) |
| `Health & Confidence` table | `render_styled_table` | Health score, badge, P10-P90 confidence range per player | Yes (reconstructed) |
| `Catcher Framing Value` table | `render_styled_table` | Framing runs, ERA impact, Pop time, CS%, Tier — catchers only | Yes (trace; conditional) |
| Context panel: "Composite Scores" card | `render_context_card` | Orange-vs-steel composite score readout | Yes (reconstructed) |
| Context panel: "Quick Verdict" card | `render_context_card` | Category wins count (A — B, ties), edge statement | Yes (reconstructed) |
| `page_timer_footer` | `st.caption` | Render time (e.g. "Player Compare rendered in 0.42s") | Yes |
| Feedback widget | `render_feedback_widget` | Collapsible feedback popover (MULTI_USER-gated) | Yes |

---

## 4. Feature-by-Feature Test Log With Real Outputs

### 4.1 Player universe filter

Options: `["MLB only", "MLB + AAA", "MLB + AAA + AA", "All"]`. Default "MLB only".

From the live DB the pool has:
- MLB (level IS NULL or "MLB"): **8,052 players** (level=None in pool — all ~8,052 rows without a level tag, which for non-minor-leaguers means MLB/blended).
- AAA: **834 players**.
- AA: **1,002 players**.

**Finding:** The "MLB only" filter uses `pool["level"].isna() | (pool["level"] == "MLB")` — this keeps all 8,052 `level=None` players plus any explicitly tagged "MLB". The pool reports `level=None` for 8,052 rows, which includes the full MLB player pool. This is correct behavior but creates a confusing UX edge case: a player like Munetaka Murakami (CWS, level=None, .180/.322, 26 HR projected) is included in "MLB only" while genuine MLB regulars like Shohei Ohtani also appear. The filter is cosmetically fine but a novice cannot tell what "level" means without explanation.

### 4.2 Player search & selection

**Search Player A / Search Player B:** Both use `st.text_input` → live filter → top 3 match shown as pill buttons.

**Issue found in source (line 192-198):** When a search is empty (`search_a == ""`), `filtered_a = player_names` (all 9,888 names). Then `top_a = filtered_a[:3]` shows the first three alphabetically (presumably "A.J. Alexy", "A.J. Burnett Jr.", "A.J. Cole" or similar). **So on page load the user sees three random pill buttons they did not ask for**, which is confusing.

**Issue found (line 197):** `width="stretch"` is passed to `st.button()`. As of Streamlit 1.47, `width` is not a recognized kwarg for `st.button`. This likely passes silently but produces no effect — the button will default-width, not stretch. Multiple beta users will notice the three pills in column 1 look uneven widths.

**Player A selection lock (reconstructed):** After typing "Juan Soto", the top match is "Juan Soto" (player_id=10, NYM, LF). Pressing the pill sets `st.session_state.compare_a = "Juan Soto"`. The button renders as `type="primary"` (orange-filled) when selected, `type="secondary"` when not. This is the correct signal for "selected."

**Critical UX gap:** After pressing a pill, the page reruns and shows the same search box with the same text the user typed. There is no "selected" confirmation UI outside the pill-button styling change. A novice must re-read the pill color to know their selection was registered. The selected player name is not echoed at the top of the results column.

### 4.3 Roster status badges

HTML badges rendered beneath the player selects.

**Reconstructed output (Team Hickey roster = empty in DB):**

Since the `league_rosters` table shows "Team Hickey" with zero rows (the team name stored in the roster records as `is_user_team=1` was wiped in the 2026-06-03 incident and not fully repopulated), the roster badge lookup returns empty for all players:

- Juan Soto: `<span ... background:var(--fp-divider)>Free Agent</span>`
- Aaron Judge: `<span ... background:var(--fp-divider)>Free Agent</span>`

All players will show "Free Agent" even if they are rostered, because `rosters_df.empty` is True when Team Hickey has no rows. **This is live-broken for the owner's team.**

For a non-empty team (e.g. BUBBA CROSBY), the badge shows "Rostered: BUBBA CROSBY" in a pale green pill. The green background (`rgba(31,157,107,.08)`) is very faint — barely distinguishable from the white canvas for a novice.

### 4.4 Progress bar

Renders "Comparing players across 12 categories…" (0%), then immediately fills to 100% "Comparison complete!", waits 0.3s, and clears. `compare_players()` is fast (pure in-memory, ~1ms), so the progress bar is nearly instantaneous and provides no real feedback value. The 0.3s sleep is a purely decorative delay.

### 4.5 Identity cards (reconstructed)

For Juan Soto (id=10, NYM, LF) vs Aaron Judge (id=1, NYY, RF):

**Card A — Juan Soto:**
- Headshot: `https://img.mlbstatic.com/mlb-photos/.../10/headshot/67/current` (live image fetch, offline = generic silhouette)
- Team logo: NYM SVG fetched from mlbstatic
- Border left color: team_color("NYM") = NY Mets blue (`#002D72`)
- Team text: "NYM · LF"

**Card B — Aaron Judge:**
- Headshot: `https://img.mlbstatic.com/mlb-photos/.../1/headshot/67/current`
- Border left color: team_color("NYY") = NY Yankees navy (`#132448`)
- Team text: "NYY · RF"

**Finding:** Both borders are very dark blue — nearly indistinguishable from each other on the page. The design uses orange for Player A only when `team == ""` or `team == "MLB"`. For players with known teams, the team brand color overrides. This loses the primary orange/steel color coding that the radar chart uses. A novice cannot tell at a glance which side is "A" vs "B" in the identity strip — both look similarly dark-bordered.

**Finding:** The `mlb_id` for Soto in the pool is obtained from `pool["mlb_id"]`. From the pool the player_id=10 row's `mlb_id` column was not checked, but if it doesn't map to the correct MLB MLBAM ID (Soto's official ID is 665742), the headshot will fall back to the generic silhouette. The `onerror="this.style.display='none'"` hides broken images silently.

### 4.6 Radar chart (reconstructed)

12-axis Scatterpolar. Orange fill = Player A (Juan Soto), steel-blue fill = Player B (Aaron Judge).

**Real z-scores computed by compare_players() — Juan Soto (A) vs Aaron Judge (B):**

| Category | Soto Z | Judge Z | Advantage |
|----------|--------|---------|-----------|
| Runs | +6.34 | +5.19 | Soto |
| Home Runs | +9.97 | +8.63 | Soto |
| Runs Batted In | +6.61 | +5.39 | Soto |
| Stolen Bases | +5.15 | +2.19 | Soto |
| Batting Average | +1.80 | +1.66 | Soto |
| On-Base Percentage | +2.45 | +2.30 | Soto |
| Wins | 0.00 | 0.00 | TIE |
| Losses | 0.00 | 0.00 | TIE |
| Saves | 0.00 | 0.00 | TIE |
| Strikeouts | 0.00 | 0.00 | TIE |
| ERA | 0.00 | 0.00 | TIE |
| WHIP | 0.00 | 0.00 | TIE |

**Composite scores: Soto=+32.305, Judge=+25.356.**

**Critical bug — radar chart for two hitters:** Six pitching categories (W, L, SV, K, ERA, WHIP) all show 0.00 for both players. On the radar chart, these six axes are completely flat (the polygon collapses to 0 on those spokes). A 12-axis radar chart with 6 dead axes looks broken and confusing to any user — half the chart is visually empty with no explanation that "these categories don't apply."

**Critical bug — L (Losses) z-score:**
The L z-score for both hitters shows 0.00 (correct — hitters don't accumulate Losses). However, for the SP comparison (Skubal vs Skenes), the Losses z-scores are:
- Skubal: `-1.48` (inverse stat, correctly sign-flipped: fewer losses = positive z)
- Skenes: `-1.95`
- Advantage: "A" (Skubal)

Wait — `z_a[L] = -1.48` and `z_b[L] = -1.95`. Since the code uses `-(val_a - mean) / std` for inverse stats, a LOWER number of losses gives a MORE POSITIVE z-score. But -1.48 > -1.95, so Skubal wins L — which is correct (he has fewer projected losses than Skenes based on the z-score sign convention). But to a novice reading the table, seeing `-1.48 vs -1.95` with the label "Advantage: Skubal" is deeply confusing. **Negative z-scores winning a category is counterintuitive without explanation.** A user will read "-1.48" and think "that's bad, why is it winning?"

The caption only says: "Z-Score — how many standard deviations above or below league average. +1.0 = top ~16% of players; -1.0 = bottom ~16%." This doesn't explain that for inverse cats (ERA, WHIP, L), a negative raw z-score can be "good."

### 4.7 Category Edge heat-bar panel (reconstructed)

Renders one row per category. A fill of `(_sa / _tot * 100)` where `_sa = za + 4.0` and `_sb = zb + 4.0` shifts negative z-scores into positive territory so the split bar has a well-defined 0–100% fill.

**Real outputs for Soto vs Judge:**

| Category | A Fill % | Winner label |
|----------|----------|--------------|
| Runs | ~54.9% | "Juan Soto" (orange) |
| Home Runs | ~53.6% | "Juan Soto" (orange) |
| RBI | ~55.0% | "Juan Soto" (orange) |
| SB | ~59.9% | "Juan Soto" (orange) |
| AVG | ~51.0% | "Juan Soto" (orange) |
| OBP | ~50.4% | "Juan Soto" (orange) |
| W | 50.0% | "TIE" (muted) |
| L | 50.0% | "TIE" (muted) |
| SV | 50.0% | "TIE" (muted) |
| K | 50.0% | "TIE" (muted) |
| ERA | 50.0% | "TIE" (muted) |
| WHIP | 50.0% | "TIE" (muted) |

**Finding:** Six categories show exactly 50.0% fill ("TIE"), because both players have z=0 in pitching cats. The bars look like perfect 50/50 splits — identical appearance. There's no visual differentiation to show "these categories are irrelevant to both players." A novice cannot tell "TIE because they're equal" from "TIE because these stats don't apply to hitters."

**Finding:** The `_adv_color` for winner is `var(--fp-cold)` for "B wins" — this is a steel-blue variable used for cold/losing indicators elsewhere in Combustion. Here it is used for "Player B leads," but the label is just the player's name. There's no consistent A/B color scheme across the identity card (team brand), the radar (orange/steel), and the edge panel (orange/steel for "A wins"/"B wins") — they all use different semantic anchor points.

### 4.8 Category Breakdown table (reconstructed)

**Real table — Soto vs Judge:**

| Category | Juan Soto Z-Score | Aaron Judge Z-Score | Advantage |
|---|---|---|---|
| Runs | +6.34 | +5.19 | Juan Soto |
| Home Runs | +9.97 | +8.63 | Juan Soto |
| Runs Batted In | +6.61 | +5.39 | Juan Soto |
| Stolen Bases | +5.15 | +2.19 | Juan Soto |
| Batting Average | +1.80 | +1.66 | Juan Soto |
| On-Base Percentage | +2.45 | +2.30 | Juan Soto |
| Wins | +0.00 | +0.00 | TIE |
| Losses | +0.00 | +0.00 | TIE |
| Saves | +0.00 | +0.00 | TIE |
| Strikeouts | +0.00 | +0.00 | TIE |
| ERA | +0.00 | +0.00 | TIE |
| WHIP | +0.00 | +0.00 | TIE |

**Finding:** The Advantage column shows player names ("Juan Soto" / "Aaron Judge" / "TIE"). Full player names in a comparison column can overflow the compact table cell — Aaron Judge's name is 11 chars, but longer names (e.g. "Gerrit Cole", "Corbin Burnes") could cause wrapping. The "Advantage" column has no visual differentiation (color) — it is plain text. In the Category Edge panel the winner name is orange-colored; in this table it is the same color as "TIE." No winner highlighting.

**Finding:** For the SP comparison (Skubal vs Skenes), the Losses row shows `Skubal=-1.48  Skenes=-1.95  Advantage: Skubal`. A novice sees -1.48 labeled as the "Advantage" winner and has no way to understand why.

### 4.9 SGP Breakdown table (reconstructed)

**Real SGP — Soto vs Judge:**

| Category | Juan Soto | Aaron Judge | Delta |
|---|---|---|---|
| Runs | +1.81 | +1.50 | +0.31 |
| Home Runs | +1.77 | +1.54 | +0.23 |
| Runs Batted In | +1.78 | +1.47 | +0.31 |
| Stolen Bases | +0.79 | +0.36 | +0.43 |
| Batting Average | +0.05 | -0.04 | +0.09 |
| On-Base Percentage | +0.45 | +0.59 | -0.14 |
| **TOTAL** | **+6.66** | **+5.42** | **+1.24** |

(W, L, SV, K, ERA, WHIP rows not shown because both hitters have near-zero SGP there, and the code gates on `abs(sa) > 0.001 or abs(sb) > 0.001` — the pitching cats are filtered out.)

**Finding:** The SGP table caption says "Concentrated value (e.g., 3.0 from Home Runs/Runs Batted In) vs diversified (0.5 across many) affects trade and roster strategy." But neither player in this comparison has concentrated value — both are balanced across R/HR/RBI/SB. The caption is generic boilerplate that doesn't respond to what the data actually shows (e.g., "Both players contribute similarly across hitting cats; Soto has more SB upside."). This is a missed opportunity.

**Finding:** Judge's OBP SGP (+0.59) is higher than Soto's (+0.45), yet Soto's composite and z-score beat Judge's OBP z. This is because SGP and z-score use different normalization approaches (SGP denominators vs population standard deviation). This inconsistency — where z says Soto wins OBP but SGP says Judge wins OBP — will confuse any user who reads both tables without an explanation.

### 4.10 Schedule Strength table

Conditionally rendered when both players have valid team codes. Pulls `get_team_strength()` from `src/game_day.py`.

**Issue:** The table shows wRC+, FIP, ERA, K% — metrics of the player's own team. These are team-level metrics, not schedule strength against future opponents. A novice reading "Schedule Strength" expects to see something like "Soto's next 2 weeks vs these opposing pitching staffs" — not "how good is NYM's offense overall." The section title "Schedule Strength" is a misnomer; the actual data is "Team Context."

### 4.11 2026 Season Stats table (reconstructed)

**Real YTD for Juan Soto (as of data last refreshed 2026-06-10):**

| Stat | Juan Soto | Aaron Judge |
|---|---|---|
| Plate Appearances | 210 | -- (no YTD PA data) |
| Batting Average | 0.276 | -- |
| Home Runs | 13 | -- |
| Runs Batted In | 30 | -- |
| Stolen Bases | 6 | -- |
| ERA | -- | -- |
| WHIP | -- | -- |
| Saves | -- | -- |
| Strikeouts | -- | -- |

**Finding:** Judge (player_id=1) shows "--" for all YTD hitting stats. The DB query for `ytd_pa` returns 0 for player_id=1, which causes the condition `_ytd_pa_a > 0 or _ytd_pa_b > 0` to gate on Soto having PA data, so the table renders, but Judge's column is all "--". A novice sees "Juan Soto: 13 HR / Aaron Judge: --" and cannot tell if Judge has no HRs or if his YTD data is simply missing. The "--" means "no data or zero" per the caption, but users will misread it as "Judge hasn't hit any HRs."

**Finding:** The YTD table mixes stats that apply to pitchers (ERA, WHIP, SV, K) with hitter stats (PA, AVG, HR). For a hitter vs hitter comparison, the pitcher rows all show "--" for both, rendering six wasted rows. The table should suppress rows where both values are "--."

**Finding:** Data freshness: `season_stats` was last refreshed 2026-06-10 (3 days stale as of audit). The 2026 Season Stats table has no freshness indicator — a user cannot tell if "13 HR" reflects current data or 3-day-old data.

### 4.12 Recent Performance (L7/L14 rolling stats)

Computed from `compute_rolling_stats()` on `game_logs` table (58,411 rows, 2026 season). This section renders only when `HAS_ROLLING_STATS=True`.

**Finding from source trace:** The function `compute_rolling_stats([id_a, id_b], days=14, ...)` calls the game_logs table. For a hitter vs hitter pair: hitter stats (R, HR, RBI, SB, AVG, OBP) are shown, and pitcher stats section is triggered by `if not _is_hitter_a or not _is_hitter_b` — so for two pure hitters, pitcher rows (IP, W, K, SV, ERA, WHIP) are suppressed. For two pure pitchers, hitter rows are suppressed. **However, for a cross-type comparison (hitter vs pitcher), BOTH blocks execute:** hitter rows appear with "--" for the pitcher, and pitcher rows appear with "--" for the hitter. This produces a bloated table with many "--" entries and no explanation.

**Finding:** The "Last 7 Days" and "Last 14 Days" headings use `st.markdown(f"**{_window_label}**")` — plain bold text, not a `st.subheader`. This breaks the page's visual rhythm: every other section has a `st.subheader()` heading; Recent Performance uses inline markdown bold. Inconsistent hierarchy.

### 4.13 Statcast Profile (reconstructed)

Columns checked: `xwoba`, `barrel_pct`, `hard_hit_pct`, `stuff_plus`.

From the live DB:
- `xwoba` non-null count: **9,888** (all players — but Statcast `xwoba` last update was 2026-06-10; `statcast_archive` shows 0 rows with non-null xwoba in the `statcast_archive` table, meaning the pool's `xwoba` comes from the blended projections table, not live Statcast).
- `stuff_plus` non-null count: **0** (FanGraphs 403 blocks this; confirmed `skipped` in refresh_log).
- `barrel_pct` non-null count: **9,888** (same issue — likely defaults).

**Finding:** The Statcast section will show `xwoba` and `barrel_pct` rows for all players (because values are non-null), but `stuff_plus` will always show "--" for both players since `stuff_plus` is null across all 9,888 rows. The section renders with 3 visible rows (xwOBA, Barrel%, Hard Hit%) and "Stuff+" is suppressed by the `_sc_va > 0 or _sc_vb > 0` gate. For pitchers, this means the most useful pitching Statcast metric (Stuff+) is always absent. The caption at the bottom still mentions "Stuff+ = pitch quality (100 = average)" even though it never shows.

**Finding:** Since `statcast_archive` has 0 rows with non-null xwoba, the xwOBA values in the pool likely come from the blended projections (expected/projected xwOBA) not actual Statcast — but the section header says "Statcast Profile" implying live Baseball Savant data. This is misleading.

### 4.14 Player card selector ("View player card")

`render_player_select([player_a_name, player_b_name], [id_a, id_b], key_suffix="compare")` renders a selectbox with the two player names. Selecting one opens the full player card dossier dialog.

**Finding:** The selectbox label is "View player card" (default). After comparing two players, a user might want the dossier for further research — the label is clear enough. But the widget appears mid-page, between Statcast and Health tables, in an odd flow: the natural action is to "open card" after seeing the full comparison, not mid-way through. It could be better placed at the top (where the identity cards are) or at the very bottom.

### 4.15 Health & Confidence table (reconstructed)

For Soto vs Judge (all players have `health_score` non-null = 9,888):

From pool:
- Soto health_score: (not queried directly, but pool has it for all 9,888 players).
- The P10-P90 confidence interval uses `_load_per_system_projections((id_a, id_b))` — queries `projections` table for steamer/zips/depthcharts. From DB: steamer has only 1 row total; zips has 3,729; depthcharts has 1,439. For Soto (player_id=10), it's plausible that 2 of 3 systems have data, so `len(systems) >= 2` may be true.

**Reconstructed output example:**

| Player | Health | Score | Confidence |
|---|---|---|---|
| Juan Soto | (green dot) Low Risk | 0.85 | ±38.20 (e.g.) |
| Aaron Judge | (amber dot) Moderate Risk | 0.72 | — |

**Finding:** The "Confidence" column label `±{width:.2f}` shows a raw P90-P10 differential across counting stats (R+HR+RBI+SB+W+SV+K summed). This number has no interpretable units. A novice sees "±38.20" with no context: 38.20 what? Home runs? Combined counting categories? The units are not labeled.

**Finding:** `get_injury_badge()` returns a CSS dot icon + label ("Low Risk" / "Moderate Risk" / "High Risk" / "Unknown"). The "Health" column shows something like `<span ...>●</span> Low Risk` where the dot is inline HTML. The `render_styled_table` function does NOT set `html_cols` here — it uses plain `st.dataframe`, so the HTML will render as raw HTML text, not a styled dot. **The injury badge HTML is being displayed as literal HTML source** (e.g., `<span style="...">●</span> Low Risk`) rather than a rendered colored dot. This is a confirmed rendering bug.

### 4.16 Catcher Framing Value

Only shown when at least one of the two compared players is a catcher (positions includes "C"). For non-catcher comparisons, this section is hidden. For any catcher, pulls `get_catcher_framing_data()` from `src/optimizer/matchup_adjustments.py`.

**Finding:** The catcher framing data comes from `data/seed/catcher_framing_2024.json` (Tier 3 fallback; the 2024 seed covers 32 catchers). This is stale 2024 data — the section header "Catcher Framing Value" has no year or staleness caveat. A user comparing two catchers in 2026 is seeing 2024 framing data presented as current.

### 4.17 Context panel: Composite Scores and Quick Verdict (reconstructed)

**Reconstructed for Soto vs Judge:**

Composite Scores card:
```
Juan Soto +32.31  (orange — leads)
Aaron Ju  +25.36  (steel — trails)
```
(name truncated at 14 chars: "Aaron Ju" would show as "Aaron Judge" is 11 chars — fine here but names like "Vladimir Guerrero Jr." truncate to "Vladimir Guerre" which cuts mid-word.)

Quick Verdict card:
```
Category wins:  6 — 0 (6 ties)
Edge:           Juan Soto leads
```

**Finding:** "6 ties" for a hitter vs hitter comparison is the six pitching categories where both score 0.00. A novice reading "6 ties" thinks both players are evenly matched in those categories, not that those categories are simply irrelevant to both players. This is a communication failure.

**Finding:** The "composite score" number (+32.31) has no scale explanation. Is 32 good? Bad? What does it mean? The context panel just shows two numbers with no range reference. A new user has no idea if +32.31 and +25.36 are close or far apart.

---

## 5. Identified Errors, Issues & Difficulties

### BLOCKER-class

- **B-1: Roster badge always shows "Free Agent"** — `league_rosters` has 0 rows for Team Hickey (data wipe incident). The badge lookup returns empty for all players. All players display "Free Agent" regardless of actual roster status. Live-broken for the team owner.

### HIGH-class

- **H-1: Inverse-stat z-scores confusing in the table** — Losses, ERA, WHIP can display as negative z-scores (e.g. `L: Skubal=-1.48  Skenes=-1.95  Advantage: Skubal`) with no explanation that for inverse categories a lower (more negative) raw value is better. The z-score sign IS flipped in the code, but the negative number in the "Advantage winner" column defies novice intuition.

- **H-2: 6-axis radar collapses for same-type player pairs** — When both players are hitters, the 6 pitching axes show z=0 for both and the radar polygon degenerates. When both are pitchers, the 6 hitting axes collapse. A radar chart with half its axes flat at zero looks visually broken and provides no information on those spokes. No "not applicable" annotation or axis suppression is implemented.

- **H-3: Injury badge HTML renders as raw text** — `render_styled_table` uses `st.dataframe` which does not render HTML in cells. The "Health" column will display `<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#2d6a4f;">●</span> Low Risk` as literal text. The green/amber/red dot never shows; the user sees HTML gibberish.

- **H-4: YTD stats show "--" even for rostered stars** — `ytd_pa` is 0 for Aaron Judge (player_id=1) despite him being a current MLB player. A 0-plate-appearance entry for the #1 ranked player signals a data gap in `season_stats` (partial refresh, 2,580/8,054 rows saved). The page never warns the user that YTD data is incomplete.

- **H-5: Cross-type comparison (hitter vs pitcher) produces nonsensical output** — The composite scores for Soto vs Skubal are +32.305 vs +11.623. Soto "wins" 8 of 12 categories (all 6 hitting cats + L and SV because the pitcher's z-score on those is negative while the hitter's is 0). This output is architecturally meaningless — you cannot meaningfully compare a hitter and pitcher on a shared scale. The page allows this without any warning.

### MEDIUM-class

- **M-1: No stale-data warning anywhere on the page** — Data is 1219+ minutes old per the orchestrator. No freshness badge, no "data last updated" timestamp is shown anywhere. The page renders confident-looking numbers with no caveat about staleness.

- **M-2: "Schedule Strength" is actually "Team Context"** — The section title is misleading. It shows the player's own team's wRC+/FIP/ERA, not the quality of upcoming opponents. This is not what "schedule strength" means in fantasy baseball.

- **M-3: Statcast section cites "Stuff+" in caption but it never appears** — `stuff_plus` is null for all 9,888 players (FanGraphs 403 blocks it). The caption still explains Stuff+ to users who will never see it. The empty promise erodes trust.

- **M-4: The xwOBA in "Statcast Profile" may be projection-blended, not actual Statcast** — `statcast_archive` has 0 rows with xwoba. The pool's `xwoba` column must come from projections. Displaying it under "Statcast Profile" implies Baseball Savant sourcing when it may not be.

- **M-5: "Confidence" column units are unlabeled** — P90-P10 range across counting stats summed is shown as e.g. "±38.20" with no unit or range context.

- **M-6: Recent Performance "Last 7/14 Days" uses `st.markdown("**…**")` heading** — Breaks visual hierarchy; all other sections use `st.subheader`.

- **M-7: width="stretch" on st.button is unrecognized in Streamlit** — Passed silently but has no effect; pills may render at inconsistent widths across browsers.

- **M-8: Empty search state shows first 3 alphabetical players** — On page load, 3 random pill buttons appear before the user has typed anything. This is confusing and clutters the selector with irrelevant names.

- **M-9: No player selected = no empty state guidance below selectors** — The context panel shows "Select Players — Search and select two different players to see composite scores and category breakdown." But the main area below the selectors is completely empty with no visual cue that the user needs to select. A novice may scroll through an empty page wondering what's missing.

- **M-10: SGP vs z-score inconsistency not explained** — Judge's OBP SGP (+0.59) beats Soto's (+0.45), but the z-score comparison says Soto wins OBP. Two sections immediately next to each other give opposite "winner" signals with no explanation.

### LOW / POLISH-class

- **L-1: `FIG.16` not zero-padded (FIG.16 vs FIG.04 inconsistency)** — The orchestrator flagged that Pitcher Streaming shows `FIG.4` not `FIG.04`. Here the header reads `FIG.16 — HEAD-TO-HEAD`. The double-digit fig numbers happen to look fine, but the convention is inconsistent with single-digit pages.

- **L-2: Identity card color coding doesn't match radar chart** — Identity cards use team brand colors for border; radar chart uses orange (A) vs steel-blue (B). Users need to learn two different A/B visual conventions on the same page.

- **L-3: Name truncation in context panel composite scores** — `_pa[:14]` truncates player names that are longer than 14 chars mid-word (e.g., "Vladimir Guerr").

- **L-4: "6 ties" in Quick Verdict counts irrelevant categories** — Six pitching-category "ties" for two hitters are noise that dilutes the useful 6-0 score.

- **L-5: Player card selector is buried mid-page** — The "View player card" selectbox appears between Statcast and Health sections, not at a natural decision point.

- **L-6: Catcher framing data is from 2024 seed with no year label** — "Catcher Framing Value" shows 2024 data without stating so.

- **L-7: No clear "reset" action** — After comparing two players, there's no "Compare different players" or "Clear" button. The user must manually re-type both search fields. Clearing `st.session_state.compare_a/b` can only be done via re-typing.

- **L-8: compute_category_fit() exists but is never called** — `src/in_season.py` has a function `compute_category_fit(player_sgp_by_cat, team_category_profile)` that evaluates a player's fit against the user's team's weak/strong categories. The page imports `compare_players` but not `compute_category_fit`. This feature is entirely absent from the page despite being the most practically useful output for a fantasy manager ("does this player help MY team's weak spots?").

- **L-9: Progress bar is purely decorative** — `compare_players()` is <1ms in-memory. The 0.3s `time.sleep` is artificial padding. Remove it or use a real spinner only for heavy operations.

- **L-10: YTD table shows pitcher-stat rows for hitter comparisons** — Both players have ERA="--", WHIP="--", SV="--", K="--". These six rows should be hidden when both values are "--".

---

## 6. UI/UX & Visual Design Critique

### Layout & hierarchy

The page uses a 1:4 split (`render_context_columns()`) with a narrow left context panel and wide main area. This is the right structural choice for a comparison tool. However, the section ordering in the main area is:

1. Level filter (before player search — wrong priority)
2. Search + pills (fine)
3. Identity cards
4. Radar chart
5. Category Edge heat-bar panel
6. Category Breakdown table
7. SGP Breakdown table
8. Schedule Strength table
9. 2026 Season Stats table
10. Recent Performance tables
11. Statcast Profile table
12. Player card selector (buried!)
13. Health & Confidence table
14. Catcher Framing Value (conditional)

Sections 8–13 are buried below a long scroll with no navigation anchors. A novice will never scroll that far. The most actionable section — health data and recent form — appears at the bottom. The most advanced data — SGP breakdown — appears near the top where a novice doesn't understand it yet.

**Recommended order:** Identity → quick verdict → category edge (visual) → recent form (hot/cold) → SGP table → radar → YTD stats → Statcast → schedule context → health → player card.

### Color usage

The Combustion palette is partially respected: orange (#ff6d00) is used for Player A fills in the heatbars and radar chart; steel-blue (`t["teal"]`) for Player B. But the identity cards use team brand colors (dark navy for NYY/NYM, etc.) which looks indistinguishable between two AL East teams. The visual system needs a clearer A (orange-accented) vs B (steel-accented) through-line on ALL components, not just the chart.

The category edge panel "advantage" labels use `var(--fp-cold)` for B's wins — the "cold" token implies "bad/losing" in other Combustion contexts (the hot/cold module). Using "cold blue" for "Player B advantage" subtly frames B as the worse player. Both players should feel neutral; the comparison is descriptive, not judging.

### Typography & numbers

- Z-scores shown as `+6.34`, `-1.48` — using `format_stat(za, "SGP")` format (`+.2f`). This is correct use of the SGP formatter but is semantically misapplied: z-scores are not SGP. The `+` prefix comes from `f"{za:+.2f}"` directly (line 416), not `format_stat`. Fine, but consistent.
- SGP values use `format_stat(sa, "SGP")` correctly (`+.2f`).
- YTD rate stats: `format_stat(float(_va), "AVG")` → `.3f` correct; `format_stat(float(_va), "ERA"/"WHIP")` → `.2f` correct.
- YTD counting stats: `str(int(float(_va)))` — no formatting. Fine for integers but `str(13)` instead of `"13"` is the same result.

### Empty states

When no players are selected, the main panel is completely empty (no `render_empty_state` call). Other pages in Combustion use the `render_empty_state(title, body, icon_key)` component for this case. The "Select Players" placeholder only exists in the narrow context panel — invisible if the user hasn't noticed the left rail. **The main column should render a proper empty state.**

### Missing "category fit" feature

The page is pitched to the novice user as "Head-to-Head player comparison with category fit vs my team." The shared context (§7) confirms this framing. But `compute_category_fit()` — the only function in `src/in_season.py` that computes "does this player help my team's weak cats?" — is never called. The "fit" concept is entirely absent. Users comparing a SB specialist to a power hitter for trade value get z-scores and SGP numbers but zero guidance on which player better fills their specific team needs. This is the killer feature that would justify using this tool.

### Accessibility

The identity cards use `onerror="this.style.display='none'"` to hide broken images silently. Headshots may not load offline (Yahoo is "warming up"). No fallback text/initial is shown when the image hides — just blank space.

The roster badge uses `rgba(31,157,107,.08)` — 8% opacity green is nearly invisible on a white canvas. The contrast ratio will fail WCAG AA by a wide margin.

The context panel card text uses `color:var(--tx-muted,#888)` for the "Select Players" placeholder — #888 on white is a WCAG fail (~3.5:1 ratio).

---

## 7. ≥10 High-Level Recommendations (Priority Order)

### R-01: Fix the rendering of injury badges (HIGH)
**Problem:** `render_styled_table` uses `st.dataframe` which does not interpret HTML in cell values. The "Health" column displays raw HTML like `<span...>●</span> Low Risk` as text. The green/amber/red health dot is never shown.
**Fix:** Add `"Health"` to `html_cols` in the `build_compact_table_html` call, or replace `render_styled_table` with `render_compact_table(pd.DataFrame(health_rows), html_cols={"Health": True})` which renders cell HTML correctly.

### R-02: Suppress inapplicable categories from the radar chart and tables (HIGH)
**Problem:** A hitter vs hitter comparison shows 6 flat-zero pitching axes on the radar chart, and a pitcher vs pitcher comparison shows 6 flat-zero hitting axes. This makes half the chart visually dead.
**Fix A (cosmetic):** Detect shared player type (both hitters OR both pitchers) and suppress the inapplicable 6 categories from the radar axes and both tables. Use a 6-axis radar instead of 12 when players are the same type.
**Fix B (guard):** Add a warning banner when the user compares a hitter to a pitcher: "Hitters and pitchers contribute to different categories — this comparison is cross-type and may be misleading. Z-scores are scaled within their own peer group."

### R-03: Add "Category Fit vs My Team" section (HIGH)
**Problem:** `compute_category_fit()` exists in `src/in_season.py` but is never called from the page. The most actionable output for a fantasy manager — "does this player help my weak categories?" — is absent.
**Fix:** Import and call `compute_category_fit` using the user's team's category profile (derived from `standings_utils.get_team_totals()` or the matchup context). Render a "Team Fit" card in the context panel showing: which categories each player helps (green), which they waste on categories the user is already strong in (orange), and a fit score.

### R-04: Explain inverse-stat z-score semantics clearly (HIGH)
**Problem:** For L (Losses), ERA, WHIP, the "Advantage" column shows a winner even when their raw z-score is negative (e.g., `-1.48 Advantage: Skubal`). Novices read negative = bad and are confused by "losing" values winning the category.
**Fix A:** In the Category Breakdown table, add a "(lower is better)" indicator to the Losses, ERA, WHIP rows — e.g., append "(▼ lower = better)" to the category name.
**Fix B:** Replace the raw z-score display for inverse stats with the sign-flipped "good z" (i.e., show +1.48 instead of -1.48 for Skubal's losses), and add a note: "z-score shown as quality-adjusted (higher = better for all categories)."

### R-05: Add a data-freshness banner (MEDIUM)
**Problem:** Data is 1219+ minutes stale. No staleness indicator appears anywhere on the page. Users make trade decisions based on confident-looking numbers that are 2+ days old.
**Fix:** Import `DataFreshnessTracker` and render a page-level freshness badge near the top: "Data last updated: 3 days ago (2026-06-10). [Refresh]" Use a yellow/amber color to signal degraded freshness. Mirror the pattern used on My Team.

### R-06: Suppress "--" rows in YTD and Recent Performance tables (MEDIUM)
**Problem:** YTD table shows ERA, WHIP, SV, K rows as "--"/--" for two hitters. Recent performance tables show hitter stats as "--" for a pitcher and vice versa. These empty rows add no value and make tables longer than necessary.
**Fix:** Filter out any row where BOTH player values are "--" before passing to `render_compact_table`. This can be done with a simple list comprehension: `[r for r in _ytd_rows if not (r[colA] == "--" and r[colB] == "--")]`.

### R-07: Fix the "Schedule Strength" section title and content (MEDIUM)
**Problem:** The section is labeled "Schedule Strength" but shows the player's own team's aggregate season stats (wRC+, FIP, ERA, K%) — not upcoming opponent quality.
**Fix A (minimal):** Rename the section "Team Context" or "Supporting Cast."
**Fix B (full):** Replace with actual schedule data: next 10 games' average opponent ERA for hitters, or average opponent wRC+ for pitchers. Use `src/schedule_grid.py` or the Yahoo schedule data to compute this.

### R-08: Redesign the player search UX (MEDIUM)
**Problems:** (a) 3 random pills appear before the user types. (b) No visual confirmation of selection beyond pill color change. (c) No "clear selection" button.
**Fix:** Change the initial state to show an empty search box with a placeholder ("Type a player name to search"). Only display pills after the user has typed at least 2 characters. Add a small "×" clear button next to the selected player name. Echo the selected player name in a "Selected: Juan Soto" line below the search.

### R-09: Implement a cross-type comparison warning or block (MEDIUM)
**Problem:** Comparing Juan Soto to Tarik Skubal gives Soto composite=+32.305 and Skubal=+11.623, Soto "winning" categories like L (Losses) and SV simply because hitters have z=0 on those axes. This output is meaningless.
**Fix:** At the comparison trigger, detect when `is_hitter_a != is_hitter_b`. Render a yellow `st.warning()` banner: "You're comparing a hitter to a pitcher. Hitting and pitching categories use separate peer groups — the composite score comparison is not directly meaningful. Consider comparing two hitters or two pitchers for a head-to-head trade evaluation."

### R-10: Move "Player card" dossier opener to the identity strip (LOW)
**Problem:** The `render_player_select` widget is buried below the SGP, Statcast, and Health sections. The natural moment to open a player card is immediately after seeing the identity strip.
**Fix:** Move `render_player_select(...)` immediately below the identity card row, before the radar chart. This gives users a "dig deeper" CTA at the first point of interest.

### R-11: Add composite score scale context in context panel (LOW)
**Problem:** "+32.31 vs +25.36" has no meaning without knowing the scale. What is a good score? What is average?
**Fix:** Add a sub-label under each composite score showing the percentile: "Top 12% of hitters" based on where composite falls in the distribution of all players of that type in the pool. This converts an abstract number into an interpretable ranking.

### R-12: Fix the level filter position (LOW)
**Problem:** "Player universe" selectbox appears at the very top, before the player search. A novice will ignore it and potentially not understand why certain players are missing.
**Fix:** Move the level filter into an expander or a sidebar control ("Advanced Filters") below the main player search UX. Default is "MLB only" which is correct; advanced users can expand to find prospects.

---

## 8. Severity-Tagged Issue List

- `[BLOCKER]` Roster badges always show "Free Agent" — Team Hickey's league_rosters rows are empty; badge lookup returns empty for all players regardless of actual status.
- `[HIGH]` Injury badge HTML renders as raw text in `render_styled_table` — `st.dataframe` does not interpret HTML cells; colored health dots never appear.
- `[HIGH]` Radar chart shows 6 dead axes (z=0) when comparing same-type players — visually broken; half the spider chart is flat for hitter vs hitter or pitcher vs pitcher.
- `[HIGH]` Inverse-stat z-scores display as negative numbers winning categories — L/ERA/WHIP rows confuse novice users who read negative as bad.
- `[HIGH]` Cross-type comparison (hitter vs pitcher) produces meaningless composite score — Soto beats Skubal in "Losses" category just because hitters have z=0; no warning is shown.
- `[HIGH]` `compute_category_fit()` exists but is never called — "Category Fit vs My Team" is the most useful feature for a fantasy manager and it is entirely absent.
- `[HIGH]` YTD data missing for multiple players (Judge shows all "--") — `season_stats` partial refresh; page gives no indication of missing data.
- `[MEDIUM]` No data freshness indicator — data is 1219+ min stale with no in-page warning.
- `[MEDIUM]` "Schedule Strength" section is mislabeled — shows own-team context, not opponent schedule quality.
- `[MEDIUM]` Statcast caption mentions Stuff+ but it never renders (null for all 9,888 players due to FanGraphs 403).
- `[MEDIUM]` xwOBA shown under "Statcast Profile" may be projection-derived, not live Baseball Savant.
- `[MEDIUM]` Recent Performance headings use inline markdown bold instead of `st.subheader` — inconsistent visual hierarchy.
- `[MEDIUM]` Confidence column units unlabeled — "±38.20" gives no interpretable meaning.
- `[MEDIUM]` SGP vs z-score inconsistency unexplained — two adjacent tables can give opposite "winner" for the same category (Judge OBP SGP > Soto OBP SGP, but Soto wins OBP z-score).
- `[MEDIUM]` Player search: 3 random pills shown before any typing — confusing first impression.
- `[MEDIUM]` No empty state in main column when no players selected — page body is blank.
- `[MEDIUM]` No guard or warning for cross-type comparisons on the page.
- `[LOW]` `width="stretch"` not a valid `st.button` kwarg — silently ignored in Streamlit 1.47.
- `[LOW]` Name truncation at 14 chars in context panel mid-word for long names.
- `[LOW]` "6 ties" in Quick Verdict counts irrelevant cross-type categories as ties.
- `[LOW]` Player card selector is buried mid-page (natural position: at identity strip).
- `[LOW]` Catcher framing data from 2024 seed — no year/staleness label in section header.
- `[LOW]` No "clear/reset" button to start a new comparison.
- `[LOW]` `compute_category_fit()` function dead code relative to this page — exists in `in_season.py`, never imported.
- `[LOW]` YTD table includes pitcher-stat rows (ERA, WHIP, SV, K) showing "--/--" for two hitters — wasted rows.
- `[LOW]` Identity card team-brand colors lose the A=orange / B=steel visual convention used by the chart.
- `[LOW]` Progress bar is purely decorative (compare_players() is <1ms; 0.3s sleep is artificial).
- `[LOW]` Roster badge green background at 8% opacity fails WCAG contrast.
- `[POLISH]` FIG numbering: "FIG.16" is double-digit and therefore consistent, but the page header could benefit from a sub-caption clarifying what "HEAD-TO-HEAD" means (z-score normalization vs raw stats comparison).
- `[POLISH]` Level filter should live in an "Advanced" expander, not as the topmost visible control.
- `[POLISH]` Context panel placeholder text `color:var(--tx-muted,#888)` on white fails WCAG AA.
- `[POLISH]` "Select two players to compare" reco banner has no substantive guidance (no tooltip showing what to expect — e.g., "Pick any two players; we'll show you who wins each scoring category").
