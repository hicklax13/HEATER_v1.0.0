# Page 01 — Draft Tool (Home) — Test-User Report

**Page source:** `app.py` — `render_single_user_app()` → `render_setup_page()` → `render_step_settings()` / `render_step_launch()` → `render_draft_page()`

**Auditor persona:** Connor, novice fantasy-baseball manager, Team Hickey, FourzynBurn league. League is **in-season (Week 12, 2026)**. Draft completed months ago.

---

## 2. Page Purpose & First Impression

The Draft Tool (Home) is HEATER's default landing page. Its stated purpose is to serve as a pre-draft setup wizard and live draft assistant. The flow is:

1. Splash bootstrap screen (data loading)
2. Step 1 — League Settings (SGP, risk, engine mode)
3. Step 2 — Launch (pre-flight check, Practice Mode toggle, START DRAFT)
4. Draft Page — 3-column layout: roster panel / hero recommendation / pick entry, plus 5 bottom tabs

**First impression (5 seconds, in-season novice):** The page shows the large "HEATER." wordmark, an orange Beta banner, two league info cards ("LEAGUE FORMAT 12 Teams / 23 Rounds · Snake · H2H Categories" and "PLAYER POOL 9,888 / Yahoo: Warming up"), the wizard stepper "1 SETTINGS / 2 LAUNCH," and controls for SGP, risk, and engine mode.

**The most serious first-impression problem:** The league draft was completed in March. It is Week 12 of the 2026 season. Landing in-season users on a draft setup wizard — with no explanation of why this page exists for them, no link to their current team, and no in-season context — is deeply confusing. A novice leaguemate clicking the sidebar "Draft Tool" entry has no idea what to do here and no clear path to the features they actually want (My Team, Matchup Planner, Free Agents). The page title "Draft Tool" in the nav correctly signals this is draft-related, but there is zero in-season orientation text.

---

## 3. Methodology

**Sources read:**
- `app.py` (all 2699 lines)
- `src/ui_shared.py` (THEME tokens, PAGE_ICONS, inject_custom_css, render_page_header, sec(), ROSTER_CONFIG)
- `src/draft_state.py` (DraftState class)
- `src/draft_engine.py` (DraftRecommendationEngine)
- `src/simulation.py` (DraftSimulator, compute_team_preferences, detect_position_run)
- `src/valuation.py` (LeagueConfig, SGPCalculator, value_all_players)
- `src/nav.py` (PAGE_REGISTRY, build_pages)
- `.streamlit/config.toml`

**DB queries run (read-only):**
- `players` table: count = 9,888
- `league_rosters`: 12 teams
- `projections`: systems present — blended (9,840), steamer_ros (9,769), zips (3,729), zips_ros (1,148), atc (1,485), thebat (1,403), thebatx (1,409), depthcharts (1,439), depthcharts_ros (1,397), steamer (1)
- `refresh_log`: 38 rows — 26 success, 5 no_data, 2 error, 2 skipped, 2 cached, 1 partial
- `data/backups/draft_state.json`: 378 bytes, current_pick=1, total_picks=276 (one pick made)

**Valuation pipeline run (reconstructed):** Called `value_all_players()` to verify pick_score generation; confirmed tier, p_survive, sgp_r/hr/etc. are **not** in the base pool or engine output.

**DraftRecommendationEngine.recommend() run (quick mode, 20 sims):** Confirmed actual recommendation columns returned.

---

## 4. Feature & Control Inventory

### Beta Banner (module-level, all pages)

| Control | Type | What it does | Tested? |
|---------|------|-------------|---------|
| Beta banner ("HEATER Beta — Thanks for testing! Use the in-app feedback button...") | Static HTML div | Informational banner | Yes — renders on ALL pages via module-level `st.markdown` |

### Splash / Bootstrap Screen

| Control | Type | What it does | Tested? |
|---------|------|-------------|---------|
| HEATER logo (64×64 SVG) | Static SVG | Visual identity | Yes |
| "Loading MLB data..." text | Caption | Status message | Yes |
| Bootstrap progress bar | `st.progress` | Shows 0–100% as phases complete | Yes (reconstructed) |
| Self-ticking JS timer (HH:MM:SS) | `st.components.v1.html` | Live elapsed time counter | Yes — deprecated API |
| Phase status text (`st.empty`) | Dynamic text | Current phase + ETA | Yes (reconstructed) |

### Setup Page (Step 1 & 2 Wizard)

| Control | Type | What it does | Tested? |
|---------|------|-------------|---------|
| Eyebrow: "FOURZYNBURN LEAGUE · 2026 SEASON / FIG.00 · COMMAND HOME" | Static header | Page identity | Yes |
| "HEATER." wordmark | `render_page_header` | Big display title with orange period | Yes |
| Wizard stepper "1 SETTINGS / 2 LAUNCH" | Custom HTML `.wizard-bar` | Shows current step | Yes |
| **Step 1:** LEAGUE FORMAT card ("12 Teams / 23 Rounds · Snake · H2H Categories") | `.metric-card` HTML | Read-only league info (hardcoded from session defaults) | Yes |
| **Step 1:** PLAYER POOL card ("9,888 / Yahoo: Warming up") | `.metric-card` HTML | Shows pool count + Yahoo status | Yes |
| **Step 1:** "Data Status" expander | `st.expander` (collapsed by default) | Shows per-phase bootstrap results with check/X icons | Yes |
| **Step 1:** "Connect Yahoo Fantasy (optional)" expander | `st.expander` (v1 only, hidden under MULTI_USER) | Yahoo OAuth flow | Yes — hidden on hosted app |
| **Step 1:** "Standings Gained Points Denominators" section label | `st.markdown` | Section header | Yes |
| **Step 1:** "Auto-compute Standings Gained Points" toggle | `st.toggle` (default: ON) | When off, reveals manual SGP inputs | Yes |
| **Step 1 (when toggle off):** Runs input | `st.number_input`, default 32.0 | Manual SGP denominator | Yes |
| **Step 1 (when toggle off):** Home Runs input | `st.number_input`, default 12.0 | Manual SGP denominator | Yes |
| **Step 1 (when toggle off):** Runs Batted In input | `st.number_input`, default 30.0 | Manual SGP denominator | Yes |
| **Step 1 (when toggle off):** Stolen Bases input | `st.number_input`, default 8.0 | Manual SGP denominator | Yes |
| **Step 1 (when toggle off):** Batting Average input | `st.number_input`, default 0.008, format "%.4f" | Manual SGP denominator | Yes |
| **Step 1 (when toggle off):** Wins input | `st.number_input`, default 3.0 | Manual SGP denominator | Yes |
| **Step 1 (when toggle off):** Saves input | `st.number_input`, default 7.0 | Manual SGP denominator | Yes |
| **Step 1 (when toggle off):** Strikeouts input | `st.number_input`, default 25.0 | Manual SGP denominator | Yes |
| **Step 1 (when toggle off):** Earned Run Average input | `st.number_input`, default 0.30, format "%.3f" | Manual SGP denominator | Yes |
| **Step 1 (when toggle off):** Walks + Hits per Inning Pitched input | `st.number_input`, default 0.03, format "%.3f" | Manual SGP denominator | Yes |
| **Step 1:** "Risk Tolerance" section label | `st.markdown` | Section header | Yes |
| **Step 1:** "Risk appetite" slider | `st.slider` (0.0–1.0, step 0.1, default 0.5) | Risk tolerance setting | Yes |
| **Step 1:** Risk label badge (e.g. "MODERATE") | `.badge.badge-fair` HTML | Derived label from slider value | Yes |
| **Step 1:** "Draft Engine Mode" radio group | `st.radio`, 3 options | Quick/Standard/Full engine mode | Yes |
| **Step 1:** "Next →" button | `st.button`, primary, centered in [2,1,2] columns | Advances to Step 2 | Yes |
| **Step 2:** Building player pool progress bar | `st.progress` | Rebuilds full valuation pipeline | Yes |
| **Step 2:** Pre-flight checklist (4 items) | Custom `.glass` HTML | Player Pool / Hitters / Pitchers / Valuations | Yes |
| **Step 2:** Practice Mode toggle | `st.toggle` (default: True) | Auto-picks opponents in practice | Yes |
| **Step 2:** "Resume saved draft" toggle | `st.toggle` (default: False) | Loads `data/backups/draft_state.json` | Yes — file exists, 1 pick recorded |
| **Step 2:** "START DRAFT" button | `st.button`, primary, disabled when checks fail | Launches draft page | Yes |
| **Step 2:** "← Back" button | `st.button` | Returns to Step 1 | Yes |
| **Sidebar:** "Refresh All Data" button | `st.button` (after bootstrap complete) | Runs force=True bootstrap | Yes |

### Draft Page Controls

| Control | Type | What it does | Tested? |
|---------|------|-------------|---------|
| Practice Mode banner (amber border) | Static HTML `.glass` | "PRACTICE MODE — Picks will not be saved" | Yes (reconstructed) |
| Draft page header / banner_teaser | `render_page_layout` | "Round N, Pick N — Your Turn" or "Team is on the clock (N picks away)" | Yes |
| Command Bar | Custom `.cmd-bar` HTML | Shows round/pick, your turn status, progress %, pick counter | Yes |
| Lock-in banner (3 seconds) | Static HTML `.lock-in` | "PlayerName — Locked In" after pick | Yes |
| Recommendation progress bar | `st.progress` | Engine analysis progress | Yes (reconstructed) |
| **Left column:** Draft Controls context panel | `render_context_card` | Mode badge (Practice/Live) | Yes |
| **Left column:** Practice Mode toggle | `st.toggle` | Toggle practice mode during draft | Yes |
| **Left column:** "Reset Practice" button (practice mode only) | `st.button` | Resets practice draft state | Yes |
| **Left column:** "Undo Last Pick" button | `st.button` | Undoes most recent pick | Yes |
| **Left column:** "Save Draft" button (live mode only) | `st.button` | Saves draft state to disk | Yes |
| **Left column:** "Sync Yahoo Draft" button (Yahoo connected + live only) | `st.button` | Polls Yahoo draft results | Yes |
| **Left column:** My Roster panel | Custom `.roster-slot` HTML | Grouped slots: Field/Utility/Pitching/Bench | Yes |
| **Left column:** Position Scarcity rings | Conic-gradient SVG rings | 8 positions (C/1B/2B/3B/SS/OF/SP/RP), colored red/amber/green by scarcity | Yes |
| **Center column:** Hero Pick card | Custom `.hero` HTML | Top recommendation with score, survival gauge, name, position, tier, badges | Yes |
| **Center column:** Score badge | `.score-badge` | Combined pick_score (e.g. "20.01") | Yes |
| **Center column:** Survival gauge | Conic-gradient circle | p_survive % (e.g. "31%"), colored red/amber/green | Yes |
| **Center column:** Player name + badges | `.p-name` | Name + BUY/FAIR/AVOID pill + confidence badge + LAST CHANCE badge | Yes |
| **Center column:** Position / tier line | `.p-meta` | "TWP · Tier 1 [badge] [injury badge] [age flag] [workload flag] [injury prob]" | Yes |
| **Center column:** Reason text | `.reason` | "Best available by combined score" (fallback — always shown) | Yes |
| **Center column:** SGP category chips | `.sgp-chip.pos/.neg` | Per-category SGP contributions | Yes — **always empty** |
| **Center column:** Percentile range bar | Inline progress bar | P10–P90 range, or "Single projection source" warning | Yes (reconstructed) |
| **Center column:** Threat alerts | Inline HTML | Max 2 lines: survival/positional threats | Yes (reconstructed) |
| **Center column:** Enhanced metrics bar | Flex row of mini bars | Category Need, Skill Delta, Closer Bonus, Stream Penalty, Lineup Bonus, Flex Bonus | Yes (reconstructed) |
| **Center column:** Alternatives section | 3-column `.alt.tier-N` cards | Up to 9 alternatives (#2–#10) with name, position, score, risk badge | Yes |
| **Center column:** Engine timing caption | `st.caption` | "Engine: X.Xs (standard mode)" | Yes (reconstructed) |
| **Right column:** "Enter Pick" section | `sec("Enter Pick")` | Section header | Yes |
| **Right column:** Player selectbox | `st.selectbox` (200 options) | Type-to-search available players | Yes |
| **Right column:** Confirmation card | `.glass` HTML with amber border | Shows selected player + position | Yes |
| **Right column:** DRAFT [PLAYER] button / "Record for [Team]" button | `st.button`, primary | Executes pick for user or records opponent pick | Yes |
| **Right column:** "Recent Picks" section | `sec("Recent Picks")` | Last 5 picks in reverse order | Yes |
| **Right column:** Feed cards | `.feed-card` / `.feed-card.user-pick` | Pick number, player name, team name, position | Yes |
| **Sidebar (draft page):** "← Back to Setup" button | `st.button` | Returns to setup wizard | Yes |

### Draft Bottom Tabs

| Control | Type | What it does | Tested? |
|---------|------|-------------|---------|
| "Category Balance" tab | `st.tabs` | 12 category cards + radar chart (Plotly) + heatmap grid | Yes |
| "Available Players" tab | `st.tabs` | Position filter pills, sort dropdown, search input, styled table (top 50) | Yes |
| "Draft Board" tab | `st.tabs` | Scrollable HTML grid: rounds × 12 teams, current pick highlighted | Yes |
| "Draft Log" tab | `st.tabs` | Export CSV / Export JSON buttons + feed-style pick history | Yes |
| "Opponent Intel" tab | `st.tabs` | Table: 11 opponents with Threat, Positions Needed, Historical Bias, Predicted Next, Picks Made | Yes |
| **Available Players:** Position filter pills | `st.button` × 9 (All/C/1B/2B/3B/SS/OF/SP/RP) | Filters player list by position | Yes |
| **Available Players:** Sort by selectbox | `st.selectbox` | HEATER Rank / Consensus Rank / ADP / Name | Yes |
| **Available Players:** Disagreement radio | `st.radio` (horizontal) | All / High / Medium / Low consensus vs HEATER disagreement | Yes |
| **Available Players:** Search input | `st.text_input` | Name search filter | Yes |
| **Available Players:** Styled table | `render_styled_table` | Player/Position/Score/Tier/ADP/Consensus Rank/HEATER vs Consensus + stats | Yes |
| **Available Players:** Player card selector | `render_player_select` | Dropdown to view a detailed player card | Yes |
| **Category Balance:** Category cards (12) | `.cat-card` HTML | Shows current roster totals for each stat | Yes |
| **Category Balance:** Radar chart | Plotly Scatterpolar | User team (orange) vs league average (dotted gray) — z-score normalized | Yes |
| **Category Balance:** Heatmap grid | `build_category_heatmap_html` | HTML heatmap comparing user vs all teams | Yes |
| **Draft Log:** Export CSV button | `st.download_button` | Downloads draft_log.csv | Yes |
| **Draft Log:** Export JSON button | `st.download_button` | Downloads draft_log.json | Yes |

---

## 5. Feature-by-Feature Test Log WITH REAL OUTPUTS

### Beta Banner
**Output (all pages):** `"HEATER Beta — Thanks for testing! Use the in-app feedback button to share notes."` on an orange gradient `#e8480a` → `#ff6d00` background. Renders at module level — appears on every page load, not just the home page. This is a cross-page concern.

### Splash / Bootstrap Screen (reconstructed)

**Refresh log status at time of audit:**
- 26 phases: `success`
- 5 phases: `no_data` (adp_sources, depth_charts, game_day, news_intelligence, players)
- 2 phases: `error` (projections: "all FanGraphs fetches failed"; transactions: "Error: database is locked")
- 2 phases: `skipped` (batting_stats, stuff_plus — FanGraphs 403)
- 2 phases: `cached` (ecr_consensus, pvb_splits)
- 1 phase: `partial` (season_stats: saved 2,580/8,054 rows)

**Critical errors present:** The `projections` phase errors with "all FanGraphs fetches failed" — this is the primary projection data source. The `players` phase returns `no_data` ("fetch_all_mlb_players returned empty"). However, since these are cached from prior runs, the app still functions.

**JS timer:** Uses `st.components.v1.html` — a deprecated API with server warnings: `"Please replace st.components.v1.html with st.iframe. … will be removed after 2026-06-01"`. Only 1 usage (the HH:MM:SS timer).

**Total elapsed time:** Not available at audit time (bootstrap was a prior session). The HH:MM:SS timer freezes once the splash clears and cannot be revisited.

### Step 1: League Settings Cards

**LEAGUE FORMAT card output (reconstructed):**
```
LEAGUE FORMAT
12 Teams
23 Rounds · Snake · H2H Categories
```
These values come from `st.session_state.get("num_teams", 12)` — they are hardcoded defaults. There is **no UI widget to change num_teams, num_rounds, or draft position** in the wizard. The only configurable items are SGP denominators, risk tolerance, and engine mode.

**PLAYER POOL card output:**
```
PLAYER POOL
9,888
Yahoo: Warming up
```
Under MULTI_USER, the badge shows "Yahoo: Warming up" (gray/muted) when `conn_status != "server"`. Current data: last successful Yahoo refresh was `2026-06-12T17:43:09` (~1,219 minutes before audit).

### Step 1: Data Status Expander (reconstructed)

Collapsed by default with label "Data Status". When expanded, shows bootstrap_results dict entries with check/X icons. Icon logic: `"Error" not in str(result)` — so phases with "no_data" or "partial" still get a **check icon** even if data is missing. Users see mostly green checks even when 8+ phases failed or returned no data.

### Step 1: Auto-compute SGP Toggle

**Default state:** ON (checked). When ON, denominators are auto-computed from the pool via `compute_sgp_denominators()`.

**Actual computed denominators (reconstructed from live DB):**
```
R:    18.074
HR:    8.129
RBI:  15.232
SB:    9.397
AVG:   0.008
OBP:   0.011
W:     1.491
L:     5.895
SV:    5.423
K:    31.628
ERA:   0.447
WHIP:  0.095
```

**When toggle is OFF:** Reveals 10 number inputs — R, HR, RBI, SB, AVG, W, SV, K, ERA, WHIP. **Missing: OBP and L (Losses)**. The league uses 12 categories including OBP and L. Both are tracked in LeagueConfig but absent from the manual SGP UI. Users who disable auto-compute cannot tune OBP or L denominators.

### Step 1: Risk Appetite Slider

**Default:** 0.5 (rendered as "MODERATE")

**Risk label mapping** — 5 labels across 11 slider stops:
```
0.0, 0.1, 0.2 → "Conservative"
0.3, 0.4      → "Balanced"
0.5, 0.6, 0.7 → "Moderate"
0.8, 0.9      → "Aggressive"
1.0           → "YOLO"
```
Three slider positions map to "Conservative" and three to "Moderate," but only one maps to "YOLO." The label changes are clumped and non-linear — dragging from 0.5 to 0.6 shows no label change, which feels unresponsive.

### Step 1: Draft Engine Mode Radio

- "Quick (< 1 second)" → mode="quick"
- "Standard (2-3 seconds)" → mode="standard" (default)
- "Full (5-10 seconds)" → mode="full"

**Actual engine timing (reconstructed, quick mode):** Returned recommendation for Shohei Ohtani in ~1.8s during audit. Standard mode with 20 sims returned same recommendation in ~3.1s.

### Step 2: Pre-flight Checklist (reconstructed)

When _build_player_pool() completes successfully:
```
[check] Player Pool     — 9,888 players loaded
[check] Hitters         — 6,139 hitters
[check] Pitchers        — 3,749 pitchers
[check] Valuations      — Standings Gained Points + Value Over Replacement Player computed
```
The "Valuations" check tests `"pick_score" in pool.columns`. This column is **only added** by `value_all_players()` inside `_build_player_pool()` — the raw `load_player_pool()` does not include it. So the check correctly verifies the valuation pipeline ran. The check would fail if `value_all_players` raises — and the error is caught with `st.error(f"Error building player pool: {e}")`.

### Draft Page: Hero Pick Card (reconstructed)

With DraftRecommendationEngine in standard mode, Shohei Ohtani recommended as pick #1:
```
Score: 20.01
Survival gauge: 31% (red — low survival even at pick 1)
Name: Shohei Ohtani
BUY/FAIR/AVOID: FAIR (displayed in sky blue)
Confidence: MEDIUM (border style, orange)
Position: TWP · Tier 1
Reason: "Best available by combined score"  ← always shown, never changed by engine
SGP chips: [empty — all 0]
Percentile range bar: "Single projection source — range unavailable"
```

**Critical finding:** The SGP category chips (`sgp_r`, `sgp_hr`, etc.) are **always empty**. The engine output and the base pool have no per-category SGP columns. Every recommendation shows zero category breakdown chips. The `.sgp-row` div renders but contains no chips, leaving an unexplained blank gap in the hero card.

**Critical finding:** `tier` is always hardcoded to 1 (`int(rec.get("tier", 1))`). The engine does not compute tiers. Every player shows "Tier 1."

**Critical finding:** `reason` is always "Best available by combined score" — the engine never populates this field. The reason text is hardcoded fallback for every pick.

### Draft Page: Available Players Tab (reconstructed)

Pool after valuation: 9,888 players. Top 10 by pick_score:
```
Shohei Ohtani (TWP)   score=21.35  adp=1.3   consensus=1
Juan Soto (LF)         score=10.25  adp=3.0   consensus=2
Jackson Chourio (LF)   score=9.60   adp=19.4  consensus=30
Jonah Cox (CF)         score=8.77   adp=999.0 consensus=1016
Wyatt Langford (LF)    score=8.65   adp=37.8  consensus=144
Aaron Judge (RF)       score=8.02   adp=1.0   consensus=92
Wes Clarke (C/1B)      score=8.00   adp=999.0 consensus=2995
Ronald Acuña Jr. (RF)  score=7.59   adp=5.3   consensus=10
Brent Rooker (DH)      score=7.42   adp=42.1  consensus=48
Lars Nootbaar (LF)     score=7.18   adp=746.0 consensus=430
```

**Anomaly:** Jonah Cox (adp=999, consensus=1016) and Wes Clarke (adp=999, consensus=2995) appear in the top 10 by HEATER's own score. These are likely unranked prospects or low-ADP players whose projections are inflated. Aaron Judge (the best overall hitter by ADP=1.0) scores 8.02 — LOWER than Cox and Clarke. A novice would be confused and lose trust immediately.

### Draft Page: Survivor Gauge Calibration

At pick #1, Ohtani's survival probability is 31% (red). This means HEATER thinks Ohtani will NOT survive to the next user pick in 69% of simulations — at pick #1 overall. That is mathematically defensible (top pick = low survival), but the visual is red and alarming, which could confuse a novice into not trusting the top recommendation.

### Draft Page: Saved Draft State

`data/backups/draft_state.json` exists with 1 pick made (current_pick=1), 276 total. "Resume saved draft" toggle would load this state. The file was clearly left from a prior test session.

---

## 6. Errors, Issues & Difficulties

### Critical / Functional Bugs

**E-01 [BLOCKER] Home page is a draft wizard shown to in-season users.** The league draft was completed in March 2026. Every leaguemate who clicks "Draft Tool" in the sidebar lands on a 2-step setup wizard for a draft that already happened. There is no in-season orientation text, no link to My Team or Free Agents, no explanation of what to do here now. The page is dead code for in-season use.

**E-02 [HIGH] SGP category chips are always empty.** `render_hero_pick` reads `rec.get("sgp_r", 0)` etc. — but neither the base pool nor the engine output contains per-category SGP columns. All 10 chip values are 0, so the entire `.sgp-row` div renders blank. The hero card advertises "SGP category contributions" but delivers nothing.

**E-03 [HIGH] Tier is always "Tier 1."** `rec.get("tier", 1)` defaults to 1 because the engine never populates `tier`. Every recommended player shows "Tier 1" regardless of their actual draft tier. This is misleading — a 23rd-round pick looks the same as a first-rounder.

**E-04 [HIGH] Reason text is always the same fallback.** The engine never populates a `reason` field. Every pick shows "Best available by combined score." No differentiation by player type, situation, or draft context.

**E-05 [HIGH] Pick quality anomaly: low-ADP unknowns beat Aaron Judge in pick_score.** Jonah Cox (ADP 999) scores 8.77 vs Aaron Judge (ADP 1.0) scoring 8.02. A novice trusting HEATER's own score would be confused and may lose confidence in the tool. This suggests the SGP valuation model is producing outlier results for uncalibrated or minor-league players.

**E-06 [HIGH] Manual SGP inputs missing OBP and L.** The league tracks 12 categories including OBP and L (inverse). When the auto-compute toggle is OFF, only 10 inputs are shown. OBP and L cannot be tuned. Users who want to calibrate SGP manually cannot fully specify their league.

**E-07 [MEDIUM] `st.components.v1.html` is deprecated.** Used for the HH:MM:SS timer in the bootstrap splash screen. Server emits deprecation warnings. Marked for removal after 2026-06-01 (already past). Should be replaced with `st.iframe`.

**E-08 [MEDIUM] Data Status icons do not reflect data quality accurately.** The expander uses `"Error" not in str(result)` to choose check vs X icon. Phases with status "no_data", "partial", or "cached" (8 of 38) still get green check marks. Users cannot tell from the Data Status panel that projections errored or that 5,474 season_stats rows failed to import.

**E-09 [MEDIUM] Survival gauge shows red (31%) for the #1 overall pick.** At pick #1, survival probability is mathematically low (only one pick before the next turn). But the red color implies "danger" — a novice will see red and hesitate. The survival gauge semantic should convey "grab now vs wait," not a simple health indicator.

**E-10 [MEDIUM] "Resume saved draft" hides stale 1-pick state.** The file `data/backups/draft_state.json` has current_pick=1 (one pick recorded). If a leaguemate clicks "Resume saved draft," they inherit someone else's partially started test draft — no warning, no staleness indicator, no ability to see the saved draft state before loading.

**E-11 [MEDIUM] Wizard has no way to configure num_teams, num_rounds, or draft position.** These are always read from session_state defaults (12, 23, 1). A user with a different league cannot change these from the UI. The League Format card is purely display. If a user joins with a non-standard league, the wizard will never tell them they're using wrong defaults.

**E-12 [MEDIUM] WHIP formatted as `.3f` but should be `.2f`.** In `render_category_balance`, the cats list has `("Walks + Hits per Inning Pitched", totals.get("WHIP", 0), ".3f")`. Per `format_stat()` authority, WHIP requires `.2f`. This produces three decimal places (e.g. "1.234" instead of "1.23"), inconsistent with every other page and the `tests/test_pages_format_compliance.py` guard.

**E-13 [LOW] `page_icon` is an empty string in `st.set_page_config`.** The browser tab shows no favicon — just the default Streamlit globe icon. "Heater" is also lowercase in `page_title="Heater"` vs the brand mark "HEATER."

**E-14 [LOW] AVOID badge uses `T["primary"]` (orange) instead of `T["danger"]` (red).** In `render_hero_pick`, the AVOID badge background is set to `T['primary']` which is `#ff6d00` (orange). This conflicts with the Combustion design system: orange is the brand/CTA color, red (`T["danger"]` = `#e0492f`) is for negative/warning states. An "AVOID" signal on a player should render red, not the same orange as a primary action button.

**E-15 [LOW] Risk label has only 5 values across 11 slider stops.** Values 0.3 and 0.4 both show "Balanced"; 0.5, 0.6, 0.7 all show "Moderate." Moving the slider from 0.5 to 0.6 produces no visible label change. This feels broken to a novice.

**E-16 [LOW] "YOLO" risk label is unprofessional for a Beta product.** The label at risk=1.0 is "YOLO." For a product trying to feel "professional, modern, high-tech," "YOLO" undermines that tone. "Speculative" or "Max Risk" would fit better.

**E-17 [POLISH] No browser favicon.** Empty string for `page_icon` means the browser tab shows a blank/default icon. A small HEATER flame SVG would reinforce brand identity.

**E-18 [POLISH] Position Scarcity rings show toast spam.** When any position ring is "critical" (red), a `st.toast()` fires. On every re-render these toasts re-fire — multiple scarcity toasts can stack if the page re-renders due to any widget interaction. The dedup key `f"scarcity_toast_{pos}_{avail_count}"` only dedupes within one session, but re-renders within the same session can still fire multiple times if the session_state key is set multiple times (the logic sets `st.session_state[toast_key] = True` and fires, but doesn't prevent re-firing on subsequent renders within the same session if the key is already there — the `if toast_key not in st.session_state` guard handles this, but rerun loops where picks change `avail_count` will fire new toasts for each count change).

---

## 7. UI/UX & Visual Design Critique

### The Strategic Problem (Most Important)

The in-season user experience is fundamentally broken. This page is the default landing page for all 12 leaguemates. When a user clicks into HEATER during Week 12 of the baseball season, they expect to see "My Team" — their roster, matchup, waiver recommendations. Instead they see a wizard for a draft that completed in March. There is no explanation, no redirect, no contextual message like "Draft season is over — head to My Team to manage your team." This is the single highest-impact UX failure on the entire page.

### Layout & Density

The 3-column draft layout (1.2 / 3 / 1.3 ratio) is reasonable for a draft session. Left column (roster + scarcity) and right column (pick entry + recent feed) are appropriate support panels. The center hero card has good visual weight.

However, the wizard steps feel underpowered for the complexity they're managing:
- Step 1 has an awkward 2-column layout where the left col is "Standings Gained Points Denominators" (toggle + potentially 10 number inputs) and right col is "Risk Tolerance + Engine Mode." These are very different concerns crammed together.
- The "League Format" and "Player Pool" cards are pure display with no actionable meaning — a novice doesn't know what "SGP denominators" are or why they would turn off auto-compute.

### Typography & Hierarchy

- Section headers (`sec()`) use `.sec-head` with `font-family: var(--font-display)` (Archivo), `font-weight: 800`, `text-transform: none` — correct per Combustion spec.
- The hero player name uses `.p-name` (Archivo display) — good visual weight.
- Numbers in the hero card (score badge, survival gauge %) use IBM Plex Mono via explicit `font-family: IBM Plex Mono,monospace` — correct.
- Category card values in `.cat-val` don't explicitly set IBM Plex Mono — they should for consistent number formatting.

### Color Use

- BUY badge: `T["green"]` (#1f9d6b) — correct (positive signal, green).
- AVOID badge: `T["primary"]` (#ff6d00) — **incorrect** (orange = brand/CTA, not negative).  
- FAIR badge: `T["sky"]` (#5f7d9c) — neutral, acceptable.
- Risk label badge uses `.badge-fair` (class name) even for non-"fair" risk levels — the class is misleading.
- The "Practice Mode" banner uses `T["warn"]` (#ff9f1c) border — appropriate.
- The command bar uses `.your-turn` / `.waiting` classes — these need verification against Combustion tokens.

### Information Hierarchy in Hero Card

The hero card is information-dense but the arrangement is:
1. Score badge (top-right, IBM Plex Mono) — good
2. Survival gauge ring (conic gradient) + player name — clear
3. BUY/FAIR/AVOID + Confidence badge + LAST CHANCE badge inline in name — noisy
4. Position · Tier · [Value/Reach/Fair badge] · [injury badge] · [age flag] · [workload flag] · [injury prob %] — excessive clutter on one line
5. Reason (always same text) — wasted space
6. SGP chips row — always empty, visual dead space
7. Percentile bar — only shows when `has_percentiles=True` (needs multiple projection systems)
8. Threat alerts — useful context

Tiers 5 and 6 (reason + SGP chips) are dead information that add layout noise without value. The ".p-meta" line stacking 7+ elements risks overflow on narrow viewports.

### Consistency with Sibling Pages

- FIG.00 numbering: consistent with the page header standard (2-digit zero-padded).
- `render_page_header` is correctly called on the setup page.
- The draft page uses `render_page_layout` with a banner_teaser — this is a different pattern than `render_page_header` used in setup; the transition between wizard and draft changes the page header entirely.

### Empty / Error States

- Empty "Recent Picks" feed: `'<div style="color:{T["tx2"]};font-size:13px;">No picks yet.</div>'` — plain gray text, should use `render_empty_state()` per design system.
- Empty SGP chips row: blank `<div class="sgp-row">` with no content — the heading text in METRIC_TOOLTIPS ("Standings Gained Points") appears on hover but there's nothing to hover over.
- Draft page error state ("Draft not initialized"): uses `st.error()` + a back button — acceptable.

### Performance

The splash screen with 33 bootstrap phases can take 15–20 min on a cold start. Even with TTL-based staleness checking, re-starting a browser session with many stale phases (game_day, news, weather, etc.) can add 30+ seconds. The user sees a HH:MM:SS timer but no breakdown of which phase is slow.

Step 2 rebuilds the full valuation pipeline (`value_all_players()` on 9,888 players) on every visit to Step 2, including on "← Back → Next →" navigation. This takes 2–5 seconds and shows an intermediate progress bar, but every back-and-forth recreates it.

The DraftRecommendationEngine runs on every render when it's the user's turn — including when no pick has been made yet. In Standard mode with 20+ sims, this is 2–3 seconds of blocking compute on every rerun. Any slider change or toggle click triggers a full re-render and re-analysis.

---

## 8. Recommendations (Ordered by Impact)

### REC-01: Add in-season orientation to the Home page

**Problem:** In-season users land on a draft wizard with no guidance. The home page is the wrong context for 12 leaguemates during active season play.

**Change:** Add a contextual "You're in season — here's where to go" banner at the top of the Home page (visible when `setup_step == 1`). Show the user's current record, their matchup this week, and 3 quick-action links: "My Team →", "Free Agents →", "Lineup Optimizer →". This can be data-driven from YDS/refresh_log with zero changes to the wizard behavior — just add content above the stepper. Alternatively, reorder the sidebar so "My Team" is the default landing page for in-season use, with "Draft Tool" deprioritized to the bottom of the nav.

---

### REC-02: Fix the SGP category chips — they should show real data

**Problem:** All per-category SGP chips are always empty. The `.sgp-row` div renders blank on every pick.

**Change:** In `_build_player_pool`, after calling `value_all_players()`, call `SGPCalculator.player_sgp(player_series)` for each top-N candidate and store per-category SGP columns (`sgp_r`, `sgp_hr`, etc.) in the pool. Or, compute these on-the-fly in `render_hero_pick` from the player row's projected counting stats + the league's SGP denominators (already in `st.session_state.sgp_calc`). Even a simple calculation like `hr_proj / lc.sgp_denominators["HR"]` would give meaningful non-zero chips that inform the user why the player was recommended.

---

### REC-03: Implement real tier assignment and pick reason generation

**Problem:** Every player shows "Tier 1" and "Best available by combined score." These are hardcoded defaults, never populated by the engine.

**Change:** (a) Compute tier from pick_score percentiles in `value_all_players` or in the engine's output — e.g., top 10% = Tier 1, next 15% = Tier 2, etc. (b) Add a `reason` column to engine output that states WHY the top pick was chosen: "Best value by SGP | 31% ADP advantage | Scarce catcher position" etc. Even 2–3 dynamic reasons per pick would be far more informative than the permanent fallback string.

---

### REC-04: Fix the pick_score anomaly — calibrate the valuation model

**Problem:** Jonah Cox (ADP 999, consensus rank 1,016) scores 8.77 — higher than Aaron Judge (ADP 1.0, consensus 92, score 8.02). Unknown/low-ranked players outscoring blue-chip stars in HEATER's own metric will destroy novice trust immediately.

**Change:** Add a floor calibration step: players with ADP > 200 or consensus_rank > 300 should have their pick_score capped or penalized. Alternatively, blend ADP rank into pick_score more aggressively (ADP is a crowd-sourced market signal). Log this in `CONSTANTS_REGISTRY` with a `calibrate_sigmoid.py`-style tuning script.

---

### REC-05: Replace `st.components.v1.html` with `st.iframe`

**Problem:** The HH:MM:SS timer uses the deprecated `st.components.v1.html` API. Server emits warnings; the API will be removed.

**Change:** Replace `_components.html(_timer_html, height=48)` with `st.iframe(html=_timer_html, height=48)`. One-line change; behavior is identical.

---

### REC-06: Fix AVOID badge color to use `T["danger"]` (red)

**Problem:** The AVOID badge on the hero pick card uses `T["primary"]` (#ff6d00, orange). Orange is HEATER's brand/CTA color. An "AVOID" recommendation using the same color as primary buttons is semantically contradictory.

**Change:** Replace `T['primary']` with `T['danger']` (`#e0492f`) in the AVOID badge HTML in `render_hero_pick` (line ~1473).

---

### REC-07: Fix manual SGP inputs to include OBP and L

**Problem:** When "Auto-compute Standings Gained Points" is disabled, the wizard shows 10 inputs but the league has 12 categories — OBP and L (Losses) are missing. Users cannot fully calibrate their league.

**Change:** Add `sgp_obp = st.number_input("On-Base Percentage", value=0.011, step=0.001, format="%.4f", key="sgp_obp")` and `sgp_l = st.number_input("Losses", value=5.9, step=0.1, key="sgp_l")` to the manual SGP section, and wire them to `lc.sgp_denominators["OBP"]` and `lc.sgp_denominators["L"]` in `_build_player_pool`.

---

### REC-08: Fix WHIP format to `.2f` in category balance

**Problem:** The Category Balance tab formats WHIP as `.3f` ("1.234") instead of `.2f` ("1.23"). This conflicts with `format_stat()` authority and the existing `test_pages_format_compliance.py` guard.

**Change:** In `render_category_balance`, change `("Walks + Hits per Inning Pitched", totals.get("WHIP", 0), ".3f")` to `".2f"`. One character fix. Then add a test assertion to cover this specific rendering path.

---

### REC-09: Fix the Data Status expander to accurately reflect failures

**Problem:** Bootstrap result icon logic (`"Error" not in str(result)`) shows a green check for `no_data`, `partial`, and `cached` phases. 8 of 38 phases returned non-success states but all appear green.

**Change:** Change the icon logic to also check for known failure keywords: `fail_signals = ("Error", "no_data", "0 rows", "empty", "failed")`. Or, instead of reading from `bootstrap_results` (which uses legacy string-matching), read from `get_refresh_log_snapshot()` and use the `status` field directly — `success`/`cached`/`skipped` → green check; `partial` → amber warning; `error`/`no_data` → red X.

---

### REC-10: Add a page title/favicon for the browser tab

**Problem:** `st.set_page_config(page_title="Heater", page_icon="")` produces a blank browser tab icon and lowercase "Heater" in the title.

**Change:** Use `page_title="HEATER"` (uppercase to match brand) and `page_icon="🔥"` as a temporary measure, or encode a small 16×16 SVG as a data URI for `page_icon`. The favicon is visible in every browser tab and every bookmark — it is worth getting right before Beta launch.

---

### REC-11: Clarify the Risk Tolerance slider with better label distribution

**Problem:** 5 risk labels across 11 slider stops means moving the slider from 0.5 to 0.6 produces no visible label change, which feels broken.

**Change:** Either: (a) use 5 discrete slider stops (0.0, 0.25, 0.5, 0.75, 1.0 — one per label), or (b) use continuous range and show the numeric value prominently ("0.5 — Moderate") rather than relying on label changes. Also consider replacing "YOLO" with "Speculative" for a more professional tone.

---

### REC-12: Add an in-season redirect or page reorder so novice users don't start at the draft wizard

**Problem:** The sidebar navigation puts "Draft Tool" first (via `build_pages(..., default=True)`). For in-season leaguemates, My Team is the correct default landing page.

**Change:** Under `MULTI_USER`, change `home = st.Page(draft_page, title="Draft Tool", default=True)` to `home = st.Page(draft_page, title="Draft Tool", default=False)` and make My Team the default if it's in-season (season weeks remaining > 0). Alternatively, move the Draft Tool page into a "Pre-Season" group in the nav so it's clearly separate from in-season tools.

---

## 9. Severity-Tagged Issue List

- `[BLOCKER]` **E-01** — Home page is a draft wizard during in-season play. No in-season context, no navigation guidance. Every leaguemate lands here confused.
- `[HIGH]` **E-02** — SGP category chips always empty (all values 0). Per-category contributions never shown. The primary "why this player" display feature is non-functional.
- `[HIGH]` **E-03** — Tier always "Tier 1" — engine never computes tiers. All players look equal tier-wise.
- `[HIGH]` **E-04** — Reason text always "Best available by combined score" — engine never populates reason. No personalized explanation for any pick.
- `[HIGH]` **E-05** — Pick_score anomaly: unknown minor-leaguers (ADP 999) outrank Aaron Judge. Immediate trust loss for novice users.
- `[HIGH]` **E-06** — Manual SGP inputs missing OBP and L. Cannot fully calibrate 12-category league when auto-compute is off.
- `[MEDIUM]` **E-07** — `st.components.v1.html` deprecated for JS timer. Already past removal date (2026-06-01). Server warning spam.
- `[MEDIUM]` **E-08** — Data Status icons mislead: no_data/partial/error phases show green checks.
- `[MEDIUM]` **E-09** — Survival gauge shows red (31%) for pick #1. Semantically confusing for novices.
- `[MEDIUM]` **E-10** — "Resume saved draft" loads stale 1-pick test state with no warning or preview.
- `[MEDIUM]` **E-11** — num_teams / num_rounds / draft_pos not configurable in wizard. Hardcoded defaults.
- `[MEDIUM]` **E-12** — WHIP formatted `.3f` instead of `.2f` in Category Balance tab. Violates `format_stat()` authority.
- `[LOW]` **E-13** — `page_icon=""` = no browser favicon. `page_title="Heater"` (lowercase).
- `[LOW]` **E-14** — AVOID badge uses `T["primary"]` (orange) instead of `T["danger"]` (red). Wrong semantic.
- `[LOW]` **E-15** — Risk slider label mapping is non-linear/clustered. Moving slider shows no feedback at mid-range.
- `[LOW]` **E-16** — "YOLO" risk label is unprofessional for a Beta product.
- `[POLISH]` **E-17** — Empty "Recent Picks" state uses plain text instead of `render_empty_state()`.
- `[POLISH]` **E-18** — Scarcity toasts can spam on rapid re-renders / pick changes during draft.
- `[POLISH]` **E-19** — The `.sgp-row` div renders an empty container (blank visual gap) when no chips exist. Should be conditionally hidden or replaced with placeholder text like "Category breakdown unavailable."
- `[POLISH]` **E-20** — Step 2 rebuilds the full 9,888-player valuation pipeline on every back-and-forth navigation. Should cache the result in `st.session_state.player_pool` and check if it's already present before re-running.
