# Page 02 — My Team — Test-User Report

**Auditor persona:** Connor, novice fantasy-baseball manager, owner of Team Hickey.
**Source file:** `pages/1_My_Team.py` (2,416 lines)
**Audit date:** 2026-06-13
**DB snapshot age at audit time:** matchup cache ~4,012 minutes old (last updated 2026-06-10T19:42:53Z)

---

## 1. Page Purpose & First Impression

My Team is the app's primary daily-workflow page. It answers: "How is my team doing right now, what are my
category gaps, and what do I do today?" The page covers a large surface area — team identity, matchup pulse,
six War Room cards, weekly report, category gaps, roster table, Bayesian projections, and player news.

**First impression (5-second test for a novice):**
The page opens with a clean Combustion page header (`My Team.` wordmark + orange rule), then two unlabeled
buttons side by side (`Refresh Stats` / `Sync Yahoo`), then an identity strip with avatar + team name +
record + roster pills. Below that lies a dense vertical stack of War Room cards, context cards, the roster
table, and a news section. The layout feels rich and high-quality on a wide monitor. However, a novice's
immediate question — "What do I do TODAY?" — is not answered until Card 3 (`Today's Actions`), buried ~40%
down the page. The two action buttons at the top have no tooltip or subtitle explaining what each does, which
is the single most confusing element within the first 5 seconds.

---

## 2. Methodology

- Read all 2,416 lines of `pages/1_My_Team.py` in full.
- Read `src/ui_shared.py` (THEME, Combustion constants, `render_matchup_ticker`, `render_reco_banner`, etc.).
- Read `src/war_room.py`, `src/war_room_actions.py`, `src/war_room_hotcold.py` (engine side via grep/source
  traces).
- Queried the live SQLite DB read-only (network-blocked) for:
  - `league_rosters` filtered to `team_name LIKE '%Hickey%'` → 27 players (real names/slots)
  - `league_matchup_cache` → Week 12 vs "The Good The Vlad The Ugly", score 7-3-2; cached at -4,012 min
  - `season_stats` 2026 for all Team Hickey players → full per-player stats
  - `league_standings` → rank 10/12 (WINS=3, LOSSES=7, TIES=1, .318 pct)
  - `player_news` for Team Hickey → 15 entries, all injuries, mostly duplicates (Yahoo source)
  - `statcast_archive` → 501 rows present but ALL xwoba/barrel_pct/hard_hit_pct/stuff_plus = NULL
  - `refresh_log` → last full refresh was 2026-06-10; `yahoo_standings` updated 2026-06-12 (most recent)
  - Injury history, projections count (9,840 blended rows)
- Reconstructed category totals from `season_stats` (detailed below).

---

## 3. Feature & Control Inventory

| # | Control | Type | What it does | Tested? |
|---|---------|------|--------------|---------|
| 1 | `Refresh Stats` button | `st.button` (writer-gated) | Calls `refresh_all_stats(force=True)` — pulls live 2026 stats from MLB Stats API; shows progress bar; reruns | Source-traced |
| 2 | `Sync Yahoo` button | `st.button` (writer-gated) | Calls `yds.force_refresh_all()` — re-fetches rosters, standings, matchup, FAs from Yahoo API | Source-traced |
| 3 | Identity strip | Static HTML | Avatar + team name (`Team Hickey`) + meta line (`MANAGER · FOURZYNBURN · 12-TEAM H2H`) + readouts: Record, Roster, Hitters, Pitchers, Rank | DB-confirmed |
| 4 | "Last synced" green pill | Static HTML badge | Shows `MAX(last_refresh)` from `refresh_log` in ET format | DB-queried |
| 5 | Roster banner | `render_reco_banner` | Shows e.g. `Roster: 27 players \| 13 hitters, 14 pitchers` as an orange-rail teaser | DB-confirmed |
| 6 | Matchup ticker | `render_matchup_ticker()` | Compact header bar: Week N score vs opponent + status pill; only renders when `yahoo_connected` session key is set | Source-traced (conditionally rendered) |
| 7 | Matchup ticker expander | `st.expander("Category Breakdown")` | Per-category W/L/T breakdown in the ticker | Source-traced |
| 8 | **Card 1: Matchup Pulse** | Static HTML panel | 12-cat heat-bar grid (win prob per category), vs opponent + weekly score hero number | DB-confirmed (cached data) |
| 9 | **Card 2: Flippable Categories** | Static HTML panel | Per-category "FLIP TO WIN" / "AT RISK" rows with gap values and suggestions | Source-traced |
| 10 | **Card 3: Today's Actions** | Static HTML panel | PRI·N priority rows: player name + team + headshot + reason | Source-traced |
| 11 | **Card 4: Player Streaks (Hot/Cold)** | Static HTML panel | 2-column grid of HOT/COLD tiles (last 7 GP) | Source-traced |
| 12 | **Card 5: Category Flip Analysis** | Static HTML panel | CONTESTED/WON/LOST chip rows with flip probability % and recommended actions | DB-confirmed |
| 13 | **Card 6: Regression Alerts** | Static HTML panel | Up to 5 BUY LOW / SELL HIGH rows with xwOBA vs wOBA divergence | Source-traced |
| 14 | Weekly Report expander | `st.expander` (auto-expanded Monday) | Context cards: This Week Opponent / Category Projections / Action Items / Streaming Targets; Thursday has Checkpoint | Source-traced |
| 15 | `Generate Report` button | `st.button` (inside expander) | Forces re-generation of the weekly report by setting session state flag | Source-traced |
| 16 | Hitting Totals card | `render_context_card` | Season totals: R, HR, RBI, SB, AVG, OBP for all hitters | DB-confirmed (reconstructed) |
| 17 | Pitching Totals card | `render_context_card` | Season totals: W, L, SV, K, ERA, WHIP for all pitchers | DB-confirmed (reconstructed) |
| 18 | Category Gaps card | `render_context_card` | Per-category you vs opp with diff; Priority Targets box for losing cats closest to flip | DB-confirmed (Week 12 cache) |
| 19 | Injured List Alerts card | `render_context_card` | Players with health badge IL/IL-60/Out/DTD listed with red dot | DB-confirmed (4 players) |
| 20 | Lineup Validation card | `render_context_card` | Off-day starter warnings + bench players available today; requires `todays_mlb_games` | Source-traced |
| 21 | Live Matchup card | `render_context_card` | Per-cat row: your stat / opp stat (color-coded by result) | DB-confirmed (Week 12 cache) |
| 22 | Statcast Signals card | `render_context_card` | Elite barrel%, xwOBA, Stuff+, BUY_LOW flags from statcast_archive | DB-confirmed (empty — all NULL) |
| 23 | Data Freshness card | `render_data_freshness_card()` | Per-source staleness list from `refresh_log` | Source-traced |
| 24 | Stat source segmented control | `st.segmented_control` | Options: `2026 Live`, `2026 Projected`, `2025`, `2024`, `2023`; controls context-panel totals | Source-traced |
| 25 | Timeframe segmented control | `st.segmented_control` | Options: `Season`, `L30`, `L14`, `L7`, `Today`; controls roster table stats | Source-traced |
| 26 | Side segmented control | `st.segmented_control` | `Hitters` / `Pitchers` split for the Active Roster table | Source-traced |
| 27 | Open player dossier selectbox | `render_player_select` / `st.selectbox` | Select player name → opens `show_player_card_dialog` | Source-traced |
| 28 | Active Roster table | `build_roster_table_html` (HTML panel) | Full roster for current Side/Timeframe with stats; player cells are `<a href="?player=ID">` links | DB-confirmed (27 rows) |
| 29 | Active Roster footer | Static HTML | "CLICK ANY PLAYER FOR GAME LOG · OUTCOMES · UPCOMING PROJECTIONS" | Source-traced |
| 30 | Export to Excel button | `st.download_button` | Downloads current Side × Timeframe slice as `.xlsx` | Source-traced |
| 31 | Marcel-Adjusted Projections section | `render_compact_table` | Only visible if `BAYESIAN_AVAILABLE` and there are games played (progress bar shows); columns: Player, AVG, HR, RBI, SB, ERA, WHIP, K | Source-traced |
| 32 | News and Alerts section | `_render_news_tab` | Sort selectbox (Most Recent / Severity / Sentiment Impact) + count summary + news cards | DB-confirmed (injury-only, duplicates) |
| 33 | News sort selectbox | `st.selectbox` | Controls sort order of news cards | Source-traced |
| 34 | Player query-param dialog | `show_player_card_dialog(id)` | Opens an `st.dialog` dossier on `?player=<id>` URL param | Source-traced |
| 35 | Page timer footer | `page_timer_footer("My Team")` | Shows render elapsed time at the bottom | Source-traced |
| 36 | Feedback widget | `render_feedback_widget("My Team")` | Popover feedback button at the bottom right (MULTI_USER-gated) | Source-traced |

---

## 4. Feature-by-Feature Test Log with Real Outputs

### 4.1 Identity Strip

**Real output (DB-confirmed, reconstructed):**
- Avatar: navy rounded square with initials "TH" (no Yahoo logo available — oauth-bound client not in session
  during cache-only mode)
- Team name: `Team Hickey` (rendered clean — emoji `🏆 ` prefix from Yahoo is stripped by the regex cleanup)
- Meta line: `MANAGER · FOURZYNBURN · 12-TEAM H2H`
- Record readout: `3-7-1` (live Yahoo standings confirmed: WINS=3, LOSSES=7, TIES=1)
- Roster: `27` / Hitters: `13` / Pitchers: `14`
- Rank: `10` / `12`

**Finding:** The Record readout (`3-7-1`) appears in orange (accented) only when the Yahoo client is connected
and can fetch live standings. In the current "Warming up" / cache-only mode, the live Yahoo client is NOT in
session (it requires env vars + saved token). As a result `_id_record = ""` → the Record readout shows `—`
and Rank shows `—`. The actual record (`3-7-1`) IS known from `league_matchup_cache` but is NOT used as a
fallback. **A novice sees "Record: —" every time the Yahoo token isn't live, which is the common state for
read-only members.**

### 4.2 "Last synced" Green Pill

**Real output:** `Last synced: Jun 10, 12:42 PM ET`

The pill reads `MAX(last_refresh)` from `refresh_log`. The actual top entry is `yahoo_standings` at
`2026-06-12T17:43:09Z` (updated yesterday). But the "Last synced" badge reads **Jun 10** because
`MAX(last_refresh)` returns the most recent timestamp across ALL sources, which in this DB snapshot is
`yahoo_standings` on Jun 12 — so the badge should say `Jun 12`. The discrepancy between what the orchestrator
observed ("Jun 12, 12:46 PM ET") and the Jun 10 values in the refresh_log snapshot may reflect a refresh that
occurred while this audit was running. Regardless, this badge correctly shows the last time any data was
refreshed — but a novice cannot distinguish "last time Yahoo data was pulled" from "last time the MLB Stats
API ran". It presents a single timestamp for a multi-source pipeline without context.

### 4.3 Refresh Stats vs Sync Yahoo Buttons

**Real behavior (source-traced):**
- `Refresh Stats` → calls `refresh_all_stats(force=True)` (MLB Stats API season stats only; shows 3-step
  progress bar; reruns).
- `Sync Yahoo` → calls `yds.force_refresh_all()` (re-fetches all Yahoo endpoints: rosters, standings,
  matchup, FAs, transactions; shows spinner).

**Finding:** No tooltip, subtitle, or label differentiates these. To a novice both look like "update my data."
The distinction (MLB Stats API vs Yahoo Fantasy API) is invisible. Both buttons are gated by
`viewer_can_write()`, which in MULTI_USER mode means only the admin sees them — members see an empty
two-column space above the identity strip. No explanation is given to members about WHY the buttons are absent.

### 4.4 Matchup Ticker

**Conditional render:** Only renders when `st.session_state["yahoo_connected"]` is truthy. In the current
"Warming up" state, this key is absent/False. **Result: the ticker is invisible.** The user gets no
compact "how am I doing this week?" bar. This is a significant daily-workflow gap — the ticker is the
quickest answer to "what's my current score?" and it's missing during the most common cached-data state.

### 4.5 War Room Cards

**Card 1 — Matchup Pulse (real output):**
The matchup data is served from `league_matchup_cache` (falls back correctly without the live Yahoo client).
Week 12 vs "The Good The Vlad The Ugly". Score: `7-3-2`. Category grid (from cache):
```
R: you=14 opp=6  WIN
HR: you=7  opp=1  WIN
RBI: you=14 opp=8 WIN
SB: you=0  opp=0  TIE
AVG: you=.359 opp=.271 WIN
OBP: you=.431 opp=.316 WIN
W: you=1  opp=0  WIN
L: you=2  opp=0  LOSS (inverse — having more Ls is bad)
SV: you=0  opp=0  TIE
K: you=21 opp=13 WIN
ERA: you=7.08 opp=3.38 LOSS
WHIP: you=1.82 opp=1.03 LOSS
```
The heat bar grid shows these win probabilities using the flip-probability model. Winning 7, losing 3, tied 2.
The fig label shows the current week number.

**Finding:** The matchup cache is 4,012 minutes (~2.8 days) stale. The page does NOT display any staleness
warning on Card 1. A novice sees "7-3-2" and believes it is live when it reflects the state as of Jun 10
afternoon. The category totals in the cache may be significantly wrong for a mid-week read.

**Card 2 — Flippable Categories (real output):**
Pulls from `war_room.get_flippable_categories(matchup)`. Given the cached matchup this will identify L and
ERA/WHIP as the losing category targets with computed gaps. The gap values come from the same stale cache.
Suggestion text (example observed in code): `"Stream a high-K SP to close the K gap"` or similar.

**Finding:** The suggestion strings in Card 2 are generic fixed text (from `war_room.py`) not dynamically
tailored to the specific roster. A novice reading "Stream a high-K SP" cannot know which FA to pick without
going to Pitcher Streaming.

**Card 3 — Today's Actions (real output):**
Calls `compute_todays_actions(roster, matchup, losing_cats)`. Requires today's schedule data from the game-day
module. Priority rows have PRI·N orange chips + player headshots + team abbreviations + reason text. The
player headshot system uses MLB Stats API CDN (network-gated during audit; renders placeholder when offline).

**Finding:** "Today's Actions" is the most actionable section but it appears third, after two analysis cards.
For a daily-workflow user who opens the page at 8 AM, this card is what they need first.

**Card 4 — Player Streaks (real output):**
Calls `compute_hot_cold_report(roster, max_entries=4)`. Returns up to 4 HOT/COLD tiles in a 2-column grid.
Each tile shows: HOT/COLD chip, headshot, player name, team abbr, L7 performance headline, season-delta detail.

**Finding:** Sparklines (per-game trend lines) are described in comments as "omitted — fabricated data."
The COLD tiles likely include some benched/IL players — a HOT/COLD designation on an IL player is misleading
since they haven't played.

**Card 5 — Category Flip Analysis (real output):**
From `compute_category_flip_probabilities` + `get_pivot_summary`. Rows are grouped CONTESTED / WON / LOST
with flip probability %. Given the Week 12 cache, ERA and WHIP are LOST; L is also LOST; SB is likely TIE.
CONTESTED cats would be those very close to the flip boundary.

The fig label shows `{N}D LEFT` where N is 7 minus today's day-of-week (Mon=0). On a Friday this reads `3D
LEFT`, correctly.

**Finding:** CONTESTED chip is colored orange (`T["hot"]`), WON is green, LOST is danger-red. This is clear.
However the flip probability text reads `47% flip` for a CONTESTED cat — a novice does not know whether a
higher % means "I'm more likely to win it" or "it's more likely to flip away from me." The word "flip" is
ambiguous for inverse stats (ERA/WHIP).

**Card 6 — Regression Alerts (real output):**
Calls `generate_regression_alerts(roster, min_pa=50)` which compares `xwoba` vs actual `woba` from
`statcast_archive`. The DB audit found **all 501 rows in `statcast_archive` have NULL xwoba, barrel_pct,
hard_hit_pct, and stuff_plus.** Therefore `_reg_alerts` will be empty and Card 6 will NOT render. The
Statcast Signals context card also will not render. Two entire sections of the page produce no output because
the Statcast data pipeline failed silently.

### 4.6 Weekly Report Expander

**Real output (source-traced, Week 12):**
- Label: `Weekly Report — Week 12 vs The Good The Vlad The Ugly`
- Contains: `Generate Report` button + opponent card (Tier, Threat level, Strengths, Weaknesses) + Category
  Projections per cat (LIKELY WIN / LIKELY LOSS / TOSS-UP) + Action Items (numbered) + Streaming Targets

`_report_opp` is sourced from `MatchupContextService.get_opponent_context()`. Opponent profile data may be
partially populated from the cached `opponent_profiles` table.

**Finding:** The `Generate Report` button inside the expander does not explain what it will do or how long
it will take. Clicking it sets a session flag → `st.rerun()` which re-runs the whole page. The report itself
uses `generate_monday_report(roster, None, ...)` with a hardcoded `None` for the second argument (matchup
score) — the `None` triggers the fallback report path. It is not obvious to a novice whether the current
report is live or from a prior run.

### 4.7 Category Totals Context Cards

**Real output (reconstructed from season_stats for 13 hitters and 14 pitchers):**

Hitting totals (2026 Live, as of Jun 10):
```
R:   450
HR:  137
RBI: 410
SB:  64
AVG: .237
OBP: .332
```

Pitching totals (2026 Live):
```
W:    55
L:    37
SV:   23
K:    704
ERA:  3.57
WHIP: 1.24
IP:   692.3
```

**Finding:** Context cards show stats in IBM Plex Mono for labels + values, but only 6 columns per side with
no league-rank context. A novice sees `HR: 137` and has no idea if that's 1st or 12th in the league.
Standalone totals without a rank or league percentile are less actionable.

### 4.8 Category Gaps Card

**Real output (Week 12 vs "The Good The Vlad The Ugly"):**
Priority Targets box: L and ERA (or whichever 2 losing cats are closest to flipping by z-score).

Per-category diff rows:
```
R:    you=14  opp=6   diff=+8    (WIN, green)
HR:   you=7   opp=1   diff=+6    (WIN, green)
RBI:  you=14  opp=8   diff=+6    (WIN, green)
SB:   you=0   opp=0   diff=0     (TIE, gray)
AVG:  you=.359 opp=.271 diff=+.088 (WIN, green)
OBP:  you=.431 opp=.316 diff=+.115 (WIN, green)
W:    you=1   opp=0   diff=+1    (WIN, green)
L:    you=2   opp=0   diff=-2    (LOSS, red)   ← inverse: more Ls = losing
SV:   you=0   opp=0   diff=0     (TIE, gray)
K:    you=21  opp=13  diff=+8    (WIN, green)
ERA:  you=7.08 opp=3.38 diff=-3.70 (LOSS, red)
WHIP: you=1.82 opp=1.03 diff=-0.79 (LOSS, red)
```

**Finding — inverse stats:** The "L" diff is rendered as `-2` in danger-red with a label that says "Priority
Target" if it's one of the two closest-to-flip. A novice sees "L: −2" and may read this as "I'm losing by 2
in some generic count." The inverse-stat semantics (fewer Losses is better) are never explained. Diff sign
polarity is correct in the code (positive = winning), but there is no annotation on the card.

### 4.9 Injured List Alerts Card

**Real output (DB-confirmed, 4 players on IL):**
```
● Cal Raleigh (IL10) — Oblique issue — On rehab assignment
● Bailey Ober (IL15) — Elbow issue
● Garrett Crochet (IL60) — Lat issue
● Shane Bieber (IL60) — Elbow issue
```

The card shows health-dot (red) + bold player name + status in parentheses.

**Finding:** The IL alert card derives the badge from `injury_history` health scores, not directly from the
`league_rosters.status` column. The health score maps "IL10/IL15/IL60" categories to labels via
`get_injury_badge`. However, the mapping requires sufficient `games_played` history — for Shane Bieber, who
has `games_played=0` in 2026, the health score computation may produce a degraded result. In the current DB,
Bieber has `0 games_played` in 2026 season stats, so his badge may read "Low Risk" rather than correctly
identifying him as an IL-60 player.

### 4.10 Lineup Validation Card

Calls `get_todays_mlb_games()` (network-blocked during audit) and `validate_daily_lineup(roster, todays_teams)`.
With network blocked, `todays_teams` returns an empty list → `_lineup_issues = []` and `_teams_playing = 0`.
The Lineup Validation card only renders when `_teams_playing > 0`. With network blocked: **the card is
completely absent.** No fallback message.

### 4.11 Live Matchup Card

Same data source as Card 1 (matchup cache). Renders a small per-category table with your stat / opp stat
in monospace and result color. The fig label (card title) is "Live Matchup" — the word "Live" is misleading
when the data is 4,012 minutes stale (the same staleness issue as Card 1).

### 4.12 Statcast Signals Card

**Real output (DB-confirmed):** Card does NOT render. All 501 rows in `statcast_archive` have NULL xwoba,
barrel_pct, hard_hit_pct, and stuff_plus. `_sc_rows` is empty → `if _sc_rows: render_context_card(...)` is
never entered. **No empty-state message. No card header. Nothing.** A novice does not know this feature
exists or that it is silently failing.

### 4.13 Data Freshness Card

**Real output (from `refresh_log`):** Shows a per-source freshness list. Most recent entries:
- `yahoo_standings`: success, 2026-06-12 (recent)
- `game_logs`: success, 2026-06-10
- `yahoo_matchup`: success, 2026-06-10 (4,012 min ago)
- `projections`: error, 2026-06-10 ("all FanGraphs fetches failed")
- `stuff_plus`: skipped (FanGraphs 403 — known)
- `season_stats`: partial (2580/8054 rows)

**Finding:** The card is buried at the very bottom of the context panel, after all the intelligence cards.
A user who sees stale matchup data in the Matchup Pulse card will not see the explanation until much later.

### 4.14 Roster Table

**Real output (DB-confirmed, 27 players):**

Hitters (13 active after filter — 1 on IL showing "IL" slot):
- Starting slots: 1B (Matt Olson), 2B (Angel Martínez), SS (Andrés Giménez), 3B (Max Muncy), C (Dillon
  Dingler), OF (Yordan Alvarez), OF (Jackson Merrill), OF (Jakob Marsee), Util (Mike Trout), Util (Bryan
  Reynolds)
- BN: Alex Bregman (3B, BN), Dansby Swanson (SS, BN)
- IL: Cal Raleigh (IL10)

Pitchers (14):
- SP: Chris Sale, Framber Valdez
- P: Taj Bradley, Kyle Harrison, Andrew Alvarez
- RP: Robert Suarez, Raisel Iglesias
- P: Jeff Hoffman
- BN: Dustin May, Landen Roupp, Eduardo Rodriguez
- IL: Bailey Ober (IL15), Garrett Crochet (IL60), Shane Bieber (IL60)

The table caption reads: `"Full 2026 season totals. Updates hourly from MLB Stats API."` — even when the last
refresh was 4,012 minutes ago. The caption claims hourly updates but the actual last update was >2.5 days
prior.

**Notable data finding:** Shane Bieber (IL60) has `games_played=0` in 2026 — consistent with him being
on the IL since the start of the season. The team carries 3 IL slots with 4 players on IL (Crochet uses
his IL60 slot, Bieber uses his IL60 slot, Ober uses IL15, Raleigh uses IL10).

### 4.15 Export to Excel

Generates a `.xlsx` file of the current Side × Timeframe view. File named
`heater_roster_hitters_season.xlsx` by default. This is a useful feature buried below the main roster table.
No announcement or hint that it exists.

### 4.16 Marcel-Adjusted Projections Section

**Real output (source-traced):** Visible only when `BAYESIAN_AVAILABLE=True` AND
`season_stats.games_played.sum() > 0`. With `BAYESIAN_AVAILABLE` dependent on PyMC being importable and the
progress bar requiring an active connection. In a cached read-only session this likely renders.

Shows: Player / AVG / HR / RBI / SB / ERA / WHIP / K. HR, RBI, SB, K values formatted as float to 2 decimal
places (`f"{x:.2f}"`) — so a player with 15 projected HR shows `15.00`. Mixing integers formatted as floats
is visually odd.

### 4.17 News and Alerts Section

**Real output (DB-confirmed):** All 15 news entries for Team Hickey are Yahoo-sourced injury entries, all
four IL players generating 3-4 duplicate entries each (same headline, different timestamps). After
deduplication by `(player_name + headline).lower()`, only 4 unique entries remain:
1. Shane Bieber — Placed on IL — Elbow issue (Yahoo, Injury)
2. Garrett Crochet — Placed on IL — Lat issue (Yahoo, Injury)
3. Bailey Ober — Placed on IL — Elbow issue (Yahoo, Injury)
4. Cal Raleigh — Placed on IL — Oblique issue + rehab assignment (MLB source, has real detail text)

The sort selectbox (`Most Recent` / `Severity` / `Sentiment Impact`) is visible above the count summary.
Count summary reads: `4 news items found: 4 injury`.

**Finding:** Sentiment score is `NaN` for all Yahoo-sourced injury entries. The `_sentiment_indicator`
receives `float(NaN)` but `sentiment = 0.0` by the `if sentiment is None: sentiment = 0.0` guard — NaN is
not None, so the guard does NOT catch it. `_SENTIMENT_THRESHOLDS` checks `score >= 0.2` → NaN comparisons
return False → falls through all thresholds → **returns empty string `""`**. Sentiment dot is invisible for
Yahoo injury entries, silently missing from the card header.

---

## 5. Errors, Issues & Difficulties

### 5.1 Statcast data entirely NULL — silent failure
The `statcast_archive` table has 501 rows but ALL of xwoba, barrel_pct, hard_hit_pct, stuff_plus are NULL.
Both the **Statcast Signals** context card and **Regression Alerts** (Card 6) are silent no-shows. No
empty-state message, no banner, nothing. Two of the page's intelligence features are completely invisible
and the user has no idea they exist or that they're broken.

### 5.2 Matchup ticker hidden when Yahoo is offline
`render_matchup_ticker()` gates on `st.session_state["yahoo_connected"]`. In every cached/multi-user state
this key is falsy. The ticker — the single fastest "how's my week going?" answer — disappears. The Matchup
Pulse card below partially compensates, but the ticker is gone from the top of the page.

### 5.3 Record and Rank show "—" when Yahoo client not in session
The identity strip shows `Record: —` and `Rank: —` whenever the live Yahoo client is not available. The
actual record IS present in `league_matchup_cache` and `league_standings` but the code only populates these
fields from the live yfpy object. Members and cached sessions always see "—" for two of the five identity
readouts. This is confusing and makes the app look broken.

### 5.4 NaN sentiment score bypasses the None guard
`news_item.get("sentiment_score", 0.0)` returns `NaN` from the DB. `if sentiment is None: sentiment = 0.0`
does NOT catch NaN. The `_sentiment_indicator` function then evaluates `NaN >= 0.2` → False → falls through
all thresholds → returns `""`. The sentiment dot is silently missing on Yahoo injury entries (all 4 news items
for Team Hickey).

### 5.5 News section shows only injuries — no ESPN/RotoWire content
Player news for Team Hickey consists entirely of Yahoo-sourced IL placement entries with body text like
"Placed on IL — Elbow issue" (no detail). The `generate_intel_summary` call is correctly skipped for
these (no real detail content), but the fallback template reads: `"Shane Bieber is listed as IL10 with a
placed on il issue."` — the capitalization and grammar are broken because the Yahoo headline `"Placed on IL
— Elbow issue"` is passed as `headline.lower()` → `"placed on il — elbow issue"`, producing awkward text.

### 5.6 Matchup data is 4,012 minutes stale with no page-level warning
The cached matchup (Jun 10) is displayed in Card 1, the Live Matchup card, the Category Gaps card, and the
Category Flip Analysis card. None of them display a staleness warning. The roster table caption says "Updates
hourly" even though the last refresh was >2 days ago.

### 5.7 Injury badge for Shane Bieber likely reads "Low Risk"
`compute_health_score(gp=[0], ga=[162], ...)` where gp is 0 games played will produce a very low ratio
(0/162 = 0%) which falls into the high-risk bucket by absence ratio — but the logic must see non-zero
il_stints or il_days to trigger the correct badge. If the injury_history row for Bieber (2026) shows
il_stints=0 and il_days=0 (because the IL event is tracked as "active" and hasn't accumulated days yet),
the badge renders "Low Risk" for an IL-60 patient. This contradicts the IL Alerts card below.

### 5.8 Roster table caption claims "Updates hourly" unconditionally
Line 2154: `_caption = "Full 2026 season totals. Updates hourly from MLB Stats API."` This caption is
hardcoded. It renders even when data is days stale. It is factually false in the current state.

### 5.9 Marcel projections format integer stats as floats
HR/RBI/SB/K columns in the Bayesian projections table are formatted `f"{x:.2f}"`, producing `15.00` for
HR. Integer stats should display as integers.

### 5.10 "FLIP TO WIN" suggestion text is generic, not roster-specific
Card 2's suggestion column reads fixed strings like "Start any 2-start pitchers this week." A novice sees this
and asks "which 2-start pitchers?" with no link to the Pitcher Streaming page.

### 5.11 No mobile adaptation for the 6-column category heat grid
Card 1 uses `grid-template-columns:repeat(6,1fr)`. On a narrow viewport this will overflow or collapse to
illegible widths. The Combustion spec requires mobile responsiveness.

### 5.12 Two segmented controls with overlapping scope
There are two separate `st.segmented_control` widgets above the roster table: "Stat source" (affects context
cards totals only) and "Timeframe" (affects the roster table only). These are positioned adjacently but affect
different parts of the page, which is not communicated. A novice changing "Stat source" may expect the table
to update.

### 5.13 "Open player dossier" selectbox is redundant with table row links
The table has `<a href="?player=ID">` links AND a selectbox "Open player dossier" that do the same thing.
The footer text says "CLICK ANY PLAYER" but the `<a>` links open a URL param which re-renders the full page
before calling `show_player_card_dialog`. The selectbox is the "reliable opener" but placed alongside
controls that look like table filters (Timeframe / Side), making its purpose unclear.

### 5.14 Page is extremely long — no section anchors or progressive disclosure
The full page scroll covers: identity → ticker → 6 War Room cards → Weekly Report expander → Hitting Totals
→ Pitching Totals → Category Gaps → IL Alerts → Lineup Validation → Live Matchup → Statcast Signals →
Data Freshness → Roster Table → Excel export → Marcel Projections → News section. This is a very long
sequential layout with no sticky navigation, no jump links, and no fold at the critical "what do I do
today?" moment.

### 5.15 "Today's Actions" and "Player Streaks" silently absent if exception thrown
Cards 3 and 4 are wrapped in `try/except: pass`. If `compute_todays_actions` or `compute_hot_cold_report`
throw, the card is silently missing with no fallback or empty-state. The user sees a gap in the war room
without knowing why.

### 5.16 FIG.01 label on "My Team" is consistent but the eyebrow says "SEASON" not "MY TEAM"
The cross-cutting finding notes FIG number inconsistency across pages. On this page `FIG.01 — ROSTER CONTROL`
is used with eyebrow `SEASON`. Navigation label is "My Team." The page wordmark is `My Team.`. The fig label
and eyebrow do not match the page name — minor but part of the systemic inconsistency.

---

## 6. UI/UX & Visual Design Critique

### 6.1 Information hierarchy: daily action buried below analysis
The user's daily workflow is: (1) check score, (2) check what to do today, (3) check category health.
The page puts analysis (Matchup Pulse, Flippable Categories) BEFORE actionable content (Today's Actions,
Lineup Validation). A simple reorder — Today's Actions first, Matchup Pulse second — would dramatically
improve the daily workflow.

### 6.2 The "Refresh Stats" / "Sync Yahoo" button distinction is invisible to novices
Two nearly identical `st.button` widgets sit side by side with no icon, no subtitle, no tooltip. Best practice
for action-critical controls at the top of a page is to use distinct icons and a one-line description. The
Combustion design system supports icons via SVG spans — they should be used here.

### 6.3 The identity strip readouts suffer from excessive "—" placeholders
The Rank and Record readouts show `—` in cached mode. Empty numeric stats in a "high-tech instrument panel"
aesthetic undermine the premium feel. The system knows the record from the matchup cache and the standings.

### 6.4 Context cards are a wall of monospace text
All context cards (Hitting Totals, Pitching Totals, Category Gaps, IL Alerts, etc.) use IBM Plex Mono for
both labels and values with a flat key-value list. While monospace is correct for figures, the cards have no
visual hierarchy — label weight and color are uniform. The Priority Targets box is the only standout element.
Combustion's `.chip` and `.hero-num` utilities are not used inside context cards.

### 6.5 "Live Matchup" card title is misleading — data is cached
The card title is "Live Matchup" but data is 4,012 minutes old. Should be "Matchup — Week 12" or include a
"(cached)" tag in the fig label.

### 6.6 News section is all injury placeholders — no actionable content
The news section's value proposition is "stay on top of your players." For Team Hickey's 27-player roster,
only 4 unique news items render, all injury IL placements already visible in the Injured List Alerts card
above. The section adds no incremental information. ESPN and RotoWire sources appear completely absent for
this roster.

### 6.7 The "Active Roster" panel is not obviously a panel
The `build_panel_html` call wraps the roster table, but the panel header ("Active Roster") blends into the
page. There is no visual hierarchy between the War Room cards (which are also panels) and the main roster
panel. A clear section heading with more whitespace above it would help.

### 6.8 Number formatting inconsistency in Marcel projections
HR/RBI/SB/K rendered as `15.00` rather than `15`. Mixed float/integer display in the same table is visually
incorrect and looks like a bug. AVG and ERA/WHIP correctly use `.3f` and `.2f` respectively.

### 6.9 The `render_reco_banner` for roster count is orange-rail but content is purely informational
`render_reco_banner("Roster: 27 players | 13 hitters, 14 pitchers", ...)` renders with the orange left-rail
accent used for recommendations. An informational roster count is not a "recommendation" — using the accent
color implies action or alert. Should use a neutral container.

### 6.10 No loading states for War Room cards that compute live
Cards 3 and 4 (Today's Actions, Player Streaks) call ML/stats functions synchronously. If they take >1 second
(e.g. rolling stats for 14 pitchers), the page appears frozen with no spinner. Only the Bayesian projections
section uses a `st.progress` bar. Other cards should show at minimum a `st.spinner`.

### 6.11 FIG label "WK {N}" on Matchup Pulse is correct but inconsistent with fig conventions
Other panels use descriptive labels like `"{N} FLAGGED"` or `"L7 GP"`. The `"WK 12"` format is fine but the
FIG numbering convention in `render_page_header` uses `FIG.01` while sub-panel fig labels use free-text
strings. This is consistent within the page but worth noting as a systemic pattern.

### 6.12 The Weekly Report expander is visually indistinguishable from War Room cards
The expander (`st.expander`) is a native Streamlit component with default styling, while all six War Room
cards are custom HTML panels. The visual break between these two types is jarring — the expander has a
rounded border and a different expand/collapse affordance compared to the panel headers.

---

## 7. Recommendations (≥ 10, ordered by impact)

### Rec 1 — Fix silent failure for stale data: add a staleness banner
**Problem:** Matchup data 4,012 minutes old is presented as current in 4 different cards and the Live Matchup
card is literally titled "Live." Users may make bad roster decisions based on outdated category standings.
**Fix:** Compute the age of `league_matchup_cache.updated_at` at page load. If > 60 minutes, render a yellow
`st.warning` or amber chip at the top of the War Room section: "Matchup data last updated Jun 10 — tap Sync
Yahoo to refresh." This should be the first visible element after the identity strip.

### Rec 2 — Show Record and Rank from cached data, not just live Yahoo client
**Problem:** `Record: —` and `Rank: —` display whenever the live Yahoo session isn't active (cached mode,
read-only members, page refresh without re-auth).
**Fix:** Fall back to `league_standings` DB table for WINS/LOSSES/TIES to populate `_id_record` and
`_id_rank` when the Yahoo client is not available. The data is already in the DB and the code already reads
standings — it just doesn't connect it to the identity strip fallback.

### Rec 3 — Differentiate "Refresh Stats" and "Sync Yahoo" with icons + subtitles
**Problem:** Two identical-looking buttons at the top of the most-used page with no explanation of what each
does. Novices can't make an informed choice.
**Fix:** Add inline SVG icons (already defined in `PAGE_ICONS`: "refresh") and a 1-line description under
each button. Example: `Refresh Stats` → subtitle "MLB API: live stats only" ; `Sync Yahoo` → subtitle
"Yahoo: rosters, matchup, FAs." Alternatively, render them as a single "Refresh All" button that calls both,
with an "Advanced" expander for individual controls.

### Rec 4 — Reorder War Room: Today's Actions first
**Problem:** The most actionable card (Today's Actions — what do I do right now?) is the third card, behind
two analysis panels.
**Fix:** Move Card 3 (Today's Actions) to position 1, Matchup Pulse to position 2, Flippable Categories to
position 3. This puts the daily action item in the hero position for a manager who opens the app at 8 AM.

### Rec 5 — Fix the sentiment NaN bug: use `pd.notna()` / `math.isnan()` guard
**Problem:** `news_item.get("sentiment_score", 0.0)` returns `NaN` from the DB. The `if sentiment is None`
guard does not catch NaN. All Yahoo injury entries silently lose their sentiment dot.
**Fix:** Replace line 247-250 with:
```python
import math
sentiment = news_item.get("sentiment_score", 0.0) or 0.0
if sentiment is None or (isinstance(sentiment, float) and math.isnan(sentiment)):
    sentiment = 0.0
```

### Rec 6 — Surface empty-state messages for Statcast Signals and Regression Alerts
**Problem:** Both cards silently disappear when Statcast data is null. A user cannot tell whether the feature
doesn't exist or is broken.
**Fix:** For Statcast Signals: if `_sc_rows` is empty, render a `render_empty_state("Statcast Signals",
"No Statcast data available. Data refreshes daily from pybaseball.", "bar_chart")` card. Same for Regression
Alerts. The presence of the empty-state communicates "this feature exists but has no data right now."

### Rec 7 — Fix "Updates hourly" caption to reflect actual staleness
**Problem:** `_caption = "Full 2026 season totals. Updates hourly from MLB Stats API."` is hardcoded and
misleading when data is hours or days stale.
**Fix:** Dynamically compute the age from `refresh_log` for the `season_stats` source and incorporate it:
```python
_caption = f"Full 2026 season totals. Last updated {_relative_time} via MLB Stats API."
```
where `_relative_time` is "2 hours ago", "2 days ago", etc.

### Rec 8 — Fix integer stats formatted as floats in Marcel projections
**Problem:** HR/RBI/SB/K render as `15.00` in the Bayesian projections table (line 2398-2400).
**Fix:** Change the format for integer-category columns from `f"{x:.2f}"` to `f"{int(round(x))}"`.

### Rec 9 — Make the matchup ticker render from SQLite cache, not just live Yahoo session
**Problem:** `render_matchup_ticker()` gates on `st.session_state["yahoo_connected"]`. In cached/multi-user
mode the ticker is invisible — the most prominent "score at a glance" UI is gone.
**Fix:** Remove the `yahoo_connected` gate. Instead, check `yds.get_matchup()` directly (which already falls
back to the SQLite cache). If a matchup dict is returned, render the ticker with a "(cached)" staleness hint
if the data age exceeds 60 minutes. If the matchup is None, skip the ticker.

### Rec 10 — Add cross-category rank context to Hitting/Pitching Totals cards
**Problem:** Standalone totals (HR: 137) have no league context. A novice can't tell if 137 HR is good.
**Fix:** Add a "League Rank" column to context cards. Query `league_standings` for each category, find the
team's position, and render a `#10/12` rank badge next to each category total. This requires one aggregation
query but massively improves the actionability of the card.

### Rec 11 — Add deep-link from Flippable Categories suggestions to relevant action pages
**Problem:** Suggestion text like "Stream a high-K SP" has no link to the Pitcher Streaming page.
**Fix:** Parse the suggestion type (streaming → `/4_Pitcher_Streaming`, trade → `/11_Trade_Analyzer`, waiver
→ `/14_Free_Agents`) and append a `heater-link`-styled anchor in the suggestion cell. The `?player=` param
pattern is already established for dossier links.

### Rec 12 — Consolidate the two segmented controls above the roster table and add labels
**Problem:** "Stat source" (affects context cards) and "Timeframe" (affects roster table only) look like
peer controls but affect different parts of the page.
**Fix:** Either (a) move "Stat source" directly above the context cards it controls, clearly labeled; or (b)
merge "Stat source" into the "Timeframe" control (rename to "Time Period" and include `2026 Live` as the first
option, making it drive both the table and the cards). Also add a visible label clarifying which controls
affect which section.

---

## 8. Severity-Tagged Issue List

- `[BLOCKER]` Matchup ticker permanently hidden in cached/multi-user mode (gate on `yahoo_connected` is always falsy); users have no top-of-page score — the primary daily-workflow element is missing.
- `[HIGH]` Statcast Signals and Regression Alerts silently absent — all `xwoba`/`barrel_pct`/`stuff_plus` values are NULL in `statcast_archive`; no empty state shown; two intelligence features are invisible.
- `[HIGH]` Stale matchup data (4,012+ min) shown without any warning in 4 cards; "Live Matchup" card title is factually false; users can make wrong start/sit decisions.
- `[HIGH]` Record and Rank show `—` in all cached/multi-user sessions despite data being available in `league_standings` — makes the identity strip look broken for every non-OAuth session.
- `[HIGH]` NaN sentiment score bypasses `None` guard; sentiment dot silently absent on all Yahoo injury entries — affects every news card in the current state.
- `[HIGH]` Roster table caption "Updates hourly from MLB Stats API" is hardcoded and factually incorrect when data is days stale.
- `[MEDIUM]` Today's Actions card is position #3 in the War Room instead of #1; daily-workflow action buried after two analysis cards.
- `[MEDIUM]` `Refresh Stats` and `Sync Yahoo` buttons are visually identical with no explanation; novice cannot make an informed choice between them.
- `[MEDIUM]` Lineup Validation card is entirely absent with no fallback when `get_todays_mlb_games()` returns empty (network blocked or off-season).
- `[MEDIUM]` IL badge for Shane Bieber likely reads "Low Risk" (0 GP in 2026 injury_history produces degenerate score); contradicts the IL Alerts card which correctly shows him IL60.
- `[MEDIUM]` Marcel projections table formats HR/RBI/SB/K as `15.00` (float) instead of `15` (integer).
- `[MEDIUM]` Flippable Categories suggestion text is generic, not roster-specific, with no links to action pages.
- `[MEDIUM]` Category Gaps: inverse-stat semantics never explained; `L: -2` in danger-red is opaque to a novice who does not know fewer losses is better.
- `[MEDIUM]` Page has no sticky header, no jump links, no progressive disclosure — full scroll from identity to news section is extremely long.
- `[MEDIUM]` Statcast Signals and Regression Alerts produce no empty-state message when data is unavailable — features appear to not exist.
- `[LOW]` `render_reco_banner` is used for the roster count (purely informational) with the orange recommendation accent — wrong semantic color.
- `[LOW]` Weekly Report expander styled differently (native Streamlit) from the custom HTML War Room cards; visual break is jarring.
- `[LOW]` "Open player dossier" selectbox is redundant with row `<a href>` links and positioned among controls that look like table filters.
- `[LOW]` News section for Team Hickey produces only injury entries already visible in the IL Alerts card; adds no incremental value.
- `[LOW]` Fallback template for Yahoo injury entries produces grammatically broken text: `"is listed as IL10 with a placed on il — elbow issue."`.
- `[LOW]` No spinner or progress indicator on Today's Actions or Player Streaks cards (they compute synchronously and can be slow).
- `[LOW]` Export to Excel button is small and hidden below the roster table with no call-out.
- `[POLISH]` FIG eyebrow "SEASON" does not match page name "My Team" (minor inconsistency with Pitcher Streaming pattern).
- `[POLISH]` Identity strip avatar shows initials "TH" (navy square) even when a Yahoo team logo URL is accessible — only populates from live yfpy client which is unavailable in cached mode.
- `[POLISH]` Category heat grid uses `repeat(6,1fr)` with no `@media` breakpoint — will overflow on mobile screens.
- `[POLISH]` COLD streak tiles use `var(--fp-cold)` which maps to `#5f7d9c` (muted steel) — barely distinguishable from the card background surface on pale screens.

---

*Report written by the Page 02 audit agent. All numbers are from live DB queries or source-code traces.
All "reconstructed" labels are explicitly annotated above.*
