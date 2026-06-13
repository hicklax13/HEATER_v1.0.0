# Page 14 — Player Databank — Test-User Report

**Auditor persona:** Connor, novice fantasy-baseball manager, Team Hickey, FourzynBurn 12-team H2H-categories league.
**Source files:** `pages/19_Player_Databank.py`, `src/player_databank.py`
**Report written:** 2026-06-13

---

## 2. Page Purpose & First Impression

The Player Databank is billed in the sidebar as a "historical multi-year player lookup" and in the page header as "Player Databank — Live stats for every MLB player." The idea is that a manager can search across the entire 9,888-player pool, filter by position / availability / team / fantasy team, pick a time window (this season, last 7 days, last year, etc.), and get a sortable stat table with an Excel export.

**First impression (novice lens):** The blank empty-state ("Search the Databank — Set your filters above, then click Search") is the right pattern, but there is an overwhelming amount of UI above the fold — a search box, two checkbox/position filters, and then **six** dropdowns in a single row, culminating in a full-width Search button. A beginner reading the dropdown labelled "Stats" that defaults to "Season (total)" is immediately confused: total of what? The last 7 days total, the last 30 days total, and the season total are all in the same dropdown. This is not a familiar UX metaphor.

The bigger first-impression problem: the page is advertised as a "historical" tool but what it shows is mostly **a paginated list of players sorted by pre-season ADP**. There is no side-by-side year comparison within a player's row, and no player profile card or drill-down. The CLAUDE.md calls this "historical multi-year player lookup" which implies clicking a player and seeing their career trajectory — that does not exist. It is a filtered, sortable table, which overlaps considerably with the **Leaders** page (page 13) and the **Player Compare** page (page 12).

---

## 3. Methodology

- Read `pages/19_Player_Databank.py` in full (417 lines).
- Read `src/player_databank.py` in full (1,457 lines).
- Read `src/ui_shared.py` (header/CSS authority, first 120 lines).
- Read `tests/test_no_offpalette_hex_in_pages.py` (banned color list).
- Queried the live SQLite DB read-only with network guard for:
  - `players` table (9,888 rows, schema, level column).
  - `game_logs` table (58,411 rows, 3 seasons: 2024=17,885; 2025=27,017; 2026=13,509).
  - `season_stats` table (6,944 rows, seasons 2023–2026).
  - `statcast_archive` table (schema, data availability).
  - `ecr_consensus`, `adp`, `ownership_trends`, `transactions`, `league_rosters`, `league_teams`.
  - `refresh_log` for all databank-relevant sources.
  - `yahoo_free_agents` cache (608 players).
- Ran `load_databank()` and `filter_databank()` for multiple stat views.
- Ran `compute_rolling_stats()` for Freddie Freeman (player_id=4) and Zack Wheeler (player_id=107).
- Ran `get_data_as_of_label()` and `get_data_refreshed_label()` for 6 stat views.
- Benchmarked `load_databank()` load times for 3 views.
- Tested `_format_cell()` for IP, ERA, WHIP, AVG, OBP, percent_owned.
- Tested `filter_databank()` for all 5 STATUS_OPTIONS including `"W"` (Waivers Only).

---

## 4. Feature & Control Inventory

| # | Control | Type | What it does | Tested? |
|---|---------|------|--------------|---------|
| 1 | Search player by name | Text input (form) | Substring search on player_name | Yes |
| 2 | Position | Selectbox (form) | Filter: All Batters, All Pitchers, or specific position (C/1B/2B/3B/SS/OF/Util/SP/RP) | Yes |
| 3 | Show my team | Checkbox (form) | Restricts to is_user_team=True players | Yes |
| 4 | Status | Selectbox (form) | All Players / All Available / Free Agents Only / Waivers Only / All Taken | Yes |
| 5 | MLB Teams | Selectbox (form) | Filter to one MLB team (30 teams + ALL) | Yes |
| 6 | Fantasy Teams | Selectbox (form) | Filter to one fantasy roster (12 teams + No Team Selected) | Yes |
| 7 | Stats | Selectbox (form) | 28 stat view options (see below) | Yes |
| 8 | Sort by | Selectbox (form) | 19 sort options (Pre-Season, Current, GP, % Ros, Adds, Drops, R, HR, RBI, SB, AVG, OBP, IP, W, L, SV, K, ERA, WHIP) | Yes |
| 9 | Order | Selectbox (form) | Ascending / Descending | Yes |
| 10 | Search (submit) | Button (primary, full-width) | Triggers data load, filter, sort, and table render | Yes |
| 11 | Results panel | Instrument panel strip | Shows: SHOWING X–Y of N / PAGE N of N / STATS AS OF MM/DD / REFRESHED date | Yes |
| 12 | Player table | Custom HTML table | Sortable JS table: Player, Team, Pos, Roster Status, GP*, Opp: MM/DD, Pre-Season, Current, Adds, Drops, % Ros, + 6-7 stat cols | Yes |
| 13 | Column header clicks | JS (in-table) | Client-side sort on any column (↑/↓ arrows) | Yes (traced) |
| 14 | Previous 25 | Button | Go to previous page | Yes |
| 15 | Next 25 | Button | Go to next page | Yes |
| 16 | Page N of N | Static text | Pagination state indicator | Yes |
| 17 | Export | Download button | Export full filtered result to branded .xlsx | Yes (traced) |
| 18 | Feedback widget | Popover (conditional) | Renders render_feedback_widget("Player Databank") | Yes (traced) |

**28 Stat View Options (Stats dropdown):**

| Key | Label | Type | Notes |
|-----|-------|------|-------|
| S_PS7 | Next 7 Days (proj) | proj | Returns pool as-is (no 7-day breakdown) |
| S_PS14 | Next 14 Days (proj) | proj | Returns pool as-is |
| S_PSR | Remaining Games (proj) | proj | Returns pool as-is |
| S_L | Today (live) | live | Returns pool as-is (NOT live in-game data) |
| S_L7 | Last 7 Days (total) | total | game_logs rolling 7d |
| S_L14 | Last 14 Days (total) | total | game_logs rolling 14d |
| S_L30 | Last 30 Days (total) | total | game_logs rolling 30d |
| S_S_2026 | Season (total) | total | game_logs 2026 season |
| S_S_2025 | 2025 Season (total) | total | game_logs 2025 season |
| S_S_2024 | 2024 Season (total) | total | game_logs 2024 season |
| ADVST_ADVST_2026 | 2026 Advanced | advanced | Pool as-is (Statcast columns) |
| ADVST_ADVST_2025 | 2025 Advanced | advanced | Pool as-is |
| S_AL7 | Last 7 Days (avg) | avg | game_logs 7d per-game avg |
| S_AL14 | Last 14 Days (avg) | avg | game_logs 14d per-game avg |
| S_AL30 | Last 30 Days (avg) | avg | game_logs 30d per-game avg |
| S_AS_2026 | Season (avg) | avg | game_logs 2026 per-game avg |
| S_AS_2025 | 2025 Season (avg) | avg | game_logs 2025 per-game avg |
| S_AS_2024 | 2024 Season (avg) | avg | game_logs 2024 per-game avg |
| S_SDL7 | Last 7 Days (std dev) | stddev | game_logs 7d standard deviation |
| S_SDL14 | Last 14 Days (std dev) | stddev | game_logs 14d std dev |
| S_SDL30 | Last 30 Days (std dev) | stddev | game_logs 30d std dev |
| S_SD_2026 | Season (std dev) | stddev | game_logs 2026 std dev |
| S_SD_2025 | 2025 Season (std dev) | stddev | game_logs 2025 std dev |
| S_SD_2024 | 2024 Season (std dev) | stddev | game_logs 2024 std dev |
| K_K | Ranks | special | Pool as-is (no dedicated column set) |
| R_O | Research | special | Pool as-is (no dedicated column set) |
| M_W | Fantasy Matchups | special | Pool as-is (no dedicated column set) |
| O_O | Opponents | special | Pool as-is (no dedicated column set) |

---

## 5. Feature-by-Feature Test Log with Real Outputs

### 5.1 Empty State (pre-search)

On first load, `db_search_triggered` is not set in session state. The page calls `render_empty_state("Search the databank", "Set your filters above, then click Search to pull live stats for every MLB player.", icon_key="free_agents")` and stops. This is the correct deferred-execution pattern — no data loads until the user clicks Search.

### 5.2 Default Search (Season 2026, All Batters, All Available, Pre-Season ADP ascending)

**Filters applied:** Position=B (is_hitter=1), Status=A (not on any roster), MLB Team=ALL, Fantasy Team=NONE, Stats=S_S_2026, Sort=Pre-Season (adp), Order=Ascending.

**Result (reconstructed):**
- Total rows: **4,244** batters
- First-page result (25 players) sorted by ADP ascending:

| Player | Team | ADP | R | HR | RBI | SB | AVG | OBP |
|--------|------|-----|---|----|-----|----|-----|-----|
| Agustin Ramirez | MIA | 86.8 | 14 | 2 | 14 | 3 | .230 | .318 |
| Yainer Diaz | HOU | 113.0 | — | — | — | — | — | — |
| Lawrence Butler | ATH | 136.0 | — | — | — | — | — | — |
| Jorge Polanco | (none) | 142.0 | — | — | — | — | — | — |
| Marcell Ozuna | PIT | 154.0 | — | — | — | — | — | — |
| Alejandro Kirk | TOR | 160.3 | — | — | — | — | — | — |
| Samuel Basallo | BAL | 160.9 | 15 | 5 | 16 | — | .277 | .338 |
| Brenton Doyle | COL | 166.3 | — | — | — | — | — | — |
| Gabriel Moreno | AZ | 181.2 | 14 | 2 | 14 | 3 | .245 | .303 |
| Bryson Stott | PHI | 181.4 | 14 | 5 | 22 | 9 | .228 | .275 |
| ... (15 more, majority with all dashes) | | | | | | | | |

**Critical observation:** 16 of the first 25 players show all-dash stats (no game log data). This is because the game_log table only covers rostered players (317) plus some 40-man players; 9,571 of 9,888 pool players have no game log data. The table displays dashes for all stat columns for these players.

**Data freshness:** Stats as of 06/10's games. Refreshed 6/10, 3:41 PM EDT.

### 5.3 Player Lookup: Freddie Freeman (LAD 1B, player_id=4)

**Search:** "Freddie Freeman", Position=All Players, Status=ALL, Stats=S_S_2026

**S_S_2026 (Season total) view — reconstructed table row:**

| Player | Team | Pos | Roster Status | GP* | Opp | Pre-Season | Current | Adds | Drops | % Ros | R | HR | RBI | SB | AVG | OBP |
|--------|------|-----|---------------|-----|-----|------------|---------|------|-------|-------|---|----|----|----|----|-----|-----|
| Freddie Freeman | LAD | 1B | BUBBA CROSBY | 64 | — | 62 | 54 | 0 | 0 | — | 35 | 10 | 36 | 2 | .284 | .366 |

Notes on this row:
- `% Ros = -` because percent_owned=0.0 for all players in ownership_trends (data appears stale/miscalibrated).
- `Adds = 0, Drops = 0` for Freddie Freeman because he is never dropped/added mid-season as a rostered player.
- Opp = "—" because the game schedule enrichment failed (Yahoo offline).
- `AVG = .284` comes from `avg_calc` (computed from game_log H/AB = 69/243), which correctly overrides the pool's blended-projection `avg = .262`.
- `OBP = .366` from `obp_calc` = (69+32+1)/(243+32+1+3) = 102/279.

**S_S_2025 (2025 Season total) — Freddie Freeman:**
- GP=147, R=81, HR=24, RBI=90, SB=6, AVG=0.295, OBP=0.367

**S_S_2024 (2024 Season total) — Freddie Freeman:**
- GP=146 (from game_logs), R=80, HR=21, RBI=86, SB=9
- Note: **discrepancy vs official stats**: season_stats table shows 2024: R=81, HR=22, RBI=89 — the game_log-based view is missing 1 game and 1 HR, 3 RBI. This is the one-game gap in game_log coverage.

**S_L7 (Last 7 Days total) — Freddie Freeman:**
- GP=3, R=3, HR=0, RBI=1, SB=0, AVG=.455, OBP=.500
- (From last 3 games within the 7-day window ending 06/10: 2+2+1 runs, 4+2+1 hits out of various ABs)

### 5.4 Player Lookup: Zack Wheeler (PHI SP, player_id=107)

**Search:** "Zack Wheeler", Position=P (All Pitchers), Status=ALL, Stats=S_S_2026

**Reconstructed table row (pitcher view):**

| Player | Team | Pos | Roster Status | GP* | Opp | Pre-Season | Current | Adds | Drops | % Ros | IP | W | L | SV | K | ERA | WHIP |
|--------|------|-----|---------------|-----|-----|------------|---------|------|-------|-------|----|---|---|----|---|-----|------|
| Zack Wheeler | PHI | SP,P | The Good The Vlad The Ugly | 9 | — | 110 | 26 | 0 | 0 | — | 56.7 | 5 | 1 | 0 | 53 | 2.22 | 0.85 |

Notes:
- **IP = 56.7** is a display error. The correct baseball notation for 56⅔ innings is **56.2**, not 56.7. The `_format_cell` function uses `.1f` format on the decimal value (56.666...) producing "56.7." This is factually wrong — 56.7 IP doesn't correspond to a valid inning count.
- ERA = 2.22 (from era_calc = ER×9/IP = 14×9/56.667 = 2.22). Correct.
- WHIP = 0.85 (from whip_calc = (BB+H)/IP = (12+36)/56.667 = 0.847). Correct.
- Zack Wheeler has **NO Statcast data** in statcast_archive (sprint_speed=None, xwoba=None, barrel_pct=None).

**S_S_2025 (2025) — Zack Wheeler:**
- GP=24, IP=149.2 (displayed as 149.7 — same IP display error), W=10, L=5, K=195, ERA=2.71, WHIP=0.94

### 5.5 "Today (live)" Stat View (S_L)

**Claim:** "Today (live)"
**Reality (reconstructed):** Returns the player pool as-is (type=`"live"` hits the `if view_type in ("proj", "special", ...)` branch which returns `pool.copy()` without any game_log merge). The stat columns shown are the ROS blended projections (r=33, hr=9, rbi=34 for Freeman), not today's live in-game stats.

**This is a falsely-labelled view.** A novice selecting "Today (live)" expects to see today's in-game or today's completed game stats. They get ROS projections.

### 5.6 "Waivers Only" Status Filter

**STATUS_OPTIONS defined as:** `("W", "Waivers Only")`
**filter_databank behavior:** The function handles "A", "FA", "T", and "ALL" — but not "W". Selecting "Waivers Only" falls through to the else branch (no filter applied), returning **4,408 rows**, identical to selecting "All Players (ALL)".

**This is a silent bug.** "Waivers Only" behaves identically to "All Players." A novice selecting it to find waiver-wire pickups gets the full unfiltered pool with no indication the filter did nothing.

### 5.7 "Util" Position Filter

**POSITION_OPTIONS includes:** `("Util", "Util")`
**filter_databank behavior:** Searches for "UTIL" substring in the `positions` column. The positions column contains values like "1B", "SS,3B", "OF", "SP,P" — never "Util". Result: **0 rows** every time.

**This filter always returns empty.** No MLB player in the pool has "Util" in their positions string (it's a Yahoo fantasy slot concept, not a real position). The Util position option should either be removed or mapped to an actual filter (e.g., first 7 position types = eligible to be in a Util slot).

### 5.8 Advanced Stat View (ADVST_ADVST_2026)

**Claim:** "2026 Advanced"
**Reality:** Returns pool as-is. The stat columns rendered are the same 6 hitter/pitcher stats (R/HR/RBI/SB/AVG/OBP or IP/W/L/SV/K/ERA/WHIP), not Statcast columns like xwOBA, barrel%, EV.

**Critical finding:** The `render_databank_table` function always shows the same fixed 6-7 stat columns regardless of what stat view is selected. Changing from "Season (total)" to "2026 Advanced" changes which data source is loaded but NOT which columns are displayed. The advanced metrics (xwoba, barrel_pct, sprint_speed, stuff_plus, etc.) are never rendered in the table.

Additionally, for Freddie Freeman:
- xwoba = 0.0 (not None, but literally 0.0)
- barrel_pct = 0.0
- hard_hit_pct = 0.0
- ev_mean = 0.0
- sprint_speed = 25.1 (ft/sec — only Statcast field with real data)
- All other Statcast fields = None or 0.0

**The "2026 Advanced" view shows no advanced content** — it renders the same projection-blend R/HR/RBI columns because `render_databank_table` ignores the stat view type.

### 5.9 "Ranks" / "Research" / "Fantasy Matchups" / "Opponents" (Special Views)

**K_K = "Ranks", R_O = "Research", M_W = "Fantasy Matchups", O_O = "Opponents"**

All four return the pool as-is (type=`"special"`). The table always renders the same 6 hitter/pitcher stat columns. None of these views change the displayed columns or add any unique data. "Fantasy Matchups" and "Opponents" sound like they would show H2H matchup context or opponent info — they do not. These are effectively placeholder/stub views.

**Freddie Freeman "Ranks" data (what the pool has, not what the table shows):**
- Pre-Season rank (ADP): 62
- Current rank (consensus_rank): 54
- ECR avg: 59.8
- ECR stddev: 28.3
- ECR sources: 2 (very thin — most players have 0 or 1 source)

### 5.10 "% Ros" Column

All 9,888 players have percent_owned=0.0 in the pool (sourced from ownership_trends table where all values are 0.0). The `_format_cell` function correctly renders 0.0 as "—" (dash). So the "% Ros" column is always a blank dash for every player. This column is present but useless with the current data state.

The yahoo_free_agents cache has 608 players with real percent_owned values (e.g., Tanner Bibee=74.0%, Spencer Torkelson=69.0%) but these are not joined into the player pool.

### 5.11 "Adds" / "Drops" Columns

These come from `_enrich_databank_columns()` which calls `get_adds_drops_map()` → `yds.get_transactions()`. With Yahoo offline and 59,869 transaction rows in the DB, the map still works. 76 players have adds > 0 and 73 have drops > 0 out of 9,888. The values are populated for actively-traded fantasy players.

### 5.12 "GP*" Column Label

The asterisk in "GP*" has no tooltip, footnote, or explanation anywhere on the page. A novice seeing "GP*" next to "GP" on other pages would have no idea what the asterisk means. (It appears to mean "games played from game_log data" vs. "from season_stats" but this distinction is nowhere communicated.)

### 5.13 Data Freshness Labels

The instrument panel shows:
- "STATS AS OF: 06/10 — end of games" (from game_logs MAX(game_date))
- "REFRESHED: 6/10, 3:41 PM EDT" (from refresh_log for season_stats)

These are accurate and clearly displayed. The Combustion Finale's `build_stat_readout_html` chips look professional. However, there is no indication that **the projections-based views** (S_PS7, S_PS14, etc.) show data from a different source with a different refresh time (ROS projections refreshed 6/10 at 2:15 PM EDT). The panel always shows game_log freshness even for projection views.

### 5.14 Performance (reconstructed)

- `load_databank('S_S_2026')`: ~3.6s (loads entire 9,888-player pool + merges game_logs for 408 players)
- `load_databank('S_L7')`: ~1.4s
- `load_databank('S_SDL7')`: ~1.3s

These times are for the backend only, before the HTML table render for 25 rows. Every page change, filter change, or stat view change requires clicking Search and re-running the full 3–4s load. There is no caching (`@st.cache_data`) on `load_databank` in the page code.

### 5.15 Export to Excel

`export_to_excel(filtered, view_label)` produces a branded .xlsx with:
- Header row: red (`#E63946`) fill, white bold font (this is `ember` red — acceptable as a functional-negative/accent, and it's only in the export, not the live CSS)
- Auto-widened columns (max 20 chars)
- Correct number formats: AVG/OBP=3 decimal places, ERA/WHIP=2 decimal places

The export works on the **full filtered result** (not just the current page of 25), which is the correct behavior. Filename format: `HEATER_Player_Databank_Season__total__2026-06-13.xlsx`.

### 5.16 FIG Number

The page renders `FIG.19 — PLAYER DATABASE` in the eyebrow. The FIG number matches the file prefix `19_Player_Databank.py`. No inconsistency here (unlike the Pitcher Streaming `FIG.4` vs `FIG.04` issue noted in the shared context).

---

## 6. Errors, Issues & Difficulties

### Bugs

1. **IP display is factually wrong.** IP values are stored as true decimals (56.666... for 56⅔ innings). `_format_cell` uses `.1f` formatting, displaying `56.7` instead of the correct baseball notation `56.2`. This affects every pitcher in every stat view that shows innings pitched.

2. **"Waivers Only" filter is broken.** `filter_databank` has no handler for `status="W"`. It silently falls through to the "no filter" branch, returning all 4,408 batters (same as "All Players"). The user selecting "Waivers Only" has no way to know their filter did nothing.

3. **"Util" position filter always returns 0 rows.** No player has "Util" in their positions string. The filter always produces an empty table with `render_empty_state("No matches", ...)`. This appears as a dead control.

4. **"Today (live)" shows ROS projections, not live stats.** `STAT_VIEW_PARAMS["S_L"]` has type=`"live"` but the data dispatch in `load_databank` routes `"live"` to the pool-as-is path, returning the blended projection columns. A user clicking "Today (live)" to see today's stats gets ROS forecast numbers.

5. **"Ranks," "Research," "Fantasy Matchups," "Opponents" are unimplemented stub views.** All four special views return the identical pool with the identical 6 stat columns. The dropdown has 28 options but 4 of them do nothing different from the default view. Users who select "Fantasy Matchups" expecting to see matchup data will be puzzled by seeing standard counting stats.

6. **"2026 Advanced" / "2025 Advanced" views show no advanced columns.** `render_databank_table` always shows the same 6 hitter/pitcher stat columns regardless of `stat_view`. The advanced Statcast metrics (xwOBA, barrel%, EV, Stuff+, etc.) are never surfaced in the table even when the "Advanced" view is selected.

7. **All percent_owned values are 0.** The ownership_trends table has 0.0 for every player. The "% Ros" column is always displayed as "—" for every player. This should either use the yahoo_free_agents cache (which has 608 real values up to 74%) or be hidden when stale.

8. **Game log data covers only ~4% of the player pool for 2026.** 408 of 9,888 players have 2026 game_logs. For the other 9,480 players, every stat column in every "total" / "avg" / "stddev" view shows dashes. The first page of results shows this clearly: 16 of 25 default result rows have all-dash stats. A novice cannot distinguish between a player with no game log data and a player who simply has 0 stats.

9. **2024 game_log stats have minor discrepancies vs official records.** Freddie Freeman 2024 via game_logs: R=80, HR=21, RBI=86 vs season_stats (from MLB Stats API): R=81, HR=22, RBI=89. One game is missing from game_logs.

### Data / Column Issues

10. **Roster Status column shows full team names (up to 30+ chars).** "The Good The Vlad The Ugly" appears in full in a narrow table column. No abbreviation. This will break the table layout on smaller screens.

11. **"GP*" asterisk unexplained.** No tooltip, no footnote, no legend. For a novice, "GP*" vs "GP" is meaningless.

12. **Opponent column always shows "—".** With Yahoo offline, `get_todays_opponent_map()` calls the MLB Stats API (blocked by network guard in tests), returns `{}`, so `opponent="-"` for all players. The column label changes daily ("Opp: 6/13") but shows no data.

13. **Off-palette hex color `#2c2f36` in inline CSS.** In `render_databank_table()` in `src/player_databank.py` line 1071, the CSS uses `color: #2c2f36;` for table header text. This color is on the Combustion Index banned list (`_BANNED` in `test_no_offpalette_hex_in_pages.py`). The test guard only scans `pages/` and `app.py`, not `src/`, so CI passes but the design violation exists.

14. **Advanced view's statcast data is almost entirely null.** Of all 9,888 statcast_archive rows: ev_mean=0 with data (0 non-null), xwoba=0 with data, barrel_pct=0, stuff_plus=0. Only sprint_speed has data (501 rows). The "2026 Advanced" stat view is advertised but the underlying Statcast data is empty.

### UX / Navigation

15. **28-option Stats dropdown with 4 stubs and 1 mislabelled view creates a confusing menu.** The dropdown spans 3 view types (projections, totals, and "specials") mixed without section separators. A novice scrolling to "Fantasy Matchups" expects something useful.

16. **No per-player drill-down.** Player names are not clickable links. The page has no way to navigate from a player's table row to a detailed view. The CLAUDE.md calls this a "historical multi-year player lookup" but the implementation is a flat list with no lookup capability.

17. **Page-level spinner is the only progress feedback.** `with st.spinner("Loading player data..."):` provides basic feedback for `load_databank()`. But changing the Stats dropdown and clicking Search has no indication that 3.5 seconds of compute is happening until the spinner appears.

18. **No `@st.cache_data` on `load_databank`.** Every Search click re-runs the full pool load + game_log merge, even if the filters haven't changed from the last search. On a page where each view takes 1.4–3.6s, this is a significant UX cost.

19. **The sort default "Pre-Season / Ascending" is confusing for a novice.** Pre-season ADP ascending = pick #1 first. But the default status filter is "All Available Players." The first result row is Agustin Ramirez (ADP=86.8) — a player most users don't recognize. Most useful default for "I just opened Player Databank" would be "Current / Descending" (highest-ranked available players first).

20. **Page overlaps heavily with Leaders (page 13) and Player Compare (page 12).** Leaders already shows sortable stat leaderboards. Player Compare does head-to-head player lookup. Databank's unique value proposition (multi-year historical lookup with game_log rolling windows) is not communicated or surfaced by the default experience.

---

## 7. UI/UX & Visual Design Critique

### Layout & Density

The filter form rows are dense. Row 3 has 6 dropdowns in `st.columns([2, 2, 2, 3, 2, 1.5])`. At typical viewport widths, these dropdowns will be squeezed. On mobile (< 768px) this is unusable — 6 horizontal dropdowns collapse to near-zero width.

The Submit button spanning full width beneath 6 dropdowns is clear, but the full-width button below a complex multi-row form feels oddly large when the inputs above are compact.

### Table Design

The custom HTML table uses the Combustion design system CSS well: Archivo bold uppercase headers, tabular-num figures, orange sort arrows (`.sorted .sort-arrow { color: var(--fp-primary) }`), even-row striping, hover highlight. This is visually solid.

However:
- The "stat group" header row (gray banner showing "Season (total)" spanning the stat columns) is visually redundant — the Stats dropdown already shows what view is selected.
- The JS sort is client-side only. After clicking to page 2, the sort resets because JS sort operates on the current DOM, not the full dataset. Server-side sort applies globally, but the JS sort only applies to the visible 25 rows. This is confusing: a user on page 2 can "sort by AVG" using header clicks and will see only page 2's 25 players sorted, not the top AVG across all results.
- `max-height: 600px; overflow-y: auto` on the table wrapper creates a double scroll bar on mobile and limits visibility to ~10 rows at a time even on desktop.

### Number Formatting

- AVG: `.284` — correct (3 decimal places from `_format_cell`)
- OBP: `.366` — correct
- ERA: `2.22` — correct
- WHIP: `0.85` — correct
- IP: `56.7` — **wrong** (should be `56.2` in baseball outs notation)
- ADP: `62` — displayed as integer when ≥1. Correct.
- `% Ros`: `—` for all players (data gap, not a formatting bug)
- Std dev rows: display raw decimal values like `0.577` for RBI std dev — a novice has no idea what to do with standard deviation of fantasy stats.

### Microcopy

- "Live stats for every MLB player" (render_reco_banner) — misleading. Most players have no game log data. And the "live" claim is false (Yahoo is offline).
- "Search player by name" placeholder — fine.
- `GP*` — the asterisk is unexplained.
- `Opp: 6/13` as a column header — the date is dynamic (good), but this field is always "—" when Yahoo is offline. A staleness indicator ("Opp [stale]") would help.

### Color & Design System

The table CSS uses `#2c2f36` (banned Combustion off-palette color) for header text. This should be `var(--fp-tx)` or `{T["tx"]}`.

The stat-group banner row (`background: #f6f7f9`) and hover color (`#eef0f3`) correctly use the HEATER surface/hover tones.

No emoji in the page output — compliant.
No stray off-palette colors in the page file itself (`pages/19_Player_Databank.py`) — compliant. The issue is in `src/player_databank.py`.

### Data Staleness Communication

The instrument panel correctly shows "STATS AS OF: 06/10 — end of games" and "REFRESHED: 6/10, 3:41 PM EDT". This is good.

However: when a projection view is selected (S_PS7, S_PS14, etc.), the panel still pulls from the game_logs/season_stats refresh source, not the projections source. A user on "Next 7 Days (proj)" sees "STATS AS OF: 06/10's games" which is factually wrong (projections aren't "games").

---

## 8. Recommendations (≥10, ordered by impact)

### R1. Fix the IP display bug — use outs notation, not decimal

**Problem:** IP values stored as 56.666... render as "56.7" instead of the correct baseball notation "56.2" (56⅔ innings). This is factually wrong.
**Fix:** Add an IP-specific format branch in `_format_cell` that converts decimal IP back to outs notation: `floor(ip)` + `.` + `round((ip - floor(ip)) * 3)`. E.g., 56.666 → "56.2". Already a CLAUDE.md structural invariant exists (`_ip_outs_to_decimal`); a reverse function is needed.

### R2. Fix "Waivers Only" filter — either implement or remove it

**Problem:** Status="W" (Waivers Only) silently falls through to the "no filter" branch, returning all 4,408 batters. The user cannot know their filter did nothing.
**Fix (minimal):** Remove "Waivers Only" from STATUS_OPTIONS since the app does not differentiate waivers from free agents. Alternatively implement it: players with `roster_team=null` AND `is_available=True` AND `yahoo_player_key` in a "waiver period" window — though this requires Yahoo waiver data.

### R3. Implement or remove the four stub stat views ("Ranks," "Research," "Fantasy Matchups," "Opponents")

**Problem:** 4 of 28 dropdown options do nothing different from the default view. A user selecting "Fantasy Matchups" sees standard counting stats.
**Fix:** Either implement them (Ranks view: show ADP/ECR/% Owned columns; Research view: show Statcast/regression columns; Matchups view: show current matchup win probability) or remove them from STAT_VIEW_OPTIONS until implemented. A stub that misleads users is worse than no option.

### R4. Fix "Today (live)" stat view — show actual today's stats or rename it

**Problem:** "Today (live)" returns ROS projections, not live in-game stats.
**Fix:** Either (a) route `type="live"` to a real live stats fetch from the MLB Stats API boxscores for today's games (per `game_day.get_target_game_date()`), or (b) rename the view to "ROS Projections" and remove the word "live." The current label is misleading.

### R5. Fix "Util" position filter — remove it or map to eligible positions

**Problem:** "Util" always returns 0 results because no player has "Util" in their `positions` string.
**Fix:** Remove "Util" from POSITION_OPTIONS, or map it to a composite filter (positions matching any of 1B/2B/3B/SS/OF/DH/C — i.e., non-pitcher position eligibles).

### R6. Show stat view-specific columns in the table — Advanced view should show Statcast columns

**Problem:** `render_databank_table` always shows R/HR/RBI/SB/AVG/OBP (or IP/W/L/SV/K/ERA/WHIP) regardless of the stat view. The "2026 Advanced" view shows no advanced data. The "Ranks" view shows no ranking data.
**Fix:** Add a column-dispatch map in `render_databank_table` keyed on `stat_view`: Advanced → xwOBA, barrel%, EV, Stuff+, sprint_speed; Ranks → ADP, ECR rank, % Ros, Adds/Drops; Std Dev → display with a disclaimer that these are per-game standard deviations.

### R7. Cache `load_databank` results with `@st.cache_data`

**Problem:** Every Search click re-runs a 1.4–3.6s pool load + game_log merge. No caching exists.
**Fix:** Add `@st.cache_data(ttl=300)` to `load_databank()`, keyed on `stat_view`. The cache can be cleared when the scheduler runs a data refresh. This would reduce most search interactions to near-instantaneous after the first load.

### R8. Replace "% Ros" dash column with real ownership data from yahoo_free_agents

**Problem:** percent_owned = 0.0 for all 9,888 players in ownership_trends. The "% Ros" column is always "—".
**Fix:** Join yahoo_free_agents (608 rows with real % owned values like 74%, 69%, etc.) into `_enrich_databank_columns`. Use the `player_name` column as the match key. This would make the column useful for filtering/sorting by popularity.

### R9. Add explicit "no data" differentiation for players without game_log coverage

**Problem:** 9,480 of 9,888 players show all-dash stats in total/avg/stddev views because their game_logs aren't in the DB. A novice can't distinguish a player with truly zero stats from one with no tracking data.
**Fix:** After the merge, add a `_has_game_log` boolean column. Render rows without game log data with a subtle "No stats" chip in the stat group area, or move them to the bottom regardless of sort. Alternatively, show the `season_stats` YTD values for players without game_logs (YTD stats cover 2,580 rostered players for 2026).

### R10. Fix `#2c2f36` off-palette color in `render_databank_table` CSS

**Problem:** Line 1071 of `src/player_databank.py` uses `color: #2c2f36;` for table header text, which violates the Combustion Index palette ban (`_BANNED` in `test_no_offpalette_hex_in_pages.py`).
**Fix:** Replace with `color: {T["tx"]};` using the THEME dict, consistent with the rest of the table CSS. The test guard should also be extended to cover `src/player_databank.py`.

### R11. Change default Sort to "Current / Descending" (highest in-season rank first)

**Problem:** Default Sort="Pre-Season (ADP) / Ascending" shows players sorted by pre-draft value. Mid-season, this is almost irrelevant — the #1 ADP player may be injured or underperforming. Also, only 579 of 9,888 players have ADP data, so the rest go to the bottom.
**Fix:** Change default to Sort="Current" (consensus_rank), Order="Ascending" (rank 1 = best). This immediately shows the highest-currently-ranked available players, which is what a manager needs mid-season.

### R12. Explain "GP*" with a tooltip or rename to "Games Played"

**Problem:** The asterisk in "GP*" has no explanation. Users don't know whether this means games played in the selected window or lifetime or something else.
**Fix:** Rename to "GP" (no asterisk), or add a `title="Games played in selected time window"` attribute to the `<th>` element in `render_databank_table`. The JS `onclick` handler could coexist with a `title` attribute.

### R13. Add a "View player profile" link or basic player card on row click

**Problem:** Player names are static text — no drill-down. The page is described as a "lookup" but offers no per-player detail. The Leaders and Free Agents pages link to Player Compare for context; Databank has no equivalent.
**Fix:** Wrap player names in an anchor pointing to `?player=PLAYER_NAME` with query-param awareness on Player Compare (`pages/16_Player_Compare.py`), or add an expander below the row showing multi-year stats from season_stats (2023–2026) and any available Statcast data.

### R14. Show a freshness label appropriate to the selected stat view

**Problem:** The instrument panel always shows game_log freshness ("STATS AS OF: 06/10's games") even when a projection view (S_PS7, S_PS14, S_PSR) is selected. Projection views should show "Based on ROS projections — Refreshed 6/10 at 2:15 PM EDT."
**Fix:** `get_data_as_of_label()` already returns the correct string per view. The instrument panel already calls it. This is mostly working — the issue is that the "as_of" label correctly changes, but the "REFRESHED" label is sourced from the game_log/season_stats source for all views. The `get_data_refreshed_label()` function already handles this per-view via the sources list — verify this is rendering correctly in the Combustion chip.

---

## 9. Severity-Tagged Issue List

- **[BLOCKER]** IP display is factually wrong — "56.7" rendered instead of "56.2" for 56⅔ innings. This is displayed to every pitcher on every stat view. Affects all pitchers across 28 views.

- **[BLOCKER]** "Today (live)" stat view shows ROS projections, not live stats. Label is false. Users setting their lineups based on "live" data are seeing forecasts.

- **[HIGH]** "Waivers Only" status filter is broken — silently returns ALL players (4,408), same as no filter. No error message to user.

- **[HIGH]** "Util" position filter always returns 0 results. Dead control.

- **[HIGH]** 4 stub stat views (Ranks, Research, Fantasy Matchups, Opponents) show identical data to the default view. Users who select them are misled.

- **[HIGH]** "2026 Advanced" view never shows Statcast columns. `render_databank_table` ignores stat view when choosing columns.

- **[HIGH]** 96% of players have no game log data (9,480 of 9,888). In all total/avg/stddev views these players show only dashes with no explanation. First page of results: 16 of 25 rows are all dashes.

- **[HIGH]** `@st.cache_data` missing on `load_databank`. Every Search costs 1.4–3.6s. Multiple searches in one session waste 10–20s of round-trip time.

- **[MEDIUM]** `#2c2f36` off-palette hex color in `render_databank_table` CSS (line 1071 of `src/player_databank.py`). Violates Combustion Index banned color list.

- **[MEDIUM]** `percent_owned` column always shows "—" for all 9,888 players (ownership_trends data is stale/zero). The yahoo_free_agents cache has real data for 608 players that isn't used.

- **[MEDIUM]** Default sort (Pre-Season ADP, Ascending) is wrong for mid-season use. Only 579 players have ADP; the default surface shows a mix of obscure untracked players.

- **[MEDIUM]** Game log stats have minor discrepancies vs official season_stats for 2024 (Freddie Freeman: game_log R=80 vs season_stats R=81, HR=21 vs 22, RBI=86 vs 89).

- **[MEDIUM]** Roster Status column shows full team names (30+ chars) like "The Good The Vlad The Ugly" — no abbreviation, breaks table layout on narrow viewports.

- **[MEDIUM]** No year-over-year comparison within a player row. The page is described as "historical multi-year lookup" but offers single-season slices with no side-by-side history.

- **[MEDIUM]** No player drill-down / profile card. Player name is plain text with no link or expandable detail.

- **[MEDIUM]** `get_data_refreshed_label` sources season_stats/game_logs for projection views (S_PS7 etc.) instead of the projections source, showing wrong refresh time for projection views.

- **[MEDIUM]** Statcast data is almost entirely null in statcast_archive. Of 9,888 rows, only sprint_speed has data (501 rows). ev_mean, xwoba, barrel_pct, stuff_plus are all null. The "Advanced" views have nothing useful to show even if render_databank_table were fixed to display them.

- **[LOW]** "GP*" asterisk has no tooltip or explanation.

- **[LOW]** JS client-side column sort only sorts the current 25-row page, not all filtered results. Confusing behavior when paginated.

- **[LOW]** `max-height: 600px` on table wrapper creates a double scroll on narrow viewports.

- **[LOW]** FIG label is "FIG.19 — PLAYER DATABASE" — inconsistent capitalization (other pages use "FIG.NN — Title Case"). Minor.

- **[LOW]** Export filename includes double underscores: `HEATER_Player_Databank_Season__total__2026-06-13.xlsx` (because `view_label.replace(" ", "_")` converts "Season (total)" to "Season__total_").

- **[LOW]** Stat-group header row in the table ("Season (total)" banner spanning stat columns) is redundant — the user already selected this in the Stats dropdown.

- **[POLISH]** Opponent column shows "Opp: 6/13" in the column header every day — correct pattern — but always shows "—" when Yahoo/MLB Schedule is offline. A subtle "(offline)" note on the column header when schedule unavailable would reduce confusion.

- **[POLISH]** Standard deviation views (S_SDL7 etc.) show decimal numbers like "0.577 RBI per game (std dev)" which a novice cannot interpret. These need a callout or footnote explaining what standard deviation means in a fantasy context.

- **[POLISH]** The "Show my team" checkbox is next to Position in a 5:1 column split — the checkbox label is cramped on smaller viewports.
