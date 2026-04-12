# Player Databank — Design Spec

**Date:** April 12, 2026
**Status:** Approved

---

## Overview

A new HEATER page (`pages/2_Player_Databank.py`) that replicates Yahoo Fantasy Baseball's "Player List" tab with HEATER branding. Provides a live, filterable, sortable player database showing all MLB players with stats across every H2H scoring category. Custom HTML table with JavaScript sorting, 28 stat view options, 5-axis filtering, and Excel export.

## Page Placement

- **File:** `pages/2_Player_Databank.py` (appears after "My Team" in sidebar)
- **Support module:** `src/player_databank.py` (data assembly, filtering, rendering, export)
- **New DB table:** `game_logs` (per-game stats for rolling windows, averages, std dev)

## Filter Bar (5 axes + search + checkbox)

### 1. Position Filter (pill-style radio buttons, custom HTML)
Options: All Batters (default), All Pitchers, P (Probable), C, 1B, 2B, 3B, SS, OF, Util, SP, RP
Plus: "Show my team" checkbox

### 2. Status Dropdown
Options: All Players, All Available Players (default), Available Players (today), Available Players (tomorrow), Available IL Eligible, Free Agents Only, Waivers Only, Keepers Only, My Watch List, All Taken Players

### 3. MLB Teams Dropdown
Options: All Teams (default) + 30 MLB teams

### 4. Fantasy Teams Dropdown
Options: No Team Selected (default) + all 12 league teams from Yahoo API

### 5. Stats Dropdown (28 options across 6 categories)

**Projected Stats:** Next 7 Days (proj), Next 14 Days (proj), Remaining Games (proj)

**Season Stats (totals):** Today (live), Last 7/14/30 Days (total), Season (total), 2025 Season (total), 2024 Season (total)

**Advanced Stats:** 2026, 2025

**Average Stats:** Last 7/14/30 Days (avg), Season (avg), 2025 Season (avg), 2024 Season (avg)

**Standard Deviation:** Last 7/14/30 Days (std dev), Season (std dev), 2025 Season (std dev), 2024 Season (std dev)

**Fantasy:** Ranks, Research, Fantasy Matchups, Opponents

### 6. Search
Text input: "Search player by name"

## Table Design (Custom HTML + JavaScript)

### Batting Columns
Player Name, Opp, Roster Status, GP*, Pre-Season rank, Current rank, % Ros, H/AB*, R, HR, RBI, SB, AVG, OBP

### Pitching Columns
Player Name, Opp, Roster Status, GP*, Pre-Season rank, Current rank, % Ros, IP*, W, L, SV, K, ERA, WHIP

### Advanced Stats Columns
Batters: xwOBA, xBA, Barrel%, Hard Hit%, EV, BABIP, K%, BB%
Pitchers: xwOBA (against), Stuff+, K%, BB%, BABIP, FIP, xFIP

### Styling (HEATER thermal theme)
- Card: white with glassmorphic shadow
- Header: #1d1d1f background, white text
- Sortable headers: clickable with triangle indicators
- Row hover: #fff7ed (light amber)
- Current rank: #e63946 (flame)
- Player links: #457b9d (sky blue)
- Stat formatting: AVG/OBP .3f, ERA/WHIP .2f, counting stats integer

### JavaScript Sorting
Click header to toggle asc/desc sort. Numeric sort for stats, alpha for text. State persisted in session_state.

## Excel Export

- Button: top-right, HEATER-branded (flame gradient)
- Exports current filters/sort as .xlsx
- Filename: HEATER_Player_Databank_{view}_{date}.xlsx
- Header row: flame background, white bold text
- Uses openpyxl (existing dependency)

## Data Layer

### New Table: game_logs
21 columns (player_id, game_date, season + batting/pitching stats), PK on (player_id, game_date)

### Bootstrap
New phase with 1hr staleness (current season), 30d (historical 2024/2025)
Source: MLB Stats API game logs

### Computation
Rolling windows, averages, std dev computed on-the-fly from game_logs via pandas

## Key Function Signatures

```python
def load_databank(stat_view: str, season: int = 2026) -> pd.DataFrame
def filter_databank(df, position, status, mlb_team, fantasy_team, search, show_my_team) -> pd.DataFrame
def compute_rolling_stats(player_ids, days, stat_type) -> pd.DataFrame
def render_databank_table(df, stat_view, sort_col, sort_dir, is_pitcher) -> str
def export_to_excel(df, stat_view_label) -> bytes
def fetch_game_logs_from_api(season, force) -> int
```
