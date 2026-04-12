# Player Databank — Implementation Plan

## Context

Connor wants a new HEATER page (`pages/2_Player_Databank.py`) that replicates Yahoo Fantasy Baseball's "Player List" tab with HEATER branding. The page provides a live, filterable, sortable player database showing all MLB players with stats across every H2H scoring category. Uses a custom HTML table with JavaScript sorting (Approach B), 28 stat view options, 5-axis filtering, and Excel export.

The design was brainstormed and approved across 4 design sections. The user confirmed:
- Approach B (custom HTML table with JS sorting)
- HEATER branding/colorways throughout
- All 28 stat dropdown options relevant to H2H Categories
- Excel export button preserving current filters/sort/formatting
- Page named "Player Databank", positioned after "My Team" as `pages/2_Player_Databank.py`

## Implementation: 9 Tasks

### Task 1: Database Schema — game_logs Table
- **Modify:** `src/database.py` — add `CREATE TABLE IF NOT EXISTS game_logs` with 21 columns (player_id, game_date, season, pa, ab, h, r, hr, rbi, sb, bb, hbp, sf, ip, w, l, sv, k, er, bb_allowed, h_allowed) and PRIMARY KEY (player_id, game_date)
- **Test:** `tests/test_player_databank.py` — verify table exists, columns correct, PK enforced

### Task 2: Game Log Fetching from MLB Stats API
- **Create:** `src/player_databank.py` with constants (STAT_VIEW_OPTIONS, STAT_VIEW_PARAMS, HITTING_COLS_TOTAL, PITCHING_COLS_TOTAL) + `load_game_logs()` and `fetch_game_logs_from_api()` functions
- Uses existing `src/database.py:get_connection()`, `statsapi.player_stat_data()`
- **Test:** load_game_logs with empty DB, with inserted test data, date range filtering

### Task 3: Rolling Stats Computation
- **Modify:** `src/player_databank.py` — add `compute_rolling_stats(player_ids, days, stat_type)` supporting "total", "avg", "stddev"
- Computes rate stats properly (AVG=h/ab, OBP=(h+bb+hbp)/(ab+bb+hbp+sf), ERA=er*9/ip, WHIP=(bb+h)/ip)
- Uses existing `load_game_logs()` from Task 2
- **Test:** total sums correctly, avg divides by GP, stddev > 0 for varying data

### Task 4: Data Assembly — load_databank() and filter_databank()
- **Modify:** `src/player_databank.py` — add `load_databank(stat_view)` that dispatches to correct computation based on STAT_VIEW_PARAMS, and `filter_databank(df, position, status, mlb_team, fantasy_team, search, show_my_team)`
- Uses existing `src/database.py:load_player_pool()` for base data
- 28 stat views handled via parameter dispatch (season_total, rolling total/avg/stddev, projected, advanced, ranks, research, matchups, opponents)
- **Test:** returns DataFrame, has required columns, position filter works, search filter works, team filter works

### Task 5: Custom HTML Table Renderer
- **Modify:** `src/player_databank.py` — add `render_databank_table(df, stat_view, sort_col, sort_dir, is_pitcher)` and `_format_cell(value, col)`
- HEATER thermal theme: header `#1d1d1f`, hover `#fff7ed`, flame accent `#e63946`, sky links `#457b9d`
- Uses existing `src/ui_shared.py:T` (theme dict)
- Embedded JavaScript `sortTable()` function for client-side column sorting
- Batting columns: R, HR, RBI, SB, AVG, OBP; Pitching columns: IP, W, L, SV, K, ERA, WHIP
- **Test:** returns HTML string, contains table tag, includes sortTable JS, formats AVG as .3f, shows ERA/WHIP for pitchers

### Task 6: Excel Export
- **Modify:** `src/player_databank.py` — add `export_to_excel(df, stat_view_label)` returning bytes
- Uses existing `openpyxl` (in requirements.txt)
- HEATER branding: flame header fill (#E63946), white bold font, auto-column-width
- Stat formatting preserved (.3f for AVG/OBP, .2f for ERA/WHIP)
- **Test:** returns bytes, valid XLSX (starts with PK header)

### Task 7: Streamlit Page
- **Create:** `pages/2_Player_Databank.py` — full page with search bar, position pill selector, 4 filter dropdowns, export button, player count, HTML table
- **Modify:** `src/ui_shared.py` — add `PAGE_ICONS["databank"]` SVG
- Uses: `render_page_layout()`, `inject_custom_css()`, `get_yahoo_data_service()`, `st.download_button()`, `st.markdown(unsafe_allow_html=True)`
- Position pills rendered as styled HTML spans; actual filter driven by `st.selectbox`
- Fantasy Teams dropdown populated from Yahoo API `get_rosters()`
- Export button generates `HEATER_Player_Databank_{view}_{date}.xlsx`

### Task 8: Bootstrap Integration
- **Modify:** `src/data_bootstrap.py` — add "game_logs" phase with 1hr staleness calling `fetch_game_logs_from_api()`
- Fetches 2026 (current), 2025, 2024 seasons
- Historical fetched once (30-day staleness)

### Task 9: Final Verification
- Run full test suite (`python -m pytest tests/test_player_databank.py -v`)
- Run ruff format + lint
- Run full project tests (`python -m pytest -x -q`)
- Visual verification in browser

## Critical Files to Reuse
- `src/database.py` — `get_connection()`, `init_db()`, `load_player_pool()`
- `src/ui_shared.py` — `T` (theme), `inject_custom_css()`, `render_page_layout()`, `PAGE_ICONS`, `format_stat()`
- `src/yahoo_data_service.py` — `get_yahoo_data_service()`, `.get_rosters()`, `.is_connected()`
- `src/valuation.py` — `LeagueConfig` for category definitions
- `src/data_bootstrap.py` — existing phase pattern for adding game_logs phase

## Verification
1. `python -m pytest tests/test_player_databank.py -v` — all databank-specific tests pass
2. `python -m ruff check . && python -m ruff format --check .` — lint clean
3. `python -m pytest -x -q` — full suite passes (~3180+ tests)
4. `streamlit run app.py` → navigate to Player Databank → verify filters, sorting, export all work
5. Push to master, verify CI passes
