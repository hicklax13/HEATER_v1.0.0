# Full Debug Plan — HEATER v1.0.0

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all 18 open bugs, load complete player data (projections, ADP, teams, FAs) for the 2026 MLB season, and fully wire Yahoo Fantasy league integration.

**Architecture:** 5-phase fix plan ordered by dependency chain. Phase 1 fixes team fields (quick win). Phase 2 restores projections+ADP via FanGraphs HTML `__NEXT_DATA__` extraction (bypasses 403). Phase 3 expands the player pool and FA coverage. Phase 4 wires missing Yahoo features. Phase 5 polishes data quality.

**Tech Stack:** Python 3.13, Streamlit, SQLite, requests, statsapi, yfpy, BeautifulSoup4

**Key Discovery:** FanGraphs JSON API returns 403, but the HTML pages return 200 with full projection data embedded in `<script id="__NEXT_DATA__">`. Steamer alone has 4,186 hitters + 5,162 pitchers + ADP. This is the primary fix for BUG-001/002.

---

## Phase 1: Fix Team Field (BUG-003)

**Bugs resolved:** BUG-003
**Unblocks:** Phases 3, 5 (team-based features, opponent data)

**Files:**
- Modify: `src/live_stats.py` (lines 351-441)

### Task 1.1: Add team_id → abbreviation lookup

- [ ] **Step 1:** Add `_build_team_id_map(season)` helper function near line 25 of `src/live_stats.py`:
  ```python
  _team_id_cache: dict[int, dict[int, str]] = {}

  def _build_team_id_map(season: int = 2026) -> dict[int, str]:
      if season in _team_id_cache:
          return _team_id_cache[season]
      try:
          data = statsapi.get("teams", {"sportId": 1, "season": season})
          mapping = {t["id"]: t["abbreviation"] for t in data.get("teams", [])}
          _team_id_cache[season] = mapping
          return mapping
      except Exception:
          return {}
  ```

- [ ] **Step 2:** In `fetch_all_mlb_players()` (line 356), add `team_map = _build_team_id_map(season)` before the API call.

- [ ] **Step 3:** Change line 377 from:
  ```python
  "team": p.get("currentTeam", {}).get("abbreviation", ""),
  ```
  to:
  ```python
  "team": team_map.get(p.get("currentTeam", {}).get("id"), ""),
  ```

- [ ] **Step 4:** Apply same fix in `fetch_extended_roster()` — add `team_map = _build_team_id_map(season)` and replace the `.get("abbreviation", "")` pattern.

- [ ] **Step 5:** Run `python -m pytest tests/ -k "live_stats" -v` — verify existing tests pass.

- [ ] **Step 6:** Verify live: `python -c "from src.live_stats import fetch_all_mlb_players; df=fetch_all_mlb_players(); print(df[df['team']!=''].shape[0], 'players with teams')"` — should show ~779.

---

## Phase 2: Restore Projections + ADP (BUG-001, BUG-002, BUG-005, BUG-011)

**Bugs resolved:** BUG-001, BUG-002, BUG-005, BUG-011
**Unblocks:** Phase 3 (FA ranking), Phase 5 (ECR, opponent data)

**Files:**
- Modify: `src/data_pipeline.py` (lines 150-661)
- Modify: `src/data_bootstrap.py` (add player_id_map phase)
- Modify: `src/prospect_engine.py` (line 233-251)

### Task 2.1: Add __NEXT_DATA__ extraction to data_pipeline.py

The FanGraphs JSON API (`/api/projections`) returns 403, but the HTML pages return 200 with projection data embedded in `<script id="__NEXT_DATA__">`. Confirmed: Steamer returns 4,186 hitters + 5,162 pitchers.

- [ ] **Step 1:** Add a new function `_fetch_fg_html_projections(system, stats)` in `src/data_pipeline.py` after `fetch_projections()`:
  ```python
  def _fetch_fg_html_projections(system: str, stats: str) -> tuple[pd.DataFrame, list[dict]]:
      """Fallback: extract projections from FanGraphs HTML __NEXT_DATA__ tag.

      The JSON API returns 403, but HTML pages embed full data in Next.js
      server-side props. Returns same format as fetch_projections().
      """
      import json as _json
      import re as _re

      url = f"https://www.fangraphs.com/projections?type={system}&stats={stats}&pos=all&team=0&lg=all&players=0"
      try:
          resp = _SESSION.get(url, timeout=20)
          resp.raise_for_status()
      except requests.exceptions.RequestException as exc:
          raise FetchError(f"HTML fetch failed for {system}/{stats}: {exc}") from exc

      match = _re.search(
          r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', resp.text
      )
      if not match:
          raise FetchError(f"No __NEXT_DATA__ found for {system}/{stats}")

      try:
          nd = _json.loads(match.group(1))
          raw = nd["props"]["pageProps"]["dehydratedState"]["queries"][0]["state"]["data"]
      except (KeyError, IndexError, _json.JSONDecodeError) as exc:
          raise FetchError(f"Failed to parse __NEXT_DATA__ for {system}/{stats}: {exc}") from exc

      if not raw:
          raise FetchError(f"Empty data in __NEXT_DATA__ for {system}/{stats}")

      # Normalize column names to match JSON API format
      for player in raw:
          if "ShortName" in player and "PlayerName" not in player:
              player["PlayerName"] = player["ShortName"]
          if "SO" in player and "K" not in player:
              player["K"] = player["SO"]

      if stats == "bat":
          return normalize_hitter_json(raw), raw
      else:
          return normalize_pitcher_json(raw), raw
  ```

- [ ] **Step 2:** Modify `fetch_projections()` to use HTML fallback. After the retry loop fails (around line 223), before raising FetchError, try the HTML path:
  ```python
  # Fallback: try HTML __NEXT_DATA__ extraction
  try:
      logger.info("JSON API failed for %s/%s, trying HTML extraction...", system, stats)
      return _fetch_fg_html_projections(system, stats)
  except FetchError:
      pass  # Fall through to original error
  raise FetchError(f"Failed to fetch {system}/{stats}: {last_exc}") from last_exc
  ```

- [ ] **Step 3:** Run `python -m pytest tests/test_data_pipeline.py -v` — the live network tests should now PASS (HTML extraction works).

### Task 2.2: Fix ADP extraction

- [ ] **Step 4:** ADP is already extracted in `refresh_if_stale()` at line 651-657 from the `raw_data` dict. The raw JSON from `__NEXT_DATA__` includes an `ADP` field. Verify: the existing `extract_adp()` function checks for `player.get("ADP")` — confirm this key exists in the HTML-extracted data (confirmed: ZiPS has `ADP` field).

- [ ] **Step 5:** If the ADP field name differs between HTML and JSON API, add a normalization step in `_fetch_fg_html_projections` to ensure `ADP` key is present.

### Task 2.3: Populate player_id_map

- [ ] **Step 6:** Add `_bootstrap_player_id_map(progress)` in `src/data_bootstrap.py`:
  ```python
  def _bootstrap_player_id_map(progress):
      """Populate player_id_map from players table cross-references."""
      from src.database import get_connection
      conn = get_connection()
      try:
          # Copy mlb_id from players table
          conn.execute("""
              INSERT OR REPLACE INTO player_id_map (player_id, mlb_id, name)
              SELECT player_id, mlb_id, name FROM players
              WHERE mlb_id IS NOT NULL AND mlb_id != 0
          """)
          # Copy fangraphs_id
          conn.execute("""
              UPDATE player_id_map SET fg_id = (
                  SELECT p.fangraphs_id FROM players p
                  WHERE p.player_id = player_id_map.player_id
              ) WHERE EXISTS (
                  SELECT 1 FROM players p
                  WHERE p.player_id = player_id_map.player_id
                  AND p.fangraphs_id IS NOT NULL
              )
          """)
          conn.commit()
          count = conn.execute("SELECT COUNT(*) FROM player_id_map").fetchone()[0]
          return f"Player ID map: {count} entries"
      except Exception as e:
          return f"Player ID map: error ({e})"
      finally:
          conn.close()
  ```

- [ ] **Step 7:** Wire into `bootstrap_all_data()` after the deduplication phase.

### Task 2.4: Fix prospect rankings fallback

- [ ] **Step 8:** In `src/prospect_engine.py`, update `fetch_fg_board()` to use the same `__NEXT_DATA__` HTML extraction approach for the prospects board page.

- [ ] **Step 9:** Run full test suite: `python -m pytest tests/ --tb=short -q`

---

## Phase 3: Expand Player Pool + Free Agent Data (BUG-004, BUG-018, BUG-019)

**Bugs resolved:** BUG-004, BUG-018, BUG-019
**Depends on:** Phase 1 (team fields), Phase 2 (projections)

**Files:**
- Modify: `src/yahoo_api.py` (line 853 — get_free_agents pagination)
- Modify: `src/data_bootstrap.py` (add Yahoo FA phase)
- Modify: `src/database.py` (add surrogate projections function)

### Task 3.1: Paginate Yahoo free agents

- [ ] **Step 1:** Add `get_all_free_agents(max_players=500)` wrapper in `src/yahoo_api.py` that paginates `get_free_agents()` calls:
  ```python
  def get_all_free_agents(self, max_players: int = 500) -> pd.DataFrame:
      batch_size = 50
      all_dfs = []
      for start in range(0, max_players, batch_size):
          df = self.get_free_agents(count=batch_size, start=start)
          if df.empty:
              break
          all_dfs.append(df)
          if len(df) < batch_size:
              break
      return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
  ```

- [ ] **Step 2:** Add `_bootstrap_yahoo_free_agents(progress, yahoo_client)` in `src/data_bootstrap.py` that calls `yahoo_client.get_all_free_agents(500)`, upserts new players, and stores ownership data.

- [ ] **Step 3:** Wire into `bootstrap_all_data()` after the Yahoo phase, only when `yahoo_client` is provided.

### Task 3.2: Generate surrogate projections for projectionless players

- [ ] **Step 4:** Add `generate_surrogate_projections()` in `src/database.py` that creates projection rows for players who have 2025 season_stats but no blended projections, pro-rating to a full season.

- [ ] **Step 5:** Wire into bootstrap after the projections phase.

- [ ] **Step 6:** Run tests: `python -m pytest tests/ -k "yahoo" -v`

---

## Phase 4: Yahoo Integration Fixes (BUG-017, BUG-016, BUG-007, BUG-013, BUG-009)

**Bugs resolved:** BUG-017, BUG-016, BUG-007, BUG-013, BUG-009
**Independent of Phases 1-3**

**Files:**
- Modify: `src/yahoo_api.py` (sync_to_db, injury headline, game_key)
- Modify: `src/database.py` (add league_teams table)

### Task 4.1: Wire transactions into sync_to_db

- [ ] **Step 1:** In `sync_to_db()` (after rosters block around line 1230), add:
  ```python
  # --- Transactions ---
  try:
      txn_df = self.get_league_transactions()
      if not txn_df.empty:
          from src.live_stats import match_player_id
          for _, row in txn_df.iterrows():
              pid = match_player_id(row.get("player_name", ""), "")
              if pid is None:
                  continue
              conn = get_connection()
              try:
                  conn.execute(
                      "INSERT OR IGNORE INTO transactions "
                      "(player_id, type, team_from, team_to, timestamp) "
                      "VALUES (?, ?, ?, ?, ?)",
                      (pid, row.get("type",""), row.get("team_from",""),
                       row.get("team_to",""), row.get("timestamp","")),
                  )
                  conn.commit()
                  counts["transactions"] = counts.get("transactions", 0) + 1
              finally:
                  conn.close()
  except Exception:
      logger.exception("Failed to sync transactions.")
  ```

### Task 4.2: Extract team logos + manager names

- [ ] **Step 2:** Add `league_teams` table to `init_db()` schema in `src/database.py`.

- [ ] **Step 3:** In `get_all_rosters()` (`src/yahoo_api.py:717`), after extracting `team_name`, extract `team_logos[0].url` and `managers[0].nickname`.

- [ ] **Step 4:** In `sync_to_db()`, after roster sync, upsert team metadata into `league_teams` table.

### Task 4.3: Fix injury headline + news constraint

- [ ] **Step 5:** Fix headline at `src/yahoo_api.py` line ~1203: prefix body part with "Placed on IL —".

- [ ] **Step 6:** Add `published_at=datetime.now(UTC).isoformat()` to the INSERT statement for Yahoo news.

### Task 4.4: Validate game_key

- [ ] **Step 7:** Add warning log in `_resolve_game_key()` if resolved key != "469".

- [ ] **Step 8:** Run: `python -m pytest tests/test_yahoo_api.py tests/test_news_fetcher.py -v`

---

## Phase 5: Data Quality Fixes (BUG-006, BUG-010, BUG-015)

**Bugs resolved:** BUG-006, BUG-010, BUG-015
**Depends on:** Phase 2 (projections needed for ECR)

**Files:**
- Modify: `src/data_bootstrap.py` (ADP success reporting, validation)
- Modify: `src/ecr.py` (minimum-sources check)

### Task 5.1: Fix false success reporting

- [ ] **Step 1:** In `_bootstrap_adp_sources()` at line 449, change to only report success when data was actually stored.

### Task 5.2: Improve ECR consensus

- [ ] **Step 2:** In `refresh_ecr_consensus()`, add a minimum-sources check (>= 2 required for meaningful consensus).

- [ ] **Step 3:** Add ECR source that reads from the `adp` table directly (FG ADP from Phase 2).

### Task 5.3: Add post-bootstrap validation

- [ ] **Step 4:** Add validation logging at end of `bootstrap_all_data()` — log player count, projection count, ADP count, team-field coverage.

- [ ] **Step 5:** Run full test suite: `python -m pytest tests/ --tb=short -q` — all 2022+ tests pass.

---

## Final Validation Checklist

- [ ] Full test suite: `python -m pytest` — 0 failures
- [ ] Lint: `python -m ruff check .` — clean
- [ ] Format: `python -m ruff format --check .` — clean
- [ ] Bootstrap end-to-end: run `streamlit run app.py` and verify:
  - [ ] Players: > 1,000 with team fields populated
  - [ ] Projections: > 500 blended rows
  - [ ] ADP: > 200 rows with adp < 500
  - [ ] player_id_map: > 500 rows
  - [ ] Transactions: > 0 rows
  - [ ] League teams with logos: 12 rows
  - [ ] ECR consensus with >= 2 sources: > 100 rows
  - [ ] Yahoo rosters: 264 entries across 12 teams
  - [ ] Free agents: > 1,000 available with rankings

---

## Bug → Phase Mapping

| Bug | Phase | Fix |
|-----|-------|-----|
| BUG-001 | 2 | `__NEXT_DATA__` HTML extraction bypasses 403 |
| BUG-002 | 2 | ADP from HTML extraction (ADP field in Steamer data) |
| BUG-003 | 1 | `team_id → abbreviation` lookup map from teams endpoint |
| BUG-004 | 3 | Extended roster + Yahoo FA fetch |
| BUG-005 | 2 | `_bootstrap_player_id_map()` consolidates mlb_id + fg_id |
| BUG-006 | 5 | Unblocked by Phase 2 + ECR improvements |
| BUG-007 | 4 | Add `published_at` to Yahoo news inserts |
| BUG-009 | 4 | Validate game_key == 469 with warning |
| BUG-010 | 5 | Check actual row count before marking success |
| BUG-011 | 2 | HTML extraction for prospect board page |
| BUG-013 | 4 | Template headline for body-part injury notes |
| BUG-015 | 5 | Auto-resolves when Phases 1+2 complete |
| BUG-016 | 4 | Extract `team_logos` + `managers` from yfpy |
| BUG-017 | 4 | Wire `get_league_transactions()` into `sync_to_db()` |
| BUG-018 | 3 | Surrogate projections from 2025 season stats |
| BUG-019 | 3 | Paginated `get_all_free_agents()` + bootstrap phase |
