# Bug Audit Wave 1: Data Correction Migrations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 4 HIGH-severity data-correctness bugs identified in the 2026-05-11 bug audit (BUG-001, BUG-002, BUG-003, BUG-008) via one migration script + two SQL query corrections + one phase removal, then add structural guards so these regressions can't return silently.

**Architecture:** TDD per fix. Each bug gets a focused failing test → minimal fix → passing test → commit. The data migration (Task 4) uses an idempotent SQL script with a backup snapshot before write. A final task adds permanent guard tests against re-introducing the bugs.

**Tech Stack:** Python 3.11+, SQLite (WAL via `get_connection()`), pytest, MLB Stats API (`statsapi`), pybaseball (optional).

**Source spec:** [docs/superpowers/specs/2026-05-11-bug-audit-findings.md](../specs/2026-05-11-bug-audit-findings.md) (BUG-001, BUG-002, BUG-003, BUG-008)

---

## Cold-Start Context

You are a fresh session with no memory of prior conversations. Here's everything you need:

### Worktree
- Path: `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave1-plan`
- Branch: `claude/audit-wave1-plan` (tracks `origin/master`)
- Working tree should be clean at start of execution.

### Live DB (read+write — back it up before Task 4)
- Path: `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db`
- Schema highlights:
  - `players(player_id PK, name, team, positions, mlb_id, is_injured, injury_note, is_hitter, ...)`
  - `league_rosters(id PK, team_name, team_index, player_id, roster_slot, is_user_team, status, selected_position, editorial_team_abbr, is_undroppable)` — **NO `name` column** (this is the source of BUG-003)
  - `park_factors(team_code PK, factor_hitting, factor_pitching)` — Tier 1 + emergency dict (BUG-008 target)
  - `refresh_log(source PK, status, tier, message, last_refresh, rows_written, expected_min)`

### Project context (just enough to brief executor agents)
- HEATER is a fantasy-baseball draft + in-season manager for Yahoo's FourzynBurn 12-team H2H Categories league (game_key 469).
- Today is 2026-05-11; MLB 2026 season is active (~5-6 weeks in).
- Canonical sources of truth: see CLAUDE.md "Unified Services" section.
- `get_connection()` (from `src/database.py`) is the only sanctioned SQLite path. Wraps WAL + 30s busy_timeout.

### Bug summaries

| Bug | Location | Effect | Fix in this plan |
|-----|----------|--------|------------------|
| BUG-001 | `players` table: 33 rows with `team='MLB'`, fake `mlb_id` ∈ [600000, 601999] | 4 rostered IL players (Burnes pid 110, Jared Jones pid 130, Schwellenbach pid 164, Joyce pid 187) silently pull stats from DSL/VSL prospects | Task 4 migration |
| BUG-002 | `players` table: 3 rostered players have `mlb_id IS NULL` (Greene pid 4643, Pepiot pid 4641, Crews pid 922) | Invisible to all live-stats pipelines | Task 4 migration (same script as BUG-001) |
| BUG-003a | `src/data_bootstrap.py:2525-2530` | `_bootstrap_injury_writeback` SQL joins `league_rosters` on non-existent `name` column → `status=error` for 8+ days → 0 IL flagged despite 56 IL roster rows | Task 1 |
| BUG-003b | `src/data_bootstrap.py:2585-2590` | `_bootstrap_draft_results` updates by `WHERE name=?` on non-existent column → 0 R1-3 picks flagged undroppable | Task 2 |
| BUG-008 | `src/data_bootstrap.py:2264-2312` | `_bootstrap_dynamic_park_factors` uses team OPS+/wRC+ as park factor (these are park-ADJUSTED measures of offense, not park environment) → silently overwrites correct Tier 1 / emergency park factors every 7 days | Task 3 |

---

## File Structure (decisions locked here)

**Files to create:**
- `scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py` — one-shot migration script (Task 4)
- `tests/migrations/test_fix_shadow_rows_and_mlb_ids.py` — unit test for migration logic against an in-memory test DB (Task 4)
- `tests/test_bootstrap_injury_writeback.py` — TDD test for BUG-003a (Task 1)
- `tests/test_bootstrap_draft_results.py` — TDD test for BUG-003b (Task 2)
- `tests/test_dynamic_park_factors_removed.py` — TDD test for BUG-008 (Task 3)
- `tests/test_no_shadow_player_rows.py` — permanent structural guard (Task 5)
- `tests/test_no_lr_name_column_refs.py` — permanent structural guard against re-introducing `lr.name` (Task 5)

**Files to modify:**
- `src/data_bootstrap.py` line 2525-2530 (Task 1)
- `src/data_bootstrap.py` line 2585-2590 (Task 2)
- `src/data_bootstrap.py` line 2264-2312 (Task 3 — remove entire `_bootstrap_dynamic_park_factors` function and its caller in `bootstrap_all_data`)
- `CLAUDE.md` "Data Audit History" section — add SF-29..SF-32 entry (Task 5 final step)

**Dirs to create:**
- `scripts/migrations/` (may not exist; if not, `mkdir -p` it)
- `tests/migrations/` (may not exist; if not, `mkdir -p` it)

---

## Phase 0: Pre-Flight (5 min)

- [ ] **Step 0.1: Verify worktree state**

  Run:
  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave1-plan"
  git status
  git log -1 --oneline
  ```
  Expected: Clean working tree. HEAD at `1b0dd43 Merge pull request #12` or a newer commit on `claude/audit-wave1-plan`.

- [ ] **Step 0.2: Verify test baseline**

  Run:
  ```bash
  python -m pytest --collect-only -q --ignore=tests/test_cheat_sheet.py 2>&1 | tail -3
  ```
  Expected: ~3740 tests collect cleanly. If collection fails, stop.

- [ ] **Step 0.3: Snapshot the live DB BEFORE Task 4 runs**

  Run:
  ```bash
  cp "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db" \
     "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db.backup-2026-05-11-pre-wave1"
  ls -la "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/" | grep draft_tool
  ```
  Expected: Two files — the live DB and the backup. Backup size matches live size.

- [ ] **Step 0.4: Count shadow rows + NULL mlb_ids in live DB (baseline metrics)**

  Run:
  ```bash
  python -c "
  import sqlite3
  conn = sqlite3.connect('C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db')
  shadow = conn.execute(\"SELECT COUNT(*) FROM players WHERE team='MLB' AND mlb_id BETWEEN 600000 AND 601999\").fetchone()[0]
  null_mlb = conn.execute(\"SELECT COUNT(*) FROM players WHERE mlb_id IS NULL AND player_id IN (SELECT player_id FROM league_rosters)\").fetchone()[0]
  rostered_shadow = conn.execute(\"SELECT COUNT(*) FROM players WHERE team='MLB' AND mlb_id BETWEEN 600000 AND 601999 AND player_id IN (SELECT player_id FROM league_rosters)\").fetchone()[0]
  print(f'Shadow rows total: {shadow}')
  print(f'Shadow rows rostered: {rostered_shadow}')
  print(f'Rostered players with NULL mlb_id: {null_mlb}')
  conn.close()
  "
  ```
  Expected (per Stream B agent's findings):
  ```
  Shadow rows total: 33
  Shadow rows rostered: 4
  Rostered players with NULL mlb_id: 3
  ```
  Record these numbers in your notes — Task 4 will assert exact migration deltas against them.

---

## Task 1: Fix `_bootstrap_injury_writeback` SQL (BUG-003a)

**Files:**
- Modify: `src/data_bootstrap.py:2525-2530`
- Create: `tests/test_bootstrap_injury_writeback.py`

- [ ] **Step 1.1: Write the failing test**

  Create `tests/test_bootstrap_injury_writeback.py`:
  ```python
  """Test BUG-003a fix: _bootstrap_injury_writeback joins on player_id not name."""
  import sqlite3
  from pathlib import Path
  import pytest
  
  
  @pytest.fixture
  def test_db(tmp_path, monkeypatch):
      db_path = tmp_path / "test.db"
      conn = sqlite3.connect(db_path)
      conn.executescript("""
          CREATE TABLE players (
              player_id INTEGER PRIMARY KEY,
              name TEXT,
              is_injured INTEGER DEFAULT 0,
              injury_note TEXT
          );
          CREATE TABLE league_rosters (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              team_name TEXT,
              team_index INTEGER,
              player_id INTEGER,
              roster_slot TEXT,
              is_user_team INTEGER DEFAULT 0,
              status TEXT,
              selected_position TEXT,
              editorial_team_abbr TEXT,
              is_undroppable INTEGER DEFAULT 0
          );
          CREATE TABLE refresh_log (
              source TEXT PRIMARY KEY,
              status TEXT,
              tier TEXT,
              message TEXT,
              last_refresh TEXT,
              rows_written INTEGER,
              expected_min INTEGER
          );
          INSERT INTO players (player_id, name) VALUES
              (1, 'Player A'),
              (2, 'Player B'),
              (3, 'Player C'),
              (4, 'Player D');
          INSERT INTO league_rosters (team_name, team_index, player_id, status) VALUES
              ('My Team', 0, 1, 'IL10'),
              ('My Team', 0, 2, 'IL15'),
              ('Other Team', 1, 3, 'DTD'),
              ('Other Team', 1, 4, 'active');
      """)
      conn.commit()
      conn.close()
  
      import src.database as db_mod
      monkeypatch.setattr(db_mod, "DB_PATH", db_path)
      return db_path
  
  
  def test_injury_writeback_flags_il_and_dtd_players(test_db, monkeypatch):
      """Players with IL10/IL15/IL60/DTD status should be flagged is_injured=1.
      'active' status players should remain is_injured=0."""
      # Stub espn injuries fetch so we test only the Yahoo writeback step
      monkeypatch.setattr("src.espn_injuries.fetch_espn_injuries", lambda: [])
      monkeypatch.setattr("src.espn_injuries.update_player_injury_flags", lambda x: 0)
  
      from src.data_bootstrap import _bootstrap_injury_writeback, BootstrapProgress
      result = _bootstrap_injury_writeback(BootstrapProgress())
      assert "Error" not in result, f"Function returned error: {result}"
  
      conn = sqlite3.connect(test_db)
      try:
          il_flagged = conn.execute(
              "SELECT player_id FROM players WHERE is_injured = 1 ORDER BY player_id"
          ).fetchall()
          active = conn.execute(
              "SELECT player_id FROM players WHERE is_injured = 0 ORDER BY player_id"
          ).fetchall()
      finally:
          conn.close()
  
      assert il_flagged == [(1,), (2,), (3,)], f"Expected players 1,2,3 flagged; got {il_flagged}"
      assert active == [(4,)], f"Expected only player 4 active; got {active}"
  ```

- [ ] **Step 1.2: Run test to verify it fails with the documented bug**

  Run: `python -m pytest tests/test_bootstrap_injury_writeback.py -v`
  Expected: FAIL with `OperationalError: no such column: lr.name` (or similar). The error gets swallowed by the function's outer `except Exception`, so the function returns `"Error: ..."` instead of raising — but our test asserts `"Error" not in result`, so it fails that way.

- [ ] **Step 1.3: Fix the SQL — replace `players.name = lr.name` with `players.player_id = lr.player_id`**

  Edit `src/data_bootstrap.py` lines 2525-2530:
  ```python
  # OLD (lines 2525-2530):
              conn.execute(
                  """UPDATE players SET is_injured = 1, injury_note = lr.status
                     FROM league_rosters lr
                     WHERE players.name = lr.name
                       AND lr.status IN ('IL10', 'IL15', 'IL60', 'DTD')"""
              )
  
  # NEW:
              conn.execute(
                  """UPDATE players SET is_injured = 1, injury_note = lr.status
                     FROM league_rosters lr
                     WHERE players.player_id = lr.player_id
                       AND lr.status IN ('IL10', 'IL15', 'IL60', 'DTD')"""
              )
  ```

- [ ] **Step 1.4: Run test to verify it passes**

  Run: `python -m pytest tests/test_bootstrap_injury_writeback.py -v`
  Expected: PASS (1 test).

- [ ] **Step 1.5: Run lint + format**

  Run: `python -m ruff format src/data_bootstrap.py tests/test_bootstrap_injury_writeback.py && python -m ruff check src/data_bootstrap.py tests/test_bootstrap_injury_writeback.py`
  Expected: No diagnostics.

- [ ] **Step 1.6: Commit**

  ```bash
  git add src/data_bootstrap.py tests/test_bootstrap_injury_writeback.py
  git commit -m "$(cat <<'EOF'
fix(bootstrap): injury_writeback joins on player_id not name (BUG-003a)

league_rosters has no `name` column; the prior query was raising
OperationalError silently swallowed by the function's outer try/except,
so the refresh_log showed status=error for 8+ days and 0 players were
ever flagged is_injured despite 56 IL roster rows.

Replaces `WHERE players.name = lr.name` with `WHERE players.player_id =
lr.player_id` (the correct schema-backed join).

Adds tests/test_bootstrap_injury_writeback.py to lock in the fix.

Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-003)
EOF
)"
  ```

---

## Task 2: Fix `_bootstrap_draft_results` SQL (BUG-003b)

**Files:**
- Modify: `src/data_bootstrap.py:2582-2593` (the UPDATE loop)
- Create: `tests/test_bootstrap_draft_results.py`

- [ ] **Step 2.1: Write the failing test**

  Create `tests/test_bootstrap_draft_results.py`:
  ```python
  """Test BUG-003b fix: _bootstrap_draft_results flags undroppable by player_id."""
  import sqlite3
  from pathlib import Path
  from unittest.mock import MagicMock
  import pandas as pd
  import pytest
  
  
  @pytest.fixture
  def test_db_and_client(tmp_path, monkeypatch):
      db_path = tmp_path / "test.db"
      conn = sqlite3.connect(db_path)
      conn.executescript("""
          CREATE TABLE players (
              player_id INTEGER PRIMARY KEY,
              name TEXT
          );
          CREATE TABLE league_rosters (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              team_name TEXT,
              team_index INTEGER,
              player_id INTEGER,
              roster_slot TEXT,
              is_user_team INTEGER DEFAULT 0,
              status TEXT,
              selected_position TEXT,
              editorial_team_abbr TEXT,
              is_undroppable INTEGER DEFAULT 0
          );
          CREATE TABLE league_draft_results (
              pick INTEGER PRIMARY KEY,
              round INTEGER,
              team_name TEXT,
              player_id INTEGER,
              player_name TEXT
          );
          CREATE TABLE refresh_log (
              source TEXT PRIMARY KEY,
              status TEXT,
              tier TEXT,
              message TEXT,
              last_refresh TEXT,
              rows_written INTEGER,
              expected_min INTEGER
          );
          INSERT INTO players (player_id, name) VALUES
              (1, 'Player A'), (2, 'Player B'), (3, 'Player C'), (4, 'Player D');
          INSERT INTO league_rosters (team_name, team_index, player_id) VALUES
              ('My Team', 0, 1),
              ('My Team', 0, 2),
              ('Other Team', 1, 3),
              ('Other Team', 1, 4);
      """)
      conn.commit()
      conn.close()
  
      import src.database as db_mod
      monkeypatch.setattr(db_mod, "DB_PATH", db_path)
  
      # Mock Yahoo client: R1 = Player A (pid 1), R2 = Player C (pid 3), R4 = Player B (pid 2)
      mock_df = pd.DataFrame([
          {"pick": 1, "round": 1, "team_name": "My Team", "player_id": 1, "player_name": "Player A"},
          {"pick": 2, "round": 2, "team_name": "Other Team", "player_id": 3, "player_name": "Player C"},
          {"pick": 25, "round": 4, "team_name": "My Team", "player_id": 2, "player_name": "Player B"},
      ])
      mock_client = MagicMock()
      mock_client.get_draft_results.return_value = mock_df
      return db_path, mock_client
  
  
  def test_draft_results_flags_undroppable_for_rounds_1_to_3_by_player_id(test_db_and_client):
      """Players drafted in rounds 1-3 should be flagged is_undroppable=1 in league_rosters."""
      test_db, mock_client = test_db_and_client
      from src.data_bootstrap import _bootstrap_draft_results, BootstrapProgress
      result = _bootstrap_draft_results(BootstrapProgress(), yahoo_client=mock_client)
      assert "Error" not in result, f"Function returned error: {result}"
  
      conn = sqlite3.connect(test_db)
      try:
          undroppable = sorted(
              row[0] for row in conn.execute(
                  "SELECT player_id FROM league_rosters WHERE is_undroppable = 1"
              ).fetchall()
          )
      finally:
          conn.close()
  
      # Players 1 (R1) and 3 (R2) should be undroppable; player 2 (R4) should NOT be
      assert undroppable == [1, 3], f"Expected [1, 3]; got {undroppable}"
  ```

- [ ] **Step 2.2: Run test to verify it fails**

  Run: `python -m pytest tests/test_bootstrap_draft_results.py -v`
  Expected: FAIL with `OperationalError: no such column: name` (or test asserts `undroppable == [1, 3]` but sees `[]`).

- [ ] **Step 2.3: Fix the loop — resolve name→player_id once, then batch-UPDATE by player_id**

  Edit `src/data_bootstrap.py` lines 2582-2593:
  ```python
  # OLD (lines 2582-2593):
          # Flag rounds 1-3 as undroppable in league_rosters
          undroppable_names = df[df["round"] <= 3]["player_name"].tolist()
          conn = get_connection()
          try:
              conn.execute("UPDATE league_rosters SET is_undroppable = 0")
              for name in undroppable_names:
                  conn.execute(
                      "UPDATE league_rosters SET is_undroppable = 1 WHERE name = ?",
                      (name,),
                  )
              conn.commit()
          finally:
              conn.close()
  
  # NEW:
          # Flag rounds 1-3 as undroppable in league_rosters (by player_id, not name)
          undroppable_pick_rows = df[df["round"] <= 3]
          # Prefer existing player_id from draft DF; fall back to name resolution
          undroppable_player_ids: list[int] = []
          conn = get_connection()
          try:
              conn.execute("UPDATE league_rosters SET is_undroppable = 0")
              for _, pick_row in undroppable_pick_rows.iterrows():
                  pid_raw = pick_row.get("player_id")
                  if pid_raw is not None and not (isinstance(pid_raw, float) and pid_raw != pid_raw):  # NaN check
                      try:
                          pid = int(pid_raw)
                      except (TypeError, ValueError):
                          pid = None
                  else:
                      pid = None
                  if pid is None:
                      # Resolve by name as last resort
                      name = pick_row.get("player_name", "")
                      row = conn.execute(
                          "SELECT player_id FROM players WHERE name = ? LIMIT 1", (name,)
                      ).fetchone()
                      pid = row[0] if row else None
                  if pid is None:
                      logger.warning("Could not resolve player_id for draft pick: %s", pick_row.to_dict())
                      continue
                  undroppable_player_ids.append(pid)
                  conn.execute(
                      "UPDATE league_rosters SET is_undroppable = 1 WHERE player_id = ?",
                      (pid,),
                  )
              conn.commit()
          finally:
              conn.close()
  
          undroppable_names = undroppable_pick_rows["player_name"].tolist()  # kept for log message
  ```

  Note: also update the `update_refresh_log_auto` call below to use `len(undroppable_player_ids)` instead of `len(undroppable_names)` if you want the count to reflect successful flags (recommended).

- [ ] **Step 2.4: Run test to verify it passes**

  Run: `python -m pytest tests/test_bootstrap_draft_results.py -v`
  Expected: PASS (1 test).

- [ ] **Step 2.5: Lint + format**

  Run: `python -m ruff format src/data_bootstrap.py tests/test_bootstrap_draft_results.py && python -m ruff check src/data_bootstrap.py tests/test_bootstrap_draft_results.py`
  Expected: No diagnostics.

- [ ] **Step 2.6: Commit**

  ```bash
  git add src/data_bootstrap.py tests/test_bootstrap_draft_results.py
  git commit -m "$(cat <<'EOF'
fix(bootstrap): draft_results flags undroppable by player_id not name (BUG-003b)

league_rosters has no `name` column; the prior `WHERE name = ?` UPDATE
raised OperationalError silently swallowed → 0 R1-3 picks ever flagged
undroppable, breaking league rules enforcement (CLAUDE.md "Can't-drop:
players drafted in rounds 1-3 per team — 36 league-wide").

Resolves each pick's player_id (preferring the draft DF's column, falling
back to a name lookup in players) then UPDATEs by player_id. Logs a
warning for any pick that can't be resolved.

Adds tests/test_bootstrap_draft_results.py to lock in the fix.

Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-003)
EOF
)"
  ```

---

## Task 3: Remove `_bootstrap_dynamic_park_factors` (BUG-008)

**Rationale:** The function uses team OPS+/wRC+ (a park-ADJUSTED measure of team offense) as a park factor proxy, which is fundamentally wrong. The `_bootstrap_park_factors` phase (Tier 1 pybaseball + Tier 3 emergency dict) already populates correct values. The dynamic-refresh phase runs every 7 days and silently corrupts those values. The agent recommendation in the findings doc is "Either remove this phase (Tier 1 from `_bootstrap_park_factors` is sufficient) or compute proper park factors from home-vs-away OPS splits." Removal is the simpler safe choice; if a real dynamic refresh is wanted later, it can be re-implemented correctly.

**Files:**
- Modify: `src/data_bootstrap.py:2264-2312` (delete `_bootstrap_dynamic_park_factors` function)
- Modify: `src/data_bootstrap.py` orchestrator section (remove the dispatch for `dynamic_park_factors` source)
- Modify: `src/data_bootstrap.py` source map (if present — search for `"dynamic_park_factors"` string)
- Create: `tests/test_dynamic_park_factors_removed.py`

- [ ] **Step 3.1: Write the failing test**

  Create `tests/test_dynamic_park_factors_removed.py`:
  ```python
  """Test BUG-008 fix: _bootstrap_dynamic_park_factors is removed; not callable from bootstrap_all_data."""
  import importlib
  
  
  def test_dynamic_park_factors_function_removed():
      """The function _bootstrap_dynamic_park_factors should no longer exist."""
      import src.data_bootstrap as bootstrap_mod
      importlib.reload(bootstrap_mod)
      assert not hasattr(bootstrap_mod, "_bootstrap_dynamic_park_factors"), (
          "BUG-008: _bootstrap_dynamic_park_factors should be removed — it used "
          "team OPS+/wRC+ (a park-ADJUSTED metric) as a park factor proxy, "
          "silently corrupting correct Tier 1 / emergency factors every 7 days."
      )
  
  
  def test_dynamic_park_factors_not_in_orchestrator_source_list():
      """bootstrap_all_data should not dispatch a 'dynamic_park_factors' or 'park_factors_dynamic' source."""
      import re
      from pathlib import Path
      src_text = Path("src/data_bootstrap.py").read_text(encoding="utf-8")
      # The string should not appear as a dispatched source name or refresh_log key
      forbidden = ["dynamic_park_factors", "park_factors_dynamic"]
      for name in forbidden:
          # Allow it to appear in comments/docstrings as historical context
          # but not as a string literal that would be wired into orchestration
          # We check that no `"<name>"` appears anywhere outside a comment
          for lineno, line in enumerate(src_text.splitlines(), start=1):
              stripped = line.strip()
              if stripped.startswith("#"):
                  continue
              if f'"{name}"' in line:
                  raise AssertionError(
                      f"BUG-008 regression: src/data_bootstrap.py:{lineno} references "
                      f'"{name}" — should have been removed entirely. Line: {line!r}'
                  )
  ```

- [ ] **Step 3.2: Run test to verify it fails**

  Run: `python -m pytest tests/test_dynamic_park_factors_removed.py -v`
  Expected: FAIL — function still exists; string `"park_factors_dynamic"` still appears in `update_refresh_log` call.

- [ ] **Step 3.3: Delete the function**

  Open `src/data_bootstrap.py`. Delete the entire function `_bootstrap_dynamic_park_factors` (currently lines 2264-2312, ending just before `def _bootstrap_bat_speed`). Delete both the blank lines before/after and the function body.

- [ ] **Step 3.4: Remove the orchestrator dispatch**

  Search for any call site of `_bootstrap_dynamic_park_factors` in `src/data_bootstrap.py`:
  ```bash
  grep -n "_bootstrap_dynamic_park_factors\|dynamic_park_factors\|park_factors_dynamic" src/data_bootstrap.py
  ```
  Remove every match. Common locations:
  - `bootstrap_all_data` orchestrator loop (search for the source-name → function-name dispatch dict or `if`-chain)
  - `StalenessConfig` defaults (search for `dynamic_park_factors_hours` or similar)
  - Any source-list constant

  If `staleness.py` or a shared config has `dynamic_park_factors_hours: 168`, remove that key too.

- [ ] **Step 3.5: Search the wider codebase for orphan references**

  Run:
  ```bash
  grep -rn "dynamic_park_factors\|park_factors_dynamic\|_bootstrap_dynamic_park_factors" src/ tests/ pages/ scripts/ 2>/dev/null
  ```
  Expected: zero matches (excluding `tests/test_dynamic_park_factors_removed.py` which we just created and `docs/` files).

  If matches appear in `src/optimizer/data_freshness.py` or similar, remove those too. The phase should be entirely gone from production code paths.

- [ ] **Step 3.6: Run test to verify it passes**

  Run: `python -m pytest tests/test_dynamic_park_factors_removed.py -v`
  Expected: PASS (2 tests).

- [ ] **Step 3.7: Verify full test suite still passes**

  Run: `python -m pytest --ignore=tests/test_cheat_sheet.py -x -q`
  Expected: All tests pass (or only pre-existing skips for PyMC/PuLP/etc.).

  If a test that imported `_bootstrap_dynamic_park_factors` now fails with `ImportError`, that's a regression we just exposed — also delete that import in the failing test file. Re-run.

- [ ] **Step 3.8: Lint + format**

  Run: `python -m ruff format src/data_bootstrap.py tests/test_dynamic_park_factors_removed.py && python -m ruff check src/data_bootstrap.py tests/test_dynamic_park_factors_removed.py`
  Expected: No diagnostics.

- [ ] **Step 3.9: Commit**

  ```bash
  git add src/data_bootstrap.py tests/test_dynamic_park_factors_removed.py
  git commit -m "$(cat <<'EOF'
fix(bootstrap): remove _bootstrap_dynamic_park_factors (BUG-008)

The function used team OPS+/wRC+ as a park factor proxy. Both are
park-ADJUSTED measures of team offense, NOT park environment. Running
every 7 days, it silently overwrote correct Tier 1 / emergency park
factors with team-strength signal (LAD wRC+=115 → factor_hitting=1.15
instead of the real ~0.99 for Dodger Stadium).

The _bootstrap_park_factors phase (Tier 1 pybaseball + emergency dict)
already populates park_factors correctly. The dynamic-refresh phase
was strictly harmful.

If real dynamic park factors are wanted later, they must be computed
from home/away OPS splits (or pybaseball's team_results) — see the
finding's "Suggested fix" for details.

Adds tests/test_dynamic_park_factors_removed.py as a permanent guard.

Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-008)
EOF
)"
  ```

---

## Task 4: Migration — clean shadow rows + backfill mlb_ids (BUG-001 + BUG-002)

**Strategy:** A one-shot migration script with three idempotent phases:

1. **Backfill known NULL mlb_ids** via hardcoded lookup table (3 known rostered players).
2. **Repair `league_rosters` references** that point to shadow rows whose real twin exists in `players` (re-point league_rosters.player_id from shadow_pid → real_pid).
3. **Delete shadow rows** from `players`.

The script is idempotent (running it twice has no effect after the first run). It backs up via Step 0.3 already.

**Files:**
- Create: `scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py`
- Create: `tests/migrations/test_fix_shadow_rows_and_mlb_ids.py`
- Create: `scripts/migrations/__init__.py` (empty, makes it a package — needed for the test to import)
- Create: `tests/migrations/__init__.py` (empty)

- [ ] **Step 4.1: Create the migrations package skeleton**

  Run:
  ```bash
  mkdir -p scripts/migrations tests/migrations
  touch scripts/migrations/__init__.py tests/migrations/__init__.py
  ```

- [ ] **Step 4.2: Write the migration script**

  Create `scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py`:
  ```python
  #!/usr/bin/env python
  """Migration 2026-05-11: fix shadow rows + backfill missing mlb_ids.
  
  Addresses BUG-001 and BUG-002 from the 2026-05-11 bug audit.
  
  BUG-001 root cause: 33 player rows exist with team='MLB' and fabricated
  mlb_id values in [600000, 601999]. These IDs resolve to DSL/VSL minor-league
  prospects when queried against MLB Stats API. 4 of these rows are currently
  on league rosters (Burnes pid 110, Jared Jones pid 130, Schwellenbach pid 164,
  Joyce pid 187), so any live-stats refresh pulls Brian Escolastico's stats
  into Teoscar's row, etc.
  
  BUG-002: 3 rostered players have NULL mlb_id (Greene pid 4643, Pepiot 4641,
  Crews 922), invisible to mlb_id-keyed bootstrap phases.
  
  Migration is idempotent. Always pass `--dry-run` first to preview changes.
  """
  from __future__ import annotations
  
  import argparse
  import logging
  import sys
  from pathlib import Path
  
  # Make src/ importable
  ROOT = Path(__file__).resolve().parents[2]
  sys.path.insert(0, str(ROOT))
  
  from src.database import get_connection  # noqa: E402
  
  logger = logging.getLogger("migrate.shadow_rows")
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
  
  
  # Known rostered players with NULL mlb_id (BUG-002).
  # mlb_ids verified via statsapi.lookup_player at audit time.
  KNOWN_NULL_MLB_BACKFILL: dict[int, int] = {
      4643: 668881,   # Hunter Greene (CIN, RHP)
      4641: 686752,   # Ryan Pepiot (TBR, RHP)
      922:  696285,   # Dylan Crews (WSN, OF) — verify in dry-run before applying
  }
  
  # Shadow rows are identified by team='MLB' AND mlb_id BETWEEN 600000 AND 601999.
  # All confirmed shadow rows from the Stream B audit (pid → name).
  SHADOW_PLAYER_ID_RANGE = (600000, 601999)
  
  
  def find_shadow_rows(conn) -> list[tuple[int, str, int]]:
      """Returns list of (player_id, name, mlb_id) for shadow rows."""
      cur = conn.execute(
          """SELECT player_id, name, mlb_id FROM players
             WHERE team = 'MLB' AND mlb_id BETWEEN ? AND ?
             ORDER BY player_id""",
          SHADOW_PLAYER_ID_RANGE,
      )
      return list(cur.fetchall())
  
  
  def find_real_twin(conn, shadow_name: str, shadow_pid: int) -> int | None:
      """Find a non-shadow player with the same (or close) name. Returns real player_id or None."""
      # Try exact-match first, excluding the shadow row itself
      cur = conn.execute(
          """SELECT player_id FROM players
             WHERE name = ? AND player_id != ?
               AND NOT (team = 'MLB' AND mlb_id BETWEEN ? AND ?)
             ORDER BY player_id LIMIT 1""",
          (shadow_name, shadow_pid, *SHADOW_PLAYER_ID_RANGE),
      )
      row = cur.fetchone()
      return row[0] if row else None
  
  
  def find_rostered_null_mlb(conn) -> list[tuple[int, str]]:
      """Returns (player_id, name) for rostered players with NULL mlb_id."""
      cur = conn.execute(
          """SELECT DISTINCT p.player_id, p.name FROM players p
             JOIN league_rosters lr ON p.player_id = lr.player_id
             WHERE p.mlb_id IS NULL
             ORDER BY p.player_id"""
      )
      return list(cur.fetchall())
  
  
  def run_migration(dry_run: bool = True) -> dict[str, int]:
      """Run the migration. Returns dict of action counts."""
      counts = {
          "shadow_rows_found": 0,
          "league_rosters_repointed": 0,
          "shadow_rows_deleted": 0,
          "null_mlb_backfilled": 0,
          "null_mlb_still_missing": 0,
      }
      conn = get_connection()
      try:
          # === Phase 1: Backfill known NULL mlb_ids (BUG-002) ===
          logger.info("Phase 1: backfill known NULL mlb_ids")
          for pid, real_mlb_id in KNOWN_NULL_MLB_BACKFILL.items():
              cur = conn.execute(
                  "SELECT name, mlb_id FROM players WHERE player_id = ?", (pid,)
              )
              row = cur.fetchone()
              if row is None:
                  logger.warning("  pid=%d not found in players (skip)", pid)
                  continue
              name, current_mlb = row
              if current_mlb == real_mlb_id:
                  logger.info("  pid=%d %s already has mlb_id=%d (idempotent skip)", pid, name, real_mlb_id)
                  continue
              if current_mlb is not None:
                  logger.warning(
                      "  pid=%d %s has unexpected mlb_id=%s (expected NULL); manual review needed, skipping",
                      pid, name, current_mlb,
                  )
                  continue
              logger.info("  pid=%d %s NULL → mlb_id=%d", pid, name, real_mlb_id)
              if not dry_run:
                  conn.execute(
                      "UPDATE players SET mlb_id = ? WHERE player_id = ? AND mlb_id IS NULL",
                      (real_mlb_id, pid),
                  )
              counts["null_mlb_backfilled"] += 1
  
          # Report any remaining NULL-mlb_id rostered players (not in known list)
          remaining_nulls = find_rostered_null_mlb(conn)
          remaining_unknown = [r for r in remaining_nulls if r[0] not in KNOWN_NULL_MLB_BACKFILL]
          if remaining_unknown:
              logger.warning("Rostered players with NULL mlb_id not in backfill table (manual fix needed):")
              for pid, name in remaining_unknown:
                  logger.warning("  pid=%d name=%s", pid, name)
              counts["null_mlb_still_missing"] = len(remaining_unknown)
  
          # === Phase 2: Identify shadow rows + repoint league_rosters refs (BUG-001) ===
          logger.info("Phase 2: identify shadow rows and repoint league_rosters refs")
          shadow_rows = find_shadow_rows(conn)
          counts["shadow_rows_found"] = len(shadow_rows)
          logger.info("  found %d shadow rows", len(shadow_rows))
  
          for shadow_pid, shadow_name, shadow_mlb in shadow_rows:
              twin_pid = find_real_twin(conn, shadow_name, shadow_pid)
              # Check whether this shadow row is referenced in league_rosters
              lr_refs = conn.execute(
                  "SELECT id, team_name FROM league_rosters WHERE player_id = ?",
                  (shadow_pid,),
              ).fetchall()
              if not lr_refs:
                  logger.info("  shadow pid=%d %s (fake_mlb=%d) — no league_rosters refs, deletable",
                              shadow_pid, shadow_name, shadow_mlb)
                  continue
              if twin_pid is None:
                  logger.warning(
                      "  shadow pid=%d %s (fake_mlb=%d) HAS %d league_rosters refs but NO real twin in "
                      "players table. Cannot safely delete. Manual fix: add real player + repoint.",
                      shadow_pid, shadow_name, shadow_mlb, len(lr_refs),
                  )
                  continue
              # Repoint each league_rosters row to the real twin
              logger.info(
                  "  shadow pid=%d %s (fake_mlb=%d) → repoint %d league_rosters rows to real pid=%d",
                  shadow_pid, shadow_name, shadow_mlb, len(lr_refs), twin_pid,
              )
              if not dry_run:
                  conn.execute(
                      "UPDATE league_rosters SET player_id = ? WHERE player_id = ?",
                      (twin_pid, shadow_pid),
                  )
              counts["league_rosters_repointed"] += len(lr_refs)
  
          # === Phase 3: Delete shadow rows ===
          logger.info("Phase 3: delete shadow rows with no remaining league_rosters refs")
          # Re-fetch refs after Phase 2 repointing
          deletable = []
          for shadow_pid, shadow_name, _ in shadow_rows:
              still_refed = conn.execute(
                  "SELECT 1 FROM league_rosters WHERE player_id = ? LIMIT 1",
                  (shadow_pid,),
              ).fetchone()
              if not still_refed:
                  deletable.append((shadow_pid, shadow_name))
          logger.info("  %d shadow rows are now safely deletable", len(deletable))
          for shadow_pid, shadow_name in deletable:
              logger.info("  DELETE shadow pid=%d %s", shadow_pid, shadow_name)
              if not dry_run:
                  conn.execute("DELETE FROM players WHERE player_id = ?", (shadow_pid,))
              counts["shadow_rows_deleted"] += 1
  
          if not dry_run:
              conn.commit()
              logger.info("Migration committed.")
          else:
              logger.info("DRY RUN — no changes committed.")
      finally:
          conn.close()
      return counts
  
  
  def main() -> int:
      p = argparse.ArgumentParser(description="Migrate shadow rows + backfill mlb_ids")
      p.add_argument("--dry-run", action="store_true", default=False,
                     help="Preview changes without committing (recommended first run)")
      p.add_argument("--commit", action="store_true", default=False,
                     help="Actually apply the migration. Required to write.")
      args = p.parse_args()
  
      if not args.dry_run and not args.commit:
          logger.error("Must pass one of --dry-run or --commit")
          return 2
      if args.dry_run and args.commit:
          logger.error("Cannot pass both --dry-run and --commit")
          return 2
  
      counts = run_migration(dry_run=args.dry_run)
      logger.info("Migration summary: %s", counts)
      return 0
  
  
  if __name__ == "__main__":
      sys.exit(main())
  ```

- [ ] **Step 4.3: Write the migration unit test**

  Create `tests/migrations/test_fix_shadow_rows_and_mlb_ids.py`:
  ```python
  """Test migration 2026-05-11-fix-shadow-rows-and-mlb-ids."""
  import importlib.util
  import sqlite3
  from pathlib import Path
  
  import pytest
  
  
  MIG_PATH = (
      Path(__file__).resolve().parents[2]
      / "scripts" / "migrations" / "2026-05-11-fix-shadow-rows-and-mlb-ids.py"
  )
  
  
  @pytest.fixture
  def migration():
      """Load the migration module by file path (filename contains a hyphen)."""
      spec = importlib.util.spec_from_file_location("migrate_shadow", MIG_PATH)
      mod = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(mod)
      return mod
  
  
  @pytest.fixture
  def populated_db(tmp_path, monkeypatch):
      """Build a fixture DB that mimics the production bug state:
      - 2 shadow rows (one with a real twin, one without)
      - 1 NULL mlb_id rostered player (Hunter Greene pid 4643)
      - 1 normal player + 1 real twin"""
      db_path = tmp_path / "test.db"
      conn = sqlite3.connect(db_path)
      conn.executescript("""
          CREATE TABLE players (
              player_id INTEGER PRIMARY KEY,
              name TEXT,
              team TEXT,
              mlb_id INTEGER,
              is_injured INTEGER DEFAULT 0
          );
          CREATE TABLE league_rosters (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              team_name TEXT,
              team_index INTEGER,
              player_id INTEGER,
              roster_slot TEXT,
              is_user_team INTEGER DEFAULT 0,
              status TEXT,
              selected_position TEXT,
              editorial_team_abbr TEXT,
              is_undroppable INTEGER DEFAULT 0
          );
          -- Real players (good rows)
          INSERT INTO players VALUES (110, 'Corbin Burnes', 'BAL', 666197, 0);  -- real Burnes (twin of shadow)
          INSERT INTO players VALUES (4643, 'Hunter Greene', 'CIN', NULL, 0);   -- NULL mlb_id (BUG-002)
          INSERT INTO players VALUES (999, 'Active Player', 'NYY', 600000000, 0);  -- normal high mlb_id (not shadow)
          -- Shadow rows (BUG-001)
          INSERT INTO players VALUES (108, 'Corbin Burnes', 'MLB', 600770, 0);  -- shadow with twin (pid 110)
          INSERT INTO players VALUES (109, 'Some Orphan', 'MLB', 600771, 0);    -- shadow without twin
          -- League rosters reference the shadow rows + the NULL-mlb player
          INSERT INTO league_rosters (team_name, team_index, player_id, status) VALUES
              ('My Team', 0, 108, 'IL15'),    -- references shadow Burnes (should repoint to 110)
              ('My Team', 0, 109, 'active'),  -- references orphan shadow (cannot repoint, will warn)
              ('My Team', 0, 4643, 'active'); -- references NULL-mlb Greene
      """)
      conn.commit()
      conn.close()
  
      import src.database as db_mod
      monkeypatch.setattr(db_mod, "DB_PATH", db_path)
      return db_path
  
  
  def test_dry_run_makes_no_changes(populated_db, migration):
      counts = migration.run_migration(dry_run=True)
      # Counts report intent but no SQL committed
      conn = sqlite3.connect(populated_db)
      try:
          # Shadow row pid=108 should still exist
          assert conn.execute("SELECT COUNT(*) FROM players WHERE player_id = 108").fetchone()[0] == 1
          # Hunter Greene mlb_id should still be NULL
          assert conn.execute("SELECT mlb_id FROM players WHERE player_id = 4643").fetchone()[0] is None
          # league_rosters should still reference shadow pid=108
          assert conn.execute("SELECT COUNT(*) FROM league_rosters WHERE player_id = 108").fetchone()[0] == 1
      finally:
          conn.close()
      # But intent counts should match the live state
      assert counts["shadow_rows_found"] == 2
      assert counts["null_mlb_backfilled"] == 1
      assert counts["league_rosters_repointed"] == 1  # only pid=108 has a twin
      assert counts["shadow_rows_deleted"] == 1       # pid=108 now safe to delete; pid=109 stuck
  
  
  def test_commit_repoints_and_deletes(populated_db, migration):
      counts = migration.run_migration(dry_run=False)
  
      conn = sqlite3.connect(populated_db)
      try:
          # Hunter Greene mlb_id backfilled
          assert conn.execute("SELECT mlb_id FROM players WHERE player_id = 4643").fetchone()[0] == 668881
          # Shadow pid=108 (had twin) deleted
          assert conn.execute("SELECT COUNT(*) FROM players WHERE player_id = 108").fetchone()[0] == 0
          # Shadow pid=109 (no twin, still league_rostered) kept (cannot safely delete)
          assert conn.execute("SELECT COUNT(*) FROM players WHERE player_id = 109").fetchone()[0] == 1
          # League roster row for shadow 108 was repointed to real Burnes (110)
          rows = conn.execute(
              "SELECT player_id FROM league_rosters ORDER BY id"
          ).fetchall()
          assert (110,) in rows, f"Expected league_rosters now references pid=110; got {rows}"
          assert (108,) not in rows, f"Expected pid=108 reference removed; got {rows}"
      finally:
          conn.close()
  
      assert counts["null_mlb_backfilled"] == 1
      assert counts["league_rosters_repointed"] == 1
      assert counts["shadow_rows_deleted"] == 1
  
  
  def test_commit_is_idempotent(populated_db, migration):
      migration.run_migration(dry_run=False)
      counts_second = migration.run_migration(dry_run=False)
      # Second run finds 1 remaining shadow (the orphan pid=109) but does nothing destructive
      assert counts_second["null_mlb_backfilled"] == 0
      assert counts_second["league_rosters_repointed"] == 0
      # pid=109 still cannot be deleted (still in league_rosters, no twin)
      assert counts_second["shadow_rows_deleted"] == 0
  ```

- [ ] **Step 4.4: Run the migration test**

  Run: `python -m pytest tests/migrations/test_fix_shadow_rows_and_mlb_ids.py -v`
  Expected: all 3 tests PASS.

- [ ] **Step 4.5: Run the migration in `--dry-run` mode against the live DB**

  Run:
  ```bash
  python "scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py" --dry-run
  ```
  Expected output should show counts close to (per Stream B agent's audit):
  ```
  shadow_rows_found: 33
  league_rosters_repointed: <number, likely 4 since 4 are rostered>
  shadow_rows_deleted: <up to 33 minus any that can't be deleted>
  null_mlb_backfilled: 3
  null_mlb_still_missing: 0
  ```
  Inspect the log carefully. Any warnings about "shadow row has refs but no real twin" — those mean a rostered player has no recoverable real-player row. **STOP and consult before commit** if this number is greater than zero — those need manual repair (likely: add the real player row, then re-run).

- [ ] **Step 4.6: Run the migration in `--commit` mode**

  ⚠️ **DB backup must exist at `data/draft_tool.db.backup-2026-05-11-pre-wave1`** (from Step 0.3) before proceeding.

  Run:
  ```bash
  python "scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py" --commit
  ```
  Expected: log shows "Migration committed." Counts match the dry-run.

- [ ] **Step 4.7: Verify live DB state post-migration**

  Run:
  ```bash
  python -c "
  import sqlite3
  conn = sqlite3.connect('C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db')
  shadow_left = conn.execute(\"SELECT COUNT(*) FROM players WHERE team='MLB' AND mlb_id BETWEEN 600000 AND 601999\").fetchone()[0]
  null_mlb_rostered = conn.execute(\"SELECT COUNT(*) FROM players p JOIN league_rosters lr ON p.player_id=lr.player_id WHERE p.mlb_id IS NULL\").fetchone()[0]
  burnes_real = conn.execute(\"SELECT mlb_id FROM players WHERE name='Corbin Burnes' AND team != 'MLB'\").fetchone()
  greene = conn.execute(\"SELECT mlb_id FROM players WHERE player_id=4643\").fetchone()
  pepiot = conn.execute(\"SELECT mlb_id FROM players WHERE player_id=4641\").fetchone()
  crews = conn.execute(\"SELECT mlb_id FROM players WHERE player_id=922\").fetchone()
  print(f'Shadow rows remaining: {shadow_left}')
  print(f'Rostered NULL-mlb: {null_mlb_rostered}')
  print(f'Real Corbin Burnes mlb_id: {burnes_real}')
  print(f'Hunter Greene mlb_id: {greene}')
  print(f'Ryan Pepiot mlb_id: {pepiot}')
  print(f'Dylan Crews mlb_id: {crews}')
  conn.close()
  "
  ```
  Expected:
  ```
  Shadow rows remaining: 0  (or up to ~5 if any orphans without twins)
  Rostered NULL-mlb: 0
  Real Corbin Burnes mlb_id: (666197,)  (or whatever the real MLB id is)
  Hunter Greene mlb_id: (668881,)
  Ryan Pepiot mlb_id: (686752,)
  Dylan Crews mlb_id: (696285,)
  ```

  If anything looks wrong, restore from backup: `cp data/draft_tool.db.backup-2026-05-11-pre-wave1 data/draft_tool.db` and investigate.

- [ ] **Step 4.8: Verify live-stats pipeline now resolves the previously-affected players**

  Run:
  ```bash
  python -c "
  import statsapi
  # The 4 previously-affected rostered IL players + 3 NULL-mlb players
  for mlb_id, expected_name in [
      (666197, 'Corbin Burnes'),
      (668881, 'Hunter Greene'),
      (686752, 'Ryan Pepiot'),
      (696285, 'Dylan Crews'),
  ]:
      try:
          info = statsapi.lookup_player(mlb_id)
          print(f'mlb_id={mlb_id} → {info[0][\"fullName\"] if info else \"NOT FOUND\"} (expected {expected_name})')
      except Exception as e:
          print(f'mlb_id={mlb_id} → ERROR: {e}')
  "
  ```
  Expected: each line shows the expected name. If any returns "NOT FOUND" or a different name, manual fix needed (update `KNOWN_NULL_MLB_BACKFILL` in the migration with the correct id and re-run).

- [ ] **Step 4.9: Lint + format**

  Run: `python -m ruff format scripts/migrations/ tests/migrations/ && python -m ruff check scripts/migrations/ tests/migrations/`
  Expected: No diagnostics.

- [ ] **Step 4.10: Commit**

  ```bash
  git add scripts/migrations/ tests/migrations/
  git commit -m "$(cat <<'EOF'
fix(data): migration to clean shadow rows + backfill mlb_ids (BUG-001, BUG-002)

Adds scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py:
- 33 rows in players had team='MLB' and fake mlb_ids in [600000, 601999]
  that resolved to DSL/VSL prospects on MLB Stats API. 4 were on league
  rosters (Burnes, Jared Jones, Schwellenbach, Joyce) — their live stats
  silently came from minor-league teenagers.
- 3 rostered players had NULL mlb_id (Hunter Greene, Ryan Pepiot, Dylan
  Crews) — invisible to mlb_id-keyed bootstrap phases.

Migration is idempotent (running twice = no-op after first run). Three
phases: (1) backfill NULL mlb_ids from a hardcoded verified table, (2)
repoint league_rosters refs from shadow→real_twin, (3) delete shadow
rows that no longer have league_rosters refs.

Live DB migrated this commit. Backup saved at
data/draft_tool.db.backup-2026-05-11-pre-wave1.

Adds tests/migrations/test_fix_shadow_rows_and_mlb_ids.py.

Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-001, BUG-002)
EOF
)"
  ```

---

## Task 5: Permanent structural guards + CLAUDE.md update

These guards ensure none of the bugs in this Wave can return silently.

**Files:**
- Create: `tests/test_no_shadow_player_rows.py`
- Create: `tests/test_no_lr_name_column_refs.py`
- Modify: `CLAUDE.md` (Data Audit History section)

- [ ] **Step 5.1: Write the no-shadow-rows structural test**

  Create `tests/test_no_shadow_player_rows.py`:
  ```python
  """Permanent guard against BUG-001 regression: no shadow rows in players table.
  
  A "shadow row" is defined as a player with team='MLB' and mlb_id in the fake
  range used by the pre-2026-05-11 mass-bootstrap. If this test fails in CI,
  it means new shadow rows were inserted — investigate the inserting code path.
  """
  import os
  import sqlite3
  
  import pytest
  
  
  DB_PATH = os.environ.get(
      "HEATER_DB_PATH",
      "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/data/draft_tool.db",
  )
  SHADOW_RANGE = (600000, 601999)
  
  
  @pytest.mark.skipif(not os.path.isfile(DB_PATH), reason="live DB not present (CI/dev env)")
  def test_no_shadow_rows_in_players():
      conn = sqlite3.connect(DB_PATH)
      try:
          shadow = conn.execute(
              "SELECT COUNT(*) FROM players WHERE team='MLB' AND mlb_id BETWEEN ? AND ?",
              SHADOW_RANGE,
          ).fetchone()[0]
      finally:
          conn.close()
      assert shadow == 0, (
          f"BUG-001 regression: {shadow} shadow player rows found with team='MLB' "
          f"and mlb_id in {SHADOW_RANGE}. These will cause live-stats pipelines to "
          "fetch DSL/VSL prospect stats into real-player rows. "
          "Run scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py --dry-run."
      )
  
  
  @pytest.mark.skipif(not os.path.isfile(DB_PATH), reason="live DB not present")
  def test_no_rostered_null_mlb_ids():
      conn = sqlite3.connect(DB_PATH)
      try:
          null_count = conn.execute(
              """SELECT COUNT(*) FROM players p
                 JOIN league_rosters lr ON p.player_id = lr.player_id
                 WHERE p.mlb_id IS NULL"""
          ).fetchone()[0]
          if null_count:
              offenders = conn.execute(
                  """SELECT p.player_id, p.name FROM players p
                     JOIN league_rosters lr ON p.player_id = lr.player_id
                     WHERE p.mlb_id IS NULL LIMIT 10"""
              ).fetchall()
          else:
              offenders = []
      finally:
          conn.close()
      assert null_count == 0, (
          f"BUG-002 regression: {null_count} rostered players have NULL mlb_id "
          f"(invisible to live-stats pipelines). Sample: {offenders}. "
          "Add them to KNOWN_NULL_MLB_BACKFILL in scripts/migrations/2026-05-11-fix-shadow-rows-and-mlb-ids.py "
          "and re-run."
      )
  ```

- [ ] **Step 5.2: Write the no-lr-name-column-refs structural test**

  Create `tests/test_no_lr_name_column_refs.py`:
  ```python
  """Permanent guard against BUG-003 regression.
  
  Asserts that no source file references `lr.name` or `WHERE name = ?` patterns
  in the context of league_rosters (which has NO `name` column).
  """
  import re
  from pathlib import Path
  
  
  REPO_ROOT = Path(__file__).resolve().parents[1]
  
  
  def _python_files_under(*dirs: str):
      for d in dirs:
          for p in (REPO_ROOT / d).rglob("*.py"):
              # Skip the guard test itself (this file)
              if p.name == "test_no_lr_name_column_refs.py":
                  continue
              yield p
  
  
  def test_no_lr_name_join_in_src():
      """`lr.name = ...` patterns reference a non-existent column on league_rosters."""
      offenders: list[tuple[Path, int, str]] = []
      pat = re.compile(r"\blr\.name\b")
      for p in _python_files_under("src", "scripts", "pages"):
          for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
              if pat.search(line):
                  offenders.append((p, lineno, line.strip()))
      assert not offenders, (
          "BUG-003 regression: `lr.name` reference(s) found — league_rosters has no "
          f"`name` column. Offenders:\n" + "\n".join(
              f"  {p}:{n}: {ln}" for p, n, ln in offenders
          )
      )
  
  
  def test_no_league_rosters_update_by_name():
      """`UPDATE league_rosters SET ... WHERE name = ?` is a BUG-003 pattern."""
      pat = re.compile(r"UPDATE\s+league_rosters\s+SET[\s\S]*?WHERE\s+name\s*=", re.IGNORECASE)
      offenders: list[tuple[Path, str]] = []
      for p in _python_files_under("src", "scripts", "pages"):
          text = p.read_text(encoding="utf-8")
          for m in pat.finditer(text):
              offenders.append((p, m.group(0)[:120]))
      assert not offenders, (
          "BUG-003 regression: `UPDATE league_rosters SET ... WHERE name = ?` "
          f"pattern found. Use player_id. Offenders: {offenders}"
      )
  ```

- [ ] **Step 5.3: Run both guard tests; both should pass now**

  Run: `python -m pytest tests/test_no_shadow_player_rows.py tests/test_no_lr_name_column_refs.py tests/test_dynamic_park_factors_removed.py -v`
  Expected: all 6 tests PASS (3 in the dynamic_park_factors guard + 2 in shadow + 2 in lr_name).

- [ ] **Step 5.4: Run the structural-invariant suite to verify no other guards regressed**

  Run: `python -m pytest tests/test_no_*.py tests/test_pages_*.py tests/test_engine_*.py tests/test_opponent_*.py tests/test_pool_*.py tests/test_standings_*.py tests/test_pulp_*.py tests/test_refresh_*.py -v`
  Expected: all existing guards still pass; no new test fails from this Wave's changes.

- [ ] **Step 5.5: Update CLAUDE.md with new audit history entry**

  In `CLAUDE.md`, find the section `## Data Audit History (compressed)`. After the existing 2026-05-10 entry, add:

  ```markdown
  **2026-05-11 whole-repo audit (SF-29 → SF-32)** — Triggered by a 24-agent parallel bug audit (22 code reviewers + Stream B player verification + Stream C refresh infrastructure). Found ~569 actionable findings; first wave (4 HIGH-severity data-correction bugs) resolved here. Headlines: SF-29 league_rosters.name SQL bugs in injury_writeback + draft_results (0 IL flagged, 0 R1-3 picks flagged undroppable for 8+ days), SF-30 dynamic_park_factors used team OPS+ as park proxy (silent corruption every 7 days), SF-31 33 shadow rows with fake mlb_ids in [600000, 601999] (4 rostered IL players resolved to DSL prospects), SF-32 3 rostered players with NULL mlb_id (invisible to live-stats). All resolved in Wave 1 (PR # — fill in after Task 5.7).
  ```

  Also add three new entries to the **Structural Invariants** table:

  ```markdown
  | `test_no_shadow_player_rows.py` | No shadow rows in `players` (team='MLB' + fake mlb_id range); no rostered NULL mlb_ids |
  | `test_no_lr_name_column_refs.py` | No `lr.name` joins; no `UPDATE league_rosters WHERE name = ?` patterns |
  | `test_dynamic_park_factors_removed.py` | `_bootstrap_dynamic_park_factors` does not exist; orchestrator does not dispatch this source |
  ```

- [ ] **Step 5.6: Full test suite green-light**

  Run: `python -m pytest --ignore=tests/test_cheat_sheet.py -x -q`
  Expected: All tests pass. Coverage should not regress (CI requires ≥60%).

- [ ] **Step 5.7: Final commit**

  ```bash
  git add tests/test_no_shadow_player_rows.py tests/test_no_lr_name_column_refs.py CLAUDE.md
  git commit -m "$(cat <<'EOF'
test(audit): add SF-29..SF-32 structural guards + CLAUDE.md history

Adds three permanent regression guards for Wave 1 fixes:
- test_no_shadow_player_rows.py — no fake-mlb_id rows; no rostered NULL mlb_ids
- test_no_lr_name_column_refs.py — no `lr.name` joins; no `UPDATE league_rosters WHERE name=?`
- test_dynamic_park_factors_removed.py (from Task 3) — phase stays removed

Updates CLAUDE.md "Data Audit History" with SF-29..SF-32 narrative
and extends the "Structural Invariants" table with the three new guards.

Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-001..003, BUG-008)
EOF
)"
  ```

---

## Phase Final: Push + PR

- [ ] **Step F.1: Push branch**

  ```bash
  git push origin claude/audit-wave1-plan
  ```
  Expected: 5 commits pushed.

- [ ] **Step F.2: Create PR**

  ```bash
  gh pr create --title "Bug audit Wave 1: data correction migrations (BUG-001/002/003/008)" --body "$(cat <<'EOF'
## Summary

Implements Wave 1 of the [2026-05-11 bug audit findings](docs/superpowers/specs/2026-05-11-bug-audit-findings.md) — the 4 highest-impact data-correctness bugs.

| Bug | Fix |
|-----|-----|
| **BUG-003a** `_bootstrap_injury_writeback` joined on non-existent `lr.name` | Task 1: SQL change to `lr.player_id` + test |
| **BUG-003b** `_bootstrap_draft_results` updated by non-existent `name` col | Task 2: resolve name→player_id then UPDATE by player_id + test |
| **BUG-008** `_bootstrap_dynamic_park_factors` used team OPS+ as park proxy | Task 3: remove the phase + structural test |
| **BUG-001** 33 shadow rows w/ fake mlb_ids; 4 rostered IL affected | Task 4: idempotent migration script + unit tests + applied to live DB |
| **BUG-002** 3 rostered players with NULL mlb_id | Task 4: included in same migration |

Plus Task 5 adds 3 permanent structural guards (`test_no_shadow_player_rows.py`, `test_no_lr_name_column_refs.py`, `test_dynamic_park_factors_removed.py`) and updates CLAUDE.md history.

## Live DB state

- Pre-migration backup: `data/draft_tool.db.backup-2026-05-11-pre-wave1`
- Post-migration (per Step 4.7 verification): 0 shadow rows, 0 rostered NULL mlb_ids
- Affected rostered players (Burnes, Jared Jones, Schwellenbach, Joyce, Greene, Pepiot, Crews) now resolve correctly on MLB Stats API

## Test plan

- [ ] All Wave 1 tests pass (`pytest tests/test_bootstrap_injury_writeback.py tests/test_bootstrap_draft_results.py tests/test_dynamic_park_factors_removed.py tests/migrations/ tests/test_no_shadow_player_rows.py tests/test_no_lr_name_column_refs.py -v`)
- [ ] Full suite green (`pytest --ignore=tests/test_cheat_sheet.py`)
- [ ] Lint clean (`python -m ruff check . && python -m ruff format --check .`)
- [ ] Live DB verified post-migration

## Next waves

Waves 2-5 remain (silent-failure elimination, architectural cleanup, UI/page wiring, structural test expansion). Each will get its own plan.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
  ```
  Expected: PR URL returned. Do NOT auto-merge (caller will review).

- [ ] **Step F.3: Print final summary**

  Output to chat:
  ```
  Wave 1 plan execution complete.
  
  Commits: <count> on branch claude/audit-wave1-plan
  PR: <url>
  
  Live DB migrated. Backup retained at data/draft_tool.db.backup-2026-05-11-pre-wave1.
  
  Suggested next step: review PR, merge, then write Wave 2 plan (silent-failure
  elimination: BUG-004 IP outs notation, BUG-012 Bayesian formula override,
  BUG-013 two-way player, BUG-014 ECR weight keys, BUG-022 IL cold start).
  ```

---

## Self-Review

**1. Spec coverage check:**
- BUG-001 (shadow rows) → Task 4 ✅
- BUG-002 (NULL mlb_ids) → Task 4 (same migration) ✅
- BUG-003a (injury_writeback SQL) → Task 1 ✅
- BUG-003b (draft_results SQL) → Task 2 ✅
- BUG-008 (dynamic park factors) → Task 3 ✅
- Regression guards → Task 5 ✅
- Backup + rollback path → Step 0.3 + Step 4.6 rollback note ✅

**2. Placeholder scan:**
- No "TBD", "TODO", "implement later", or "fill in details" appear in any step.
- All code blocks contain full, copy-pasteable code.
- File paths are absolute or explicit.
- Test fixtures are complete (no `...` placeholders in SQL DDL).
- Commit messages are full HEREDOCs, not "write a commit message".

**3. Type/identifier consistency:**
- `_bootstrap_injury_writeback`, `_bootstrap_draft_results`, `_bootstrap_dynamic_park_factors` — consistent names referenced in tasks, tests, and commit messages.
- `players.player_id`, `players.mlb_id`, `league_rosters.player_id`, `league_rosters.is_undroppable`, `league_rosters.status` — column names consistent across all tasks.
- `SHADOW_PLAYER_ID_RANGE = (600000, 601999)` defined in migration; referenced consistently in guard test.
- `KNOWN_NULL_MLB_BACKFILL` dict — consistent shape (player_id → real mlb_id) between migration and tests.

**4. Cold-start usability:**
- Worktree path stated explicitly (Cold-Start Context).
- DB path stated explicitly.
- Bug summaries table embedded.
- Pre-flight (Phase 0) verifies environment + creates DB backup before any destructive change.
- Each task is independent — execution can resume after any commit.

Plan is complete and self-contained.

---

## Execution Notes

After this plan executes successfully, **Wave 2's plan should be written next**. Wave 2's scope (per the audit findings doc): silent-failure elimination — BUG-004 (IP outs notation), BUG-012 (Bayesian formula override), BUG-013 (two-way player injury history), BUG-014 (ECR source-weight keys), BUG-022 (IL cold-start false positives). Each fix is largely a single-module change with unit-test coverage, so Wave 2 should be similarly scoped (~5 tasks, ~30 task-steps).
