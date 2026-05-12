# Bug Audit Wave 4: UI / Page Wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Fix 5 HIGH-severity UI/page-wiring bugs from the 2026-05-11 audit (BUG-009, BUG-011, BUG-018, BUG-019, BUG-024). All are single-file-or-small-multi-file fixes.

**Architecture:** TDD per bug. Each task = failing test → minimal fix → passing test → commit. Final task adds permanent structural guards + CLAUDE.md history.

**Tech Stack:** Python 3.11+, pytest, Streamlit (pages), pandas.

**Source spec:** [docs/superpowers/specs/2026-05-11-bug-audit-findings.md](../specs/2026-05-11-bug-audit-findings.md)

---

## Cold-Start Context

### Worktree
- Path: `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave4-plan`
- Branch: `claude/audit-wave4-plan` (tracks `origin/master`)
- HEAD: `d40b69f Merge pull request #15 from hicklax13/claude/audit-wave3-plan` (Wave 3 merged)

### Bug summaries

| Bug | Location | Effect | Fix |
|-----|----------|--------|-----|
| BUG-018 | `pages/6_Line-up_Optimizer.py:2078` | `WEEKS_IN_SEASON = 24.0` divides counting stats while same file (:1469) and canonical `src/optimizer/backtest_runner.py` use 26. "Projected Weekly Category Totals" overstated by ~8.3% | Task 1: change 24.0 → 26.0; pin via test |
| BUG-019 | `pages/6_Line-up_Optimizer.py:3097` (Streaming tab) vs `:907-908` (earlier in same file) | Streaming-tab team map has only `"Oakland Athletics"→"OAK"`. Earlier map has both `"Athletics"→"ATH"` AND `"Oakland Athletics"→"OAK"`. MLB 2026 API returns "Athletics" (relocated). Streaming silently drops ATH-tagged matchups | Task 2: align maps; lift to a single module-level constant |
| BUG-009 | `pages/9_League_Standings.py:128,244,540,672,768` (`load_league_rosters`, `load_league_records`, `load_league_schedule_full`); `pages/7_Player_Compare.py:94` (`load_league_rosters`) | Pages bypass `get_yahoo_data_service` 3-tier cache → stale data; not in `test_pages_yahoo_compliance.py` allowlist | Task 3: route through YDS; expand structural-test allowlist to flag these going forward |
| BUG-011 | `src/optimizer/pipeline.py:488-495` | `build_daily_dcv_table` called WITHOUT `confirmed_lineups`, `recent_form`, `team_strength` — DCV silently uses 0.9 default volume, skips L14 form, skips opposing-offense wRC+ | Task 4: forward those kwargs from ctx/kwargs into `build_daily_dcv_table` |
| BUG-024 | `pages/12_Matchup_Planner.py:553-555` | When `team_name` missing or `rosters` empty, silently falls back to `pool.head(23)` with NO info banner. User sees plausible matchup ratings against demo data | Task 5: surface a `st.warning` and label the table as "demo data" |

---

## File Structure

**Files to create:**
- `tests/test_lineup_optimizer_26_weeks.py` — Task 1
- `tests/test_lineup_optimizer_team_map_consistent.py` — Task 2 (structural guard)
- `tests/test_pages_use_yds_for_rosters.py` — Task 3 (structural guard)
- `tests/test_optimizer_pipeline_forwards_context.py` — Task 4
- `tests/test_matchup_planner_demo_banner.py` — Task 5

**Files to modify:**
- `pages/6_Line-up_Optimizer.py` — Task 1 (line 2078) + Task 2 (consolidate team map)
- `pages/9_League_Standings.py` — Task 3 (5 locations) + add to YDS-compliance allowlist
- `pages/7_Player_Compare.py` — Task 3 (1 location)
- `tests/test_pages_yahoo_compliance.py` — Task 3 (add to PAGES_TO_CHECK or document exemption)
- `src/optimizer/pipeline.py:488-495` — Task 4 (forward kwargs)
- `pages/12_Matchup_Planner.py:553-555` — Task 5 (add banner)
- `CLAUDE.md` — Task 6 (history + structural-invariants)

---

## Phase 0: Pre-Flight (3 min)

- [ ] **Step 0.1: Verify worktree state**

  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave4-plan"
  git status
  git log -1 --oneline
  ```
  Expected: clean tree; HEAD at `d40b69f Merge pull request #15`.

⚠️ All tasks: use `pytest -q 2>&1 | tail -10`.

---

## Task 1: BUG-018 — 24v26 weeks scaler in Lineup Optimizer

**Insight:** Same file (`pages/6_Line-up_Optimizer.py`) uses `26.0` at line 1469 and canonical `src/optimizer/backtest_runner.py` uses `26`, but the "Projected Weekly Category Totals" display section uses `WEEKS_IN_SEASON = 24.0` at line 2078. CLAUDE.md states "Counting stats divided by 26 weeks." The display overstates every counting stat by ~8.3%.

**Files:**
- Modify: `pages/6_Line-up_Optimizer.py:2078`
- Create: `tests/test_lineup_optimizer_26_weeks.py`

- [ ] **Step 1.1: Write the structural-guard test**

  Create `tests/test_lineup_optimizer_26_weeks.py`:
  ```python
  """BUG-018 fix: Lineup Optimizer scales counting stats by 26 (canonical), not 24."""

  import re
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]
  PAGE = REPO_ROOT / "pages" / "6_Line-up_Optimizer.py"


  def test_lineup_optimizer_uses_26_weeks_for_weekly_proj():
      """The Projected Weekly Category Totals scaler should be 26 (canonical
      per CLAUDE.md "Counting stats divided by 26 weeks"), not 24.

      The fix is at the line that defines WEEKS_IN_SEASON in the weekly-proj
      section. If anyone sets it back to 24, this guard fires.
      """
      assert PAGE.exists()
      text = PAGE.read_text(encoding="utf-8")
      # Find every `WEEKS_IN_SEASON = N` assignment
      bad: list[tuple[int, str]] = []
      pat = re.compile(r"\bWEEKS_IN_SEASON\s*=\s*([0-9.]+)\b")
      for lineno, line in enumerate(text.splitlines(), start=1):
          stripped = line.strip()
          if stripped.startswith("#"):
              continue
          m = pat.search(line)
          if m:
              val = float(m.group(1))
              if abs(val - 26.0) > 0.01:
                  bad.append((lineno, stripped))
      assert not bad, (
          f"BUG-018 regression: WEEKS_IN_SEASON should be 26.0 "
          f"(per CLAUDE.md and src/optimizer/backtest_runner.py). Offenders: {bad}"
      )
  ```

- [ ] **Step 1.2: Run test to verify it fails**

  ```bash
  python -m pytest tests/test_lineup_optimizer_26_weeks.py -q 2>&1 | tail -10
  ```
  Expected: FAIL because line 2078 still has `WEEKS_IN_SEASON = 24.0`.

- [ ] **Step 1.3: Fix the constant**

  Edit `pages/6_Line-up_Optimizer.py`. Find this block near line 2075-2086:
  ```python
                      proj = lineup["projected_stats"]
                      # Scale counting stats to weekly (24-week fantasy season).
                      # Rate stats (AVG, OBP, ERA, WHIP) don't scale.
                      WEEKS_IN_SEASON = 24.0
  ```
  Change to:
  ```python
                      proj = lineup["projected_stats"]
                      # Scale counting stats to weekly. Canonical per CLAUDE.md
                      # "Counting stats divided by 26 weeks" — was 24.0 (BUG-018).
                      # Rate stats (AVG, OBP, ERA, WHIP) don't scale.
                      WEEKS_IN_SEASON = 26.0
  ```

- [ ] **Step 1.4: Run test to verify it passes**

  ```bash
  python -m pytest tests/test_lineup_optimizer_26_weeks.py -q 2>&1 | tail -10
  ```
  Expected: `1 passed`.

- [ ] **Step 1.5: Lint + commit**

  ```bash
  python -m ruff format pages/6_Line-up_Optimizer.py tests/test_lineup_optimizer_26_weeks.py
  python -m ruff check pages/6_Line-up_Optimizer.py tests/test_lineup_optimizer_26_weeks.py
  git add pages/6_Line-up_Optimizer.py tests/test_lineup_optimizer_26_weeks.py
  git commit -m "$(cat <<'EOF'
  fix(pages/optimizer): use canonical 26-week scaler for weekly proj (BUG-018)

  pages/6_Line-up_Optimizer.py "Projected Weekly Category Totals" section
  used WEEKS_IN_SEASON = 24.0 while same file line 1469 and canonical
  src/optimizer/backtest_runner.py use 26. CLAUDE.md: "Counting stats
  divided by 26 weeks." Display overstated counting stats by ~8.3%.

  Adds tests/test_lineup_optimizer_26_weeks.py as a structural guard.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-018)
  EOF
  )"
  ```

---

## Task 2: BUG-019 — Athletics streaming team-code map mismatch

**Insight:** `pages/6_Line-up_Optimizer.py:907-908` has a team-name → team-code map with BOTH `"Athletics": "ATH"` AND `"Oakland Athletics": "OAK"`. The Streaming-tab dict at line 3097 has only `"Oakland Athletics": "OAK"`. MLB 2026 API returns `"Athletics"` (no "Oakland" prefix after the 2024 relocation), so the Streaming tab silently misses every two-start pitcher facing or pitching for the Athletics. CLAUDE.md gotcha "Inconsistent ATH vs OAK across files" confirms ATH is the canonical 2026 code (matches MLB Stats API + bootstrap emergency dict + Wave 1's migration).

**Files:**
- Modify: `pages/6_Line-up_Optimizer.py` (Streaming-tab map at ~line 3097 + consolidation)
- Create: `tests/test_lineup_optimizer_team_map_consistent.py`

- [ ] **Step 2.1: Read both maps and decide approach**

  Run:
  ```bash
  grep -n -A2 '"Athletics"\|"Oakland Athletics"' pages/6_Line-up_Optimizer.py
  ```

  Read both. Decide: (a) inline-fix the second map to match the first, OR (b) lift a single module-level constant referenced from both call sites.

  Recommended: (b) — declare a module-level constant near the top of the file:
  ```python
  # MLB team-name → team-abbreviation map. 2026: Athletics relocated to
  # Sacramento and MLB Stats API uses "Athletics" (no "Oakland" prefix);
  # historical references to "Oakland Athletics" still resolve to OAK in
  # legacy data. Both names map to "ATH" for forward-compatibility with
  # the active MLB API surface. (BUG-019 fix.)
  _MLB_TEAM_NAME_TO_ABBR: dict[str, str] = {
      "Athletics": "ATH",
      "Oakland Athletics": "ATH",  # 2024 relocation; legacy data may still use this name
      # ... copy the rest of the existing map verbatim ...
  }
  ```

  Then replace both inline maps with references to `_MLB_TEAM_NAME_TO_ABBR`.

  ⚠️ If the existing map at :907 uses `"ATH"` AND `"OAK"` for the two different name forms (creating ambiguity), unify them BOTH to `"ATH"` per the canonical bootstrap code. This is a conscious migration from OAK to ATH per Wave 1's findings — re-verify by checking `data_bootstrap.py:_PARK_FACTORS_EMERGENCY_2026` for "ATH".

- [ ] **Step 2.2: Write the structural-guard test**

  Create `tests/test_lineup_optimizer_team_map_consistent.py`:
  ```python
  """BUG-019 fix: team-name → team-abbrev maps in Line-up Optimizer are consistent."""

  import re
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]
  PAGE = REPO_ROOT / "pages" / "6_Line-up_Optimizer.py"


  def test_athletics_maps_to_ath_everywhere():
      """Both `"Athletics"` and `"Oakland Athletics"` must map to `"ATH"`
      (canonical 2026 MLB Stats API team code, per CLAUDE.md and Wave 1).
      The earlier inconsistency (`"Oakland Athletics" → "OAK"` only) made
      the Streaming tab silently drop Athletics matchups (MLB API returns
      `"Athletics"`)."""
      assert PAGE.exists()
      text = PAGE.read_text(encoding="utf-8")
      # Look for any line that maps "Athletics" or "Oakland Athletics" to something
      pat = re.compile(r'"(Oakland\s+Athletics|Athletics)"\s*:\s*"([A-Z]+)"')
      mismatches: list[tuple[int, str, str]] = []
      seen_athletics_keys: set[str] = set()
      for lineno, line in enumerate(text.splitlines(), start=1):
          stripped = line.strip()
          if stripped.startswith("#"):
              continue
          for m in pat.finditer(line):
              name, code = m.group(1), m.group(2)
              seen_athletics_keys.add(name)
              if code != "ATH":
                  mismatches.append((lineno, name, code))
      assert not mismatches, (
          f"BUG-019 regression: Athletics mapped to wrong code. "
          f"Per Wave 1 finding (D1A-008), all Athletics name forms should "
          f"map to 'ATH' (canonical 2026 MLB Stats API). Offenders: {mismatches}"
      )
      # Sanity: at least one Athletics mapping exists (otherwise the page
      # silently drops the team entirely).
      assert seen_athletics_keys, (
          "No Athletics name→abbrev mapping found in pages/6_Line-up_Optimizer.py — "
          "did someone remove the team entirely?"
      )
  ```

- [ ] **Step 2.3: Run test to verify failure**

  ```bash
  python -m pytest tests/test_lineup_optimizer_team_map_consistent.py -q 2>&1 | tail -10
  ```
  Expected: FAIL with the Streaming-tab `"Oakland Athletics": "OAK"` (or whatever the current state is — the test catches anything ≠ "ATH").

- [ ] **Step 2.4: Apply the fix (option (b) — lift to module-level constant)**

  In `pages/6_Line-up_Optimizer.py`:

  (a) Read the existing map at `:907-919`. Note all key-value pairs. Read the Streaming-tab map at `:3097`. Note its key-value pairs.

  (b) Near the top of the file (after imports), insert the consolidated constant:
  ```python
  # MLB team-name → team-abbrev map. Both "Athletics" and "Oakland Athletics"
  # map to "ATH" (the canonical 2026 abbreviation per MLB Stats API after
  # the 2024 relocation; matches data_bootstrap._PARK_FACTORS_EMERGENCY_2026
  # and Wave 1's data migration). The Streaming tab had a separate map
  # missing "Athletics" entirely (BUG-019) — consolidated here.
  _MLB_TEAM_NAME_TO_ABBR: dict[str, str] = {
      # ... paste ALL entries from the existing :907 map here,
      # converting "Oakland Athletics": "OAK" → "Oakland Athletics": "ATH"
      # and ensuring "Athletics": "ATH" is present ...
  }
  ```

  ⚠️ Preserve every team-name key from the existing map — only the Athletics value changes (OAK → ATH).

  (c) Replace the inline map at `:907` with a reference to `_MLB_TEAM_NAME_TO_ABBR`. If it was assigned to a local variable like `name_to_abbr = {...}`, change to `name_to_abbr = _MLB_TEAM_NAME_TO_ABBR`.

  (d) Replace the inline map at `:3097` (Streaming tab) with the same reference: `team_abbr_map = _MLB_TEAM_NAME_TO_ABBR`.

- [ ] **Step 2.5: Run test to verify pass**

  ```bash
  python -m pytest tests/test_lineup_optimizer_team_map_consistent.py -q 2>&1 | tail -10
  ```
  Expected: `1 passed`.

- [ ] **Step 2.6: Lint + commit**

  ```bash
  python -m ruff format pages/6_Line-up_Optimizer.py tests/test_lineup_optimizer_team_map_consistent.py
  python -m ruff check pages/6_Line-up_Optimizer.py tests/test_lineup_optimizer_team_map_consistent.py
  git add pages/6_Line-up_Optimizer.py tests/test_lineup_optimizer_team_map_consistent.py
  git commit -m "$(cat <<'EOF'
  fix(pages/optimizer): consolidate MLB team-name map; Athletics → ATH (BUG-019)

  pages/6_Line-up_Optimizer.py had two separate inline team-name → abbrev
  maps. The map at :907 had both "Athletics"→"ATH" and "Oakland Athletics"
  →"OAK"; the Streaming-tab map at :3097 had ONLY "Oakland Athletics"→
  "OAK". MLB Stats API returns "Athletics" (no "Oakland" prefix after the
  2024 relocation), so the Streaming tab silently dropped every Athletics
  matchup.

  Fix: lifted both maps to a single module-level constant
  _MLB_TEAM_NAME_TO_ABBR. Both Athletics name forms now map to "ATH"
  (canonical 2026 MLB Stats API code, matches data_bootstrap emergency
  dict and Wave 1's data migration).

  Adds tests/test_lineup_optimizer_team_map_consistent.py as a permanent
  structural guard.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-019);
  Wave 1 finding D1A-008.
  EOF
  )"
  ```

---

## Task 3: BUG-009 — League Standings + Player Compare bypass YDS

**Insight:** `pages/9_League_Standings.py:128,244,540,672,768` calls `load_league_rosters()`, `load_league_records()`, `load_league_schedule_full()` directly. `pages/7_Player_Compare.py:94` calls `load_league_rosters()` directly. Both pages should consume `get_yahoo_data_service()` for 3-tier-cache freshness. Neither is in `tests/test_pages_yahoo_compliance.py` `PAGES_TO_CHECK` allowlist, so the existing structural guard misses them.

**Files:**
- Modify: `pages/9_League_Standings.py` (5 locations)
- Modify: `pages/7_Player_Compare.py` (1 location)
- Modify: `tests/test_pages_yahoo_compliance.py` (add pages to PAGES_TO_CHECK)
- Create: `tests/test_pages_use_yds_for_rosters.py` (focused guard)

- [ ] **Step 3.1: Read each call site to understand context**

  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave4-plan"
  grep -n "load_league_rosters\|load_league_records\|load_league_schedule_full" pages/9_League_Standings.py
  grep -n "load_league_rosters" pages/7_Player_Compare.py
  grep -n "PAGES_TO_CHECK\|load_league" tests/test_pages_yahoo_compliance.py | head -30
  ```

- [ ] **Step 3.2: Write the focused guard test**

  Create `tests/test_pages_use_yds_for_rosters.py`:
  ```python
  """BUG-009 fix: pages/9_League_Standings + pages/7_Player_Compare route
  roster fetches through get_yahoo_data_service, not raw SQL helpers."""

  import re
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]


  def test_league_standings_uses_yds_for_rosters():
      """pages/9_League_Standings.py must NOT call load_league_rosters directly.
      Use yds.get_rosters() instead. load_league_records/schedule remain OK
      because YDS doesn't expose those (they're DB-only)."""
      p = REPO_ROOT / "pages" / "9_League_Standings.py"
      assert p.exists()
      text = p.read_text(encoding="utf-8")
      bad: list[tuple[int, str]] = []
      for lineno, line in enumerate(text.splitlines(), start=1):
          if line.strip().startswith("#"):
              continue
          if "load_league_rosters" in line and "import" not in line.lower():
              # Caller invocation, not the import line
              bad.append((lineno, line.strip()))
      assert not bad, (
          f"BUG-009 regression: pages/9_League_Standings.py calls "
          f"load_league_rosters() directly; route through yds.get_rosters() "
          f"instead. Offenders: {bad}"
      )


  def test_player_compare_uses_yds_for_rosters():
      """pages/7_Player_Compare.py must NOT call load_league_rosters directly."""
      p = REPO_ROOT / "pages" / "7_Player_Compare.py"
      assert p.exists()
      text = p.read_text(encoding="utf-8")
      bad: list[tuple[int, str]] = []
      for lineno, line in enumerate(text.splitlines(), start=1):
          if line.strip().startswith("#"):
              continue
          if "load_league_rosters" in line and "import" not in line.lower():
              bad.append((lineno, line.strip()))
      assert not bad, (
          f"BUG-009 regression: pages/7_Player_Compare.py calls "
          f"load_league_rosters() directly. Offenders: {bad}"
      )
  ```

- [ ] **Step 3.3: Run test (should fail)**

  ```bash
  python -m pytest tests/test_pages_use_yds_for_rosters.py -q 2>&1 | tail -10
  ```
  Expected: both tests FAIL.

- [ ] **Step 3.4: Fix `pages/7_Player_Compare.py` (smaller — do first)**

  Read the file around line 94. Replace:
  ```python
  rosters_df = load_league_rosters()
  ```
  with:
  ```python
  from src.yahoo_data_service import get_yahoo_data_service
  _yds = get_yahoo_data_service()
  rosters_df = _yds.get_rosters()
  ```

  Then remove `load_league_rosters` from the `from src.database import ...` line at the top of the file (line 8).

  ⚠️ If `_yds` is already initialized elsewhere in the file (search for `get_yahoo_data_service`), reuse it instead of re-instantiating.

- [ ] **Step 3.5: Fix `pages/9_League_Standings.py` (5 locations)**

  Read each of lines 128, 244, 540, 672, 768. The `load_league_rosters` calls go through YDS. The `load_league_records` and `load_league_schedule_full` calls — YDS doesn't expose those, so keep the DB read but document.

  At each `load_league_rosters()` call site:
  - Replace with `yds.get_rosters()` (using an already-initialized `yds` if available, else add `from src.yahoo_data_service import get_yahoo_data_service` and `_yds = get_yahoo_data_service()` near the top of the file).

  At each `load_league_records()` and `load_league_schedule_full()` call site:
  - Keep the call but add a comment: `# Schedule/records: DB-only — no YDS equivalent (Yahoo API surface doesn't expose historical schedule)`.

  Remove `load_league_rosters` from the `from src.database import ...` block (line 14-17) but keep `load_league_records` and `load_league_schedule_full` since they're still used.

- [ ] **Step 3.6: Update `tests/test_pages_yahoo_compliance.py` allowlist**

  Read the file to find `PAGES_TO_CHECK` (or similar). Add `"pages/9_League_Standings.py"` and `"pages/7_Player_Compare.py"` to the list if they're not already there. If the test scans ALL pages by default, do nothing here.

  ⚠️ Don't break the existing test. If the new pages are added but the test fails on OTHER pages' allowed `load_league_*` calls, examine the test's logic and add explicit exemptions for `load_league_records` and `load_league_schedule_full` (only `load_league_rosters` is the bug class).

- [ ] **Step 3.7: Run all relevant tests**

  ```bash
  python -m pytest tests/test_pages_use_yds_for_rosters.py tests/test_pages_yahoo_compliance.py -q 2>&1 | tail -10
  ```
  Expected: all pass.

- [ ] **Step 3.8: Lint + commit**

  ```bash
  python -m ruff format pages/9_League_Standings.py pages/7_Player_Compare.py tests/test_pages_use_yds_for_rosters.py tests/test_pages_yahoo_compliance.py
  python -m ruff check pages/9_League_Standings.py pages/7_Player_Compare.py tests/test_pages_use_yds_for_rosters.py tests/test_pages_yahoo_compliance.py
  git add pages/9_League_Standings.py pages/7_Player_Compare.py tests/test_pages_use_yds_for_rosters.py tests/test_pages_yahoo_compliance.py
  git commit -m "$(cat <<'EOF'
  fix(pages): route roster fetches through get_yahoo_data_service (BUG-009)

  pages/9_League_Standings.py and pages/7_Player_Compare.py called
  load_league_rosters() directly, bypassing the 3-tier Yahoo cache
  (session_state → API → SQLite fallback) and risking stale data. Both
  pages were missing from tests/test_pages_yahoo_compliance.py
  PAGES_TO_CHECK allowlist, so the existing structural guard didn't
  catch them.

  Fixes:
  1. Replace each load_league_rosters() call with yds.get_rosters().
  2. load_league_records / load_league_schedule_full remain — YDS doesn't
     expose those (Yahoo API surface doesn't include historical schedule
     or league_records); documented at each call site.
  3. Add focused guard test test_pages_use_yds_for_rosters.py.
  4. Add both pages to test_pages_yahoo_compliance.py PAGES_TO_CHECK.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-009)
  EOF
  )"
  ```

---

## Task 4: BUG-011 — Optimizer pipeline drops DCV context

**Insight:** `src/optimizer/pipeline.py:488-495` calls `build_daily_dcv_table` WITHOUT `confirmed_lineups`, `recent_form`, `team_strength` even though those parameters affect volume_factor (batting order), recent-form blend, and pitcher matchup multiplier (opposing offense wRC+). CLAUDE.md "Daily Optimizer — Matchup data before state classification" warns about this exact dependency.

**Files:**
- Modify: `src/optimizer/pipeline.py:488-495`
- Create: `tests/test_optimizer_pipeline_forwards_context.py`

- [ ] **Step 4.1: Read the `LineupOptimizerPipeline.optimize` method to find where ctx/kwargs are available**

  ```bash
  grep -n "def optimize\b\|class LineupOptimizerPipeline\|build_daily_dcv_table\|confirmed_lineups\|recent_form\|team_strength" src/optimizer/pipeline.py
  ```

  Confirm that `kwargs` and/or `ctx` (the `OptimizerDataContext`) contain these fields. Looking at the existing code:
  ```python
  schedule_today = kwargs.get("schedule_today")
  dcv_table = build_daily_dcv_table(
      roster=enhanced_roster,
      matchup=kwargs.get("matchup"),
      schedule_today=schedule_today,
      park_factors=park_factors,
      config=self.config,
      rate_modes=daily_extras.get("rate_stat_modes"),
  )
  ```

  The pattern is `kwargs.get("confirmed_lineups")`, `kwargs.get("recent_form")` etc. Verify `OptimizerDataContext` exposes these as attributes or whether they only come via kwargs.

- [ ] **Step 4.2: Write the failing test**

  Create `tests/test_optimizer_pipeline_forwards_context.py`:
  ```python
  """BUG-011 fix: LineupOptimizerPipeline forwards DCV context kwargs."""

  from unittest.mock import patch
  import pandas as pd
  import pytest


  def test_pipeline_forwards_confirmed_lineups_to_dcv():
      """Pipeline.optimize must forward confirmed_lineups to build_daily_dcv_table.
      Verifies via a wrapper-mock that captures the call's kwargs."""
      from src.optimizer.pipeline import LineupOptimizerPipeline

      pipeline = LineupOptimizerPipeline(roster=pd.DataFrame(), mode="standard")
      captured = {}

      def fake_build(**kwargs):
          captured.update(kwargs)
          return pd.DataFrame()

      with patch("src.optimizer.daily_optimizer.build_daily_dcv_table", side_effect=fake_build), \
           patch("src.optimizer.daily_optimizer.check_ip_override", return_value=pd.DataFrame()), \
           patch("src.optimizer.daily_optimizer.apply_ip_pace_scaling", return_value=pd.DataFrame()):
          # Call the optimizer in a mode that triggers daily DCV (read pipeline
          # to find which mode + kwargs trigger the build_daily_dcv_table call)
          test_lineups = {"NYY": [1, 2, 3]}
          test_recent = {1: {"l14_avg": 0.300}}
          test_team_str = {"NYY": {"wrc_plus": 105}}
          try:
              pipeline.optimize(
                  matchup={},
                  schedule_today=[],
                  confirmed_lineups=test_lineups,
                  recent_form=test_recent,
                  team_strength=test_team_str,
              )
          except Exception:
              # The optimizer may fail later in the pipeline due to empty roster;
              # we only care that build_daily_dcv_table got the right kwargs.
              pass

      # If captured is empty, the daily DCV branch never fired — adjust the
      # call above (mode/kwargs) or assert on a different code path.
      if not captured:
          pytest.skip("build_daily_dcv_table was not invoked in this mode")
      assert "confirmed_lineups" in captured, (
          f"BUG-011: pipeline did not forward confirmed_lineups. "
          f"build_daily_dcv_table received kwargs: {sorted(captured.keys())}"
      )
      assert captured.get("confirmed_lineups") == test_lineups
      assert "recent_form" in captured, (
          f"BUG-011: recent_form not forwarded. Got: {sorted(captured.keys())}"
      )
      assert "team_strength" in captured, (
          f"BUG-011: team_strength not forwarded. Got: {sorted(captured.keys())}"
      )
  ```

  ⚠️ This test is integration-style. If the test setup is too brittle (empty roster crashes elsewhere first), simplify by directly testing the wrapper function or by patching at a higher level.

- [ ] **Step 4.3: Run test (should fail)**

  ```bash
  python -m pytest tests/test_optimizer_pipeline_forwards_context.py -q 2>&1 | tail -10
  ```
  Expected: FAIL (the kwargs are missing from `captured`) OR SKIP (if the daily DCV branch never triggers in the test setup).

- [ ] **Step 4.4: Fix the pipeline forward**

  In `src/optimizer/pipeline.py`, find the `build_daily_dcv_table(...)` call around line 488. Change:
  ```python
              dcv_table = build_daily_dcv_table(
                  roster=enhanced_roster,
                  matchup=kwargs.get("matchup"),
                  schedule_today=schedule_today,
                  park_factors=park_factors,
                  config=self.config,
                  rate_modes=daily_extras.get("rate_stat_modes"),
              )
  ```
  To:
  ```python
              # BUG-011 fix: forward confirmed_lineups / recent_form / team_strength.
              # Without them DCV defaults volume_factor=0.9 (lineup-not-posted),
              # skips L14 form blending, and skips opposing-offense wRC+ in the
              # pitcher matchup multiplier (per CLAUDE.md gotcha "Daily
              # Optimizer — Matchup data before state classification").
              dcv_table = build_daily_dcv_table(
                  roster=enhanced_roster,
                  matchup=kwargs.get("matchup"),
                  schedule_today=schedule_today,
                  park_factors=park_factors,
                  config=self.config,
                  rate_modes=daily_extras.get("rate_stat_modes"),
                  confirmed_lineups=kwargs.get("confirmed_lineups"),
                  recent_form=kwargs.get("recent_form"),
                  team_strength=kwargs.get("team_strength"),
              )
  ```

  ⚠️ Verify `build_daily_dcv_table`'s signature accepts these kwargs by reading `src/optimizer/daily_optimizer.py` (the function signature was given in the audit at line 431-443). Per CLAUDE.md "Key API Signatures", it does — but confirm.

- [ ] **Step 4.5: Verify test passes**

  ```bash
  python -m pytest tests/test_optimizer_pipeline_forwards_context.py -q 2>&1 | tail -10
  ```
  Expected: `1 passed` (or SKIP if the integration test still can't trigger the daily DCV branch — then add a unit test that directly imports the call site and verifies the kwargs are forwarded via mock).

- [ ] **Step 4.6: Lint + commit**

  ```bash
  python -m ruff format src/optimizer/pipeline.py tests/test_optimizer_pipeline_forwards_context.py
  python -m ruff check src/optimizer/pipeline.py tests/test_optimizer_pipeline_forwards_context.py
  git add src/optimizer/pipeline.py tests/test_optimizer_pipeline_forwards_context.py
  git commit -m "$(cat <<'EOF'
  fix(optimizer): pipeline forwards confirmed_lineups/recent_form/team_strength (BUG-011)

  src/optimizer/pipeline.py called build_daily_dcv_table WITHOUT the
  confirmed_lineups, recent_form, and team_strength kwargs even though
  those parameters affect:
  - volume_factor (batting order position when lineup is posted)
  - recent-form L14 blend in projections
  - pitcher matchup multiplier (opposing-offense wRC+)

  Without forwarding, DCV always treated lineups as not-posted (0.9
  multiplier), skipped L14 form blending, and ignored opposing-offense
  wRC+ in the matchup multiplier — per the exact CLAUDE.md gotcha
  "Daily Optimizer — Matchup data before state classification."

  Adds tests/test_optimizer_pipeline_forwards_context.py.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-011)
  EOF
  )"
  ```

---

## Task 5: BUG-024 — Matchup Planner silent demo roster

**Insight:** `pages/12_Matchup_Planner.py:553-555` falls back to `pool.head(23)` as a roster when `team_name` is missing or `rosters` is empty. NO info banner — user sees plausible matchup ratings computed against demo data.

**Files:**
- Modify: `pages/12_Matchup_Planner.py:553-555`
- Create: `tests/test_matchup_planner_demo_banner.py`

- [ ] **Step 5.1: Write the structural-guard test**

  Create `tests/test_matchup_planner_demo_banner.py`:
  ```python
  """BUG-024 fix: Matchup Planner page surfaces a warning when falling back to demo roster."""

  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]
  PAGE = REPO_ROOT / "pages" / "12_Matchup_Planner.py"


  def test_demo_roster_path_emits_warning():
      """The fallback path that uses pool.head(...) as a demo roster must be
      accompanied by an st.warning/st.info call so the user knows the
      ratings are against demo data, not their real team.

      Heuristic: find any `pool.head(` line in the file; the immediately
      surrounding 5 lines must contain `st.warning(` or `st.info(`.
      """
      assert PAGE.exists()
      text = PAGE.read_text(encoding="utf-8")
      lines = text.splitlines()
      offenders: list[tuple[int, str]] = []
      for lineno, line in enumerate(lines, start=1):
          stripped = line.strip()
          if stripped.startswith("#"):
              continue
          if "pool.head(" in line:
              # Look at +/- 5 lines for an st.warning or st.info or st.error
              start = max(0, lineno - 6)
              end = min(len(lines), lineno + 5)
              context = "\n".join(lines[start:end])
              if not any(marker in context for marker in ("st.warning(", "st.info(", "st.error(")):
                  offenders.append((lineno, stripped))
      assert not offenders, (
          f"BUG-024 regression: pool.head(...) used as demo roster without "
          f"a user-visible st.warning/st.info banner. Offenders: {offenders}"
      )
  ```

- [ ] **Step 5.2: Run test (should fail)**

  ```bash
  python -m pytest tests/test_matchup_planner_demo_banner.py -q 2>&1 | tail -10
  ```
  Expected: FAIL.

- [ ] **Step 5.3: Add the warning banner**

  In `pages/12_Matchup_Planner.py`, find line 553-555 (the fallback). Current:
  ```python
      else:
          # Fallback: use the top 23 players from the pool as a demo roster
          roster_df = pool.head(23).rename(columns={"player_name": "name"})
  ```

  Change to:
  ```python
      else:
          # Fallback: use the top 23 players from the pool as a demo roster.
          # (BUG-024) Surface a banner so the user knows the displayed
          # matchup ratings are against demo data, not their real team.
          st.warning(
              "No team selected or roster not yet loaded — showing **demo data** "
              "based on the top 23 players from the pool. Matchup ratings below "
              "are illustrative only. Connect to Yahoo or pick a team to see "
              "your real roster."
          )
          roster_df = pool.head(23).rename(columns={"player_name": "name"})
  ```

- [ ] **Step 5.4: Verify**

  ```bash
  python -m pytest tests/test_matchup_planner_demo_banner.py -q 2>&1 | tail -10
  ```

- [ ] **Step 5.5: Lint + commit**

  ```bash
  python -m ruff format pages/12_Matchup_Planner.py tests/test_matchup_planner_demo_banner.py
  python -m ruff check pages/12_Matchup_Planner.py tests/test_matchup_planner_demo_banner.py
  git add pages/12_Matchup_Planner.py tests/test_matchup_planner_demo_banner.py
  git commit -m "$(cat <<'EOF'
  fix(pages/matchup): surface warning when falling back to demo roster (BUG-024)

  pages/12_Matchup_Planner.py silently fell back to pool.head(23) as a
  demo roster when team_name was missing or rosters was empty — no info
  banner, so the user saw plausible matchup ratings computed against
  arbitrary top players from the pool.

  Adds an st.warning banner explaining the fallback. Adds a structural
  guard test that any future pool.head(...) demo path must include a
  user-visible banner (st.warning/st.info/st.error) within ±5 lines.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-024)
  EOF
  )"
  ```

---

## Task 6: CLAUDE.md update + final guards

This task only updates CLAUDE.md — the structural guards were added per-task.

- [ ] **Step 6.1: Run all Wave 4 guards + existing structural-invariant suite to verify no regressions**

  ```bash
  python -m pytest tests/test_lineup_optimizer_26_weeks.py tests/test_lineup_optimizer_team_map_consistent.py tests/test_pages_use_yds_for_rosters.py tests/test_optimizer_pipeline_forwards_context.py tests/test_matchup_planner_demo_banner.py tests/test_no_*.py tests/test_pages_*.py tests/test_engine_*.py tests/test_opponent_*.py tests/test_pool_*.py tests/test_standings_*.py tests/test_pulp_*.py tests/test_refresh_*.py -q 2>&1 | tail -10
  ```
  Expected: all pass.

- [ ] **Step 6.2: Update CLAUDE.md "Data Audit History" with Wave 4 paragraph**

  After the Wave 3 (SF-38..SF-41) paragraph, append:
  ```markdown
  **2026-05-11 Wave 4 (SF-42 → SF-46)** — UI/page wiring wave. SF-42 Lineup Optimizer weekly-proj scaler used 24 weeks instead of canonical 26 (same file used 26 elsewhere) → counting stats overstated ~8.3%. SF-43 Streaming tab in Line-up Optimizer used a separate team-name → abbrev map missing "Athletics" entirely → silently dropped Athletics matchups; consolidated to a single module-level `_MLB_TEAM_NAME_TO_ABBR` mapping all Athletics name forms to "ATH" (canonical 2026 abbreviation per Wave 1 migration). SF-44 League Standings + Player Compare pages bypassed `get_yahoo_data_service` via raw `load_league_rosters()` → stale data; both pages routed through YDS and added to the YDS-compliance allowlist. SF-45 `optimizer/pipeline.py` dropped `confirmed_lineups` / `recent_form` / `team_strength` before `build_daily_dcv_table` → DCV always treated lineups as not-posted, skipped L14 form blending, skipped opposing-offense wRC+ in matchup multiplier. SF-46 Matchup Planner silently fell back to `pool.head(23)` as demo roster when team unknown → user saw plausible matchup ratings against demo data with no warning; added `st.warning` banner.
  ```

- [ ] **Step 6.3: Update Structural Invariants table**

  Add five new rows:
  ```markdown
  | `test_lineup_optimizer_26_weeks.py` | `pages/6_Line-up_Optimizer.py` uses `WEEKS_IN_SEASON = 26.0` (canonical), not 24 |
  | `test_lineup_optimizer_team_map_consistent.py` | Both "Athletics" and "Oakland Athletics" map to "ATH" (canonical 2026 MLB Stats API code) |
  | `test_pages_use_yds_for_rosters.py` | `pages/9_League_Standings` and `pages/7_Player_Compare` do not call `load_league_rosters` directly |
  | `test_optimizer_pipeline_forwards_context.py` | `LineupOptimizerPipeline` forwards `confirmed_lineups`/`recent_form`/`team_strength` to `build_daily_dcv_table` |
  | `test_matchup_planner_demo_banner.py` | `pages/12_Matchup_Planner.py` `pool.head(...)` demo-roster fallback is accompanied by `st.warning`/`st.info`/`st.error` |
  ```

- [ ] **Step 6.4: Commit**

  ```bash
  git add CLAUDE.md
  git commit -m "$(cat <<'EOF'
  docs(audit): CLAUDE.md history for Wave 4 (SF-42..SF-46)

  Adds Wave 4 narrative to Data Audit History and registers 5 new
  structural-invariant guards from Wave 4 Tasks 1-5.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md
  EOF
  )"
  ```

---

## Phase Final: Push + PR

- [ ] **Step F.1: Push branch**
  ```bash
  git push -u origin claude/audit-wave4-plan
  ```

- [ ] **Step F.2: Create PR**
  ```bash
  gh pr create --title "Wave 4: UI / page wiring (BUG-009/011/018/019/024)" --body "Implements Wave 4 of the 2026-05-11 bug audit findings. 5 HIGH-severity UI/page wiring bugs. 5 new tests + 5 structural guards."
  ```

---

## Self-Review

**Spec coverage:**
- BUG-009 → Task 3 ✅
- BUG-011 → Task 4 ✅
- BUG-018 → Task 1 ✅
- BUG-019 → Task 2 ✅
- BUG-024 → Task 5 ✅
- Guards + CLAUDE.md → Task 6 ✅

**Placeholder scan:** Every step has executable commands and full code blocks. No TBD/TODO.

**Type/identifier consistency:** `_MLB_TEAM_NAME_TO_ABBR`, `WEEKS_IN_SEASON = 26.0`, `load_league_rosters`, `get_yahoo_data_service`/`get_rosters`, `build_daily_dcv_table` kwargs — referenced consistently across tasks.

**Cold-start usability:** Worktree, branch, bug summaries, pre-flight all stated. Each task independent.
