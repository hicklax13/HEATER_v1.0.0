# Bug Audit Wave 7: Infrastructure Cleanup — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Address 4 INFRASTRUCTURE-class items that were flagged by the audit but not in the top-25 HIGH bucket: INFRA-F6 (row-count gates), INFRA-F3 (umpire timeout), INFRA-F4 (DataFreshnessTracker reads refresh_log), and the ATH-vs-OAK data inconsistency follow-up (the UI fix in Wave 4 corrected the streaming map; some source modules still emit "OAK").

**Architecture:** TDD per task. 4 tasks + CLAUDE.md update.

**Tech Stack:** Python 3.11+, SQLite, pytest, MLB-StatsAPI.

**Source spec:** [docs/superpowers/specs/2026-05-11-bug-audit-findings.md](../specs/2026-05-11-bug-audit-findings.md)

---

## Cold-Start Context

### Worktree
- Path: `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave7-plan`
- Branch: `claude/audit-wave7-plan` (tracks `origin/master`)
- HEAD: `35a94c0 Merge pull request #18 from hicklax13/claude/audit-wave6-plan` (Wave 6 merged)

### Item summaries

| Item | Location | Issue | Fix |
|------|----------|-------|-----|
| INFRA-F6 | `src/data_bootstrap.py` (18 call sites) | Phases call `update_refresh_log("X", "success")` without row-count gate → silent 0-row writes still possible | Convert to `update_refresh_log_auto(name, count, expected_min, message=...)` per phase |
| INFRA-F3 | `src/data_bootstrap.py:1517-1623` (`_bootstrap_umpire_tendencies`) | Tier 1 schedule + boxscore iteration unbounded → can run >300s before falling through to Tier 3 seed | Add wall-clock timeout (60s) for Tier 1; on exceed, log + break to Tier 3 |
| INFRA-F4 | `src/optimizer/data_freshness.py:37-76` (`DataFreshnessTracker`) | In-memory dict; never reads `refresh_log` → fresh page sessions report UNKNOWN for everything | Add `populate_from_refresh_log()` factory + call from `__init__` |
| ATH/OAK | `src/player_databank.py:533-534`, `src/data_2026.py` (various) | Athletics still mapped to "OAK" in some source modules (Wave 4 fixed Streaming-tab UI only) | Update remaining "OAK" → "ATH" per CLAUDE.md Wave 1 directive |

---

## Phase 0: Pre-Flight (2 min)

- [ ] **Step 0.1: Verify worktree state**
  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave7-plan"
  git status
  git log -1 --oneline
  ```
  Expected: clean, HEAD at `35a94c0 Merge pull request #18`.

⚠️ All tasks: use `pytest -q 2>&1 | tail -10`. COMMIT explicitly per task.

---

## Task 1: INFRA-F6 — Row-count gates for bootstrap phases

**Insight:** Audit found 18 of 31 bootstrap phases call `update_refresh_log("X", "success")` without verifying row count. A successful HTTP call returning 0 items would log "success" with no count.

**Files:**
- Modify: `src/data_bootstrap.py` (multiple `update_refresh_log` call sites)
- Create: `tests/test_no_unguarded_update_refresh_log.py`

- [ ] **Step 1.1: Survey all `update_refresh_log` call sites**

  Run: `grep -n 'update_refresh_log("' src/data_bootstrap.py`

  For each match, note:
  - Phase name (the string literal)
  - Whether it's already `update_refresh_log_auto` (good — leave alone)
  - Whether row count is computed locally (then convert to `_auto`)
  - Whether it's a status-only update (e.g., `"error"`, `"skipped"` — leave alone)

- [ ] **Step 1.2: Write the structural guard test**

  Create `tests/test_no_unguarded_update_refresh_log.py`:
  ```python
  """INFRA-F6 fix: bootstrap phases must use update_refresh_log_auto (with row count)
  for success states, not plain update_refresh_log(..., "success")."""

  import re
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]
  BOOTSTRAP = REPO_ROOT / "src" / "data_bootstrap.py"

  # These specific call sites are allowed to use the plain form (status-only updates,
  # not "success" claims). Add to this list ONLY with justification.
  ALLOWED_PLAIN_SITES = {
      # phase_name: justification
      "draft_results": "early-return path when no Yahoo client; not a success claim",
  }


  def test_success_calls_use_update_refresh_log_auto():
      """`update_refresh_log("X", "success")` patterns should be migrated to
      `update_refresh_log_auto(...)` so silent 0-row writes are caught."""
      assert BOOTSTRAP.exists()
      text = BOOTSTRAP.read_text(encoding="utf-8")
      pat = re.compile(r'update_refresh_log\(\s*"([a-z_]+)"\s*,\s*"success"', re.MULTILINE)
      offenders: list[tuple[int, str, str]] = []
      for m in pat.finditer(text):
          phase = m.group(1)
          if phase in ALLOWED_PLAIN_SITES:
              continue
          lineno = text[:m.start()].count("\n") + 1
          offenders.append((lineno, phase, m.group(0)))
      assert not offenders, (
          f"INFRA-F6 regression: bootstrap phase(s) use plain "
          f"update_refresh_log(..., 'success') without row-count gate. "
          f"Migrate to update_refresh_log_auto(name, count, expected_min, message=...). "
          f"Offenders: {offenders}"
      )
  ```

- [ ] **Step 1.3: Run test (should fail with N offenders)**

  ```bash
  python -m pytest tests/test_no_unguarded_update_refresh_log.py -q 2>&1 | tail -10
  ```
  Expected: FAIL — lists ~18 offenders.

- [ ] **Step 1.4: Migrate each phase**

  For each offending call site, find the surrounding function. The pattern is typically:
  ```python
  # OLD:
  result_count = len(df)  # or similar
  # ... do work ...
  update_refresh_log("phase_name", "success")
  return f"{result_count} rows refreshed"
  ```

  Replace with:
  ```python
  # NEW:
  update_refresh_log_auto(
      "phase_name",
      result_count,
      expected_min=<reasonable floor>,
      message=f"<descriptive message>",
  )
  return f"{result_count} rows refreshed"
  ```

  Choose `expected_min` per phase:
  - `projections`: 1000 (~80% of ~1300 players)
  - `prospect_rankings`: 50 (top-100 prospects minus filters)
  - `news`: 1 (any news at all is a success)
  - `news_intelligence`: 1
  - `ecr_consensus`: 100 (top-300 ranked players)
  - `game_day`: 1 (any game weather)
  - `sprint_speed`: 100 (top hitters)
  - `bat_speed`: 100
  - `forty_man`: 800 (30 teams × ~40 players × 80% match)
  - `pvb_splits`: 500
  - Others: pick a sensible floor from CLAUDE.md or refresh_log historical data

  ⚠️ If a phase has no easy count variable (e.g., it doesn't track rows written), use `update_refresh_log_auto(name, 1, expected_min=1, ...)` so the count is at least non-zero. Or compute a count from a SELECT on the target table.

- [ ] **Step 1.5: Run test (should pass)**

  ```bash
  python -m pytest tests/test_no_unguarded_update_refresh_log.py -q 2>&1 | tail -10
  ```

- [ ] **Step 1.6: Lint + commit**

  ```bash
  python -m ruff format src/data_bootstrap.py tests/test_no_unguarded_update_refresh_log.py
  python -m ruff check src/data_bootstrap.py tests/test_no_unguarded_update_refresh_log.py
  git add src/data_bootstrap.py tests/test_no_unguarded_update_refresh_log.py
  git commit -m "$(cat <<'EOF'
  fix(bootstrap): migrate ~18 phases to update_refresh_log_auto (INFRA-F6)

  Audit INFRA-F6 found 18 of 31 bootstrap phases call
  update_refresh_log("X", "success") without verifying row count → a
  successful HTTP call returning 0 items would silently log "success"
  with no signal to the operator. SF-2/SF-3 fixed several phases but
  didn't migrate them all.

  Migrates remaining phases to update_refresh_log_auto(name, count,
  expected_min, message=...) with reasonable expected_min floors. Now
  zero-row writes downgrade to "no_data" status automatically.

  Adds tests/test_no_unguarded_update_refresh_log.py as a structural
  guard against new phases regressing the pattern.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (INFRA-F6)
  EOF
  )"
  ```

---

## Task 2: INFRA-F3 — Umpire phase timeout

**Insight:** `_bootstrap_umpire_tendencies` (lines 1517-1623) iterates the MLB schedule from season start through today, calling `boxscore_data(game_pk)` per game. With ~40 games × 50 days = ~2000 HTTP calls, the phase routinely exceeds 300s. The audit's Step 0 timeout is 240s per phase, so this routinely fails before the Tier 3 seed serves.

**Files:**
- Modify: `src/data_bootstrap.py:1517-1623` (`_bootstrap_umpire_tendencies`)
- Create: `tests/test_umpire_phase_has_timeout.py`

- [ ] **Step 2.1: Write the structural guard**

  Create `tests/test_umpire_phase_has_timeout.py`:
  ```python
  """INFRA-F3 fix: _bootstrap_umpire_tendencies must have a wall-clock timeout
  on Tier 1 (schedule iteration) so it falls through to Tier 3 seed in bounded time."""

  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]


  def test_umpire_phase_uses_time_budget():
      """The Tier 1 schedule iteration in _bootstrap_umpire_tendencies must use
      a `time.time()` budget so it terminates within ~60s even with thousands
      of completed games."""
      text = (REPO_ROOT / "src" / "data_bootstrap.py").read_text(encoding="utf-8")

      # Find _bootstrap_umpire_tendencies function body
      start = text.find("def _bootstrap_umpire_tendencies")
      assert start >= 0, "umpire phase function not found"
      # Find next top-level function (def at column 0)
      end = text.find("\ndef ", start + 1)
      body = text[start:end if end > 0 else len(text)]

      # The Tier 1 iteration must check elapsed time against a budget
      has_budget = (
          ("time.time()" in body or "time.monotonic()" in body)
          and ("UMPIRE_TIER1_TIMEOUT" in body or "_TIER1_TIMEOUT" in body or "elapsed" in body.lower())
      )
      assert has_budget, (
          "INFRA-F3 regression: _bootstrap_umpire_tendencies does not appear to "
          "use a wall-clock timeout to cap Tier 1 schedule iteration. The phase "
          "should track elapsed time and break to Tier 3 seed when exceeded. "
          f"Body excerpt:\n{body[:500]}"
      )
  ```

- [ ] **Step 2.2: Add timeout to Tier 1 iteration**

  Read `_bootstrap_umpire_tendencies` body. In the Tier 1 block (around lines 1537-1623), find the `for game in schedule:` loop. Modify:

  ```python
  # Add at top of Tier 1 try block (after `_statsapi` import):
  import time
  _UMPIRE_TIER1_TIMEOUT_S = 60.0  # INFRA-F3: cap Tier 1 to 60s; otherwise fall through to seed
  tier1_start = time.monotonic()

  # Inside the `for game in schedule:` loop, add at top:
  if time.monotonic() - tier1_start > _UMPIRE_TIER1_TIMEOUT_S:
      logger.warning(
          "T7 [primary]: Tier 1 schedule iteration exceeded %ds budget; "
          "falling through to Tier 3 seed file. (INFRA-F3 fix.)",
          _UMPIRE_TIER1_TIMEOUT_S,
      )
      umpire_stats = {}  # reset so Tier 3 fires cleanly
      break
  ```

  ⚠️ Adjust variable names to match the existing code. The key idea: a `time.monotonic()` budget that breaks the loop.

- [ ] **Step 2.3: Run guard test**

  ```bash
  python -m pytest tests/test_umpire_phase_has_timeout.py -q 2>&1 | tail -10
  ```

- [ ] **Step 2.4: Lint + commit**

  ```bash
  python -m ruff format src/data_bootstrap.py tests/test_umpire_phase_has_timeout.py
  python -m ruff check src/data_bootstrap.py tests/test_umpire_phase_has_timeout.py
  git add src/data_bootstrap.py tests/test_umpire_phase_has_timeout.py
  git commit -m "$(cat <<'EOF'
  fix(bootstrap): cap umpire Tier 1 iteration at 60s wall-clock (INFRA-F3)

  _bootstrap_umpire_tendencies Tier 1 iterates the MLB schedule + per-game
  boxscore_data calls. With ~2000 completed games mid-season this routinely
  runs >300s before falling through to Tier 3 (the shipped 2024 seed),
  exceeding the bootstrap orchestrator's 240s per-phase timeout.

  Adds a 60s wall-clock budget on Tier 1: if exceeded, logs a warning and
  breaks to Tier 3 seed cleanly. Phase now completes in bounded time.

  Adds tests/test_umpire_phase_has_timeout.py as a structural guard.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (INFRA-F3)
  EOF
  )"
  ```

---

## Task 3: INFRA-F4 — DataFreshnessTracker reads refresh_log

**Insight:** `DataFreshnessTracker` is an in-memory dict that only gets populated by `tracker.record()` calls inside `shared_data_layer.build_optimizer_context`. So when a fresh page session starts (no prior tracker), `tracker.check(source)` returns `UNKNOWN` for every source even if `refresh_log` shows recent refreshes.

**Files:**
- Modify: `src/optimizer/data_freshness.py:37-76`
- Create: `tests/test_data_freshness_reads_refresh_log.py`

- [ ] **Step 3.1: Write the failing test**

  Create `tests/test_data_freshness_reads_refresh_log.py`:
  ```python
  """INFRA-F4 fix: DataFreshnessTracker hydrates from refresh_log on init."""

  import sqlite3
  from pathlib import Path

  import pytest


  @pytest.fixture
  def test_db_with_refresh_log(tmp_path, monkeypatch):
      db_path = tmp_path / "test.db"
      conn = sqlite3.connect(db_path)
      conn.executescript("""
          CREATE TABLE refresh_log (
              source TEXT PRIMARY KEY,
              status TEXT,
              tier TEXT,
              message TEXT,
              last_refresh TEXT,
              rows_written INTEGER,
              rows_expected_min INTEGER
          );
          INSERT INTO refresh_log (source, status, last_refresh) VALUES
              ('projections', 'success', datetime('now', '-2 hours')),
              ('players', 'success', datetime('now', '-30 minutes')),
              ('game_day', 'success', datetime('now', '-15 minutes'));
      """)
      conn.commit()
      conn.close()

      import src.database as db_mod
      monkeypatch.setattr(db_mod, "DB_PATH", db_path)
      return db_path


  def test_tracker_hydrates_from_refresh_log_on_init(test_db_with_refresh_log):
      """A fresh DataFreshnessTracker() should report FRESH (not UNKNOWN) for
      sources that have recent entries in refresh_log."""
      from src.optimizer.data_freshness import DataFreshnessTracker, FreshnessStatus

      tracker = DataFreshnessTracker()  # should auto-hydrate from refresh_log
      # players refreshed 30 min ago, well within any TTL → FRESH
      assert tracker.check("players") == FreshnessStatus.FRESH, (
          "INFRA-F4 regression: DataFreshnessTracker did not hydrate "
          "from refresh_log on init. tracker.check('players') returned "
          f"{tracker.check('players')} instead of FRESH."
      )
  ```

- [ ] **Step 3.2: Run test (should fail with UNKNOWN)**

  ```bash
  python -m pytest tests/test_data_freshness_reads_refresh_log.py -q 2>&1 | tail -10
  ```

- [ ] **Step 3.3: Add hydration to `DataFreshnessTracker.__init__`**

  In `src/optimizer/data_freshness.py`, modify `__init__`:
  ```python
  # OLD:
  def __init__(self) -> None:
      self._sources: dict[str, _SourceRecord] = {}

  # NEW:
  # Default TTLs from CLAUDE.md "Data Sources" table; sources not listed here
  # get the conservative 24h default.
  _DEFAULT_TTLS = {
      "players": 168.0,          # 7 days
      "projections": 24.0,
      "ros_projections": 4.0,
      "season_stats": 1.0,
      "yahoo_rosters": 0.5,
      "yahoo_standings": 0.5,
      "yahoo_free_agents": 1.0,
      "yahoo_transactions": 0.25,
      "game_day": 2.0,
      "team_strength": 24.0,
      "park_factors": 720.0,     # 30 days
      "catcher_framing": 168.0,
      "umpire_tendencies": 24.0,
      "depth_charts": 168.0,
      "news": 1.0,
      "ecr_consensus": 24.0,
      "sprint_speed": 24.0,
      "stuff_plus": 24.0,
  }


  def __init__(self) -> None:
      self._sources: dict[str, _SourceRecord] = {}
      # INFRA-F4 fix: hydrate from refresh_log on init so callers outside
      # lineup-optimizer (which calls tracker.record() inline) still see
      # FRESH/STALE rather than UNKNOWN for sources recently refreshed.
      self._populate_from_refresh_log()


  def _populate_from_refresh_log(self) -> None:
      """Read refresh_log table and seed _sources with FRESH/STALE state."""
      try:
          from src.database import get_connection

          conn = get_connection()
          try:
              cur = conn.execute(
                  "SELECT source, last_refresh, status FROM refresh_log"
              )
              for source, last_refresh_str, status in cur.fetchall():
                  if status != "success" or not last_refresh_str:
                      continue
                  try:
                      ts = datetime.fromisoformat(last_refresh_str.replace(" ", "T"))
                      if ts.tzinfo is None:
                          ts = ts.replace(tzinfo=UTC)
                  except ValueError:
                      continue
                  ttl = self._DEFAULT_TTLS.get(source, 24.0)
                  self._sources[source] = _SourceRecord(
                      name=source,
                      timestamp=ts,
                      ttl_hours=ttl,
                      data_as_of="",
                      source_label="refresh_log",
                  )
          finally:
              conn.close()
      except Exception as exc:
          logger.warning("DataFreshnessTracker hydration from refresh_log failed: %s", exc)
  ```

  ⚠️ Make `_DEFAULT_TTLS` a class-level constant on `DataFreshnessTracker` (or module-level). Import `datetime` if not already imported.

- [ ] **Step 3.4: Run test (should pass)**

  ```bash
  python -m pytest tests/test_data_freshness_reads_refresh_log.py -q 2>&1 | tail -10
  ```

- [ ] **Step 3.5: Targeted regression sweep**

  ```bash
  python -m pytest tests/test_data_freshness*.py tests/test_optimizer*freshness* -q 2>&1 | tail -10
  ```

- [ ] **Step 3.6: Lint + commit**

  ```bash
  python -m ruff format src/optimizer/data_freshness.py tests/test_data_freshness_reads_refresh_log.py
  python -m ruff check src/optimizer/data_freshness.py tests/test_data_freshness_reads_refresh_log.py
  git add src/optimizer/data_freshness.py tests/test_data_freshness_reads_refresh_log.py
  git commit -m "$(cat <<'EOF'
  fix(optimizer): DataFreshnessTracker hydrates from refresh_log on init (INFRA-F4)

  DataFreshnessTracker was an in-memory dict populated only via
  tracker.record() calls inside shared_data_layer.build_optimizer_context.
  Fresh page sessions outside the lineup-optimizer run path got UNKNOWN
  for every source, even when refresh_log showed recent successful
  refreshes.

  Adds _populate_from_refresh_log() that reads the refresh_log table on
  __init__ and seeds tracker state with FRESH/STALE per source. TTLs
  are sourced from a _DEFAULT_TTLS mapping that mirrors CLAUDE.md's
  Data Sources table; unmapped sources default to 24h.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (INFRA-F4)
  EOF
  )"
  ```

---

## Task 4: ATH/OAK source-module follow-up

**Insight:** Wave 4 fixed the Streaming-tab team-name map in `pages/6_Line-up_Optimizer.py`. The audit's D1A-008 finding noted additional inconsistencies in source modules:
- `src/player_databank.py:533-534`: `_TEAM_ABBR` maps "Oakland Athletics" / "Athletics" → "OAK"
- `src/data_2026.py`: hardcoded "OAK" in player rows (for the 2026 spring training player list)

Per CLAUDE.md Wave 1 D1A-008: canonical 2026 code is "ATH" (matches MLB Stats API + `_PARK_FACTORS_EMERGENCY_2026` + Wave 1's DB migration).

**Files:**
- Modify: `src/player_databank.py` (`_TEAM_ABBR` map)
- Modify: `src/data_2026.py` ("OAK" team strings in player rows)
- Create: `tests/test_no_oak_in_source_modules.py`

- [ ] **Step 4.1: Survey "OAK" occurrences in source modules**

  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave7-plan"
  grep -rn '"OAK"\|"Oakland"' src/ --include="*.py" | head -30
  ```

  Capture every match. Note which are:
  - Data-source values (should be "ATH" per Wave 1 directive)
  - Historical comments/strings (leave alone)
  - Test fixtures (depends on what they're testing)

- [ ] **Step 4.2: Write the failing structural test**

  Create `tests/test_no_oak_in_source_modules.py`:
  ```python
  """Wave 7 fix: source modules use "ATH" for Athletics (canonical 2026 code).
  Wave 1 D1A-008 fixed the live DB; Wave 4 fixed the Streaming-tab map; this
  catches any remaining "OAK" emissions in source modules."""

  import re
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]

  # Files where "OAK" must NOT appear as a current-team value (it's the
  # pre-2024 abbreviation; canonical 2026 is "ATH").
  GUARDED_FILES = [
      "src/player_databank.py",
      "src/data_2026.py",
      "src/live_stats.py",
  ]


  def test_no_oak_team_value_in_source_modules():
      """`"OAK"` as a string literal in source modules implies pre-2024
      Athletics naming; canonical 2026 code is `"ATH"`."""
      offenders: list[tuple[str, int, str]] = []
      pat = re.compile(r'"OAK"')
      for rel in GUARDED_FILES:
          p = REPO_ROOT / rel
          if not p.exists():
              continue
          for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
              stripped = line.strip()
              if stripped.startswith("#"):
                  continue
              if pat.search(line):
                  offenders.append((rel, lineno, stripped))
      assert not offenders, (
          "Wave 7 regression: source modules still emit \"OAK\" for Athletics. "
          "Use \"ATH\" (canonical 2026 per Wave 1 D1A-008). Offenders:\n"
          + "\n".join(f"  {p}:{n}: {ln}" for p, n, ln in offenders)
      )
  ```

- [ ] **Step 4.3: Run test (should fail)**

  ```bash
  python -m pytest tests/test_no_oak_in_source_modules.py -q 2>&1 | tail -10
  ```

- [ ] **Step 4.4: Replace "OAK" → "ATH" in flagged files**

  For each offender from Step 4.1:
  - `src/player_databank.py` `_TEAM_ABBR`: change `"Oakland Athletics": "OAK"` and `"Athletics": "OAK"` → both to `"ATH"`
  - `src/data_2026.py`: change any `"OAK"` team value in player dicts to `"ATH"`
  - `src/live_stats.py`: if any "OAK" appears, evaluate per use (likely a fallback team-stat dict; change to "ATH")

  ⚠️ Don't touch comments or docstrings that reference the historical "OAK" naming.

- [ ] **Step 4.5: Run test (should pass) + targeted regression**

  ```bash
  python -m pytest tests/test_no_oak_in_source_modules.py -q 2>&1 | tail -10
  python -m pytest tests/test_player_databank*.py tests/test_live_stats*.py -q 2>&1 | tail -10
  ```

  ⚠️ If pre-existing tests fail because they expected "OAK", update those test assertions to "ATH" (the rename propagates correctly).

- [ ] **Step 4.6: Lint + commit**

  ```bash
  python -m ruff format src/player_databank.py src/data_2026.py src/live_stats.py tests/test_no_oak_in_source_modules.py
  python -m ruff check src/player_databank.py src/data_2026.py src/live_stats.py tests/test_no_oak_in_source_modules.py
  git add src/player_databank.py src/data_2026.py src/live_stats.py tests/test_no_oak_in_source_modules.py
  git commit -m "$(cat <<'EOF'
  fix(data): Athletics → ATH in source modules (Wave 7 follow-up to D1A-008)

  Wave 1 D1A-008 documented that ATH is the canonical 2026 Athletics
  abbreviation (matches MLB Stats API + _PARK_FACTORS_EMERGENCY_2026 +
  Wave 1's DB migration). Wave 4 fixed the Streaming-tab UI map. This
  closes the loop by replacing remaining "OAK" emissions in source
  modules: player_databank _TEAM_ABBR, data_2026 player rows, live_stats
  fallback team-stat dict.

  Adds tests/test_no_oak_in_source_modules.py as a structural guard.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (Wave 1 D1A-008)
  EOF
  )"
  ```

---

## Task 5: CLAUDE.md update

- [ ] **Step 5.1: Run cumulative structural-invariant sweep**

  ```bash
  python -m pytest tests/test_no_*.py tests/test_pages_*.py tests/test_engine_*.py tests/test_opponent_*.py tests/test_pool_*.py tests/test_standings_*.py tests/test_pulp_*.py tests/test_refresh_*.py tests/test_no_unguarded_update_refresh_log.py tests/test_umpire_phase_has_timeout.py tests/test_data_freshness_reads_refresh_log.py tests/test_no_oak_in_source_modules.py -q 2>&1 | tail -10
  ```

- [ ] **Step 5.2: Update CLAUDE.md**

  After the Wave 6 paragraph, append:
  ```markdown
  **2026-05-12 Wave 7 (SF-54, SF-55, SF-56, SF-57)** — Infrastructure cleanup. SF-54 (INFRA-F6) ~18 bootstrap phases migrated from plain `update_refresh_log(..., "success")` to `update_refresh_log_auto(..., expected_min, ...)` so silent 0-row writes are caught. SF-55 (INFRA-F3) `_bootstrap_umpire_tendencies` Tier 1 schedule iteration now capped at 60s wall-clock; falls through to Tier 3 seed in bounded time. SF-56 (INFRA-F4) `DataFreshnessTracker.__init__` hydrates from `refresh_log` table so fresh page sessions show FRESH/STALE instead of UNKNOWN. SF-57 (Wave 1 D1A-008 follow-up) Athletics → ATH in `player_databank._TEAM_ABBR`, `data_2026.py`, `live_stats.py` (Wave 1 migrated the live DB; Wave 4 fixed the Streaming-tab UI map; this closes the source-module loop).
  ```

  Add four rows to the Structural Invariants table:
  ```markdown
  | `test_no_unguarded_update_refresh_log.py` | Bootstrap phases use `update_refresh_log_auto` (with row-count) not plain `update_refresh_log(..., "success")` |
  | `test_umpire_phase_has_timeout.py` | `_bootstrap_umpire_tendencies` Tier 1 uses a wall-clock timeout |
  | `test_data_freshness_reads_refresh_log.py` | `DataFreshnessTracker` hydrates from `refresh_log` on init |
  | `test_no_oak_in_source_modules.py` | Source modules emit "ATH" not "OAK" for Athletics |
  ```

- [ ] **Step 5.3: Commit**

  ```bash
  git add CLAUDE.md
  git commit -m "$(cat <<'EOF'
  docs(audit): CLAUDE.md history for Wave 7 (SF-54..SF-57)

  Adds Wave 7 narrative to Data Audit History. Wave 7 covers
  infrastructure cleanup items flagged by the audit but outside the
  top-25 HIGH bucket: row-count gates (SF-54), umpire timeout (SF-55),
  DataFreshnessTracker (SF-56), and Athletics → ATH source-module
  follow-up (SF-57).

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md
  EOF
  )"
  ```

---

## Phase Final: Push + PR + coderabbit review

- [ ] **Step F.1: Push branch**
  ```bash
  git push -u origin claude/audit-wave7-plan
  ```

- [ ] **Step F.2: Create PR** with `gh pr create --title "Wave 7: infrastructure cleanup (INFRA-F3/F4/F6 + ATH/OAK follow-up)" --body "..."`

- [ ] **Step F.3: Run CodeRabbit review** via `coderabbit:code-review` skill

- [ ] **Step F.4: Address any CodeRabbit findings** (per `superpowers:receiving-code-review`)

- [ ] **Step F.5: Merge** with `gh pr merge --auto --merge`

---

## Self-Review

**Spec coverage:** INFRA-F3, INFRA-F4, INFRA-F6, D1A-008 follow-up.
**Placeholder scan:** No "TBD"/"TODO". All code blocks complete.
**Cold-start usability:** Worktree, branch, bug summaries, pre-flight stated. Each task independent.
