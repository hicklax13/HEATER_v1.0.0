# Bug Audit Wave 5: Miscellaneous Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development.

**Goal:** Fix 5 HIGH/MEDIUM-severity bugs from the 2026-05-11 audit (BUG-017, BUG-020, BUG-021, BUG-023, BUG-025) that are smaller-scope than the prior waves. BUG-010 (_LC singletons) and BUG-015 (survival calibrator ADP plumbing) are reserved for a dedicated Wave 6.

**Architecture:** TDD per task. Each task = failing test → minimal fix → passing test → commit. Final task: CLAUDE.md history.

**Tech Stack:** Python 3.11+, pytest, pandas, Streamlit config.

**Source spec:** [docs/superpowers/specs/2026-05-11-bug-audit-findings.md](../specs/2026-05-11-bug-audit-findings.md)

---

## Cold-Start Context

### Worktree
- Path: `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave5-plan`
- Branch: `claude/audit-wave5-plan` (tracks `origin/master`)
- HEAD: `2cd2c1d Merge pull request #16 from hicklax13/claude/audit-wave4-plan`

### Bug summaries

| Bug | Location | Effect | Fix |
|-----|----------|--------|-----|
| BUG-017 | `src/trade_intelligence.py:543-547` | `compute_fa_comparisons` initializes `best_fa_value=0.0` then only overrides on `val > best_fa_value`. For negative-SGP players (target_sgp < 0), the FA comparison defaults to 0 → `fa_pct = 0/neg = 0` → marks "has_alternative=False" even when alternative FAs exist | Task 1: seed `best_fa_value` from the first candidate, compare on absolute-value basis OR sign-aware |
| BUG-020 | `src/closer_monitor.py:211` | Exact `name + team` match for closer lookup fails on accent/punctuation/suffix mismatches (José Suárez, Andrés Muñoz, etc.). Affected closers get `projected_sv=0` → low job security alert | Task 2: normalize names via `_normalize_pitcher_name` (canonical helper per CLAUDE.md) before match |
| BUG-021 | `src/playoff_sim.py:164` (`season_weeks = 22.0`) + earlier `_PLAYOFF_SPOTS = 6` | Hardcoded constants contradict CLAUDE.md (26-week season + 4 playoff spots in FourzynBurn). Counting projections off by ~18%; playoff_pct includes top-6 instead of top-4 | Task 3: parameterize from `LeagueConfig` or accept as function args |
| BUG-023 | `_try_reconnect_yahoo` in `app.py:183-225` only | Headless callers (CI cron, ops scripts, calibrate_constants.py) can't reconnect — every refresh outside Streamlit fails | Task 4: move `_try_reconnect_yahoo` from `app.py` into `src/yahoo_api.py` (or new `src/yahoo_reconnect.py`); have `app.py` re-export for backward compat |
| BUG-025 | `.streamlit/config.toml:10-11` | `enableXsrfProtection=false` and `enableCORS=false` disable CSRF defense; exploitable if Streamlit exposed beyond localhost | Task 5: set both to `true`; document the localhost-only assumption if needed |

---

## File Structure

**Files to create:**
- `tests/test_compute_fa_comparisons_negative_sgp.py` — Task 1
- `tests/test_closer_monitor_name_normalization.py` — Task 2
- `tests/test_playoff_sim_uses_league_config.py` — Task 3
- `tests/test_yahoo_reconnect_in_yahoo_api.py` — Task 4
- `tests/test_streamlit_security_settings.py` — Task 5

**Files to modify:**
- `src/trade_intelligence.py:543-547` — Task 1
- `src/closer_monitor.py:211` — Task 2
- `src/playoff_sim.py:164` (`season_weeks`) + the earlier `_PLAYOFF_SPOTS` constant — Task 3
- `src/yahoo_api.py` — Task 4 (add `_try_reconnect_yahoo` or `try_reconnect_yahoo`)
- `app.py` — Task 4 (re-export from yahoo_api or replace local def with import)
- `.streamlit/config.toml:10-11` — Task 5
- `CLAUDE.md` — Task 6 (history + structural-invariants)

---

## Phase 0: Pre-Flight (2 min)

- [ ] **Step 0.1: Verify worktree**

  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave5-plan"
  git status
  git log -1 --oneline
  ```
  Expected: clean, HEAD at `2cd2c1d Merge pull request #16`.

⚠️ All tasks: use `pytest -q 2>&1 | tail -10`.

---

## Task 1: BUG-017 — `compute_fa_comparisons` picks wrong FA for negative-SGP players

**Insight:** `src/trade_intelligence.py:538-547`:
```python
best_fa_name = ""
best_fa_value = 0.0
for pos in positions:
    pos = pos.strip()
    candidates = fa_by_pos.get(pos, [])
    if candidates:
        name, val = candidates[0]
        if val > best_fa_value:
            best_fa_name = name
            best_fa_value = val
```
- `candidates` is pre-sorted descending by SGP at each position.
- `candidates[0]` is the highest-SGP candidate at that position.
- BUG: for negative-SGP players (e.g., bad relievers), the highest-SGP FA may STILL be negative. Initialization `best_fa_value=0.0` + override `if val > 0` means no negative-SGP FA ever wins → `best_fa_value` stays 0 → downstream `fa_pct = 0 / target_sgp = 0` → `has_alternative=False` even when alternatives exist.

**Files:**
- Modify: `src/trade_intelligence.py:538-547`
- Create: `tests/test_compute_fa_comparisons_negative_sgp.py`

- [ ] **Step 1.1: Write the failing test**

  Create `tests/test_compute_fa_comparisons_negative_sgp.py`:
  ```python
  """BUG-017 fix: compute_fa_comparisons handles negative-SGP players."""

  import pandas as pd


  def test_compute_fa_comparisons_finds_alternative_for_negative_sgp():
      """Bad reliever on opponent's roster (negative SGP) should still find
      the best available FA at their position — not default to fa_value=0."""
      from src.trade_intelligence import compute_fa_comparisons
      from src.valuation import LeagueConfig

      # Build a tiny pool: opponent has a bad reliever (negative SGP), and there
      # are 2 FAs at the same position — also negative-SGP but less negative.
      cols_hitter_baseline = {
          "r": 0, "hr": 0, "rbi": 0, "sb": 0, "ab": 0, "h": 0, "bb": 0, "hbp": 0, "sf": 0,
          "avg": 0, "obp": 0, "pa": 0,
      }
      pool = pd.DataFrame([
          # Opponent's bad reliever (pid 1, negative SGP)
          {"player_id": 1, "name": "BadReliever", "player_name": "BadReliever",
           "positions": "RP", "is_hitter": 0, **cols_hitter_baseline,
           "w": 0, "l": 8, "sv": 2, "k": 30, "ip": 50, "era": 6.0, "whip": 1.8,
           "er": 33, "bb_allowed": 30, "h_allowed": 60},
          # FA #1 — also bad (negative SGP)
          {"player_id": 100, "name": "BadFA1", "player_name": "BadFA1",
           "positions": "RP", "is_hitter": 0, **cols_hitter_baseline,
           "w": 0, "l": 6, "sv": 1, "k": 25, "ip": 40, "era": 5.5, "whip": 1.7,
           "er": 24, "bb_allowed": 22, "h_allowed": 46},
          # FA #2 — better but still negative
          {"player_id": 101, "name": "OkFA", "player_name": "OkFA",
           "positions": "RP", "is_hitter": 0, **cols_hitter_baseline,
           "w": 1, "l": 4, "sv": 5, "k": 35, "ip": 45, "era": 4.2, "whip": 1.3,
           "er": 21, "bb_allowed": 15, "h_allowed": 43},
      ])
      fa_pool = pool[pool["player_id"] != 1].copy()
      result = compute_fa_comparisons(
          opponent_player_ids=[1],
          player_pool=pool,
          fa_pool=fa_pool,
          config=LeagueConfig(),
      )
      assert 1 in result, f"Missing pid=1 in result: {result}"
      r = result[1]
      # The bad reliever's "best FA" should be one of the available FAs,
      # NOT empty/zero. Specifically the best one (OkFA, pid 101) should win
      # because it has the highest SGP among the FAs at RP (least negative).
      assert r["fa_name"] != "", (
          f"BUG-017: best FA name empty for negative-SGP player. Got: {r}"
      )
      # fa_value should NOT be 0 — it should be the OkFA's actual SGP (negative)
      assert abs(r["fa_value"]) > 0.01, (
          f"BUG-017: best FA value is 0.0, indicating no alternative was found. "
          f"Got: {r}"
      )
  ```

  NOTE: This test depends on `compute_fa_comparisons`'s actual signature. Read the function signature in `src/trade_intelligence.py` and adapt the call (parameter names, order). The key assertions are about `fa_name` and `fa_value`, not the exact call shape.

- [ ] **Step 1.2: Run test (should fail)**

  ```bash
  python -m pytest tests/test_compute_fa_comparisons_negative_sgp.py -q 2>&1 | tail -10
  ```
  Expected: FAIL — `fa_name` is empty AND `fa_value` is 0.0.

- [ ] **Step 1.3: Fix the comparison loop**

  In `src/trade_intelligence.py`, find the loop around line 538-547. Replace:
  ```python
          best_fa_name = ""
          best_fa_value = 0.0
          for pos in positions:
              pos = pos.strip()
              candidates = fa_by_pos.get(pos, [])
              if candidates:
                  name, val = candidates[0]
                  if val > best_fa_value:
                      best_fa_name = name
                      best_fa_value = val
  ```

  With:
  ```python
          best_fa_name = ""
          best_fa_value: float | None = None  # None = no seed yet
          for pos in positions:
              pos = pos.strip()
              candidates = fa_by_pos.get(pos, [])
              if candidates:
                  name, val = candidates[0]
                  # BUG-017 fix: seed unconditionally on first candidate so
                  # negative-SGP players still get an alternative; subsequent
                  # positions only override if their FA has a higher SGP.
                  if best_fa_value is None or val > best_fa_value:
                      best_fa_name = name
                      best_fa_value = val
          if best_fa_value is None:
              best_fa_value = 0.0
  ```

  This way: the first candidate (regardless of sign) seeds `best_fa_value`, then subsequent positions override only if higher.

- [ ] **Step 1.4: Run test (should pass)**

  ```bash
  python -m pytest tests/test_compute_fa_comparisons_negative_sgp.py -q 2>&1 | tail -10
  ```
  Expected: `1 passed`.

- [ ] **Step 1.5: Targeted regression sweep**

  ```bash
  python -m pytest tests/test_trade_intelligence*.py tests/test_compute_fa* -q 2>&1 | tail -10
  ```

- [ ] **Step 1.6: Lint + commit**

  ```bash
  python -m ruff format src/trade_intelligence.py tests/test_compute_fa_comparisons_negative_sgp.py
  python -m ruff check src/trade_intelligence.py tests/test_compute_fa_comparisons_negative_sgp.py
  git add src/trade_intelligence.py tests/test_compute_fa_comparisons_negative_sgp.py
  git commit -m "$(cat <<'EOF'
  fix(trade): compute_fa_comparisons finds alternative for negative-SGP players (BUG-017)

  The function initialized `best_fa_value = 0.0` then only overrode on
  `val > best_fa_value`. For players with negative SGP (e.g., bad
  relievers — high ERA/WHIP/L outweighs K/SV in totals_sgp), the
  highest-SGP FA candidate at their position may also be negative. None
  of those negative FAs ever beat the initial 0.0 → best_fa_value
  stayed at 0 → downstream fa_pct = 0/target_sgp = 0 → has_alternative=
  False, even when alternatives existed.

  Fix: seed best_fa_value from the first candidate (None sentinel),
  then compare on subsequent positions. The least-negative FA wins for
  multi-position players, which is the desired "best alternative"
  semantic for the negative-SGP case.

  Adds tests/test_compute_fa_comparisons_negative_sgp.py.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-017)
  EOF
  )"
  ```

---

## Task 2: BUG-020 — Closer monitor name-only exact match fails on accent/punctuation

**Insight:** `src/closer_monitor.py:211`:
```python
match = player_pool[(player_pool["name"] == closer_name) & (player_pool["team"] == team)]
```
Exact string equality. Fails for "José Suárez" vs "Jose Suarez", "Andrés Muñoz" vs "Andres Munoz", suffixes like "Jr." Affected closer rows get `projected_sv=0, era=0, whip=0`, computed `job_security` collapses to `0.6 * hierarchy + 0.0 * sv_component`, demoting healthy closers.

CLAUDE.md mentions a canonical `_normalize_pitcher_name` helper used for probable-starter matching. Use the same helper here.

**Files:**
- Modify: `src/closer_monitor.py:211`
- Create: `tests/test_closer_monitor_name_normalization.py`

- [ ] **Step 2.1: Locate `_normalize_pitcher_name`**

  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave5-plan"
  grep -rn "def _normalize_pitcher_name\|_normalize_pitcher_name" src/ | head -10
  ```

  Confirm where it lives. Likely `src/optimizer/daily_optimizer.py` or `src/optimizer/streaming.py`. Reuse it; don't reimplement.

- [ ] **Step 2.2: Write the failing test**

  Create `tests/test_closer_monitor_name_normalization.py`:
  ```python
  """BUG-020 fix: closer_monitor matches names robustly via _normalize_pitcher_name."""

  import pandas as pd


  def test_closer_with_accented_name_matched():
      """A closer in depth_data with no accent should still match the player_pool
      row that has accents (or vice versa)."""
      from src.closer_monitor import build_closer_grid

      depth_data = {
          "NYM": {"closer": "Edwin Diaz", "setup": [], "closer_confidence": 0.95},
      }
      pool = pd.DataFrame([
          {"player_id": 1, "name": "Edwin Díaz", "team": "NYM", "sv": 30,
           "era": 1.50, "whip": 0.90, "mlb_id": 621242},
      ])
      grid = build_closer_grid(depth_data, player_pool=pool)
      assert len(grid) == 1, f"Expected 1 team in grid; got {grid}"
      row = grid[0]
      assert row["closer_name"] == "Edwin Diaz"
      # The match should have populated SV/ERA/WHIP from the pool row
      assert row["projected_sv"] == 30, (
          f"BUG-020: projected_sv = {row['projected_sv']} indicates name match "
          f"failed for 'Edwin Diaz' (depth_data) vs 'Edwin Díaz' (pool, accented)"
      )
      assert abs(row["era"] - 1.50) < 0.01
      assert abs(row["whip"] - 0.90) < 0.01


  def test_closer_with_suffix_matched():
      """Suffix mismatch ('Jr.' / 'II') should not break the match."""
      from src.closer_monitor import build_closer_grid

      depth_data = {
          "TBR": {"closer": "Pete Fairbanks", "setup": [], "closer_confidence": 0.85},
      }
      pool = pd.DataFrame([
          # Different suffix form in pool (unlikely for Fairbanks but stress-test)
          {"player_id": 1, "name": "Pete Fairbanks Jr.", "team": "TBR", "sv": 25,
           "era": 2.00, "whip": 1.00, "mlb_id": 668941},
      ])
      grid = build_closer_grid(depth_data, player_pool=pool)
      row = grid[0]
      assert row["projected_sv"] == 25, (
          f"BUG-020: projected_sv = {row['projected_sv']} indicates suffix "
          f"variation broke the match"
      )
  ```

- [ ] **Step 2.3: Run test (should fail)**

  ```bash
  python -m pytest tests/test_closer_monitor_name_normalization.py -q 2>&1 | tail -10
  ```
  Expected: FAIL (projected_sv = 0).

- [ ] **Step 2.4: Fix the match**

  In `src/closer_monitor.py`, find line 211. Current:
  ```python
          mlb_id = None
          if player_pool is not None and not player_pool.empty:
              match = player_pool[(player_pool["name"] == closer_name) & (player_pool["team"] == team)]
  ```

  Replace with:
  ```python
          mlb_id = None
          if player_pool is not None and not player_pool.empty:
              # BUG-020: normalize names before matching to handle accents,
              # punctuation, and suffix variations (José vs Jose, Jr., II).
              # Use the canonical _normalize_pitcher_name helper.
              from src.optimizer.daily_optimizer import _normalize_pitcher_name
              norm_closer = _normalize_pitcher_name(closer_name)
              # Pre-normalize the pool's name column for vectorized comparison
              pool_names_norm = player_pool["name"].apply(_normalize_pitcher_name)
              match = player_pool[
                  (pool_names_norm == norm_closer)
                  & (player_pool["team"] == team)
              ]
  ```

  ⚠️ If `_normalize_pitcher_name` lives elsewhere (per Step 2.1's grep), adjust the import path. If it doesn't exist as a standalone helper, define a local `_normalize_name` in `closer_monitor.py` that strips accents (`unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()`), suffixes (`Jr.`, `II`, etc.), and punctuation.

- [ ] **Step 2.5: Run test (should pass)**

  ```bash
  python -m pytest tests/test_closer_monitor_name_normalization.py -q 2>&1 | tail -10
  ```
  Expected: `2 passed`.

- [ ] **Step 2.6: Lint + commit**

  ```bash
  python -m ruff format src/closer_monitor.py tests/test_closer_monitor_name_normalization.py
  python -m ruff check src/closer_monitor.py tests/test_closer_monitor_name_normalization.py
  git add src/closer_monitor.py tests/test_closer_monitor_name_normalization.py
  git commit -m "$(cat <<'EOF'
  fix(closer_monitor): normalize names for closer-to-pool match (BUG-020)

  src/closer_monitor.py used exact string equality on `name + team` to
  match a closer from depth_data to player_pool. Fails on accent/
  punctuation/suffix mismatches (José Suárez, Andrés Muñoz, etc.).
  Affected closer rows got projected_sv=0/era=0/whip=0 and a deflated
  job_security score, demoting healthy closers in the UI grid.

  Fix: normalize both names via the canonical _normalize_pitcher_name
  helper (the same one used for probable-starter matching elsewhere) and
  match on the normalized form.

  Adds tests/test_closer_monitor_name_normalization.py.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-020)
  EOF
  )"
  ```

---

## Task 3: BUG-021 — `playoff_sim.py` hardcoded season_weeks + _PLAYOFF_SPOTS

**Insight:** `src/playoff_sim.py:164` has `season_weeks = 22.0`. The earlier `_PLAYOFF_SPOTS = 6` (find via grep) also contradicts CLAUDE.md (FourzynBurn: 26-week season, top-4 playoff). Counting projections divided by 22 mid-season are wrong; playoff_count summing top-6 over-counts.

**Files:**
- Modify: `src/playoff_sim.py` (constants)
- Create: `tests/test_playoff_sim_uses_league_config.py`

- [ ] **Step 3.1: Locate both constants**

  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave5-plan"
  grep -n "season_weeks\|_PLAYOFF_SPOTS\|playoff_spots\|22\.0\|= 6\b" src/playoff_sim.py | head -30
  ```

- [ ] **Step 3.2: Write the failing test**

  Create `tests/test_playoff_sim_uses_league_config.py`:
  ```python
  """BUG-021 fix: playoff_sim uses canonical season_weeks=26 and playoff_spots=4."""

  import re
  from pathlib import Path

  REPO_ROOT = Path(__file__).resolve().parents[1]
  PSIM = REPO_ROOT / "src" / "playoff_sim.py"


  def test_no_hardcoded_22_season_weeks():
      """season_weeks should be 26 (CLAUDE.md canonical for 26-week MLB season),
      not 22. If sourced from LeagueConfig it should not appear as a literal here."""
      assert PSIM.exists()
      text = PSIM.read_text(encoding="utf-8")
      bad: list[tuple[int, str]] = []
      pat = re.compile(r"\bseason_weeks\s*=\s*([0-9.]+)\b")
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
          f"BUG-021 regression: season_weeks should be 26 (CLAUDE.md canonical), "
          f"not other value. Offenders: {bad}"
      )


  def test_no_hardcoded_6_playoff_spots():
      """_PLAYOFF_SPOTS should be 4 (FourzynBurn league: top-4 playoff per
      CLAUDE.md), not 6."""
      assert PSIM.exists()
      text = PSIM.read_text(encoding="utf-8")
      bad: list[tuple[int, str]] = []
      pat = re.compile(r"\b_PLAYOFF_SPOTS\s*=\s*([0-9]+)\b")
      for lineno, line in enumerate(text.splitlines(), start=1):
          stripped = line.strip()
          if stripped.startswith("#"):
              continue
          m = pat.search(line)
          if m:
              val = int(m.group(1))
              if val != 4:
                  bad.append((lineno, stripped))
      assert not bad, (
          f"BUG-021 regression: _PLAYOFF_SPOTS should be 4 (FourzynBurn top-4 "
          f"playoff per CLAUDE.md). Offenders: {bad}"
      )
  ```

- [ ] **Step 3.3: Run test (should fail)**

  ```bash
  python -m pytest tests/test_playoff_sim_uses_league_config.py -q 2>&1 | tail -10
  ```

- [ ] **Step 3.4: Fix the constants**

  In `src/playoff_sim.py`:
  - Change `season_weeks = 22.0` (line 164) → `season_weeks = 26.0`. Update the comment if it references 22.
  - Find `_PLAYOFF_SPOTS = 6` (use Step 3.1's grep result for the exact line). Change to `_PLAYOFF_SPOTS = 4`.

  Add a comment near each:
  ```python
  # BUG-021: was 22.0; canonical per CLAUDE.md (26-week MLB regular season)
  season_weeks = 26.0
  ```
  ```python
  # BUG-021: was 6; FourzynBurn league has top-4 playoff per CLAUDE.md
  _PLAYOFF_SPOTS = 4
  ```

- [ ] **Step 3.5: Run test + targeted regression sweep**

  ```bash
  python -m pytest tests/test_playoff_sim_uses_league_config.py tests/test_playoff_sim*.py -q 2>&1 | tail -10
  ```

  ⚠️ If existing playoff_sim tests fail because they were calibrated against the old 22/6 values, update the test fixtures' expected outputs to match the new 26/4. Don't break passing tests by re-running old expected values against new constants.

- [ ] **Step 3.6: Lint + commit**

  ```bash
  python -m ruff format src/playoff_sim.py tests/test_playoff_sim_uses_league_config.py
  python -m ruff check src/playoff_sim.py tests/test_playoff_sim_uses_league_config.py
  git add src/playoff_sim.py tests/test_playoff_sim_uses_league_config.py
  git commit -m "$(cat <<'EOF'
  fix(playoff_sim): correct season_weeks=26 and _PLAYOFF_SPOTS=4 (BUG-021)

  src/playoff_sim.py had hardcoded constants contradicting CLAUDE.md:
  - season_weeks = 22.0 (canonical: 26-week MLB regular season)
  - _PLAYOFF_SPOTS = 6 (FourzynBurn: top-4 playoff)

  Counting-stat weekly projections were off by ~18% and playoff_pct
  summed top-6 ranks instead of top-4.

  Adds tests/test_playoff_sim_uses_league_config.py as a structural
  guard against either constant drifting back.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-021)
  EOF
  )"
  ```

---

## Task 4: BUG-023 — Move `_try_reconnect_yahoo` out of `app.py`

**Insight:** `_try_reconnect_yahoo` at `app.py:183-225` is the only path that re-establishes Yahoo OAuth from a stored token. Non-Streamlit callers (CI cron `.github/workflows/refresh.yml`, ops scripts, `calibrate_constants.py`) cannot reconnect → Yahoo phases stale 8+ days outside Streamlit per the audit's INFRA-F7 finding.

**Files:**
- Modify: `src/yahoo_api.py` (add module-level `try_reconnect_yahoo()` function)
- Modify: `app.py` (import + use the new function, or keep a thin wrapper for backward compat)
- Create: `tests/test_yahoo_reconnect_in_yahoo_api.py`

- [ ] **Step 4.1: Read `_try_reconnect_yahoo` in `app.py`**

  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave5-plan"
  sed -n '180,230p' app.py
  ```

  Capture the function body verbatim (you'll move it as-is, with minor edits if needed to remove Streamlit-specific code like `st.error` calls).

- [ ] **Step 4.2: Write the failing test**

  Create `tests/test_yahoo_reconnect_in_yahoo_api.py`:
  ```python
  """BUG-023 fix: try_reconnect_yahoo lives in src/yahoo_api.py for headless callers."""

  import inspect


  def test_try_reconnect_yahoo_callable_from_yahoo_api():
      """The reconnect helper must be importable from src.yahoo_api so
      non-Streamlit callers (CI, ops scripts) can also reconnect."""
      from src.yahoo_api import try_reconnect_yahoo

      # Sanity: it's a callable that returns YahooFantasyClient or None
      assert callable(try_reconnect_yahoo)
      sig = inspect.signature(try_reconnect_yahoo)
      # No required positional args (it reads env + token file)
      required = [
          p for p in sig.parameters.values()
          if p.default is inspect.Parameter.empty
          and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
      ]
      assert not required, (
          f"BUG-023: try_reconnect_yahoo should not require positional args "
          f"(it reads env + token). Required: {required}"
      )


  def test_app_imports_or_uses_reconnect_helper():
      """app.py should either import try_reconnect_yahoo from src.yahoo_api
      or define a thin wrapper that calls it. Plain `def _try_reconnect_yahoo`
      in app.py with its own duplicated logic is the BUG-023 state."""
      from pathlib import Path

      repo = Path(__file__).resolve().parents[1]
      app_text = (repo / "app.py").read_text(encoding="utf-8")
      # Either app.py imports try_reconnect_yahoo, OR if it still defines
      # _try_reconnect_yahoo, the body must be ≤5 lines (i.e., a thin wrapper).
      if "from src.yahoo_api import" in app_text and "try_reconnect_yahoo" in app_text:
          # Direct import — preferred
          return
      # Otherwise verify the local function is thin (delegates to yahoo_api)
      import re
      m = re.search(
          r"def _try_reconnect_yahoo[^\n]*\n((?:[ \t][^\n]*\n)+)",
          app_text,
      )
      if m:
          body = m.group(1)
          # Count non-blank, non-comment body lines
          lines = [
              ln for ln in body.splitlines()
              if ln.strip() and not ln.strip().startswith("#")
          ]
          assert len(lines) <= 5, (
              f"BUG-023 regression: app.py defines _try_reconnect_yahoo with "
              f"{len(lines)} non-trivial body lines; should be a thin wrapper "
              f"(≤5 lines) that calls src.yahoo_api.try_reconnect_yahoo "
              f"so headless callers can reuse the logic."
          )
  ```

- [ ] **Step 4.3: Run test (should fail)**

  ```bash
  python -m pytest tests/test_yahoo_reconnect_in_yahoo_api.py -q 2>&1 | tail -10
  ```
  Expected: FAIL (function not yet in yahoo_api, app.py's _try_reconnect_yahoo is fat).

- [ ] **Step 4.4: Move the function to `src/yahoo_api.py`**

  (a) Open `src/yahoo_api.py`. At the END of the file (after any existing functions/classes), add:
  ```python
  def try_reconnect_yahoo() -> "YahooFantasyClient | None":
      """Headless Yahoo OAuth reconnect.

      Reads YAHOO_LEAGUE_ID from env and the cached OAuth token from
      data/yahoo_token.json. Returns a connected YahooFantasyClient or None.

      This function is callable from CI cron, ops scripts, and
      calibrate_constants.py — anywhere outside Streamlit. The Streamlit
      app.py keeps a thin wrapper for backward compat. (BUG-023 fix.)
      """
      import json
      import os
      from pathlib import Path

      yahoo_league_id = os.environ.get("YAHOO_LEAGUE_ID", "").strip()
      if not yahoo_league_id:
          return None

      try:
          client = YahooFantasyClient(league_id=yahoo_league_id)
      except Exception:
          return None

      # ... PASTE the rest of the original _try_reconnect_yahoo body here,
      # adapted to remove Streamlit st.* calls. Use logger.info/logger.warning
      # for status, NOT st.error/st.warning.
      # Refer to the original function at app.py:183-225 for the full token-
      # loading + authentication flow.

      if client.is_connected() if hasattr(client, "is_connected") else False:
          return client
      return None
  ```

  ⚠️ Copy the FULL original logic from `app.py:_try_reconnect_yahoo`. Replace any `st.error(...)` / `st.warning(...)` with `logger.error(...)` / `logger.warning(...)`. Replace `st.session_state[...]` reads with env-var fallbacks (the function should be fully self-contained).

  ⚠️ Make sure the `YahooFantasyClient.is_connected` or `is_authenticated` check uses the right attribute. Per Wave 3 Task 4's finding, the actual attribute is `is_authenticated` (a property), not `is_connected()`. Verify by reading `src/yahoo_api.py` to see what the class exposes.

  (b) In `app.py`, replace the original `_try_reconnect_yahoo` definition with a thin wrapper:
  ```python
  def _try_reconnect_yahoo() -> "YahooFantasyClient | None":
      """Streamlit-side wrapper: delegates to src.yahoo_api.try_reconnect_yahoo.
      Kept for backward compat with existing call sites in app.py."""
      from src.yahoo_api import try_reconnect_yahoo
      return try_reconnect_yahoo()
  ```

  This preserves all `app.py:_try_reconnect_yahoo()` call sites while routing through the headless implementation.

- [ ] **Step 4.5: Run test (should pass)**

  ```bash
  python -m pytest tests/test_yahoo_reconnect_in_yahoo_api.py -q 2>&1 | tail -10
  ```

- [ ] **Step 4.6: Targeted regression sweep**

  ```bash
  python -m pytest tests/test_yahoo_api*.py tests/test_yahoo*.py -q 2>&1 | tail -10
  ```

- [ ] **Step 4.7: Lint + commit**

  ```bash
  python -m ruff format src/yahoo_api.py app.py tests/test_yahoo_reconnect_in_yahoo_api.py
  python -m ruff check src/yahoo_api.py app.py tests/test_yahoo_reconnect_in_yahoo_api.py
  git add src/yahoo_api.py app.py tests/test_yahoo_reconnect_in_yahoo_api.py
  git commit -m "$(cat <<'EOF'
  fix(yahoo_api): move _try_reconnect_yahoo into yahoo_api for headless callers (BUG-023)

  _try_reconnect_yahoo lived only in app.py, so non-Streamlit callers
  (CI cron in .github/workflows/refresh.yml, ops scripts,
  calibrate_constants.py) couldn't reconnect to Yahoo. Result: Yahoo
  phases stale 8+ days outside Streamlit (per audit INFRA-F7).

  Fix: moved the function to src/yahoo_api.try_reconnect_yahoo (public,
  st.* calls replaced with logger.*). app.py keeps a thin one-line
  wrapper for backward compat with its existing call sites.

  Adds tests/test_yahoo_reconnect_in_yahoo_api.py.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-023)
  EOF
  )"
  ```

---

## Task 5: BUG-025 — Streamlit XSRF / CORS

**Insight:** `.streamlit/config.toml:10-11` has `enableXsrfProtection=false` and `enableCORS=false`. CLAUDE.md confirms app is "in-season production-active". If exposed beyond localhost, both disables open CSRF attack surface (especially on Yahoo OAuth form submits).

**Files:**
- Modify: `.streamlit/config.toml:10-11`
- Create: `tests/test_streamlit_security_settings.py`

- [ ] **Step 5.1: Write the failing test**

  Create `tests/test_streamlit_security_settings.py`:
  ```python
  """BUG-025 fix: Streamlit XSRF + CORS protections enabled."""

  import re
  from pathlib import Path

  CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


  def test_xsrf_protection_enabled():
      """enableXsrfProtection should be true; false leaves the app vulnerable
      to CSRF (especially Yahoo OAuth form submits)."""
      assert CONFIG.exists()
      text = CONFIG.read_text(encoding="utf-8")
      m = re.search(r"enableXsrfProtection\s*=\s*(true|false)", text, re.IGNORECASE)
      assert m, "enableXsrfProtection setting missing from .streamlit/config.toml"
      assert m.group(1).lower() == "true", (
          f"BUG-025 regression: enableXsrfProtection = {m.group(1)}. "
          "Must be true; false exposes the app to CSRF when hosted beyond localhost."
      )


  def test_cors_enabled():
      """enableCORS should be true (default behavior); false in combination
      with disabled XSRF is the BUG-025 high-severity config."""
      assert CONFIG.exists()
      text = CONFIG.read_text(encoding="utf-8")
      m = re.search(r"enableCORS\s*=\s*(true|false)", text, re.IGNORECASE)
      # CORS may simply not be set (default true) — that's also acceptable
      if m is None:
          return  # default is true; no setting needed
      assert m.group(1).lower() == "true", (
          f"BUG-025 regression: enableCORS = {m.group(1)}. "
          "Set to true or remove the line (default is true)."
      )
  ```

- [ ] **Step 5.2: Run test (should fail)**

  ```bash
  python -m pytest tests/test_streamlit_security_settings.py -q 2>&1 | tail -10
  ```

- [ ] **Step 5.3: Fix the config**

  In `.streamlit/config.toml`:
  ```toml
  # OLD:
  [server]
  headless = true
  enableXsrfProtection = false
  enableCORS = false

  # NEW:
  [server]
  headless = true
  # BUG-025: XSRF + CORS protections enabled. If a specific local-dev
  # behavior requires either disabled, override via STREAMLIT_SERVER_*
  # env var instead of disabling in committed config.
  enableXsrfProtection = true
  enableCORS = true
  ```

- [ ] **Step 5.4: Verify**

  ```bash
  python -m pytest tests/test_streamlit_security_settings.py -q 2>&1 | tail -10
  ```

- [ ] **Step 5.5: Commit**

  ```bash
  git add .streamlit/config.toml tests/test_streamlit_security_settings.py
  git commit -m "$(cat <<'EOF'
  fix(streamlit): enable XSRF + CORS protections (BUG-025)

  .streamlit/config.toml had enableXsrfProtection=false and
  enableCORS=false. CLAUDE.md describes the app as "in-season
  production-active"; if hosted beyond localhost, the disabled CSRF
  defense is exploitable (especially via Yahoo OAuth form submits).

  Sets both to true. Local-dev quirks that require one of them
  disabled should override via STREAMLIT_SERVER_* env vars at run
  time, not in committed config.

  Adds tests/test_streamlit_security_settings.py.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-025)
  EOF
  )"
  ```

---

## Task 6: CLAUDE.md update

- [ ] **Step 6.1: Run all Wave 5 + existing structural-invariant tests**

  ```bash
  python -m pytest tests/test_compute_fa_comparisons_negative_sgp.py tests/test_closer_monitor_name_normalization.py tests/test_playoff_sim_uses_league_config.py tests/test_yahoo_reconnect_in_yahoo_api.py tests/test_streamlit_security_settings.py tests/test_no_*.py tests/test_pages_*.py tests/test_engine_*.py tests/test_opponent_*.py tests/test_pool_*.py tests/test_standings_*.py tests/test_pulp_*.py tests/test_refresh_*.py -q 2>&1 | tail -10
  ```

- [ ] **Step 6.2: Update CLAUDE.md history**

  After the Wave 4 paragraph, append:
  ```markdown
  **2026-05-11 Wave 5 (SF-47 → SF-51)** — Miscellaneous high-impact fixes. SF-47 `compute_fa_comparisons` defaulted `best_fa_value=0.0` and only overrode on `val > 0.0` → negative-SGP players (bad relievers) got no FA alternative. SF-48 closer_monitor used exact `name + team` equality → José/Jose accent mismatches deflated job_security; now normalizes via `_normalize_pitcher_name`. SF-49 `playoff_sim.season_weeks=22` and `_PLAYOFF_SPOTS=6` contradicted CLAUDE.md (26-week season, top-4 playoff per FourzynBurn) → fixed. SF-50 `_try_reconnect_yahoo` lived in `app.py` only → headless callers couldn't reconnect (Yahoo phases stale 8+ days outside Streamlit per INFRA-F7). Moved to `src/yahoo_api.try_reconnect_yahoo`. SF-51 `.streamlit/config.toml` had XSRF + CORS disabled → CSRF attack surface; set both to true.
  ```

- [ ] **Step 6.3: Update Structural Invariants table**

  Add five rows:
  ```markdown
  | `test_compute_fa_comparisons_negative_sgp.py` | `compute_fa_comparisons` seeds best FA from first candidate (handles negative-SGP) |
  | `test_closer_monitor_name_normalization.py` | `closer_monitor.build_closer_grid` normalizes names before matching pool |
  | `test_playoff_sim_uses_league_config.py` | `playoff_sim` uses `season_weeks=26` and `_PLAYOFF_SPOTS=4` |
  | `test_yahoo_reconnect_in_yahoo_api.py` | `try_reconnect_yahoo` lives in `src/yahoo_api`; `app.py` is a thin wrapper |
  | `test_streamlit_security_settings.py` | `.streamlit/config.toml` has `enableXsrfProtection=true` and `enableCORS=true` |
  ```

- [ ] **Step 6.4: Commit**

  ```bash
  git add CLAUDE.md
  git commit -m "docs(audit): CLAUDE.md history for Wave 5 (SF-47..SF-51)"
  ```

---

## Phase Final

- [ ] **Step F.1: Push + PR**
  ```bash
  git push -u origin claude/audit-wave5-plan
  gh pr create --title "Wave 5: miscellaneous fixes (BUG-017/020/021/023/025)" --body "..."
  ```

---

## Self-Review

**Spec coverage:** BUG-017, BUG-020, BUG-021, BUG-023, BUG-025 each have a task. BUG-010 + BUG-015 explicitly deferred to Wave 6.

**Placeholder scan:** No "TBD"/"TODO"/"implement later".

**Type/identifier consistency:** `try_reconnect_yahoo`, `_normalize_pitcher_name`, `_PLAYOFF_SPOTS`, `season_weeks`, `enableXsrfProtection`, `enableCORS` — referenced consistently.

**Cold-start usability:** Worktree, branch, bug summaries, pre-flight all stated. Tasks independent.
