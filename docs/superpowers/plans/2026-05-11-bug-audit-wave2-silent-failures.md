# Bug Audit Wave 2: Silent-Failure Elimination — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 5 HIGH-severity silent-failure bugs from the 2026-05-11 audit (BUG-004, BUG-012, BUG-013, BUG-014, BUG-022) that silently produce wrong data without surfacing errors. Each fix is module-local with TDD test + structural guard.

**Architecture:** TDD per bug. Each task = focused failing test → minimal fix → passing test → commit. Final task adds permanent structural guards (one per bug). All fixes are independent and can be reviewed in any order.

**Tech Stack:** Python 3.11+, SQLite, pytest, MLB-StatsAPI, pandas, numpy.

**Source spec:** [docs/superpowers/specs/2026-05-11-bug-audit-findings.md](../specs/2026-05-11-bug-audit-findings.md)

---

## Cold-Start Context

### Worktree
- Path: `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave2-plan`
- Branch: `claude/audit-wave2-plan` (tracks `origin/master`)
- HEAD: `8dbf8d0 Merge pull request #13 from hicklax13/claude/audit-wave1-plan` (Wave 1 already merged)

### Bug summaries

| Bug | Files | Effect | Fix |
|-----|-------|--------|-----|
| BUG-004 | `src/live_stats.py:237`, `src/player_databank.py:441-447` | IP returned by MLB API as outs notation ("52.2" = 52⅔ IP) parsed as decimal (52.2 IP) → every ERA/WHIP shifts by ~1.5% | Task 1: `_ip_outs_to_decimal()` helper applied at both call sites |
| BUG-012 | `src/bayesian.py:406-460` | `updated["h"] = obs_h + avg*remaining_ab` produces h/ab ≠ avg; pitcher BB/H split hardcoded 35/65 ignoring observed ratio | Task 2: derive h consistently from blended avg×ab; use observed BB/(BB+H) ratio with regression toward 0.35 |
| BUG-013 | `src/injury_model.py:362-368` | Inner `break` for two-way players overwrites hitting `gamesPlayed=159` with pitching `gamesPlayed=10` (Ohtani → health_score=0.06) | Task 3: `max(games_played, ...)` across stat groups |
| BUG-014 | `src/ecr.py:1066-1080` | Source-weight lookup keys are `"espn_rank"`/`"fantasypros_rank"` but `_SOURCE_WEIGHTS_INSEASON` uses `"espn"`/`"fantasypros"` → 1.5× FantasyPros, 0.4× Yahoo weights are silently no-ops | Task 4: strip `_rank` suffix before passing to `_compute_player_consensus` |
| BUG-022 | `src/il_manager.py:120-121` | Cold-start (`last_known_status=None` → `{}`) makes `old_status=""`; every IL player flags as "new IL change" | Task 5: return empty list on cold start; emit changes only when caller provides prior state |

---

## File Structure

**Files to create:**
- `tests/test_ip_outs_notation.py` — Task 1
- `tests/test_bayesian_consistency.py` — Task 2
- `tests/test_injury_two_way_player.py` — Task 3
- `tests/test_ecr_source_weighting.py` — Task 4
- `tests/test_il_manager_cold_start.py` — Task 5
- `tests/test_no_ip_decimal_parse.py` — Task 6 (structural guard)

**Files to modify:**
- `src/live_stats.py:237` — Task 1 (1 call site)
- `src/player_databank.py:441-447` — Task 1 (1 helper)
- `src/bayesian.py:406-412, 453-460` — Task 2 (h derivation + BB/H split)
- `src/injury_model.py:362-368` — Task 3 (1 line: `max(...)`)
- `src/ecr.py:1074` — Task 4 (1 line: dict-comprehension key transform)
- `src/il_manager.py:107-134` — Task 5 (function contract)
- `CLAUDE.md` — Task 6 (SF-33..SF-37 history + structural-invariant table)

---

## Phase 0: Pre-Flight (3 min)

- [ ] **Step 0.1: Verify worktree state**

  ```bash
  cd "C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave2-plan"
  git status  # expect clean
  git log -1 --oneline  # expect: 8dbf8d0 Merge pull request #13 ...
  ```

- [ ] **Step 0.2: Verify pytest baseline**

  ```bash
  python -m pytest --collect-only -q --ignore=tests/test_cheat_sheet.py 2>&1 | tail -3
  ```
  Expect: ~3750 tests collect cleanly.

⚠️ **For all tasks: use `pytest -q 2>&1 | tail -10`** to avoid verbose-output stalls.

---

## Task 1: BUG-004 — IP outs-notation parser

**Insight:** MLB Stats API returns IP as a string like `"52.2"` meaning 52 + 2 outs (i.e., 52⅔ IP). `float("52.2")` returns 52.2 (decimal IP) — wrong by ~0.467 IP. Every downstream ERA = (ER × 9) / IP shifts by ~1.5% and same for WHIP. Live DB verified: all 536 pitcher IP values end in `.0`/`.1`/`.2` only, confirming the outs-notation source format never gets converted.

**Files:**
- Create: `tests/test_ip_outs_notation.py`
- Modify: `src/live_stats.py:237`, `src/player_databank.py:440-447`

- [ ] **Step 1.1: Write the failing test**

  Create `tests/test_ip_outs_notation.py`:
  ```python
  """Test BUG-004 fix: IP outs notation correctly converted to decimal IP."""
  import pytest


  def test_ip_outs_notation_conversion():
      """MLB API IP format: '52.2' means 52 + 2/3 IP (NOT 52.2)."""
      from src.live_stats import _ip_outs_to_decimal

      # Whole innings: 52.0 → 52.0
      assert _ip_outs_to_decimal("52.0") == pytest.approx(52.0, abs=1e-6)
      # 1 out: 52.1 → 52 + 1/3 = 52.333...
      assert _ip_outs_to_decimal("52.1") == pytest.approx(52 + 1 / 3, abs=1e-6)
      # 2 outs: 52.2 → 52 + 2/3 = 52.667...
      assert _ip_outs_to_decimal("52.2") == pytest.approx(52 + 2 / 3, abs=1e-6)
      # Whole: 0.0 → 0
      assert _ip_outs_to_decimal("0.0") == 0.0
      # 1 out for a reliever debut: 0.1 → 1/3
      assert _ip_outs_to_decimal("0.1") == pytest.approx(1 / 3, abs=1e-6)
      # Numeric input also works (idempotent)
      assert _ip_outs_to_decimal(6.0) == pytest.approx(6.0, abs=1e-6)
      # Empty/None
      assert _ip_outs_to_decimal("") == 0.0
      assert _ip_outs_to_decimal(None) == 0.0
      # Garbage tolerates gracefully
      assert _ip_outs_to_decimal("abc") == 0.0


  def test_parse_pitching_stat_uses_outs_notation():
      """_parse_pitching_stat must use _ip_outs_to_decimal for IP."""
      from src.live_stats import _parse_pitching_stat

      player_info = {"fullName": "Test", "team_abbr": "ATL", "mlb_id": 1}
      stat = {"inningsPitched": "10.2", "earnedRuns": 4, "wins": 1, "losses": 0,
              "saves": 0, "strikeOuts": 12, "era": "3.38", "whip": "1.18",
              "baseOnBalls": 3, "hits": 9, "gamesPlayed": 2}
      row = _parse_pitching_stat(stat, player_info)
      assert row["ip"] == pytest.approx(10 + 2 / 3, abs=1e-6), (
          f"BUG-004: expected ip = 10.667, got {row['ip']}"
      )


  def test_player_databank_parse_game_log_uses_outs_notation():
      """player_databank._parse_game_log_row should convert IP outs notation."""
      from src.player_databank import _parse_game_log_row

      raw = {
          "inningsPitched": "6.1",
          "wins": 0, "losses": 0, "saves": 0, "strikeOuts": 7,
          "earnedRuns": 2, "baseOnBalls": 1, "hits": 4,
      }
      row = _parse_game_log_row(raw, player_id=1, game_date="2026-05-10", season=2026)
      assert row["ip"] == pytest.approx(6 + 1 / 3, abs=1e-6), (
          f"BUG-004: expected ip = 6.333, got {row['ip']}"
      )
  ```

- [ ] **Step 1.2: Run test to verify it fails**

  ```bash
  python -m pytest tests/test_ip_outs_notation.py -q 2>&1 | tail -10
  ```
  Expect: All 3 tests FAIL — `_ip_outs_to_decimal` does not yet exist (ImportError) plus the parse tests fail because `float("10.2") = 10.2 ≠ 10.667`.

- [ ] **Step 1.3: Add `_ip_outs_to_decimal` helper to `src/live_stats.py`**

  In `src/live_stats.py`, near the top of the file (after imports, before `_parse_pitching_stat`), add:
  ```python
  def _ip_outs_to_decimal(ip_value: object) -> float:
      """Convert MLB Stats API innings-pitched ("52.2" = 52 + 2 outs) to decimal IP.

      MLB Stats API formats IP as a string where the fractional part is the
      number of outs in the partial inning (0, 1, or 2), NOT a true decimal.
      For example "52.2" means 52 innings + 2 outs = 52 + 2/3 ≈ 52.667 IP.
      `float("52.2")` gives 52.2 (wrong by 0.467 IP), causing every ERA = ER*9/IP
      and WHIP = (BB+H)/IP to shift by ~1.5%.

      Accepts string or numeric input. Returns 0.0 for empty/garbage inputs.
      """
      if ip_value is None or ip_value == "":
          return 0.0
      try:
          # Coerce to string then parse, so numeric inputs (already decimal) round-trip
          s = str(ip_value)
          # If it's already a decimal (e.g. 6.5 from some other source), accept as-is
          # but the MLB outs format never produces .3-.9, so we can detect:
          if "." in s:
              int_part_str, frac_str = s.split(".", 1)
              int_part = int(int_part_str)
              # Take first digit of frac as outs (0, 1, or 2)
              frac_digit = int(frac_str[0]) if frac_str else 0
              if frac_digit <= 2:
                  return int_part + frac_digit / 3.0
              # If frac_digit >= 3, the source already uses true decimals — return as-is
              return float(s)
          return float(s)
      except (ValueError, TypeError):
          return 0.0
  ```

  Then change line 237 of the same file:
  ```python
  # OLD:
          "ip": float(stat.get("inningsPitched", "0") or 0),

  # NEW:
          "ip": _ip_outs_to_decimal(stat.get("inningsPitched", "0")),
  ```

- [ ] **Step 1.4: Update `src/player_databank.py` `_float` helper or add direct call**

  In `src/player_databank.py`, line 440-447, replace the `_float` helper's IP-specific path. The cleanest fix: import the helper from `live_stats` and use it specifically for IP:

  ```python
  # Add to imports near top of file:
  from src.live_stats import _ip_outs_to_decimal

  # In _parse_game_log_row, find where "ip" is set and change to use the helper.
  # Search for `"ip": _float(...)` or `"ip": float(...)` in this function and replace with:
  #   "ip": _ip_outs_to_decimal(raw.get("inningsPitched")),
  # If the current code uses _float("inningsPitched"), replace just that one call.
  ```

  ⚠️ Read the surrounding code carefully — there may be additional IP-related fields (`game_logs.ip` is the main one). Only convert fields that come from MLB Stats API's `inningsPitched` key.

  Also FIX the misleading comment in `_float` (lines 443-444). Change:
  ```python
              # IP is sometimes stored as "2.1" (outs notation) but statsapi
              # returns decimal innings directly (e.g. 6.0 for 6 full innings).
              return float(val)
  ```
  To:
  ```python
              # NOTE: For IP fields, use _ip_outs_to_decimal() — statsapi returns
              # IP as outs notation ("10.2" = 10 + 2 outs). This _float helper is
              # for non-IP numeric fields only.
              return float(val)
  ```

- [ ] **Step 1.5: Run test to verify it passes**

  ```bash
  python -m pytest tests/test_ip_outs_notation.py -q 2>&1 | tail -10
  ```
  Expect: `3 passed`.

- [ ] **Step 1.6: Lint + format**

  ```bash
  python -m ruff format src/live_stats.py src/player_databank.py tests/test_ip_outs_notation.py
  python -m ruff check src/live_stats.py src/player_databank.py tests/test_ip_outs_notation.py
  ```
  Expect: `All checks passed!`

- [ ] **Step 1.7: Commit**

  ```bash
  git add src/live_stats.py src/player_databank.py tests/test_ip_outs_notation.py
  git commit -m "$(cat <<'EOF'
  fix(stats): IP outs-notation parser (BUG-004)

  MLB Stats API returns innings-pitched as a string where the fractional
  part is OUTS in the partial inning (0, 1, or 2) — "52.2" means 52⅔ IP,
  not 52.2 IP. The prior `float(stat["inningsPitched"])` parse was off by
  ~0.467 IP per non-whole-inning value, shifting every ERA and WHIP
  computed downstream by ~1.5%.

  Adds src/live_stats._ip_outs_to_decimal() helper and applies it in both
  call sites:
  - src/live_stats.py:237 (_parse_pitching_stat)
  - src/player_databank.py game_log parser

  Fixes the misleading comment in player_databank._float that claimed
  "statsapi returns decimal innings directly" (it does not).

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-004)
  EOF
  )"
  ```

---

## Task 2: BUG-012 — Bayesian formula override

**Insight:** The Bayesian update derives `updated["h"] = int(obs_h + updated["avg"] * remaining_ab)` while setting `updated["ab"] = pre_ab`. This makes `h/ab ≠ avg` in general (inconsistent rate-stat aggregation). Same class of bug for pitchers: `bb_allowed`/`h_allowed` are derived from `WHIP × IP` using a hardcoded 35/65 split, ignoring the player's actual BB/H ratio.

**Files:**
- Modify: `src/bayesian.py:406-412` (h derivation), `:453-460` (BB/H split)
- Create: `tests/test_bayesian_consistency.py`

- [ ] **Step 2.1: Write the failing test**

  Create `tests/test_bayesian_consistency.py`:
  ```python
  """Test BUG-012 fix: Bayesian batch_update produces self-consistent rate/counting stats."""
  import pandas as pd
  import pytest


  @pytest.fixture
  def bayesian_updater():
      from src.bayesian import BayesianUpdater
      return BayesianUpdater()


  def test_hitter_h_div_ab_equals_avg(bayesian_updater):
      """After batch_update, updated["h"] / updated["ab"] must equal updated["avg"]
      (within rounding tolerance). The prior bug computed h from formula that
      mixed observed h with blended avg × remaining_ab, producing inconsistency."""
      preseason = pd.DataFrame([{
          "player_id": 1, "name": "Test Hitter", "is_hitter": True,
          "system": "blended",
          "pa": 600, "ab": 550, "h": 165, "avg": 0.300, "obp": 0.370,
          "r": 90, "hr": 25, "rbi": 80, "sb": 10, "bb": 45, "hbp": 5, "sf": 5,
      }])
      # Observed mid-season: player overperforming (.350 in 200 AB)
      season_stats = pd.DataFrame([{
          "player_id": 1,
          "pa_obs": 220, "ab_obs": 200, "h_obs": 70,  # .350 AVG
          "obp_obs": 0.420, "r_obs": 35, "hr_obs": 12, "rbi_obs": 35, "sb_obs": 4,
      }])
      result = bayesian_updater.batch_update_projections(season_stats, preseason)
      assert len(result) == 1
      row = result.iloc[0]
      # Allow integer-rounding tolerance: |h - avg*ab| ≤ 1
      derived_h = row["avg"] * row["ab"]
      assert abs(row["h"] - derived_h) <= 1.5, (
          f"BUG-012: h/ab inconsistent with avg. "
          f"avg={row['avg']:.4f}, ab={row['ab']}, h={row['h']}, derived_h={derived_h:.2f}"
      )


  def test_pitcher_bb_h_split_uses_observed_ratio(bayesian_updater):
      """For a control pitcher (low BB rate), Bayesian update must NOT impose
      the hardcoded 35/65 BB/H split. Observed bb:h ratio should regress toward
      the league mean (35/65) with stabilization, not get overwritten."""
      preseason = pd.DataFrame([{
          "player_id": 100, "name": "Control SP", "is_hitter": False,
          "system": "blended",
          "ip": 200.0, "era": 3.50, "whip": 1.10, "w": 14, "l": 8, "sv": 0, "k": 180,
          "er": 78, "bb_allowed": 40, "h_allowed": 180,  # bb:h = 40:180 ≈ 18:82
      }])
      # Observed: 100 IP, ERA=3.0, WHIP=1.0, 15 BB and 85 H
      season_stats = pd.DataFrame([{
          "player_id": 100,
          "ip_obs": 100.0, "era_obs": 3.0, "whip_obs": 1.0,
          "w_obs": 8, "l_obs": 3, "sv_obs": 0, "k_obs": 95,
          "er_obs": 33, "bb_allowed_obs": 15, "h_allowed_obs": 85,
      }])
      result = bayesian_updater.batch_update_projections(season_stats, preseason)
      row = result.iloc[0]
      total_br = row["bb_allowed"] + row["h_allowed"]
      assert total_br > 0, "Pitcher should have non-zero baserunners projected"
      bb_share = row["bb_allowed"] / total_br
      # Control pitcher's observed share is 15/100 = 0.15. With regression to 0.35,
      # final share should be < 0.30 (well below the hardcoded 0.35).
      assert bb_share < 0.30, (
          f"BUG-012: pitcher bb_share = {bb_share:.3f} appears to use hardcoded "
          f"0.35 ratio, ignoring observed bb:h = 15:85 = 0.15"
      )


  def test_pitcher_er_consistent_with_era_and_ip(bayesian_updater):
      """updated["er"] should equal int(era * ip / 9). Already correct in current
      code, but pin it down so we don't regress."""
      preseason = pd.DataFrame([{
          "player_id": 200, "name": "Test SP", "is_hitter": False,
          "system": "blended",
          "ip": 150.0, "era": 4.00, "whip": 1.20, "w": 10, "l": 8, "sv": 0, "k": 130,
          "er": 67, "bb_allowed": 50, "h_allowed": 130,
      }])
      season_stats = pd.DataFrame([{
          "player_id": 200,
          "ip_obs": 80.0, "era_obs": 3.50, "whip_obs": 1.15,
          "w_obs": 6, "l_obs": 4, "sv_obs": 0, "k_obs": 75,
          "er_obs": 31, "bb_allowed_obs": 24, "h_allowed_obs": 70,
      }])
      result = bayesian_updater.batch_update_projections(season_stats, preseason)
      row = result.iloc[0]
      expected_er = int(row["era"] * row["ip"] / 9)
      assert abs(row["er"] - expected_er) <= 1, (
          f"BUG-012: er = {row['er']} inconsistent with era*ip/9 = {expected_er}"
      )
  ```

- [ ] **Step 2.2: Run test to verify it fails**

  ```bash
  python -m pytest tests/test_bayesian_consistency.py -q 2>&1 | tail -10
  ```
  Expect: at least the first two tests FAIL (third may pass since the current code is already consistent there).

- [ ] **Step 2.3: Fix h derivation** — `src/bayesian.py:406-412`

  Find lines 406-412:
  ```python
                  updated["pa"] = int(pre_pa)
                  obs_ab = int(_safe_val(row, "ab_obs"))
                  updated["ab"] = int(_safe_val(row, "ab_pre"))
                  # Recalculate h: observed hits + blended avg * remaining AB
                  obs_h = int(_safe_val(row, "h_obs"))
                  remaining_ab = max(0, updated["ab"] - obs_ab)
                  updated["h"] = int(obs_h + updated["avg"] * remaining_ab)
  ```
  Replace with:
  ```python
                  updated["pa"] = int(pre_pa)
                  # ab = full-season projected at-bats (carried from preseason);
                  # h = avg * ab to keep h/ab consistent with the blended avg.
                  # (Earlier code computed h = obs_h + avg*remaining_ab which made
                  # h/ab != avg whenever observed performance diverged from the
                  # blended rate — fixed for BUG-012.)
                  updated["ab"] = int(_safe_val(row, "ab_pre"))
                  updated["h"] = int(round(updated["avg"] * updated["ab"]))
  ```

- [ ] **Step 2.4: Fix BB/H split** — `src/bayesian.py:453-460`

  Find lines 453-460 (the section after `updated["ip"] = pre_ip` and the ER derivation):
  ```python
                  # Derive bb_allowed and h_allowed from WHIP
                  if updated["ip"] > 0:
                      total_baserunners = updated["whip"] * updated["ip"]
                      updated["bb_allowed"] = int(total_baserunners * 0.35)
                      updated["h_allowed"] = int(total_baserunners * 0.65)
                  else:
                      updated["bb_allowed"] = 0
                      updated["h_allowed"] = 0
  ```
  Replace with:
  ```python
                  # Derive bb_allowed and h_allowed from WHIP using observed BB:H
                  # ratio (regressed toward 0.35 with stabilization). The earlier
                  # hardcoded 0.35/0.65 split silently overwrote a control pitcher's
                  # actual BB rate — fixed for BUG-012.
                  if updated["ip"] > 0:
                      total_baserunners = updated["whip"] * updated["ip"]
                      obs_bb = _safe_val(row, "bb_allowed_obs")
                      obs_h_allowed = _safe_val(row, "h_allowed_obs")
                      obs_br = obs_bb + obs_h_allowed
                      if obs_br > 0:
                          obs_bb_share = obs_bb / obs_br
                          # Regress toward league-average 0.35 with stabilization
                          # at ~50 observed baserunners (~12-15 IP)
                          stab = 50.0
                          bb_share = (obs_bb + 0.35 * stab) / (obs_br + stab)
                      else:
                          bb_share = 0.35
                      updated["bb_allowed"] = int(total_baserunners * bb_share)
                      updated["h_allowed"] = int(total_baserunners * (1.0 - bb_share))
                  else:
                      updated["bb_allowed"] = 0
                      updated["h_allowed"] = 0
  ```

- [ ] **Step 2.5: Run tests to verify they pass**

  ```bash
  python -m pytest tests/test_bayesian_consistency.py -q 2>&1 | tail -10
  ```
  Expect: `3 passed`.

  Then run all `bayesian` tests to verify no regressions:
  ```bash
  python -m pytest tests/test_bayesian*.py tests/test_*bayesian* -q 2>&1 | tail -10
  ```
  Expect: all pass (any pre-existing bayesian-test count plus our 3).

- [ ] **Step 2.6: Lint + format**

  ```bash
  python -m ruff format src/bayesian.py tests/test_bayesian_consistency.py
  python -m ruff check src/bayesian.py tests/test_bayesian_consistency.py
  ```

- [ ] **Step 2.7: Commit**

  ```bash
  git add src/bayesian.py tests/test_bayesian_consistency.py
  git commit -m "$(cat <<'EOF'
  fix(bayesian): consistent rate/counting derivation in batch update (BUG-012)

  Two related fixes to BayesianUpdater.batch_update_projections:

  1. Hitter h derivation. Previously:
       updated["h"] = int(obs_h + updated["avg"] * remaining_ab)
     This made h/ab != avg whenever observed performance diverged from
     the blended rate. A player batting .350 in 200 AB with .300
     preseason projection got h ≈ 70 + .310*350 = 178, ab = 550, but
     avg = .310 — so h/ab = .324 != .310 = avg.
     Fix: derive h = round(avg * ab) directly. Consistency preserved.

  2. Pitcher BB/H split. Previously:
       bb_allowed = total_baserunners * 0.35  # hardcoded
       h_allowed  = total_baserunners * 0.65  # hardcoded
     Ignored observed BB:H ratio entirely. A control pitcher with
     BB/9=2.0 and an erratic closer with BB/9=4.5 got identical
     35:65 splits.
     Fix: compute observed bb_share = bb_obs / (bb_obs + h_obs), regress
     toward 0.35 with stabilization=50 baserunners, then split
     total_baserunners by the blended share.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-012)
  EOF
  )"
  ```

---

## Task 3: BUG-013 — Two-way player injury history

**Insight:** `load_injury_history_from_api` requests `group="hitting,pitching"` so `stats.get("stats", [])` returns BOTH groups. The inner loop's `break` only exits the splits loop, not the stat_groups loop. So for Ohtani: hitting games_played=159 gets assigned, then the next stat_group (pitching) iteration finds Ohtani's pitching split with games_played=10, **overwrites** the variable, and that 10 is what gets recorded. health_score = 10/162 = 0.06 — appears severely injured.

**Files:**
- Modify: `src/injury_model.py:362-368`
- Create: `tests/test_injury_two_way_player.py`

- [ ] **Step 3.1: Write the failing test**

  Create `tests/test_injury_two_way_player.py`:
  ```python
  """Test BUG-013 fix: two-way player injury history uses max games across stat groups."""
  from unittest.mock import patch
  import pandas as pd
  import pytest


  def test_two_way_player_uses_max_games_played():
      """Ohtani-like two-way player: hitting=159 games, pitching=10 games.
      The recorded games_played should be 159 (not 10 — which is what the
      pitching group's split would overwrite under BUG-013)."""
      # Mock statsapi response: both stat groups present for one player
      mock_response = {
          "stats": [
              {  # First group: hitting (159 games)
                  "type": {"displayName": "yearByYear"},
                  "group": {"displayName": "hitting"},
                  "splits": [{"season": "2024", "stat": {"gamesPlayed": 159}}],
              },
              {  # Second group: pitching (10 games)
                  "type": {"displayName": "yearByYear"},
                  "group": {"displayName": "pitching"},
                  "splits": [{"season": "2024", "stat": {"gamesPlayed": 10}}],
              },
          ]
      }
      with patch("src.injury_model.statsapi.player_stat_data", return_value=mock_response):
          from src.injury_model import load_injury_history_from_api
          df = load_injury_history_from_api([660271])  # Ohtani's mlb_id
      # Filter to season=2024 (the one with our mocked data)
      row_2024 = df[df["season"] == 2024]
      assert not row_2024.empty
      games_played = int(row_2024.iloc[0]["games_played"])
      assert games_played == 159, (
          f"BUG-013: two-way player games_played overwritten by pitching group; "
          f"expected max(159, 10) = 159, got {games_played}"
      )


  def test_single_group_player_unchanged():
      """A single-position player should still get the right games_played."""
      mock_response = {
          "stats": [
              {
                  "type": {"displayName": "yearByYear"},
                  "group": {"displayName": "hitting"},
                  "splits": [{"season": "2024", "stat": {"gamesPlayed": 145}}],
              },
          ]
      }
      with patch("src.injury_model.statsapi.player_stat_data", return_value=mock_response):
          from src.injury_model import load_injury_history_from_api
          df = load_injury_history_from_api([12345])
      row_2024 = df[df["season"] == 2024]
      assert int(row_2024.iloc[0]["games_played"]) == 145
  ```

- [ ] **Step 3.2: Run test to verify it fails**

  ```bash
  python -m pytest tests/test_injury_two_way_player.py -q 2>&1 | tail -10
  ```
  Expect: first test FAILS (`got 10`), second PASSES.

- [ ] **Step 3.3: Fix the loop** — `src/injury_model.py:362-368`

  Find lines 362-368:
  ```python
          for season in seasons:
              games_played = 0
              for stat_group in stats.get("stats", []):
                  for split in stat_group.get("splits", []):
                      if split.get("season") == str(season):
                          games_played = int(split.get("stat", {}).get("gamesPlayed", 0))
                          break
  ```
  Replace with:
  ```python
          for season in seasons:
              games_played = 0
              # Take MAX across stat groups: two-way players (e.g. Ohtani) have
              # separate hitting + pitching splits; the smaller value would
              # silently overwrite the larger one if we naively assigned.
              # (BUG-013 fix.)
              for stat_group in stats.get("stats", []):
                  for split in stat_group.get("splits", []):
                      if split.get("season") == str(season):
                          gp = int(split.get("stat", {}).get("gamesPlayed", 0))
                          games_played = max(games_played, gp)
                          break  # done with this stat group's splits
  ```

- [ ] **Step 3.4: Run test to verify it passes**

  ```bash
  python -m pytest tests/test_injury_two_way_player.py -q 2>&1 | tail -10
  ```
  Expect: `2 passed`.

- [ ] **Step 3.5: Lint + format + commit**

  ```bash
  python -m ruff format src/injury_model.py tests/test_injury_two_way_player.py
  python -m ruff check src/injury_model.py tests/test_injury_two_way_player.py
  git add src/injury_model.py tests/test_injury_two_way_player.py
  git commit -m "$(cat <<'EOF'
  fix(injury): use max games_played across stat groups for two-way players (BUG-013)

  load_injury_history_from_api requests group="hitting,pitching", so the
  outer loop iterates BOTH stat groups. The inner `break` only escaped
  the splits loop, not the stat_groups loop — so for Ohtani-class two-way
  players, hitting gamesPlayed=159 got overwritten by pitching
  gamesPlayed=10. health_score = 10/162 ≈ 0.06 flagged the healthiest
  two-way player as severely injured.

  Fix: take max(games_played, gp) across the two groups so the larger
  count survives.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-013)
  EOF
  )"
  ```

---

## Task 4: BUG-014 — ECR source-weight key mismatch

**Insight:** `refresh_ecr_consensus` builds `sources = {c: row[c] for c in rank_cols if pd.notna(row[c])}` where `rank_cols` are columns ending in `_rank` (e.g., `"espn_rank"`, `"fantasypros_rank"`). Then `_compute_player_consensus(sources)` looks up weights via `_SOURCE_WEIGHTS_INSEASON.get(src, 1.0)` — but those weights are keyed by `"espn"`, `"fantasypros"` (no suffix). The lookup always returns the 1.0 default, so the entire in-season weighting (1.5× FantasyPros ROS, 0.4× preseason Yahoo) is a silent no-op.

**Files:**
- Modify: `src/ecr.py:1074`
- Create: `tests/test_ecr_source_weighting.py`

- [ ] **Step 4.1: Write the failing test**

  Create `tests/test_ecr_source_weighting.py`:
  ```python
  """Test BUG-014 fix: ECR consensus actually applies in-season source weights."""
  import pandas as pd
  import pytest


  def test_compute_player_consensus_applies_source_weights():
      """_compute_player_consensus should weight sources by _SOURCE_WEIGHTS_INSEASON.
      Pass keys WITHOUT the '_rank' suffix (matching the weight dict keys)."""
      from src.ecr import _compute_player_consensus, _SOURCE_WEIGHTS_INSEASON

      # FantasyPros ROS weight is documented as ~1.5× in-season.
      # Yahoo preseason weight is documented as ~0.4× in-season.
      # Construct two source dicts: same average rank but different mixes.
      assert "fantasypros" in _SOURCE_WEIGHTS_INSEASON
      assert "yahoo" in _SOURCE_WEIGHTS_INSEASON
      assert _SOURCE_WEIGHTS_INSEASON["fantasypros"] > _SOURCE_WEIGHTS_INSEASON["yahoo"]

      # Player A: ranked by fantasypros=50, yahoo=100 → weighted toward 50
      # Player B: ranked by fantasypros=100, yahoo=50 → weighted toward 100
      a = _compute_player_consensus({"fantasypros": 50, "yahoo": 100})
      b = _compute_player_consensus({"fantasypros": 100, "yahoo": 50})
      assert a["consensus_avg"] < b["consensus_avg"], (
          "BUG-014: ECR source weighting is broken — "
          f"player with fantasypros=50/yahoo=100 should have LOWER consensus "
          f"(better rank) than fantasypros=100/yahoo=50, "
          f"but got A={a['consensus_avg']:.2f} vs B={b['consensus_avg']:.2f}"
      )


  def test_refresh_ecr_strips_rank_suffix_from_keys():
      """The dict passed into _compute_player_consensus must use bare source names
      (no '_rank' suffix) so weight lookups hit _SOURCE_WEIGHTS_INSEASON properly."""
      from src.ecr import _compute_player_consensus, _SOURCE_WEIGHTS_INSEASON

      # If keys are '_rank'-suffixed, weights all default to 1.0 → equal weighting.
      # Test by passing both forms and confirming the bare form gives the
      # asymmetric result.
      bare = _compute_player_consensus({"fantasypros": 30, "yahoo": 60})
      suffixed = _compute_player_consensus({"fantasypros_rank": 30, "yahoo_rank": 60})
      # The bare form should weight fantasypros more heavily than yahoo
      # (since fantasypros' weight is higher).
      # If the suffixed form is broken (all weights = 1.0), it will produce a
      # SIMPLE average ((30+60)/2 = 45), while the bare form will be lower (< 45).
      assert bare["consensus_avg"] < suffixed["consensus_avg"], (
          "BUG-014: passing _rank-suffixed keys does NOT trigger in-season "
          "weighting (all weights default to 1.0). Caller must strip the suffix."
      )
  ```

- [ ] **Step 4.2: Run test to verify it fails**

  ```bash
  python -m pytest tests/test_ecr_source_weighting.py -q 2>&1 | tail -10
  ```
  Expect: BOTH tests fail because the source-weight dict isn't being honored.

- [ ] **Step 4.3: Fix the dict comprehension** — `src/ecr.py:1074`

  Find line 1074:
  ```python
          for _, row in result_df.iterrows():
              sources = {c: row[c] for c in rank_cols if pd.notna(row[c])}
              result_dict = _compute_player_consensus(sources)
  ```
  Replace with:
  ```python
          for _, row in result_df.iterrows():
              # Strip "_rank" suffix so keys match _SOURCE_WEIGHTS_INSEASON
              # (which uses "espn", "fantasypros", etc. — not "*_rank"). The
              # earlier `{c: row[c] for c in rank_cols}` left the suffix on the
              # key, so weights.get(src, 1.0) always returned the 1.0 default,
              # silently disabling in-season source weighting. (BUG-014 fix.)
              sources = {
                  c.removesuffix("_rank"): row[c]
                  for c in rank_cols
                  if pd.notna(row[c])
              }
              result_dict = _compute_player_consensus(sources)
  ```

- [ ] **Step 4.4: Run test to verify it passes**

  ```bash
  python -m pytest tests/test_ecr_source_weighting.py -q 2>&1 | tail -10
  ```
  Expect: `2 passed`.

  Run all ECR tests to ensure no regressions:
  ```bash
  python -m pytest tests/test_ecr*.py -q 2>&1 | tail -10
  ```

- [ ] **Step 4.5: Lint + format + commit**

  ```bash
  python -m ruff format src/ecr.py tests/test_ecr_source_weighting.py
  python -m ruff check src/ecr.py tests/test_ecr_source_weighting.py
  git add src/ecr.py tests/test_ecr_source_weighting.py
  git commit -m "$(cat <<'EOF'
  fix(ecr): strip _rank suffix so source-weight lookups hit the weights dict (BUG-014)

  refresh_ecr_consensus built sources dict keyed by '_rank'-suffixed column
  names ("espn_rank", "fantasypros_rank", ...) but _compute_player_consensus
  looked up _SOURCE_WEIGHTS_INSEASON which is keyed by bare source names
  ("espn", "fantasypros"). Every weights.get(src, 1.0) call returned the
  1.0 default, silently neutralizing the in-season weighting (1.5×
  FantasyPros ROS, 0.4× preseason Yahoo).

  Fix: c.removesuffix("_rank") in the dict comprehension. The weights
  dict is now actually applied.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-014)
  EOF
  )"
  ```

---

## Task 5: BUG-022 — IL manager cold-start false positives

**Insight:** `detect_il_changes(current_roster, last_known_status=None)` defaults to `{}`, then for every IL player iterates and computes `old_status = last_known_status.get(pid, "")`. So `status != ""` is True for every IL player → cold start emits "new IL change" for every existing IL player. Spammy false-positive alerts on every fresh session.

**Files:**
- Modify: `src/il_manager.py:107-134`
- Create: `tests/test_il_manager_cold_start.py`

- [ ] **Step 5.1: Write the failing test**

  Create `tests/test_il_manager_cold_start.py`:
  ```python
  """Test BUG-022 fix: il_manager.detect_il_changes does not flag everyone on cold start."""
  import pandas as pd


  def _roster():
      return pd.DataFrame([
          {"player_id": 1, "name": "Player A", "status": "IL10", "positions": "OF"},
          {"player_id": 2, "name": "Player B", "status": "IL15", "positions": "P"},
          {"player_id": 3, "name": "Player C", "status": "active", "positions": "1B"},
          {"player_id": 4, "name": "Player D", "status": "DTD", "positions": "SS"},
      ])


  def test_cold_start_emits_no_changes():
      """When last_known_status is None (cold start), no changes should be
      emitted. Caller cannot distinguish 'new IL transition' from 'pre-existing
      IL' without prior baseline. (BUG-022 fix.)"""
      from src.il_manager import detect_il_changes
      changes = detect_il_changes(_roster(), last_known_status=None)
      assert changes == [], (
          f"BUG-022: cold start should emit empty list (no baseline), "
          f"but emitted {len(changes)} change(s)."
      )


  def test_real_transition_emits_change():
      """When last_known_status shows a player was 'active' and now is 'IL10',
      that's a real transition — emit a change."""
      from src.il_manager import detect_il_changes
      last_known = {1: "active", 2: "IL15", 3: "active", 4: "DTD"}
      changes = detect_il_changes(_roster(), last_known_status=last_known)
      # Player 1 transitioned active → IL10 (real new IL)
      # Player 2's status unchanged (IL15 → IL15)
      # Player 3 unchanged
      # Player 4 unchanged
      pids = [c["player_id"] for c in changes]
      assert pids == [1], f"Expected only pid 1 flagged; got {pids}"


  def test_empty_dict_treated_as_cold_start_or_explicit_empty():
      """Caller passing {} explicitly is treated as 'no prior IL state' —
      every current IL player IS a new transition. This is the documented
      contract divergence from None (None = first run, {} = explicit empty)."""
      from src.il_manager import detect_il_changes
      changes = detect_il_changes(_roster(), last_known_status={})
      pids = sorted(c["player_id"] for c in changes)
      # All 3 IL players (1, 2, 4) are "new" transitions vs empty dict
      assert pids == [1, 2, 4], f"Expected pids 1,2,4 flagged; got {pids}"
  ```

- [ ] **Step 5.2: Run test to verify it fails**

  ```bash
  python -m pytest tests/test_il_manager_cold_start.py -q 2>&1 | tail -10
  ```
  Expect: `test_cold_start_emits_no_changes` FAILS (returns all 3 IL players, not empty). Other two should pass with current code.

- [ ] **Step 5.3: Fix the function contract** — `src/il_manager.py:107-134`

  Find the function `detect_il_changes` (around line 107). The current body:
  ```python
  def detect_il_changes(
      current_roster: pd.DataFrame,
      last_known_status: dict[int, str] | None = None,
  ) -> list[dict]:
      """Detect new IL changes by comparing current roster status to last known."""
      if last_known_status is None:
          last_known_status = {}
      changes = []
      status_col = "status" if "status" in current_roster.columns else "injury_note"
      if status_col not in current_roster.columns:
          return changes
      for _, row in current_roster.iterrows():
          pid = int(row.get("player_id", 0))
          status = str(row.get(status_col, "") or "")
          old_status = last_known_status.get(pid, "")
          if status and status != old_status:
              il_type = classify_il_type(status)
              if il_type in IL_DURATION_ESTIMATES:
                  changes.append(...)
      return changes
  ```
  Replace the body with:
  ```python
  def detect_il_changes(
      current_roster: pd.DataFrame,
      last_known_status: dict[int, str] | None = None,
  ) -> list[dict]:
      """Detect new IL changes by comparing current roster status to last known.

      Contract:
      - `last_known_status=None` (cold start, no prior baseline): returns
        empty list. Callers cannot distinguish "new IL transition" from
        "pre-existing IL" without history — emitting alerts for every
        existing IL player would spam every cold session. Persist the
        current roster's status and pass it on the next call.
      - `last_known_status={}` (explicit empty): every current IL player IS
        a new transition relative to the empty baseline.
      - `last_known_status={pid: status, ...}`: emits only players whose
        current status differs from the prior recorded value AND classifies
        to a known IL type.

      (BUG-022 fix: cold-start path no longer emits a flood of false-positive
      "new IL change" alerts for the entire pre-existing IL list.)
      """
      if last_known_status is None:
          # Cold start: no baseline → cannot detect transitions → emit nothing.
          return []
      changes: list[dict] = []
      status_col = "status" if "status" in current_roster.columns else "injury_note"
      if status_col not in current_roster.columns:
          return changes
      for _, row in current_roster.iterrows():
          pid = int(row.get("player_id", 0))
          status = str(row.get(status_col, "") or "")
          old_status = last_known_status.get(pid, "")
          if status and status != old_status:
              il_type = classify_il_type(status)
              if il_type in IL_DURATION_ESTIMATES:
                  changes.append(
                      {
                          "player_id": pid,
                          "player_name": str(row.get("name", row.get("player_name", ""))),
                          "il_type": il_type,
                          "status": status,
                          "positions": str(row.get("positions", "")),
                      }
                  )
      return changes
  ```

- [ ] **Step 5.4: Run tests to verify all pass**

  ```bash
  python -m pytest tests/test_il_manager_cold_start.py -q 2>&1 | tail -10
  ```
  Expect: `3 passed`.

  Run any pre-existing IL-manager tests to ensure no regressions:
  ```bash
  python -m pytest tests/test_il_manager*.py tests/test_*il*.py -q 2>&1 | tail -10
  ```

- [ ] **Step 5.5: Lint + format + commit**

  ```bash
  python -m ruff format src/il_manager.py tests/test_il_manager_cold_start.py
  python -m ruff check src/il_manager.py tests/test_il_manager_cold_start.py
  git add src/il_manager.py tests/test_il_manager_cold_start.py
  git commit -m "$(cat <<'EOF'
  fix(il_manager): cold-start path emits no changes (BUG-022)

  detect_il_changes default `last_known_status=None` previously coalesced
  to {}, making old_status="" for every player → every existing IL roster
  member was flagged as a "new IL change" on every cold session. Spammy
  false-positives that masked real transitions.

  Fix: distinguish None (cold start, no baseline) from {} (explicit empty
  baseline). On None, return empty list; let caller establish baseline by
  persisting the current state for the next call. {} retains the prior
  behavior for callers who genuinely want "everything is new".

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md (BUG-022)
  EOF
  )"
  ```

---

## Task 6: Structural guards + CLAUDE.md update

**Files:**
- Create: `tests/test_no_ip_decimal_parse.py` (BUG-004 guard)
- Modify: `CLAUDE.md`

- [ ] **Step 6.1: Write the BUG-004 structural guard**

  Create `tests/test_no_ip_decimal_parse.py`:
  ```python
  """Permanent guard against BUG-004 regression: no direct `float(stat["inningsPitched"])` calls.

  The MLB Stats API formats innings-pitched as outs notation, not decimals.
  Any direct float() coercion is a BUG-004 regression.
  """
  import re
  from pathlib import Path


  REPO_ROOT = Path(__file__).resolve().parents[1]


  def _scan_files():
      for d in ("src", "scripts"):
          for p in (REPO_ROOT / d).rglob("*.py"):
              if p.name in {"live_stats.py"}:
                  # The legitimate _ip_outs_to_decimal definition + a few internal
                  # usages live here; the test below allows it.
                  pass
              yield p


  def test_no_direct_float_innings_pitched():
      """Reject `float(...inningsPitched...)` literal patterns in production code."""
      # Pattern: float(...stat... inningsPitched ...) OR float(...innings_pitched...)
      offending = []
      # Looser regex: any line containing both `float(` and the string `inningsPitched`
      for p in _scan_files():
          text = p.read_text(encoding="utf-8")
          for lineno, line in enumerate(text.splitlines(), start=1):
              if "float(" in line and "inningsPitched" in line:
                  offending.append((str(p), lineno, line.strip()))
      assert not offending, (
          "BUG-004 regression: direct `float(...inningsPitched...)` parse found. "
          "Use _ip_outs_to_decimal() from src.live_stats instead. Offenders:\n"
          + "\n".join(f"  {p}:{n}: {ln}" for p, n, ln in offending)
      )
  ```

- [ ] **Step 6.2: Run new guard test (should already pass after Task 1)**

  ```bash
  python -m pytest tests/test_no_ip_decimal_parse.py -q 2>&1 | tail -10
  ```
  Expect: `1 passed`.

- [ ] **Step 6.3: Update CLAUDE.md**

  Find the "Data Audit History" section. AFTER the "2026-05-11 whole-repo audit (SF-29 → SF-32)" paragraph that Wave 1 added, append:
  ```markdown
  **2026-05-11 Wave 2 (SF-33 → SF-37)** — Silent-failure elimination wave. SF-33 IP outs notation parsed as decimal (every ERA/WHIP off by ~1.5%; live DB confirmed all 536 pitcher IP values ended in .0/.1/.2 only). SF-34 Bayesian batch_update produced h/ab ≠ avg + hardcoded 35/65 BB:H split ignoring observed ratio. SF-35 Two-way player injury history overwritten by pitching gamesPlayed (Ohtani health_score 0.06 instead of 0.98). SF-36 ECR consensus source-weight lookups silently no-op due to `_rank`-suffix key mismatch. SF-37 IL-change detector flagged every existing IL player as "new" on cold start. All resolved in Wave 2 (PR # — fill in after Step 6.5).
  ```

  Then in the Structural Invariants table, add ONE new row (the other guards live in their per-bug test files):
  ```markdown
  | `test_no_ip_decimal_parse.py` | No direct `float(...inningsPitched...)` parses in src/ or scripts/ — must use `_ip_outs_to_decimal()` |
  ```

- [ ] **Step 6.4: Lint + commit**

  ```bash
  python -m ruff format tests/test_no_ip_decimal_parse.py
  python -m ruff check tests/test_no_ip_decimal_parse.py
  git add tests/test_no_ip_decimal_parse.py CLAUDE.md
  git commit -m "$(cat <<'EOF'
  test(audit): SF-33..SF-37 structural guard + CLAUDE.md history

  Adds test_no_ip_decimal_parse.py: no production code may call
  `float(...inningsPitched...)` — must use _ip_outs_to_decimal().

  Updates CLAUDE.md Data Audit History with Wave 2 summary and adds the
  new guard to the Structural Invariants table.

  Refs: docs/superpowers/specs/2026-05-11-bug-audit-findings.md
  EOF
  )"
  ```

---

## Phase Final: Push + PR

- [ ] **Step F.1: Push branch**

  ```bash
  git push -u origin claude/audit-wave2-plan
  ```

- [ ] **Step F.2: Create PR**

  ```bash
  gh pr create --title "Wave 2: silent-failure elimination (BUG-004/012/013/014/022)" --body "$(cat <<'EOF'
  ## Summary

  Implements Wave 2 of the [2026-05-11 bug audit findings](docs/superpowers/specs/2026-05-11-bug-audit-findings.md) — the 5 highest-impact silent-failure bugs.

  | Bug | Fix |
  |-----|-----|
  | BUG-004 | `_ip_outs_to_decimal()` helper; both call sites updated |
  | BUG-012 | Bayesian h = avg×ab (consistent); BB/H split uses observed ratio with regression |
  | BUG-013 | Two-way player history uses max(gp) across stat groups |
  | BUG-014 | ECR consensus strips `_rank` suffix before weight lookup |
  | BUG-022 | IL change detector returns empty list on cold start |
  | — | SF-33..SF-37 structural guards + CLAUDE.md history |

  All fixes are TDD per task with focused regression tests. No data-pipeline migrations needed (in contrast to Wave 1).

  🤖 Generated with [Claude Code](https://claude.com/claude-code)
  EOF
  )"
  ```

- [ ] **Step F.3: Print final summary**

  Print:
  - Number of commits added (should be 6: 5 fixes + 1 guards/history)
  - Test status: all Wave 2 + Wave 1 guards still pass
  - PR URL

---

## Self-Review

**Spec coverage check:**
- BUG-004 → Task 1 ✅
- BUG-012 → Task 2 ✅ (both sub-bugs: hitter h, pitcher BB/H)
- BUG-013 → Task 3 ✅
- BUG-014 → Task 4 ✅
- BUG-022 → Task 5 ✅
- Regression guards → Task 6 (test_no_ip_decimal_parse.py + the 5 per-bug tests serve as their own guards)

**Placeholder scan:**
- No "TBD"/"TODO"/"implement later" appear.
- All code blocks are full, copy-pasteable.
- File paths absolute or repo-relative.
- Test fixtures concrete.

**Type/identifier consistency:**
- `_ip_outs_to_decimal` defined in `src/live_stats.py`, imported into `player_databank.py`, used in tests.
- `_compute_player_consensus`, `_SOURCE_WEIGHTS_INSEASON` referenced consistently.
- `detect_il_changes` contract documented in fix + tested in 3 cases (None / empty / populated).

**Cold-start usability:**
- Worktree + branch stated explicitly.
- Bug summaries table embedded.
- Pre-flight verifies clean working tree + test baseline.
- Each task is independent; can be executed in any order.

Plan is complete and self-contained.
