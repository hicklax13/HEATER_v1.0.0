# Wave 8a: Behavioral MEDIUM/HIGH Fixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Address ~20 high-confidence behavioral bugs found by the 2026-05-11 whole-repo bug audit that escaped the Waves 1-7 top-25 bucket. Each fix is a clear-cut bug with documented file:line and observable wrong behavior.

**Architecture:** TDD per task — write failing test, implement fix, verify, commit. Six thematic groups, ~20 fixes total, ~6 commits.

**Tech Stack:** Python 3.11+, SQLite, pytest, pandas.

**Source spec:** [docs/superpowers/specs/2026-05-11-bug-audit-findings.md](../specs/2026-05-11-bug-audit-findings.md)

**Cwd:** `C:/Users/conno/OneDrive/Desktop/HEATER_v1.0.0/.claude/worktrees/audit-wave8a`
**Branch:** `claude/audit-wave8a-behavioral` (tracks `master` at `3a7e9e8`)

⚠️ All tasks: use `pytest -q 2>&1 | tail -10`. Commit explicitly per group.

---

## Group 1: SGP / Category math bugs (~4 fixes)

| Audit ID | File:Line | Bug |
|----------|-----------|-----|
| D5A-008 | `src/waiver_wire.py:535` | Per-category deltas divide by per-player SGP denom (rate stats off 100-1000x in UI) |
| D5B-030 | `src/start_sit_widget.py:24` | Hardcoded SGP denominators bypass LeagueConfig (SF-21/SF-25 violation) |
| D4B-011 | `src/engine/portfolio/category_analysis.py:200` | Inverse-stat gainability hardcoded `gap < 0.5` for L/ERA/WHIP (vastly different scales) |
| D5A-018 (MEDIUM) | `src/leaders.py:517` (likely `compute_category_leaders`) | Omits L category in inverse-stat leaders |

**Files:**
- Modify: `src/waiver_wire.py`, `src/start_sit_widget.py`, `src/engine/portfolio/category_analysis.py`, `src/leaders.py`
- Create: `tests/test_wave8a_sgp_category_math.py`

- [ ] **Step 1.1**: Write failing tests for each bug (one test per fix)
- [ ] **Step 1.2**: Fix D5A-008 — use league-level SGP denom from `LeagueConfig.sgp_denominators` not per-player
- [ ] **Step 1.3**: Fix D5B-030 — replace hardcoded denoms with `LeagueConfig().sgp_denominators[cat]`
- [ ] **Step 1.4**: Fix D4B-011 — scale `gap` threshold per-category: L/ERA/WHIP use category-typical scale, not bare 0.5
- [ ] **Step 1.5**: Fix D5A-018 — ensure L is in the inverse-stat leader rotation
- [ ] **Step 1.6**: Run tests, lint, commit:
  ```
  fix(audit): SGP/category math bugs (Wave 8a/1)
  
  D5A-008/D5B-030/D4B-011/D5A-018: 4 behavioral bugs in SGP and
  category math.
  ```

---

## Group 2: Silent fallbacks and error-swallowing (~5 fixes)

| Audit ID | File:Line | Bug |
|----------|-----------|-----|
| D5B-031 | `src/weekly_report.py:184` | `check_daily_lineup` returns `[]` when `todays_games is None` (silent skip) |
| D5B-034 | `src/alerts.py:443` | `compute_swap_impacts` outer bare-except → returns `[]` for any failure |
| D5B-042 | `src/leaders.py:517` | `compute_projection_skew` bare except → `projection_skew=""` for all players |
| D4B-005 | `src/engine/output/trade_evaluator.py:846` | DB roster rebuild swallows errors → uniform category weights without UI signal |
| D4B-020 | `src/trade_finder.py:1052,1351` | `_player_sgp_volume_aware` returns 0.0 silently on missing player |

**Pattern fix for each:** Replace bare `except: return []` with:
1. Specific exception class (or remove try/except if not needed)
2. `logger.warning(...)` with context
3. Surface the failure in the return value (e.g. add a `status: "error"` field to dict returns) OR raise for callers that should handle

**Files:**
- Modify: `src/weekly_report.py`, `src/alerts.py`, `src/leaders.py`, `src/engine/output/trade_evaluator.py`, `src/trade_finder.py`
- Create: `tests/test_wave8a_silent_fallbacks.py`

- [ ] **Step 2.1**: Write tests verifying logger.warning is called on each failure path
- [ ] **Step 2.2**: Fix each site per pattern above
- [ ] **Step 2.3**: Run tests, lint, commit

---

## Group 3: Pitcher / rate-stat bugs (~5 fixes)

| Audit ID | File:Line | Bug |
|----------|-----------|-----|
| D4B-002 | `src/trade_simulator.py:122` | Odd `n_sims` leaves fake zero in surpluses array → biased toward 0 |
| D4B-003 | `src/trade_simulator.py:324` | Pitcher rate-stat fallback: missing WHIP defaults to 0 IP → 0 baserunners contribution |
| D6B-001 | `src/game_day.py:962` | ERA/WHIP return 0.00 on no-IP → display looks ELITE (should show "—" / NaN) |
| D6B-002..003 | `src/game_day.py:715` | `row["fip"] = row["team_era"]` in statsapi fallback — FIP=ERA persisted silently |
| D6B-022 | `src/simulation.py:430` | `np.maximum(sgp_values, 0.01)` loses sign for inverse-stat-heavy pitchers (negative SGP floored to +0.01) |

**Files:**
- Modify: `src/trade_simulator.py`, `src/game_day.py`, `src/simulation.py`
- Create: `tests/test_wave8a_pitcher_rate_bugs.py`

- [ ] **Step 3.1**: Tests for each bug
- [ ] **Step 3.2**: Fixes per-file
- [ ] **Step 3.3**: Verify, lint, commit

---

## Group 4: Cold-start / initialization bugs (~3 fixes)

| Audit ID | File:Line | Bug |
|----------|-----------|-----|
| D5A-013 | `src/standings_utils.py:108` | `clear_cache` references `_cached_fa_pool` before its definition → NameError risk on cold start |
| D6A-001 | `src/game_day.py:32` | Hardcoded UTC-4 offset ignores DST |
| D2A-002 | `src/yahoo_api.py:1689` | `_INVERSE_CATS` evaluated at class-body time → config changes ignored |

**Files:**
- Modify: `src/standings_utils.py`, `src/game_day.py`, `src/yahoo_api.py`
- Create: `tests/test_wave8a_cold_start_init.py`

- [ ] **Step 4.1**: Tests for each
- [ ] **Step 4.2**: Fix D5A-013 — define `_cached_fa_pool` at module level OR move `clear_cache` definition after it
- [ ] **Step 4.3**: Fix D6A-001 — use `zoneinfo.ZoneInfo("America/New_York")` instead of hardcoded `-4`
- [ ] **Step 4.4**: Fix D2A-002 — evaluate `_INVERSE_CATS` at instance-init time, not class-body
- [ ] **Step 4.5**: Verify, lint, commit

---

## Group 5: Pool / mask / shape bugs (~3 fixes)

| Audit ID | File:Line | Bug |
|----------|-----------|-----|
| D6A-002,003 | `src/draft_engine.py:740,779,954` | `is_hitter=True` default flips pitcher mask in 3 enhancement stages — silently no-ops when col missing |
| D6B-011..013 | `src/draft_engine.py:739,779,817` | `pool.get("is_hitter", True)` returns scalar when col missing → broken Series broadcast |
| D6B-017 | `src/draft_state.py:266` | NaN→0 coercion in roster totals mis-computes OBP denom |
| D3A-003 | `src/optimizer/projections.py:154` | K3 block crashes when only one of `xwoba_delta`/`babip_delta` is present (defensive check missing) |

**Files:**
- Modify: `src/draft_engine.py`, `src/draft_state.py`, `src/optimizer/projections.py`
- Create: `tests/test_wave8a_pool_mask.py`

- [ ] **Step 5.1**: Tests
- [ ] **Step 5.2**: Fix is_hitter default — raise KeyError or use `is_hitter=False` default (safer; non-hitter applies to all)
- [ ] **Step 5.3**: Fix NaN coercion in OBP denom — exclude NaN rows from `ab + bb + hbp + sf` aggregation
- [ ] **Step 5.4**: Fix K3 block — check both keys present before computing
- [ ] **Step 5.5**: Verify, lint, commit

---

## Group 6: Miscellaneous behavioral fixes (~2 fixes)

| Audit ID | File:Line | Bug |
|----------|-----------|-----|
| D5B-037 | `src/ip_tracker.py:62` | `ip_per_start = ip_season / 30.0` hardcodes 30 starts → undercounts remaining IP for SP |
| D5B-014 | `src/matchup_context.py:344` | `get_matchup_adjustments` returns un-adjusted roster on failure (indistinguishable from success) |

**Files:**
- Modify: `src/ip_tracker.py`, `src/matchup_context.py`
- Create: `tests/test_wave8a_misc.py`

- [ ] **Step 6.1**: Tests for each
- [ ] **Step 6.2**: Fix `ip_per_start` — use the actual `games_started` if available; fall back to league average (~25 for a 162-game season SP)
- [ ] **Step 6.3**: Fix `get_matchup_adjustments` — return tuple `(roster, applied: bool)` or raise on failure, never silently no-op
- [ ] **Step 6.4**: Verify, lint, commit

---

## Group 7: CLAUDE.md + cumulative sweep

- [ ] **Step 7.1**: Run full Wave-8a-introduced + structural-invariant sweep
- [ ] **Step 7.2**: Update CLAUDE.md "Data Audit History" with Wave 8a paragraph (SF-58..SF-77)
- [ ] **Step 7.3**: Commit:
  ```
  docs(audit): CLAUDE.md history for Wave 8a (SF-58..SF-77)
  ```

---

## Phase Final: Push + PR + CodeRabbit + auto-merge

- [ ] **Step F.1**: Push branch
- [ ] **Step F.2**: Create PR with summary of 6 groups × ~3 fixes
- [ ] **Step F.3**: Dispatch `coderabbit:code-reviewer` on the PR
- [ ] **Step F.4**: Address any IMPORTANT findings
- [ ] **Step F.5**: Wait for CI, merge via `gh pr merge --merge --delete-branch`

---

## Self-Review

**Spec coverage:** 20 audit findings across 6 thematic groups. All have specific file:line citations and clear behavioral impact described.
**Placeholder scan:** None — every fix has a defined approach.
**Cold-start usability:** Worktree path, branch, source spec, per-group file lists provided.

## Notes for the implementer

- Run pytest verbose output through `2>&1 | tail -10` always.
- If a fix requires more than 30 minutes of design work or touches >3 files, STOP and report as `NEEDS_CONTEXT` — that fix probably wasn't the right level of granularity for Wave 8a.
- Keep the structural-invariant guard discipline: any new behavioral test should be small + clear + named `test_wave8a_*`.
- If existing tests break because they were asserting the OLD buggy behavior, update them — but document each test change in the commit message.
- The 6 groups are designed to be independent. If Group 3 blocks on something, skip to Group 4. Report blockers.
