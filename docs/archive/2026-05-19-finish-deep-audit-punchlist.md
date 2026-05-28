# Deep Audit Punchlist Completion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete every remaining item in `docs/2026-05-17-deep-audit-punchlist.md` (Sections 2/3/4/5/6/7 + new Section 8 path-scrub + new Section 9 preflight tooling) as a single mega-PR with semantic per-section commits, then leave only `master` branch.

**Architecture:**
- TDD for new helper extracts (Section 5) — failing test first.
- Structural-invariant guard tests for every dedup (Section 3) — ensures regressions fail CI.
- Surgical edits + targeted tests for bug fixes (Section 2 autonomous).
- `AskUserQuestion` for two deferred design calls (L3, L7) — do NOT guess.
- Single branch `claude/finish-deep-audit-punchlist`, semantic per-commit messages, squash-merge at end.

**Tech Stack:** Python 3.12+/3.14, pytest, ruff (lint+format), gh CLI, PowerShell on Windows.

**Phase 0 baseline (already captured):** 3947 passing, 7 failing (pre-existing on master), 17 skipped.

**Hard CI gate per commit:** `python -m ruff check . && python -m ruff format --check . && python -m pytest --ignore=tests/test_cheat_sheet.py -q` — pass count must remain ≥ 3947, no NEW failures.

---

## File Structure Overview

### Files to CREATE
- `scripts/preflight.py` — env + access verification (already written in Phase 0)
- `tests/test_no_stale_folder_path.py` — Section 8 structural-invariant guard
- `tests/test_normalize_player_name_canonical.py` — D2 guard + unit tests
- `tests/test_no_inverse_cats_literals.py` — D4 structural-invariant guard
- `tests/test_no_rate_stats_literals.py` — D6 structural-invariant guard
- `tests/test_render_position_filter.py` — Section 5 helper unit test
- `tests/test_weeks_remaining_canonical.py` — Section 5 helper unit test
- `src/league_rules.py` — add `weeks_remaining()` wrapper (file already exists with other rules; appending)

### Files to MODIFY (by section)
- **Section 2 L8** (dead `sf` lines): `src/valuation.py:451, 536`
- **Section 2 L14** (stale comment): `src/engine/portfolio/category_analysis.py:32`
- **Section 2 L3** (after design pause): `src/standings_engine.py:188`
- **Section 2 L7** (after design pause): `src/optimizer/matchup_adjustments.py:427` (1 site) OR `src/start_sit.py:758`, `src/matchup_planner.py:193`, `src/draft_engine.py:619-623`, `src/two_start.py:321` (4 sites)
- **Section 3 D2**: `src/valuation.py` (add canonical), `src/league_manager.py`, `src/live_stats.py`, `src/optimizer/daily_optimizer.py`
- **Section 3 D4**: `pages/6_League_Standings.py:77`, `src/draft_engine.py:129`, `scripts/draft_vs_current.py:12`, `scripts/optimal_roster_sim.py:12`, `src/waiver_wire.py:1067` (fallback simplification)
- **Section 3 D6**: `pages/6_League_Standings.py:76,149`, `src/player_card.py:57`, `src/engine/monte_carlo/trade_simulator.py:299`, `src/ui_shared.py:2343`, `src/waiver_wire.py:50`, `src/optimizer/category_urgency.py:111,279`, `src/optimizer/pivot_advisor.py:104` (fallback simplification)
- **Section 4**: `src/ecr.py:976-997`, F841 audit, fresh orphan sweep
- **Section 5 render_position_filter**: `src/ui_shared.py` (add helper), `pages/12_Trade_Finder.py:978`, `pages/14_Free_Agents.py:331`, `pages/20_Draft_Simulator.py:527`, `pages/17_Leaders.py:342`
- **Section 5 weeks_remaining**: `src/league_rules.py` (add wrapper), `src/validation/dynamic_context.py:32` (fix `FANTASY_REGULAR_SEASON_WEEKS=22→26` bonus), `pages/5_Matchup_Planner.py:171`, `pages/12_Trade_Finder.py:941-942`, `pages/6_League_Standings.py:458` (consolidate alternate fn)
- **Section 6**: `docs/architecture.md`, `docs/Research.md`, move `docs/2026-05-17-deep-audit-punchlist.md` → `docs/archive/`
- **Section 7**: `CLAUDE.md` (local-path refs only)
- **Section 8**: any file containing `HEATER_v1.0.0` as a LOCAL folder path (NOT GitHub URL, NOT OneDrive WARNING block); plus `__pycache__/` purge

---

## Phase 1 — Branch + preflight commit

### Task 1: Create branch + commit scripts/preflight.py

**Files:**
- Create branch: `claude/finish-deep-audit-punchlist`
- Commit: `scripts/preflight.py` (already on disk from Phase 0)

- [ ] **Step 1: Verify clean tree + on master**

```bash
git status --porcelain                       # expect only ?? scripts/preflight.py
git branch --show-current                    # expect: master
git pull --ff-only origin master             # already up to date
```

- [ ] **Step 2: Create branch**

```bash
git switch -c claude/finish-deep-audit-punchlist
```

- [ ] **Step 3: Stage + commit preflight**

```bash
git add scripts/preflight.py
git commit -m "$(cat <<'EOF'
chore(scripts): add preflight env+access verification

Pre-flight sanity check for HEATER deep-audit completion sessions.
Verifies: filesystem RW, Python 3.12+, required imports (pandas, numpy,
streamlit, yfpy, etc.), optional deps (pymc, xgboost), env vars,
yahoo_token.json shape, gh auth + push permission, pre-commit hook,
git tree cleanliness, worktree count.

Exit codes: 0 (clean), 1 (hard fail), 2 (warnings to review).

Run: python scripts/preflight.py

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify commit**

```bash
git log --oneline -2                         # expect new commit + 29fdaed parent
```

---

## Section 2 — Logic / math bugs (autonomous subset)

### Task 2: L8 — delete dead `sf` lines in valuation.py

**Files:**
- Modify: `src/valuation.py:451` (line `player.get("sf", 0) or 0` — no-op)
- Modify: `src/valuation.py:536` (same pattern in `_marginal_obp_sgp`)

- [ ] **Step 1: Read current state**

`src/valuation.py:451` (inside `_rate_stat_sgp`, OBP branch):
```python
            hbp = player.get("hbp", 0) or 0
            player.get("sf", 0) or 0                          # ← dead
            if pa == 0:
                return 0
```

`src/valuation.py:536` (inside `_marginal_obp_sgp`):
```python
            hbp = player.get("hbp", 0) or 0
            player.get("sf", 0) or 0                          # ← dead
            pa = player.get("pa", 0) or 0
```

- [ ] **Step 2: Delete both dead lines + add a single-line note above the OBP num denominator**

Edit `src/valuation.py:451` — DELETE the `player.get("sf", 0) or 0` line. Add a comment on the line before `if pa == 0`:
```python
            # OBP denominator ≈ pa (covers ab+bb+hbp+sf); sf not separately needed.
```

Same edit at line 536: DELETE the dead `sf` get, add comment near `pa = player.get(...)`.

- [ ] **Step 3: Run targeted tests**

```bash
python -m pytest tests/test_valuation.py tests/test_sgp_calculator.py -q 2>&1 | tail -10
```
Expected: 0 failures.

- [ ] **Step 4: Run ruff (F841 will drop by 2 if it tracked them)**

```bash
python -m ruff check src/valuation.py
```
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add src/valuation.py
git commit -m "fix(valuation): Section 2 L8 — remove dead sf get/discard lines

PR #33 classified L8 as 'not a bug; pa approximates ab+bb+hbp+sf'.
This removes the dead 'player.get(\"sf\", 0) or 0' statements at
valuation.py:451 (_rate_stat_sgp OBP) and 536 (_marginal_obp_sgp);
both compute a value and immediately discard it. Adds a 1-line
comment noting pa is the OBP denominator.

No behavior change. Drops F841-style dead-code residue.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: L14 — refresh stale 22-week comment in category_analysis.py

**Files:**
- Modify: `src/engine/portfolio/category_analysis.py:32` (single-line comment update)

- [ ] **Step 1: Read current state**

`src/engine/portfolio/category_analysis.py:30-46`:
```python
# Approximate weekly production rates per roster, used for punt estimation.
# Based on a competitive 12-team roster in a full season (~22 weeks).
WEEKLY_RATE_DEFAULTS: dict[str, float] = {
    "R": 35.0,
    ...
}
```

- [ ] **Step 2: Edit the comment block**

Replace the 2-line comment with:
```python
# Approximate weekly production rates per roster, used for punt estimation.
# Calibrated for FourzynBurn 26-week regular season (LeagueConfig.season_weeks).
# Example sanity check: 35 R/wk × 26 wk = 910 R/season (realistic upper-bound team).
```

- [ ] **Step 3: Run targeted tests**

```bash
python -m pytest tests/test_category_analysis.py tests/test_engine_portfolio.py -q 2>&1 | tail -10
```
Expected: 0 new failures (test files may not exist; broader test run if so).

- [ ] **Step 4: Commit**

```bash
git add src/engine/portfolio/category_analysis.py
git commit -m "docs(category_analysis): Section 2 L14 — refresh 22→26 week calibration comment

PR #33 classified L14 as 'only stale comment; constants calibrated for
26 weeks (35×26=910 realistic season total)'. This updates the comment
block on WEEKLY_RATE_DEFAULTS to reflect FourzynBurn's 26-week season
and references LeagueConfig.season_weeks as the source of truth.

No behavior change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Section 2 — Logic bugs (DESIGN PAUSE)

### Task 4: L3 — change divisor from weeks_remaining → config.season_weeks (USER DECISION: 26)

**Files:** `src/standings_engine.py:188` (the `weeks = max(weeks_remaining, 1)` line in `_estimate_team_weekly_stats`)

**Pre-resolved design decision** (2026-05-19): User selected **Option A — pool has full-season projections; divide by `config.season_weeks` (26)**. Rationale: matches L1/L2/L4/L10/L11 fixes from PR #33; relative win-prob comparison is invariant to divisor but absolute weekly-mean becomes interpretable.

- [ ] **Step 1: Edit src/standings_engine.py:188**

```python
# OLD: weeks = max(weeks_remaining, 1)
# 2026-05-19 L3 fix: pool has FULL-SEASON projections; divide by season_weeks
# (26 for FourzynBurn) to get true weekly mean. Caller-passed weeks_remaining
# was unused for weekly-mean calc (relative win-prob is divisor-invariant when
# both teams use same divisor; absolute weekly mean is now interpretable).
weeks = max(config.season_weeks, 1)
```

- [ ] **Step 2: Run targeted tests**

```bash
python -m pytest tests/test_standings_engine.py tests/test_league_standings_integration.py tests/test_standings_calibration.py -q 2>&1 | tail -15
```

If tests previously pinned specific weekly-mean values calibrated for `weeks_remaining=9` or `=16`, update expected values to reflect 26-week divisor.

- [ ] **Step 3: Run broader regression**

```bash
python -m pytest --ignore=tests/test_cheat_sheet.py -q 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add src/standings_engine.py [tests modified]
git commit -m "fix(standings): Section 2 L3 — divisor → config.season_weeks (26)

PR #33 deferred L3 pending design review. User decision 2026-05-19:
pool has full-season projections; divide by config.season_weeks (26)
to produce interpretable weekly means.

Behavioral note: relative win probability is invariant to the divisor
(both teams scale identically); only absolute weekly-mean changes.
Tests pinned to old weeks_remaining=N values updated to 26.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: L7 — standardize pitcher park inversion to `1/pf` (USER DECISION)

**Files:** `src/start_sit.py:756-759`, `src/matchup_planner.py:193`, `src/draft_engine.py:619-623`, `src/two_start.py:321`

**Pre-resolved design decision** (2026-05-19): User selected **Option A — standardize on `1/pf` (reciprocal) everywhere**. Rationale: mathematically cleaner (multiplicative inverse), matches newer DCV pipeline, and matches matchup_adjustments.py:427's explicit "more accurate" comment.

- [ ] **Step 1: Edit src/start_sit.py:756-759**

```python
# OLD:
# pf = 2.0 - pf
# pf = max(pf, 0.5)  # floor to avoid extreme values
# 2026-05-19 L7 fix: standardize on reciprocal inversion (Coors pf=1.38 → 0.725).
# Reciprocal is multiplicative inverse — symmetric with how pf is applied to hitters.
pf = 1.0 / pf if pf > 0 else 1.0
```
Remove the floor (`max(pf, 0.5)`) — reciprocal of any positive pf is always positive and bounded.

- [ ] **Step 2: Edit src/matchup_planner.py:193**

```python
# OLD: inverse_park = max(2.0 - pf, 0.5)
# 2026-05-19 L7 fix: reciprocal inversion (canonical, see L7 design note).
inverse_park = 1.0 / pf if pf > 0 else 1.0
```

- [ ] **Step 3: Edit src/draft_engine.py:619-623**

```python
# OLD:
# pool.loc[is_pitcher, "park_factor_adj"] = 2.0 - pool.loc[is_pitcher, "park_factor_adj"]
# 2026-05-19 L7 fix: reciprocal inversion (canonical).
# Vectorized: pandas does element-wise reciprocal cleanly; guard pf > 0 to avoid div0.
pf_series = pool.loc[is_pitcher, "park_factor_adj"]
pool.loc[is_pitcher, "park_factor_adj"] = pf_series.where(pf_series <= 0, 1.0 / pf_series).fillna(1.0)
```

- [ ] **Step 4: Read + edit src/two_start.py:321**

```bash
sed -n '315,335p' src/two_start.py
```
Find where `park_adj` is inverted (if at all in this file — may already pass through). If `2-pf` or `2.0 - pf` appears, replace with `1.0 / pf if pf > 0 else 1.0`. If `park_adj` is just consumed (no inversion in this file), no edit needed; note in commit message.

- [ ] **Step 5: Run targeted tests**

```bash
python -m pytest tests/test_start_sit.py tests/test_matchup_planner.py tests/test_optimizer.py tests/test_draft_engine.py tests/test_two_start.py tests/test_optimizer_matchup_adjustments.py -q 2>&1 | tail -15
```

Tests that pinned `2 - pf` numeric values (e.g. expected `0.62` for Coors-affected pitcher) need updating to expect `0.725` (= `1/1.38`). Update them with the new value + a 1-line comment citing L7 canonicalization.

- [ ] **Step 6: Run broader regression**

```bash
python -m pytest --ignore=tests/test_cheat_sheet.py -q 2>&1 | tail -5
```

- [ ] **Step 7: Commit**

```bash
git add src/start_sit.py src/matchup_planner.py src/draft_engine.py src/two_start.py [tests updated]
git commit -m "fix(park-factor): Section 2 L7 — standardize pitcher park inversion to 1/pf

PR #33+#38 deferred L7 (5 sites, 2 inversion conventions). User decision
2026-05-19: standardize on the reciprocal (1/pf) convention used by
src/optimizer/matchup_adjustments.py:427 — mathematically cleaner
(multiplicative inverse of the factor) and consistent with the newer
DCV pipeline.

4 sites updated (start_sit, matchup_planner, draft_engine, two_start
if inverting there). Pitcher recommendations at Coors shift ~10% gentler
(pf 1.38: 2-pf=0.62 → 1/pf=0.725); tests pinned to old values updated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Section 3 — Dedup consolidation

### Task 6: D2 — consolidate normalize_player_name into valuation.py

**Files:**
- Modify: `src/valuation.py` (append canonical helper)
- Modify: `src/league_manager.py:206-217` (delete `_normalize_name`, import canonical)
- Modify: `src/live_stats.py:57-66` (delete `_normalize_name`, import canonical)
- Modify: `src/optimizer/daily_optimizer.py:184-201` (delete `_normalize_pitcher_name`, import canonical with rename alias if needed)
- Create: `tests/test_normalize_player_name_canonical.py` (guard + unit tests)

- [ ] **Step 1: TDD — write the failing test**

`tests/test_normalize_player_name_canonical.py`:
```python
"""D2 consolidation: normalize_player_name() lives in src.valuation only.

Guards against re-introducing per-module name-normalization implementations.
"""
from __future__ import annotations

from pathlib import Path
import re

import pytest

from src.valuation import normalize_player_name


@pytest.mark.parametrize("raw, expected", [
    ("Iván Rodríguez", "ivan rodriguez"),
    ("José Ramírez", "jose ramirez"),
    ("Vladimir Guerrero Jr.", "vladimir guerrero"),
    ("Robinson Canó III", "robinson cano"),
    ("Cal Ripken Sr.", "cal ripken"),
    ("Shohei Ohtani (Pitcher)", "shohei ohtani"),
    ("Shohei Ohtani (Batter)", "shohei ohtani"),
    ("Mike O'Neill", "mike oneill"),
    ("Hyun-Jin Ryu", "hyunjin ryu"),
    ("", ""),
    (None, ""),
])
def test_canonical_normalization(raw, expected):
    assert normalize_player_name(raw) == expected


def test_no_other_normalize_name_defs_in_src():
    """Structural guard: no other `def _normalize_name(` or `def _normalize_pitcher_name(`
    in src/. The canonical normalize_player_name in valuation.py is the only impl."""
    src = Path("src")
    pattern = re.compile(r"^def (_normalize_name|_normalize_pitcher_name)\b", re.MULTILINE)
    offenders = []
    for f in src.rglob("*.py"):
        text = f.read_text(encoding="utf-8", errors="ignore")
        if pattern.search(text):
            offenders.append(str(f))
    assert not offenders, (
        f"Found duplicate name-normalization impl(s); consolidate into "
        f"src.valuation.normalize_player_name: {offenders}"
    )
```

- [ ] **Step 2: Run test — expect ImportError (function doesn't exist yet)**

```bash
python -m pytest tests/test_normalize_player_name_canonical.py -v 2>&1 | tail -10
```
Expected: ImportError or AttributeError on `normalize_player_name`.

- [ ] **Step 3: Add canonical helper to valuation.py**

Insert near the top of `src/valuation.py` (after imports, before LeagueConfig):
```python
import re
import unicodedata


def normalize_player_name(name: object) -> str:
    """Normalize a player name for robust cross-source matching.

    Single canonical implementation (Section 3 D2 consolidation). Handles:

      * Yahoo parenthetical role suffixes: "Shohei Ohtani (Pitcher)" → "shohei ohtani"
      * Unicode accents via NFKD: "José Ramírez" → "jose ramirez"
      * Generational suffixes: "Vladimir Guerrero Jr." → "vladimir guerrero"
      * Casing: always returns lowercase
      * Punctuation: stripped except internal whitespace
      * Type safety: non-string / None / empty → "" (caller-safe)

    All consumers (`src/league_manager.py`, `src/live_stats.py`,
    `src/optimizer/daily_optimizer.py`, and any future caller) must
    import from here — duplicate impls are blocked by
    tests/test_normalize_player_name_canonical.py.
    """
    if not name or not isinstance(name, str):
        return ""
    name = re.sub(r"\s*\((?:Pitcher|Batter|P|B)\)\s*$", "", name).strip()
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii").lower().strip()
    for suffix in (" jr.", " jr", " sr.", " sr", " iii", " iv", " ii"):
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return " ".join(s.split())
```

- [ ] **Step 4: Replace callers**

In `src/league_manager.py`:
- DELETE the `_normalize_name` function (lines 206-217).
- Replace imports — add `from src.valuation import normalize_player_name`.
- Replace 2 call sites at lines 248, 255: `_normalize_name(n)` → `normalize_player_name(n)`.

In `src/live_stats.py`:
- DELETE `_normalize_name` (lines 57-66).
- Add `from src.valuation import normalize_player_name`.
- Replace call at line 84: `_normalize_name(player_name)` → `normalize_player_name(player_name)`.

In `src/optimizer/daily_optimizer.py`:
- DELETE `_normalize_pitcher_name` (lines 184-201).
- Add `from src.valuation import normalize_player_name as _normalize_pitcher_name` (preserves call-site names).

- [ ] **Step 5: Run test — must pass + no other failures**

```bash
python -m pytest tests/test_normalize_player_name_canonical.py -v 2>&1 | tail -15
python -m pytest tests/test_valuation.py tests/test_league_manager.py tests/test_live_stats.py tests/test_optimizer_daily.py -q 2>&1 | tail -10
```
Expected: PASS for guard. Existing tests stable.

- [ ] **Step 6: Run full suite (broader regression check)**

```bash
python -m pytest --ignore=tests/test_cheat_sheet.py -q 2>&1 | tail -5
```
Expected: ≥ 3947 passing.

- [ ] **Step 7: Commit**

```bash
git add src/valuation.py src/league_manager.py src/live_stats.py src/optimizer/daily_optimizer.py tests/test_normalize_player_name_canonical.py
git commit -m "refactor(name-norm): Section 3 D2 — consolidate normalize_player_name into valuation

Three subtly-different impls of player-name normalization existed:
- src/league_manager.py:206 (_normalize_name): NFD + suffix strip
- src/live_stats.py:57 (_normalize_name): NFKD + Yahoo (Pitcher) regex
- src/optimizer/daily_optimizer.py:184 (_normalize_pitcher_name): all of above + punctuation

Adds canonical src.valuation.normalize_player_name() that is the SUPERSET
of all three (handles Yahoo suffix + NFKD accents + lowercase + generational
suffix + punctuation). Replaces 3+ call sites. Adds
tests/test_normalize_player_name_canonical.py guard against re-introducing
per-module variants.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: D4 — replace INVERSE_CATS literals with LeagueConfig().inverse_stats

**Files:**
- Modify: `pages/6_League_Standings.py:77`
- Modify: `src/draft_engine.py:129`
- Modify: `scripts/draft_vs_current.py:12`
- Modify: `scripts/optimal_roster_sim.py:12`
- Modify: `src/waiver_wire.py:1067` (fallback simplification — fallback no longer needed)
- Create: `tests/test_no_inverse_cats_literals.py` (structural-invariant guard)

**Out of scope (intentional non-inverse-stat sets):**
- `src/war_room.py:55-56` — pitching-rate-cluster strategy rules
- `src/waiver_wire.py:713,717` — `_INVERSE_RATE_CATS` / `_PITCHING_RATE` (different concepts)
- `src/trade_intelligence.py:74` — `PITCHING_RATE_CLUSTER`
- `src/opponent_intel.py:359`, `src/optimizer/pivot_advisor.py:327` — strategy conditions
- `src/ui_shared.py:2519` — display formatting
- `tests/conftest.py:67`, `tests/test_trade_engine.py:151` — test literals (acceptable)

- [ ] **Step 1: TDD — write structural-invariant guard**

`tests/test_no_inverse_cats_literals.py`:
```python
"""D4: No hardcoded {"L", "ERA", "WHIP"} INVERSE-STAT sets outside LeagueConfig.

Allowed exceptions:
  * src/valuation.py — defines the canonical LeagueConfig.inverse_stats
  * tests/ — test literals are acceptable
  * docs/, CLAUDE.md — documentation
"""
from __future__ import annotations

from pathlib import Path
import re

# Inverse-stat set pattern: {"L", "ERA", "WHIP"} in any order
PATTERN = re.compile(
    r'\{\s*(?:"[LEW]\w*"\s*,\s*){2}"[LEW]\w*"\s*\}'
)
# Stricter: actual L/ERA/WHIP literal
STRICT = re.compile(r'\{[^}]*"L"[^}]*"ERA"[^}]*"WHIP"[^}]*\}|\{[^}]*"WHIP"[^}]*"ERA"[^}]*"L"[^}]*\}')

ALLOWED_FILES = {
    Path("src/valuation.py"),     # canonical definition
}
ALLOWED_DIRS = {"tests", "docs"}


def test_no_inverse_cats_literal_outside_league_config():
    root = Path(".")
    offenders = []
    for f in list(root.rglob("src/**/*.py")) + list(root.rglob("pages/**/*.py")) + list(root.rglob("scripts/**/*.py")):
        rel = f.relative_to(root)
        if rel in ALLOWED_FILES:
            continue
        if rel.parts[0] in ALLOWED_DIRS:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for m in STRICT.finditer(text):
            # Skip if literal is on a comment line
            lineno = text[:m.start()].count("\n") + 1
            line = text.splitlines()[lineno - 1].lstrip()
            if line.startswith("#"):
                continue
            offenders.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    assert not offenders, (
        "Hardcoded inverse-stat literal {\"L\", \"ERA\", \"WHIP\"} found. "
        "Use LeagueConfig().inverse_stats instead:\n  " + "\n  ".join(offenders)
    )
```

- [ ] **Step 2: Run test — expect failures listing offending sites**

```bash
python -m pytest tests/test_no_inverse_cats_literals.py -v 2>&1 | tail -15
```
Expected: FAIL with offender list (pages/6_League_Standings.py:77, src/draft_engine.py:129, scripts/draft_vs_current.py:12, scripts/optimal_roster_sim.py:12).

- [ ] **Step 3: Replace each site**

`pages/6_League_Standings.py:77`:
```python
# OLD: _INVERSE_CATS = {"L", "ERA", "WHIP"}
from src.valuation import LeagueConfig as _LC
_INVERSE_CATS = _LC().inverse_stats
```

`src/draft_engine.py:129`:
```python
# OLD: INVERSE_CATS_UPPER: set[str] = {"L", "ERA", "WHIP"}
# Use LeagueConfig.inverse_stats lazily inside functions (module-level singletons banned per SF-21).
def _inverse_cats_upper() -> set[str]:
    from src.valuation import LeagueConfig
    return LeagueConfig().inverse_stats
```
Update internal callers to use `_inverse_cats_upper()` instead of `INVERSE_CATS_UPPER`. If module-level constant is widely referenced, accept that change requires a sweep — keep as a SET via `{*LeagueConfig().inverse_stats}` if call-site shape requires.

`scripts/draft_vs_current.py:12` and `scripts/optimal_roster_sim.py:12`:
```python
# OLD: inverse_cats = {"L", "ERA", "WHIP"}
from src.valuation import LeagueConfig
inverse_cats = LeagueConfig().inverse_stats
```

`src/waiver_wire.py:1067`:
```python
# OLD: inverse_cats = config.inverse_stats if config else {"L", "ERA", "WHIP"}
# Simplify: LeagueConfig is always constructible; fallback redundant.
from src.valuation import LeagueConfig
inverse_cats = (config or LeagueConfig()).inverse_stats
```

- [ ] **Step 4: Run guard test — must PASS now**

```bash
python -m pytest tests/test_no_inverse_cats_literals.py -v 2>&1 | tail -10
```
Expected: PASS.

- [ ] **Step 5: Run broader regression**

```bash
python -m pytest --ignore=tests/test_cheat_sheet.py -q 2>&1 | tail -5
```
Expected: ≥ 3947.

- [ ] **Step 6: Commit**

```bash
git add pages/6_League_Standings.py src/draft_engine.py scripts/draft_vs_current.py scripts/optimal_roster_sim.py src/waiver_wire.py tests/test_no_inverse_cats_literals.py
git commit -m "refactor(inverse-cats): Section 3 D4 — replace literal {L,ERA,WHIP} with LeagueConfig

Five sites had literal inverse-stat sets that drift from
LeagueConfig.inverse_stats (the canonical source). Replaced with
LeagueConfig().inverse_stats accessor. Adds
tests/test_no_inverse_cats_literals.py structural guard against
re-introducing the literal.

NOT changed (intentional non-inverse-stat sets — different concept):
- src/war_room.py (pitching-rate-cluster rules)
- src/waiver_wire.py:713,717 (_INVERSE_RATE_CATS, _PITCHING_RATE)
- src/trade_intelligence.py:74 (PITCHING_RATE_CLUSTER)
- src/opponent_intel.py, src/optimizer/pivot_advisor.py (strategy conditions)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: D6 — replace RATE_STATS sets with LeagueConfig().rate_stats

**Files:**
- Modify: `pages/6_League_Standings.py:76,149`
- Modify: `src/player_card.py:57`
- Modify: `src/engine/monte_carlo/trade_simulator.py:299`
- Modify: `src/ui_shared.py:2343`
- Modify: `src/waiver_wire.py:50`
- Modify: `src/optimizer/category_urgency.py:111,279` (fallback simplification)
- Modify: `src/optimizer/pivot_advisor.py:104` (fallback simplification)
- Create: `tests/test_no_rate_stats_literals.py`

- [ ] **Step 1: TDD — write guard test**

Pattern matches `{"AVG", "OBP", "ERA", "WHIP"}` in any order. ALLOWED: `src/valuation.py` (canonical), `tests/`, `docs/`.

```python
"""D6: No hardcoded {"AVG", "OBP", "ERA", "WHIP"} RATE-STAT sets outside LeagueConfig."""
from __future__ import annotations
from pathlib import Path
import re

STRICT = re.compile(
    r'\{[^}]*"AVG"[^}]*"OBP"[^}]*"ERA"[^}]*"WHIP"[^}]*\}'
)

ALLOWED_FILES = {Path("src/valuation.py")}
ALLOWED_DIRS = {"tests", "docs"}


def test_no_rate_stats_literal_outside_league_config():
    root = Path(".")
    offenders = []
    for f in list(root.rglob("src/**/*.py")) + list(root.rglob("pages/**/*.py")) + list(root.rglob("scripts/**/*.py")):
        rel = f.relative_to(root)
        if rel in ALLOWED_FILES or rel.parts[0] in ALLOWED_DIRS:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")
        for m in STRICT.finditer(text):
            lineno = text[:m.start()].count("\n") + 1
            line = text.splitlines()[lineno - 1].lstrip()
            if line.startswith("#"):
                continue
            offenders.append(f"{rel}:{lineno}: {line.strip()[:100]}")
    assert not offenders, (
        "Hardcoded rate-stat literal {AVG,OBP,ERA,WHIP} found. "
        "Use LeagueConfig().rate_stats instead:\n  " + "\n  ".join(offenders)
    )
```

- [ ] **Step 2: Run — expect FAIL with offender list**

- [ ] **Step 3: Replace each site (same pattern as D4)**

Module-level constants → lazy LeagueConfig access:
```python
# OLD: _RATE_STATS = {"AVG", "OBP", "ERA", "WHIP"}
from src.valuation import LeagueConfig as _LC
_RATE_STATS = _LC().rate_stats
```

Getattr-fallback patterns → just use config (no fallback):
```python
# OLD: rate_stats = getattr(config, "rate_stats", {"AVG", "OBP", "ERA", "WHIP"})
# config is always a LeagueConfig; the fallback was defensive but unreachable.
rate_stats = config.rate_stats
```

- [ ] **Step 4: Run guard — PASS**

- [ ] **Step 5: Broader test sweep**

```bash
python -m pytest --ignore=tests/test_cheat_sheet.py -q 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git commit -m "refactor(rate-stats): Section 3 D6 — replace literal {AVG,OBP,ERA,WHIP} with LeagueConfig

Six sites had literal rate-stat sets + two getattr-fallback patterns
that drift from LeagueConfig.rate_stats. Simplified to use accessor
directly. Adds tests/test_no_rate_stats_literals.py guard.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Section 5 — Helper extracts

### Task 9: Extract render_position_filter helper

**Files:**
- Modify: `src/ui_shared.py` (append helper)
- Modify: `pages/12_Trade_Finder.py:978`, `pages/14_Free_Agents.py:331`, `pages/20_Draft_Simulator.py:527`, `pages/17_Leaders.py:342`
- Create: `tests/test_render_position_filter.py`

- [ ] **Step 1: TDD — unit test (mock streamlit)**

```python
"""Section 5: render_position_filter() helper extract."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import pandas as pd
import pytest


def test_render_position_filter_returns_selected_value():
    from src.ui_shared import render_position_filter
    pool = pd.DataFrame({"position": ["C", "1B", "SS", "OF", "SP", "RP"]})
    with patch("streamlit.selectbox", return_value="OF") as sel:
        result = render_position_filter(pool, key="test_key", default="All")
    assert result == "OF"
    sel.assert_called_once()
    # First positional arg or `options` kwarg should include "All" + position list
    args, kwargs = sel.call_args
    options = kwargs.get("options") or (args[1] if len(args) > 1 else None)
    assert options is not None and "All" in options
    assert {"C", "1B", "SS", "OF", "SP", "RP"}.issubset(set(options))


def test_render_position_filter_default_used_when_no_value():
    from src.ui_shared import render_position_filter
    pool = pd.DataFrame({"position": ["C", "1B"]})
    with patch("streamlit.selectbox", return_value="All") as sel:
        result = render_position_filter(pool, key="k2", default="All")
    assert result == "All"


def test_render_position_filter_canonical_order():
    """Positions appear in canonical roster order: C, 1B, 2B, 3B, SS, OF, SP, RP."""
    from src.ui_shared import render_position_filter
    pool = pd.DataFrame({"position": ["RP", "SP", "OF", "SS", "3B", "2B", "1B", "C"]})
    captured_options = {}
    def _capture(label, options, **kwargs):
        captured_options["opts"] = options
        return options[0]
    with patch("streamlit.selectbox", side_effect=_capture):
        render_position_filter(pool, key="k3")
    assert captured_options["opts"] == ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
```

- [ ] **Step 2: Run — expect ImportError**

- [ ] **Step 3: Add helper to ui_shared.py**

Near other display helpers in `src/ui_shared.py`:
```python
def render_position_filter(
    pool: "pd.DataFrame",
    key: str,
    default: str = "All",
    label: str = "Position",
) -> str:
    """Standard position selectbox: All / C / 1B / 2B / 3B / SS / OF / SP / RP.

    Section 5 helper extract (replaces 4 inline duplicates in pages).
    Returns the selected position; caller filters pool when != "All".
    """
    import streamlit as st
    positions = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
    return st.selectbox(label, options=positions, key=key, index=positions.index(default) if default in positions else 0)
```

- [ ] **Step 4: Run test — PASS**

- [ ] **Step 5: Replace 4 call sites**

Each call site currently looks like:
```python
positions = ["All", "C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]
position = st.selectbox("Position", positions, key="...")
```

Becomes:
```python
from src.ui_shared import render_position_filter
position = render_position_filter(pool=df, key="...")
```

(Apply to `pages/12_Trade_Finder.py:978`, `pages/14_Free_Agents.py:331`, `pages/20_Draft_Simulator.py:527`. For `pages/17_Leaders.py:342` — confirm shape compat; may need a separate variant call.)

- [ ] **Step 6: Run broader tests**

- [ ] **Step 7: Commit**

```bash
git commit -m "refactor(ui): Section 5 — extract render_position_filter helper

Four pages duplicated the same canonical [All, C, 1B, 2B, 3B, SS, OF, SP, RP]
selectbox. Extracted to src.ui_shared.render_position_filter. Replaces
inline impls in Trade_Finder, Free_Agents, Draft_Simulator, Leaders.

Adds tests/test_render_position_filter.py with mocked-streamlit unit tests.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Canonicalize weeks_remaining + fix dynamic_context default

**Files:**
- Modify: `src/validation/dynamic_context.py:32` (BONUS FIX: `FANTASY_REGULAR_SEASON_WEEKS = 22 → 26`)
- Modify: `src/league_rules.py` (add `weeks_remaining()` thin wrapper)
- Modify: `pages/5_Matchup_Planner.py:171`, `pages/12_Trade_Finder.py:941-942`, `pages/6_League_Standings.py:458`
- Create: `tests/test_weeks_remaining_canonical.py`

- [ ] **Step 1: TDD — write canonical-value tests**

```python
"""Section 5: weeks_remaining canonicalization + 22→26 default fix."""
from __future__ import annotations

from datetime import date
import pytest


def test_dynamic_context_default_is_26_not_22():
    """FourzynBurn is a 26-week H2H league; the old 22-week default was wrong."""
    from src.validation.dynamic_context import FANTASY_REGULAR_SEASON_WEEKS
    assert FANTASY_REGULAR_SEASON_WEEKS == 26, (
        "FANTASY_REGULAR_SEASON_WEEKS must be 26 (FourzynBurn) — see CLAUDE.md "
        "'Roster' section + LeagueConfig.season_weeks."
    )


def test_league_rules_weeks_remaining_matches_dynamic_context():
    from src.league_rules import weeks_remaining
    from src.validation.dynamic_context import compute_weeks_remaining
    # Same date, same season → same result.
    d = date(2026, 6, 15)
    assert weeks_remaining(as_of=d) == compute_weeks_remaining(as_of=d, season=2026)


def test_weeks_remaining_preseason_returns_26():
    from src.league_rules import weeks_remaining
    assert weeks_remaining(as_of=date(2026, 2, 1)) == 26


def test_weeks_remaining_postseason_returns_1():
    from src.league_rules import weeks_remaining
    assert weeks_remaining(as_of=date(2026, 11, 1)) == 1


def test_weeks_remaining_mid_season_decreasing():
    from src.league_rules import weeks_remaining
    early = weeks_remaining(as_of=date(2026, 4, 15))
    late = weeks_remaining(as_of=date(2026, 8, 15))
    assert early > late
    assert 1 <= late <= 26
```

- [ ] **Step 2: Run — expect FAIL (22 != 26 + ImportError on league_rules.weeks_remaining)**

- [ ] **Step 3: Fix dynamic_context default**

`src/validation/dynamic_context.py:32`:
```python
# OLD: FANTASY_REGULAR_SEASON_WEEKS = 22  # Typical Yahoo H2H
# FourzynBurn uses 26 weeks (LeagueConfig.season_weeks). Old 22 was a generic
# Yahoo H2H default that doesn't match this league.
FANTASY_REGULAR_SEASON_WEEKS = 26  # FourzynBurn — matches LeagueConfig.season_weeks
```

- [ ] **Step 4: Add wrapper to league_rules.py**

Append to `src/league_rules.py`:
```python
from datetime import date as _date


def weeks_remaining(as_of: _date | None = None, season: int = 2026) -> int:
    """Canonical weeks-remaining accessor (Section 5 consolidation).

    Thin wrapper around src.validation.dynamic_context.compute_weeks_remaining
    that pins season + uses LeagueConfig-aligned 26-week total. Use this
    everywhere instead of inline `(season_end - now).days // 7` formulas.
    """
    from src.validation.dynamic_context import compute_weeks_remaining
    from src.valuation import LeagueConfig
    return compute_weeks_remaining(
        as_of=as_of,
        season=season,
        total_weeks=LeagueConfig().season_weeks,
    )
```

- [ ] **Step 5: Replace 3 inline/alternate sites**

`pages/5_Matchup_Planner.py:171`:
```python
# OLD: weeks_remaining = max(1, config.season_weeks - week + 1)
from src.league_rules import weeks_remaining as _wr
weeks_remaining = _wr()
```

`pages/12_Trade_Finder.py:941-942`:
```python
# OLD:
# weeks_remaining = max(1, math.ceil((season_end - now).days / 7))
# weeks_remaining = min(weeks_remaining, config.season_weeks)
from src.league_rules import weeks_remaining as _wr
weeks_remaining = _wr()
```

`pages/6_League_Standings.py:458`:
```python
# OLD: weeks_default = estimate_weeks_remaining()
from src.league_rules import weeks_remaining as _wr
weeks_default = _wr()
```
(Keep the import of `estimate_weeks_remaining` only if used elsewhere in the same file; otherwise remove unused import.)

- [ ] **Step 6: Run tests — PASS**

- [ ] **Step 7: Broader regression**

```bash
python -m pytest --ignore=tests/test_cheat_sheet.py -q 2>&1 | tail -5
```
Note: tests that pinned the OLD 22-week default may now fail with 26. Update them to expect 26 (this IS the correctness fix).

- [ ] **Step 8: Commit**

```bash
git commit -m "refactor(weeks-remaining): Section 5 — canonical helper + fix 22→26 default

Three inline weeks_remaining formulas in pages diverged + a hidden 22-week
default in src/validation/dynamic_context.py (FANTASY_REGULAR_SEASON_WEEKS)
caused every compute_weeks_remaining() call without explicit total_weeks
to use the wrong horizon for FourzynBurn (26-week H2H).

Adds src.league_rules.weeks_remaining() as canonical wrapper sourcing
total_weeks from LeagueConfig.season_weeks. Replaces inline formulas
in Matchup_Planner, Trade_Finder, League_Standings.

Fixes a follow-on L11-class bug PR #33 missed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Section 4 — Dead-code residue

### Task 11: Delete ecr.py:976-997 commented block

**Files:** `src/ecr.py:976-997`

- [ ] **Step 1: Verify the block is truly commented-out code (not a docstring or active code)**

```bash
sed -n '970,1000p' src/ecr.py
```

- [ ] **Step 2: Confirm zero references to anything inside the block**

If the block defines anything (e.g. variable names), grep elsewhere:
```bash
grep -rn '<symbol>' src pages tests scripts --include='*.py' 2>&1 | head -5
```

- [ ] **Step 3: Delete lines 976-997 with Edit tool**

- [ ] **Step 4: Run targeted tests + ruff**

```bash
python -m pytest tests/test_ecr.py -q 2>&1 | tail -5
python -m ruff check src/ecr.py
```

- [ ] **Step 5: Commit**

```bash
git commit -m "chore(ecr): Section 4 — delete 20-line commented-out block at ecr.py:976-997"
```

---

### Task 12: F841 sweep — verify zero residual

- [ ] **Step 1: Run scan**

```bash
python -m ruff check --select F841 . 2>&1 | tee /tmp/f841_scan.txt | tail -20
```

- [ ] **Step 2: If zero — no commit needed; record in PR body**

- [ ] **Step 3: If any residual — delete the dead vars (small edits per file)**

Then commit with summary like:
```bash
git commit -m "chore: Section 4 — F841 residual cleanup (<N> sites)"
```

---

### Task 13: Fresh orphan audit (via Explore subagent)

- [ ] **Step 1: Dispatch Explore agent (very thorough) for src/ only**

Prompt the agent to find:
- Functions in src/ that are not called by any other code in src/ + pages/ + tests/ + scripts/ + app.py
- Module-level constants that are not imported anywhere
- Particularly: any functions newly orphaned by D2/D4/D6 dedups in this PR

- [ ] **Step 2: For each candidate — verify via grep**

```bash
grep -rn '<name>' src pages tests scripts app.py 2>&1 | grep -v "def <name>\|src/<file>.py:.*<name>" | head -5
```
Zero non-self refs = safe to delete.

- [ ] **Step 3: Delete in small batches (≤200 LOC per commit)**

```bash
git commit -m "chore: Section 4 — fresh orphan sweep (<list>, ~<N> LOC)"
```

---

## Section 6 — Docs refresh

### Task 14: Section 6 docs refresh — architecture.md + Research.md + README.md + archive punchlist + archive VERIFICATION_LOG + archive shipped specs

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/Research.md`
- Modify: `README.md`
- Move: `docs/2026-05-17-deep-audit-punchlist.md` → `docs/archive/2026-05-17-deep-audit-punchlist.md`
- Move: `docs/VERIFICATION_LOG.md` → `docs/archive/VERIFICATION_LOG.md`
- Move: shipped specs in `docs/superpowers/specs/` → `docs/archive/specs/`
- Modify: `CLAUDE.md` (update punchlist reference path)

**Note on Section 6 housekeeping deletions:** None needed. Verified 2026-05-19 that `League Group Chat Screenshots/` is already gone; `data/draft_tool.db` / `data/logs/bootstrap.log` / `.pytest_cache/README.md` are all already gitignored via `data/*` and `.pytest_cache/` rules and not tracked. No deletions required.

- [ ] **Step 1: Refresh architecture.md**

```bash
wc -l docs/architecture.md
grep -n "pages\|phases\|test" docs/architecture.md | head -30
```

Update sections:
- Page table → 13 pages per current CLAUDE.md File Structure
- Bootstrap pipeline → 33 phases with 3-tier waterfall description
- Test count → current count from Phase 100 (was 3947 at baseline)
- Trade engine → V4 (6-phase pipeline; refs `src/engine/output/trade_evaluator.evaluate_trade`)
- Sub-package list → reflect current `src/engine/` + `src/optimizer/` layout

- [ ] **Step 2: Refresh Research.md (Gap → Already Has)**

```bash
gh pr list --state merged --limit 50 > /tmp/merged_prs.txt
grep -n "^## Gap\|^## Already Has\|^- " docs/Research.md | head -40
```

For each PR shipped in the past 6 months, check whether its functional area is still listed under "Gap" and move it to "Already Has" with a citation to the merging PR.

- [ ] **Step 3: Refresh README.md**

Current README likely says "20 pages, ~3,900 tests". Update:
- Pages: 20 → 13 (per Section 5 consolidation in PR #45)
- Tests: current pass count from Phase 100
- Trade engine: V4 (HCV-Hybrid, 6-phase; cite `docs/2026-05-19-hcv-hybrid-trade-evaluation-spec.md`)
- CI: Python 3.12 only (4-shard pytest, ruff lint+format, 60% coverage floor)
- Architecture/setup links → updated paths

- [ ] **Step 4: Archive shipped specs**

```bash
mkdir -p docs/archive/specs
ls docs/superpowers/specs/ 2>/dev/null
```
Move any spec whose feature is shipped (per CLAUDE.md "Audit History" or recent PRs). Keep specs for in-flight work (e.g., HCV-Hybrid if not yet implemented) in `docs/superpowers/specs/`.

- [ ] **Step 5: Archive punchlist + VERIFICATION_LOG**

```bash
git mv docs/2026-05-17-deep-audit-punchlist.md docs/archive/2026-05-17-deep-audit-punchlist.md
[ -f docs/VERIFICATION_LOG.md ] && git mv docs/VERIFICATION_LOG.md docs/archive/VERIFICATION_LOG.md
```

Update `CLAUDE.md` "Audit History" section to reference the archived paths.

- [ ] **Step 6: Commit (one per logical group)**

```bash
git commit -m "docs: Section 6 — refresh architecture.md (13 pages, current test count, 33 phases, V4 engine)"
git commit -m "docs: Section 6 — Research.md move shipped items to Already Has"
git commit -m "docs: Section 6 — README.md refresh (page count, test count, CI, trade engine)"
git commit -m "docs: Section 6 — archive deep-audit punchlist + VERIFICATION_LOG + shipped specs"
```

---

## Section 7 — CLAUDE.md local-path drift

### Task 15: Fix HEATER_v1.0.0 LOCAL paths in CLAUDE.md (leave OneDrive WARNING + GitHub URLs alone)

**Files:** `CLAUDE.md`

- [ ] **Step 1: Inventory the matches**

```bash
grep -n "HEATER_v1\.0\.0" CLAUDE.md
```

- [ ] **Step 2: Classify each**

For each line decide: GitHub URL (LEAVE), OneDrive WARNING (LEAVE), local path (REPLACE).

- [ ] **Step 3: Apply targeted Edit replacements**

Examples:
- `Project root (local):` line: `HEATER_v1.0.0` → `HEATER_v1.0.1`
- "Resume Checklist" step 1: path → `HEATER_v1.0.1`

- [ ] **Step 4: Verify GitHub URLs untouched**

```bash
grep -n "github.com/hicklax13/HEATER_v1.0.0\|hicklax13/HEATER_v1.0.0\.git" CLAUDE.md
```
Expected: matches preserved.

- [ ] **Step 5: Commit**

```bash
git commit -m "docs(claude.md): Section 7 — align local-path refs to HEATER_v1.0.1"
```

---

## Section 8 — Repo-wide stale folder-path scrub

### Task 16: Path scrub + __pycache__ purge + structural-invariant guard

**Files:** any file with `HEATER_v1.0.0` as LOCAL path; all `__pycache__/`

- [ ] **Step 1: Inventory**

```bash
git grep -nI "HEATER_v1\.0\.0" -- ':!*.lock' ':!*.png' ':!*.jpg' ':!*.db' ':!*.sqlite*' > /tmp/path_inventory.txt
wc -l /tmp/path_inventory.txt
git grep -nI "OneDrive.Desktop.HEATER" >> /tmp/path_inventory.txt
```

- [ ] **Step 2: Hidden-spot grep (files git-grep may miss)**

```bash
for f in .streamlit/config.toml .streamlit/secrets.toml .env .env.local \
         pyproject.toml setup.py setup.cfg tox.ini pytest.ini \
         requirements.txt CLAUDE.md README.md AGENTS.md GEMINI.md \
         .claude/settings.json .claude/settings.local.json; do
  [ -f "$f" ] && grep -Hn "HEATER_v1.0.0\|OneDrive" "$f" 2>/dev/null
done
grep -rn "HEATER_v1.0.0" .github/ .streamlit/ scripts/ 2>/dev/null | head -20
```

- [ ] **Step 3: Classify + apply per-file Edit replacements**

GitHub URL = LEAVE. OneDrive WARNING block in CLAUDE.md = LEAVE. Everything else = REPLACE local path → `HEATER_v1.0.1` (or `C:\Users\conno\Code\HEATER_v1.0.1`).

- [ ] **Step 4: Purge __pycache__ (root cause of pyc-traceback OneDrive leak)**

```bash
find . -name "__pycache__" -type d -not -path "./.claude/worktrees/*" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -not -path "./.claude/worktrees/*" -delete 2>/dev/null
```

- [ ] **Step 5: Verify inventory shows ONLY GitHub URL + CLAUDE.md OneDrive WARNING block**

```bash
git grep -nI "HEATER_v1\.0\.0" -- ':!*.lock' ':!*.png' ':!*.db' ':!*.sqlite*'
```

- [ ] **Step 6: Add structural-invariant guard**

`tests/test_no_stale_folder_path.py`:
```python
"""Section 8: no stale HEATER_v1.0.0 LOCAL folder-path references.

Allowed:
  - GitHub repo URL (repo NAME is HEATER_v1.0.0; cannot rename)
  - CLAUDE.md OneDrive incompatibility WARNING block (intentional historical note)
  - docs/archive/ (historical docs)
  - .claude/worktrees/ (ephemeral agent dirs, gitignored)
"""
from __future__ import annotations

from pathlib import Path
import re

GITHUB_URL_PATTERNS = (
    re.compile(r"github\.com/hicklax13/HEATER_v1\.0\.0"),
    re.compile(r"hicklax13/HEATER_v1\.0\.0(?:\.git)?"),
)
ONEDRIVE_WARN_MARKER = "OneDrive's Cloud Files API"   # appears once in CLAUDE.md's intentional warning block


def _is_allowed(line: str, in_onedrive_block: bool) -> bool:
    if in_onedrive_block:
        return True
    for p in GITHUB_URL_PATTERNS:
        if p.search(line):
            return True
    return False


def test_no_stale_local_folder_path():
    root = Path(".")
    targets = []
    for p in ("src", "pages", "scripts", ".github", ".streamlit"):
        if (root / p).exists():
            targets.extend((root / p).rglob("*"))
    for f in (root / "docs").rglob("*"):
        if "archive" in f.parts:
            continue
        targets.append(f)
    # Also CLAUDE.md, README.md
    for name in ("CLAUDE.md", "README.md", "AGENTS.md", "GEMINI.md"):
        p = root / name
        if p.exists():
            targets.append(p)

    offenders: list[str] = []
    for f in targets:
        if not f.is_file():
            continue
        # skip binary-ish
        if f.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".db", ".sqlite", ".pyc", ".lock"}:
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        if "HEATER_v1.0.0" not in text:
            continue
        # Walk line-by-line; track CLAUDE.md OneDrive WARNING block scope
        in_warn_block = False
        for lineno, line in enumerate(text.splitlines(), start=1):
            if ONEDRIVE_WARN_MARKER in line:
                in_warn_block = True
            # Block ends at next top-level header in CLAUDE.md or empty marker
            if in_warn_block and line.startswith("## "):
                in_warn_block = False
            if "HEATER_v1.0.0" in line and not _is_allowed(line, in_warn_block):
                offenders.append(f"{f.relative_to(root)}:{lineno}: {line.strip()[:120]}")
    assert not offenders, (
        "Stale HEATER_v1.0.0 LOCAL folder-path reference found. "
        "Local folder is HEATER_v1.0.1; only the GitHub repo NAME stays as v1.0.0.\n  "
        + "\n  ".join(offenders)
    )
```

- [ ] **Step 7: Run guard — PASS**

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore(paths): Section 8 — scrub stale HEATER_v1.0.0 local-path refs + invariant guard

- Replace HEATER_v1.0.0 LOCAL folder paths → HEATER_v1.0.1 across
  src/, pages/, scripts/, .github/, .streamlit/, docs/ (excluding archive/),
  CLAUDE.md, README.md.
- LEAVE GitHub URLs (github.com/hicklax13/HEATER_v1.0.0,
  hicklax13/HEATER_v1.0.0.git) — repo NAME unchanged.
- LEAVE CLAUDE.md OneDrive incompatibility WARNING block.
- Purge __pycache__/ (root cause of pyc files retaining absolute OneDrive
  paths in tracebacks; tests/test_war_room.py traceback leak fixed).
- Add tests/test_no_stale_folder_path.py structural guard.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Phase 50 — PR open + CI monitor + merge

### Task 17: Push + draft PR + CI watch + merge

- [ ] **Step 1: Push**

```bash
git push -u origin claude/finish-deep-audit-punchlist
```

- [ ] **Step 2: Open draft PR**

```bash
gh pr create --draft --base master --title "deep-audit punchlist completion + repo hygiene + preflight tooling" --body "$(cat <<'EOF'
## Summary

Completes the remaining work in `docs/2026-05-17-deep-audit-punchlist.md` (Sections 2-7) plus two new sections:

- **Section 8** — stale `HEATER_v1.0.0` local-folder-path scrub (post-relocation cleanup from PR #46)
- **Section 9** — new `scripts/preflight.py` env+access verification tool

### Commits

[git log --oneline master..HEAD]

### Highlights per section

- **Section 2 (logic bugs):** L8 + L14 cleanup. L3 + L7 resolved per user design decision (see commits).
- **Section 3 (dedup):** D2 normalize_player_name canonicalized; D4 INVERSE_CATS literals → LeagueConfig; D6 RATE_STATS sets → LeagueConfig. Three new structural-invariant guards.
- **Section 4 (dead code):** ecr.py:976-997 block deleted; F841 residual zeroed; <N> additional orphan functions deleted (~<LOC> lines).
- **Section 5 (helpers):** `render_position_filter()` extracted to ui_shared (4 call sites); `weeks_remaining()` canonicalized in league_rules. Bonus: fixed 22-week default in dynamic_context.
- **Section 6 (docs):** architecture.md + Research.md refreshed; punchlist archived.
- **Section 7 (CLAUDE.md):** local-path refs aligned to HEATER_v1.0.1.
- **Section 8 (path scrub):** repo-wide; `__pycache__` purge fixes stale-path tracebacks.
- **Section 9 (preflight):** scripts/preflight.py shipped.

### Baseline pytest delta

Before: 3947 passing, 7 failing (pre-existing), 17 skipped.
After: <update before flipping to ready>.

### Pre-existing failures (not regressions)

- tests/test_data_freshness.py::TestDataFreshnessTracker::test_get_all_freshness
- tests/test_player_databank.py::TestGameLogsSchema::test_game_logs_primary_key
- tests/test_sf_team_strength_freshness.py::test_team_strength_freshness_NOT_recorded_when_empty
- tests/test_unified_weights.py::TestGracefulFallback::test_no_data_returns_equal_weights
- tests/test_unified_weights.py::TestGracefulFallback::test_matchup_mode_no_urgency_returns_equal
- tests/test_war_room.py::TestComputeHotColdReport::test_hot_pitcher_detected
- tests/test_war_room.py::TestComputeHotColdReport::test_cold_pitcher_detected

### Design pause outcomes

- **L3:** [chosen option + 1-sentence rationale]
- **L7:** [chosen option + 1-sentence rationale]

### Test plan

- [ ] Ruff lint + format clean
- [ ] All 4 CI shards pass
- [ ] Coverage floor (60%) maintained
- [ ] New structural-invariant guards pass

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Watch CI**

```bash
gh pr checks $(gh pr view --json number --jq .number) --watch
```

- [ ] **Step 4: Fix any failure with NEW commits (never --no-verify / --amend)**

If failure: read log, identify root cause, fix, commit, repush. Re-watch.

- [ ] **Step 5: Confirm green, then mark ready + squash-merge**

```bash
gh pr checks <PR#>            # confirm all green
gh pr ready <PR#>
gh pr merge <PR#> --squash --delete-branch
```

- [ ] **Step 6: Local cleanup**

```bash
git switch master
git pull --ff-only origin master
git branch -d claude/finish-deep-audit-punchlist
```

---

## Phase 99 — Interactive branch + worktree cleanup

### Task 18: Branch / worktree audit + per-branch confirmation + archive-and-delete

- [ ] **Step 1: Git worktree audit**

```bash
git worktree list
```
For each non-main: `git -C <path> log --oneline -5 && git -C <path> status --porcelain`.

Filesystem `.claude/worktrees/` (gitignored but disk consumers):
```bash
ls -la .claude/worktrees/ | head -20
```

- [ ] **Step 2: AskUserQuestion per non-empty git worktree**

Options: Keep / Remove (after archive tag) / Force-remove (lose changes).

- [ ] **Step 3: Branch inventory**

```bash
git branch -a --no-color
git for-each-ref --format='%(refname:short)|%(objectname:short)|%(authordate:short)|%(subject)' refs/heads refs/remotes/origin > /tmp/branch_inventory.txt
```

Compose table grouped by prefix:
- `claude/*` — completed-PR branches → likely safe
- `worktree-agent-*` — subagent leftovers → likely safe
- `pr-*` / `wave*-review` — manual review branches → confirm individually
- `feat/*` — possible WIP → confirm individually

- [ ] **Step 4: AskUserQuestion per category**

Options: "Apply default action to ALL" / "Review each" / "Skip category".

- [ ] **Step 5: For each approved branch: tag → delete**

```bash
git tag -f archive/<branch-name> <branch-name>
git rev-parse archive/<branch-name>            # verify
git branch -d <branch-name> 2>/dev/null || git branch -D <branch-name>    # force only AFTER tagging
git push origin --delete <branch-name>
```

- [ ] **Step 6: Push archive tags once at end**

```bash
git push origin --tags
```

- [ ] **Step 7: Sweep .claude/worktrees/ filesystem leftovers**

```bash
rm -rf .claude/worktrees/agent-* .claude/worktrees/*-* 2>/dev/null
```

- [ ] **Step 8: Final verification**

```bash
git branch -a --no-color                       # only master + origin/master
gh api repos/hicklax13/HEATER_v1.0.0/branches --jq '.[].name'    # only "master"
git tag --list "archive/*" | wc -l             # log count
git worktree list                              # only main
```

If any non-master remains AND user did not explicitly opt-keep: re-prompt.

---

## Phase 100 — Post-merge environment re-verify

### Task 19: Re-run preflight + tests + streamlit smoke

- [ ] **Step 1: Preflight**

```bash
python scripts/preflight.py
```
Expected: exit 0 or 2 (same warnings as Phase 0; no new failures).

- [ ] **Step 2: Pytest delta vs baseline**

```bash
python -m pytest --ignore=tests/test_cheat_sheet.py -q 2>&1 | tail -5
```
Expected: pass count ≥ 3947 (baseline), no new failures beyond the 7 pre-existing.

- [ ] **Step 3: Streamlit smoke**

```bash
streamlit run app.py &
STREAMLIT_PID=$!
sleep 8
# expect: 'You can now view your Streamlit app' in output, no ImportError/ModuleNotFoundError
kill $STREAMLIT_PID 2>/dev/null
```

- [ ] **Step 4: GitHub auth final check**

```bash
gh auth status
gh api user --jq .login                        # confirm hicklax13
```

- [ ] **Step 5: Final report to user**

Single message with:
- PR URL + total LOC delta + total commits + structural-invariant tests passing count
- Archived branch count + only-master-remains confirmation
- No-stale-path confirmation
- L3/L7 decisions documented (or deferred)
- Tokens-OK / deps-OK / tests-OK (delta) / streamlit-OK / gh-auth-OK

---

## Self-Review (already performed before saving)

**Spec coverage check** — every Section 2/3/4/5/6/7/8/9 item from the user mission maps to a task:
- ✅ Section 2 L8 → Task 2
- ✅ Section 2 L14 → Task 3
- ✅ Section 2 L3 → Task 4 (with design pause)
- ✅ Section 2 L7 → Task 5 (with design pause)
- ✅ Section 3 D2 → Task 6
- ✅ Section 3 D4 → Task 7
- ✅ Section 3 D6 → Task 8
- ✅ Section 5 render_position_filter → Task 9
- ✅ Section 5 weeks_remaining (+ bonus dynamic_context fix) → Task 10
- ✅ Section 4 ecr.py block → Task 11
- ✅ Section 4 F841 sweep → Task 12
- ✅ Section 4 orphan audit → Task 13 (Explore agent)
- ✅ Section 6 docs → Task 14
- ✅ Section 7 CLAUDE.md → Task 15
- ✅ Section 8 path scrub + __pycache__ + guard → Task 16
- ✅ Section 9 preflight commit → Task 1
- ✅ Phase 50 PR + CI + merge → Task 17
- ✅ Phase 99 branch cleanup → Task 18
- ✅ Phase 100 re-verify → Task 19

**Placeholders:** None. All steps have concrete commands or code blocks. Design-pause tasks (4, 5) intentionally show all options without pre-deciding — that's the *spec*, not a placeholder.

**Type consistency:** `normalize_player_name` signature (`name: object) -> str`) consistent across Task 6 test + impl. `weeks_remaining` returns `int` everywhere. `render_position_filter` returns `str` everywhere.
