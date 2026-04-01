# Post-Debug Cleanup & Hardening Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 9 remaining MEDIUM bugs from the systematic debug, update CLAUDE.md, and verify the app works end-to-end with live Yahoo data.

**Architecture:** Three parallel workstreams: (1) bug fixes across optimizer and trade modules, (2) CLAUDE.md documentation update, (3) live app verification via browser testing.

**Tech Stack:** Python, Streamlit, PuLP, pandas, pytest, Playwright (browser testing)

---

## Workstream Overview

```
Agent 1 — Bug Fixer: Optimizer Modules        ─┐
  (ALP-001, ALP-002, OP-006, OP-011, LP-004)   │
                                                 ├─→ Test Suite → Commit → Push
Agent 2 — Bug Fixer: Trade Modules             ─┤
  (OTA-001, OTA-002, TF-005, TF-009)            │
                                                 │
Agent 3 — Docs: Update CLAUDE.md               ─┤
  (48 bugs fixed, acceptance panel, page audit)  │
                                                 │
Agent 4 — Verify: Live App Testing             ─┘
  (Trade Analyzer acceptance panel, Trade Finder,
   Lineup Optimizer daily tab, My Team page)
```

---

## Task 1: Fix Optimizer MEDIUM Bugs (Agent 1)

**Files:**
- Modify: `src/optimizer/advanced_lp.py`
- Modify: `src/optimizer/pipeline.py`
- Modify: `src/optimizer/matchup_adjustments.py`
- Test: `tests/test_optimizer_advanced_lp.py`, `tests/test_optimizer_pipeline.py`, `tests/test_optimizer_matchups.py`

### Bug ALP-001: Dead code branch for inverse cats in maximin

- [ ] **Step 1: Locate the dead branch**

In `maximin_lineup()`, line 169 filters out inverse cats:
```python
cats = [c for c in cats if c not in INVERSE_CATS]
```
Lines 200-209 contain an unreachable `if cat in INVERSE_CATS:` branch.

- [ ] **Step 2: Remove dead code**

Delete the unreachable inverse-cat branch (lines 200-209). Add a comment explaining why:
```python
# Note: inverse cats (L, ERA, WHIP) excluded at line 169.
# Their SGP scale factors are incompatible with counting stats in maximin.
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_optimizer_advanced_lp.py -x -q`
Expected: All pass (dead code removal has no functional impact)

### Bug ALP-002: primary_value missing weight multiplier

- [ ] **Step 4: Fix primary_value computation**

In `epsilon_constraint_lineup()`, find the post-solve computation:
```python
raw_primary = sum(float(primary_vals[i]) for i in started)
primary_value = raw_primary / sf_primary
```
Change to include weight:
```python
raw_primary = sum(float(primary_vals[i]) for i in started)
primary_value = raw_primary * w_primary / sf_primary
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_optimizer_advanced_lp.py -x -q`
Expected: All pass

### Bug OP-006: Risk adjustments assume 0-based index

- [ ] **Step 6: Reset index before risk adjustments**

In `_solve_lineup()`, after `adjusted_roster = roster.copy()`, add:
```python
adjusted_roster = adjusted_roster.reset_index(drop=True)
```

- [ ] **Step 7: Run tests**

Run: `python -m pytest tests/test_optimizer_pipeline.py -x -q`
Expected: All pass

### Bug OP-011: Empty park_factors dict blocks Stage 2

- [ ] **Step 8: Fix park_factors truthiness check**

In `optimize()`, find the Stage 2 condition:
```python
if self._preset["enable_matchups"] and _MATCHUP_AVAILABLE and week_schedule and park_factors:
```
Change `park_factors` to `park_factors is not None`:
```python
if self._preset["enable_matchups"] and _MATCHUP_AVAILABLE and week_schedule and park_factors is not None:
```

- [ ] **Step 9: Run tests**

Run: `python -m pytest tests/test_optimizer_pipeline.py -x -q`
Expected: All pass

### Bug LP-004: Pitchers always get neutral 1.0 park factor

- [ ] **Step 10: Apply dampened park factor for pitchers**

In `park_factor_adjustment()`, find:
```python
if not is_hitter:
    return 1.0
```
Change to:
```python
if not is_hitter:
    # Dampen park factor effect for pitchers (rate stats partially captured)
    return 1.0 + (pf - 1.0) * 0.3  # 30% of hitter park effect
```

- [ ] **Step 11: Update tests for pitcher park factors**

Any test that asserts `park_factor_adjustment(..., is_hitter=False) == 1.0` needs updating.
The new expected value for a Coors pitcher (pf=1.38) is: `1.0 + (1.38-1.0)*0.3 = 1.114`.

- [ ] **Step 12: Run tests and commit**

Run: `python -m pytest tests/test_optimizer_matchups.py tests/test_optimizer_advanced_lp.py tests/test_optimizer_pipeline.py -x -q`
Expected: All pass

```bash
git add src/optimizer/advanced_lp.py src/optimizer/pipeline.py src/optimizer/matchup_adjustments.py tests/
git commit -m "fix: 5 medium optimizer bugs (ALP-001, ALP-002, OP-006, OP-011, LP-004)"
```

---

## Task 2: Fix Trade Module MEDIUM Bugs (Agent 2)

**Files:**
- Modify: `src/opponent_trade_analysis.py`
- Modify: `src/trade_finder.py`
- Test: `tests/test_opponent_trade_analysis.py`, `tests/test_trade_finder.py`

### Bug OTA-001: config param accepted but never used

- [ ] **Step 1: Remove unused config parameter**

In `compute_opponent_needs()`, remove the `config` parameter since `category_gap_analysis()` uses its own module-level config. Update the docstring.

Also update any callers that pass `config` (check `src/trade_finder.py` line 585).

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_opponent_trade_analysis.py -x -q`
Expected: All pass

### Bug OTA-002: Inverse-stat delta sign flip inconsistent

- [ ] **Step 3: Fix sign convention in analyze_from_opponent_view()**

Store raw delta in `opp_category_deltas`, use flipped value only for classification:
```python
raw_delta = after_val - before_val
improvement = -raw_delta if cat in config.inverse_stats else raw_delta
deltas[cat] = round(raw_delta, 3)  # raw stat change for display
if rank >= 8 and improvement > 0:
    weak_helped.append(cat)
elif rank <= 4 and improvement < 0:
    strong_hurt.append(cat)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_opponent_trade_analysis.py -x -q`
Expected: All pass (or update tests for new sign convention)

### Bug TF-005: ERA/WHIP of 0.00 not counted as contributing

- [ ] **Step 5: Fix rate stat zero check**

In `_count_contributing_categories()`, change:
```python
if cat in config.inverse_stats:
    if val > 0 and val < benchmark * 1.5:
        count += 1
```
To:
```python
if cat in config.inverse_stats:
    ip = float(player_row.get("ip", 0) or 0)
    if cat in ("ERA", "WHIP") and ip <= 0:
        continue  # No IP = doesn't contribute to pitching rates
    if val < benchmark * 1.5:
        count += 1
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_trade_finder.py -x -q`
Expected: All pass

### Bug TF-009: 1-for-1 and 2-for-1 composite scores incomparable

- [ ] **Step 7: Normalize 2-for-1 composite score**

In `scan_2_for_1()`, remove `ROSTER_SPOT_SGP` from the composite formula and add it as a fixed bonus:
```python
user_delta_for_composite = user_delta - ROSTER_SPOT_SGP
composite = (
    0.30 * user_delta_for_composite
    + 0.15 * adp_fairness * 2.0
    + 0.15 * 0.5 * 2.0  # ECR placeholder (neutral for 2-for-1)
    + 0.20 * p_accept * 3.0
    + 0.10 * max(opp_delta, 0)
    + 0.10 * need_match * 2.0
) + 0.1  # Small fixed bonus for roster flexibility
```

- [ ] **Step 8: Run tests and commit**

Run: `python -m pytest tests/test_trade_finder.py tests/test_opponent_trade_analysis.py -x -q`
Expected: All pass

```bash
git add src/opponent_trade_analysis.py src/trade_finder.py tests/
git commit -m "fix: 4 medium trade bugs (OTA-001, OTA-002, TF-005, TF-009)"
```

---

## Task 3: Update CLAUDE.md (Agent 3)

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Update test count and file counts**

Update:
- Test count: verify with `python -m pytest --co -q` and update the number
- Test file count: verify with `ls tests/test_*.py | wc -l`

- [ ] **Step 2: Add systematic debug summary**

Add to the "Key Fixes" section or create a new "Key Fixes (April 1, 2026)" section:
```markdown
## Key Fixes (April 1, 2026)

- **Systematic Debug** — 9 parallel agents audited 16 files, found 49 bugs (1 critical, 16 high, 24 medium, 8 low)
- **TI-001 CRITICAL:** ERA/WHIP recalculated with unreduced er/bb_allowed/h_allowed — injured pitchers got worse rate stats
- **TI-002 HIGH:** Health score double-dip: IL15 players lost 36% instead of 16%
- **TI-003 HIGH:** "C" in "CF" substring match gave center fielders catcher scarcity premium
- **TF-001/TF-002 HIGH:** 2-for-1 drop cost always zero — opponent roster squeeze never penalized
- **OP-001 HIGH:** Statcast step called with player name instead of mlb_id — entire step was dead code
- **CU-001 HIGH:** math.exp() overflow on extreme WHIP gaps crashed optimizer
- **DO-001 HIGH:** Arizona="AZ"/Oakland="ATH" abbreviations broke park factor lookups
- **Acceptance Analysis Panel** — Trade Analyzer now shows ADP Fairness, ECR Fairness, Acceptance Probability, Acceptance Tier
- **Page Audit** — 6 page bugs fixed (health_dict consistency, Start/Sit schedule, pitcher detection, rate stat formatting)
```

- [ ] **Step 3: Update current session state**

Update the "Season State" section if anything changed (e.g., new test count).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with systematic debug session (49 bugs, acceptance panel)"
```

---

## Task 4: Live App Verification (Agent 4 — Browser Testing)

**Prerequisites:** App must be running (`streamlit run app.py`)

- [ ] **Step 1: Launch app and verify Yahoo connection**

Navigate to app, verify auto-reconnect works, check data freshness.

- [ ] **Step 2: Test Trade Analyzer acceptance panel**

1. Navigate to Trade Analyzer page
2. Select a giving player from your roster
3. Select a receiving player from another team
4. Click "Analyze Trade"
5. Verify the Acceptance Analysis section appears with 4 metrics:
   - Acceptance Probability (percentage)
   - ADP Fairness (percentage)
   - ECR Fairness (percentage)
   - Acceptance Tier (High/Medium/Low)
6. Verify ADP/ECR detail expander works

- [ ] **Step 3: Test Trade Finder**

1. Navigate to Trade Finder page
2. Check By Partner tab — verify trades show with new columns (ADP, ECR, opponent needs)
3. Check By Value tab — verify composite scores look reasonable
4. Check Trade Readiness tab — verify pitchers appear (TI-005 fix)

- [ ] **Step 4: Test Lineup Optimizer daily tab**

1. Navigate to Lineup Optimizer page
2. Select Daily Optimize tab
3. Verify DCV scores are differentiated (not all identical — DO-002 fix)
4. Verify Arizona/Oakland players show game status (not always off-day — DO-001 fix)
5. Verify park factors are applied (LP-001 fix)

- [ ] **Step 5: Test My Team page**

1. Navigate to My Team page
2. Verify AVG/OBP show 3 decimal places (.260 not .26)
3. Verify ERA/WHIP show 3 decimal places (3.500 not 3.50)
4. Verify IP Watch counts multi-position pitchers (MT-001 fix)

- [ ] **Step 6: Test Player Compare**

1. Navigate to Player Compare page
2. Compare two players
3. Verify health badges reflect IL/DTD status (not just raw injury history)

- [ ] **Step 7: Document any issues found**

If any issues found, log them and create follow-up fixes.

---

## Execution Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                    PARALLEL PHASE (20 min)                       │
│                                                                  │
│  Agent 1: Fix 5 optimizer bugs ──────────────────────┐          │
│  Agent 2: Fix 4 trade module bugs ──────────────────┤          │
│  Agent 3: Update CLAUDE.md ─────────────────────────┘          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    MERGE PHASE (5 min)                           │
│                                                                  │
│  Run full test suite (pytest -x -q)                             │
│  Run lint + format (ruff check . && ruff format --check .)      │
│  Commit all changes                                              │
│  Push to master                                                  │
│  Verify CI passes                                                │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                    VERIFY PHASE (10 min)                         │
│                                                                  │
│  Agent 4: Launch app + browser test all 5 pages                 │
│  Document any issues                                             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Total estimated time:** 35 minutes with parallel agents

**Agent assignment:**
- Agent 1: `general-purpose` — optimizer module fixes
- Agent 2: `general-purpose` — trade module fixes
- Agent 3: `general-purpose` — CLAUDE.md documentation
- Agent 4: Launch app manually, then use Claude in Chrome for browser testing
