# FA Engine Overhaul — Implementation Plan

**Date**: 2026-05-20
**Trigger**: Crochet/Kirk bad recommendation. 4 research agents confirmed systemic engine issues.
**Scope**: All 4 phases sequential (~20-25h total). Strict IL protection.
**Status**: PENDING USER APPROVAL before any P1 code is touched.

---

## Executive summary

The Free Agents page (`pages/14_Free_Agents.py`) calls `waiver_wire.compute_add_drop_recommendations` — an older engine missing nearly every feature published industry best-practice expects (opponent context, IL handling, stat-window blending, position scarcity, return-date scaling). A NEWER engine, `src/optimizer/fa_recommender.py::recommend_fa_moves`, already exists with most of these features — but it's only wired to the Lineup Optimizer page.

**This plan**:
1. Rewires the Free Agents page to use `fa_recommender.recommend_fa_moves` (P1)
2. Fixes the root-cause IL-zeroing bug that survives the rewire (P1)
3. Adds the missing features both engines lack — stat-window blending, position scarcity, return-date scaling, dynamic IL stash list, regression-flag wiring, sustainability rewrite, composite formula fix, rate-stat volume weighting (P2 + P3)
4. Cleans up 24 unjustified magic constants (P3)
5. Ships cosmetic polish (P4)

After all 4 phases the engine will be aligned with the published methodology of FantasyPros, Razzball, RotoWire, and Pitcher List — and the Crochet/Kirk class of bug will be eliminated at the root.

**Realism note**: "best possible algorithm" is not the goal — that's a multi-year research program. The goal is "aligned with industry best-practice and free of the bad-recommendation bug class." That is achievable.

---

## Architecture overview — what changes

```
BEFORE:
  pages/14_Free_Agents.py
    -> compute_add_drop_recommendations    [waiver_wire.py — OLDER ENGINE]
        -> _roster_category_totals          [in_season.py — ZEROES IL PLAYERS]
        -> compute_drop_cost                [waiver_wire.py — no scarcity]
        -> compute_net_swap_value           [waiver_wire.py — no urgency]
        -> compute_sustainability_score     [waiver_wire.py — 5-bucket step]

  pages/2_Line-up_Optimizer.py
    -> recommend_fa_moves                  [fa_recommender.py — NEWER ENGINE, but missing features]
        -> compute_category_urgency        [category_urgency.py — well-designed]
        -> sustainability + base value     [fa_recommender.py — additive urgency bug]


AFTER:
  pages/14_Free_Agents.py     -+
                                |--> recommend_fa_moves (NEW canonical engine)
  pages/2_Line-up_Optimizer.py -+
                                |
                                v
                  fa_recommender.recommend_fa_moves
                          |
                          +-- OptimizerDataContext (existing shared layer)
                          +-- IL-stash filter (strict — never drop rostered IL)
                          +-- _roster_category_totals — IL players keep their projection × return-date scalar
                          +-- compute_vorp positional scarcity (existing in valuation.py — wired in)
                          +-- stat-window blend (0.10·L14 + 0.20·YTD + 0.70·ROS)
                          +-- composite formula fix (urgency as multiplier, not additive)
                          +-- sustainability rewrite (xwOBA-wOBA + Stuff+/FIP, not BABIP-only)
                          +-- dynamic IL stash list (from rosters.status, not hardcoded names)
                          +-- regression-flag scalar (existing column, never read — wire it in)
                          +-- rate-stat volume weighting
                          +-- 24 magic constants -> constants_registry.py with citations
```

---

## Phase 1 — Critical fixes (PR1, PR2, PR3 — ~8-10h)

Goal: kill the Crochet/Kirk class of bug. Single user-facing outcome: top-30 SP on IL15 never gets recommended for drop.

### PR1: Wire Free Agents page to the newer `recommend_fa_moves` engine (~3-4h)

**Files modified**:
- `pages/14_Free_Agents.py` — replace `compute_add_drop_recommendations` call with `recommend_fa_moves(ctx)` via `OptimizerDataContext`
- New helper in `pages/14_Free_Agents.py` or `src/optimizer/shared_data_layer.py` to translate `recommend_fa_moves` output into the existing UI shape (preserving column names so the Markdown tables don't break)

**Decisions baked in**:
- `OptimizerDataContext` is built the same way the Lineup Optimizer page does — single source of truth for roster_ids, player_pool, urgency weights, IL list
- Strict IL policy (per user direction): IL stash list includes ANY rostered player with `status IN (IL10, IL15, IL60, IL, DTD)`. Never recommend dropping any of them. UI shows them as "PROTECTED" with the existing banner.
- The old `compute_add_drop_recommendations` stays callable for backward compatibility, but unused by the FA page. Removed in P3 once integration tests confirm parity.

**Tests**:
- `test_fa_page_uses_recommend_fa_moves.py` — AST check that the FA page imports + calls `recommend_fa_moves`, not `compute_add_drop_recommendations`
- Existing `test_fa_il_stash_unified_protection.py` (PR #89) is preserved as a UI-layer guard
- New `test_fa_page_output_shape.py` — mock the engine, render the page, assert table columns + expander structure unchanged

**Output shape diff for users**:
- Recommended Adds/Drops table: same 5 columns (Add, Position, Net SGP Delta, Sustainability, Drop)
- Why expanders: same structure, but reasoning strings will be different (engine produces different copy)
- Top pickup callout: same format
- Streamers section: unchanged (different code path)

### PR2: Fix IL-zeroing root cause in `_roster_category_totals` (~2h)

**File modified**: `src/in_season.py:42-94`

**Current behavior**:
```python
_IL_STATUSES = frozenset({"IL10", "IL15", "IL60", "IL-10", "IL-15", "IL-60"})
for _, p in roster.iterrows():
    status = str(p.get("status", "") or "").upper().strip()
    if status in _IL_STATUSES:
        continue   # ← BUG: IL players contribute zero
    totals["R"] += int(p.get("r", 0) or 0)
    ...
```

**New behavior** (strict IL policy):
```python
for _, p in roster.iterrows():
    status = str(p.get("status", "") or "").upper().strip()
    weight = _il_weight_from_status(status, expected_return_days=p.get("expected_return_days"))
    # weight is 0.0 for "Suspended" / unavailable, scaled for IL by return date,
    # 1.0 for active. NEVER zero for IL — that's the bug.
    totals["R"] += int(p.get("r", 0) or 0) * weight
    ...
```

**`_il_weight_from_status`**:
- `active` / empty → 1.0
- `IL10` → 0.85 (returning in <2 weeks; mostly available ROS)
- `IL15` → 0.70 (15-day; lose ~3 weeks)
- `IL60` → 0.20 (long stash; most of season but still has value)
- `DTD` → 0.95 (day-to-day)
- `Suspended` / `Restricted` → 0.0 (truly unavailable)

These weights are research-defensible (per RotoWire stash valuation, FantasyPros injury rankings); registered in `constants_registry.py` with citations.

**Tests**:
- `test_il_weight_from_status.py` — parametrized over all statuses
- `test_roster_category_totals_il_weighted.py` — Crochet on IL15 contributes 70% of his projected K to team K total
- Regression guard: old "IL zeroed" assertion replaced with new "IL weighted" assertion

### PR3: Wire positional scarcity into `recommend_fa_moves` scoring (~2-3h)

**Problem**: `compute_vorp` in `src/valuation.py:685-713` already implements scarcity (C / SS / 2B scarce-position bonus), but `fa_recommender.py` doesn't call it. Composite scoring is currently position-blind.

**Fix**: In `fa_recommender.py::_compute_base_value`, multiply by a positional scarcity factor derived from `compute_per_category_replacement` (the dynamic per-position replacement-level function in `src/valuation.py:1358`).

```python
def _compute_base_value(player, ctx):
    sgp = ... # existing calc
    # NEW: positional scarcity multiplier
    scarcity_mult = _positional_scarcity_factor(player["positions"], ctx)
    return sgp * scarcity_mult
```

Where `_positional_scarcity_factor` returns:
- For multi-position players: takes the BEST (scarcest) position
- C: 1.20× (or dynamic from replacement-level math — to be decided per design Q below)
- 2B / SS: 1.10×
- Others: 1.00×
- The current CLAUDE.md multipliers are kept as defaults but moved to `constants_registry.py`

**Open design question** (flagged for user before implementation): is the existing `compute_vorp` scarce-position bonus computation the right one, or should we use dynamic replacement-level math instead? Industry consensus (Razzball, FanGraphs auction calculator) prefers dynamic replacement. Recommended approach: use dynamic for scoring, keep explicit multipliers as backstop. **Default to dynamic unless user prefers otherwise.**

**Tests**:
- `test_fa_recommender_positional_scarcity.py` — top-2 catcher scores higher than 30th-OF with equivalent raw SGP
- `test_no_position_blindness_regression.py` — explicit regression guard

---

## Phase 2 — Engine quality (PR4, PR5, PR6 — ~6-8h)

Goal: engine reflects current player performance, not just preseason projection.

### PR4: Stat-window blending in `_compute_base_value` (~3-4h)

**Problem**: `ytd_*` and `recent_form` columns are loaded into the pool but never consumed by either recommender. Engine evaluates purely on ROS/preseason projection.

**Fix** (per published industry methodology):
```python
# Blend weights:
#   0.70 × ROS projection  (anchor)
#   0.20 × YTD             (in-season data)
#   0.10 × last-14-day     (recent form)
# Weights configurable via constants_registry; defaults match Smart Fantasy
# Baseball + The Athletic published guidance.

def _blended_projection(player, ctx):
    ros = _read_ros(player)
    ytd_rate, ytd_volume = _read_ytd(player)
    l14 = _read_l14(player, ctx)
    # Sample-size gating: skip YTD/L14 if below stabilization thresholds
    # (K% at 60 PA, BB% at 120 PA per Pizza Cutter / Carleton).
    if ytd_volume < _STABILIZATION_THRESHOLDS["pa"]:
        return ros
    return 0.70 * ros + 0.20 * ytd_rate + 0.10 * l14
```

**Tests**:
- `test_blend_weights_match_canon.py` — 0.10 + 0.20 + 0.70 = 1.00
- `test_blend_short_sample_falls_through_to_ros.py` — players with <60 PA YTD evaluated on pure ROS
- `test_blend_long_sample_uses_full_blend.py` — Aaron Judge w/ 200 PA YTD uses full blend
- `test_blend_constants_in_registry.py` — weights live in `constants_registry.py` not magic literals

### PR5: Sustainability score rewrite (~1-2h)

**Problem**: 5-bucket step function based on BABIP only. Pitcher logic is INVERTED ("high ERA = high sustainability" = "expected to improve"). Displayed as a percentage which the user reads as "% chance of sustaining."

**Fix**: replace with calibrated logistic on multiple signals.
```python
def compute_sustainability(player, is_hitter):
    if is_hitter:
        xwoba_gap = player.get("xwoba_minus_woba", 0)  # canonical regression signal
        babip_z = z_score(player["babip"], player["career_babip"])
        ev_trend = player.get("ev_mean_yoy_delta", 0)
        # Logistic combination
        logit = 2.5 * xwoba_gap - 0.6 * babip_z + 0.4 * ev_trend
        return _sigmoid(logit)  # returns 0-1 calibrated probability
    else:
        stuff_plus_yoy = ...
        siera_minus_era = ...
        k_bb_pct_z = ...
        ...
```

**Naming fix**: The UI string "Strong underlying metrics support continued production" stops firing when sustainability is high for the wrong reason. Add separate "regression-favorable" copy for pitchers with high ERA + good underlying skills.

**Tests**:
- `test_sustainability_continuous.py` — output is continuous 0-1, not stepped
- `test_sustainability_pitcher_inversion_fixed.py` — pitcher with 2.10 ERA + great Stuff+ scores HIGHER than pitcher with 5.50 ERA, not lower
- `test_sustainability_signals_combine.py` — xwOBA-wOBA gap drives the hitter signal

### PR6: Composite formula fix — multiplicative urgency (~1-2h)

**Problem**: `composite = base_value * sustainability * ownership_mult * floor_mult + urgency_boost` in `fa_recommender.py:267`. The urgency_boost is ADDITIVE and scales with number-of-categories the FA contributes to. So multi-category contributors (e.g., 5-tool OFs) get over-rewarded vs concentrated-value players (top SPs touching only W/K/ERA/WHIP).

**Fix**:
```python
# Make urgency a multiplicative per-category weight, applied INSIDE base_value
# instead of additively outside.
def _compute_base_value(player, ctx):
    contributions = {}
    for cat in categories:
        marginal = _category_marginal_sgp(player, cat)
        urgency_mult = ctx.category_urgency[cat]  # 0.1 - 3.0 scale
        contributions[cat] = marginal * urgency_mult
    return sum(contributions.values())

# Composite stays purely multiplicative:
composite = base_value * sustainability * ownership_mult * floor_mult
```

This eliminates the "many cats > few high-value cats" rank inflation.

**Tests**:
- `test_composite_no_additive_urgency.py` — `urgency_boost` no longer added separately
- `test_concentrated_value_competitive.py` — top SP scores competitive with multi-cat OF given equal raw SGP

---

## Phase 3 — Quality + maintenance (PR7-PR10 — ~4-6h)

Goal: engine accuracy refinements + tech debt cleanup.

### PR7: Dynamic IL stash list (~1h)
Remove `IL_STASH_NAMES = {"Shane Bieber", "Spencer Strider"}` from `src/alerts.py:28`. Derive dynamically from `league_rosters.status IN (IL10, IL15, IL60, IL, DTD)`. UI keeps the same protected-drop banner. Hardcoded list is removed only AFTER PR1 + PR2 ship.

### PR8: IL return-date scaling (~1-2h)
Already mostly covered by PR2's `_il_weight_from_status`. PR8 extends this to consume ESPN injury return dates (already loaded into `shared_data_layer.py:957-969`) for more granular weighting. IL10 with 3-day return weight = 0.95; IL10 with 12-day return weight = 0.70.

### PR9: 24 magic constants → constants_registry (~2h)
Move all 24 unjustified magic constants from `fa_recommender.py` + `waiver_wire.py` into `constants_registry.py` with:
- Citation (paper / blog / "internal heuristic")
- Bounds (min/max plausible values)
- Sensitivity tag (does varying it change recommendations meaningfully?)

Includes the `_STREAM_WIN_PROB_MIN=0.2755`, `_FLOOR_PA_MIN=50`, `_HITTER_GAMES_PER_DAY=0.85`, `WEEKLY_RATE_DEFAULTS`, etc.

### PR10: Regression-flag wiring + rate-stat volume weighting (~1h)
- `regression_flag` column (BUY_LOW / SELL_HIGH) is loaded into the pool but never read by either recommender. Use it as a final ranking-adjustment scalar (`base_value × 1.05` for BUY_LOW, `× 0.95` for SELL_HIGH).
- Fix `compute_matchup_targeted_adds` rate-stat volume weighting at `waiver_wire.py:758-761`.

---

## Phase 4 — Cosmetic polish (PR11-PR14 — ~1-2h)

Bundled from the earlier FA page review (PRs B, C, D + ECR format):

- PR11: ECR shown as integer "909", not "909.00"
- PR12: Position string dedup ("2B/3B,3B" → "2B,3B")
- PR13: "Why" expander shows "Other categories: +X.XX SGP" reconciliation line
- PR14: FA list pagination — "Show all 519" toggle (currently caps at 200)

---

## Sequencing + PR breakdown

```
P1  (parallel-safe, but recommended serial for review clarity)
 ├── PR1: Wire FA page to recommend_fa_moves
 ├── PR2: Fix IL zeroing in _roster_category_totals  [BLOCKS PR3]
 └── PR3: Positional scarcity via compute_vorp        [depends on PR1 + PR2]

P2  (PR4 + PR5 + PR6 mostly independent, can interleave)
 ├── PR4: Stat-window blending
 ├── PR5: Sustainability rewrite
 └── PR6: Composite formula multiplicative

P3  (depends on P1 complete)
 ├── PR7: Dynamic IL stash list                       [depends on PR1]
 ├── PR8: IL return-date scaling                      [depends on PR2]
 ├── PR9: Magic constants → registry
 └── PR10: Regression-flag wiring + rate-stat volume

P4  (independent, can ship anytime)
 ├── PR11: ECR integer formatting
 ├── PR12: Position string dedup
 ├── PR13: Why expander reconciliation
 └── PR14: FA pagination toggle
```

**Total: 14 PRs over ~20-25 hours of work.**

After each PR: CI run, your review, merge, brief regression check on the live app.

---

## Test strategy

Per CLAUDE.md convention:
- Every PR ships with at least one structural-invariant test (`test_no_*.py` / `test_sf*.py`)
- Behavioral tests verify the *intended* recommendation behavior on canonical scenarios (e.g., Crochet/Kirk should NEVER score positive)
- AST-based tests pin design decisions (e.g., "FA page imports `recommend_fa_moves`, not `compute_add_drop_recommendations`")
- After P1 lands: integration test that processes the actual Crochet/Kirk scenario through the engine and asserts the recommendation does NOT come back as a positive-value swap

---

## Open design questions (flagged for your input before P1 starts)

1. **Position scarcity multipliers**: dynamic from `compute_per_category_replacement` (industry consensus, more accurate), OR explicit fixed multipliers (C×1.20, 2B/SS×1.10) from CLAUDE.md (simpler, predictable)? **Default: dynamic. Fallback: fixed.**

2. **IL weights in `_il_weight_from_status`**: the defaults (IL10=0.85, IL15=0.70, IL60=0.20) are research-defensible but not citation-anchored. Acceptable to ship with these, refine later based on actual data?

3. **Stat-window blend weights**: 0.10·L14 + 0.20·YTD + 0.70·ROS is the published canonical blend. Comfortable shipping with these as defaults, or do you want to be more aggressive on recent-form (e.g., 0.20·L14 + 0.20·YTD + 0.60·ROS)?

4. **Composite formula direction**: when fixing the additive→multiplicative urgency bug, the *direction* of the change matters. If urgency is moved INSIDE base_value as a per-category multiplier, the composite formula becomes `composite = base_value × sustainability × ownership_mult × floor_mult`. That's a cleaner separation but means urgency stops being visible as a separate "boost" in any debug output. Acceptable?

5. **Deprecation of `waiver_wire.compute_add_drop_recommendations`**: after P1 wires the FA page through `recommend_fa_moves`, do we DELETE the old function entirely (clean break), or DEPRECATE it (mark for removal, keep importable for backward compat for a sweep cycle)? Default: deprecate with a `DeprecationWarning`, remove in 1 sweep cycle.

---

## Risks + unknowns

- **Output-shape compatibility**: `recommend_fa_moves` returns a slightly different dict shape than `compute_add_drop_recommendations`. PR1 needs a translation layer. Risk: subtle UI regressions if column names change. Mitigation: explicit snapshot tests of the page output before and after.
- **Performance**: P1 + P2 might increase per-page load time (more computation per FA). Currently 47.55s; tolerance is probably 60s. Will measure and optimize if needed.
- **Composite formula direction (Q4)**: if I'm wrong about the urgency-as-multiplier-inside being cleaner, I'll need to revert to additive-but-bounded.
- **IL weight tuning**: the IL10=0.85 / IL15=0.70 weights might be wrong empirically. We'd need real season data to validate. Ship with defaults, iterate later.

---

## Acceptance criteria

**P1 done when**:
- [ ] Free Agents page imports `recommend_fa_moves` and uses it as the primary recommendation engine
- [ ] `_roster_category_totals` no longer zeroes IL players — they keep their projection × IL weight
- [ ] `_compute_base_value` multiplies by positional scarcity factor
- [ ] Crochet/Kirk scenario test asserts the swap does NOT score positive
- [ ] All structural-invariant tests pass
- [ ] CI green
- [ ] User clicks Refresh All Data and confirms: no top-30 SP appears as drop candidate in Recommended Adds/Drops

**P2 done when**:
- [ ] Blended projections used for FA scoring
- [ ] Sustainability is a calibrated probability, not a step function
- [ ] Composite formula multiplicative-only

**P3 done when**:
- [ ] IL_STASH_NAMES hardcoded list removed
- [ ] Return-date scaling active
- [ ] All 24 magic constants moved to constants_registry with citations
- [ ] regression_flag wired into composite

**P4 done when**:
- [ ] All cosmetic polish PRs shipped

**Full overhaul done when**: I run a manual test of 5 random scenarios on the Free Agents page and every recommendation passes the "industry-best-practice smell test" — i.e., would a serious fantasy analyst at FantasyPros/Razzball/Pitcher List endorse it?

---

## What I need from you to start P1

1. ☐ Approval of this plan as a whole
2. ☐ Answer to design question #1 (dynamic scarcity vs fixed multipliers — recommend dynamic)
3. ☐ Acknowledgment of design question #2 (IL weight defaults — ship and iterate)
4. ☐ Answer to design question #5 (deprecate old function — recommend deprecate, remove in 1 cycle)

Questions 3 + 4 (blend weights + composite direction) I'll default to the recommended values unless you say otherwise.

Once approved, I start with PR1 immediately.
