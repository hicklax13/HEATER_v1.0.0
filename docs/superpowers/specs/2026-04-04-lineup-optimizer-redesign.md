# Lineup Optimizer Redesign — Matchup-Aware, Roster-Accurate

## Problem

The Lineup Optimizer page (pages/5_Lineup.py, 1800+ lines) has 5 interrelated bugs that make it unreliable for in-season lineup decisions:

1. Live matchup data (7-4-1 score with per-category values) never reaches the optimizer — category weights aren't matchup-aware
2. IP budget calculation produces nonsensical results (1.23/20 despite active pitchers)
3. Roster displays in random/alphabetical order instead of Yahoo Fantasy slot order
4. Projected totals are full-season, not weekly — misleading for H2H matchup optimization
5. The page is 1800+ lines with tangled concerns (data loading, optimization, display, IP tracking, matchup analysis)

## Design

### 1. Shared Roster Sort Order (src/ui_shared.py)

Add a `sort_roster_for_display()` function that sorts players into Yahoo Fantasy slot order. This function is used on EVERY page that displays the roster.

**Slot order for hitters:** C, 1B, 2B, 3B, SS, OF, OF, OF, Util, Util, BN, IL
**Slot order for pitchers:** SP, SP, RP, RP, P, P, P, P, BN, IL

**Logic:**
- Use the `selected_position` column from `league_rosters` (Yahoo's actual slot assignment)
- If `selected_position` is empty/missing, infer from `positions` column using priority order
- Starters sort by slot order above, bench players sort alphabetically after starters, IL last

**Apply to:** pages/1_My_Team.py, pages/5_Lineup.py, pages/3_Trade_Analyzer.py, and any other page rendering the roster table.

### 2. Live Matchup Data Flow (pages/5_Lineup.py)

Replace the broken standings-based `my_totals`/`opp_totals` with data from `yds.get_matchup()`.

The matchup dict from Yahoo has this structure:
```python
{
    "week": 2,
    "opp_name": "Baty Babies",
    "wins": 7, "losses": 4, "ties": 1,
    "categories": [
        {"cat": "R", "you": 18, "opp": 15, "result": "WIN"},
        {"cat": "HR", "you": 7, "opp": 6, "result": "WIN"},
        {"cat": "RBI", "you": 22, "opp": 23, "result": "LOSS"},
        ...
    ]
}
```

**Build `my_totals` and `opp_totals` from this data**, not from `league_standings`. This gives the optimizer real-time category values to compute urgency weights.

**Urgency weight formula:**
- Categories you're losing: weight × 1.5
- Categories you're tied: weight × 1.2
- Categories you're winning by small margin (<10%): weight × 1.0
- Categories you're winning comfortably: weight × 0.6

### 3. IP Budget Fix (pages/5_Lineup.py + src/ip_tracker.py)

The page builds pitcher dicts for IP tracking but doesn't include `status` or `positions` fields. Fix the dict construction to include all fields the updated `ip_tracker.py` needs:

```python
pitcher_dict = {
    "name": row["player_name"],
    "ip": float(row.get("ip", 0)),
    "positions": str(row.get("positions", "")),
    "status": str(row.get("status", "active")),
    "is_starter": "SP" in str(row.get("positions", "")),
}
```

Also: the IP tracker uses ROS projection IP, but early in the season each pitcher only has actual IP (e.g., Crochet 11.0 actual, 225 projected). The tracker should use **projected** IP for the formula (how many IP they're expected to pitch ROS), not actual IP accrued so far. Source projected IP from the `projections` table.

### 4. Weekly Projection Scaling

The "Projected Category Totals" table shows full-season numbers (651R, 186HR). For a weekly H2H matchup, show **weekly projections** instead:

```python
weekly_projection = full_season_projection / 26.0  # 26-week season
```

This gives managers a realistic sense of what to expect this week, not the full season.

### 5. Matchup-Aware Optimizer Wiring

The `compute_urgency_weights()` function in `src/optimizer/category_urgency.py` already exists and computes per-category urgency from matchup data. The problem is it's never called because the matchup data doesn't reach it.

**Fix the wiring:**
1. In pages/5_Lineup.py, after loading the live matchup, call `compute_urgency_weights(matchup)`
2. Pass the urgency dict to the pipeline via a new kwarg or by pre-multiplying category weights
3. The pipeline's `_compute_category_weights()` returns base weights — multiply each by the urgency factor

**IP-aware pitching weight adjustment:**
- If projected weekly IP < 20 (minimum): boost W, K, ERA, WHIP weights by 1.5x
- If projected weekly IP >= 20: reduce pitching weights by 0.7x to focus on hitting

## Implementation — Agent Architecture

### Agent 1: Roster Sort Order
**Scope:** `src/ui_shared.py` + all 5 pages that display rosters
- Add `sort_roster_for_display(roster_df)` to `src/ui_shared.py`
- Define `SLOT_ORDER_HITTERS` and `SLOT_ORDER_PITCHERS` constants
- Apply in: pages/1_My_Team.py, pages/5_Lineup.py, pages/3_Trade_Analyzer.py, pages/4_Free_Agents.py, pages/10_Trade_Finder.py

### Agent 2: Matchup Data + Urgency Wiring
**Scope:** `pages/5_Lineup.py` (lines 305-400, 475-530, 545-620)
- Replace standings-based team_totals with live matchup data
- Build proper my_totals/opp_totals from matchup["categories"]
- Call compute_urgency_weights() and multiply into category weights
- Wire urgency into pipeline.optimize() call
- Fix opponent selector to auto-detect from live matchup

### Agent 3: IP Budget + Projection Fixes
**Scope:** `pages/5_Lineup.py` (lines 354-381) + `src/ip_tracker.py`
- Fix pitcher dict construction to include status, positions, projected IP
- Use projected IP (from projections table) instead of actual IP for rate calculation
- Add weekly projection scaling to the "Projected Category Totals" display
- Verify IP tracker formula works with real roster data

## Verification

1. `python -m pytest tests/ -x -q` — all 2300 tests pass
2. Restart app, go to Lineup page:
   - Roster sorted in Yahoo slot order (C first, IL last)
   - IP budget shows realistic number (30-50 IP range, not 1.23)
   - Opponent auto-detected as "Baty Babies" from live matchup
   - Click Optimize: categories you're losing (RBI, OBP, W, K) have higher weights
   - Bieber/Strider on bench, Bibee starting
   - Projected totals scaled to weekly (not 651R full-season)
3. Check My Team page — roster also in slot order

## Files Modified

| File | Agent | Changes |
|------|-------|---------|
| src/ui_shared.py | 1 | Add sort_roster_for_display() |
| pages/1_My_Team.py | 1 | Apply roster sort |
| pages/5_Lineup.py | 1, 2, 3 | Sort + matchup wiring + IP fix |
| pages/3_Trade_Analyzer.py | 1 | Apply roster sort |
| src/ip_tracker.py | 3 | Accept projected IP, use for rate calc |
