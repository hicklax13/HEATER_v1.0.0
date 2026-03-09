# Fantasy Baseball Draft Tool

## Project Status: UI/UX Redesign Complete ✅ (Broadcast Booth Theme)

**All phases complete.** Design plan: `.claude/plans/optimized-wiggling-meerkat.md`

All core backend modules (Phases 1-7 of original build + 9 accuracy enhancements) are complete. A complete `app.py` rewrite (~1900 lines) implements the "Broadcast Booth" dark-theme UI with 4-step setup wizard, 3-column draft page, and sports-broadcast styling. Phase 7 verification passed all 15 checks plus cleanup.

## What's Been Done (UI Redesign)

### Phase 0: Infrastructure ✅
- `.streamlit/config.toml` dark theme created
- `THEME` dict with all design tokens at top of `app.py`
- `inject_custom_css()` function (~400 lines) with Google Fonts, component styles, animations

### Phase 1: Setup Wizard ✅
- 4-step wizard: Import → Settings → Connect → Launch
- Wizard progress bar with amber active / green completed states
- Load Sample Data button with in-process import

### Phase 2: Draft Page Layout ✅
- 3-column layout: My Roster | Center (hero pick + alternatives) | Enter Pick
- Command bar: Round/Pick, YOUR TURN badge, progress bar

### Phases 3-6: Components + Animations + Streamlit Features ✅
- Hero pick card, alternative cards, roster diamond, scarcity rings
- Pick entry panel, recent picks feed, category radar chart
- Draft board, draft log, available players table
- Amber glow pulse, card hover lift, skeleton loading
- `@st.fragment`, `st.toast()`, `st.status()`

### Phase 7: Verification ✅
11 bugs found and fixed during verification. All 15 checklist items passed.

<details>
<summary>Bugs fixed (click to expand)</summary>

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | "No player data found" on Step 4 | `create_blended_projections()` deletes `system='blended'` rows, but sample data only has 'blended' | Skip blend when no non-blended projections exist |
| 2 | `SGPCalculator.compute_sgp_denominators` AttributeError | `compute_sgp_denominators` is a standalone function, not a method | Import and call as standalone function |
| 3 | `compute_replacement_levels()` missing args | Function signature is `(pool, config, sgp_calc)` but was called with `(pool)` | Fixed to `(pool, lc, sgp)` |
| 4 | `value_all_players()` wrong args | Signature is `(pool, config, ...)` but was called with `(pool, sgp, repl, lc)` | Fixed to `(pool, lc, replacement_levels=repl)` |
| 5 | Hitter/pitcher count shows 0/190 | Code checked `player_type` column but DB uses `is_hitter` | Changed to `pool["is_hitter"].sum()` |
| 6 | `KeyError: 'player_name'` on draft page | DB column is `name` but UI code uses `player_name` | Added `rename(columns={"name": "player_name"})` in `_build_player_pool()` |
| 7 | Hero pick card rendering error | Missing column references after rename | Fixed column mappings in hero card rendering |
| 8 | Alternative cards not displaying | Score calculation used wrong column names | Aligned with renamed columns |
| 9 | Radar chart Plotly fillcolor error | Plotly 6.x rejects 8-digit hex colors (`#RRGGBBAA`) | Changed to `rgba(r,g,b,a)` format |
| 10 | Radar chart Plotly not installed | Streamlit's Python (3.13 Store) didn't have plotly | Installed plotly in correct Python environment |
| 11 | Draft board snake order wrong | `overall` pick number used display column index instead of pick position | Introduced `pos_in_round` variable for correct snake reversal |

</details>

### Verification Checklist ✅
- [x] 1. Setup wizard step navigation (forward/back/skip)
- [x] 2. Data import through new UI (Load Sample Data works, 190 players)
- [x] 3. Draft page 3-column layout renders
- [x] 4. Hero pick card: correct player + stats + survival gauge
- [x] 5. Alternative cards: 5 players with scores
- [x] 6. Roster diamond: filled/empty slots correct
- [x] 7. Scarcity rings: correct color thresholds
- [x] 8. Pick entry search + draft button works
- [x] 9. Practice mode auto-pick functions
- [x] 10. Undo reverses correctly
- [x] 11. Radar chart renders (or graceful fallback)
- [x] 12. Draft board dark theme + snake order
- [x] 13. Text contrast passes on dark background
- [x] 14. MC simulation <8 seconds (4.45s with 6-round horizon optimization)
- [x] 15. `@st.fragment` — verified unnecessary (architecture avoids redundant reruns)
- [x] Debug logging cleaned up from Load Sample Data handler
- [x] `?skip=1` debug shortcut removed

## Purpose

A live fantasy baseball draft assistant for use on a laptop during a Yahoo Sports snake draft. Recommends the optimal player to draft each time the user is on the clock, accounting for league settings, draft state, category scarcity, positional needs, opponent behavior, and risk.

## League Context

- **Provider:** Yahoo Sports
- **League:** FourzynBurn | **Team:** Team Hickey
- **Format:** 12-team snake draft, 23 rounds
- **Scoring:** 5x5 roto categories (R, HR, RBI, SB, AVG / W, SV, K, ERA, WHIP)
- **Manager skill:** Opponents are extremely high-skilled; user is a novice

## Roster Positions

| Slot | Count | Slot | Count |
|------|-------|------|-------|
| C    | 1     | Util | 2     |
| 1B   | 1     | P    | 4     |
| 2B   | 1     | SP   | 2     |
| 3B   | 1     | RP   | 2     |
| SS   | 1     | BN   | 5     |
| OF   | 3     | MI/CI | 0   |

**Total roster size:** 23 (matches draft rounds)

## Tech Stack

- **Framework:** Streamlit (Python)
- **Database:** SQLite (`data/draft_tool.db`)
- **Core libs:** pandas, NumPy, SciPy, Plotly
- **Yahoo API:** yfpy (optional, for live league sync)

## File Structure

```
app.py                  — Main Streamlit app (~1900 lines): setup wizard + draft page
.streamlit/config.toml  — Dark theme configuration
.claude/launch.json     — Dev server config for preview tools
src/
  database.py           — SQLite schema, CSV import, projection blending, player pool loading
  valuation.py          — SGP calculator, replacement levels, VORP, category weights, full valuation
  draft_state.py        — Draft state management, roster tracking, snake pick order, JSON persistence
  simulation.py         — Monte Carlo draft simulation, opponent modeling, survival probability, urgency
  yahoo_api.py          — Yahoo Fantasy API integration (yfpy OAuth, league sync, draft polling)
  validation.py         — Validation utilities
load_sample_data.py     — Generates ~190 sample players for testing
data/
  draft_tool.db         — SQLite database (created at runtime)
  backups/              — Draft state JSON backups
  yahoo_creds.json      — Yahoo API credentials (not committed)
```

## Architecture & Key Algorithms

### Valuation Pipeline
1. **Projection blending** — Weighted average of multiple projection systems (Steamer, ZiPS, etc.)
2. **SGP (Standings Gain Points)** — Converts raw stats to standings-point movement; auto-computed denominators available
3. **VORP** — Value Over Replacement Player with multi-position flexibility premium
4. **Category weights** — Dynamic: standings-based when league data available, draft-progress-aware scaling
5. **pick_score** = weighted SGP + positional scarcity + VORP bonus

### Draft Engine
- **Monte Carlo simulation** (100-300 sims) with opponent roster tracking
- **Survival probability** — Normal CDF with positional scarcity adjustment
- **Urgency** = (1 - P_survive) * positional_dropoff
- **Combined score** = MC mean SGP + urgency * 0.4
- **Tier assignment** via natural breaks algorithm

### Nine Accuracy Improvements (Implemented)
1. Dynamic replacement levels (recalculated as players are drafted)
2. Actual roster totals in rate stat SGP calculations
3. Opponent roster tracking in simulations
4. Multi-position eligibility bonus in VORP (+0.12 per extra position, +0.08 for scarce positions)
5. Dynamic category targets from standings data
6. Full Monte Carlo integration into pick recommendations
7. Projection confidence discount (low PA/IP players get up to 20% discount)
8. Bench slot optimization (late-draft bonus for multi-position flexibility)
9. Draft position awareness in urgency calculations

## Running the App

```bash
# First time: load sample data
python load_sample_data.py

# Launch the app
streamlit run app.py
```

## Key Technical Notes

- Python 3.14 + SQLite returns bytes for some integer columns; fixed with explicit CAST in SQL and `pd.to_numeric()` coercion
- Snake draft pick order: `round % 2 == 0` forward, `round % 2 == 1` reverse
- MC simulation uses NumPy vectorized availability tracking (`is_available` boolean mask)
- SGP denominators stored in `LeagueConfig` dataclass, user-configurable in UI
- Auto-SGP denominators computed by distributing top players across simulated teams, measuring stddev
- Draft state persists to JSON in `data/backups/` for crash recovery
- Sample data uses `system='blended'` — do NOT call `create_blended_projections()` when only blended data exists
- DB column is `name` but UI expects `player_name` — normalized via rename in `_build_player_pool()`
- `compute_replacement_levels(pool, config, sgp_calc)` and `compute_sgp_denominators(pool, config)` are standalone functions in `src/valuation.py`, NOT methods on SGPCalculator
- MC simulation uses 6-round horizon limit for performance (4.45s vs 10.42s without limit)
- Plotly 6.x does NOT accept 8-digit hex colors; use `rgba(r,g,b,a)` format instead

## Data Sources (for real draft)

- **Projections:** FanGraphs CSV exports (Steamer, ZiPS, Depth Charts)
- **ADP:** FantasyPros consensus ADP
- **Yahoo integration:** yfpy library for league settings import and live draft polling
