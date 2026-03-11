# Plan 4: Surface the Analytics — Design Spec

**Date:** 2026-03-11
**Approach:** Integration First — wire existing Plan 3 backend into UI with minimal new logic
**Status:** Approved

## Context

Plan 3 added 5 backend analytics modules (~1,900 LOC, 96 tests):
- Bayesian projection updater (`src/bayesian.py`)
- Injury risk model (`src/injury_model.py`)
- Percentile forecasts (`src/valuation.py` extensions)
- Enhanced opponent model (`src/simulation.py`, `src/draft_state.py` extensions)
- Yahoo Fantasy API client (`src/yahoo_api.py`)

None of these surface in the draft page UI (`app.py`) or in-season pages (except `5_Lineup_Optimizer.py`). The draft page still shows point-estimate SGP, basic survival probability, and ADP-only opponent modeling.

## Design Decisions

- **Info density:** Contextual badges directly on hero card and alternatives (compact, no extra clicks)
- **Opponent intel:** Threat alerts on hero card + full Opponent Intel tab in bottom tabs
- **Yahoo OAuth:** Setup wizard Step 1 with "Connect Yahoo" button, graceful fallback to manual CSV
- **Architecture:** Import existing functions, render their output. No new algorithms or pipeline refactors.

---

## Section 1: Draft Page — Injury Badges & Risk Flags

### Data Flow
1. On draft page init, query `injury_history` table for all players
2. Compute `health_score` per player via `compute_health_score()`
3. Build a `health_scores` DataFrame with columns `player_id`, `health_score`, and `age_risk_mult` (computed from `compute_age_risk()` using player age and is_hitter flag). Store the `{player_id: health_score}` dict in session state for badge lookups.
4. Apply `apply_injury_adjustment(projections_df, health_scores_df)` to counting stat projections so injured players rank lower naturally. Both DataFrames must share `player_id` for the merge.

### Hero Card Display
```
🏆 #1 PICK: Aaron Judge
OF • NYY  🟢 Healthy  ⚠️ Age 33
```

- **Injury badge:** `get_injury_badge(health_score)` → 🟢 ≥0.90, 🟡 0.75-0.89, 🔴 <0.75 (matches actual thresholds in injury_model.py:294)
- **Age flag:** ⚠️ for hitters 30+ or pitchers 28+ (matches age-risk curves in injury_model.py)
- **Workload flag:** 🔥 for pitchers with >40 IP increase year-over-year

### Alternative Cards
Same badges, compact form (icon only, no label text).

---

## Section 2: Draft Page — Percentile Ranges

### Data Flow
1. On draft page init, compute volatility via `compute_projection_volatility()` across available projection systems
2. Check if volatility is all zeros (single projection source / sample data):
   - If yes: skip steps 3-4, set `has_percentiles = False`, show "single source" note in UI
   - If no: continue to step 3
3. Apply `add_process_risk(volatility_df, historical_correlations)` to widen low-correlation stats (AVG, SB)
4. Generate P10/P50/P90 with `compute_percentile_projections(base=base_df, volatility=vol_df)` — note: parameter names are `base` and `volatility`, NOT `base_df`/`volatility_df`
5. Convert raw stat percentiles to SGP units via `SGPCalculator`

### Hero Card Display
```
Score: 8.42  Surv: 12%
P10: 6.1  ██████████░░░  P90: 10.7
```
Inline bar: filled portion = where P50 sits within P10-P90 range. Wider = more uncertainty.

### Alternative Cards
Compact text: `6.1 — 10.7` (no bar).

### MC Simulation Upgrade
- **Important:** `evaluate_candidates()` in `simulation.py` does NOT currently pass through `use_percentile_sampling` or `sgp_volatility` to `simulate_draft()`. Must update `evaluate_candidates()` to accept and forward these two parameters.
- Enable `use_percentile_sampling=True` in `simulate_draft()` (via updated `evaluate_candidates()`)
- Pass `sgp_volatility` array so MC sims draw from Normal(mean, vol) instead of point estimates
- Display `risk_adjusted_sgp` (MC_mean - 0.5 * MC_std) alongside combo score
- When `has_percentiles` is False (single source), skip percentile sampling — use existing point estimates

---

## Section 3: Draft Page — Opponent Intelligence

### Threat Alerts (Hero Card)
On each pick:
1. Build draft history DataFrame by joining `draft_picks` with player data to get `team_key` and `positions` columns (required by `compute_team_preferences()`). The `draft_picks` table alone does NOT have `positions` — must join on `player_id`.
2. Call `compute_team_preferences(draft_history_df)` → returns dict keyed by string `team_key`
3. Call `get_positional_needs(draft_state_dict, team_id, roster_config)` for each team picking before user's next turn — note: `team_id` is a 0-based integer index, NOT the string `team_key`. Must maintain a mapping between `team_key` strings and integer `team_id` indices (the `team_index` field in the pick log provides this).
4. Generate threat assessment by combining preferences (string-keyed) and needs (int-keyed) via the mapping

Display rules (1-2 lines max):
- Team with >60% positional bias toward hero pick's position → `🔥 Team 6 targets SS early (72% bias)`
- 3+ teams need hero pick's position → `⚠️ Low survival: 3 teams need OF before you`
- Survival >80% → `✅ Likely available next round`

### Opponent Intel Tab
New tab in bottom tabs section (alongside Draft Board, All Picks, Category Standings):
- Table: Team name | Positions needed | Historical bias | Predicted next pick position
- Color-coded rows by threat level to user's target players
- Refreshes after every pick

### Data Sources
- `draft_picks` JOIN `player_pool` (on `player_id`) → DataFrame with `team_key` and `positions` columns → `compute_team_preferences()` → per-team bias dict (keyed by string `team_key`)
- Current draft state dict → `get_positional_needs(state, team_id_int, roster_config)` per team → needs dict (keyed by position string, only positions with remaining > 0)
- **Team ID mapping:** Build `{team_key_str: team_id_int}` from the pick log's `team_index` field. Both opponent intel sources use this mapping to reconcile their different keying schemes.
- Falls back to ADP-only when no draft history exists (compute_team_preferences returns empty dict for None input)

---

## Section 4: In-Season Page Enhancements

### My Team (`1_My_Team.py`)
- Injury badges (🟢🟡🔴) next to each roster player
- Bayesian-adjusted projections: call `BayesianUpdater.batch_update_projections()` when season stats exist AND `games_played > 0` for at least some players. If all `games_played` are 0 (sample data / pre-season), skip Bayesian update and show preseason projections without the "Updated" tag.
- Show `📊 Updated` tag next to stats that have been regressed with observed data (only when actual games have been played)

### Trade Analyzer (`2_Trade_Analyzer.py`)
- P10/P90 range for each player in trade proposal (upside/downside risk)
- Injury badges next to traded players
- No changes to core MC analysis

### Player Compare (`3_Player_Compare.py`)
- Health score comparison row in breakdown table
- "Projection confidence" row showing P10-P90 range width per player

### Lineup Optimizer (`5_Lineup_Optimizer.py`)
- Two-start SP detection via MLB Stats API schedule data
- Health score penalty in LP objective function (risky players slightly penalized)

---

## Section 5: Yahoo API Integration

### Setup Wizard Step 1
```
┌─────────────────────────────────────┐
│  🏈 Connect Yahoo Fantasy           │
│  [Authorize with Yahoo]             │
│  ── or ──                           │
│  Configure manually below...        │
└─────────────────────────────────────┘
```

### OAuth Flow
- Use yfpy's own browser-based OAuth flow (NOT `streamlit-oauth` — the two are incompatible token formats)
- `YahooFantasyClient.authenticate()` already accepts a `token_data: dict | None` parameter (line 137 of yahoo_api.py) but currently ignores it. **Must update** `authenticate()` to consume `token_data` when provided, skipping yfpy's browser flow.
- Flow: User clicks "Authorize with Yahoo" → yfpy opens browser OAuth → token stored in `data/` directory → yfpy client created
- On subsequent sessions, yfpy auto-refreshes from stored token file (no re-auth needed)

### Auto-Populate on Connect
- League settings (teams, roster slots, scoring) → pre-fill config form
- All 12 team rosters → `league_rosters` table
- Current standings → `league_standings` table
- Draft results → `draft_picks` table

### Credential Handling
- `YAHOO_CLIENT_ID` and `YAHOO_CLIENT_SECRET` from environment variables
- If missing, "Connect Yahoo" button is hidden (no error states)
- All code behind `YFPY_AVAILABLE` checks — manual CSV always works

### In-Season Sync
- "🔄 Sync Yahoo" button on `1_My_Team.py`
- One-click refresh of rosters and standings from Yahoo

---

## Section 6: CI Improvements & Practice Mode

### CI Coverage Reporting
- Add `pytest-cov` with `--cov=src --cov-report=term-missing`
- Coverage floor: 75% (not 80% — optional deps like PyMC/PuLP/yfpy unavailable in CI)
- Coverage summary visible in CI output

### Practice Mode
- "🎮 Practice Mode" toggle in draft page header
- When enabled: draft state is session-state-only (NOT written to `draft_state.json` or `draft_picks` table). A separate `st.session_state["practice_draft_state"]` dict holds all practice picks. On page refresh, practice state resets (intentional — practice is ephemeral).
- Yellow banner: "PRACTICE MODE — picks won't be saved"
- "Reset Practice" button clears `st.session_state["practice_draft_state"]`
- All analytics (MC, opponent intel, injury, percentiles) work identically — they read from whichever draft state dict is active (real or practice)
- Full dress rehearsal capability

---

## Files Modified

| File | Changes |
|------|---------|
| `app.py` | Import injury/percentile/opponent modules, add badges to hero card and alternatives, add percentile bars, add threat alerts, add Opponent Intel tab, add practice mode toggle, build team_key→team_id mapping |
| `src/simulation.py` | Update `evaluate_candidates()` to accept and forward `use_percentile_sampling` and `sgp_volatility` to `simulate_draft()` |
| `src/yahoo_api.py` | Update `authenticate()` to consume `token_data` dict when provided |
| `pages/1_My_Team.py` | Import injury + Bayesian modules, add badges and updated projections, add Yahoo sync button |
| `pages/2_Trade_Analyzer.py` | Import injury + percentile modules, add badges and P10/P90 ranges |
| `pages/3_Player_Compare.py` | Import injury + percentile modules, add health score and confidence rows |
| `pages/5_Lineup_Optimizer.py` | Add two-start SP detection, health score penalty |
| `.github/workflows/ci.yml` | Add pytest-cov configuration |

## New Files
None — this is purely integration work. Practice mode uses session state, not a new DB table.

## Testing Strategy
- Existing 150 tests continue to pass (backend unchanged)
- Add integration tests verifying import paths work (modules load cleanly in page context)
- Manual testing of draft page UI with sample data
- CI coverage reporting validates no regressions
