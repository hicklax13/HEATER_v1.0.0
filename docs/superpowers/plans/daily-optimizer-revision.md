# Daily Line-up Optimizer Revision Plan

**Goal:** the daily optimizer returns the optimal daily lineup (highest total
weighted DCV under Yahoo slot constraints), with every guard from the success
criteria intact.

**Branch:** `claude/revise-lineup-optimizer-T6y8I`

## Success criteria (from `/goal`)

1. For a fixed roster + game date, optimizer selects the lineup with the
   highest total weighted DCV reachable under Yahoo slot constraints
   (C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P).
2. Locked games (in-progress/final/past start time) get `volume_factor=0`
   and never start.
3. Pure-P / SP-only pitchers not in today's probables get `volume=0`
   (SP gate).
4. `forced_start` flag fires when LP starts a player with
   `matchup_mult < 0.70` or `total_dcv < median*0.5`.
5. Structural-invariant tests pass.
6. `python -m pytest --ignore=tests/test_cheat_sheet.py` is green.

## Survey findings (already-correct)

- **Criterion 2 (locked games):** `daily_optimizer.py:631-664, 786-787`
  already detects `in progress / final / game over / completed` AND
  `game_datetime <= now_utc`, then forces `volume = 0.0` and
  `reason = "LOCKED"`. ✓
- **Criterion 3 (SP gate):** `daily_optimizer.py:757-781` handles
  `positions == {"SP"}` and `positions == {"P"}` token sets; normalises
  names for accent/suffix matching against `probable_starters`;
  SP/RP hybrids keep `0.9` baseline. ✓ Guarded by `test_sp_gate_trace.py`.
- **Criterion 4 (forced_start):** `pages/6_Line-up_Optimizer.py:1498-1522`
  computes `forced_start` using `_FORCED_MATCHUP_THRESHOLD = 0.70` and
  `_FORCED_DCV_RATIO = 0.50` (median × 0.5), splits by hitter/pitcher
  median. ✓
- **Criterion 1 (slot-aware DCV maximisation in the page):** the page's
  `_run_lp_for_group` (`pages/6_Line-up_Optimizer.py:1256-1277`) runs the
  LP twice (hitters / pitchers) with `total_dcv` shifted into the `r`
  column and `category_weights={"r": 1.0}`, so the LP maximises total
  DCV under the Yahoo slot map. ✓

## Bugs / gaps fixed in this revision

### B1 — `src/optimizer/pipeline.py:251-255` weeks_remaining baseline is 24

`weeks_remaining` is computed from `24 - _weeks_elapsed`, but
FourzynBurn is a 26-week season (canonical per CLAUDE.md and
`test_lineup_optimizer_26_weeks`). Two-week early-season under-count
inflates `alpha` (in pipeline callers) and bleeds through every
remaining-games scaler. → bump to `26`.

### B2 — `pages/6_Line-up_Optimizer.py:534, 842` page-level 24-week constants

Auto-compute uses `_TOTAL_WEEKS = 24` to seed `weeks_remaining`, and the
Rest-of-Season alpha formula divides by `24.0`. Same 26-week
canonical issue. → both bumped to 26. (Note: this is **not** caught by
`test_lineup_optimizer_26_weeks` because that test only inspects
`WEEKS_IN_SEASON = ...` literals, which were already 26.0 at line
2079.)

### B3 — `src/optimizer/daily_optimizer.py:1158-1169` retry drops `team_strength`

The all-zero-DCV retry path recursively calls `build_daily_dcv_table`
without forwarding `team_strength` (or the `ctx`). When the retry
fires, pitcher matchup multipliers silently lose the opposing-offense
wRC+ adjustment — pitchers facing the Yankees / Marlins look
identical. → forward `team_strength` (and forward `ctx=None` since the
parent already merged it).

### B4 — `src/optimizer/daily_optimizer.py:745-755, 818` raw-team lookups

`confirmed_lineups[team]` and `weather_by_team.get(team)` use raw
upper-cased `team` (the SQL column) rather than `team_canon` (output
of `canonicalize_team`). For teams whose code drifted (OAK→ATH,
WSN→WSH, CHW→CWS, AZ→ARI), the lookup silently misses → players get
`in_lineup=None` (volume 0.9 baseline) and lose weather. →
prefer canonical lookup, fall back to raw.

### B5 — `src/optimizer/pipeline.py:565-597` `daily_lineup` ignores slot constraints

The naive `dcv[dcv["volume_factor"] > 0].head(18)` shortcut returns
the top-18 by `total_dcv` regardless of Yahoo slots, so external
consumers of `LineupOptimizerPipeline(mode="daily").optimize()[
"daily_lineup"]` would get a roster with 7 OFs / 0 catchers.
The page sidesteps this by running its own LP-on-DCV; pipeline
should do the same. → replace shortcut with a slot-aware LP using
the same DCV-shift-into-`r` trick the page uses, split hitters /
pitchers to keep negative-DCV pitcher slots from collapsing.

## Test plan

1. Run only-fast targeted tests first (Quick path):
   - `tests/test_daily_optimizer.py`
   - `tests/test_sp_gate_trace.py`
   - `tests/test_optimizer_pipeline_forwards_context.py`
   - `tests/test_lineup_optimizer_26_weeks.py`
   - `tests/test_no_lc_singletons_in_optimizer.py`
   - `tests/test_optimizer_pipeline.py`
2. Then the full suite: `python -m pytest --ignore=tests/test_cheat_sheet.py`.
3. No new tests are required — every change is covered by existing
   structural-invariant guards.

## Constraints respected

- No hardcoded category lists added — only existing literals touched.
- No new `_LC = ...` module-level singletons.
- No direct `sqlite3.connect` introduced; no scripts touched.
- `PULP_AVAILABLE` left untouched; pipeline keeps importing the
  `LineupOptimizer` class.
- Sigmoid k-values still read from `CONSTANTS_REGISTRY` at runtime
  (`category_urgency.py` is unchanged).
- No new files created — all edits go into existing modules and this
  single checklist (which the goal explicitly requested).
