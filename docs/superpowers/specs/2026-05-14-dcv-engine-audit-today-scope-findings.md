# DCV Engine Audit — Today Scope — Findings

**Date:** 2026-05-14
**Auditor:** Claude Opus 4.7 (1M context) — single-session sequential audit
**Spec:** [design](2026-05-14-dcv-engine-audit-today-scope-design.md)
**Context:** [context bundle](2026-05-14-dcv-engine-audit-today-scope-context.md)

## Audit Provenance

The original 6-agent parallel dispatch failed: all 6 agents hit stream timeouts after ~2 hours each without writing their report files. Recovery: I executed the audit myself in a single session, applying all 6 agent lenses (algorithms, constants, league-config, inputs/data, MatchupContext+sigmoid, edge cases) to a sequential read of the in-scope files. The findings carry the same `DCV-AX-NNN` IDs as if the agents had produced them — A1=algorithms, A2=constants, A3=league-config, A4=inputs/data, A5=matchup+sigmoid, A6=edge cases — for traceability to the original audit design.

## Executive Summary

- **HIGH:** 2
- **MED:** 14
- **LOW:** 12
- **Demoted (HIGH → MED):** 0 — both HIGH findings carry indisputable proof artifacts.

**Headline risks:**

1. **DCV-A1-001** (HIGH) — Four hardcoded replacement-level baselines (`_REPL_AVG=0.240`, `_REPL_OBP=0.305`, `_REPL_ERA=4.50`, `_REPL_WHIP=1.35`) control all rate-stat DCV. A 0.005 shift in `_REPL_AVG` changes per-player AVG DCV by ~16-33%. The constants have no citation, no entry in `CONSTANTS_REGISTRY`, and cannot be perturbed by sensitivity analysis. They were calibrated by inspection, not by research.
2. **DCV-A1-002** (HIGH) — `apply_stud_floor` uses pre-SF-25 SGP math (`sgp += val/denom` for all categories including rate stats), producing a ranking dominated by rate stats by a 45:1 factor over counting stats. The "studs" identified for floor protection are biased toward high-AVG hitters; high-HR/SB/SV/K hitters and pitchers are under-protected. Direct contradiction of the SF-25 consolidation that established `SGPCalculator.totals_sgp` as the sole SGP path.

**Confidence statement:** After this audit, the DCV engine for "Today" scope is correct against FourzynBurn settings to approximately **80%** confidence. The 20% gap is driven by:
- 2 HIGH findings affecting rate-stat DCV magnitude and stud-floor ranking
- 14 MED findings (mostly constant drift between code, registry, and CLAUDE.md, plus several dead/inconsistent fallback paths)
- 6 open questions (see section below) where the audit cannot conclude without your input on FourzynBurn-specific behavior

Confidence is bounded by the fact that most of the engine math (sigmoid urgency, weighted rate-stat aggregation in SGPCalculator, IL/SUSP exclusion, locked-teams logic, SP gate handling after Wave 10) is correct and well-cited. The bugs are concentrated in two pre-SF-25 leftover code paths and in constant-citation hygiene.

## Methodology

Single-session sequential audit by Claude. Files read in this order:
1. `src/optimizer/constants_registry.py` (334 LOC) — full read
2. `src/optimizer/daily_optimizer.py` (1302 LOC) — full read, multiple passes
3. `src/optimizer/shared_data_layer.py` (1105 LOC) — partial read focused on constants and category-weight assembly
4. `src/optimizer/category_urgency.py` (274 LOC) — full read
5. `src/matchup_context.py` (549 LOC) — focused read on `get_category_weights` and 3-mode logic
6. `src/game_day.py` — `get_target_game_date` and `_FINAL_STATUSES`
7. `pages/6_Line-up_Optimizer.py` — Today-scope branch (lines 939-1011)
8. `CLAUDE.md` — referenced throughout for citations

All findings reference exact line numbers and quote the relevant code or doc. Indisputability rule applied: HIGH findings carry math derivation or direct citation. Demotion rule was not triggered (both HIGH findings have indisputable proof).

## Findings Catalog

### HIGH (must-fix before next H2H matchup)

#### DCV-A1-001 — Hardcoded `_REPL_*` baselines without citation

```
ID:       DCV-A1-001
Severity: HIGH
File:     src/optimizer/daily_optimizer.py:993-996
Symptom:  Four magic replacement-level baselines (_REPL_AVG=0.240, _REPL_OBP=0.305, _REPL_ERA=4.50, _REPL_WHIP=1.35) drive the entire rate-stat DCV computation without any research citation or registry entry. Sensitivity analysis cannot perturb them.
Expected: Replacement-level baselines should be (a) cited (e.g., "12-team H2H mixed league deep replacement, calibrated from N=X player-seasons of standings data") and (b) registered in CONSTANTS_REGISTRY so sigmoid_calibrator / sensitivity_analysis can perturb them.
Actual:   Lines 992-996:
              # Replacement levels (12-team H2H mixed league, deep replacement):
              _REPL_AVG = 0.240
              _REPL_OBP = 0.305
              _REPL_ERA = 4.50
              _REPL_WHIP = 1.35
          Comment says "deep replacement" without citation. Values are not in CONSTANTS_REGISTRY.
Proof:    Math derivation — sensitivity of player DCV to _REPL_AVG:
          For a hitter at AVG=0.270 with 500 AB:
          - annual_sgp = (h - ab × _REPL_AVG) / _RAW_SGP_DENOM["AVG"]
          - With _REPL_AVG=0.240: (135 - 500×0.240) / 22 = 15/22 = 0.682
          - With _REPL_AVG=0.245: (135 - 500×0.245) / 22 = 12.5/22 = 0.568
          - With _REPL_AVG=0.250: (135 - 500×0.250) / 22 = 10/22 = 0.455
          - Diff from 0.240 → 0.245: -16.7% in AVG DCV
          - Diff from 0.240 → 0.250: -33.3% in AVG DCV
          A 0.005 absolute shift in the constant produces ~16% relative change. This exceeds the HIGH threshold of "5%+ DCV mis-ranking."
          Around the replacement level itself, the sign of contribution flips:
          - Player at AVG=0.245 with _REPL=0.240: positive contribution
          - Same player with _REPL=0.250: negative contribution
Repro:    Run build_daily_dcv_table with a roster including a hitter at AVG=0.270/500AB. Inspect dcv_avg. Mutate _REPL_AVG by ±0.005. Compare. Difference will be 15-33% in either direction.
Fix:      (1) Cite each _REPL_* with a source (research paper, calibrated value from N seasons of standings data, or empirical 12-team H2H mixed-league baseline). (2) Add to CONSTANTS_REGISTRY with bounds and sensitivity. (3) Read at call-site so sigmoid_calibrator can perturb them.
```

#### DCV-A1-002 — `apply_stud_floor` uses pre-SF-25 SGP math; rate stats dominate ranking

```
ID:       DCV-A1-002
Severity: HIGH
File:     src/optimizer/daily_optimizer.py:441-452
Symptom:  apply_stud_floor computes player ranking SGP as `sgp += val / denom` for every category including rate stats. For AVG (denom=0.004), a player's contribution is 67.5 SGP units (val/denom for AVG=0.270). For HR (denom=20), the same player contributes 1.5 SGP units. The 45:1 ratio means rate stats completely dominate stud ranking; the "studs" identified are biased toward high-AVG hitters.
Expected: Per SF-25 (CLAUDE.md "SGP through SGPCalculator.totals_sgp only — All 7 prior local reinventions migrated. Structurally guarded."), stud_floor should call sgp_calc.totals_sgp({...}) on the player's projected stats. The SGPCalculator handles rate-stat aggregation correctly using team-volume-weighted normalization, not raw rate / denom.
Actual:   Lines 441-452:
              for cat in config.all_categories:
                  val = float(row.get(cat.lower(), 0) or 0)
                  denom = config.sgp_denominators.get(cat, 1.0)
                  if abs(denom) > 1e-9:
                      if cat in config.inverse_stats:
                          sgp -= val / denom
                      else:
                          sgp += val / denom
              total_sgp[pid] = sgp
          Direct val/denom division for all 12 cats, no team-volume normalization, no Bayesian replacement-level subtraction.
Proof:    Direct citation — CLAUDE.md line documenting SF-25:
              "SGPCalculator.totals_sgp is sole SGP path — 7 prior local reinventions deleted/migrated"
          And the structural guard in tests:
              tests/test_no_hardcoded_categories_in_src.py
          plus 5 sibling guards. This 8th occurrence in apply_stud_floor is unguarded because the test family checks for hardcoded *category lists*, not for raw `val/denom` division. The function bypasses SGPCalculator entirely.
          Math derivation of bias:
          - For HR=30 (counting): sgp += 30/20 = 1.5
          - For AVG=0.270 (rate): sgp += 0.270/0.004 = 67.5
          - For SB=20: sgp += 20/8 = 2.5
          - For ERA=3.50 (inverse rate): sgp -= 3.50/0.20 = -17.5
          - For K=180: sgp += 180/50 = 3.6
          Counting stats contribute O(1-5) SGP each. Rate stats contribute O(50-100) per cat. A high-AVG hitter outranks an elite power-speed hitter solely because of the rate-stat scaling artifact.
Repro:    Build a small player pool with two hitters:
              Hitter A: AVG=0.300, HR=20, SB=20, R=80, RBI=80, OBP=0.360
              Hitter B: AVG=0.260, HR=40, SB=40, R=100, RBI=110, OBP=0.330
          Hitter B is objectively more valuable in 6-cat scoring (more HR, SB, R, RBI; only slightly worse rates). Run total_sgp calculation. Hitter A will rank higher because (0.300+0.360)/(0.004+0.005) ≈ 73 SGP vs Hitter B's (0.260+0.330)/(0.004+0.005) ≈ 65 SGP — and that swamps all counting-stat differences.
Fix:      Replace lines 441-452 with a call to sgp_calc.totals_sgp(...) per player. Add a structural-invariant test that flags any `val / denom` SGP pattern in src/optimizer/.
```

### MED (should-fix before season-end)

#### DCV-A1-003 — Doubleheader handling dead in daily DCV path

```
ID:       DCV-A1-003
Severity: MED
File:     src/optimizer/daily_optimizer.py:779
Symptom:  compute_volume_factor accepts is_doubleheader and returns 2.0 for confirmed-starter doubleheader. The caller in build_daily_dcv_table never passes is_doubleheader, so the doubleheader 2.0 boost is dead code.
Expected: Per CLAUDE.md "Volume factor — 2.0 = confirmed in doubleheader starting lineup," doubleheader detection should flow through. Yahoo lineups DO show doubleheader twin-bills; statsapi.schedule returns separate game entries for each leg.
Actual:   Line 779: volume = compute_volume_factor(team_plays, in_lineup)
          Function signature: def compute_volume_factor(team_playing_today, in_confirmed_lineup, is_doubleheader=False).
          The caller never inspects schedule_today for duplicate (home, away) team pairs to detect doubleheaders.
Proof:    Direct citation — CLAUDE.md "compute_volume_factor — 2.0 = confirmed in doubleheader starting lineup." Documents intended behavior; code does not implement it.
Repro:    A player on a team with a confirmed doubleheader scheduled today and in both starting lineups should produce volume_factor=2.0. Today's code returns 1.0 for the same scenario.
Fix:      Detect doubleheaders by counting (home_team, away_team) pairs in schedule_today (>1 = doubleheader for that pair). Pass is_doubleheader to compute_volume_factor when the player's team is in a doubleheader pair.
```

#### DCV-A1-004 — `_form_weight` dynamic range drifts from registry and from shared_data_layer

```
ID:       DCV-A1-004
Severity: MED
File:     src/optimizer/daily_optimizer.py:904-905
Symptom:  Three different "recent form blend weight" values across the codebase: daily_optimizer's dynamic cap is 0.25; CONSTANTS_REGISTRY says 0.20; shared_data_layer._RECENT_FORM_WEIGHT_TODAY=0.25.
Expected: Single source of truth in CONSTANTS_REGISTRY, consumed at runtime by all callers. Value should reflect calibrated max blend weight.
Actual:   daily_optimizer.py:904-905:
              _form_weight = min(0.25, 0.10 + (form_games - 7) * 0.02)
          CONSTANTS_REGISTRY (line 167-175): recent_form_blend = 0.20.
          shared_data_layer.py:58: _RECENT_FORM_WEIGHT_TODAY = 0.25.
          daily_optimizer's CLAUDE.md gotcha says "Dynamic weight: linear interpolation from 0.10 (7 games) to scope max (14+ games). Below 7 games = 0 weight." — but doesn't specify the "scope max" value.
Proof:    Direct citation — CONSTANTS_REGISTRY entry says 0.20; daily_optimizer code uses 0.25. Drift between the canonical registry value and the actual runtime cap.
Repro:    Inspect a player with form_games=14: code blends 25% L14 vs 75% preseason. CONSTANTS_REGISTRY would suggest 20% / 80%. shared_data_layer's _RECENT_FORM_WEIGHT_TODAY=0.25 matches the code but not the registry.
Fix:      Decide canonical value (0.20 per registry, OR 0.25 per code) and propagate. Update registry citation. Remove the duplicate in shared_data_layer.
```

#### DCV-A1-005 — `_RAW_SGP_DENOM` hardcoded from assumed team-season volumes

```
ID:       DCV-A1-005
Severity: MED
File:     src/optimizer/daily_optimizer.py:1003
Symptom:  Hardcoded raw-unit SGP denominators (AVG=22 hits, OBP=30 OB events, ERA=31 ER, WHIP=28 BB+H) derived from assumed team-season volumes (5500 AB, 6100 PA, 1400 IP). These assumptions ignore actual league volumes; a roster with 6000 actual AB would be misvalued.
Expected: Derive _RAW_SGP_DENOM from actual league-config sgp_denominators × current team-volume estimates, or read from a calibrated registry entry. Or document the assumption clearly with sensitivity analysis showing the misvaluation bound.
Actual:   Lines 1003-1009 (with derivation in comments 998-1002):
              _RAW_SGP_DENOM = {"AVG": 22.0, "OBP": 30.0, "ERA": 31.0, "WHIP": 28.0}
          Comment derivation:
              AVG: 0.004 × 5500 AB ≈ 22 hits
              OBP: 0.005 × 6100 PA ≈ 30 on-base events
              ERA: 0.20 × 1400 IP / 9 ≈ 31 ER
              WHIP: 0.020 × 1400 IP ≈ 28 walks+hits
Proof:    Math derivation — for a 12-team league with actual 1500 team-season IP (above the assumed 1400):
          ERA: 0.20 × 1500 / 9 ≈ 33.3 ER (vs hardcoded 31).
          A pitcher's annual ERA-SGP would differ by 31/33.3 = -7%. Below the HIGH threshold but still real.
Repro:    Compare actual league IP volume (from team standings totals over a full season) to 1400. Difference scales _RAW_SGP_DENOM linearly, shifting all ERA/WHIP DCV proportionally.
Fix:      Either (a) derive from config.sgp_denominators × actual team-season volumes from standings, or (b) add to CONSTANTS_REGISTRY with current MLB-baseline citation.
```

#### DCV-A1-006 — SP/RP hybrids classified as full starters by `"SP" in positions`

```
ID:       DCV-A1-006
Severity: MED
File:     src/optimizer/daily_optimizer.py:1008
Symptom:  SP/RP hybrid pitchers (positions="SP,RP,P") are flagged as full starters by `_is_starter_pitcher = "SP" in positions`, getting the 1/30 daily fraction. These hybrids actually make occasional spot starts (5-10 per year) and shouldn't be over-weighted as starters.
Expected: Distinguish pure-SP (1/30), pure-RP (1/50), and SP/RP-hybrid (1/40 or weighted average) daily fractions.
Actual:   Line 1008-1009:
              _is_starter_pitcher = "SP" in str(player.get("positions", "")).upper()
              _pitcher_daily_frac = 1.0 / 30.0 if _is_starter_pitcher else 1.0 / 50.0
          Returns 1/30 for any pitcher whose positions string contains "SP", including SP,RP hybrids.
Proof:    A pitcher tagged "SP,RP,P" typically makes ~5-10 starts and ~20-30 relief appearances per season — 25-40 total appearances. Using 1/30 over-weights their daily ERA/WHIP contribution by ~25-50%.
Repro:    Query player pool for `positions LIKE '%SP%' AND positions LIKE '%RP%'`. For each, the current code applies SP daily-fraction. Expected: blended fraction.
Fix:      Detect SP/RP hybrid via `("SP" in pos and "RP" in pos)` and use a blended fraction (e.g., 1/40 or 1/(0.3×30 + 0.7×50) weighted).
```

#### DCV-A1-007 — `STABILIZATION_POINTS["r"]=460` matches OBP value (likely typo)

```
ID:       DCV-A1-007
Severity: MED
File:     src/optimizer/daily_optimizer.py:177
Symptom:  STABILIZATION_POINTS["r"]=460 equals STABILIZATION_POINTS["obp"]=460. Runs stabilization research typically cites 200-300 PA, not 460. Likely a copy-paste error from OBP.
Expected: R stabilization point per FanGraphs research is ~250-300 PA (runs are opportunity-dependent and stabilize faster than rates like OBP).
Actual:   Lines 176-188:
              STABILIZATION_POINTS: dict[str, float] = {
                  "r": 460,    # ← same as OBP, likely typo
                  "hr": 170,
                  ...
                  "obp": 460,
                  ...
              }
          Note CONSTANTS_REGISTRY does NOT have stabilization_r entry, so no canonical citation exists.
Proof:    FanGraphs stabilization series (cited in registry for HR/AVG/OBP/etc.) gives different points for different stats. R = OBP by exact value is statistically improbable; likely a copy-paste during dict authoring.
Repro:    compute_blended_projection for R with observed counts; the prior weighting will be miscalibrated. Effect bounded because R is a counting stat and the function may not actually use STABILIZATION_POINTS["r"] depending on stat_key normalization.
Fix:      Verify intended R stabilization point from FanGraphs research (likely ~250) and update. Add to CONSTANTS_REGISTRY with citation.
```

#### DCV-A1-008 — STABILIZATION_POINTS for W/L/SV conceptually unsuitable

```
ID:       DCV-A1-008
Severity: MED
File:     src/optimizer/daily_optimizer.py:182-184
Symptom:  STABILIZATION_POINTS for w=200, l=200, sv=100. W/L/SV are discrete outcomes per appearance, not rates that stabilize. Bayesian rate-blend using these values is conceptually misapplied.
Expected: Either (a) document why these are stabilization points despite W/L/SV not being rates (likely "PA-equivalent" for opportunity), with citation, or (b) skip the Bayesian blend for these stats and use raw counts.
Actual:   Lines 182-184:
              "w": 200,
              "l": 200,
              "sv": 100,
          No citation. compute_blended_projection treats these as rate stabilization points.
Proof:    Bayesian rate stabilization works for rate stats (HR rate, K rate, ERA = ER/9IP, etc.) because rates have well-defined regression-to-mean behavior. W/L/SV are binary outcomes per appearance multiplied by opportunity (starts, save situations). Treating them as rates is a categorical mismatch.
Repro:    A pitcher who has accumulated 3 W in 8 starts (rate 0.375) gets blended with a presumed prior. The "prior rate" implied by STABILIZATION_POINTS is opaque — is it 0.375 wins/start? Where is that calibrated from?
Fix:      Either move W/L/SV out of compute_blended_projection (use raw counts) OR document the rate denominator (per start? per appearance? per opportunity?) with citation.
```

#### DCV-A1-009 — League-avg xFIP=4.20 hardcoded in pitcher quality multiplier

```
ID:       DCV-A1-009
Severity: MED
File:     src/optimizer/daily_optimizer.py:376
Symptom:  Line 376: `quality = max(0.5, min(2.0, 2.0 - pitcher_xfip / 4.20))`. The 4.20 is league-avg xFIP, hardcoded. Current MLB xFIP varies yearly (~4.05 in 2023, ~4.15 in 2024, ~4.30 in some seasons). A 0.20 shift in baseline changes hitter matchup multiplier meaningfully.
Expected: Read current league-avg xFIP from team_strength aggregation or from a CONSTANTS_REGISTRY entry updated annually.
Actual:   Hardcoded 4.20.
Proof:    For a pitcher at xFIP=3.50:
          - With baseline 4.20: quality = 2.0 - 3.50/4.20 = 1.167. Hitter mult ×= 1/1.167 = 0.857.
          - With baseline 4.00: quality = 2.0 - 3.50/4.00 = 1.125. Hitter mult ×= 1/1.125 = 0.889.
          - Diff: 3.7% in hitter matchup multiplier. Below HIGH threshold but real and unnecessary.
Repro:    Compute current MLB league-avg xFIP from pool. Compare to 4.20. Multiply expected diff into hitter DCV.
Fix:      Add league_avg_xfip to CONSTANTS_REGISTRY with annual update procedure, or derive from team_strength on every call.
```

#### DCV-A1-010 — `check_ip_override` `ip_minimum=20.0` uncited

```
ID:       DCV-A1-010
Severity: MED
File:     src/optimizer/daily_optimizer.py:1188-1192
Symptom:  check_ip_override default ip_minimum=20.0 forces a pitcher start when weekly IP < 20. FourzynBurn has no explicit per-matchup IP minimum (Yahoo allows 0 IP, you just lose pitching cats). Why 20?
Expected: Either (a) document why 20 IP is the threshold (heuristic for "enough K/W/SV opportunity"), with citation; OR (b) make it user-configurable via league settings.
Actual:   Line 1192: `ip_minimum: float = 20.0` with no citation.
Proof:    FourzynBurn settings (CLAUDE.md "League Context") show no IP floor. The 20 IP threshold is purely heuristic.
Repro:    A team projecting 18 IP this week with all pitchers off today would get a force-start boost. Threshold change to 15 or 25 would alter behavior without principled basis.
Fix:      Document the heuristic ("avoid pitching streamer panic when projected IP gives reasonable opportunity"). Or make it a constant in CONSTANTS_REGISTRY with sensitivity bound.
```

#### DCV-A1-011 — `apply_ip_pace_scaling` doesn't include `dcv_l` (losses)

```
ID:       DCV-A1-011
Severity: MED
File:     src/optimizer/daily_optimizer.py:1283-1287
Symptom:  apply_ip_pace_scaling scales K/W/SV down when IP budget is near-exhausted but does NOT scale L (losses). L is inverse — its DCV contribution is negative. Scaling K/W/SV down while leaving L's negative contribution intact over-penalizes the pitcher relative to their actual remaining opportunity.
Expected: Scale all pitcher counting stats (K, W, SV, L) consistently. Reduced IP opportunity reduces ALL counting outcomes — both good (W/K/SV) and bad (L).
Actual:   Line 1283: `counting_cols = ["dcv_k", "dcv_w", "dcv_sv"]`. L is missing.
Proof:    For a pitcher at 85% IP budget, code scales K/W/SV by ~0.7. dcv_l stays at full magnitude. The pitcher's total_dcv is now disproportionately dragged down by L's unscaled contribution.
Repro:    A pitcher with dcv_k=2.0, dcv_w=0.5, dcv_sv=0, dcv_l=-1.0 at 85% IP usage:
          - Before scaling: total = 2.0 + 0.5 + 0 - 1.0 = 1.5
          - After scaling (k/w/sv × 0.7): total = 1.4 + 0.35 + 0 - 1.0 = 0.75
          - Correct scaling (l × 0.7 too): total = 1.4 + 0.35 + 0 - 0.7 = 1.05
          Difference: 30% under-valuation of the pitcher.
Fix:      Add "dcv_l" to counting_cols. Loss scaling logic should mirror W (proportional to IP opportunity).
```

#### DCV-A2-001 — `default_team_weekly_ip` drifts from `_STREAM_IP_TARGET`

```
ID:       DCV-A2-001
Severity: MED
File:     src/optimizer/constants_registry.py:260-268
Symptom:  CONSTANTS_REGISTRY entry default_team_weekly_ip=55.0; CLAUDE.md gotcha and streaming.py constant _STREAM_IP_TARGET=54. Off by 1 IP.
Expected: Single value in CONSTANTS_REGISTRY consumed by all callers. Registry is the canonical source per its docstring.
Actual:   Registry: 55.0. CLAUDE.md (gotcha "Pitcher leave-empty threshold — `_PITCHER_EMPTY_THRESHOLD = _pitcher_median * 0.20`") + streaming.py constants: 54. apply_ip_pace_scaling default: 55.0 (line 1245).
Proof:    Direct citation — CLAUDE.md line ~268 gotcha section documents 54; CONSTANTS_REGISTRY says 55.
Fix:      Decide canonical value. Update registry citation. Make streaming.py and apply_ip_pace_scaling read from registry.
```

#### DCV-A2-002 — `streaming_baseline_whip` drifts from `_LEAGUE_AVG_WHIP`

```
ID:       DCV-A2-002
Severity: MED
File:     src/optimizer/constants_registry.py:269-277
Symptom:  CONSTANTS_REGISTRY streaming_baseline_whip=1.25; CLAUDE.md Wave 8d constant _LEAGUE_AVG_WHIP=1.30. Off by 0.05 WHIP.
Expected: Single value, registered, cited.
Actual:   Registry: 1.25 (cited "2022-2024: ~1.22-1.28"). Wave 8d: 1.30 in streaming.py/war_room_hotcold.py.
Proof:    CLAUDE.md Wave 8d Batch 5: "_LEAGUE_AVG_ERA/WHIP/WOBA" promoted to named constants. The values listed there don't match CONSTANTS_REGISTRY values.
Fix:      Reconcile. If 1.30 is current league avg, update registry. If registry's 1.25 is intended baseline, update streaming.
```

#### DCV-A2-003 — `apply_ip_pace_scaling` weekly_ip_target default mismatch

```
ID:       DCV-A2-003
Severity: MED
File:     src/optimizer/daily_optimizer.py:1245
Symptom:  Function signature default `weekly_ip_target: float = 55.0` matches CONSTANTS_REGISTRY default_team_weekly_ip=55.0 but contradicts CLAUDE.md _STREAM_IP_TARGET=54. Same drift as DCV-A2-001.
Expected: Consume from CONSTANTS_REGISTRY at call time.
Actual:   Hardcoded 55.0 in signature.
Proof:    See DCV-A2-001.
Fix:      Read from CONSTANTS_REGISTRY.
```

#### DCV-A2-004 — Playoff weeks may not match FourzynBurn 26-week season

```
ID:       DCV-A2-004
Severity: MED
File:     src/optimizer/shared_data_layer.py:51
Symptom:  `_PLAYOFF_WEEKS = {21, 22, 23, 24}` — these are the weeks getting a 1.15× schedule premium. FourzynBurn is a 26-week season (per SF-42 fix), top-4 playoff (per SF-49). The playoff weeks for a 26-week season with top-4 should likely be weeks 23-26 or similar, not 21-24.
Expected: Playoff weeks should be parameterized by league config (FourzynBurn season_weeks=26, playoff_spots=4 → playoff weeks 23-26 or 24-26 depending on round structure).
Actual:   Hardcoded {21, 22, 23, 24}.
Proof:    Direct citation — CLAUDE.md Wave 5 SF-49: "playoff_sim hardcoded season_weeks = 22.0 and _PLAYOFF_SPOTS = 6 contradicted CLAUDE.md (FourzynBurn: 26-week season, top-4 playoff) → counting-stat weekly projections off ~18%, playoff_pct over-counted by 50%; both fixed." The fix was in playoff_sim.py; shared_data_layer's _PLAYOFF_WEEKS may have the same drift.
Fix:      Verify FourzynBurn playoff structure. If playoffs are weeks 23-26 (or similar), update _PLAYOFF_WEEKS. Parameterize by config if possible.
```

#### DCV-A5-001 — Rate stat thresholds in `classify_rate_stat_mode` uncited

```
ID:       DCV-A5-001
Severity: MED
File:     src/optimizer/category_urgency.py:181-186
Symptom:  Hardcoded thresholds for protect/abandon (ERA: 0.30/-0.50, WHIP: 0.05/-0.08, AVG: 0.020/-0.030, OBP: 0.020/-0.030). Comment says "Thresholds calibrated from research" but no citation.
Expected: Each threshold cited to a source or calibrated value. Asymmetry between protect (smaller gap) and abandon (larger gap) is design — should be documented.
Actual:   Lines 181-186, comment "Thresholds calibrated from research:" without research citation.
Proof:    The asymmetry (protect threshold smaller than abandon threshold's absolute value) implies a risk-averse design: easier to commit to protect than to abandon. This is a calibration choice that affects which categories get "abandoned" (zeroing pitcher ERA/WHIP DCV).
Fix:      Add citations or document the calibration method.
```

#### DCV-A5-002 — Urgency floor 0.10 breaks pure sigmoid; design choice but uncited

```
ID:       DCV-A5-002
Severity: MED
File:     src/optimizer/category_urgency.py:136-141
Symptom:  Urgency is floored at 0.10 even for categories being won comfortably. This is a deliberate design choice (to prevent zeroing) but isn't cited and affects every DCV computation.
Expected: Document the design rationale (preventing post-hoc multiplication from collapsing winning categories to zero) and any calibration of the 0.10 specific value.
Actual:   Lines 136-141:
              # Floor at 0.10: even categories being won comfortably should
              # contribute ~10% of their base DCV.  Without this floor, the
              # post-hoc multiplication crushes pitcher counting stats...
              urgency[cat] = max(0.10, raw)
          Comment explains *why* but not *why 0.10 specifically* (vs 0.05 or 0.15).
Proof:    A category being won by gap=5 (large) gives raw urgency ≈ 4×10⁻⁵. The floor of 0.10 means the urgency multiplier is 2500× the true sigmoid value. This isn't wrong — it's a deliberate de-emphasis ceiling — but the specific 0.10 is calibrated by feel, not by research.
Fix:      Document the calibration ("0.10 chosen because pitcher dcv_k at scale-100 mSGP per day rounds to 0.01 below this floor"). Consider adding to CONSTANTS_REGISTRY.
```

### LOW (backlog — fix when touched)

#### DCV-A1-012 — Hitter daily fraction 1/162 vs typical 1/145

```
ID:       DCV-A1-012
Severity: LOW
File:     src/optimizer/daily_optimizer.py:1007
Symptom:  _hitter_daily_frac = 1.0 / 162.0. Typical hitter plays 140-145 games not 162. Daily contribution scaled by full-season denominator under-weights hitter DCV by ~10%.
Proof:    Average regular hitter games-played: ~145 (4-5 days off, 5-10 days IL). Use 1/145 for tighter calibration.
Fix:      Replace 162.0 with 145.0 or derive from player's expected games-played.
```

#### DCV-A1-013 — Sprint speed thresholds uncited

```
ID:       DCV-A1-013
Severity: LOW
File:     src/optimizer/daily_optimizer.py:1107-1110
Symptom:  Sprint speed boost thresholds (29.0, 28.0) and multipliers (1.10, 1.05) hardcoded without citation.
Proof:    Statcast league-avg sprint speed is ~27.0. 28.0 = "fast," 29.0 = "elite." Thresholds reasonable but uncited.
Fix:      Add citation (Statcast leaderboards) and consider registering.
```

#### DCV-A1-014 — STUD_FLOOR_TOP_N=8 uncited

```
ID:       DCV-A1-014
Severity: LOW
File:     src/optimizer/daily_optimizer.py:191
Symptom:  STUD_FLOOR_TOP_N = 8 — top-8 protected players. No citation for why 8.
Proof:    FourzynBurn roster is 23 active + 5 bench. Top-8 = top ~28% of starters. Reasonable but uncited.
Fix:      Document rationale or derive from roster size.
```

#### DCV-A1-015 — `_REPL_*` not in CONSTANTS_REGISTRY (paired with HIGH DCV-A1-001)

```
ID:       DCV-A1-015
Severity: LOW
File:     src/optimizer/daily_optimizer.py:993-996
Symptom:  Structural follow-up to DCV-A1-001: even after fixing the values, they should be in CONSTANTS_REGISTRY for sensitivity_analysis.
Proof:    Sensitivity_analysis.py perturbs every CONSTANTS_REGISTRY entry. _REPL_* are not perturbable today.
Fix:      Add 4 entries to CONSTANTS_REGISTRY with bounds (e.g., _REPL_AVG: 0.230-0.250).
```

#### DCV-A1-016 — Comment "Scale: ~1.0 per 40 wRC+ points" misleading

```
ID:       DCV-A1-016
Severity: LOW
File:     src/optimizer/daily_optimizer.py:383
Symptom:  Comment says "Scale: ~1.0 per 40 wRC+ points" but actual divisor is 80, giving 0.5 per 40 wRC+. Cosmetic.
Fix:      Update comment to match formula.
```

#### DCV-A1-017 — `matchup_weights = 0.5 + urgency` offset uncited

```
ID:       DCV-A1-017
Severity: LOW
File:     src/matchup_context.py:432
Symptom:  `matchup_weights = {cat: 0.5 + urgency for cat in all_cats}`. Why +0.5 offset? Effectively makes winning-category floor = 0.60 weight, soft-de-emphasizing them.
Proof:    Without the offset, urgency-as-weight would give range [0.10, 1.00]. With offset, [0.60, 1.50]. Design choice but uncited.
Fix:      Document or remove the offset; pick canonical weight range.
```

#### DCV-A2-005 — Mode normalization inconsistency in `get_category_weights`

```
ID:       DCV-A2-005
Severity: LOW
File:     src/matchup_context.py:487-504
Symptom:  "blended" mode normalizes to mean=1.0; "matchup" and "standings" modes don't normalize, just clamp to non-negative. Cross-mode comparisons not meaningful.
Fix:      Apply normalization consistently across all 3 modes.
```

#### DCV-A4-001 — `_FINAL_STATUSES` inconsistent with daily_optimizer's locked_teams set

```
ID:       DCV-A4-001
Severity: LOW
File:     src/game_day.py:24 + src/optimizer/daily_optimizer.py:644
Symptom:  game_day uses {"final", "game over", "completed early"}; daily_optimizer uses {"in progress", "final", "game over", "completed"}. Slight mismatch on "completed" vs "completed early".
Proof:    A game with status "completed early" is final per game_day but not locked per daily_optimizer (string mismatch). Edge case — rain-shortened games.
Fix:      Centralize the FINAL_STATUSES set in one module.
```

#### DCV-A3-001 — shared_data_layer teams list uses "CHW" not "CWS"

```
ID:       DCV-A3-001
Severity: LOW
File:     src/optimizer/shared_data_layer.py:193 (line 187 starts the `teams` list)
Symptom:  Hardcoded teams list contains "CHW" (legacy); FourzynBurn uses "CWS" (Wave 1 / SF-57 canonicalization).
Proof:    Mismatch with canonical naming. May be handled by canonicalize_team but creates inconsistency.
Fix:      Replace "CHW" → "CWS". Verify "OAK" → "ATH" too.
```

#### DCV-A6-001 — Default team strength fallback values are optimistic

```
ID:       DCV-A6-001
Severity: LOW
File:     src/matchup_context.py:163-171
Symptom:  Fallback when get_team_strength fails: team_era=4.00, team_whip=1.25. League avg is closer to 4.20 / 1.27. Fallback values are slightly optimistic for "league average."
Fix:      Update fallback to current league averages.
```

#### DCV-A1-018 — `apply_stud_floor` floor multiplier 1.5× median uncited

```
ID:       DCV-A1-018
Severity: LOW
File:     src/optimizer/daily_optimizer.py:488
Symptom:  Floor value = median_dcv × 1.5 × matchup_mult. Why 1.5? Arbitrary.
Fix:      Cite or replace with derived value.
```

#### DCV-A1-019 — `_CORRELATION_PAIRS` uncited

```
ID:       DCV-A1-019
Severity: LOW
File:     src/optimizer/shared_data_layer.py:33-43
Symptom:  Hardcoded category correlation values (HR-R=0.72, HR-RBI=0.68, etc.) without citation. Used for dampening correlated categories during urgency adjustment.
Fix:      Cite (FanGraphs correlations? empirical from FourzynBurn standings?) or register.
```

## Open Questions

These cannot be resolved without your input on FourzynBurn-specific behavior or design intent:

1. **OQ-1 (DCV-A1-001):** What is the source of the `_REPL_*` baselines (0.240/0.305/4.50/1.35)? Were they calibrated against FourzynBurn league standings data, or borrowed from a generic 12-team H2H reference? If borrowed, where from?

2. **OQ-2 (DCV-A1-003):** Should doubleheader days produce 2.0× volume_factor for confirmed starters? Yahoo H2H counts both games' stats — but the LP solver picks one daily lineup that's then unchanged for both games. Does doubling DCV reflect actual production or double-count slot-fixed decisions?

3. **OQ-3 (DCV-A1-010):** What is the right `ip_minimum` for `check_ip_override`? FourzynBurn has no per-matchup IP minimum (Yahoo allows 0). Is 20 IP a fantasy-industry heuristic for "enough opportunity" or arbitrary?

4. **OQ-4 (DCV-A1-008):** Should W/L/SV be in `STABILIZATION_POINTS` at all? These are discrete-outcome counting stats, not rates. If yes, what's the rate denominator (per start, per appearance, per opportunity)?

5. **OQ-5 (DCV-A2-004):** What are FourzynBurn's playoff weeks? `_PLAYOFF_WEEKS = {21, 22, 23, 24}` may not align with the 26-week season + top-4 playoff structure (SF-49 fixed playoff_sim's hardcoded values; this is a separate site).

6. **OQ-6 (DCV-A5-002):** Is the urgency floor of 0.10 calibrated to a specific behavior (e.g., minimum pitcher counting-stat visibility) or chosen by feel? Documented either way.

## Fix-Wave Proposal

### Wave 11A — HIGH fixes (recommended first)

- **Findings:** DCV-A1-001, DCV-A1-002
- **Estimated LOC delta:**
  - DCV-A1-001: +30 lines (4 registry entries + citations + read at call site) + ~5 lines per perturbation test
  - DCV-A1-002: +15 lines (replace `for cat in config.all_categories: sgp += val/denom` with `sgp = sgp_calc.totals_sgp(...)`) + structural-invariant test
- **Estimated test count:** 4 new tests (rate-stat DCV regression with each _REPL_* perturbed; stud-floor ranking regression with hitter A vs B example from proof)
- **Estimated effort:** 1-2 sessions, TDD per fix
- **Dependencies:** Resolve OQ-1 (what are the right _REPL_* values?) before fixing values; the structural fix (move to registry + read from registry) can proceed without resolving the value question.

### Wave 11B — MED fixes (should-fix before next H2H)

- **Findings:** DCV-A1-003 (doubleheader), DCV-A1-004 (form weight drift), DCV-A1-005 (RAW_SGP_DENOM), DCV-A1-006 (SP/RP hybrid daily frac), DCV-A1-007 (R stabilization), DCV-A1-008 (W/L/SV stabilization), DCV-A1-009 (xFIP baseline), DCV-A1-010 (ip_minimum), DCV-A1-011 (apply_ip_pace_scaling missing L), DCV-A2-001..003 (registry drift), DCV-A2-004 (playoff weeks), DCV-A5-001..002 (uncited thresholds + urgency floor)
- **Estimated LOC delta:** ~100 lines across 4 files + ~15 new tests
- **Estimated effort:** 2-3 sessions
- **Dependencies:** Resolve OQ-2 (doubleheader semantics), OQ-3 (ip_minimum), OQ-4 (W/L/SV stabilization), OQ-5 (playoff weeks), OQ-6 (urgency floor)

### Backlog — LOW (12 findings)

- Address opportunistically when touching adjacent code. No wave commitment.
- Worth doing for code health and audit hygiene but no operational impact on lineup decisions.

## Indisputability Statements

Both HIGH findings carry their proofs inline in the `Proof:` field:

- **DCV-A1-001**: Math derivation showing `_REPL_AVG` sensitivity = 16-33% per 0.005 shift on a typical hitter. Sign-flip behavior demonstrated for replacement-level hitters.
- **DCV-A1-002**: Direct citation of CLAUDE.md SF-25 ("SGPCalculator.totals_sgp is sole SGP path") + math derivation showing rate-stat dominance (45:1 over counting stats) in stud-floor ranking.

Demotion rule was not triggered. Both HIGH findings carry proof artifacts; neither is "suspicion without evidence."

## What's Likely Correct (Audit Confidence Boosters)

For balance, the audit confirmed these are working correctly:

- **Sigmoid urgency math** (category_urgency.py:122-141) — correct sign, correct k-value runtime read from CONSTANTS_REGISTRY (SF-39 confirmed in place).
- **IL/SUSP exclusion** (daily_optimizer.py:235-265) — Wave 10 SUSP fix verified; all documented IL statuses present.
- **Locked-teams logic** (daily_optimizer.py:626-665) — correctly zeroes volume for in-progress/final games.
- **SP gate** (daily_optimizer.py:758-781) — Wave 10 fix verified; correctly distinguishes pure-SP, SP/RP hybrid, pure-P, and pure-RP cases.
- **Weighted rate-stat aggregation in DCV inner loop** (daily_optimizer.py:1044-1079) — correct math (`(repl_er - er) / DENOM` is the right form; sign correct for inverse stats).
- **`get_target_game_date`** (game_day.py:27-57) — DST-aware ET timezone, correct auto-advance logic, includes "completed early" status.
- **Recursion guard** (daily_optimizer.py:1147-1170) — `_retry_attempted` bounds recursion to one retry; recovers from all-zero DCV with sensible fallback.
- **Stuff+/FIP K-boost** (`_stuff_plus_k_multiplier`) — well-cited (SF-6), with safe fallback chain (Stuff+ → xFIP → FIP → neutral 1.0).

These cover the most performance-critical and previously-bug-prone code paths. The HIGH findings are concentrated in (1) leftover pre-SF-25 SGP math in apply_stud_floor and (2) replacement-level baselines lacking citation/registry.
