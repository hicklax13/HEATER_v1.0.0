# Trade Engine V3 — Implementation Plan

## Context

The trade finder produces bad recommendations because: (1) ADP is a binary filter, not a scoring component — opponents won't accept trades where they give up a higher-drafted player regardless of stats; (2) only 1-for-1 trades are supported — no 2-for-1 or 2-for-2; (3) no opponent perspective modeling — the engine doesn't know what categories the opponent needs; (4) no schedule awareness — trade urgency should depend on upcoming opponent difficulty; (5) category correlations are ignored — the power cluster (HR/R/RBI at r=0.86) gets triple-counted in SGP.

**Key user requirement:** "My league mates care about ADP deeply. They are less inclined to accept a trade that is statistically in their favor if they give up someone drafted above what they are getting." ADP must be a PRIMARY factor, not just a filter.

## Architecture

## 8 Tasks with Agent Assignments

```
Parallel Group A (start simultaneously):
  Agent A: Task 1 (ADP-weighted scoring) + Task 2 (multi-player expansion)
  Agent B: Task 3 (opponent perspective) + Task 4 (schedule urgency)
  Agent C: Task 5 (correlation adjustments)

Sequential:
  Agent C: Task 6 (enhanced acceptance — needs Tasks 1, 3, 5)
  Agent D: Task 7 (page tab enhancements — needs Tasks 1-6)
  Agent D: Task 8 (tests — needs Tasks 1-7)

```

---

## Task 1: ADP-Weighted Trade Scoring (Agent A)
**File:** `src/trade_finder.py` — modify `scan_1_for_1()` composite score

- Add `compute_adp_fairness(give_id, recv_id, pool)` → float 0-1 (1.0 = same draft round, 0.0 = extreme mismatch)
- Uses league draft round (primary) with generic ADP fallback
- New composite: 40% user_delta + **20% adp_fairness** + 20% acceptance + 10% opp_benefit + 10% need_match
- Add ADP columns to trade result dict: `give_adp_round`, `recv_adp_round`, `adp_fairness`

---

## Task 2: Multi-Player Trade Expansion (Agent A)
**File:** `src/trade_finder.py` — new functions

- `scan_2_for_1(seeds, user_ids, opp_ids, pool)` — greedy expansion from top 1-for-1 seeds
- `scan_1_for_2(seeds, ...)` — reverse direction
- `scan_2_for_2(seeds, ...)` — combine both
- Cap 3 per side, max 50 expansions per seed
- Roster spot value: `ROSTER_SPOT_SGP = 0.8` (bonus for 2-for-1, penalty for 1-for-2)
- Drop cost = SGP of worst bench player minus replacement FA
- Wire into `find_trade_opportunities()` as Tier 2 after existing Tier 1

---

## Task 3: Opponent Perspective Module (Agent B)
**New file:** `src/opponent_trade_analysis.py`

- `compute_opponent_needs(opp_team, standings, config)` — reuses `category_gap_analysis()` from opponent's perspective
- `analyze_from_opponent_view(trade, opp_team, standings, ...)` — per-category impact for opponent
- `compute_opponent_acceptance(trade, opp_needs, adp_fairness, archetype)` — enhanced acceptance with ADP + needs + archetype
- `get_opponent_archetype(team_name)` — maps from `OPPONENT_PROFILES` to trade_willingness (0-1)

---

## Task 4: Schedule-Aware Trade Urgency (Agent B)
**File:** `src/trade_intelligence.py` — add functions

- `compute_schedule_urgency(weeks_ahead=3, yds=None)` → multiplier [0.85, 1.25]
- Tier 1 opponents = +urgency, Tier 4 = -urgency
- Applied as final modifier in `find_trade_opportunities()`

---

## Task 5: Category Correlation Adjustments (Agent C)
**File:** `src/trade_intelligence.py` — add function

- `apply_correlation_adjustments(sgp_dict)` → adjusted SGP dict
- Power cluster (HR/R/RBI): 17% discount (r=0.86 over-counting)
- SB: 8% premium (independent, r<0.12)
- AVG/OBP cluster: 10% discount (r=0.70)
- ERA/WHIP cluster: 12% discount (r=0.84)
- Integrated into `_weighted_totals_sgp()` in trade_finder.py

---

## Task 6: Enhanced Acceptance Model (Agent C)
**File:** `src/trade_finder.py` — modify `estimate_acceptance_probability()`

- Loss aversion: 1.5 → **1.8** (meta-analysis consensus)
- NEW param: `adp_fairness` — penalty when opponent gives up higher-drafted player
- NEW param: `opponent_need_match` — boost when trade fills opponent's weak categories
- NEW param: `opponent_standings_rank` — bubble teams (4th-8th) get +30% acceptance
- NEW param: `opponent_trade_willingness` — from archetype (0.3 passive → 0.7 active)
- All new params have defaults for backward compatibility

---

## Task 7: Trade Finder Page Tab Enhancements (Agent D)
**File:** `pages/10_Trade_Finder.py`

- `_build_trade_df()`: add Type (1-for-1/2-for-1/etc), Give ADP, Recv ADP, ADP Fair columns
- **By Partner tab**: add opponent need indicators, archetype tier, schedule urgency badge
- **By Category Need tab**: add "Opponent Also Benefits" indicator
- **By Value tab**: include multi-player trades, ADP fairness column
- **Trade Readiness tab**: add ADP Tier column (Elite/Core/Depth/Filler), opponent need match score

---

## Task 8: Tests (Agent D)
**New files:** `tests/test_multi_player_trades.py`, `tests/test_opponent_analysis.py`, `tests/test_schedule_urgency.py`, `tests/test_correlation_adjustments.py`

~35 tests covering: ADP fairness scoring, multi-player roster accounting, opponent needs, acceptance model params, schedule urgency bounds, correlation discounts, backward compatibility

---

## Critical Files

| File | Agent | Action |
|---|---|---|
| `src/trade_finder.py` | A, C | Scoring, multi-player, acceptance |
| `src/opponent_trade_analysis.py` | B | NEW — opponent perspective |
| `src/trade_intelligence.py` | B, C | Schedule urgency, correlation |
| `pages/10_Trade_Finder.py` | D | All 4 tab enhancements |
| `src/opponent_intel.py` | B | Reuse profiles + schedule |
| `src/engine/portfolio/category_analysis.py` | B | Reuse gap analysis |
| `src/database.py` | — | Reuse `get_player_draft_round()`, `league_draft_picks` |

---

## Data Sources (all live from Yahoo)

| Data | Source | Used By |
|---|---|---|
| All 12 rosters + IL/DTD/NA | `yds.get_rosters()` | All tasks |
| League standings (per-category) | `yds.get_standings()` | Tasks 3, 5, 6 |
| League draft results | `yds.get_draft_results()` → `league_draft_picks` | Tasks 1, 6 |
| Generic ADP | `adp` table (558 entries) | Task 1 (fallback) |
| Free agent pool | `yds.get_free_agents()` | Task 2 (pickup value) |
| Transactions | `yds.get_transactions()` | Task 2 (roster moves) |
| Schedule | `yds.get_schedule()` + `OPPONENT_PROFILES` | Task 4 |
| Matchup scores | `yds.get_matchup()` | Context display |

**Gap:** League draft data may not be populated. First call to `yds.get_draft_results()` triggers Yahoo fetch + write-through to `league_draft_picks` table.

---

## Verification

1. `python -m pytest tests/test_multi_player_trades.py tests/test_opponent_analysis.py tests/test_schedule_urgency.py tests/test_correlation_adjustments.py -v`
2. `python -m pytest tests/test_trade_finder.py tests/test_trade_intelligence.py -v` (regression)
3. `python -m pytest -x -q` (full suite, 2242+ tests)
4. Start app → Trade Finder → verify all 4 tabs show new columns
5. Verify ADP fairness appears in By Value tab
6. Verify multi-player trades appear with Type column
7. Verify no trades recommending Rd 2 picks for Rd 16 picks
8. `git push origin master` → CI green
