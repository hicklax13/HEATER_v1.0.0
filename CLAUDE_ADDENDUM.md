
---

## TRADE ANALYZER ENGINE (Medallion-Grade)

### Algo Spec Location
The AUTHORITATIVE algorithm specification is: `/docs/trade-analyzer-spec.md`
Read it IN FULL before writing any trade analyzer code.

### Critical Rules for Trade Analyzer
1. NEVER use placeholder data or mock projections. Every number traces to Yahoo API, Statcast, or FanGraphs.
2. NEVER simplify the math from what the spec defines. If you cannot implement a layer, skip it entirely and move to the next. Do NOT substitute a simpler approximation without flagging it.
3. ALL trade evaluations must use MY TEAM's actual roster, standings, and category totals from the live Yahoo API. Not generic league averages.
4. Punt detection must zero out marginal value for hopeless categories.
5. Lineup optimizer must respect Yahoo's multi-position eligibility per player.
6. Opportunity cost (best FA at vacated slot) must be subtracted from every trade evaluation.
7. Implement in the phase order from the spec: Phase 1 first, each phase independently useful.

### Trade Analyzer File Structure (within existing project)
```
/src/engine/               # NEW - core algorithm modules
  /signals/                # L0-L1
  /regime/                 # L2: BOCPD + HMM
  /projections/            # L3: Bayesian projection engine
  /dependencies/           # L4: copula + covariance
  /injury/                 # L5: Cox hazard
  /matchup/                # L6: game-level projections
  /portfolio/              # L7: roster optimizer + category analysis
  /game_theory/            # L8: opponent modeling
  /dynamics/               # L9: Bellman + real options
  /simulation/             # L10: Monte Carlo
  /output/                 # L11: grading + reporting
/pages/                    # Existing Streamlit pages (Trade Analyzer page already exists)
/data/                     # Existing data directory
/tests/                    # Existing tests directory
```

### Test Cases
The spec defines 5 acceptance test cases. All must pass before a phase is considered complete.
