"""Context engine sub-package for trade evaluation.

Spec reference: Section 17 Phase 4 items 17-20

Phase 4 adds environmental and roster-level context to trade analysis:
  - Game-level matchup adjustments (Log5 batter-pitcher)
  - Injury stochastic process (Weibull duration sampling)
  - Enhanced bench/flexibility option value
  - Roster concentration risk (Herfindahl index)

Wires into:
  - src/engine/output/trade_evaluator.py: master orchestrator
  - src/engine/monte_carlo/trade_simulator.py: MC sim overlay
  - src/injury_model.py: health scores and age-risk curves
"""
