"""Engine signals submodules.

ARCHITECTURAL BOUNDARY (BUG-005):
These submodules (decay, kalman, regime, statcast) are consumed by
`src/trade_signals.py` (the Trade-Readiness UI helper) — NOT by the
core trade-engine evaluation pipeline (trade_evaluator, trade_simulator,
bayesian_blend). The original Phase 3 architecture diagram promised
"Kalman feeds Bayesian blend" but the wiring was never completed.

If you want recency-weighted decay or Kalman-filtered true-talent
estimates in the trade engine, you must EXPLICITLY route their output
into `src/engine/projections/bayesian_blend.py` via a new adapter — do
not just import these modules from the trade evaluator. The
`tests/test_engine_signals_architectural_boundary.py` guard enforces
this boundary to prevent silent "looks wired but doesn't work" regressions.
"""
