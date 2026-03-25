"""Enhanced Lineup Optimizer package.

Provides advanced lineup optimization with:
- Enhanced projections (Bayesian, Kalman, injury modeling)
- Matchup adjustments (park factors, platoon splits, opposing pitcher quality)
- H2H weekly opponent targeting + season-long optimization
- Non-linear SGP category targeting + pitcher streaming
- Stochastic optimization (mean-variance, CVaR tail-risk protection)
- Dual H2H/season-long objective
- Advanced LP formulations (maximin, epsilon-constraint, stochastic MIP)
- Master pipeline orchestrator (quick/standard/full modes)
"""

from src.optimizer.projections import build_enhanced_projections

__all__ = [
    "build_enhanced_projections",
]

# Lazy imports — these are available but not imported at package level
# to avoid circular imports and optional dependency issues:
#
# from src.optimizer.matchup_adjustments import compute_weekly_matchup_adjustments
# from src.optimizer.h2h_engine import compute_h2h_category_weights
# from src.optimizer.sgp_theory import compute_nonlinear_weights
# from src.optimizer.streaming import rank_streaming_candidates
# from src.optimizer.scenario_generator import generate_stat_scenarios
# from src.optimizer.dual_objective import blend_h2h_roto_weights
# from src.optimizer.advanced_lp import maximin_lineup
# from src.optimizer.pipeline import LineupOptimizerPipeline
