"""Trade Analyzer Engine — Phases 1-4.

Implements the roster-portfolio trade evaluation pipeline from trade-analyzer-spec.md.
  Phase 1: Z-score+SGP valuation, marginal category elasticity, gap analysis,
           punt detection, LP lineup optimization, deterministic trade grading.
  Phase 2: Bayesian Model Averaging, KDE marginals, Gaussian copula,
           10K paired Monte Carlo simulations.
  Phase 3: Statcast harvesting, exponential signal decay, Kalman filter,
           BOCPD changepoint detection, HMM regime classification.
  Phase 4: Log5 matchup engine, injury stochastic process (Weibull),
           enhanced bench option value, roster concentration risk (HHI).
"""
