"""Cross-boundary TypedDict definitions for trade engine output.

These TypedDicts document the shape of dicts returned by trade-engine
functions across module boundaries. They are *non-total* so existing
callers that omit optional MC/context/game-theory keys still type-check.

Wave 8c (audit findings YV-006/D4D-002/D7D-003..012):
The 35-key ``evaluate_trade`` return, the 10-key matchup result, and the
10-section player-card payload were documented in prose only. Consumers
fished via ``.get(key, default)`` and typos silently returned ``None``.
Promoting these to TypedDicts makes the schema machine-checkable without
changing runtime behavior (TypedDict is a structural type-hint only).
"""

from __future__ import annotations

from typing import Any, TypedDict


class GradeRange(TypedDict):
    """Output of ``_compute_grade_range`` — grade with uncertainty band."""

    grade: str
    grade_low: str
    grade_high: str
    confidence: str  # "high" | "medium" | "low"


class CategoryImpactEntry(TypedDict, total=False):
    """One row of ``category_impact`` per category."""

    before: float
    after: float
    delta: float
    weighted_sgp: float


class TradeResult(TypedDict, total=False):
    """Return shape of ``evaluate_trade`` — Phase 1-5 combined output.

    All keys are technically optional because Phase 2 (MC) and Phase 5
    (game theory) only contribute when their feature flags are enabled,
    and Phase 4 (context) keys only appear when ``concentration_data`` is
    populated. ``total=False`` lets callers consume safely with ``.get``.

    Phase 1 keys (always present when ``evaluate_trade`` returns success):
      grade, grade_range, surplus_sgp, category_impact, category_analysis,
      punt_categories, bench_cost, replacement_penalty, flexibility_penalty,
      risk_flags, verdict, confidence_pct, before_totals, after_totals,
      giving_players, receiving_players, lineup_constrained,
      drop_candidate, fa_pickup, reshuffle, total_sgp_change, mc_mean,
      mc_std.

    Phase 2 keys (when ``enable_mc=True``):
      mc_median, p5, p25, p75, p95, prob_positive, var5, cvar5, sharpe,
      confidence_interval.

    Phase 4 keys (when ``enable_context=True`` and concentration runs):
      concentration_hhi_before, concentration_hhi_after,
      concentration_delta, concentration_penalty, bench_option_detail.

    Phase 5 keys (when ``enable_game_theory=True``):
      market_values, sensitivity_report.

    Transparency:
      analytics_context — :class:`AnalyticsContext` instance.
    """

    # --- Phase 1: always present ---
    grade: str
    grade_range: GradeRange
    surplus_sgp: float
    category_impact: dict[str, CategoryImpactEntry]
    category_analysis: dict[str, Any]
    punt_categories: list[str]
    bench_cost: float
    replacement_penalty: float
    replacement_detail: dict[str, Any]
    flexibility_penalty: float
    flexibility_detail: dict[str, Any]
    risk_flags: list[str]
    verdict: str  # "ACCEPT" | "DECLINE"
    compliant: bool
    confidence_pct: float
    before_totals: dict[str, float]
    after_totals: dict[str, float]
    giving_players: list[str]
    receiving_players: list[str]
    lineup_constrained: bool
    drop_candidate: str | None
    fa_pickup: str | None
    reshuffle: dict[str, Any] | None
    total_sgp_change: float
    mc_mean: float
    mc_std: float

    # --- Phase 2: MC overlay ---
    mc_median: float
    p5: float
    p25: float
    p75: float
    p95: float
    prob_positive: float
    var5: float
    cvar5: float
    sharpe: float
    confidence_interval: tuple[float, float]
    convergence_quality: str  # "good" | "marginal" | "poor"

    # --- Phase 4: context ---
    concentration_hhi_before: float
    concentration_hhi_after: float
    concentration_delta: float
    concentration_penalty: float
    bench_option_detail: dict[str, Any] | None

    # --- Phase 5: game theory ---
    market_values: dict[int, dict[str, Any]]
    sensitivity_report: dict[str, Any]

    # --- Transparency ---
    analytics_context: Any  # AnalyticsContext (avoid import cycle)


class MCOverlayResult(TypedDict, total=False):
    """Return shape of ``_run_mc_overlay`` — Phase 2 MC metrics overlay."""

    mc_mean: float
    mc_std: float
    mc_median: float
    p5: float
    p25: float
    p75: float
    p95: float
    prob_positive: float
    var5: float
    cvar5: float
    sharpe: float
    confidence_interval: tuple[float, float]
    convergence_quality: str
    grade: str
    verdict: str
    confidence_pct: float
