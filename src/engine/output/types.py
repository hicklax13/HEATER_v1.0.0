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
    # Feature 1 (2026-05-23): IP-floor soft penalty per report Section B.6.
    # Trade-marginal penalty when post-trade weekly IP falls below the
    # 20 IP/week Yahoo floor; delta semantics so a trade that doesn't
    # worsen an already-below-floor situation contributes 0.
    ip_floor_penalty: float
    ip_floor_detail: dict[str, Any]
    # Specialist cap (report Section H.2): excess SGP credit removed when a
    # single received starter exceeds 25% of a category's standings range.
    specialist_cap_penalty: float
    specialist_cap_detail: dict[str, Any]
    # Secondary valuation diagnostics (report B.8 + B.5/C.2 + B.7 pseudocode).
    # These do NOT feed the grade — they are league-wide / variance-aware
    # sanity checks alongside the primary roster-context SGP surplus.
    delta_vorp_prp: float  # Σ VORP(receiving) − Σ VORP(giving) — report B.8
    vorp_detail: dict[str, Any]
    delta_g_score: float  # Σ g_composite(receiving) − Σ g_composite(giving)
    gscore_detail: dict[str, Any]
    drl_replacement_chain: list[dict[str, Any]]  # named bench/FA fills — report B.7
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
    # Bug A (2026-05-23): Phase 1 weighted SGP is the AUTHORITY for grade /
    # verdict / confidence_pct. The MC's own grade/verdict/confidence_pct
    # are exposed as mc_grade/mc_verdict/mc_confidence_pct so the user can
    # see when the two graders disagree, but they NEVER override the
    # top-level grade. (See _run_mc_overlay rename block.)
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
    mc_grade: str  # MC-derived grade — diagnostic only, never the top-level grade
    mc_verdict: str  # "ACCEPT" | "DECLINE" from MC mean — diagnostic only
    mc_confidence_pct: float  # MC's prob_positive expressed as %

    # --- Phase 4: context ---
    concentration_hhi_before: float
    concentration_hhi_after: float
    concentration_delta: float
    concentration_penalty: float
    bench_option_detail: dict[str, Any] | None

    # --- Phase 5: game theory ---
    market_values: dict[int, dict[str, Any]]
    sensitivity_report: dict[str, Any]

    # --- Phase 6: Feature 2 (2026-05-23) Weekly H2H matrix ---
    # Per report Section B.5 — 26-week × 12-cat win-probability matrix.
    # Present only when enable_weekly_matrix=True AND weekly_schedule +
    # league_rosters are provided. Dict with keys: before (pd.DataFrame),
    # after (pd.DataFrame), delta (pd.DataFrame), schedule (dict),
    # summary (pd.DataFrame), method (str), cv_used (dict).
    weekly_matrix: dict[str, Any]

    # --- Phase 7: Feature 3 (2026-05-23) Playoff + championship sim ---
    # Per report Section B.10 + Q(a) — the engine's PRIMARY objective.
    # Present only when enable_playoff_sim=True AND weekly_schedule +
    # league_rosters + current_wins + user_team_name are provided.
    # playoff_sim dict contains before/after/delta playoff_prob + champ_prob.
    # delta_playoff_prob and delta_champ_prob are also surfaced at top level
    # for headline display.
    playoff_sim: dict[str, Any]
    delta_playoff_prob: float
    delta_champ_prob: float

    # --- Transparency ---
    analytics_context: Any  # AnalyticsContext (avoid import cycle)


class MCOverlayResult(TypedDict, total=False):
    """Return shape of ``_run_mc_overlay`` — Phase 2 MC metrics overlay.

    Bug A (2026-05-23): The MC's grade/verdict/confidence_pct are renamed to
    mc_grade/mc_verdict/mc_confidence_pct at the boundary so the caller's
    ``result.update(mc_result)`` cannot clobber Phase 1's authoritative grade.
    """

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
    mc_grade: str
    mc_verdict: str
    mc_confidence_pct: float
