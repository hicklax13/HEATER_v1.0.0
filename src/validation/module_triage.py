"""
Module Triage — the kill/fix/keep/connect decision for every analytical module.

This is the output of the audit. Each module gets a verdict and an action plan.
Run `print_triage_report()` to see the full breakdown.

Verdicts:
    KILL     — Remove from codebase. Dead code, pseudoscience, or never called.
    FIX      — Broken but high-value. Wire in missing inputs, fix logic.
    KEEP     — Works, just needs constant calibration.
    CONNECT  — Computed but disconnected from decisions. Wire into pipeline.
    DEMOTE   — Keep code, remove from "pipeline count", label experimental.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Verdict(Enum):
    KILL = "KILL"
    FIX = "FIX"
    KEEP = "KEEP"
    CONNECT = "CONNECT"
    DEMOTE = "DEMOTE"


@dataclass
class ModuleTriage:
    name: str
    verdict: Verdict
    file: str
    reason: str
    action: str
    effort: str  # "trivial", "small", "medium", "large"
    priority: int  # 1=highest


TRIAGE: list[ModuleTriage] = [
    # =========================================================================
    # KILLED — Removed in Phase 1 kill sweep
    # =========================================================================
    ModuleTriage(
        name="Regime Detection (was projections.py)",
        verdict=Verdict.KILL,
        file="DELETED",
        reason="Was dead code — _apply_regime_adjustment() logged 'skipping' and "
        "returned input unchanged. Removed in Phase 1 kill sweep.",
        action="DONE",
        effort="trivial",
        priority=1,
    ),
    ModuleTriage(
        name="Bellman DP (was game_theory/dynamic_programming.py)",
        verdict=Verdict.KILL,
        file="DELETED",
        reason="Was fully implemented but never called from evaluate_trade(). Removed in Phase 1 kill sweep.",
        action="DONE",
        effort="trivial",
        priority=2,
    ),
    ModuleTriage(
        name="Adverse Selection (was game_theory/adverse_selection.py)",
        verdict=Verdict.KILL,
        file="DELETED",
        reason="Was pseudoscience — ~5.6% constant discount with made-up priors. Removed in Phase 1 kill sweep.",
        action="DONE",
        effort="trivial",
        priority=3,
    ),
    ModuleTriage(
        name="Multi-Period Optimization (was optimizer/multi_period.py)",
        verdict=Verdict.KILL,
        file="DELETED",
        reason="Was never called from UI — required inputs never passed. Removed in Phase 1 kill sweep.",
        action="DONE",
        effort="trivial",
        priority=4,
    ),
    # =========================================================================
    # FIX — Broken but high-value
    # =========================================================================
    ModuleTriage(
        name="Matchup Adjustments (optimizer/matchup_adjustments.py)",
        verdict=Verdict.FIX,
        file="src/optimizer/matchup_adjustments.py",
        reason="Lineup Optimizer page already wires week_schedule from "
        "get_weekly_schedule() and park_factors from PARK_FACTORS dict. "
        "Module runs when schedule data is available. Partially fixed in Phase 4.",
        action="DONE (wiring) — week_schedule and park_factors passed from page.\n"
        "Remaining: Replace hardcoded platoon splits (0.086/0.061) with calibrated values.\n"
        "Replace hardcoded temp coefficient (0.009) with calibrated value.",
        effort="small",
        priority=5,
    ),
    ModuleTriage(
        name="Start/Sit Advisor (pages/11, src/start_sit.py)",
        verdict=Verdict.FIX,
        file="pages/11_Start_Sit.py:195-201",
        reason="Was all context parameters passed as None. Now passes park_factors, "
        "standings, and weekly totals (user from standings, opponent approximated "
        "by league median). Matchup state classified dynamically. Fixed in Phase 4.",
        action="DONE — park_factors wired from PARK_FACTORS dict, standings from "
        "load_league_standings(), weekly totals built from standings pivot. "
        "Remaining: weekly_schedule requires schedule_grid.py integration, "
        "true opponent requires Yahoo weekly matchup data.",
        effort="medium",
        priority=6,
    ),
    ModuleTriage(
        name="Waiver Wire weeks_remaining (pages/10, src/waiver_wire.py)",
        verdict=Verdict.FIX,
        file="pages/10_Waiver_Wire.py:209",
        reason="Was hardcoded weeks_remaining=16. Now uses "
        "compute_weeks_remaining() from dynamic_context.py. Fixed in Phase 3.",
        action="DONE — pages/3 and pages/10 both use compute_weeks_remaining().",
        effort="small",
        priority=7,
    ),
    ModuleTriage(
        name="Power Rankings components (pages/8, src/power_rankings.py)",
        verdict=Verdict.FIX,
        file="pages/8_Standings.py:156-221",
        reason="Was 3 of 5 components hardcoded. Now passes None when data "
        "unavailable. compute_power_rating() re-weights from available components. "
        "UI shows 'N/A'. Fixed in Phase 3.",
        action="DONE — hardcoded 0.5/0.1/1.0 replaced with None, power_rating "
        "re-normalizes, UI shows 'N/A'. Dynamic computation functions exist in "
        "dynamic_context.py for when schedule/injury/matchup data is available.",
        effort="medium",
        priority=8,
    ),
    ModuleTriage(
        name="Injury Badges (src/injury_model.py)",
        verdict=Verdict.FIX,
        file="src/injury_model.py:67-136",
        reason="Only uses 3-year historical games_played. Does not check current "
        "IL status. A pitcher on IL right now shows 'Low Risk' green dot.",
        action="1. Add current_il_status check from transactions table.\n"
        "2. If player currently on IL, override to 'High Risk' regardless.\n"
        "3. Add last_transaction_date to health score inputs.",
        effort="small",
        priority=9,
    ),
    ModuleTriage(
        name="Trade Confidence Percentage",
        verdict=Verdict.FIX,
        file="src/engine/output/trade_evaluator.py:659",
        reason="confidence_pct = 50 + surplus * 25 is a linear transform with "
        "no statistical basis. Never calibrated against outcomes.",
        action="1. Replace with quality_score from AnalyticsContext (input-based).\n"
        "2. OR use MC probability (when enable_mc=True) as the confidence.\n"
        "3. Never show a single number — show confidence tier + explanation.",
        effort="small",
        priority=10,
    ),
    ModuleTriage(
        name="Bootstrap Silent Failures",
        verdict=Verdict.FIX,
        file="src/data_bootstrap.py:80-337",
        reason="API failures logged as warnings but splash screen shows green "
        "checkmarks. User has no idea data is missing/stale.",
        action="1. Return error status (not 'success') when 0 records fetched.\n"
        "2. Add AnalyticsContext.stamp_data() calls in bootstrap.\n"
        "3. Show data quality badges in UI (red/yellow/green).\n"
        "4. Add fallback to sample data on first-run API failure.",
        effort="medium",
        priority=11,
    ),
    ModuleTriage(
        name="Trade Engine Phase 1 vs Legacy Contradictions",
        verdict=Verdict.FIX,
        file="pages/3_Trade_Analyzer.py:207-237",
        reason="Phase 1 and legacy use different grading logic but UI doesn't "
        "indicate which engine ran. Same trade, different verdict.",
        action="1. Always show which engine produced the result.\n"
        "2. If falling back to legacy, show explicit warning.\n"
        "3. Standardize verdict logic across both engines.\n"
        "4. Long-term: remove legacy entirely once Phase 1 is stable.",
        effort="small",
        priority=12,
    ),
    # =========================================================================
    # KEEP — Works, needs calibration
    # =========================================================================
    ModuleTriage(
        name="Core LP Solver (src/lineup_optimizer.py)",
        verdict=Verdict.KEEP,
        file="src/lineup_optimizer.py",
        reason="The PuLP-based LP solver works correctly. It's the one module that does what it claims.",
        action="No changes needed. This is the foundation everything else feeds.",
        effort="trivial",
        priority=99,
    ),
    ModuleTriage(
        name="H2H Category Weights (optimizer/h2h_engine.py)",
        verdict=Verdict.KEEP,
        file="src/optimizer/h2h_engine.py",
        reason="Logic is sound. Hardcoded variances need calibration from "
        "actual weekly H2H data, but the framework is correct.",
        action="Calibrate weekly variances from CalibrationDataset.weekly_matchups. "
        "Load from ConstantSet instead of hardcoding.",
        effort="small",
        priority=13,
    ),
    ModuleTriage(
        name="Non-linear SGP (optimizer/sgp_theory.py)",
        verdict=Verdict.KEEP,
        file="src/optimizer/sgp_theory.py",
        reason="Logic is sound but sigma values contradict H2H engine (R: 25.0 vs 15.0). Need to reconcile.",
        action="1. Derive sigmas from actual standings data.\n"
        "2. Ensure consistency with H2H variances (same data source).\n"
        "3. Load from ConstantSet.",
        effort="small",
        priority=14,
    ),
    ModuleTriage(
        name="Phase 1 SGP Trade Evaluation",
        verdict=Verdict.KEEP,
        file="src/engine/output/trade_evaluator.py",
        reason="Core deterministic trade grading works. Needs calibrated thresholds and better confidence reporting.",
        action="Replace hardcoded GRADE_THRESHOLDS with ConstantSet values. Add AnalyticsContext for transparency.",
        effort="small",
        priority=15,
    ),
    ModuleTriage(
        name="Maximin LP (optimizer/advanced_lp.py)",
        verdict=Verdict.KEEP,
        file="src/optimizer/advanced_lp.py:165-169",
        reason="Works but excludes inverse categories (ERA, WHIP, L). "
        "This is a design choice, not a bug, but should be documented.",
        action="Add clear documentation that maximin is offense-only. "
        "Consider a separate inverse-aware maximin variant.",
        effort="trivial",
        priority=16,
    ),
    # =========================================================================
    # CONNECT — Computed but disconnected from decisions
    # =========================================================================
    ModuleTriage(
        name="CVaR → LP Solver",
        verdict=Verdict.CONNECT,
        file="src/optimizer/pipeline.py:378-401",
        reason="CVaR (Conditional Value at Risk) is computed from scenarios "
        "but never used as an LP constraint. It's displayed as a metric "
        "but doesn't influence the lineup.",
        action="Add CVaR constraint to advanced_lp.py: lineup value at 5th "
        "percentile must exceed a floor. This is the stochastic MIP "
        "that's already partially implemented but never called.",
        effort="large",
        priority=17,
    ),
    ModuleTriage(
        name="Streaming → Lineup Swaps",
        verdict=Verdict.CONNECT,
        file="src/optimizer/pipeline.py:345-376",
        reason="Streaming candidates are ranked independently. Not evaluated "
        "as actual lineup swaps (bench pitcher X for stream Y).",
        action="For each streaming candidate, compute net lineup SGP delta "
        "vs. the worst rostered pitcher. Present as 'swap X for Y: +0.3 SGP'.",
        effort="medium",
        priority=18,
    ),
    ModuleTriage(
        name="MC Trade → Deterministic Side-by-Side",
        verdict=Verdict.CONNECT,
        file="src/engine/output/trade_evaluator.py:718-741",
        reason="When enable_mc=True, MC result overwrites Phase 1 verdict. "
        "User never sees both. Should show side-by-side.",
        action="Show both verdicts: 'Deterministic: B+ | Monte Carlo: A- (55% positive)'. "
        "Let user see the disagreement and decide.",
        effort="small",
        priority=19,
    ),
    # =========================================================================
    # DEMOTE — Keep code, label experimental
    # =========================================================================
    ModuleTriage(
        name="Scenario Generation (optimizer/scenario_generator.py)",
        verdict=Verdict.DEMOTE,
        file="src/optimizer/scenario_generator.py",
        reason="Generates correlated stat scenarios using Gaussian copula. "
        "Correlations are hardcoded folklore (HR-RBI=0.85). Computed "
        "but CVaR doesn't feed LP. Potentially useful if wired in.",
        action="Label as 'Experimental' in UI. Don't count in pipeline module "
        "list. If CVaR constraint is connected (above), promote back.",
        effort="trivial",
        priority=20,
    ),
    ModuleTriage(
        name="Marcel Stabilization (optimizer/projections.py)",
        verdict=Verdict.DEMOTE,
        file="src/optimizer/projections.py:132-191",
        reason="Was mislabeled as 'Bayesian'. Renamed to 'Marcel Stabilization' in "
        "UI and docs. Internal code names (BayesianUpdater class) kept for "
        "backward compatibility. Fixed in Phase 6.",
        action="DONE — UI text renamed in pages/1, pages/2, optimizer/projections.py "
        "docstring. Internal class name BayesianUpdater left as-is to avoid "
        "breaking imports/tests.",
        effort="small",
        priority=21,
    ),
]


def print_triage_report() -> None:
    """Print the full triage report to stdout."""
    by_verdict: dict[Verdict, list[ModuleTriage]] = {}
    for t in TRIAGE:
        by_verdict.setdefault(t.verdict, []).append(t)

    print("=" * 70)
    print("HEATER MODULE TRIAGE REPORT")
    print("=" * 70)
    print()

    for verdict in Verdict:
        modules = by_verdict.get(verdict, [])
        if not modules:
            continue

        print(f"--- {verdict.value} ({len(modules)} modules) ---")
        print()

        for m in sorted(modules, key=lambda x: x.priority):
            print(f"  [{m.priority:2d}] {m.name}")
            print(f"       File: {m.file}")
            print(f"       Effort: {m.effort}")
            print(f"       Reason: {m.reason}")
            print(f"       Action: {m.action}")
            print()

    # Summary stats
    total = len(TRIAGE)
    counts = {v: len(by_verdict.get(v, [])) for v in Verdict}
    print("=" * 70)
    print("SUMMARY")
    print(f"  Total modules triaged: {total}")
    for v, c in counts.items():
        print(f"  {v.value:10s}: {c}")
    print()
    print("Priority order (do these first):")
    for m in sorted(TRIAGE, key=lambda x: x.priority)[:10]:
        print(f"  {m.priority:2d}. [{m.verdict.value:7s}] {m.name} ({m.effort})")


if __name__ == "__main__":
    print_triage_report()
