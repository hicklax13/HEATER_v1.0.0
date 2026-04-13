"""Master pipeline orchestrator for the enhanced lineup optimizer.

Chains: projections → matchup adjustments → category weights → LP solve.

Four modes:
  - Quick (<1s): Enhanced projections + mean-variance only. No scenarios.
  - Standard (2-3s): 200 scenarios + CVaR + matchup adjustments.
  - Full (5-10s): Stochastic MIP + maximin comparison + streaming + multi-period.
  - Daily (<2s): Enhanced projections + matchup + H2H + Daily Category Value.

Each module is optional and degrades gracefully when unavailable or when
the data it needs is missing.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.analytics_context import AnalyticsContext, DataQuality, ModuleStatus
from src.validation.constant_optimizer import load_constants

_CONSTANTS = load_constants()

logger = logging.getLogger(__name__)

# ── Module imports (all optional) ─────────────────────────────────────

try:
    from src.optimizer.projections import build_enhanced_projections

    _PROJECTIONS_AVAILABLE = True
except ImportError:
    _PROJECTIONS_AVAILABLE = False

try:
    from src.optimizer.matchup_adjustments import compute_weekly_matchup_adjustments

    _MATCHUP_AVAILABLE = True
except ImportError:
    _MATCHUP_AVAILABLE = False

try:
    from src.optimizer.h2h_engine import (
        compute_h2h_category_weights,
        estimate_h2h_win_probability,
    )

    _H2H_AVAILABLE = True
except ImportError:
    _H2H_AVAILABLE = False

try:
    from src.optimizer.sgp_theory import compute_nonlinear_weights

    _SGP_AVAILABLE = True
except ImportError:
    _SGP_AVAILABLE = False

try:
    from src.optimizer.streaming import rank_streaming_candidates

    _STREAMING_AVAILABLE = True
except ImportError:
    _STREAMING_AVAILABLE = False

try:
    from src.optimizer.scenario_generator import (
        compute_scenario_lineup_values,
        generate_stat_scenarios,
        mean_variance_adjustments,
    )

    _SCENARIOS_AVAILABLE = True
except ImportError:
    _SCENARIOS_AVAILABLE = False

try:
    from src.optimizer.dual_objective import blend_h2h_roto_weights, recommend_alpha

    _DUAL_AVAILABLE = True
except ImportError:
    _DUAL_AVAILABLE = False

try:
    from src.optimizer.advanced_lp import maximin_lineup

    _ADVANCED_LP_AVAILABLE = True
except ImportError:
    _ADVANCED_LP_AVAILABLE = False

try:
    from src.lineup_optimizer import LineupOptimizer

    _OPTIMIZER_AVAILABLE = True
except ImportError:
    _OPTIMIZER_AVAILABLE = False


# ── Constants ─────────────────────────────────────────────────────────

ALL_CATEGORIES: list[str] = [
    "r",
    "hr",
    "rbi",
    "sb",
    "avg",
    "obp",
    "w",
    "l",
    "sv",
    "k",
    "era",
    "whip",
]
INVERSE_CATS: set[str] = {"l", "era", "whip"}

# Default SGP denominators for weight normalization
_DEFAULT_SCALE: dict[str, float] = {
    "r": 20.0,
    "hr": 7.0,
    "rbi": 20.0,
    "sb": 5.0,
    "avg": 0.005,
    "obp": 0.005,
    "w": 3.0,
    "l": 3.0,
    "sv": 5.0,
    "k": 25.0,
    "era": 0.30,
    "whip": 0.03,
}

# Mode presets
MODE_PRESETS: dict[str, dict[str, Any]] = {
    "quick": {
        "enable_projections": True,
        "enable_matchups": False,
        "enable_scenarios": False,
        "enable_streaming": False,
        "enable_multi_period": False,
        "enable_maximin": False,
        "n_scenarios": 0,
        "risk_aversion": 0.10,
    },
    "standard": {
        "enable_projections": True,
        "enable_matchups": True,
        "enable_scenarios": True,
        "enable_streaming": False,
        "enable_multi_period": False,
        "enable_maximin": False,
        "n_scenarios": 200,
        "risk_aversion": 0.15,
    },
    "full": {
        "enable_projections": True,
        "enable_matchups": True,
        "enable_scenarios": True,
        "enable_streaming": True,
        "enable_multi_period": True,
        "enable_maximin": True,
        "n_scenarios": 500,
        "risk_aversion": 0.15,
    },
    "daily": {
        "enable_projections": True,
        "enable_matchups": True,
        "enable_h2h": True,
        "enable_sgp": False,
        "enable_streaming": False,
        "enable_scenarios": False,
        "enable_dual_objective": False,
        "enable_advanced_lp": False,
        "enable_daily_dcv": True,
        "enable_multi_period": False,
        "enable_maximin": False,
        "n_scenarios": 0,
        "risk_aversion": 0.10,
    },
}


# ── Pipeline Class ─────────────────────────────────────────────────────


class LineupOptimizerPipeline:
    """Chains all optimizer modules into a single optimization pipeline.

    Usage::

        pipeline = LineupOptimizerPipeline(roster, mode="standard")
        result = pipeline.optimize(standings=standings_df)
    """

    def __init__(
        self,
        roster: pd.DataFrame,
        mode: str = "standard",
        alpha: float = 0.5,
        weeks_remaining: int | None = None,
        config: Any | None = None,
    ):
        """Initialize the pipeline.

        Args:
            roster: Player DataFrame with stat projections and positions.
            mode: Optimization mode — "quick", "standard", "full", or "daily".
            alpha: H2H vs season-long blend (0=pure season-long, 1=pure H2H).
            weeks_remaining: Weeks left in the season. If None, computed
                dynamically from today's date and the 2026 season start.
            config: Optional LeagueConfig for SGP denominators.
        """
        if weeks_remaining is None:
            _ET = timezone(timedelta(hours=-4))
            _season_start = datetime(2026, 3, 25, tzinfo=_ET)
            _now = datetime.now(_ET)
            _weeks_elapsed = max(0, (_now - _season_start).days // 7)
            weeks_remaining = max(1, 24 - _weeks_elapsed)
        self.roster = roster.copy()
        self.mode = mode if mode in MODE_PRESETS else "standard"
        self.alpha = alpha
        self.weeks_remaining = weeks_remaining
        self.config = config
        self._preset = MODE_PRESETS[self.mode]
        self._ctx: AnalyticsContext | None = None

    def optimize(
        self,
        standings: pd.DataFrame | None = None,
        team_name: str | None = None,
        h2h_opponent_totals: dict[str, float] | None = None,
        my_totals: dict[str, float] | None = None,
        week_schedule: list[dict] | None = None,
        park_factors: dict[str, float] | None = None,
        free_agents: pd.DataFrame | None = None,
        ytd_totals: dict[str, float] | None = None,
        target_totals: dict[str, float] | None = None,
        weekly_rates: dict[str, float] | None = None,
        roto_rank: int | None = None,
        h2h_wins: int | None = None,
        h2h_losses: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the full optimization pipeline.

        All parameters are optional — the pipeline degrades gracefully
        when data is missing, skipping modules that require it.

        Keyword Args (via **kwargs, used by "daily" mode):
            matchup: dict or None — live matchup data from Yahoo API.
            schedule_today: list or None — today's MLB schedule.
            weekly_ip_projected: float — projected weekly IP (default 20.0).

        Returns:
            Dict with keys:
              - lineup: dict from LineupOptimizer.optimize_lineup()
              - category_weights: dict[str, float] used for optimization
              - h2h_analysis: dict or None (win probabilities)
              - streaming_suggestions: list or None
              - risk_metrics: dict or None (CVaR, VaR)
              - maximin_comparison: dict or None
              - recommendations: list[str] — human-readable tips
              - timing: dict[str, float] — per-stage timing
              - mode: str — optimization mode used
              - urgency_weights: dict (daily mode only)
              - rate_stat_modes: dict (daily mode only)
              - matchup_summary: dict (daily mode only)
              - daily_dcv: DataFrame (daily mode only)
              - daily_lineup: dict (daily mode only)
        """
        t0 = time.perf_counter()
        timing: dict[str, float] = {}
        recommendations: list[str] = []
        enhanced_roster = self.roster.copy()

        # --- AnalyticsContext: transparency spine for this optimization ---
        ctx = AnalyticsContext(pipeline="lineup_optimizer")
        ctx.stamp_data(
            "roster",
            DataQuality.LIVE if len(self.roster) > 0 else DataQuality.MISSING,
            record_count=len(self.roster),
        )
        self._ctx = ctx

        # ── Stage 1: Enhanced Projections ─────────────────────────
        if self._preset["enable_projections"] and _PROJECTIONS_AVAILABLE:
            t1 = time.perf_counter()
            try:
                enhanced_roster = build_enhanced_projections(
                    enhanced_roster,
                    config=self.config,
                    enable_bayesian=True,
                    enable_kalman=True,
                    enable_statcast=(self.mode in ("full", "standard")),
                    enable_injury=True,
                    weeks_remaining=self.weeks_remaining,
                )
                logger.info("Stage 1: Enhanced projections applied")
            except Exception:
                logger.warning("Stage 1: Enhanced projections failed", exc_info=True)
            timing["projections"] = time.perf_counter() - t1

        # ── Stage 2: Matchup Adjustments ─────────────────────────
        matchup_adjusted = False
        if self._preset["enable_matchups"] and _MATCHUP_AVAILABLE and week_schedule and park_factors is not None:
            t2 = time.perf_counter()
            try:
                enhanced_roster = compute_weekly_matchup_adjustments(
                    enhanced_roster,
                    week_schedule=week_schedule,
                    park_factors=park_factors or {},
                )
                matchup_adjusted = True
                logger.info("Stage 2: Matchup adjustments applied")
            except Exception:
                logger.warning("Stage 2: Matchup adjustments failed", exc_info=True)
            timing["matchups"] = time.perf_counter() - t2

        # ── Stage 3: Category Weights ─────────────────────────────
        t3 = time.perf_counter()
        category_weights = self._compute_category_weights(
            standings=standings,
            team_name=team_name,
            h2h_opponent_totals=h2h_opponent_totals,
            my_totals=my_totals,
            ytd_totals=ytd_totals,
            target_totals=target_totals,
            weekly_rates=weekly_rates,
            roto_rank=roto_rank,
            h2h_wins=h2h_wins,
            h2h_losses=h2h_losses,
        )
        timing["weights"] = time.perf_counter() - t3

        # ── Stage 4: Mean-Variance Risk Adjustment ────────────────
        risk_adjustments: dict[int, float] | None = None
        if self._preset["risk_aversion"] > 0 and _SCENARIOS_AVAILABLE:
            t4 = time.perf_counter()
            try:
                risk_adjustments = mean_variance_adjustments(
                    enhanced_roster,
                    lambda_risk=self._preset["risk_aversion"],
                )
            except Exception:
                logger.warning("Stage 4: Mean-variance failed", exc_info=True)
            timing["risk"] = time.perf_counter() - t4

        # ── Stage 5: LP Solve ─────────────────────────────────────
        t5 = time.perf_counter()
        lineup_result = self._solve_lineup(
            enhanced_roster,
            category_weights,
            risk_adjustments,
        )
        timing["solve"] = time.perf_counter() - t5

        # ── Stage 6: H2H Analysis ─────────────────────────────────
        h2h_analysis = None
        if _H2H_AVAILABLE and h2h_opponent_totals and my_totals:
            try:
                h2h_analysis = estimate_h2h_win_probability(
                    my_totals,
                    h2h_opponent_totals,
                )
                n_cats = len(ALL_CATEGORIES)
                exp_wins = h2h_analysis.get("expected_wins", n_cats / 2.0)
                if exp_wins < n_cats * 0.4:
                    recommendations.append(
                        f"Projected to win only {exp_wins:.2f}/{n_cats} categories — "
                        "consider streaming pitchers or targeting close categories."
                    )
                elif exp_wins > n_cats * 0.7:
                    recommendations.append(
                        f"Strong matchup ({exp_wins:.2f}/{n_cats} projected wins) — "
                        "consider resting borderline starters to preserve roster flexibility."
                    )
            except Exception:
                logger.warning("Stage 6: H2H analysis failed", exc_info=True)

        # ── Stage 7: Streaming Suggestions ─────────────────────────
        streaming_suggestions = None
        if (
            self._preset["enable_streaming"]
            and _STREAMING_AVAILABLE
            and free_agents is not None
            and not free_agents.empty
        ):
            t7 = time.perf_counter()
            try:
                # Filter to pitchers only
                fa_pitchers = (
                    free_agents[free_agents.get("positions", pd.Series(dtype=str)).str.contains("SP|RP", na=False)]
                    if "positions" in free_agents.columns
                    else pd.DataFrame()
                )

                if not fa_pitchers.empty:
                    fa_list = fa_pitchers.to_dict("records")
                    streaming_suggestions = rank_streaming_candidates(
                        fa_list,
                        park_factors=park_factors,
                        category_weights=category_weights,
                        max_results=5,
                    )
                    if streaming_suggestions:
                        top = streaming_suggestions[0]
                        recommendations.append(
                            f"Top streaming pickup: {top.get('player_name', '?')} (+{top.get('net_value', 0):.2f} SGP)"
                        )
            except Exception:
                logger.warning("Stage 7: Streaming failed", exc_info=True)
            timing["streaming"] = time.perf_counter() - t7

        # ── Stage 8: Scenario Analysis (CVaR) ──────────────────────
        risk_metrics = None
        if self._preset["enable_scenarios"] and _SCENARIOS_AVAILABLE and self._preset["n_scenarios"] > 0:
            t8 = time.perf_counter()
            try:
                scenarios = generate_stat_scenarios(
                    enhanced_roster,
                    n_scenarios=self._preset["n_scenarios"],
                )
                # Compute per-scenario lineup values
                assignments_map = _extract_assignments_map(
                    lineup_result,
                    enhanced_roster,
                )
                if assignments_map:
                    values = compute_scenario_lineup_values(
                        scenarios,
                        assignments_map,
                        category_weights,
                    )
                    risk_metrics = _compute_risk_metrics(values)
            except Exception:
                logger.warning("Stage 8: Scenario analysis failed", exc_info=True)
            timing["scenarios"] = time.perf_counter() - t8

        # ── Stage 9: Maximin Comparison ────────────────────────────
        maximin_comparison = None
        if self._preset["enable_maximin"] and _ADVANCED_LP_AVAILABLE:
            t9 = time.perf_counter()
            try:
                maximin_result = maximin_lineup(
                    enhanced_roster,
                    scale_factors=_DEFAULT_SCALE,
                    category_weights=category_weights,
                )
                if maximin_result.get("status") == "Optimal":
                    maximin_comparison = maximin_result
                    recommendations.append(
                        f"Maximin (balanced) lineup guarantees z={maximin_result.get('z_value', 0):.2f} "
                        "in worst category."
                    )
            except Exception:
                logger.warning("Stage 9: Maximin failed", exc_info=True)
            timing["maximin"] = time.perf_counter() - t9

        # ── V2 Daily Optimizer Stages (10-12) ─────────────────────
        # Mutable dict to collect V2 stage outputs; merged into final result.
        daily_extras: dict[str, Any] = {}

        # ── Stage 10: Category Urgency (V2) ────────────────────────
        if self._preset.get("enable_daily_dcv", False):
            t10 = time.perf_counter()
            try:
                from src.optimizer.category_urgency import compute_urgency_weights

                matchup_data = kwargs.get("matchup")
                urgency_result = compute_urgency_weights(matchup_data, self.config)
                daily_extras["urgency_weights"] = urgency_result.get("urgency", {})
                daily_extras["rate_stat_modes"] = urgency_result.get("rate_modes", {})
                daily_extras["matchup_summary"] = urgency_result.get("summary", {})
                logger.info(
                    "Stage 10 (Category Urgency): %d categories weighted",
                    len(daily_extras["urgency_weights"]),
                )
            except Exception:
                logger.warning("Stage 10 (Category Urgency) failed", exc_info=True)
            timing["category_urgency"] = time.perf_counter() - t10

        # ── Stage 11: Daily Category Value (V2) ────────────────────
        if self._preset.get("enable_daily_dcv", False):
            t11 = time.perf_counter()
            try:
                from src.optimizer.daily_optimizer import (
                    apply_ip_pace_scaling,
                    build_daily_dcv_table,
                    check_ip_override,
                )

                schedule_today = kwargs.get("schedule_today")
                dcv_table = build_daily_dcv_table(
                    roster=enhanced_roster,
                    matchup=kwargs.get("matchup"),
                    schedule_today=schedule_today,
                    park_factors=park_factors,
                    config=self.config,
                )

                # Check IP minimum override
                weekly_ip = kwargs.get("weekly_ip_projected", 20.0)
                dcv_table = check_ip_override(dcv_table, weekly_ip)

                # T3-6: Per-pitcher IP pace constraint awareness
                dcv_table = apply_ip_pace_scaling(dcv_table, weekly_ip)

                daily_extras["daily_dcv"] = dcv_table
                if not dcv_table.empty:
                    logger.info(
                        "Stage 11 (Daily DCV): %d players scored",
                        len(dcv_table),
                    )
            except Exception:
                logger.warning("Stage 11 (Daily DCV) failed", exc_info=True)
            timing["daily_dcv"] = time.perf_counter() - t11

        # ── Stage 12: Daily Lineup (V2) ──────────────────────────
        if self._preset.get("enable_daily_dcv", False) and "daily_dcv" in daily_extras:
            t12 = time.perf_counter()
            try:
                dcv = daily_extras["daily_dcv"]
                if not dcv.empty:
                    # Build daily recommended lineup from DCV table
                    # Top players by total_dcv, respecting position slots
                    starters = dcv[dcv["volume_factor"] > 0].head(18)  # Max 18 starters
                    name_col = "name" if "name" in dcv.columns else "player_name"
                    pos_col = "positions" if "positions" in dcv.columns else "pos"
                    starter_cols = [name_col, pos_col, "total_dcv"]
                    if "stud_floor_applied" in dcv.columns:
                        starter_cols.append("stud_floor_applied")
                    bench_cols = [name_col, pos_col, "total_dcv"]

                    daily_extras["daily_lineup"] = {
                        "starters": starters[[c for c in starter_cols if c in starters.columns]].to_dict("records")
                        if not starters.empty
                        else [],
                        "bench": dcv[~dcv.index.isin(starters.index)][
                            [c for c in bench_cols if c in dcv.columns]
                        ].to_dict("records")
                        if not dcv.empty
                        else [],
                    }
                    logger.info(
                        "Stage 12 (Daily Lineup): %d starters recommended",
                        len(starters),
                    )
            except Exception:
                logger.warning("Stage 12 (Daily Lineup) failed", exc_info=True)
            timing["daily_lineup"] = time.perf_counter() - t12

        # ── Stamp AnalyticsContext modules from timing data ─────────
        _stage_map = {
            "projections": ("enable_projections", _PROJECTIONS_AVAILABLE),
            "matchups": ("enable_matchups", _MATCHUP_AVAILABLE),
            "weights": (None, True),  # Always runs
            "risk": (None, _SCENARIOS_AVAILABLE),
            "solve": (None, _OPTIMIZER_AVAILABLE),  # Always runs
            "streaming": ("enable_streaming", _STREAMING_AVAILABLE),
            "scenarios": ("enable_scenarios", _SCENARIOS_AVAILABLE),
            "maximin": ("enable_maximin", _ADVANCED_LP_AVAILABLE),
            "category_urgency": ("enable_daily_dcv", True),
            "daily_dcv": ("enable_daily_dcv", True),
            "daily_lineup": ("enable_daily_dcv", True),
        }
        for stage, (setting_key, module_avail) in _stage_map.items():
            if stage in timing:
                ctx.stamp_module(stage, ModuleStatus.EXECUTED, influence=0.5)
            elif setting_key and not self._preset.get(setting_key, False):
                ctx.stamp_module(stage, ModuleStatus.SKIPPED, reason=f"Disabled in '{self.mode}' mode")
            elif not module_avail:
                ctx.stamp_module(stage, ModuleStatus.DISABLED, reason="Module unavailable (import failed)")
            else:
                ctx.stamp_module(stage, ModuleStatus.FALLBACK, reason="Failed silently")
        # H2H analysis doesn't have a preset toggle — depends on data
        if h2h_analysis is not None:
            ctx.stamp_module("h2h_analysis", ModuleStatus.EXECUTED, influence=0.4)
        elif not _H2H_AVAILABLE:
            ctx.stamp_module("h2h_analysis", ModuleStatus.DISABLED, reason="H2H module unavailable")
        else:
            ctx.stamp_module("h2h_analysis", ModuleStatus.SKIPPED, reason="No opponent/team totals provided")

        # ── Compile Result ─────────────────────────────────────────
        timing["total"] = time.perf_counter() - t0

        final_result: dict[str, Any] = {
            "lineup": lineup_result,
            "category_weights": category_weights,
            "h2h_analysis": h2h_analysis,
            "streaming_suggestions": streaming_suggestions,
            "risk_metrics": risk_metrics,
            "maximin_comparison": maximin_comparison,
            "recommendations": recommendations,
            "timing": timing,
            "mode": self.mode,
            "matchup_adjusted": matchup_adjusted,
            "analytics_context": ctx,
        }
        # Merge V2 daily optimizer outputs (if any)
        final_result.update(daily_extras)

        return final_result

    # ── Private Methods ────────────────────────────────────────────

    def _compute_category_weights(
        self,
        standings: pd.DataFrame | None = None,
        team_name: str | None = None,
        h2h_opponent_totals: dict[str, float] | None = None,
        my_totals: dict[str, float] | None = None,
        ytd_totals: dict[str, float] | None = None,
        target_totals: dict[str, float] | None = None,
        weekly_rates: dict[str, float] | None = None,
        roto_rank: int | None = None,
        h2h_wins: int | None = None,
        h2h_losses: int | None = None,
    ) -> dict[str, float]:
        """Build blended category weights from all available sources.

        Priority chain:
          1. Non-linear SGP from standings (roto component)
          2. H2H category weights from opponent totals (H2H component)
          3. Season balance urgency weights (modifier)
          4. Alpha-blend H2H and roto components
          5. Fallback: equal weights
        """
        roto_weights = {cat: 1.0 for cat in ALL_CATEGORIES}
        h2h_weights = {cat: 1.0 for cat in ALL_CATEGORIES}

        # Roto component: non-linear SGP from standings
        if _SGP_AVAILABLE and standings is not None and team_name:
            try:
                roto_weights = compute_nonlinear_weights(standings, team_name)
            except Exception:
                logger.warning("Non-linear SGP weights failed", exc_info=True)

        # H2H component: opponent-specific weights
        if _H2H_AVAILABLE and h2h_opponent_totals and my_totals:
            try:
                h2h_weights = compute_h2h_category_weights(
                    my_totals,
                    h2h_opponent_totals,
                )
            except Exception:
                logger.warning("H2H category weights failed", exc_info=True)

        # Use user-provided alpha; only auto-recommend when caller
        # left it at the default 0.5 AND record data is available.
        alpha = self.alpha
        if _DUAL_AVAILABLE and alpha == 0.5 and (roto_rank is not None or h2h_wins is not None):
            try:
                alpha = recommend_alpha(
                    self.weeks_remaining,
                    roto_rank=roto_rank,
                    h2h_record_wins=h2h_wins,
                    h2h_record_losses=h2h_losses,
                )
            except Exception:
                pass  # Keep user alpha

        # Blend H2H and roto
        if _DUAL_AVAILABLE:
            try:
                blended = blend_h2h_roto_weights(h2h_weights, roto_weights, alpha)
                return blended
            except Exception:
                logger.warning("Dual objective blending failed", exc_info=True)

        # Fallback: return roto weights directly
        return roto_weights

    def _solve_lineup(
        self,
        roster: pd.DataFrame,
        category_weights: dict[str, float],
        risk_adjustments: dict[int, float] | None = None,
    ) -> dict:
        """Solve the lineup LP with the given weights and risk adjustments."""
        if not _OPTIMIZER_AVAILABLE:
            return {
                "assignments": [],
                "bench": [],
                "projected_stats": {},
                "status": "optimizer_unavailable",
            }

        try:
            adjusted_roster = roster.copy()
            adjusted_roster = adjusted_roster.reset_index(drop=True)
            if risk_adjustments:
                # Store raw risk penalties for potential UI display
                adjusted_roster["_risk_penalty"] = 0.0
                for idx, penalty in risk_adjustments.items():
                    if idx < len(adjusted_roster):
                        adjusted_roster.iloc[idx, adjusted_roster.columns.get_loc("_risk_penalty")] = penalty

                # Pre-adjust stat values so the LP solver accounts for risk.
                # penalty is negative (from mean_variance_adjustments), so
                # risk_fraction = -penalty / max_penalty gives [0, 1] range.
                # We scale stats by (1 - risk_fraction * scale_factor).
                penalties = [risk_adjustments.get(i, 0.0) for i in range(len(adjusted_roster))]
                if penalties:
                    min_penalty = min(penalties)  # most negative = highest risk
                    if min_penalty < 0:
                        counting_cats = ["r", "hr", "rbi", "sb", "w", "sv", "k"]
                        for idx, penalty in risk_adjustments.items():
                            if idx < len(adjusted_roster) and penalty < 0:
                                # Convert negative penalty to a 0-1 risk fraction
                                # Scale factor of 0.15 caps the maximum stat reduction
                                try:
                                    _risk_cap = _CONSTANTS.get("risk_scale_cap")
                                except (TypeError, KeyError):
                                    _risk_cap = 0.15
                                risk_frac = min(abs(penalty / min_penalty), 1.0) * _risk_cap
                                for cat in counting_cats:
                                    if cat in adjusted_roster.columns:
                                        col_loc = adjusted_roster.columns.get_loc(cat)
                                        val = float(adjusted_roster.iloc[idx, col_loc])
                                        adjusted_roster.iloc[idx, col_loc] = val * (1.0 - risk_frac)

            optimizer = LineupOptimizer(adjusted_roster, self.config)
            result = optimizer.optimize_lineup(category_weights=category_weights)
            return result
        except Exception:
            logger.warning("LP solve failed", exc_info=True)
            return {
                "assignments": [],
                "bench": [],
                "projected_stats": {},
                "status": "solve_failed",
            }


# ── Helper Functions ───────────────────────────────────────────────────


def _extract_assignments_map(
    lineup_result: dict,
    roster: pd.DataFrame,
) -> dict[int, float]:
    """Convert lineup assignments to {player_index: weight} for scenario analysis."""
    assignments_map: dict[int, float] = {}
    assigned_names = {a["player_name"] for a in lineup_result.get("assignments", [])}

    name_col = "player_name" if "player_name" in roster.columns else "name"
    for idx, row in roster.iterrows():
        name = row.get(name_col, "")
        if name in assigned_names:
            # Use integer position index for scenario array indexing
            pos = roster.index.get_loc(idx)
            assignments_map[pos] = 1.0

    return assignments_map


def _compute_risk_metrics(values: np.ndarray) -> dict[str, float]:
    """Compute VaR and CVaR from scenario lineup values."""
    if len(values) == 0:
        return {"var_5": 0.0, "cvar_5": 0.0, "mean": 0.0, "std": 0.0}

    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    # 5th percentile VaR
    var_idx = max(0, int(n * 0.05) - 1)
    var_5 = float(sorted_vals[var_idx])

    # CVaR = mean of worst 5%
    tail_count = max(1, int(n * 0.05))
    cvar_5 = float(np.mean(sorted_vals[:tail_count]))

    return {
        "var_5": round(var_5, 4),
        "cvar_5": round(cvar_5, 4),
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "p10": round(float(np.percentile(values, 10)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p90": round(float(np.percentile(values, 90)), 4),
    }
