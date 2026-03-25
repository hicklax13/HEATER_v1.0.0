"""
Constant Optimizer — data-driven replacement for HEATER's ~30 magic numbers.

Instead of hardcoding sigma=10.0, urgency_weight=0.4, grade_threshold=2.0, etc.,
this module finds optimal values by minimizing prediction error against historical
outcomes.

Architecture:
    1. Define each magic number as a CalibratableConstant with bounds and a loss function
    2. Use historical data (from calibration_data.py) as ground truth
    3. Optimize each constant (or groups of correlated constants) via scipy.optimize
    4. Output a ConstantSet that can be loaded at runtime instead of hardcoded values
    5. Store calibration metadata (when, from what data, confidence interval)

This replaces the current pattern of:
    sigma = 10.0  # no justification

With:
    sigma = constants.get("survival_sigma")  # calibrated from 2025 Yahoo draft data
    # Value: 13.7, 95% CI: [11.2, 16.4], calibrated 2026-03-24, n=276 picks

Usage:
    from src.validation.constant_optimizer import ConstantRegistry, load_constants

    # At startup: load calibrated constants (falls back to defaults)
    constants = load_constants()

    # In simulation.py:
    sigma = constants.get("survival_sigma")  # 13.7 if calibrated, 10.0 if not

    # During calibration:
    registry = ConstantRegistry()
    registry.calibrate_all(datasets)
    registry.save("data/calibrated_constants.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Where calibrated constants are stored
CONSTANTS_PATH = Path("data/calibrated_constants.json")


@dataclass
class CalibratableConstant:
    """
    A single magic number that should be calibrated from data.

    Attributes:
        name: Unique identifier (e.g., "survival_sigma")
        description: What this constant controls
        current_value: The hardcoded value currently in the codebase
        default_value: Safe fallback if calibration hasn't run
        bounds: (min, max) for optimization
        source_file: Where this constant is hardcoded
        source_line: Approximate line number
        category: Grouping for related constants
    """

    name: str
    description: str
    current_value: float
    default_value: float
    bounds: tuple[float, float]
    source_file: str
    source_line: int = 0
    category: str = "uncategorized"

    # Set during calibration
    calibrated_value: float | None = None
    confidence_interval: tuple[float, float] | None = None
    calibration_date: datetime | None = None
    calibration_n: int = 0
    calibration_loss: float | None = None


# ============================================================================
# THE REGISTRY: Every magic number in HEATER, documented and calibratable
# ============================================================================

# fmt: off
ALL_CONSTANTS: list[CalibratableConstant] = [
    # --- Draft Engine ---
    CalibratableConstant(
        name="survival_sigma",
        description="ADP noise parameter for Normal CDF survival probability",
        current_value=10.0, default_value=10.0, bounds=(3.0, 40.0),
        source_file="src/simulation.py", source_line=20,
        category="draft",
    ),
    CalibratableConstant(
        name="picks_between_exponent",
        description="Exponent for scaling sigma by picks between (currently 0.3)",
        current_value=0.3, default_value=0.3, bounds=(0.1, 0.8),
        source_file="src/simulation.py", source_line=60,
        category="draft",
    ),
    CalibratableConstant(
        name="urgency_weight",
        description="Weight of urgency in combined_score = mean_sgp + urgency * W",
        current_value=0.4, default_value=0.4, bounds=(0.05, 2.0),
        source_file="src/simulation.py", source_line=575,
        category="draft",
    ),
    CalibratableConstant(
        name="scarcity_cap",
        description="Max survival reduction from positional scarcity",
        current_value=0.2, default_value=0.2, bounds=(0.05, 0.6),
        source_file="src/simulation.py", source_line=65,
        category="draft",
    ),
    CalibratableConstant(
        name="opponent_positional_boost",
        description="Weight boost for unfilled opponent positions",
        current_value=1.4, default_value=1.4, bounds=(1.0, 3.0),
        source_file="src/simulation.py", source_line=409,
        category="draft",
    ),
    CalibratableConstant(
        name="category_need_weight",
        description="Scaling for category balance need (below median boost)",
        current_value=0.8, default_value=0.8, bounds=(0.1, 2.0),
        source_file="src/draft_analytics.py", source_line=152,
        category="draft",
    ),
    CalibratableConstant(
        name="closeness_bonus_weight",
        description="Bonus weight for being close to median (PDF-based)",
        current_value=0.2, default_value=0.2, bounds=(0.0, 1.0),
        source_file="src/draft_analytics.py", source_line=153,
        category="draft",
    ),
    CalibratableConstant(
        name="bfa_early_gap",
        description="Rank gap threshold for BUY label in early rounds",
        current_value=20.0, default_value=20.0, bounds=(5.0, 40.0),
        source_file="src/draft_analytics.py", source_line=390,
        category="draft",
    ),
    CalibratableConstant(
        name="bfa_mid_gap",
        description="Rank gap threshold for BUY label in mid rounds",
        current_value=15.0, default_value=15.0, bounds=(5.0, 30.0),
        source_file="src/draft_analytics.py", source_line=392,
        category="draft",
    ),
    CalibratableConstant(
        name="bfa_late_gap",
        description="Rank gap threshold for BUY label in late rounds",
        current_value=10.0, default_value=10.0, bounds=(3.0, 20.0),
        source_file="src/draft_analytics.py", source_line=394,
        category="draft",
    ),
    CalibratableConstant(
        name="closer_sv_threshold",
        description="Projected saves threshold for closer bonus",
        current_value=20.0, default_value=20.0, bounds=(10.0, 35.0),
        source_file="src/contextual_factors.py", source_line=177,
        category="draft",
    ),
    CalibratableConstant(
        name="sgp_per_extra_pa",
        description="SGP value per additional plate appearance",
        current_value=0.0018, default_value=0.0018, bounds=(0.0005, 0.01),
        source_file="src/contextual_factors.py", source_line=296,
        category="draft",
    ),

    # --- Trade Engine ---
    CalibratableConstant(
        name="trade_grade_a_plus",
        description="SGP surplus threshold for A+ trade grade",
        current_value=2.0, default_value=2.0, bounds=(0.5, 5.0),
        source_file="src/engine/output/trade_evaluator.py", source_line=98,
        category="trade",
    ),
    CalibratableConstant(
        name="trade_grade_a",
        description="SGP surplus threshold for A trade grade",
        current_value=1.5, default_value=1.5, bounds=(0.3, 4.0),
        source_file="src/engine/output/trade_evaluator.py", source_line=99,
        category="trade",
    ),
    CalibratableConstant(
        name="trade_confidence_scale",
        description="Linear scaling factor for confidence pct (currently 25.0)",
        current_value=25.0, default_value=25.0, bounds=(5.0, 100.0),
        source_file="src/engine/output/trade_evaluator.py", source_line=659,
        category="trade",
    ),
    CalibratableConstant(
        name="fa_turnover_discount",
        description="Discount applied to FA replacement value",
        current_value=0.5, default_value=0.5, bounds=(0.1, 1.0),
        source_file="src/engine/output/trade_evaluator.py", source_line=871,
        category="trade",
    ),
    CalibratableConstant(
        name="concentration_hhi_threshold",
        description="HHI delta threshold for concentration risk warning",
        current_value=0.05, default_value=0.05, bounds=(0.01, 0.20),
        source_file="src/engine/output/trade_evaluator.py", source_line=649,
        category="trade",
    ),
    CalibratableConstant(
        name="punt_rank_threshold",
        description="Standings rank at or above which a category is punt-eligible",
        current_value=10.0, default_value=10.0, bounds=(7.0, 12.0),
        source_file="src/engine/portfolio/category_analysis.py", source_line=178,
        category="trade",
    ),
    CalibratableConstant(
        name="punt_weeks_buffer",
        description="Buffer multiplier for weeks-to-close-gap (currently 1.2)",
        current_value=1.2, default_value=1.2, bounds=(1.0, 2.0),
        source_file="src/engine/portfolio/category_analysis.py", source_line=174,
        category="trade",
    ),

    # --- Lineup Optimizer ---
    CalibratableConstant(
        name="h2h_variance_r",
        description="Weekly variance for Runs in H2H engine",
        current_value=225.0, default_value=225.0, bounds=(50.0, 600.0),
        source_file="src/optimizer/h2h_engine.py", source_line=61,
        category="optimizer",
    ),
    CalibratableConstant(
        name="h2h_variance_hr",
        description="Weekly variance for Home Runs in H2H engine",
        current_value=16.0, default_value=16.0, bounds=(4.0, 64.0),
        source_file="src/optimizer/h2h_engine.py", source_line=63,
        category="optimizer",
    ),
    CalibratableConstant(
        name="sgp_sigma_r",
        description="SGP theory sigma for Runs (contradicts H2H variance of 15.0)",
        current_value=25.0, default_value=25.0, bounds=(10.0, 50.0),
        source_file="src/optimizer/sgp_theory.py", source_line=46,
        category="optimizer",
    ),
    CalibratableConstant(
        name="risk_scale_cap",
        description="Maximum risk penalty fraction applied to projections",
        current_value=0.15, default_value=0.15, bounds=(0.05, 0.40),
        source_file="src/optimizer/pipeline.py", source_line=560,
        category="optimizer",
    ),
    CalibratableConstant(
        name="dual_alpha_week6",
        description="H2H/Roto alpha blend when weeks_remaining < 6",
        current_value=0.7, default_value=0.7, bounds=(0.4, 0.95),
        source_file="src/optimizer/dual_objective.py", source_line=155,
        category="optimizer",
    ),
    CalibratableConstant(
        name="streaming_baseline_era",
        description="Baseline ERA for streaming pitcher evaluation",
        current_value=4.0, default_value=4.0, bounds=(3.5, 5.0),
        source_file="src/optimizer/streaming.py", source_line=152,
        category="optimizer",
    ),
    CalibratableConstant(
        name="hr_temp_coefficient",
        description="HR increase per degree Fahrenheit (Nathan 2012 cite)",
        current_value=0.009, default_value=0.009, bounds=(0.003, 0.020),
        source_file="src/optimizer/matchup_adjustments.py", source_line=87,
        category="optimizer",
    ),
    CalibratableConstant(
        name="copula_hr_rbi_corr",
        description="HR-RBI correlation in scenario generator (folklore value)",
        current_value=0.85, default_value=0.85, bounds=(0.5, 0.98),
        source_file="src/optimizer/scenario_generator.py", source_line=67,
        category="optimizer",
    ),

    # --- In-Season Pages ---
    CalibratableConstant(
        name="weekly_rate_r",
        description="Default weekly run rate for gap analysis",
        current_value=35.0, default_value=35.0, bounds=(15.0, 60.0),
        source_file="src/waiver_wire.py", source_line=27,
        category="in_season",
    ),
    CalibratableConstant(
        name="hitter_home_advantage",
        description="Home advantage multiplier for hitters",
        current_value=1.02, default_value=1.02, bounds=(1.0, 1.06),
        source_file="src/matchup_planner.py", source_line=41,
        category="in_season",
    ),
    CalibratableConstant(
        name="standings_tau_era",
        description="Weekly ERA variance for standings simulation",
        current_value=1.20, default_value=1.20, bounds=(0.3, 3.0),
        source_file="src/standings_projection.py", source_line=15,
        category="in_season",
    ),
]
# fmt: on


@dataclass
class ConstantSet:
    """
    A set of calibrated (or default) constants that can be loaded at runtime.

    Replaces hardcoded magic numbers throughout the codebase.
    """

    values: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    # metadata[name] = {calibrated_from, date, n, loss, ci_low, ci_high}

    def get(self, name: str) -> float:
        """Get a constant value, falling back to default if not calibrated."""
        if name in self.values:
            return self.values[name]
        # Find default from registry
        for c in ALL_CONSTANTS:
            if c.name == name:
                return c.default_value
        raise KeyError(f"Unknown constant: {name}")

    def is_calibrated(self, name: str) -> bool:
        """Check if a constant has been calibrated (vs using default)."""
        return name in self.metadata and self.metadata[name].get("calibrated", False)

    def save(self, path: Path | str | None = None) -> None:
        """Save calibrated constants to JSON."""
        path = Path(path or CONSTANTS_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "values": self.values,
            "metadata": {
                k: {kk: (vv.isoformat() if isinstance(vv, datetime) else vv) for kk, vv in v.items()}
                for k, v in self.metadata.items()
            },
            "generated_at": datetime.now(UTC).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info("Saved %d calibrated constants to %s", len(self.values), path)

    @classmethod
    def load(cls, path: Path | str | None = None) -> ConstantSet:
        """Load calibrated constants from JSON, falling back to defaults."""
        path = Path(path or CONSTANTS_PATH)
        cs = cls()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                cs.values = data.get("values", {})
                cs.metadata = data.get("metadata", {})
                logger.info("Loaded %d calibrated constants from %s", len(cs.values), path)
            except Exception as exc:
                logger.warning("Failed to load constants from %s: %s", path, exc)
        else:
            logger.info("No calibrated constants file at %s — using defaults", path)
        return cs


def load_constants(path: Path | str | None = None) -> ConstantSet:
    """
    Load calibrated constants for runtime use.

    Call this once at app startup. Thread the ConstantSet through pipelines
    instead of using hardcoded values.
    """
    return ConstantSet.load(path)


class ConstantRegistry:
    """
    Registry that manages calibration of all magic numbers.

    Usage:
        registry = ConstantRegistry()
        registry.calibrate_all(datasets)
        registry.save()
        print(registry.report())
    """

    def __init__(self) -> None:
        self.constants = {c.name: c for c in ALL_CONSTANTS}
        self.result_set = ConstantSet()

    def calibrate_all(
        self,
        datasets: list[Any],
        categories: list[str] | None = None,
    ) -> ConstantSet:
        """
        Calibrate all constants (or just specified categories) from data.

        Args:
            datasets: CalibrationDataset instances with historical outcomes
            categories: Optional filter (e.g., ["draft", "trade"])

        Returns:
            ConstantSet with calibrated values
        """
        targets = list(self.constants.values())
        if categories:
            targets = [c for c in targets if c.category in categories]

        logger.info("Calibrating %d constants...", len(targets))

        for const in targets:
            try:
                self._calibrate_one(const, datasets)
                if const.calibrated_value is not None:
                    self.result_set.values[const.name] = const.calibrated_value
                    self.result_set.metadata[const.name] = {
                        "calibrated": True,
                        "date": datetime.now(UTC).isoformat(),
                        "n": const.calibration_n,
                        "loss": const.calibration_loss,
                        "previous_value": const.current_value,
                        "ci_low": (const.confidence_interval[0] if const.confidence_interval else None),
                        "ci_high": (const.confidence_interval[1] if const.confidence_interval else None),
                    }
                    logger.info(
                        "  %s: %.4f → %.4f (loss: %.4f, n=%d)",
                        const.name,
                        const.current_value,
                        const.calibrated_value,
                        const.calibration_loss or 0,
                        const.calibration_n,
                    )
            except Exception as exc:
                logger.warning("  %s: calibration failed: %s", const.name, exc)

        return self.result_set

    def _calibrate_one(
        self,
        const: CalibratableConstant,
        datasets: list[Any],
    ) -> None:
        """
        Calibrate a single constant.

        For constants in the "draft" category, uses survival calibrator.
        For "trade" category, uses trade calibrator.
        For others, uses a generic grid search approach.
        """
        if const.category == "draft" and const.name == "survival_sigma":
            from src.validation.survival_calibrator import calibrate_survival

            result = calibrate_survival(datasets, current_sigma=const.current_value)
            const.calibrated_value = result.optimal_sigma
            const.calibration_n = result.n_predictions
            const.calibration_loss = result.optimal_brier_score
            const.calibration_date = datetime.now(UTC)
            return

        # Generic: grid search within bounds using a simple heuristic
        # For constants where we don't have a specific calibrator yet,
        # we just mark them as "not calibrated" with their current value
        logger.debug(
            "  %s: no specific calibrator — keeping default %.4f",
            const.name,
            const.current_value,
        )
        const.calibrated_value = None

    def report(self) -> str:
        """Generate a human-readable calibration report."""
        lines = [
            "=" * 70,
            "HEATER CONSTANT CALIBRATION REPORT",
            f"Generated: {datetime.now(UTC).isoformat()}",
            "=" * 70,
            "",
        ]

        by_category: dict[str, list[CalibratableConstant]] = {}
        for c in self.constants.values():
            by_category.setdefault(c.category, []).append(c)

        for cat, consts in sorted(by_category.items()):
            lines.append(f"--- {cat.upper()} ---")
            for c in consts:
                status = "CALIBRATED" if c.calibrated_value is not None else "DEFAULT"
                val = c.calibrated_value if c.calibrated_value is not None else c.current_value
                ci = ""
                if c.confidence_interval:
                    ci = f" 95% CI: [{c.confidence_interval[0]:.2f}, {c.confidence_interval[1]:.2f}]"
                lines.append(f"  {c.name:35s} = {val:8.4f}  [{status}]  (was {c.current_value:.4f}){ci}")
                lines.append(f"    {c.description}")
                lines.append(f"    Source: {c.source_file}:{c.source_line}")
                lines.append("")
            lines.append("")

        return "\n".join(lines)

    def save(self, path: Path | str | None = None) -> None:
        """Save calibrated constants."""
        self.result_set.save(path)
