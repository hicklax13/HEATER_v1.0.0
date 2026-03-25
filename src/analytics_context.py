"""
AnalyticsContext — the transparency spine for every HEATER recommendation.

Every pipeline (draft engine, trade engine, lineup optimizer, in-season pages)
creates an AnalyticsContext at entry and threads it through every module.
Each module stamps its status. The UI reads the context to show users
exactly what produced a recommendation and how much to trust it.

This replaces the current pattern of silent try/except fallbacks that
show green checkmarks regardless of what actually ran.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ModuleStatus(Enum):
    """Status of an analytical module's execution."""

    EXECUTED = "executed"  # Ran successfully with real inputs
    FALLBACK = "fallback"  # Ran but fell back to simpler logic
    SKIPPED = "skipped"  # Intentionally skipped (mode/config)
    DISABLED = "disabled"  # Required inputs not provided
    ERROR = "error"  # Failed, caught by try/except
    NOT_APPLICABLE = "n/a"  # Module doesn't apply to this pipeline


class DataQuality(Enum):
    """Quality tier for a data source."""

    LIVE = "live"  # Fetched within staleness threshold
    STALE = "stale"  # Exists but past staleness threshold
    SAMPLE = "sample"  # Sample/demo data, not real
    MISSING = "missing"  # Not available at all
    HARDCODED = "hardcoded"  # Baked into source code


class ConfidenceTier(Enum):
    """
    Overall confidence in a recommendation.

    Unlike the current fake confidence (50 + surplus * 25), this is
    computed from input quality and module execution — i.e., HOW the
    recommendation was produced, not WHAT it says.
    """

    HIGH = "high"  # All key modules ran, real data, calibrated constants
    MEDIUM = "medium"  # Some fallbacks, some stale data
    LOW = "low"  # Major fallbacks, stale/sample data
    EXPERIMENTAL = "experimental"  # Uncalibrated, treat as directional only


@dataclass
class ModuleResult:
    """Record of a single module's execution within a pipeline."""

    name: str
    status: ModuleStatus
    fallback_reason: str | None = None
    execution_ms: float = 0.0
    # How much this module changed the output vs. skipping it.
    # 0.0 = no influence (skipped/error), 1.0 = dominant influence.
    influence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def actually_ran(self) -> bool:
        return self.status in (ModuleStatus.EXECUTED, ModuleStatus.FALLBACK)


@dataclass
class DataSourceStatus:
    """Freshness and quality of a data source feeding a recommendation."""

    source: str  # e.g., "fangraphs_steamer", "yahoo_rosters", "mlb_stats_api"
    quality: DataQuality
    last_refresh: datetime | None = None
    record_count: int = 0
    notes: str = ""

    @property
    def age_hours(self) -> float | None:
        if self.last_refresh is None:
            return None
        delta = datetime.now(UTC) - self.last_refresh
        return delta.total_seconds() / 3600


@dataclass
class AnalyticsContext:
    """
    Transparency record for a single recommendation pipeline execution.

    Created at pipeline entry. Each module stamps it. UI reads it to
    show the user exactly what happened.

    Usage:
        ctx = AnalyticsContext(pipeline="trade_engine")
        ctx.stamp_data("yahoo_rosters", DataQuality.LIVE, last_refresh=..., record_count=276)
        ctx.stamp_data("fangraphs_projections", DataQuality.STALE, notes="Only 2/7 systems fetched")

        with ctx.track_module("phase1_sgp") as mod:
            result = compute_sgp(...)
            mod.influence = 0.9

        # If module falls back:
        with ctx.track_module("phase2_mc") as mod:
            if not enable_mc:
                mod.status = ModuleStatus.DISABLED
                mod.fallback_reason = "enable_mc=False (default)"
            else:
                ...

        # UI reads:
        ctx.confidence_tier   # HIGH / MEDIUM / LOW / EXPERIMENTAL
        ctx.quality_score     # 0.0 - 1.0
        ctx.user_warnings     # ["Projections from 1 system (need 5+)", ...]
        ctx.modules_summary   # "4/7 modules ran, 2 fell back, 1 disabled"
    """

    pipeline: str  # "draft_engine", "trade_engine", "lineup_optimizer", etc.
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    modules: dict[str, ModuleResult] = field(default_factory=dict)
    data_sources: dict[str, DataSourceStatus] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _active_module: str | None = field(default=None, repr=False)
    _module_start: float = field(default=0.0, repr=False)

    # ------------------------------------------------------------------
    # Module tracking
    # ------------------------------------------------------------------

    class _ModuleTracker:
        """Context manager for timing and recording a module's execution."""

        def __init__(self, ctx: AnalyticsContext, name: str):
            self._ctx = ctx
            self._name = name
            self._result = ModuleResult(name=name, status=ModuleStatus.EXECUTED)
            ctx.modules[name] = self._result

        def __enter__(self) -> ModuleResult:
            self._ctx._active_module = self._name
            self._ctx._module_start = time.perf_counter()
            return self._result

        def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
            elapsed = (time.perf_counter() - self._ctx._module_start) * 1000
            self._result.execution_ms = round(elapsed, 1)
            self._ctx._active_module = None
            if exc_type is not None:
                self._result.status = ModuleStatus.ERROR
                self._result.fallback_reason = f"{exc_type.__name__}: {exc_val}"
                self._result.influence = 0.0
                self._ctx.warnings.append(f"Module '{self._name}' failed: {exc_val}")
                return True  # Swallow exception (graceful degradation)
            return False

    def track_module(self, name: str) -> _ModuleTracker:
        """Context manager: tracks execution, timing, and errors for a module."""
        return self._ModuleTracker(self, name)

    def stamp_module(
        self,
        name: str,
        status: ModuleStatus,
        *,
        reason: str | None = None,
        influence: float = 0.0,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Quick stamp for modules that don't need the context manager."""
        self.modules[name] = ModuleResult(
            name=name,
            status=status,
            fallback_reason=reason,
            influence=influence,
            details=details or {},
        )

    # ------------------------------------------------------------------
    # Data source tracking
    # ------------------------------------------------------------------

    def stamp_data(
        self,
        source: str,
        quality: DataQuality,
        *,
        last_refresh: datetime | None = None,
        record_count: int = 0,
        notes: str = "",
    ) -> None:
        """Record the quality and freshness of a data source."""
        self.data_sources[source] = DataSourceStatus(
            source=source,
            quality=quality,
            last_refresh=last_refresh,
            record_count=record_count,
            notes=notes,
        )

    # ------------------------------------------------------------------
    # Computed properties for UI
    # ------------------------------------------------------------------

    @property
    def quality_score(self) -> float:
        """
        0.0 - 1.0 score reflecting overall analytical quality.

        Factors:
        - What fraction of modules actually ran (vs. fell back / skipped)?
        - Are data sources fresh and complete?
        - Were any critical modules disabled?

        This is NOT a prediction accuracy score. It measures input quality
        and pipeline completeness — i.e., "how well-informed is this
        recommendation?" not "how correct is it?"
        """
        if not self.modules and not self.data_sources:
            return 0.0

        # Module score: fraction of modules that actually ran
        module_scores = []
        for mod in self.modules.values():
            if mod.status == ModuleStatus.EXECUTED:
                module_scores.append(1.0)
            elif mod.status == ModuleStatus.FALLBACK:
                module_scores.append(0.5)
            elif mod.status in (ModuleStatus.SKIPPED, ModuleStatus.NOT_APPLICABLE):
                continue  # Don't penalize intentional skips
            else:  # DISABLED, ERROR
                module_scores.append(0.0)

        module_avg = sum(module_scores) / len(module_scores) if module_scores else 0.5

        # Data score: quality of inputs
        data_scores = []
        for ds in self.data_sources.values():
            if ds.quality == DataQuality.LIVE:
                data_scores.append(1.0)
            elif ds.quality == DataQuality.STALE:
                data_scores.append(0.5)
            elif ds.quality == DataQuality.HARDCODED:
                data_scores.append(0.3)
            elif ds.quality == DataQuality.SAMPLE:
                data_scores.append(0.1)
            else:  # MISSING
                data_scores.append(0.0)

        data_avg = sum(data_scores) / len(data_scores) if data_scores else 0.5

        # Weighted: modules 60%, data 40%
        return round(0.6 * module_avg + 0.4 * data_avg, 2)

    @property
    def confidence_tier(self) -> ConfidenceTier:
        """Map quality score to a human-readable tier."""
        score = self.quality_score
        if score >= 0.80:
            return ConfidenceTier.HIGH
        if score >= 0.55:
            return ConfidenceTier.MEDIUM
        if score >= 0.30:
            return ConfidenceTier.LOW
        return ConfidenceTier.EXPERIMENTAL

    @property
    def user_warnings(self) -> list[str]:
        """Warnings that should be shown in the UI."""
        warnings = list(self.warnings)

        # Auto-generate warnings from data quality
        for ds in self.data_sources.values():
            if ds.quality == DataQuality.SAMPLE:
                warnings.append(f"Using sample data for {ds.source} — not real projections")
            elif ds.quality == DataQuality.MISSING:
                warnings.append(f"No data available for {ds.source}")
            elif ds.quality == DataQuality.STALE and ds.age_hours is not None:
                if ds.age_hours > 48:
                    warnings.append(f"{ds.source} data is {ds.age_hours:.0f}h old")
            elif ds.quality == DataQuality.HARDCODED:
                warnings.append(f"{ds.source} uses hardcoded values (not live data)")

        # Auto-generate warnings from module failures
        disabled = [m.name for m in self.modules.values() if m.status == ModuleStatus.DISABLED]
        if disabled:
            warnings.append(f"{len(disabled)} module(s) disabled due to missing inputs: " + ", ".join(disabled))

        errored = [m.name for m in self.modules.values() if m.status == ModuleStatus.ERROR]
        if errored:
            warnings.append(f"{len(errored)} module(s) failed: " + ", ".join(errored))

        return warnings

    @property
    def modules_summary(self) -> str:
        """One-line summary of module execution for UI display."""
        if not self.modules:
            return "No modules tracked"
        total = len(self.modules)
        ran = sum(1 for m in self.modules.values() if m.status == ModuleStatus.EXECUTED)
        fell_back = sum(1 for m in self.modules.values() if m.status == ModuleStatus.FALLBACK)
        failed = sum(1 for m in self.modules.values() if m.status in (ModuleStatus.ERROR, ModuleStatus.DISABLED))
        parts = [f"{ran}/{total} modules ran"]
        if fell_back:
            parts.append(f"{fell_back} used fallback")
        if failed:
            parts.append(f"{failed} unavailable")
        return ", ".join(parts)

    # ------------------------------------------------------------------
    # Serialization for logging / debugging
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize for logging, caching, or UI consumption."""
        return {
            "pipeline": self.pipeline,
            "created_at": self.created_at.isoformat(),
            "quality_score": self.quality_score,
            "confidence_tier": self.confidence_tier.value,
            "modules_summary": self.modules_summary,
            "warnings": self.user_warnings,
            "modules": {
                name: {
                    "status": m.status.value,
                    "fallback_reason": m.fallback_reason,
                    "execution_ms": m.execution_ms,
                    "influence": m.influence,
                }
                for name, m in self.modules.items()
            },
            "data_sources": {
                name: {
                    "quality": ds.quality.value,
                    "age_hours": ds.age_hours,
                    "record_count": ds.record_count,
                    "notes": ds.notes,
                }
                for name, ds in self.data_sources.items()
            },
        }
