"""Tests for AnalyticsContext — the transparency spine."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from src.analytics_context import (
    AnalyticsContext,
    ConfidenceTier,
    DataQuality,
    ModuleResult,
    ModuleStatus,
)


class TestAnalyticsContextBasics:
    """Core context tracking."""

    def test_empty_context_has_zero_quality(self):
        ctx = AnalyticsContext(pipeline="test")
        assert ctx.quality_score == 0.0
        assert ctx.confidence_tier == ConfidenceTier.EXPERIMENTAL

    def test_all_modules_executed_high_quality(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("mod1", ModuleStatus.EXECUTED, influence=0.5)
        ctx.stamp_module("mod2", ModuleStatus.EXECUTED, influence=0.5)
        ctx.stamp_data("source1", DataQuality.LIVE, record_count=100)
        assert ctx.quality_score >= 0.8
        assert ctx.confidence_tier == ConfidenceTier.HIGH

    def test_fallback_modules_reduce_quality(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("mod1", ModuleStatus.EXECUTED, influence=0.5)
        ctx.stamp_module("mod2", ModuleStatus.FALLBACK, reason="PyMC not installed")
        ctx.stamp_data("source1", DataQuality.LIVE, record_count=100)
        score = ctx.quality_score
        # 0.6*(1.0+0.5)/2 + 0.4*1.0 = 0.85 — still HIGH because data is live
        # Fallback reduces module avg but one fallback isn't catastrophic
        assert 0.7 < score < 1.0

    def test_disabled_modules_reduce_quality(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("mod1", ModuleStatus.EXECUTED, influence=0.5)
        ctx.stamp_module("mod2", ModuleStatus.DISABLED, reason="No opponent data")
        ctx.stamp_module("mod3", ModuleStatus.DISABLED, reason="No schedule")
        ctx.stamp_data("source1", DataQuality.LIVE, record_count=100)
        score = ctx.quality_score
        assert score < 0.7

    def test_stale_data_reduces_quality(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("mod1", ModuleStatus.EXECUTED, influence=1.0)
        ctx.stamp_data(
            "projections",
            DataQuality.STALE,
            last_refresh=datetime.now(UTC) - timedelta(hours=72),
            record_count=500,
        )
        score = ctx.quality_score
        # 0.6*1.0 + 0.4*0.5 = 0.80 — stale data brings it down from 1.0
        assert score <= 0.80

    def test_sample_data_severely_reduces_quality(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("mod1", ModuleStatus.EXECUTED, influence=1.0)
        ctx.stamp_data("players", DataQuality.SAMPLE, record_count=190)
        score = ctx.quality_score
        # 0.6*1.0 + 0.4*0.1 = 0.64 — sample data is low quality
        assert score < 0.7
        assert ctx.confidence_tier == ConfidenceTier.MEDIUM

    def test_missing_data_is_worst(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("mod1", ModuleStatus.EXECUTED, influence=1.0)
        ctx.stamp_data("players", DataQuality.MISSING)
        score = ctx.quality_score
        # 0.6*1.0 + 0.4*0.0 = 0.60 — missing data is worst data tier
        assert score <= 0.60

    def test_skipped_modules_dont_penalize(self):
        """Intentionally skipped modules (mode selection) shouldn't reduce quality."""
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("core", ModuleStatus.EXECUTED, influence=1.0)
        ctx.stamp_module("optional", ModuleStatus.SKIPPED, reason="Quick mode")
        ctx.stamp_data("source1", DataQuality.LIVE, record_count=100)
        # Only 'core' should count — skipped is ignored
        assert ctx.quality_score >= 0.8


class TestModuleTracking:
    """Context manager for module tracking."""

    def test_track_module_success(self):
        ctx = AnalyticsContext(pipeline="test")
        with ctx.track_module("phase1") as mod:
            mod.influence = 0.9
            # Simulate work
            _ = sum(range(100))

        assert "phase1" in ctx.modules
        assert ctx.modules["phase1"].status == ModuleStatus.EXECUTED
        assert ctx.modules["phase1"].execution_ms >= 0  # May be 0.0 on fast machines
        assert ctx.modules["phase1"].influence == 0.9

    def test_track_module_exception_swallowed(self):
        """Exceptions inside track_module are caught (graceful degradation)."""
        ctx = AnalyticsContext(pipeline="test")
        with ctx.track_module("broken_module") as mod:
            mod.influence = 0.5
            raise ValueError("Something went wrong")

        assert ctx.modules["broken_module"].status == ModuleStatus.ERROR
        assert "ValueError" in ctx.modules["broken_module"].fallback_reason
        assert ctx.modules["broken_module"].influence == 0.0

    def test_track_module_exception_generates_warning(self):
        ctx = AnalyticsContext(pipeline="test")
        with ctx.track_module("broken") as mod:
            raise RuntimeError("Oops")

        assert any("broken" in w for w in ctx.warnings)


class TestWarnings:
    """Auto-generated user-facing warnings."""

    def test_sample_data_warning(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_data("players", DataQuality.SAMPLE, record_count=190)
        warnings = ctx.user_warnings
        assert any("sample data" in w.lower() for w in warnings)

    def test_missing_data_warning(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_data("yahoo_rosters", DataQuality.MISSING)
        warnings = ctx.user_warnings
        assert any("no data" in w.lower() for w in warnings)

    def test_stale_data_warning_only_after_48h(self):
        ctx = AnalyticsContext(pipeline="test")
        # 24h old — no warning
        ctx.stamp_data(
            "stats",
            DataQuality.STALE,
            last_refresh=datetime.now(UTC) - timedelta(hours=24),
        )
        warnings = ctx.user_warnings
        assert not any("old" in w.lower() for w in warnings)

        # 72h old — warning
        ctx2 = AnalyticsContext(pipeline="test")
        ctx2.stamp_data(
            "stats",
            DataQuality.STALE,
            last_refresh=datetime.now(UTC) - timedelta(hours=72),
        )
        warnings2 = ctx2.user_warnings
        assert any("72h old" in w for w in warnings2)

    def test_disabled_modules_warning(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("matchup_adj", ModuleStatus.DISABLED, reason="No schedule")
        ctx.stamp_module("start_sit", ModuleStatus.DISABLED, reason="No opponent")
        warnings = ctx.user_warnings
        assert any("2 module(s) disabled" in w for w in warnings)

    def test_hardcoded_data_warning(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_data("park_factors", DataQuality.HARDCODED, notes="2024 values")
        warnings = ctx.user_warnings
        assert any("hardcoded" in w.lower() for w in warnings)


class TestModulesSummary:
    """One-line summary for UI display."""

    def test_all_ran(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("a", ModuleStatus.EXECUTED)
        ctx.stamp_module("b", ModuleStatus.EXECUTED)
        assert ctx.modules_summary == "2/2 modules ran"

    def test_mixed_status(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("a", ModuleStatus.EXECUTED)
        ctx.stamp_module("b", ModuleStatus.FALLBACK, reason="no data")
        ctx.stamp_module("c", ModuleStatus.DISABLED, reason="no input")
        summary = ctx.modules_summary
        assert "1/3 modules ran" in summary
        assert "1 used fallback" in summary
        assert "1 unavailable" in summary

    def test_empty(self):
        ctx = AnalyticsContext(pipeline="test")
        assert ctx.modules_summary == "No modules tracked"


class TestSerialization:
    """to_dict for logging and caching."""

    def test_roundtrip_basic(self):
        ctx = AnalyticsContext(pipeline="trade_engine")
        ctx.stamp_module("phase1", ModuleStatus.EXECUTED, influence=0.9)
        ctx.stamp_data("projections", DataQuality.LIVE, record_count=750)

        d = ctx.to_dict()
        assert d["pipeline"] == "trade_engine"
        assert d["confidence_tier"] == "high"
        assert "phase1" in d["modules"]
        assert d["modules"]["phase1"]["status"] == "executed"
        assert "projections" in d["data_sources"]
        assert d["data_sources"]["projections"]["quality"] == "live"

    def test_warnings_in_dict(self):
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_data("players", DataQuality.SAMPLE)
        d = ctx.to_dict()
        assert len(d["warnings"]) > 0


class TestConfidenceTierBoundaries:
    """Verify tier assignment at boundary values."""

    def test_tier_high_boundary(self):
        """score >= 0.80 → HIGH"""
        ctx = AnalyticsContext(pipeline="test")
        # 2 executed modules + 1 live source = high quality
        ctx.stamp_module("a", ModuleStatus.EXECUTED)
        ctx.stamp_module("b", ModuleStatus.EXECUTED)
        ctx.stamp_data("src", DataQuality.LIVE)
        assert ctx.confidence_tier == ConfidenceTier.HIGH

    def test_tier_experimental_boundary(self):
        """score < 0.30 → EXPERIMENTAL"""
        ctx = AnalyticsContext(pipeline="test")
        ctx.stamp_module("a", ModuleStatus.ERROR, reason="crash")
        ctx.stamp_module("b", ModuleStatus.DISABLED, reason="no input")
        ctx.stamp_data("src", DataQuality.MISSING)
        assert ctx.confidence_tier == ConfidenceTier.EXPERIMENTAL


class TestModuleResultProperties:
    """ModuleResult helper properties."""

    def test_actually_ran_for_executed(self):
        mr = ModuleResult(name="test", status=ModuleStatus.EXECUTED)
        assert mr.actually_ran is True

    def test_actually_ran_for_fallback(self):
        mr = ModuleResult(name="test", status=ModuleStatus.FALLBACK)
        assert mr.actually_ran is True

    def test_actually_ran_false_for_disabled(self):
        mr = ModuleResult(name="test", status=ModuleStatus.DISABLED)
        assert mr.actually_ran is False

    def test_actually_ran_false_for_error(self):
        mr = ModuleResult(name="test", status=ModuleStatus.ERROR)
        assert mr.actually_ran is False
