"""Tests for data freshness tracking."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.optimizer.data_freshness import DataFreshnessTracker, FreshnessStatus


class TestDataFreshnessTracker:
    """Verify freshness tracking and staleness detection."""

    @pytest.fixture()
    def _tracker(self):
        return DataFreshnessTracker()

    def test_record_and_check_fresh(self, _tracker):
        """Data recorded just now should be fresh."""
        _tracker.record("live_stats", ttl_hours=1.0)
        status = _tracker.check("live_stats")
        assert status == FreshnessStatus.FRESH

    def test_stale_data_detected(self, _tracker):
        """Data recorded 2 hours ago with 1-hour TTL should be stale."""
        _tracker.record(
            "live_stats",
            ttl_hours=1.0,
            timestamp=datetime.now(UTC) - timedelta(hours=2),
        )
        status = _tracker.check("live_stats")
        assert status == FreshnessStatus.STALE

    def test_unknown_source_returns_unknown(self, _tracker):
        """Unrecorded data source should return UNKNOWN status."""
        status = _tracker.check("nonexistent_source")
        assert status == FreshnessStatus.UNKNOWN

    def test_get_all_freshness(self, _tracker):
        """get_all() should return status for every recorded source."""
        _tracker.record("live_stats", ttl_hours=1.0)
        _tracker.record("projections", ttl_hours=24.0)
        result = _tracker.get_all()
        assert "live_stats" in result
        assert "projections" in result
        assert len(result) == 2

    def test_missing_data_audit(self, _tracker):
        """audit_missing() counts null/missing values in a DataFrame."""
        df = pd.DataFrame(
            [
                {
                    "player_name": "A",
                    "statcast_barrel": 0.12,
                    "recent_form_avg": None,
                },
                {
                    "player_name": "B",
                    "statcast_barrel": None,
                    "recent_form_avg": 0.280,
                },
                {
                    "player_name": "C",
                    "statcast_barrel": None,
                    "recent_form_avg": None,
                },
            ]
        )
        report = _tracker.audit_missing(df, columns=["statcast_barrel", "recent_form_avg"])
        assert report["statcast_barrel"]["missing"] == 2
        assert report["statcast_barrel"]["total"] == 3
        assert report["recent_form_avg"]["missing"] == 2

    def test_get_age_str_format(self, _tracker):
        """get_age_str() returns readable 'Xh Ym ago' format."""
        # Never-recorded source returns "never"
        assert _tracker.get_age_str("nonexistent") == "never"

        # Source recorded 2h 15m ago
        _tracker.record(
            "live_stats",
            ttl_hours=24.0,
            timestamp=datetime.now(UTC) - timedelta(hours=2, minutes=15),
        )
        age_str = _tracker.get_age_str("live_stats")
        assert "2h" in age_str
        assert "15m" in age_str
        assert "ago" in age_str

        # Source recorded just minutes ago (no hours component)
        _tracker.record(
            "recent",
            ttl_hours=1.0,
            timestamp=datetime.now(UTC) - timedelta(minutes=5),
        )
        age_str = _tracker.get_age_str("recent")
        assert "5m ago" == age_str
