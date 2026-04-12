"""Data freshness tracking for the Lineup Optimizer.

Tracks when each data source was last refreshed and whether it's
within its staleness TTL.  Surfaces missing-data audits for
quality monitoring.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class FreshnessStatus(Enum):
    """Staleness status for a data source."""

    FRESH = "fresh"
    STALE = "stale"
    UNKNOWN = "unknown"


@dataclass
class _SourceRecord:
    name: str
    timestamp: datetime
    ttl_hours: float
    data_as_of: str  # human-readable description of what the data represents
    source_label: str  # where the data came from (e.g. "Yahoo API", "MLB Stats API")


class DataFreshnessTracker:
    """Track last-refresh timestamps and TTLs for optimizer data sources."""

    def __init__(self) -> None:
        self._sources: dict[str, _SourceRecord] = {}

    def record(
        self,
        source: str,
        ttl_hours: float,
        timestamp: datetime | None = None,
        data_as_of: str = "",
        source_label: str = "",
    ) -> None:
        """Record a data fetch with its TTL.

        Args:
            source: Data source name (e.g. "live_stats", "projections").
            ttl_hours: Hours before this data is considered stale.
            timestamp: When the data was fetched.  Defaults to now.
            data_as_of: What the data represents (e.g. "Games through 4/11").
            source_label: Data provider (e.g. "Yahoo API", "MLB Stats API").
        """
        self._sources[source] = _SourceRecord(
            name=source,
            timestamp=timestamp or datetime.now(UTC),
            ttl_hours=ttl_hours,
            data_as_of=data_as_of or "",
            source_label=source_label or "",
        )

    def check(self, source: str) -> FreshnessStatus:
        """Check if a data source is fresh or stale."""
        rec = self._sources.get(source)
        if rec is None:
            return FreshnessStatus.UNKNOWN
        age = datetime.now(UTC) - rec.timestamp
        if age <= timedelta(hours=rec.ttl_hours):
            return FreshnessStatus.FRESH
        return FreshnessStatus.STALE

    def get_age_str(self, source: str) -> str:
        """Return human-readable age string like '2h 15m ago'."""
        rec = self._sources.get(source)
        if rec is None:
            return "never"
        age = datetime.now(UTC) - rec.timestamp
        hours = int(age.total_seconds() // 3600)
        mins = int((age.total_seconds() % 3600) // 60)
        if hours > 0:
            return f"{hours}h {mins}m ago"
        return f"{mins}m ago"

    def get_all(self) -> dict[str, dict]:
        """Return freshness status for all recorded sources."""
        result = {}
        for name, rec in self._sources.items():
            status = self.check(name)
            result[name] = {
                "status": status.value,
                "timestamp": rec.timestamp.isoformat(),
                "ttl_hours": rec.ttl_hours,
                "age": self.get_age_str(name),
                "data_as_of": rec.data_as_of,
                "source_label": rec.source_label,
            }
        return result

    @staticmethod
    def audit_missing(
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, dict]:
        """Audit a DataFrame for missing/null values.

        Args:
            df: DataFrame to audit.
            columns: Specific columns to check.  Defaults to all.

        Returns:
            Dict mapping column name to {missing, total, pct} counts.
        """
        cols = columns or list(df.columns)
        report: dict[str, dict] = {}
        for col in cols:
            if col not in df.columns:
                report[col] = {"missing": len(df), "total": len(df), "pct": 100.0}
                continue
            n_missing = int(df[col].isna().sum())
            n_total = len(df)
            report[col] = {
                "missing": n_missing,
                "total": n_total,
                "pct": round(n_missing / max(n_total, 1) * 100, 1),
            }
        return report
