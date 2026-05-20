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


# INFRA-F4: default TTLs for refresh_log hydration. Mirrors CLAUDE.md's
# Data Sources table; unmapped sources fall back to _DEFAULT_TTL_HOURS.
_DEFAULT_TTLS = {
    "players": 168.0,  # 7 days
    "projections": 24.0,
    "ros_projections": 4.0,
    "historical_stats": 720.0,  # 30 days
    "season_stats": 1.0,
    "yahoo_data": 0.5,
    "yahoo_rosters": 0.5,
    "yahoo_standings": 0.5,
    "yahoo_free_agents": 1.0,
    "yahoo_transactions": 0.25,
    "game_day": 2.0,
    "team_strength": 24.0,
    "park_factors": 720.0,
    "catcher_framing": 168.0,
    "umpire_tendencies": 24.0,
    "depth_charts": 168.0,
    "news": 1.0,
    "news_intelligence": 1.0,
    "ecr_consensus": 24.0,
    "adp_sources": 24.0,
    "sprint_speed": 24.0,
    "bat_speed": 24.0,
    "stuff_plus": 24.0,
    "forty_man": 168.0,
    "pvb_splits": 24.0,
    "batting_stats": 24.0,
    "injury_data": 1.0,
    "prospect_rankings": 168.0,
}
_DEFAULT_TTL_HOURS = 24.0  # for sources not in _DEFAULT_TTLS


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
        # INFRA-F4: hydrate from refresh_log on init so callers outside the
        # lineup-optimizer flow (which calls tracker.record() inline) still
        # see FRESH/STALE rather than UNKNOWN for recently refreshed sources.
        self._populate_from_refresh_log()

    def _populate_from_refresh_log(self) -> None:
        """Read refresh_log and seed _sources with timestamps from recent
        successful refreshes. Failures (non-success status, bad timestamp,
        missing table, etc.) are logged at DEBUG and do not raise."""
        try:
            from src.database import get_connection
        except Exception as exc:
            logger.debug("DataFreshnessTracker: could not import get_connection: %s", exc)
            return

        try:
            conn = get_connection()
        except Exception as exc:
            logger.debug("DataFreshnessTracker: get_connection failed: %s", exc)
            return

        try:
            cur = conn.execute("SELECT source, last_refresh, status FROM refresh_log")
            rows = cur.fetchall()
        except Exception as exc:
            logger.debug("DataFreshnessTracker: refresh_log query failed: %s", exc)
            try:
                conn.close()
            except Exception:
                pass
            return

        # 2026-05-20 SFH M-3: load partial / cached / skipped rows in addition
        # to success. Each indicates data was written (or a deliberate fallback
        # was used) — the freshness tracker should reflect the last touch so
        # the UI can show "data is 4h old (partial)" instead of "no data".
        # Skip only error / unknown (no data) and no_data (source returned
        # nothing — tracking the empty-fetch timestamp wastes UI space).
        _LOADABLE_STATUSES = {"success", "partial", "cached", "skipped"}

        for source, last_refresh_str, status in rows:
            if status not in _LOADABLE_STATUSES or not last_refresh_str:
                continue
            try:
                ts = datetime.fromisoformat(last_refresh_str.replace(" ", "T"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=UTC)
            except (ValueError, TypeError):
                continue
            ttl = _DEFAULT_TTLS.get(source, _DEFAULT_TTL_HOURS)
            self._sources[source] = _SourceRecord(
                name=source,
                timestamp=ts,
                ttl_hours=ttl,
                data_as_of="",
                source_label="refresh_log",
            )

        try:
            conn.close()
        except Exception:
            pass

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
