"""Precomputation cache for expensive trade evaluation objects.

Spec reference: Section 17 Phase 6 item 27

Caches three expensive computations that don't need to be
recomputed on every trade evaluation:
  1. Gaussian copula (fit once from historical data or defaults)
  2. SGP denominators (computed from standings, changes ~daily)
  3. Category gap analysis (changes only when standings refresh)

The cache uses a simple in-memory dict with staleness tracking.
Each entry has a TTL (time-to-live) and is lazily refreshed.

Wires into:
  - src/engine/output/trade_evaluator.py: fast repeated evaluations
  - src/engine/portfolio/copula.py: cached copula instance
  - src/engine/portfolio/category_analysis.py: cached gap analysis
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Sentinel for distinguishing "not in cache" from cached None
_SENTINEL = object()

# Default TTLs (in seconds)
DEFAULT_TTL_COPULA: int = 86400  # 24 hours — correlation structure is stable
DEFAULT_TTL_SGP: int = 3600  # 1 hour — standings-based, may change
DEFAULT_TTL_GAP_ANALYSIS: int = 3600  # 1 hour — same as standings
DEFAULT_TTL_MARKET_VALUES: int = 1800  # 30 min — opponent needs shift


class CacheEntry:
    """Single cache entry with value and staleness tracking."""

    __slots__ = ("value", "created_at", "ttl", "hits")

    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.created_at = time.monotonic()
        self.ttl = ttl
        self.hits = 0

    @property
    def is_stale(self) -> bool:
        """Check if this entry has exceeded its TTL."""
        return (time.monotonic() - self.created_at) > self.ttl

    @property
    def age_seconds(self) -> float:
        """How old this entry is in seconds."""
        return time.monotonic() - self.created_at


class TradeEvalCache:
    """In-memory cache for expensive trade evaluation computations.

    Thread-safe for single-writer scenarios (Streamlit runs in one
    thread per user). Not designed for multi-process sharing.

    Usage:
        cache = TradeEvalCache()
        cache.set("copula", copula_instance, ttl=86400)
        copula = cache.get("copula")  # Returns None if stale
    """

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}
        self._total_hits: int = 0
        self._total_misses: int = 0

    def get(self, key: str) -> Any | None:
        """Retrieve a cached value, or None if missing/stale.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found or stale.
        """
        entry = self._store.get(key)
        if entry is None:
            self._total_misses += 1
            return None

        if entry.is_stale:
            self._total_misses += 1
            del self._store[key]
            return None

        entry.hits += 1
        self._total_hits += 1
        return entry.value

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Store a value in the cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds. If None, uses default (1 hour).
        """
        if ttl is None:
            ttl = DEFAULT_TTL_SGP
        self._store[key] = CacheEntry(value=value, ttl=ttl)

    def invalidate(self, key: str) -> bool:
        """Remove a specific key from cache.

        Args:
            key: Cache key to remove.

        Returns:
            True if the key existed and was removed.
        """
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> int:
        """Remove all entries from cache.

        Returns:
            Number of entries removed.
        """
        count = len(self._store)
        self._store.clear()
        return count

    def stats(self) -> dict[str, int | float]:
        """Return cache performance statistics.

        Returns:
            Dict with:
              - size: current number of entries
              - total_hits: total successful gets
              - total_misses: total failed gets
              - hit_rate: hits / (hits + misses) percentage
              - entries: per-key stats {key: {hits, age_s, stale}}
        """
        total = self._total_hits + self._total_misses
        hit_rate = (self._total_hits / total * 100.0) if total > 0 else 0.0

        entries = {}
        for key, entry in self._store.items():
            entries[key] = {
                "hits": entry.hits,
                "age_s": round(entry.age_seconds, 1),
                "stale": entry.is_stale,
            }

        return {
            "size": len(self._store),
            "total_hits": self._total_hits,
            "total_misses": self._total_misses,
            "hit_rate": round(hit_rate, 1),
            "entries": entries,
        }

    def get_or_compute(
        self,
        key: str,
        compute_fn: Any,
        ttl: int | None = None,
    ) -> Any:
        """Get from cache, or compute and cache if missing/stale.

        This is the primary pattern for using the cache: provide a
        key and a callable that produces the value. The callable is
        only invoked if the cache doesn't have a fresh value.

        Args:
            key: Cache key.
            compute_fn: Callable that returns the value to cache.
            ttl: Time-to-live in seconds.

        Returns:
            Cached or freshly computed value.
        """
        # Use sentinel to distinguish "not in cache" from cached None
        entry = self._store.get(key)
        if entry is not None and not entry.is_stale:
            entry.hits += 1
            self._total_hits += 1
            return entry.value

        # Miss or stale — recompute
        if entry is not None:
            del self._store[key]
        self._total_misses += 1

        value = compute_fn()
        self.set(key, value, ttl=ttl)
        return value


# Module-level singleton for global cache access
_global_cache: TradeEvalCache | None = None


def get_trade_cache() -> TradeEvalCache:
    """Get or create the global trade evaluation cache.

    Returns:
        The singleton TradeEvalCache instance.
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = TradeEvalCache()
    return _global_cache


def reset_trade_cache() -> None:
    """Reset the global cache (for testing or manual refresh)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache.clear()
    _global_cache = None
