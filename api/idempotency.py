"""Mutation idempotency contract (seam + types). The in-memory store is a
single-replica placeholder; a durable Redis/Postgres store lands in Phase 9.
NOT yet enforced on live write routes — wiring happens when the durable store
and provider writes are built (Phases 5/9)."""

from __future__ import annotations

from typing import Any, Protocol

from fastapi import Header

IDEMPOTENCY_HEADER = "Idempotency-Key"


class IdempotencyStore(Protocol):
    def get(self, key: str) -> Any | None: ...
    def put(self, key: str, result: Any) -> None: ...


class InMemoryIdempotencyStore:
    """Single-process store. First write per key wins (replays are stable)."""

    def __init__(self) -> None:
        self._d: dict[str, Any] = {}

    def get(self, key: str) -> Any | None:
        return self._d.get(key)

    def put(self, key: str, result: Any) -> None:
        # First-write-wins so a replay never overwrites the original result.
        self._d.setdefault(key, result)


async def idempotency_key(
    idempotency_key: str | None = Header(default=None),
) -> str | None:
    """FastAPI dependency that surfaces the optional Idempotency-Key header."""
    return idempotency_key
