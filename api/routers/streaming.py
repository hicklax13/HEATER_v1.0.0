"""Streaming router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.streaming import StreamingResponse
from api.deps import get_streaming_service

router = APIRouter(prefix="/api", tags=["streaming"])


@router.get("/streaming", response_model=StreamingResponse)
def get_streaming(
    date: str | None = None,
    limit: int = 25,
    service=Depends(get_streaming_service),
) -> StreamingResponse:
    return service.get_streaming(date=date, limit=limit)
