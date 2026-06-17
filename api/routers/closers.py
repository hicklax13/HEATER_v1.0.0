"""Closer Monitor router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.closers import ClosersResponse
from api.deps import get_closer_service

router = APIRouter(prefix="/api", tags=["closers"])


@router.get("/closers", response_model=ClosersResponse)
def get_closers(service=Depends(get_closer_service)) -> ClosersResponse:
    return service.get_closers()
