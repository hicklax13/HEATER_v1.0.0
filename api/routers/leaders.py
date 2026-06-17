"""Leaders router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.contracts.leaders import LeadersResponse
from api.deps import get_leaders_service

router = APIRouter(prefix="/api", tags=["leaders"])


@router.get("/leaders", response_model=LeadersResponse)
def get_leaders(category: str = "HR", limit: int = 25, service=Depends(get_leaders_service)) -> LeadersResponse:
    return service.get_leaders(category, limit)
