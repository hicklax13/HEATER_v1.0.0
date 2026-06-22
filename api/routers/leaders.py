"""Leaders router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import require_login
from api.contracts.leaders import LeadersOverallResponse, LeadersResponse
from api.deps import get_leaders_overall_service, get_leaders_service

router = APIRouter(prefix="/api", tags=["leaders"], dependencies=[Depends(require_login)])


@router.get("/leaders", response_model=LeadersResponse)
def get_leaders(category: str = "HR", limit: int = 25, service=Depends(get_leaders_service)) -> LeadersResponse:
    return service.get_leaders(category, limit)


@router.get("/leaders/overall", response_model=LeadersOverallResponse)
def get_leaders_overall(
    lens: str = "overall", limit: int = 25, service=Depends(get_leaders_overall_service)
) -> LeadersOverallResponse:
    return service.get_leaders_overall(lens, limit)
