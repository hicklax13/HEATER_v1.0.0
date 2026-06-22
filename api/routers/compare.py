"""Compare router. THIN — delegates to the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from api.auth import require_login
from api.contracts.compare import CompareResponse
from api.deps import get_compare_service

router = APIRouter(prefix="/api", tags=["compare"], dependencies=[Depends(require_login)])


@router.get("/compare", response_model=CompareResponse)
def get_compare(
    ids: str = "",
    service=Depends(get_compare_service),
) -> CompareResponse:
    player_ids: list[int] = []
    for part in ids.split(","):
        part = part.strip()
        if part:
            try:
                player_ids.append(int(part))
            except ValueError:
                pass
    return service.compare(player_ids=player_ids)
