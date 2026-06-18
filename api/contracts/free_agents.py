"""Contract models for the Free Agents page. The frontend's
web/lib/data/types.ts is generated from the OpenAPI these produce."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class FreeAgentRec(BaseModel):
    add: PlayerRef
    drop: PlayerRef | None = None
    marginal_value: float
    categories_helped: list[str] = []
    ownership_pct: float | None = None
    rationale: str = ""


class FreeAgentsResponse(BaseModel):
    team_name: str
    recommendations: list[FreeAgentRec]
