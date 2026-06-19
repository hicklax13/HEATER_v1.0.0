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


class StatItem(BaseModel):
    label: str
    value: str


class FreeAgentPoolItem(BaseModel):
    player: PlayerRef
    rank: int
    value: float  # marginal value normalized to 0-100 (top FA in pool = 100)
    own_pct: float = 0.0
    own_delta: float = 0.0
    hitter: bool = True
    stats: list[StatItem] = []
    fit: str = ""  # category key this FA most helps (e.g. "SB")
    tag: str | None = None


class FreeAgentPoolResponse(BaseModel):
    top_need: str = ""  # the user's biggest category gap
    free_agents: list[FreeAgentPoolItem] = []
