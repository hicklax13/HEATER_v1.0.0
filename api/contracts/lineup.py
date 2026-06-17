"""Contract models for the Lineup Optimizer page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class LineupOptimizeRequest(BaseModel):
    team_name: str
    date: str | None = None
    scope: str = "rest_of_season"


class LineupSlot(BaseModel):
    slot: str
    player: PlayerRef
    action: str  # "START" | "SIT"
    projected: float
    forced_start: bool = False
    reason: str | None = None


class LineupOptimizeResponse(BaseModel):
    team_name: str
    date: str
    slots: list[LineupSlot]
    summary: str = ""
