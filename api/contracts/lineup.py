"""Contract models for the Lineup Optimizer page."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef


class LineupOptimizeRequest(BaseModel):
    team_name: str
    date: str | None = None
    scope: str = "rest_of_season"


class LineupSlot(BaseModel):
    slot: str
    player: PlayerRef
    action: str  # "START" | "SIT"
    projected: float = 0.0  # per-player daily value is daily-mode (DCV) → slice 2
    forced_start: bool = False
    reason: str | None = None
    status: str = "start"  # "start" | "bench" (sit/off distinction → daily-mode slice 2)


class CatImpact(BaseModel):
    """A projected category total for the optimal lineup (display string + trend)."""

    key: str
    proj: str
    trend: str = "flat"  # "up" | "down" | "flat" (flat for now — no baseline to diff)


class LineupOptimizeResponse(BaseModel):
    team_name: str
    date: str
    slots: list[LineupSlot]  # the optimal STARTERS
    summary: str = ""
    bench: list[LineupSlot] = Field(
        default_factory=list
    )  # ALL non-starter roster players (incl IL/BN — frontend filters by slot)
    optimal: bool = False  # is the user's current Yahoo lineup already the optimal one?
    impact: list[CatImpact] = Field(default_factory=list)  # projected category totals for the lineup
