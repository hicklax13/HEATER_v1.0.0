"""Contract models for the Leaders page."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef


class LeaderRow(BaseModel):
    rank: int
    player: PlayerRef
    value: float


class LeadersResponse(BaseModel):
    category: str
    rows: list[LeaderRow]


class OverallLeaderRow(BaseModel):
    rank: int
    player: PlayerRef
    value: float = 0.0  # 0-100 lens score
    stats: list[str] = Field(default_factory=list)  # 3 key stats, e.g. ["24 HR","58 R",".322 AVG"]
    trend: str = "flat"  # up/down/flat
    tag: str = ""  # ""/hot/cold/breakout/sell
    note: str = ""
    hitter: bool = True


class LeadersOverallResponse(BaseModel):
    lens: str = "overall"
    rows: list[OverallLeaderRow] = Field(default_factory=list)
