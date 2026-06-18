"""Contract models for the Leaders page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class LeaderRow(BaseModel):
    rank: int
    player: PlayerRef
    value: float


class LeadersResponse(BaseModel):
    category: str
    rows: list[LeaderRow]
