"""Contract models for the Player Databank page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class SeasonStat(BaseModel):
    year: int
    stats: dict[str, float] = {}


class DatabankResponse(BaseModel):
    player: PlayerRef
    seasons: list[SeasonStat] = []  # newest first
