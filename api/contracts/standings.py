"""Contract models for the League Standings page."""

from __future__ import annotations

from pydantic import BaseModel


class TeamStanding(BaseModel):
    rank: int
    team_name: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points: float = 0.0
    category_ranks: dict[str, int] = {}


class StandingsResponse(BaseModel):
    teams: list[TeamStanding]
