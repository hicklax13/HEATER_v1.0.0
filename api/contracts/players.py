"""Contract models for the player-id endpoints (player search + league rosters).

Both give the frontend a HEATER player_id source for the Trades build-a-trade
evaluator + compare-any-player. Both reuse the shared PlayerRef."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef


class PlayerSearchResponse(BaseModel):
    query: str
    results: list[PlayerRef] = Field(default_factory=list)  # most fantasy-relevant matches first


class LeagueRosterTeam(BaseModel):
    team_name: str
    manager: str = ""
    players: list[PlayerRef] = Field(default_factory=list)


class LeagueRostersResponse(BaseModel):
    teams: list[LeagueRosterTeam] = Field(default_factory=list)
