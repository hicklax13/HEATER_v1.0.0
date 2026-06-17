"""Contract models for the Trade Finder page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class TradeSuggestion(BaseModel):
    partner_team: str
    giving: list[PlayerRef] = []  # players YOU give
    receiving: list[PlayerRef] = []  # players you receive
    net_sgp: float = 0.0
    rationale: str = ""


class TradeFinderResponse(BaseModel):
    team_name: str
    suggestions: list[TradeSuggestion] = []
