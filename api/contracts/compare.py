"""Contract models for the Player Compare page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class ComparePlayer(BaseModel):
    player: PlayerRef
    stats: dict[str, float] = {}  # category -> projected value


class CompareResponse(BaseModel):
    categories: list[str] = []  # the categories compared, in order
    players: list[ComparePlayer] = []
