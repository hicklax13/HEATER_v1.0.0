"""Contract models shared across pages."""

from __future__ import annotations

from pydantic import BaseModel


class PlayerRef(BaseModel):
    id: int
    mlb_id: int | None = None
    name: str
    positions: str
    team_abbr: str | None = None
    team_id: int | None = None
    yahoo_player_key: str | None = None
