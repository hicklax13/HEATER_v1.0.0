"""Contract models shared across pages."""

from __future__ import annotations

from pydantic import BaseModel


class PlayerRef(BaseModel):
    id: int
    name: str
    positions: str
    yahoo_player_key: str | None = None
