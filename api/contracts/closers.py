"""Contract models for the Closer Monitor page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class CloserEntry(BaseModel):
    team: str
    closer: PlayerRef | None = None
    role: str = ""  # e.g. "Closer", "Committee", "Setup"
    confidence: str = ""  # e.g. "Firm", "Shaky"
    handcuffs: list[PlayerRef] = []


class ClosersResponse(BaseModel):
    entries: list[CloserEntry]
