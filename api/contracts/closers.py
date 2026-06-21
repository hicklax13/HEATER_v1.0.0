"""Contract models for the Closer Monitor page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class CloserEntry(BaseModel):
    team: str
    closer: PlayerRef | None = None
    role: str = ""  # e.g. "Closer", "Committee", "Setup"
    confidence: str = ""  # e.g. "Firm", "Shaky" (label derived from job_security)
    handcuffs: list[PlayerRef] = []  # next-in-line relievers, pool-resolved (mlb_id) when unambiguous
    # Saves-Finder actionable fields (numeric — surfaced from the closer grid):
    job_security: float = 0.0  # 0-1 firmness score (confidence is its label)
    security_color: str = ""  # engine heat color (e.g. green / yellow / red)
    projected_sv: float = 0.0  # projected saves
    era: float = 0.0
    whip: float = 0.0


class ClosersResponse(BaseModel):
    entries: list[CloserEntry]
