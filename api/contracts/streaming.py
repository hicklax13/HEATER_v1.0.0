"""Contract models for the Pitcher Streaming page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class StreamCandidate(BaseModel):
    player: PlayerRef
    team: str = ""
    opponent: str = ""
    score: float = 0.0
    actionable: bool = True
    status: str = ""  # e.g. "OK", "LOCKED", "FINAL"
    reason: str = ""


class StreamingResponse(BaseModel):
    date: str
    candidates: list[StreamCandidate]
