"""Contract models for the Lineup Optimize endpoint.

The frontend's web/lib/data/types.ts is generated from the OpenAPI schema
these produce (see scripts/export_openapi.py).

LineupSlot maps directly to one assignment from LineupOptimizer.optimize_lineup()
  -> "assignments": [{"slot": str, "player_name": str, "player_id": int}, ...]

LineupOptimizeRequest carries the minimum inputs the pipeline needs:
  roster_ids    — list of player DB ids on the user's roster
  team_name     — for context/logging
  mode          — "quick" | "standard" | "full"  (default "quick" for API)
  weeks_remaining — passed through to the pipeline

LineupOptimizeResponse wraps the result with recommendations and metadata.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class LineupOptimizeRequest(BaseModel):
    roster_ids: list[int] = Field(default_factory=list)
    team_name: str
    mode: str = "quick"
    weeks_remaining: int = Field(default=16, ge=0, le=26)


class LineupSlot(BaseModel):
    slot: str
    player_name: str
    player_id: int


class LineupOptimizeResponse(BaseModel):
    team_name: str
    mode: str
    slots: list[LineupSlot]
    recommendations: list[str] = []
