"""Contract models for the Free Agents endpoint.

The frontend's web/lib/data/types.ts is generated from the OpenAPI schema
these produce (see scripts/export_openapi.py).

Each FreeAgentRec maps directly to one entry from recommend_fa_moves():
  add_name / drop_name  — player names from "add_name"/"drop_name" keys
  net_sgp_delta         — "net_sgp_delta" (float, can be negative)
  urgency_categories    — "urgency_categories" (list of category strings)
  reasoning             — "reasoning" (human-readable string)
"""

from __future__ import annotations

from pydantic import BaseModel


class FreeAgentRec(BaseModel):
    add_name: str
    add_positions: str
    drop_name: str
    drop_positions: str
    net_sgp_delta: float
    urgency_categories: list[str] = []
    reasoning: str = ""


class FreeAgentsResponse(BaseModel):
    team_name: str
    adds_remaining: int
    recommendations: list[FreeAgentRec]
