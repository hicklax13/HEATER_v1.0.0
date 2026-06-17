"""Contract models for the My Team page. This is the canonical API contract;
the frontend's web/lib/data/types.ts is generated from the OpenAPI schema
these produce (see scripts/export_openapi.py)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MatchupHero(BaseModel):
    opponent: str
    week: int
    win_prob: float = Field(ge=0.0, le=1.0)
    tie_prob: float = Field(ge=0.0, le=1.0)
    loss_prob: float = Field(ge=0.0, le=1.0)


class CategoryLine(BaseModel):
    cat: str
    you: float
    opp: float
    edge: float
    win_prob: float = Field(ge=0.0, le=1.0)
    inverse: bool = False


class MyTeamResponse(BaseModel):
    team_name: str
    record: str
    rank: int
    matchup: MatchupHero | None
    categories: list[CategoryLine]
