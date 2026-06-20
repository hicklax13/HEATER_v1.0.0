"""Contract models for the My Team page. This is the canonical API contract;
the frontend's web/lib/data/types.ts is generated from the OpenAPI schema
these produce (see scripts/export_openapi.py)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef


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


class Mover(BaseModel):
    """A hot/cold player on the user's roster (trending vs projection)."""

    player: PlayerRef
    stats: list[str] = Field(default_factory=list)  # up to 2 display stats, e.g. ["18 HR", ".322 AVG"]
    trend: str = "flat"  # "up" | "down"
    tag: str = ""  # "hot" | "cold"
    context: str = ""  # short note, e.g. "Trending hot vs projection"
    rostered_by_you: bool = True


class StatusChip(BaseModel):
    """A small dashboard status badge (IL count, news count, …)."""

    label: str  # "IL", "News"
    value: int  # count
    status: str = "info"  # "ok" | "warn" | "info"


class MyTeamResponse(BaseModel):
    team_name: str
    record: str
    rank: int
    matchup: MatchupHero | None
    categories: list[CategoryLine]
    # ── Team-dashboard slice 1 (all defaulted → backward-compatible) ──
    eyebrow: str = ""
    subline: str = ""
    freshness_minutes: float | None = None
    playoff_cut_rank: int = 4
    status_chips: list[StatusChip] = Field(default_factory=list)
    movers: list[Mover] = Field(default_factory=list)
    movers_scope: str = "mine"
