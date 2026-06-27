"""Contract models for the Start/Sit page (compare verdict + apply-to-open-slots).

Reuses PlayerRef/StatItem (common) and LineupSlot/DailyMeta (lineup) — the
/optimize response is the same lineup shape the Optimizer page already renders."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef, StatItem
from api.contracts.lineup import DailyMeta, LineupSlot

# Engine-native horizons (same selector as the Optimizer page). Validated in the
# service (an unknown value degrades to rest_of_season, never raises).
_SCOPES = ("today", "rest_of_week", "rest_of_season")


class StartSitCompareRequest(BaseModel):
    team_name: str | None = None
    scope: str = "today"  # today | rest_of_week | rest_of_season
    player_ids: list[int] = Field(default_factory=list)  # 2..6 (service clamps)


class StartSitCandidate(BaseModel):
    player: PlayerRef
    start_score: float = 0.0  # 0-100 heat (normalized across the compared set)
    rank: int = 0  # 1-based, by start_score desc
    eligible_slots: list[str] = Field(default_factory=list)  # league slots this player can fill
    projected: list[StatItem] = Field(default_factory=list)  # scope-scaled projected line
    category_impact: list[StatItem] = Field(default_factory=list)  # per-cat SGP impact (display)
    matchup: str = ""  # "vs SF" / "@ COL" (empty when not playing / unknown)
    reason: str = ""  # top driver(s) of the score
    playable: bool = True  # False = IL/DTD/off-day (cannot start the scope)


class StartSitVerdict(BaseModel):
    start_ids: list[int] = Field(default_factory=list)  # players assigned to open slots
    sit_ids: list[int] = Field(default_factory=list)  # the rest (bounded by open-slot count)
    reasoning: str = ""


class StartSitCompareResponse(BaseModel):
    scope: str
    candidates: list[StartSitCandidate] = Field(default_factory=list)
    verdict: StartSitVerdict = Field(default_factory=StartSitVerdict)
    open_slots: dict[str, int] = Field(default_factory=dict)  # open lineup slots by position
    confidence: float = 0.0  # 0-1 (gap between the top two start_scores)
    confidence_label: str = "Toss-up"  # "Clear" | "Lean" | "Toss-up"


class StartSitOptimizeRequest(BaseModel):
    team_name: str | None = None
    scope: str = "today"
    player_ids: list[int] = Field(default_factory=list)


class StartSitOptimizeResponse(BaseModel):
    scope: str
    slots: list[LineupSlot] = Field(default_factory=list)  # the filled STARTERS
    bench: list[LineupSlot] = Field(default_factory=list)
    summary: str = ""
    daily: DailyMeta | None = None  # day-level context (today scope only)
