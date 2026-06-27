"""Contract models for the Lineup Optimizer page."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef, StatItem


class LineupOptimizeRequest(BaseModel):
    team_name: str
    date: str | None = None
    scope: str = "rest_of_season"
    mode: str = "standard"  # "standard" (ROS/weekly LP) | "daily" (today's DCV start/sit)


class LineupSlot(BaseModel):
    slot: str  # the RECOMMENDED slot
    player: PlayerRef
    action: str  # "START" | "SIT"
    projected: float = 0.0  # per-player daily value is daily-mode (DCV) → slice 2
    forced_start: bool = False  # started despite a poor matchup / low value (roster-forced)
    reason: str | None = None  # daily exclusion cause: "LOCKED" | "IL" | "OFF_DAY"
    status: str = "start"  # "start" | "bench"
    value: float = 0.0  # DAILY mode: 0-100 heat value (normalized DCV; best play today = 100)
    matchup: str = ""  # DAILY mode: "vs SF" / "@ COL" (empty if not playing / unknown)
    current_slot: str = ""  # the player's CURRENT Yahoo slot (lets the frontend diff swaps)
    value_breakdown: list[StatItem] = Field(
        default_factory=list
    )  # per-category DCV/value contributions (WS5 inline explainer; additive, daily/standard)


class CatImpact(BaseModel):
    """A projected category total for the optimal lineup (display string + trend)."""

    key: str
    proj: str
    trend: str = "flat"  # "up" | "down" | "flat" (flat for now — no baseline to diff)


class IpPace(BaseModel):
    """Weekly innings-pitched pacing for the daily view (from ip_tracker)."""

    projected: float
    target: float
    pace_pct: int
    status: str = ""  # "safe" | "warning" | "danger"
    message: str = ""


class Swap(BaseModel):
    """A recommended move: start this currently-benched player in `slot`."""

    player: PlayerRef
    slot: str
    value: float = 0.0


class DailyMeta(BaseModel):
    """Day-level context for the daily optimizer (populated only when mode='daily')."""

    urgency: dict[str, float] = Field(default_factory=dict)  # per-category 0-1 urgency
    rate_modes: dict[str, str] = Field(default_factory=dict)  # ERA/WHIP → protect|compete|abandon
    winning: list[str] = Field(default_factory=list)
    losing: list[str] = Field(default_factory=list)
    tied: list[str] = Field(default_factory=list)
    ip_pace: IpPace | None = None
    recommendations: list[str] = Field(default_factory=list)
    swaps: list[Swap] = Field(default_factory=list)  # benched players the optimizer wants to start


class FaSuggestion(BaseModel):
    """A 'drop X for available Y' free-agent upgrade, alongside the LP lineup.

    Composed from recommend_fa_moves(ctx) over the optimize context. The LP still
    optimizes the CURRENT roster; these are the available-pickup layer.
    """

    add: PlayerRef
    drop: PlayerRef
    net_sgp_delta: float = 0.0  # net SGP gain of the swap (add value − drop cost)
    category_impact: list[StatItem] = Field(default_factory=list)  # per-cat SGP deltas, formatted
    reasoning: str = ""
    urgency_categories: list[str] = Field(default_factory=list)  # losing/tied cats this swap helps


class LineupOptimizeResponse(BaseModel):
    team_name: str
    date: str
    slots: list[LineupSlot]  # the optimal STARTERS
    summary: str = ""
    bench: list[LineupSlot] = Field(
        default_factory=list
    )  # ALL non-starter roster players (incl IL/BN — frontend filters by slot)
    optimal: bool = False  # is the user's current Yahoo lineup already the optimal one?
    impact: list[CatImpact] = Field(default_factory=list)  # projected category totals for the lineup
    mode: str = "standard"  # echoes the request mode
    daily: DailyMeta | None = None  # day-level context, daily mode only
    fa_suggestions: list[FaSuggestion] = Field(
        default_factory=list
    )  # available "drop X for Y" pickups (composed from recommend_fa_moves)
