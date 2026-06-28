"""Contract models for the Pitcher Streaming page."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef


class StreamComponents(BaseModel):
    """The 6 stream-score factors, each in [-1, +1]."""

    matchup: float = 0.0
    env: float = 0.0
    form: float = 0.0
    lineup: float = 0.0
    sgp: float = 0.0
    winprob: float = 0.0


class StreamCandidate(BaseModel):
    player: PlayerRef
    team: str = ""
    opponent: str = ""
    is_home: bool = False
    # ISO-8601 UTC game start (statsapi game_datetime); "" if unknown. Frontend formats to local time.
    game_time: str = ""
    score: float = 0.0
    status: Literal["", "PROBABLE", "LOCKED", "FINAL", "OPEN"] = ""
    confidence: Literal["", "HIGH", "MEDIUM", "LOW"] = ""
    actionable: bool = True
    num_starts: int = 1
    net_sgp: float = 0.0
    opp_wrc_plus: float = 0.0
    opp_k_pct: float = 0.0
    park: float = 1.0
    expected_ip: float = 0.0
    expected_k: float = 0.0
    expected_er: float = 0.0
    win_pct: float = 0.0
    own_pct: float = 0.0
    risk_flags: list[str] = Field(default_factory=list)
    components: StreamComponents = Field(default_factory=StreamComponents)
    expected_line: str = ""
    rank: int = 0
    reason: str = ""


class BudgetStrip(BaseModel):
    adds_left: int = 0
    adds_total: int = 10
    ip_pace: float | None = None  # projected weekly IP from the roster's pitchers (ip_tracker); None if no roster
    ip_target: float = 54.0
    cats_in_play: list[str] = Field(default_factory=list)


class FactorDetail(BaseModel):
    key: str  # matchup/sgp/form/lineup/env/winprob
    label: str
    value: float = 0.0  # the component, [-1, +1]
    weight: float = 0.0  # the stream_score weight from CONSTANTS_REGISTRY
    detail: str = ""


class ProbableStarter(BaseModel):
    player: PlayerRef
    team: str = ""
    opponent: str = ""
    is_home: bool = False
    pos_group: str = "SP"  # SP/SP-RP/RP — engine probables are SPs
    start_likelihood: str = ""  # confirmed/likely/projected (from confidence proximity)


class PitcherScorecard(StreamCandidate):
    factors: list[FactorDetail] = Field(default_factory=list)


class StreamAnalyzeRequest(BaseModel):
    pitcher_id: int
    date: str = ""  # YYYY-MM-DD; "" → today


class StreamAnalyzeResponse(BaseModel):
    found: bool = False
    scorecard: PitcherScorecard | None = None


class StreamingResponse(BaseModel):
    date: str
    candidates: list[StreamCandidate] = Field(default_factory=list)
    top_pick: StreamCandidate | None = None
    budget: BudgetStrip = Field(default_factory=BudgetStrip)
    probables: list[ProbableStarter] = Field(default_factory=list)
    # Resolved this-week category urgency (CAT -> 0-1), from the live matchup
    # (compute_urgency_weights). Display + Bubba context only; the engine already
    # applied these to the scores. {} when no live matchup.
    urgency: dict[str, float] = Field(default_factory=dict)
