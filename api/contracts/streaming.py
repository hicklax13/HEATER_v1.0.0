"""Contract models for the Pitcher Streaming page."""

from __future__ import annotations

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
    score: float = 0.0
    status: str = ""  # raw engine value: "PROBABLE"/"LOCKED"/"FINAL"
    confidence: str = ""  # raw engine value: "HIGH"/"MEDIUM"/"LOW"
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
    ip_pace: float = 0.0  # DEFERRED — weekly IP pace plumbing is a follow-up
    ip_target: float = 54.0
    cats_in_play: list[str] = Field(default_factory=list)


class StreamingResponse(BaseModel):
    date: str
    candidates: list[StreamCandidate] = Field(default_factory=list)
    top_pick: StreamCandidate | None = None
    budget: BudgetStrip = Field(default_factory=BudgetStrip)
