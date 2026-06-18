"""Contract models for the Draft Simulator 'recommend' endpoint.

Stateless: the client holds the draft (config + pick_log) and sends it on each
call; the server rebuilds DraftState by replaying picks (exactly like
DraftState.load) and answers as a pure function — no server-side draft storage."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class DraftConfig(BaseModel):
    num_teams: int = 12
    num_rounds: int = 23
    user_team_index: int = 0  # 0-based seat of the user
    roster_config: dict[str, int] | None = None  # None → engine default


class DraftPick(BaseModel):
    pick: int  # 0-indexed overall pick number
    team_index: int
    player_id: int
    player_name: str
    positions: str  # comma-separated, e.g. "SS,3B"


class DraftRecommendRequest(BaseModel):
    config: DraftConfig = DraftConfig()
    pick_log: list[DraftPick] = []  # picks so far, in order
    top_n: int = 8  # capped server-side
    n_simulations: int = 300  # MC sims; capped server-side


class DraftClock(BaseModel):
    current_pick: int
    round: int
    pick_in_round: int
    picking_team_index: int
    is_user_turn: bool


class DraftRecommendation(BaseModel):
    player: PlayerRef
    rank: int
    score: float  # composite value (0-100)
    projected_sgp: float  # MC mean roster SGP (best-effort)
    confidence: str | None = None  # engine confidence_level: HIGH/MEDIUM/LOW
    tag: str | None = None  # e.g. buy/fair/avoid
    reason: str = ""


class DraftRecommendResponse(BaseModel):
    clock: DraftClock
    recommendations: list[DraftRecommendation]
    summary: str = ""
