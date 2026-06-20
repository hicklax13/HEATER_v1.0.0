"""Contract models for the Matchup Planner page."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import PlayerRef


class MatchupCategory(BaseModel):
    cat: str
    you: float
    opp: float
    win_prob: float
    inverse: bool = False
    win: str = ""  # "you" | "opp" | "" (tie)


class MatchPlayer(BaseModel):
    player: PlayerRef
    pos: str = ""
    status: str = ""  # basic game status (NOT live play-by-play — see live-stats follow-up)
    state: str = "none"  # sched/live/final/none
    stats: list[str] = Field(default_factory=list)  # 7-stat projected line
    badge: str | None = None  # IL / DTD


class RosterRow(BaseModel):
    slot: str = ""
    you: MatchPlayer | None = None
    opp: MatchPlayer | None = None


class MatchupResponse(BaseModel):
    team_name: str
    opponent: str = ""
    week: int = 0
    projected_cat_wins: float = 0.0
    win_prob: float = 0.0  # P(win the weekly matchup overall)
    categories: list[MatchupCategory] = []
    date_tabs: list[str] = Field(default_factory=list)
    hitter_columns: list[str] = Field(default_factory=list)
    pitcher_columns: list[str] = Field(default_factory=list)
    hitters: list[RosterRow] = Field(default_factory=list)
    pitchers: list[RosterRow] = Field(default_factory=list)
