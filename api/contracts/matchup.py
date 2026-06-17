"""Contract models for the Matchup Planner page."""

from __future__ import annotations

from pydantic import BaseModel


class MatchupCategory(BaseModel):
    cat: str
    you: float
    opp: float
    win_prob: float
    inverse: bool = False


class MatchupResponse(BaseModel):
    team_name: str
    opponent: str = ""
    week: int = 0
    projected_cat_wins: float = 0.0
    win_prob: float = 0.0  # P(win the weekly matchup overall)
    categories: list[MatchupCategory] = []
