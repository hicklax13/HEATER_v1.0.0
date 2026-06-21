"""Probable-pitcher / matchup schedule contracts (Probable Pitcher feature P1).

The 7-day grid: TEAM rows x day columns. Each cell is that team's probable SP for
the day, league-tagged (yours / taken / available) with a matchup difficulty band
and a two-start flag. Reuses the shared PlayerRef.
"""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class ProbableCell(BaseModel):
    pitcher: PlayerRef | None = None
    opponent: str = ""  # e.g. "SF" (the opposing team abbr)
    is_home: bool = False
    # 0-100 streamability score (HIGHER = easier matchup to stream against).
    difficulty: float = 0.0
    band: str = "medium"  # easy | medium | tough (derived from difficulty)
    two_start: bool = False
    availability: str = "available"  # yours | taken | available
    rostered_by: str | None = None
    status: str = ""
    confidence: str = ""


class ProbableTeamRow(BaseModel):
    team: str  # team abbreviation
    cells: list[ProbableCell | None] = []  # aligned to ProbableGridResponse.days


class ProbableGridResponse(BaseModel):
    days: list[str] = []  # the date columns, YYYY-MM-DD, ascending
    teams: list[ProbableTeamRow] = []
