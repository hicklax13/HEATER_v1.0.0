"""Contract models for the Punt Analyzer page."""

from __future__ import annotations

from pydantic import BaseModel


class PuntCategory(BaseModel):
    cat: str
    current_rank: int
    gainable: bool
    recommendation: str = ""  # e.g. "Punt", "Contend", "Hold"


class PuntResponse(BaseModel):
    team_name: str
    punt_candidates: list[str] = []  # category names recommended to punt
    categories: list[PuntCategory] = []
