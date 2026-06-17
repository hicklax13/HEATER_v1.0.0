"""Contract models for the Trade Analyzer page."""

from __future__ import annotations

from pydantic import BaseModel

from api.contracts.common import PlayerRef


class TradeEvaluateRequest(BaseModel):
    team_name: str
    giving_ids: list[int]
    receiving_ids: list[int]
    enable_mc: bool = False  # Monte Carlo is the slow opt-in path (async in B3)


class GradeRange(BaseModel):
    grade: str
    low: float
    center: float
    high: float


class CategoryImpact(BaseModel):
    cat: str
    delta: float  # SGP delta for this category from the trade


class TradeEvaluationResponse(BaseModel):
    grade: str = ""  # Phase-1 weighted-SGP grade (the authority)
    verdict: str = ""  # e.g. "Accept", "Reject", "Fair"
    confidence_pct: float = 0.0
    surplus_sgp: float = 0.0  # headline net SGP surplus
    grade_range: GradeRange | None = None
    giving: list[PlayerRef] = []
    receiving: list[PlayerRef] = []
    category_impacts: list[CategoryImpact] = []
    delta_playoff_prob: float | None = None
    delta_champ_prob: float | None = None
    mc_enabled: bool = False
    summary: str = ""
    warnings: list[str] = []  # reshuffle / IP-floor / ghost-team flags
