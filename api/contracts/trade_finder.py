"""Contract models for the Trade Finder page."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from api.contracts.common import PlayerRef
from api.contracts.trade import CategoryImpact


class TradeSuggestion(BaseModel):
    partner_team: str
    partner_record: str | None = None  # "11-1-0 · 1st" — from load_league_records
    grade: str = ""  # engine grade_trade(user_sgp_gain) — already computed by the finder
    giving: list[PlayerRef] = []  # players YOU give
    receiving: list[PlayerRef] = []  # players you receive
    net_sgp: float = 0.0
    category_impacts: list[CategoryImpact] = []  # per-cat SGP delta (service-side diff)
    rationale: str = ""


class TradeFinderResponse(BaseModel):
    team_name: str
    suggestions: list[TradeSuggestion] = []
    # Observability: WHY the suggestion list is what it is. The original bug was a
    # SILENT empty (a missed roster key), so a recurrence must be diagnosable from
    # the response. Additive/optional (default None) → backward-compatible; the
    # frontend may ignore it.
    reason: Literal["ok", "team_not_resolved", "no_pool", "no_league_data", "no_totals", "error"] | None = None
