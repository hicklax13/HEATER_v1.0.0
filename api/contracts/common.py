"""Contract models shared across pages."""

from __future__ import annotations

from pydantic import BaseModel


class PlayerRef(BaseModel):
    id: int
    mlb_id: int | None = None
    name: str
    positions: str
    team_abbr: str | None = None
    team_id: int | None = None
    yahoo_player_key: str | None = None


class StatItem(BaseModel):
    """A single labeled stat for display, e.g. {label: "HR", value: "18"}.

    Shared across pages (free agents, team movers/lever, …). The frontend renders
    label + value; counting stats are integer strings, rate stats pre-formatted.
    """

    label: str
    value: str


class Record(BaseModel):
    """Structured win-loss-tie record (additive complement to the display string)."""

    wins: int = 0
    losses: int = 0
    ties: int = 0
