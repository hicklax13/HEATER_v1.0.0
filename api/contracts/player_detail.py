"""Contract models for the canonical player card (PlayerDialog).

1:1 with the frontend ``PlayerDetail`` TS interface (web/src/lib/player-detail.ts);
the frontend adapter camelCases the snake_case fields. Slice 1 fills the DB-backed
core (identity, season line, prior years, ownership, ranks, rostered-by, ROS
projections, history); the live-source fields (l7/l14/l30 windows, game logs,
near-term projection horizons) are emitted as "—"/[] and filled by a later slice.
"""

from __future__ import annotations

from pydantic import BaseModel


class LabelValue(BaseModel):
    label: str
    value: str


class GameRow(BaseModel):
    date: str
    opp: str
    result: str
    upcoming: bool = False
    line: list[str] = []


class StatRow(BaseModel):
    cat: str
    season: str = "—"
    l30: str = "—"
    l14: str = "—"
    l7: str = "—"
    avg: str = "—"
    std: str = "—"


class PriorRow(BaseModel):
    cat: str
    y2025: str = "—"
    y2024: str = "—"


class PriorBlock(BaseModel):
    y2025_rank: int = 0
    y2024_rank: int = 0
    rows: list[PriorRow] = []


class ProjRow(BaseModel):
    cat: str
    today: str = "—"
    n7: str = "—"
    n14: str = "—"
    n30: str = "—"
    ros: str = "—"
    avg: str = "—"
    std: str = "—"


class HistoryEvent(BaseModel):
    kind: str = ""  # "drafted" | "traded" | "added" | "dropped"
    date: str = ""
    text: str = ""
    member: str = ""


class PlayerDetailResponse(BaseModel):
    mlb_id: int
    team_id: int | None = None
    name: str
    pos: str = ""
    bats: str = ""
    jersey: str = ""
    team_name: str = ""
    is_pitcher: bool = False
    own_pct: float = 0.0
    own_delta: float = 0.0
    rostered_by: str = ""
    headline: list[LabelValue] = []
    ranks: list[LabelValue] = []
    game_columns: list[str] = []
    game_log: list[GameRow] = []
    stats: list[StatRow] = []
    prior: PriorBlock = PriorBlock()
    projections: list[ProjRow] = []
    history: list[HistoryEvent] = []
