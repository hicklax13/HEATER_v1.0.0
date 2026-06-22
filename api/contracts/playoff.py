"""Contract models for the playoff-odds endpoint (Standings odds panel + Team page).

Per-team PLAYOFF odds + projected standings from one Monte Carlo run
(simulate_season_enhanced). Per-team CHAMPIONSHIP odds are intentionally NOT here —
no engine computes them league-wide yet (documented follow-up)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from api.contracts.common import Record


class PlayoffTeam(BaseModel):
    team: str
    playoff_odds: float = 0.0  # 0-100 (P of finishing top-`playoff_spots`)
    projected_wins: float = 0.0  # mean projected final wins
    projected_record: str = ""  # "W-L-T" display string (kept for backward-compat)
    projected_record_wlt: Record | None = None  # structured W-L-T (additive)
    current_wins: int = 0
    rank: int = 0  # projected finish, 1 = best playoff odds
    in_cut: bool = False  # projected inside the playoff cut
    is_user: bool = False


class PlayoffOddsResponse(BaseModel):
    team_name: str
    playoff_spots: int = 4
    you: PlayoffTeam | None = None
    league: list[PlayoffTeam] = Field(default_factory=list)  # sorted by playoff_odds desc
    n_sims: int = 0
