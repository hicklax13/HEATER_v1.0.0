"""Live draft assistant with Yahoo real-time sync polling."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class DraftPick:
    pick_number: int
    team_index: int
    player_id: int
    player_name: str
    positions: str = ""


@dataclass
class SyncResult:
    new_picks: list[DraftPick] = field(default_factory=list)
    is_user_turn: bool = False
    draft_complete: bool = False
    error: str | None = None
    total_picks: int = 0


class LiveDraftSyncer:
    """Manages live draft synchronization with Yahoo."""

    def __init__(
        self,
        yahoo_client=None,
        player_pool: pd.DataFrame | None = None,
        user_team_key: str = "",
        num_teams: int = 12,
        num_rounds: int = 23,
        user_team_index: int = 0,
    ):
        self._yahoo = yahoo_client
        self._pool = player_pool if player_pool is not None else pd.DataFrame()
        self._user_team_key = user_team_key
        self._num_teams = num_teams
        self._num_rounds = num_rounds
        self._user_team_index = user_team_index
        self._known_picks: list[DraftPick] = []
        self._consecutive_errors = 0
        self._last_poll_time: float = 0

    @property
    def known_pick_count(self) -> int:
        return len(self._known_picks)

    @property
    def is_syncing_disabled(self) -> bool:
        return self._consecutive_errors >= 3

    def resolve_player(self, yahoo_name: str, yahoo_team: str = "") -> int | None:
        """Resolve Yahoo player name to local player_id."""
        if self._pool.empty:
            return None
        name_col = "name" if "name" in self._pool.columns else "player_name"
        match = self._pool[self._pool[name_col].str.lower() == yahoo_name.lower()]
        if not match.empty:
            pid = match.iloc[0].get("player_id")
            return int(pid) if pid is not None and pid != 0 else None
        # Fuzzy fallback: check contains (regex=False for names like "J.D. Martinez")
        match = self._pool[self._pool[name_col].str.lower().str.contains(yahoo_name.lower(), na=False, regex=False)]
        if not match.empty:
            pid = match.iloc[0].get("player_id")
            return int(pid) if pid is not None and pid != 0 else None
        return None

    def add_known_pick(self, pick: DraftPick) -> None:
        """Manually add a pick to known picks."""
        self._known_picks.append(pick)

    def detect_new_picks(self, yahoo_picks: list[dict]) -> list[DraftPick]:
        """Detect picks not yet in known_picks."""
        known_count = len(self._known_picks)
        new_picks = []
        for i, yp in enumerate(yahoo_picks):
            if i >= known_count:
                pick = DraftPick(
                    pick_number=i + 1,
                    team_index=int(yp.get("team_index", 0)),
                    player_id=int(yp.get("player_id", 0)),
                    player_name=str(yp.get("player_name", "")),
                    positions=str(yp.get("positions", "")),
                )
                new_picks.append(pick)
                self._known_picks.append(pick)
        return new_picks

    def is_user_turn(self, total_picks: int) -> bool:
        """Check if it's the user's turn based on snake draft order."""
        if total_picks >= self._num_teams * self._num_rounds:
            return False
        rnd = total_picks // self._num_teams
        pos = total_picks % self._num_teams
        if rnd % 2 == 0:
            team_idx = pos
        else:
            team_idx = self._num_teams - 1 - pos
        return team_idx == self._user_team_index

    def poll_and_sync(self, yahoo_picks: list[dict] | None = None) -> SyncResult:
        """Poll for new picks and sync state.

        If yahoo_picks is None, would call Yahoo API (requires yahoo_client).
        """
        result = SyncResult()
        try:
            picks = yahoo_picks or []
            result.total_picks = len(picks)
            new = self.detect_new_picks(picks)
            result.new_picks = new
            result.is_user_turn = self.is_user_turn(len(picks))
            result.draft_complete = len(picks) >= self._num_teams * self._num_rounds
            self._consecutive_errors = 0
        except Exception as e:
            self._consecutive_errors += 1
            result.error = str(e)
        return result

    def get_last_n_picks(self, n: int = 3) -> list[DraftPick]:
        """Return the last N picks."""
        return self._known_picks[-n:]

    def full_reconciliation(self, yahoo_picks: list[dict]) -> SyncResult:
        """Full rebuild from all Yahoo picks."""
        self._known_picks = []
        return self.poll_and_sync(yahoo_picks)
