"""Draft state management: tracks picks, rosters, and category totals."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import pandas as pd

BACKUP_DIR = Path(__file__).parent.parent / "data" / "backups"


@dataclass
class RosterSlot:
    position: str
    player_id: Optional[int] = None
    player_name: Optional[str] = None


@dataclass
class TeamRoster:
    team_name: str
    team_index: int  # 0-11
    slots: list = field(default_factory=list)
    picks: list = field(default_factory=list)  # list of (pick_number, player_id, player_name)

    def filled_positions(self) -> set:
        return {s.position for s in self.slots if s.player_id is not None}

    def needs_position(self, pos: str) -> bool:
        """Check if this team still has an open slot for this position."""
        for s in self.slots:
            if s.position == pos and s.player_id is None:
                return True
        return False

    def open_positions(self) -> list:
        return [s.position for s in self.slots if s.player_id is None]

    def add_player(self, player_id: int, player_name: str, positions: str, pick_number: int):
        """Assign a player to the best available roster slot."""
        pos_list = [p.strip() for p in positions.split(",")]
        assigned = False

        # Try to fill a specific position slot first
        for pos in pos_list:
            for s in self.slots:
                if s.position == pos and s.player_id is None:
                    s.player_id = player_id
                    s.player_name = player_name
                    assigned = True
                    break
            if assigned:
                break

        # Try Util (for hitters) or P (for pitchers)
        if not assigned:
            flex_positions = ["Util", "P", "BN"]
            for flex in flex_positions:
                for s in self.slots:
                    if s.position == flex and s.player_id is None:
                        s.player_id = player_id
                        s.player_name = player_name
                        assigned = True
                        break
                if assigned:
                    break

        # Last resort: bench
        if not assigned:
            for s in self.slots:
                if s.player_id is None:
                    s.player_id = player_id
                    s.player_name = player_name
                    assigned = True
                    break

        self.picks.append((pick_number, player_id, player_name))
        return assigned


class DraftState:
    """Manages the entire draft state."""

    def __init__(self, num_teams: int = 12, num_rounds: int = 23,
                 user_team_index: int = 0, roster_config: dict = None):
        self.num_teams = num_teams
        self.num_rounds = num_rounds
        self.user_team_index = user_team_index
        self.total_picks = num_teams * num_rounds
        self.current_pick = 0  # 0-indexed overall pick number
        self.pick_log = []  # list of dicts: {pick, team, player_id, player_name}
        self.drafted_player_ids = set()

        # Default roster config
        if roster_config is None:
            roster_config = {
                "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1,
                "OF": 3, "Util": 2,
                "SP": 2, "RP": 2, "P": 4,
                "BN": 5,
            }

        # Initialize all team rosters
        self.teams = []
        for i in range(num_teams):
            slots = []
            for pos, count in roster_config.items():
                for _ in range(count):
                    slots.append(RosterSlot(position=pos))
            self.teams.append(TeamRoster(
                team_name=f"Team {i+1}" if i != user_team_index else "My Team",
                team_index=i,
                slots=slots,
            ))

    @property
    def user_team(self) -> TeamRoster:
        return self.teams[self.user_team_index]

    @property
    def current_round(self) -> int:
        return (self.current_pick // self.num_teams) + 1

    @property
    def pick_in_round(self) -> int:
        return (self.current_pick % self.num_teams) + 1

    @property
    def is_user_turn(self) -> bool:
        return self.picking_team_index() == self.user_team_index

    def picking_team_index(self, pick: int = None) -> int:
        """Determine which team picks at a given overall pick number (snake draft)."""
        if pick is None:
            pick = self.current_pick
        round_num = pick // self.num_teams  # 0-indexed round
        pos_in_round = pick % self.num_teams
        if round_num % 2 == 0:
            return pos_in_round  # forward
        else:
            return self.num_teams - 1 - pos_in_round  # reverse (snake)

    def next_user_pick(self) -> Optional[int]:
        """Return the overall pick number of the user's next pick, or None if draft is over."""
        for pick in range(self.current_pick, self.total_picks):
            if self.picking_team_index(pick) == self.user_team_index:
                return pick
        return None

    def picks_until_user_turn(self) -> int:
        """Number of picks until the user is on the clock."""
        nxt = self.next_user_pick()
        if nxt is None:
            return 999
        return nxt - self.current_pick

    def make_pick(self, player_id: int, player_name: str, positions: str,
                  team_index: int = None) -> dict:
        """Record a draft pick.

        Args:
            player_id: Unique player identifier.
            player_name: Display name.
            positions: Comma-separated position string.
            team_index: Override which team gets the pick (default: snake order).
                        Used by 'mine' command to force-assign to user's team.
        """
        team_idx = team_index if team_index is not None else self.picking_team_index()
        team = self.teams[team_idx]
        team.add_player(player_id, player_name, positions, self.current_pick)
        self.drafted_player_ids.add(player_id)

        entry = {
            "pick": self.current_pick,
            "round": self.current_round,
            "pick_in_round": self.pick_in_round,
            "team_index": team_idx,
            "team_name": team.team_name,
            "player_id": player_id,
            "player_name": player_name,
            "positions": positions,
        }
        self.pick_log.append(entry)
        self.current_pick += 1
        return entry

    def undo_last_pick(self):
        """Remove the last pick."""
        if not self.pick_log:
            return
        entry = self.pick_log.pop()
        self.current_pick -= 1
        self.drafted_player_ids.discard(entry["player_id"])

        # Remove from team roster
        team = self.teams[entry["team_index"]]
        team.picks = [(p, pid, pn) for p, pid, pn in team.picks if pid != entry["player_id"]]
        for s in team.slots:
            if s.player_id == entry["player_id"]:
                s.player_id = None
                s.player_name = None
                break

    def get_user_roster_totals(self, player_pool: pd.DataFrame) -> dict:
        """Compute category totals for the user's current roster."""
        totals = {
            "R": 0, "HR": 0, "RBI": 0, "SB": 0,
            "W": 0, "SV": 0, "K": 0,
            "ab": 0, "h": 0,
            "ip": 0, "er": 0, "bb_allowed": 0, "h_allowed": 0,
        }

        for _, pid, _ in self.user_team.picks:
            player = player_pool[player_pool["player_id"] == pid]
            if player.empty:
                continue
            p = player.iloc[0]

            totals["R"] += int(p.get("r", 0) or 0)
            totals["HR"] += int(p.get("hr", 0) or 0)
            totals["RBI"] += int(p.get("rbi", 0) or 0)
            totals["SB"] += int(p.get("sb", 0) or 0)
            totals["W"] += int(p.get("w", 0) or 0)
            totals["SV"] += int(p.get("sv", 0) or 0)
            totals["K"] += int(p.get("k", 0) or 0)
            totals["ab"] += int(p.get("ab", 0) or 0)
            totals["h"] += int(p.get("h", 0) or 0)
            totals["ip"] += float(p.get("ip", 0) or 0)
            totals["er"] += int(p.get("er", 0) or 0)
            totals["bb_allowed"] += int(p.get("bb_allowed", 0) or 0)
            totals["h_allowed"] += int(p.get("h_allowed", 0) or 0)

        # Compute rate stats
        if totals["ab"] > 0:
            totals["AVG"] = totals["h"] / totals["ab"]
        else:
            totals["AVG"] = 0
        if totals["ip"] > 0:
            totals["ERA"] = totals["er"] * 9 / totals["ip"]
            totals["WHIP"] = (totals["bb_allowed"] + totals["h_allowed"]) / totals["ip"]
        else:
            totals["ERA"] = 0
            totals["WHIP"] = 0

        return totals

    def get_team_drafted_positions(self, team_index: int) -> list:
        """Get list of primary positions drafted by a specific team."""
        positions = []
        for entry in self.pick_log:
            if entry["team_index"] == team_index:
                for p in entry["positions"].split(","):
                    p = p.strip()
                    if p:
                        positions.append(p)
        return positions

    def get_all_teams_positions(self) -> dict:
        """Get drafted positions for every team. Returns {team_index: [positions]}."""
        result = {}
        for i in range(self.num_teams):
            result[i] = self.get_team_drafted_positions(i)
        return result

    def get_all_team_roster_totals(self, player_pool: pd.DataFrame) -> list:
        """Get projected stat totals for all teams (for standings estimation).

        Returns list of dicts (one per team) with category totals.
        """
        all_totals = []
        for team in self.teams:
            t = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "W": 0, "SV": 0, "K": 0,
                 "ab": 0, "h": 0, "ip": 0, "er": 0, "bb_allowed": 0, "h_allowed": 0}
            for _, pid, _ in team.picks:
                player = player_pool[player_pool["player_id"] == pid]
                if player.empty:
                    continue
                p = player.iloc[0]
                t["R"] += int(p.get("r", 0) or 0)
                t["HR"] += int(p.get("hr", 0) or 0)
                t["RBI"] += int(p.get("rbi", 0) or 0)
                t["SB"] += int(p.get("sb", 0) or 0)
                t["W"] += int(p.get("w", 0) or 0)
                t["SV"] += int(p.get("sv", 0) or 0)
                t["K"] += int(p.get("k", 0) or 0)
                t["ab"] += int(p.get("ab", 0) or 0)
                t["h"] += int(p.get("h", 0) or 0)
                t["ip"] += float(p.get("ip", 0) or 0)
                t["er"] += int(p.get("er", 0) or 0)
                t["bb_allowed"] += int(p.get("bb_allowed", 0) or 0)
                t["h_allowed"] += int(p.get("h_allowed", 0) or 0)
            if t["ab"] > 0:
                t["AVG"] = t["h"] / t["ab"]
            else:
                t["AVG"] = 0
            if t["ip"] > 0:
                t["ERA"] = t["er"] * 9 / t["ip"]
                t["WHIP"] = (t["bb_allowed"] + t["h_allowed"]) / t["ip"]
            else:
                t["ERA"] = 0
                t["WHIP"] = 0
            all_totals.append(t)
        return all_totals

    def positions_still_needed_league(self, config_roster_slots: dict) -> dict:
        """For each position, count how many teams still need it.

        Returns {position: num_teams_needing_it}.
        """
        all_positions = self.get_all_teams_positions()
        needed = {}
        for pos in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]:
            slots = config_roster_slots.get(pos, 0)
            count = 0
            for team_idx, team_pos in all_positions.items():
                filled = sum(1 for p in team_pos if p == pos)
                if filled < slots:
                    count += 1
            needed[pos] = count
        return needed

    def available_players(self, player_pool: pd.DataFrame) -> pd.DataFrame:
        """Filter player pool to only available (undrafted) players."""
        return player_pool[~player_pool["player_id"].isin(self.drafted_player_ids)].copy()

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, filename: str = "draft_state.json"):
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        filepath = BACKUP_DIR / filename
        data = {
            "num_teams": self.num_teams,
            "num_rounds": self.num_rounds,
            "user_team_index": self.user_team_index,
            "current_pick": self.current_pick,
            "pick_log": self.pick_log,
            "drafted_player_ids": list(self.drafted_player_ids),
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return str(filepath)

    @classmethod
    def load(cls, filename: str = "draft_state.json",
             roster_config: dict = None) -> "DraftState":
        filepath = BACKUP_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"No saved draft state at {filepath}")

        with open(filepath) as f:
            data = json.load(f)

        state = cls(
            num_teams=data["num_teams"],
            num_rounds=data["num_rounds"],
            user_team_index=data["user_team_index"],
            roster_config=roster_config,
        )

        # Replay all picks to rebuild state
        state.current_pick = 0
        state.drafted_player_ids = set()
        state.pick_log = []

        for entry in data["pick_log"]:
            state.make_pick(
                entry["player_id"],
                entry["player_name"],
                entry["positions"],
            )

        return state

    def has_saved_state(self) -> bool:
        return (BACKUP_DIR / "draft_state.json").exists()
