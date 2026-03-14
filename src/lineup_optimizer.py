"""Lineup optimizer using PuLP linear programming.

Maximizes marginal SGP across all roto categories subject to roster slot
constraints. Supports category targeting based on standings gaps and
two-start pitcher identification for weekly optimization.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

try:
    from pulp import (
        PULP_CBC_CMD,
        LpBinary,
        LpMaximize,
        LpProblem,
        LpStatus,
        LpVariable,
        lpSum,
    )

    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

if TYPE_CHECKING:
    from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# ── Roster Slot Definitions ──────────────────────────────────────────
# Maps slot name to (count, eligible_positions)
# Based on Yahoo 5x5 roto default: C/1B/2B/3B/SS/3OF/2Util/2SP/2RP/4P/5BN
ROSTER_SLOTS: dict[str, tuple[int, list[str]]] = {
    "C": (1, ["C"]),
    "1B": (1, ["1B"]),
    "2B": (1, ["2B"]),
    "3B": (1, ["3B"]),
    "SS": (1, ["SS"]),
    "OF": (3, ["OF", "LF", "CF", "RF"]),
    "Util": (2, ["C", "1B", "2B", "3B", "SS", "OF", "LF", "CF", "RF", "DH"]),
    "SP": (2, ["SP"]),
    "RP": (2, ["RP"]),
    "P": (4, ["SP", "RP"]),
}

# Hitting categories for SGP computation
HITTING_CATS = ["r", "hr", "rbi", "sb", "avg"]
PITCHING_CATS = ["w", "sv", "k", "era", "whip"]
ALL_CATS = HITTING_CATS + PITCHING_CATS

# Categories where lower is better
INVERSE_CATS = {"era", "whip"}


class LineupOptimizer:
    """Optimize start/sit decisions using linear programming.

    Solves a binary assignment problem: assign each player to at most one
    roster slot, maximizing projected SGP contribution subject to position
    eligibility and slot count constraints.
    """

    def __init__(
        self,
        roster: pd.DataFrame,
        config: LeagueConfig | None = None,
        roster_slots: dict | None = None,
    ):
        """Initialize optimizer.

        Args:
            roster: Player pool with columns: player_id, player_name, positions,
                    plus stat projection columns (r, hr, rbi, sb, avg, w, sv, k, era, whip)
            config: League configuration for SGP denominators
            roster_slots: Custom slot definitions (overrides default)
        """
        self.roster = roster.copy()
        self.config = config
        self.slots = roster_slots or ROSTER_SLOTS
        self._category_weights: dict[str, float] = {cat: 1.0 for cat in ALL_CATS}

    def optimize_lineup(self, category_weights: dict[str, float] | None = None) -> dict:
        """Find the optimal lineup assignment.

        Args:
            category_weights: Optional per-category weights for objective function.
                            Higher weight = prioritize that category. Default all 1.0.

        Returns:
            Dict with keys:
            - assignments: list of {slot, player_name, player_id}
            - bench: list of player_names not starting
            - projected_stats: dict of total projected stats for starters
            - status: solver status string
        """
        if not PULP_AVAILABLE:
            logger.warning("PuLP not installed — returning greedy lineup")
            return self._greedy_fallback()

        if self.roster.empty:
            return {
                "assignments": [],
                "bench": [],
                "projected_stats": {},
                "status": "empty_roster",
            }

        weights = category_weights or self._category_weights

        # Build the LP model
        prob = LpProblem("lineup_optimizer", LpMaximize)

        # Expand slots: "OF" with count=3 becomes "OF_0", "OF_1", "OF_2"
        expanded_slots = []
        for slot_name, (count, eligible) in self.slots.items():
            for i in range(count):
                expanded_slots.append((f"{slot_name}_{i}", eligible))

        players = self.roster.index.tolist()
        player_names = (
            self.roster["player_name"].tolist() if "player_name" in self.roster.columns else [f"P{i}" for i in players]
        )

        # Decision variables: x[player][slot] = 1 if player assigned to slot
        x = {}
        for p_idx in players:
            for slot_name, eligible in expanded_slots:
                x[(p_idx, slot_name)] = LpVariable(f"x_{p_idx}_{slot_name}", cat=LpBinary)

        # Objective: maximize weighted stat contributions of starters
        # For ERA/WHIP, weight by IP so high-workload pitchers' rates matter more
        objective_terms = []
        for p_idx in players:
            row = self.roster.loc[p_idx]
            player_value = 0.0
            ip = float(row.get("ip", 0) or 0)
            for cat in ALL_CATS:
                if cat not in row.index:
                    continue
                val = float(row.get(cat, 0) or 0)
                w = weights.get(cat, 1.0)
                if cat in INVERSE_CATS:
                    # For ERA/WHIP, lower is better — weight by IP so
                    # high-workload pitchers' rates count proportionally more
                    player_value -= val * ip * w
                else:
                    player_value += val * w

            for slot_name, _ in expanded_slots:
                objective_terms.append(x[(p_idx, slot_name)] * player_value)

        prob += lpSum(objective_terms)

        # Constraint 1: Each player assigned to at most one slot
        for p_idx in players:
            prob += (
                lpSum(x[(p_idx, slot_name)] for slot_name, _ in expanded_slots) <= 1,
                f"max_one_slot_{p_idx}",
            )

        # Constraint 2: Each slot filled by at most one player
        # Using <= 1 instead of == 1 so partial lineups are valid
        # when roster has fewer players than available slots
        for slot_name, eligible in expanded_slots:
            prob += (
                lpSum(x[(p_idx, slot_name)] for p_idx in players) <= 1,
                f"fill_slot_{slot_name}",
            )

        # Constraint 3: Position eligibility
        for p_idx in players:
            row = self.roster.loc[p_idx]
            player_positions = _parse_positions(row.get("positions", ""))

            for slot_name, eligible in expanded_slots:
                if not any(pos in eligible for pos in player_positions):
                    prob += (
                        x[(p_idx, slot_name)] == 0,
                        f"ineligible_{p_idx}_{slot_name}",
                    )

        # Solve
        solver = PULP_CBC_CMD(msg=0, timeLimit=10)
        prob.solve(solver)

        status = LpStatus[prob.status]
        if status != "Optimal":
            logger.warning(f"Optimizer status: {status} — using greedy fallback")
            return self._greedy_fallback()

        # Extract assignments
        assignments = []
        assigned_players = set()
        for p_idx in players:
            for slot_name, _ in expanded_slots:
                if x[(p_idx, slot_name)].varValue and x[(p_idx, slot_name)].varValue > 0.5:
                    base_slot = slot_name.rsplit("_", 1)[0]
                    assignments.append(
                        {
                            "slot": base_slot,
                            "player_name": player_names[players.index(p_idx)],
                            "player_id": int(self.roster.loc[p_idx].get("player_id", p_idx)),
                        }
                    )
                    assigned_players.add(p_idx)

        bench = [player_names[players.index(p_idx)] for p_idx in players if p_idx not in assigned_players]

        # Compute projected stats for starters
        projected = {}
        starter_rows = self.roster.loc[list(assigned_players)]
        for cat in ALL_CATS:
            if cat in starter_rows.columns:
                projected[cat] = float(starter_rows[cat].sum())

        # Fix rate stats: compute weighted averages instead of sums
        projected = _fix_rate_stats(projected, starter_rows)

        return {
            "assignments": assignments,
            "bench": bench,
            "projected_stats": projected,
            "status": status,
        }

    def category_targeting(self, standings: pd.DataFrame, team_name: str) -> dict[str, float]:
        """Compute category weights based on standings gaps.

        Categories where a small stat improvement = big standings gain
        get higher weights in the optimizer.

        Args:
            standings: DataFrame with columns: team_name, category, total, rank
            team_name: User's team name

        Returns:
            Dict mapping category name to weight multiplier
        """
        if standings.empty:
            return {cat: 1.0 for cat in ALL_CATS}

        weights = {}
        for cat in ALL_CATS:
            cat_standings = standings[standings["category"] == cat].sort_values("total")
            if cat_standings.empty:
                weights[cat] = 1.0
                continue

            user_row = cat_standings[cat_standings["team_name"] == team_name]
            if user_row.empty:
                weights[cat] = 1.0
                continue

            user_total = float(user_row.iloc[0]["total"])
            user_rank = int(user_row.iloc[0].get("rank", 6))

            # Find gap to next rank up
            if cat in INVERSE_CATS:
                # Lower is better — find team with next lower total
                better = cat_standings[cat_standings["total"] < user_total]
            else:
                # Higher is better — find team with next higher total
                better = cat_standings[cat_standings["total"] > user_total]

            if better.empty:
                weights[cat] = 0.5  # Already in 1st — low priority
                continue

            if cat in INVERSE_CATS:
                gap = user_total - float(better.iloc[-1]["total"])
            else:
                gap = float(better.iloc[0]["total"]) - user_total

            # Smaller gap = higher weight (easier to gain a standings point)
            if gap > 0:
                weights[cat] = min(3.0, 1.0 / (gap + 0.01))
            else:
                weights[cat] = 1.0

        # Normalize so average weight is 1.0
        mean_weight = np.mean(list(weights.values()))
        if mean_weight > 0:
            weights = {k: v / mean_weight for k, v in weights.items()}

        return weights

    @staticmethod
    def identify_two_start_pitchers(roster: pd.DataFrame, schedule: dict[str, list[str]] | None = None) -> list[str]:
        """Find starting pitchers with 2 starts in the scoring period.

        Args:
            roster: Roster DataFrame with player_name and positions columns
            schedule: Optional dict mapping player_name to list of game dates.
                     If not provided, returns all SP (can't determine starts).

        Returns:
            List of player names with 2+ starts
        """
        sps = roster[roster["positions"].str.contains("SP", na=False)]

        if schedule is None:
            return sps["player_name"].tolist() if "player_name" in sps.columns else []

        two_starters = []
        name_col = "player_name" if "player_name" in sps.columns else "name"
        for _, row in sps.iterrows():
            name = row.get(name_col, "")
            if name in schedule and len(schedule[name]) >= 2:
                two_starters.append(name)

        return two_starters

    def suggest_moves(
        self,
        free_agents: pd.DataFrame,
        category_weights: dict[str, float] | None = None,
        top_n: int = 5,
    ) -> list[dict]:
        """Suggest FA pickups that maximize marginal value.

        Args:
            free_agents: Available free agents DataFrame
            category_weights: Category weights from standings-based targeting
            top_n: Number of suggestions to return

        Returns:
            List of dicts: {player_name, player_id, position, marginal_value, replaces}
        """
        if free_agents.empty or self.roster.empty:
            return []

        weights = category_weights or self._category_weights
        suggestions = []

        for _, fa in free_agents.iterrows():
            fa_value = _compute_player_value(fa, weights)
            fa_positions = _parse_positions(fa.get("positions", ""))
            fa_name = fa.get("player_name", fa.get("name", "Unknown"))

            # Find the weakest starter at an eligible position
            worst_starter = None
            worst_value = float("inf")

            for _, starter in self.roster.iterrows():
                starter_positions = _parse_positions(starter.get("positions", ""))
                if any(p in fa_positions for p in starter_positions):
                    starter_value = _compute_player_value(starter, weights)
                    if starter_value < worst_value:
                        worst_value = starter_value
                        worst_starter = starter

            if worst_starter is not None:
                marginal = fa_value - worst_value
                if marginal > 0:
                    suggestions.append(
                        {
                            "player_name": fa_name,
                            "player_id": int(fa.get("player_id", 0)),
                            "position": fa.get("positions", ""),
                            "marginal_value": round(marginal, 2),
                            "replaces": worst_starter.get(
                                "player_name",
                                worst_starter.get("name", "Unknown"),
                            ),
                        }
                    )

        suggestions.sort(key=lambda x: x["marginal_value"], reverse=True)
        return suggestions[:top_n]

    def _greedy_fallback(self) -> dict:
        """Simple greedy assignment when PuLP is unavailable."""
        if self.roster.empty:
            return {
                "assignments": [],
                "bench": [],
                "projected_stats": {},
                "status": "empty_roster",
            }

        assignments = []
        assigned = set()

        # Sort players by total stat value descending
        roster_scored = self.roster.copy()
        roster_scored["_value"] = roster_scored.apply(
            lambda r: _compute_player_value(r, self._category_weights), axis=1
        )
        roster_scored = roster_scored.sort_values("_value", ascending=False)

        for slot_name, (count, eligible) in self.slots.items():
            filled = 0
            for idx, row in roster_scored.iterrows():
                if idx in assigned or filled >= count:
                    continue
                player_positions = _parse_positions(row.get("positions", ""))
                if any(p in eligible for p in player_positions):
                    name = row.get("player_name", row.get("name", f"P{idx}"))
                    assignments.append(
                        {
                            "slot": slot_name,
                            "player_name": name,
                            "player_id": int(row.get("player_id", idx)),
                        }
                    )
                    assigned.add(idx)
                    filled += 1

        name_col = "player_name" if "player_name" in self.roster.columns else "name"
        bench = [self.roster.loc[idx].get(name_col, f"P{idx}") for idx in self.roster.index if idx not in assigned]

        projected = {}
        if assigned:
            starters = self.roster.loc[list(assigned)]
            for cat in ALL_CATS:
                if cat in starters.columns:
                    projected[cat] = float(starters[cat].sum())

            # Fix rate stats: compute weighted averages instead of sums
            projected = _fix_rate_stats(projected, starters)

        return {
            "assignments": assignments,
            "bench": bench,
            "projected_stats": projected,
            "status": "greedy_fallback",
        }


# ── Helper Functions ─────────────────────────────────────────────────


def _parse_positions(positions_str: str | None) -> list[str]:
    """Parse comma-separated position string into list."""
    if not positions_str or pd.isna(positions_str):
        return []
    return [p.strip() for p in str(positions_str).split(",") if p.strip()]


def _fix_rate_stats(projected: dict, starter_rows: pd.DataFrame) -> dict:
    """Recompute rate stats as weighted averages instead of sums.

    AVG = sum(h) / sum(ab), ERA = sum(er)*9 / sum(ip),
    WHIP = sum(bb_allowed + h_allowed) / sum(ip).
    """
    result = dict(projected)

    # AVG: weighted by at-bats
    if "avg" in result:
        if "h" in starter_rows.columns and "ab" in starter_rows.columns:
            total_ab = float(starter_rows["ab"].sum())
            total_h = float(starter_rows["h"].sum())
            result["avg"] = total_h / total_ab if total_ab > 0 else 0.0
        else:
            # Fallback: if component columns missing, leave as-is
            pass

    # ERA: weighted by innings pitched
    if "era" in result:
        if "er" in starter_rows.columns and "ip" in starter_rows.columns:
            total_ip = float(starter_rows["ip"].sum())
            total_er = float(starter_rows["er"].sum())
            result["era"] = (total_er * 9) / total_ip if total_ip > 0 else 0.0
        else:
            pass

    # WHIP: weighted by innings pitched
    if "whip" in result:
        if "ip" in starter_rows.columns:
            total_ip = float(starter_rows["ip"].sum())
            bb = float(starter_rows["bb_allowed"].sum()) if "bb_allowed" in starter_rows.columns else 0.0
            ha = float(starter_rows["h_allowed"].sum()) if "h_allowed" in starter_rows.columns else 0.0
            result["whip"] = (bb + ha) / total_ip if total_ip > 0 else 0.0
        else:
            pass

    return result


def _compute_player_value(player: pd.Series, weights: dict[str, float]) -> float:
    """Compute weighted stat value for a player.

    For inverse stats (ERA, WHIP), the contribution is weighted by IP so that
    high-workload pitchers' rates count proportionally more than relievers with
    minimal innings. This mirrors the LP solver's objective function.
    """
    value = 0.0
    ip = float(player.get("ip", 0) or 0)
    for cat in ALL_CATS:
        if cat not in player.index:
            continue
        val = float(player.get(cat, 0) or 0)
        w = weights.get(cat, 1.0)
        if cat in INVERSE_CATS:
            # Weight by IP so a 200-IP starter's ERA matters more than
            # a 1-IP reliever's ERA
            value -= val * ip * w
        else:
            value += val * w
    return value
