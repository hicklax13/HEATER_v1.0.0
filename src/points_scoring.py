"""Points-league scoring valuation (Phase 4 slice 1).

Standalone + additive: pure functions over the existing player pool, parallel to
(and never touching) the category engine (LeagueConfig/SGPCalculator). Given a
points league's per-stat weights, computes a player's projected points, ranks
players, and totals a roster. Stats HEATER does not project are flagged
'uncovered', never silently zeroed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

# Friendly stat name -> pool column, per player type. The SINGLE place to widen
# coverage later. Verified against load_player_pool() (2026-06-27).
_HITTER_STAT_COLUMNS: dict[str, str] = {
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "H": "h",
    "BB": "bb",
    "HBP": "hbp",
    "AB": "ab",
    "SF": "sf",
    "AVG": "avg",
    "OBP": "obp",
}
_PITCHER_STAT_COLUMNS: dict[str, str] = {
    "IP": "ip",
    "K": "k",
    "W": "w",
    "L": "l",
    "SV": "sv",
    "ER": "er",
    "H": "h_allowed",
    "BB": "bb_allowed",
    "ERA": "era",
    "WHIP": "whip",
}


@dataclass(frozen=True)
class PointsScoringConfig:
    """Per-stat point weights for a points league. Keys are friendly stat names
    (case-insensitive); values are points per unit of the stat (sign encodes
    inverse stats, e.g. ER = -2.0)."""

    hitter_weights: dict[str, float]
    pitcher_weights: dict[str, float]
    name: str = "custom"


@dataclass
class PointsResult:
    points: float
    breakdown: dict[str, float] = field(default_factory=dict)
    uncovered: set[str] = field(default_factory=set)


def _num(value) -> float:
    """NaN/None/inf-safe numeric coercion -> finite float (0.0 on failure)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(f):
        return 0.0
    return f


def _get(row, key):
    """Mapping-or-Series accessor; returns None when the key is absent."""
    try:
        return row.get(key)  # dict / pandas Series / Mapping all support .get
    except AttributeError:
        return row[key] if key in row else None


def _score_half(row, weights: dict, stat_columns: dict[str, str]) -> PointsResult:
    """Score one weight map against one column map (one player 'half')."""
    points = 0.0
    breakdown: dict[str, float] = {}
    uncovered: set[str] = set()
    for raw_stat, weight in weights.items():
        stat = str(raw_stat).strip().upper()
        col = stat_columns.get(stat)
        if col is None:
            uncovered.add(stat)  # HEATER does not project this stat for this type
            continue
        contribution = _num(weight) * _num(_get(row, col))
        breakdown[stat] = contribution
        points += contribution
    return PointsResult(points=points, breakdown=breakdown, uncovered=uncovered)


def _is_hitter(row) -> bool:
    return bool(_num(_get(row, "is_hitter")))


def _has_pitcher_volume(row) -> bool:
    return _num(_get(row, "ip")) > 0.0


def project_player_points(player_row, config: PointsScoringConfig) -> PointsResult:
    """Project a single player's points under `config`.

    Hitters use hitter_weights, pitchers use pitcher_weights. A two-way player
    (is_hitter AND pitcher volume — Ohtani) is scored as BOTH halves summed; its
    breakdown keys are prefixed BAT:/PIT: to disambiguate same-named stats (a
    hitter's H vs a pitcher's H allowed)."""
    is_hit = _is_hitter(player_row)
    pitches = _has_pitcher_volume(player_row)
    two_way = is_hit and pitches

    score_hit = is_hit
    score_pit = two_way or (pitches and not is_hit)

    total = 0.0
    breakdown: dict[str, float] = {}
    uncovered: set[str] = set()

    if score_hit:
        half = _score_half(player_row, config.hitter_weights, _HITTER_STAT_COLUMNS)
        total += half.points
        prefix = "BAT:" if two_way else ""
        breakdown.update({f"{prefix}{k}": v for k, v in half.breakdown.items()})
        uncovered |= half.uncovered
    if score_pit:
        half = _score_half(player_row, config.pitcher_weights, _PITCHER_STAT_COLUMNS)
        total += half.points
        prefix = "PIT:" if two_way else ""
        breakdown.update({f"{prefix}{k}": v for k, v in half.breakdown.items()})
        uncovered |= half.uncovered

    return PointsResult(points=total, breakdown=breakdown, uncovered=uncovered)


def uncovered_stats(config: PointsScoringConfig, pool=None) -> dict[str, set[str]]:
    """Which configured stats HEATER cannot score, per player type — for upfront
    UI transparency. `pool` is accepted for signature symmetry but not required
    (coverage is determined by the static stat maps)."""
    hit = {str(s).strip().upper() for s in config.hitter_weights}
    pit = {str(s).strip().upper() for s in config.pitcher_weights}
    return {
        "hitter": {s for s in hit if s not in _HITTER_STAT_COLUMNS},
        "pitcher": {s for s in pit if s not in _PITCHER_STAT_COLUMNS},
    }


def rank_players_by_points(pool: pd.DataFrame, config: PointsScoringConfig) -> pd.DataFrame:
    """Return a copy of `pool` with a `points` column, sorted points-descending.
    Never mutates the input."""
    out = pool.copy()
    out["points"] = [project_player_points(row, config).points for _, row in out.iterrows()]
    return out.sort_values("points", ascending=False, kind="mergesort").reset_index(drop=True)


def roster_points(roster_ids: list, pool: pd.DataFrame, config: PointsScoringConfig) -> float:
    """Total projected points for the players in `roster_ids` (looked up by
    player_id). Unknown ids contribute 0."""
    if pool is None or len(pool) == 0 or not roster_ids:
        return 0.0
    ids = {int(_num(i)) for i in roster_ids}
    pid = pd.to_numeric(pool["player_id"], errors="coerce")
    subset = pool[pid.isin(ids)]
    return float(sum(project_player_points(row, config).points for _, row in subset.iterrows()))


# A documented, illustrative points preset built ONLY from projected stats (so it
# never produces 'uncovered'). It is NOT a claim of any provider's exact defaults
# — a user's real weights come from their league settings (Phase 5 connectors).
STANDARD_POINTS = PointsScoringConfig(
    name="standard",
    hitter_weights={
        "R": 1.0,
        "HR": 4.0,
        "RBI": 1.0,
        "SB": 2.0,
        "H": 1.0,
        "BB": 1.0,
    },
    pitcher_weights={
        "IP": 1.0,
        "K": 1.0,
        "W": 5.0,
        "SV": 5.0,
        "ER": -2.0,
        "H": -0.5,
        "BB": -0.5,
        "L": -3.0,
    },
)
