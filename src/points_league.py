# src/points_league.py
"""Fantasy points league projections with Yahoo/ESPN/CBS scoring presets."""

from __future__ import annotations

import pandas as pd

SCORING_PRESETS: dict[str, dict[str, dict[str, float]]] = {
    "yahoo": {
        "hitting": {
            "1B": 2.6,
            "2B": 5.2,
            "3B": 7.8,
            "HR": 9.4,
            "R": 1.9,
            "RBI": 1.9,
            "BB": 2.6,
            "HBP": 2.6,
            "SB": 4.2,
            "CS": -2.6,
            "K_hitting": -0.8,
        },
        "pitching": {
            "IP": 2.4,
            "W": 5.0,
            "L": -5.0,
            "SV": 5.0,
            "H_allowed": -0.8,
            "BB_allowed": -0.8,
            "K": 1.0,
            "ER": -1.6,
        },
    },
    "espn": {
        "hitting": {
            "1B": 1,
            "2B": 2,
            "3B": 3,
            "HR": 4,
            "R": 1,
            "RBI": 1,
            "BB": 1,
            "HBP": 1,
            "SB": 1,
            "CS": -1,
        },
        "pitching": {
            "IP": 1,
            "W": 5,
            "L": -5,
            "SV": 5,
            "H_allowed": -1,
            "BB_allowed": -1,
            "K": 1,
            "ER": -2,
        },
    },
    "cbs": {
        "hitting": {
            "1B": 1,
            "2B": 2,
            "3B": 3,
            "HR": 4,
            "R": 1,
            "RBI": 1,
            "BB": 1,
            "SB": 2,
            "CS": -1,
        },
        "pitching": {
            "IP": 3,
            "W": 7,
            "L": -7,
            "SV": 10,
            "H_allowed": -1,
            "BB_allowed": -1,
            "K": 2,
            "ER": -2,
        },
    },
}


def get_scoring_preset(name: str) -> tuple[dict[str, float], dict[str, float]]:
    """Return (hitting_weights, pitching_weights) for a scoring preset."""
    preset = SCORING_PRESETS[name]
    return preset["hitting"], preset["pitching"]


def estimate_missing_batting_stats(row: pd.Series) -> dict[str, float]:
    """Estimate 1B, 2B, 3B, CS, K_hitting from available stats."""
    h = float(row.get("h", 0) or 0)
    hr = float(row.get("hr", 0) or 0)
    pa = float(row.get("pa", 0) or 0)
    sb = float(row.get("sb", 0) or 0)
    non_hr_h = max(0.0, h - hr)
    doubles = non_hr_h * 0.22
    triples = non_hr_h * 0.025
    singles = max(0.0, non_hr_h - doubles - triples)
    return {
        "1B": singles,
        "2B": doubles,
        "3B": triples,
        "CS": sb * 0.27,
        "K_hitting": pa * 0.223,
    }


def compute_fantasy_points(
    projections_df: pd.DataFrame,
    hitting_weights: dict[str, float],
    pitching_weights: dict[str, float],
) -> pd.DataFrame:
    """Compute fantasy points for all players."""
    df = projections_df.copy()
    df["fantasy_points"] = 0.0
    hitting_stat_map = {
        "R": "r",
        "RBI": "rbi",
        "HR": "hr",
        "SB": "sb",
        "BB": "bb",
        "HBP": "hbp",
    }
    pitching_stat_map = {
        "IP": "ip",
        "W": "w",
        "L": "l",
        "SV": "sv",
        "K": "k",
        "ER": "er",
        "H_allowed": "h_allowed",
        "BB_allowed": "bb_allowed",
    }
    for idx, row in df.iterrows():
        pts = 0.0
        is_hitter = bool(row.get("is_hitter", True))
        if is_hitter:
            estimated = estimate_missing_batting_stats(row)
            for stat_key, weight in hitting_weights.items():
                if stat_key in estimated:
                    pts += estimated[stat_key] * weight
                elif stat_key in hitting_stat_map:
                    col = hitting_stat_map[stat_key]
                    pts += float(row.get(col, 0) or 0) * weight
        else:
            for stat_key, weight in pitching_weights.items():
                if stat_key in pitching_stat_map:
                    col = pitching_stat_map[stat_key]
                    pts += float(row.get(col, 0) or 0) * weight
        df.at[idx, "fantasy_points"] = round(pts, 1)
    return df


def compute_points_leaders(
    stats_df: pd.DataFrame,
    hitting_weights: dict[str, float],
    pitching_weights: dict[str, float],
    top_n: int = 20,
) -> pd.DataFrame:
    """Return the top-N fantasy points scorers.

    Parameters
    ----------
    stats_df : pd.DataFrame
        Player stats (must include columns used by compute_fantasy_points).
    hitting_weights / pitching_weights : dict
        Scoring weights from a preset.
    top_n : int
        Number of leaders to return.

    Returns
    -------
    pd.DataFrame
        Sorted DataFrame with fantasy_points column.
    """
    if stats_df.empty:
        return pd.DataFrame()
    scored = compute_fantasy_points(stats_df, hitting_weights, pitching_weights)
    scored = scored.sort_values("fantasy_points", ascending=False).head(top_n)
    # Pick display columns (include mlb_id/player_id for headshots and player cards)
    display_cols = ["player_id", "name", "team", "positions", "fantasy_points", "mlb_id"]
    available = [c for c in display_cols if c in scored.columns]
    if not available:
        available = scored.columns.tolist()
    return scored[available].reset_index(drop=True)
