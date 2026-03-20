"""Category leaders, points leaders, and breakout detection."""

from __future__ import annotations

import pandas as pd

INVERSE_CATS = {"ERA", "WHIP", "L"}


def compute_category_leaders(
    season_stats_df: pd.DataFrame,
    categories: list[str] | None = None,
    min_pa: int = 50,
    min_ip: float = 20.0,
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """Compute leaders per category. Ascending sort for ERA/WHIP/L."""
    cats = categories or ["R", "HR", "RBI", "SB", "AVG", "OBP", "W", "SV", "K", "ERA", "WHIP"]
    stat_map = {
        "R": "r",
        "HR": "hr",
        "RBI": "rbi",
        "SB": "sb",
        "AVG": "avg",
        "OBP": "obp",
        "W": "w",
        "SV": "sv",
        "K": "k",
        "ERA": "era",
        "WHIP": "whip",
        "L": "l",
    }
    leaders = {}
    for cat in cats:
        col = stat_map.get(cat, cat.lower())
        if col not in season_stats_df.columns:
            continue
        df = season_stats_df.copy()
        # Filter by minimum playing time
        is_pitch = cat in {"W", "SV", "K", "ERA", "WHIP", "L"}
        if is_pitch and "ip" in df.columns:
            df = df[pd.to_numeric(df["ip"], errors="coerce").fillna(0) >= min_ip]
        elif not is_pitch and "pa" in df.columns:
            df = df[pd.to_numeric(df["pa"], errors="coerce").fillna(0) >= min_pa]
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        ascending = cat in INVERSE_CATS
        df = df.sort_values(col, ascending=ascending).head(top_n)
        leaders[cat] = df
    return leaders


def compute_points_leaders(
    season_stats_df: pd.DataFrame,
    hitting_weights: dict[str, float],
    pitching_weights: dict[str, float],
    top_n: int = 20,
) -> pd.DataFrame:
    """Top N players by fantasy points."""
    try:
        from src.points_league import compute_fantasy_points

        result = compute_fantasy_points(season_stats_df, hitting_weights, pitching_weights)
        return result.sort_values("fantasy_points", ascending=False).head(top_n).reset_index(drop=True)
    except ImportError:
        return pd.DataFrame()


def filter_leaders_by_position(leaders_df: pd.DataFrame, position: str, pos_col: str = "positions") -> pd.DataFrame:
    """Filter leaders to a specific position."""
    if pos_col not in leaders_df.columns:
        return leaders_df
    return leaders_df[leaders_df[pos_col].str.contains(position, case=False, na=False)]


def detect_breakouts(
    season_stats_df: pd.DataFrame,
    preseason_df: pd.DataFrame,
    stats: list[str] | None = None,
    threshold: float = 1.5,
) -> pd.DataFrame:
    """Detect players significantly outperforming projections.
    z = (observed - projected) / max(std, 0.01), breakout if z > threshold.
    """
    inverse_stats = {"era", "whip"}
    check_stats = stats or ["hr", "rbi", "sb", "k", "avg"]
    results = []
    name_col = "name" if "name" in season_stats_df.columns else "player_name"
    pre_name = "name" if "name" in preseason_df.columns else "player_name"
    pre_lookup = {}
    for _, row in preseason_df.iterrows():
        pre_lookup[str(row.get(pre_name, "")).lower()] = row
    for _, actual in season_stats_df.iterrows():
        pname = str(actual.get(name_col, "")).lower()
        if pname not in pre_lookup:
            continue
        projected = pre_lookup[pname]
        max_z = 0.0
        best_stat = ""
        for stat in check_stats:
            obs = float(actual.get(stat, 0) or 0)
            proj = float(projected.get(stat, 0) or 0)
            # Use 15% of projected as rough std, with stat-aware minimum
            min_std = 0.5 if stat in ("hr", "sb", "w", "sv") else 0.01 if stat in ("avg", "obp", "era", "whip") else 1.0
            std = max(min_std, abs(proj) * 0.15)
            # For inverse stats (ERA, WHIP), lower observed is better
            if stat in inverse_stats:
                z = (proj - obs) / std
            else:
                z = (obs - proj) / std
            if z > max_z:
                max_z = z
                best_stat = stat
        if max_z > threshold:
            results.append(
                {
                    "name": actual.get(name_col, ""),
                    "breakout_stat": best_stat,
                    "z_score": round(max_z, 2),
                }
            )
    return (
        pd.DataFrame(results).sort_values("z_score", ascending=False).reset_index(drop=True)
        if results
        else pd.DataFrame(columns=["name", "breakout_stat", "z_score"])
    )
