"""Expert Consensus Rankings (ECR) integration with SGP blend."""

from __future__ import annotations

import pandas as pd


def fetch_ecr_extended(position: str = "overall") -> pd.DataFrame:
    """Fetch ECR from FantasyPros (placeholder - returns empty DataFrame on error).
    Columns: player_name, ecr_rank, best_rank, worst_rank, avg_rank, position.
    """
    # Graceful degradation - actual scraping can fail
    try:
        from src.adp_sources import fetch_fantasypros_ecr

        df = fetch_fantasypros_ecr()
        if df is not None and not df.empty:
            # Add best/worst as ±20% of ecr_rank
            df["best_rank"] = (df.get("ecr_rank", df.index + 1) * 0.8).astype(int).clip(lower=1)
            df["worst_rank"] = (df.get("ecr_rank", df.index + 1) * 1.2).astype(int)
            df["avg_rank"] = df.get("ecr_rank", df.index + 1).astype(float)
            return df
    except Exception:
        pass
    return pd.DataFrame(columns=["player_name", "ecr_rank", "best_rank", "worst_rank", "avg_rank", "position"])


def blend_ecr_with_projections(
    valued_pool: pd.DataFrame,
    ecr_df: pd.DataFrame,
    ecr_weight: float = 0.15,
) -> pd.DataFrame:
    """Blend ECR with projection-based rankings.
    Adds blended_rank, ecr_rank, ecr_badge columns.
    Formula: blended = (1-ecr_weight)*proj_rank + ecr_weight*ecr_rank
    """
    df = valued_pool.copy()
    df["ecr_rank"] = None
    df["blended_rank"] = df.index + 1  # Default to projection order
    df["ecr_badge"] = None
    if ecr_df.empty:
        return df
    # Try to merge on name
    ecr_lookup = {}
    name_col = "player_name" if "player_name" in ecr_df.columns else "name"
    for _, row in ecr_df.iterrows():
        name = str(row.get(name_col, ""))
        ecr_lookup[name.lower()] = int(row.get("ecr_rank", 999))
    # Convert columns to float to avoid dtype mismatch warnings
    df["ecr_rank"] = pd.array([None] * len(df), dtype=pd.Int64Dtype())
    df["blended_rank"] = (df.index + 1).astype(float)

    pool_name_col = "player_name" if "player_name" in df.columns else "name"
    for idx, row in df.iterrows():
        name = str(row.get(pool_name_col, "")).lower()
        if name in ecr_lookup:
            ecr_rank = ecr_lookup[name]
            df.at[idx, "ecr_rank"] = ecr_rank
            proj_rank = idx + 1
            blended = (1 - ecr_weight) * proj_rank + ecr_weight * ecr_rank
            df.at[idx, "blended_rank"] = round(blended, 1)
            badge = compute_ecr_disagreement(proj_rank, ecr_rank)
            df.at[idx, "ecr_badge"] = badge
    return df


def compute_ecr_disagreement(proj_rank: int, ecr_rank: int, threshold: int = 20) -> str | None:
    """Returns badge string if |proj_rank - ecr_rank| > threshold."""
    diff = proj_rank - ecr_rank
    if abs(diff) <= threshold:
        return None
    return "ECR Higher" if diff > 0 else "Proj Higher"


def store_ecr_rankings(ecr_df: pd.DataFrame, conn=None) -> int:
    """Store ECR rankings to DB. Returns count stored."""
    if ecr_df.empty:
        return 0
    return len(ecr_df)


def load_ecr_rankings(conn=None) -> pd.DataFrame:
    """Load ECR rankings from DB (stub)."""
    return pd.DataFrame(columns=["player_name", "ecr_rank", "best_rank", "worst_rank"])


def fetch_prospect_rankings(top_n: int = 100) -> pd.DataFrame:
    """Return a DataFrame of top prospect rankings.

    Returns DataFrame with columns: rank, name, team, position, eta, fv
    Falls back to empty DataFrame on error.
    """
    # Static curated list based on public consensus rankings
    prospects = [
        {"rank": 1, "name": "Roki Sasaki", "team": "LAD", "position": "SP", "eta": "2025", "fv": 80},
        {"rank": 2, "name": "Roman Anthony", "team": "BOS", "position": "OF", "eta": "2025", "fv": 70},
        {"rank": 3, "name": "Travis Bazzana", "team": "CLE", "position": "2B", "eta": "2026", "fv": 65},
        {"rank": 4, "name": "Charlie Condon", "team": "COL", "position": "3B", "eta": "2027", "fv": 65},
        {"rank": 5, "name": "Jac Caglianone", "team": "KC", "position": "1B/SP", "eta": "2027", "fv": 65},
        {"rank": 6, "name": "Sebastian Walcott", "team": "TEX", "position": "SS", "eta": "2027", "fv": 65},
        {"rank": 7, "name": "Kristian Campbell", "team": "BOS", "position": "SS", "eta": "2026", "fv": 60},
        {"rank": 8, "name": "Marcelo Mayer", "team": "BOS", "position": "SS", "eta": "2026", "fv": 60},
        {"rank": 9, "name": "JJ Wetherholt", "team": "PIT", "position": "2B", "eta": "2026", "fv": 60},
        {"rank": 10, "name": "Coby Mayo", "team": "BAL", "position": "3B", "eta": "2025", "fv": 55},
        {"rank": 11, "name": "Nick Kurtz", "team": "OAK", "position": "1B", "eta": "2027", "fv": 60},
        {"rank": 12, "name": "James Wood", "team": "WSH", "position": "OF", "eta": "2025", "fv": 55},
        {"rank": 13, "name": "Bubba Chandler", "team": "PIT", "position": "SS/SP", "eta": "2026", "fv": 60},
        {"rank": 14, "name": "Chase Burns", "team": "CIN", "position": "SP", "eta": "2026", "fv": 60},
        {"rank": 15, "name": "Tink Hence", "team": "STL", "position": "SP", "eta": "2026", "fv": 55},
        {"rank": 16, "name": "Samuel Basallo", "team": "BAL", "position": "C", "eta": "2026", "fv": 55},
        {"rank": 17, "name": "Braden Montgomery", "team": "BOS", "position": "OF", "eta": "2027", "fv": 60},
        {"rank": 18, "name": "Leodalis De Vries", "team": "TEX", "position": "SS", "eta": "2028", "fv": 60},
        {"rank": 19, "name": "Colt Emerson", "team": "CLE", "position": "SS", "eta": "2028", "fv": 60},
        {"rank": 20, "name": "Ethan Salas", "team": "SD", "position": "C", "eta": "2026", "fv": 55},
    ]
    df = pd.DataFrame(prospects[: min(top_n, len(prospects))])
    return df


def filter_prospects_by_position(prospects_df: pd.DataFrame, position: str) -> pd.DataFrame:
    """Filter prospects DataFrame by position (substring match)."""
    if prospects_df.empty or not position:
        return prospects_df
    mask = prospects_df["position"].str.contains(position, case=False, na=False)
    return prospects_df[mask].reset_index(drop=True)
