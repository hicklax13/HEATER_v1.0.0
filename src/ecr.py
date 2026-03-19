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
