"""Closer depth chart monitor with job security scoring."""

from __future__ import annotations

import pandas as pd


def compute_job_security(hierarchy_confidence: float, projected_sv: float) -> float:
    """Compute closer job security score [0, 1].

    Formula: security = 0.6 * hierarchy_confidence + 0.4 * min(1.0, projected_sv / 30)
    """
    sv_component = min(1.0, max(0.0, projected_sv) / 30.0)
    raw = 0.6 * max(0.0, min(1.0, hierarchy_confidence)) + 0.4 * sv_component
    return max(0.0, min(1.0, raw))


def get_security_color(security: float) -> str:
    """Return color hex based on job security level.

    >= 0.7 -> green (#2d6a4f)
    >= 0.4 -> yellow (#ff9f1c)
    <  0.4 -> red (#e63946)
    """
    if security >= 0.7:
        return "#2d6a4f"
    elif security >= 0.4:
        return "#ff9f1c"
    else:
        return "#e63946"


def build_closer_grid(
    depth_data: dict,
    player_pool: pd.DataFrame | None = None,
) -> list[dict]:
    """Build 30-team closer grid from depth chart data.

    Args:
        depth_data: Dict keyed by team code, each value has keys:
            - closer: str (closer name or "Committee")
            - setup: list[str] (setup arm names)
            - closer_confidence: float [0, 1]
        player_pool: Optional DataFrame with player stats (name, team, sv, era, whip).

    Returns:
        List of dicts sorted by team code, each with:
        team, closer_name, setup_names, job_security, security_color,
        projected_sv, era, whip.
    """
    if not depth_data:
        return []

    grid = []
    for team, info in sorted(depth_data.items()):
        closer_name = info.get("closer", "Unknown")
        setup_names = info.get("setup", [])
        confidence = float(info.get("closer_confidence", 0.5))
        projected_sv = 0.0
        era = 0.0
        whip = 0.0

        mlb_id = None
        if player_pool is not None and not player_pool.empty:
            match = player_pool[(player_pool["name"] == closer_name) & (player_pool["team"] == team)]
            if not match.empty:
                row = match.iloc[0]
                _sv = row.get("sv", 0)
                projected_sv = 0.0 if pd.isna(_sv) else float(_sv or 0)
                _era = row.get("era", 0)
                era = 0.0 if pd.isna(_era) else float(_era or 0)
                _whip = row.get("whip", 0)
                whip = 0.0 if pd.isna(_whip) else float(_whip or 0)
                mlb_id = row.get("mlb_id")

        security = compute_job_security(confidence, projected_sv)
        grid.append(
            {
                "team": team,
                "closer_name": closer_name,
                "setup_names": setup_names,
                "job_security": round(security, 3),
                "security_color": get_security_color(security),
                "projected_sv": projected_sv,
                "era": era,
                "whip": whip,
                "mlb_id": mlb_id,
            }
        )

    return grid
