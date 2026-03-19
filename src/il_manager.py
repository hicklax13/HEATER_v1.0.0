"""Auto-swap IL player detection with replacement recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

IL_DURATION_ESTIMATES: dict[str, float] = {
    "IL10": 2.0,
    "IL15": 3.5,
    "IL60": 10.0,
    "DTD": 0.5,
}


@dataclass
class ILAlert:
    player_id: int
    player_name: str
    il_type: str
    expected_duration_weeks: float
    recommended_replacement_id: int | None = None
    recommended_replacement_name: str | None = None
    lost_sgp: float = 0.0
    replacement_sgp: float = 0.0
    net_sgp_impact: float = 0.0
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()


def classify_il_type(status_string: str) -> str:
    """Classify IL status from raw status string."""
    s = status_string.upper().strip()
    if "60" in s:
        return "IL60"
    elif "15" in s:
        return "IL15"
    elif "10" in s:
        return "IL10"
    elif "DTD" in s or "DAY" in s:
        return "DTD"
    elif "IL" in s or "DL" in s or "INJURED" in s:
        return "IL15"  # default IL
    return "DTD"


def estimate_il_duration(il_type: str, position: str = "") -> float:
    """Estimate IL duration in weeks."""
    base = IL_DURATION_ESTIMATES.get(il_type, 2.0)
    # Pitchers tend to be out slightly longer
    if position and position.upper() in ("SP", "RP", "P"):
        base *= 1.15
    return round(base, 1)


def compute_lost_sgp(
    player_sgp: float,
    duration_weeks: float,
    weeks_remaining: float = 22.0,
) -> float:
    """Compute SGP lost due to IL stint."""
    if weeks_remaining <= 0:
        return 0.0
    fraction = min(1.0, duration_weeks / weeks_remaining)
    return round(player_sgp * fraction, 3)


def find_best_replacement(
    vacated_positions: list[str],
    bench_players: pd.DataFrame,
    il_duration_weeks: float = 2.0,
) -> dict | None:
    """Find best bench replacement eligible at vacated position.

    Returns dict with player_id, name, expected_sgp or None.
    """
    if bench_players.empty or not vacated_positions:
        return None
    eligible = bench_players.copy()
    # Filter to players eligible at any vacated position
    pos_col = "positions" if "positions" in eligible.columns else None
    if pos_col:
        mask = eligible[pos_col].apply(lambda p: any(vp.upper() in str(p).upper() for vp in vacated_positions))
        eligible = eligible[mask]
    if eligible.empty:
        return None
    # Pick highest SGP player
    sgp_col = "pick_score" if "pick_score" in eligible.columns else None
    if sgp_col is None:
        # Pick first available
        row = eligible.iloc[0]
    else:
        row = eligible.sort_values(sgp_col, ascending=False).iloc[0]
    return {
        "player_id": int(row.get("player_id", 0)),
        "name": str(row.get("name", row.get("player_name", "Unknown"))),
        "expected_sgp": float(row.get("pick_score", 0) or 0) * (il_duration_weeks / 22.0),
    }


def detect_il_changes(
    current_roster: pd.DataFrame,
    last_known_status: dict[int, str] | None = None,
) -> list[dict]:
    """Detect new IL changes by comparing current roster status to last known."""
    if last_known_status is None:
        last_known_status = {}
    changes = []
    status_col = "status" if "status" in current_roster.columns else "injury_note"
    if status_col not in current_roster.columns:
        return changes
    for _, row in current_roster.iterrows():
        pid = int(row.get("player_id", 0))
        status = str(row.get(status_col, "") or "")
        old_status = last_known_status.get(pid, "")
        if status and status != old_status:
            il_type = classify_il_type(status)
            if il_type in IL_DURATION_ESTIMATES:
                changes.append(
                    {
                        "player_id": pid,
                        "player_name": str(row.get("name", row.get("player_name", ""))),
                        "il_type": il_type,
                        "status": status,
                        "positions": str(row.get("positions", "")),
                    }
                )
    return changes


def generate_il_alert(
    il_player: dict,
    bench_players: pd.DataFrame,
    player_sgp: float = 0.0,
    weeks_remaining: float = 22.0,
) -> ILAlert:
    """Generate a full IL alert with replacement recommendation."""
    il_type = il_player.get("il_type", "IL15")
    duration = estimate_il_duration(il_type, il_player.get("positions", ""))
    lost = compute_lost_sgp(player_sgp, duration, weeks_remaining)
    positions = [p.strip() for p in il_player.get("positions", "Util").split(",")]
    replacement = find_best_replacement(positions, bench_players, duration)
    alert = ILAlert(
        player_id=il_player.get("player_id", 0),
        player_name=il_player.get("player_name", ""),
        il_type=il_type,
        expected_duration_weeks=duration,
        lost_sgp=lost,
    )
    if replacement:
        alert.recommended_replacement_id = replacement["player_id"]
        alert.recommended_replacement_name = replacement["name"]
        alert.replacement_sgp = round(replacement["expected_sgp"], 3)
        alert.net_sgp_impact = round(replacement["expected_sgp"] - lost, 3)
    return alert
