"""War Room -- Hot/Cold performance report.

Identifies roster players significantly over/under-performing their
season stats based on last-7-game data from MLB Stats API.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# IL/inactive statuses to skip (case-insensitive)
_INACTIVE_STATUSES = {"il10", "il15", "il60", "il", "na", "dl"}

# Deviation thresholds — raised moderate floor to prevent false signals
# from near-identical L7 vs season lines (e.g., .320 vs .321)
_STRONG_THRESHOLD = 1.2
_MODERATE_THRESHOLD = 1.0


def compute_hot_cold_report(
    roster: pd.DataFrame,
    max_entries: int = 4,
) -> list[dict]:
    """Compute a Hot/Cold performance report for rostered players.

    Compares each active player's last-7-game stats against their
    season-to-date baseline to identify significant streaks.

    Parameters
    ----------
    roster : pd.DataFrame
        Roster DataFrame with columns: name, team, positions, is_hitter,
        mlb_id, player_id, avg, obp, hr, rbi, sb, r, era, whip, k, w,
        sv, ip, status.
    max_entries : int
        Maximum number of entries to return (default 4).

    Returns
    -------
    list[dict]
        Sorted by |deviation_score| descending, balanced between hot/cold.
        Each dict has: player, team, status, player_type, headline, detail,
        verdict, deviation_score.
    """
    from src.game_day import get_player_recent_form_cached

    if roster is None or roster.empty:
        return []

    entries: list[dict] = []

    for _, row in roster.iterrows():
        # Skip inactive players
        status = str(row.get("status", "") or "").strip().lower()
        if status in _INACTIVE_STATUSES:
            continue

        # Skip players without mlb_id (NaN from SQLite NULL is truthy)
        raw_mlb_id = row.get("mlb_id")
        if not pd.notna(raw_mlb_id):
            continue

        try:
            mlb_id = int(float(raw_mlb_id))
        except (ValueError, TypeError):
            continue

        # Fetch recent form
        try:
            form = get_player_recent_form_cached(mlb_id)
        except Exception:
            logger.debug("Failed to fetch form for mlb_id=%s", mlb_id)
            continue

        if not form:
            continue

        l7 = form.get("l7") or {}
        if l7.get("games", 0) < 3:
            continue

        player_name = str(row.get("name", "Unknown"))
        team = str(row.get("team", ""))
        is_hitter = bool(row.get("is_hitter", True))
        player_type = form.get("player_type", "hitter" if is_hitter else "pitcher")

        if player_type == "hitter" or (player_type not in ("hitter", "pitcher") and is_hitter):
            entry = _evaluate_hitter(row, l7, player_name, team)
        else:
            entry = _evaluate_pitcher(row, l7, player_name, team)

        if entry is not None:
            entries.append(entry)

    if not entries:
        return []

    # Sort by absolute deviation descending
    entries.sort(key=lambda e: abs(e["deviation_score"]), reverse=True)

    # Balance hot and cold: try to include both if possible
    hot = [e for e in entries if e["status"] == "hot"]
    cold = [e for e in entries if e["status"] == "cold"]

    if max_entries >= 2 and hot and cold:
        half = max_entries // 2
        # Give each side at least half, fill remainder from the other
        selected_hot = hot[:half]
        selected_cold = cold[:half]
        remaining = max_entries - len(selected_hot) - len(selected_cold)
        # Fill remaining slots from whichever side has more strong signals
        leftover_hot = hot[half:]
        leftover_cold = cold[half:]
        leftover = sorted(
            leftover_hot + leftover_cold,
            key=lambda e: abs(e["deviation_score"]),
            reverse=True,
        )
        combined = selected_hot + selected_cold + leftover[:remaining]
        combined.sort(key=lambda e: abs(e["deviation_score"]), reverse=True)
        return combined[:max_entries]

    return entries[:max_entries]


def _evaluate_hitter(row: pd.Series, l7: dict, player_name: str, team: str) -> dict | None:
    """Evaluate a hitter's hot/cold status from L7 vs season stats."""
    l7_avg = l7.get("avg", 0.0) or 0.0
    l7_hr = l7.get("hr", 0) or 0
    l7_games = l7.get("games", 1) or 1
    l7_rbi = l7.get("rbi", 0) or 0
    l7_sb = l7.get("sb", 0) or 0
    l7_r = l7.get("r", 0) or 0

    season_avg = float(row.get("avg", 0.0) or 0.0)
    season_hr = float(row.get("hr", 0) or 0)

    # AVG deviation (each 50 points = 1 unit)
    avg_dev = (l7_avg - season_avg) / 0.050

    # HR pace deviation
    hr_per_game = l7_hr / l7_games
    hr_pace = hr_per_game * 162
    season_hr_baseline = season_hr if season_hr > 0 else 15
    hr_divisor = max(season_hr_baseline * 0.3, 5)
    season_hr_projected = season_hr_baseline  # use as-is for comparison
    hr_dev = (hr_pace - season_hr_projected) / hr_divisor

    deviation_score = avg_dev * 0.6 + hr_dev * 0.4

    if abs(deviation_score) < _MODERATE_THRESHOLD:
        return None

    is_hot = deviation_score > 0
    status = "hot" if is_hot else "cold"

    # Build headline from top L7 stats
    headline = _build_hitter_headline(l7_avg, l7_hr, l7_rbi, l7_sb, l7_r, l7_games, is_hot)

    # Build detail comparing to season
    if is_hot:
        diff = l7_avg - season_avg
        detail = f"Season avg: .{int(season_avg * 1000):03d} -- outperforming by +.{int(abs(diff) * 1000):03d}"
    else:
        diff = season_avg - l7_avg
        detail = f"Season avg: .{int(season_avg * 1000):03d} -- underperforming by .{int(abs(diff) * 1000):03d}"

    if is_hot:
        verdict = "Ride the streak -- start with confidence"
    else:
        verdict = "Consider benching -- dragging AVG/OBP down"

    return {
        "player": player_name,
        "team": team,
        "status": status,
        "player_type": "hitter",
        "headline": headline,
        "detail": detail,
        "verdict": verdict,
        "deviation_score": round(deviation_score, 2),
    }


def _build_hitter_headline(avg: float, hr: int, rbi: int, sb: int, r: int, games: int, is_hot: bool) -> str:
    """Build a readable headline from L7 hitter stats."""
    parts: list[str] = []

    avg_str = f".{int(avg * 1000):03d}" if avg > 0 else ".000"
    parts.append(f"{avg_str} AVG")

    if is_hot:
        # Highlight impressive stats
        if hr > 0:
            parts.append(f"{hr} HR")
        if rbi >= 3:
            parts.append(f"{rbi} RBI")
        if sb >= 2:
            parts.append(f"{sb} SB")
    else:
        # Highlight poor stats
        parts.append(f"{hr} HR")
        if r == 0:
            parts.append(f"{r} R")

    # Keep to 2-3 stats max for readability
    stat_text = ", ".join(parts[:3])
    return f"{stat_text} in last {games} games"


def _evaluate_pitcher(row: pd.Series, l7: dict, player_name: str, team: str) -> dict | None:
    """Evaluate a pitcher's hot/cold status from L7 vs season stats."""
    l7_era = l7.get("era", 0.0)
    l7_whip = l7.get("whip", 0.0)
    l7_k = l7.get("k", 0) or 0
    l7_ip = l7.get("ip", 0.0) or 0.0
    l7_games = l7.get("games", 1) or 1

    if l7_era is None:
        l7_era = 0.0
    if l7_whip is None:
        l7_whip = 0.0

    season_era = float(row.get("era", 4.50) or 4.50)
    season_whip = float(row.get("whip", 1.30) or 1.30)
    season_k = float(row.get("k", 0) or 0)
    season_ip = float(row.get("ip", 0.0) or 0.0)

    # ERA deviation (inverted: lower L7 ERA = positive/hot)
    era_dev = (season_era - l7_era) / 1.0

    # K/9 deviation
    if l7_ip > 0:
        l7_k_per_9 = l7_k / l7_ip * 9
    else:
        l7_k_per_9 = 0.0

    if season_ip > 0:
        season_k_per_9 = season_k / season_ip * 9
    else:
        season_k_per_9 = 7.0  # league average fallback

    k_dev = (l7_k_per_9 - season_k_per_9) / 2.0

    deviation_score = era_dev * 0.6 + k_dev * 0.4

    if abs(deviation_score) < _MODERATE_THRESHOLD:
        return None

    is_hot = deviation_score > 0
    status = "hot" if is_hot else "cold"

    # Build headline
    headline = _build_pitcher_headline(l7_era, l7_whip, l7_k, l7_games, is_hot)

    # Build detail
    if is_hot:
        diff = season_era - l7_era
        detail = f"Season ERA: {season_era:.2f} -- outperforming by {diff:.2f} ERA"
    else:
        diff = l7_era - season_era
        detail = f"Season ERA: {season_era:.2f} -- struggling with +{diff:.2f} ERA above norm"

    if is_hot:
        verdict = "Ride the streak -- start with confidence"
    else:
        verdict = "Consider benching to protect ERA/WHIP"

    return {
        "player": player_name,
        "team": team,
        "status": status,
        "player_type": "pitcher",
        "headline": headline,
        "detail": detail,
        "verdict": verdict,
        "deviation_score": round(deviation_score, 2),
    }


def _build_pitcher_headline(era: float, whip: float, k: int, games: int, is_hot: bool) -> str:
    """Build a readable headline from L7 pitcher stats."""
    parts: list[str] = []

    if is_hot:
        parts.append(f"{era:.2f} ERA")
        if k > 0:
            parts.append(f"{k} K")
    else:
        parts.append(f"{era:.2f} ERA")
        if whip >= 1.40:
            parts.append(f"{whip:.2f} WHIP")
        if k == 0:
            parts.append(f"{k} K")

    stat_text = ", ".join(parts[:3])
    return f"{stat_text} in last {games} games"
