# src/schedule_grid.py
"""7-day schedule grid with matchup color-coding."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

TIER_COLORS: dict[str, str] = {
    "smash": "#2d6a4f",
    "favorable": "#40916c",
    "neutral": "#6b7280",
    "unfavorable": "#ff9f1c",
    "avoid": "#e63946",
}

_DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _get_week_dates(start_date: datetime | None = None) -> list[str]:
    """Return ISO date strings for the next 7 days starting from Monday."""
    if start_date is None:
        start_date = datetime.now(UTC)
    # Find the Monday of this week
    monday = start_date - timedelta(days=start_date.weekday())
    return [(monday + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]


def build_schedule_grid(
    roster: pd.DataFrame,
    weekly_schedule: list[dict] | None = None,
    matchup_ratings: pd.DataFrame | None = None,
    start_date: datetime | None = None,
) -> dict:
    """Build a 7-day schedule grid for the user's roster.

    Args:
        roster: DataFrame with player_id, name, team, positions, is_hitter
        weekly_schedule: List of game dicts from MLB Stats API
        matchup_ratings: DataFrame from compute_weekly_matchup_ratings()
        start_date: Override start date (default: current week)

    Returns:
        {
            "dates": ["2026-03-16", ...],  # 7 ISO dates
            "day_labels": ["Mon", "Tue", ...],
            "players": [
                {
                    "player_id": int, "name": str, "team": str,
                    "positions": str, "is_hitter": bool,
                    "days": [
                        {"date": str, "has_game": bool, "opponent": str|None,
                         "is_home": bool, "tier": str|None, "tier_color": str}
                        | None  # off day
                    ]
                }
            ],
            "games_per_day": [int, ...],  # total team games per day
        }
    """
    dates = _get_week_dates(start_date)

    # Build game lookup: {(team, date): {opponent, is_home}}
    game_lookup: dict[tuple[str, str], dict] = {}
    if weekly_schedule:
        for game in weekly_schedule:
            game_date = str(game.get("date", ""))[:10]
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            if home:
                game_lookup[(home, game_date)] = {
                    "opponent": away,
                    "is_home": True,
                }
            if away:
                game_lookup[(away, game_date)] = {
                    "opponent": home,
                    "is_home": False,
                }

    # Build rating lookup: {player_id: matchup_tier}
    rating_lookup: dict[int, str] = {}
    if matchup_ratings is not None and not matchup_ratings.empty:
        for _, row in matchup_ratings.iterrows():
            pid = int(row.get("player_id", 0))
            tier = str(row.get("matchup_tier", "neutral"))
            rating_lookup[pid] = tier

    players = []
    games_per_day = [0] * 7

    if roster is not None and not roster.empty:
        for _, player in roster.iterrows():
            pid = int(player.get("player_id", 0))
            team = str(player.get("team", ""))
            tier = rating_lookup.get(pid, "neutral")

            days = []
            for i, date in enumerate(dates):
                game_info = game_lookup.get((team, date))
                if game_info:
                    games_per_day[i] += 1
                    days.append(
                        {
                            "date": date,
                            "has_game": True,
                            "opponent": game_info["opponent"],
                            "is_home": game_info["is_home"],
                            "tier": tier,
                            "tier_color": TIER_COLORS.get(tier, TIER_COLORS["neutral"]),
                        }
                    )
                else:
                    days.append(
                        {
                            "date": date,
                            "has_game": False,
                            "opponent": None,
                            "is_home": False,
                            "tier": None,
                            "tier_color": TIER_COLORS["neutral"],
                        }
                    )

            players.append(
                {
                    "player_id": pid,
                    "name": str(player.get("name", "")),
                    "team": team,
                    "positions": str(player.get("positions", "")),
                    "is_hitter": bool(player.get("is_hitter", True)),
                    "days": days,
                }
            )

    return {
        "dates": dates,
        "day_labels": _DAY_LABELS,
        "players": players,
        "games_per_day": games_per_day,
    }


def render_schedule_html(grid: dict) -> str:
    """Render schedule grid as an HTML table string."""
    if not grid or not grid.get("players"):
        return "<p>No schedule data available.</p>"

    dates = grid["dates"]
    day_labels = grid["day_labels"]

    # Header row
    header = "<tr><th style='text-align:left;padding:6px 10px;'>Player</th>"
    for i, label in enumerate(day_labels):
        header += f"<th style='text-align:center;padding:6px 8px;'>{label}<br/><small>{dates[i][5:]}</small></th>"
    header += "</tr>"

    # Player rows
    rows = []
    for p in grid["players"]:
        row = f"<tr><td style='padding:6px 10px;font-weight:600;white-space:nowrap;'>{p['name']}</td>"
        for day in p["days"]:
            if day["has_game"]:
                color = day["tier_color"]
                prefix = "vs" if day["is_home"] else "@"
                opp = day["opponent"] or "?"
                row += (
                    f"<td style='text-align:center;padding:4px 6px;"
                    f"background:{color};color:#fff;border-radius:4px;'>"
                    f"<small>{prefix} {opp}</small></td>"
                )
            else:
                row += "<td style='text-align:center;padding:4px 6px;color:#ccc;'>-</td>"
        row += "</tr>"
        rows.append(row)

    # Games per day footer
    footer = "<tr style='border-top:2px solid #ddd;'><td style='padding:6px 10px;font-weight:600;'>Games</td>"
    for count in grid["games_per_day"]:
        footer += f"<td style='text-align:center;padding:4px 6px;font-weight:600;'>{count}</td>"
    footer += "</tr>"

    return (
        "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
        f"<thead>{header}</thead>"
        f"<tbody>{''.join(rows)}{footer}</tbody>"
        "</table>"
    )
