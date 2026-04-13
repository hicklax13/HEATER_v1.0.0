"""War Room — Today's Actions engine.

Computes specific, named roster moves for today based on MLB schedule,
matchup state, and player availability.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Category-to-stat mapping for losig category analysis
_CAT_STAT_MAP: dict[str, str] = {
    "SB": "sb",
    "HR": "hr",
    "R": "r",
    "RBI": "rbi",
    "AVG": "avg",
    "OBP": "obp",
    "K": "k",
    "W": "w",
    "SV": "sv",
    "ERA": "era",
    "WHIP": "whip",
    "L": "l",
}

# Hitting categories (for mapping bench hitters to losing cats)
_HITTING_CATS = {"SB", "HR", "R", "RBI", "AVG", "OBP"}
_PITCHING_CATS = {"K", "W", "SV", "ERA", "WHIP", "L"}

# Slots that count as "starting" (not bench / IL)
_BENCH_SLOTS = {"BN", "IL", "IL+", "DL", "NA"}


def _position_eligible(player_positions: str, slot: str) -> bool:
    """Check if a player's position list makes them eligible for a roster slot."""
    slot_upper = slot.strip().upper()
    positions = {p.strip().upper() for p in str(player_positions).split(",") if p.strip()}
    if slot_upper in ("UTIL", "UT"):
        return True
    if slot_upper == "P":
        return bool(positions & {"SP", "RP", "P"})
    if slot_upper == "OF":
        return bool(positions & {"LF", "CF", "RF", "OF"})
    return slot_upper in positions


def _is_bench(roster_slot: str, status: str = "") -> bool:
    """Return True if the roster slot is a bench/IL slot or player has IL status."""
    if str(roster_slot).strip().upper() in _BENCH_SLOTS:
        return True
    # Also check player status (IL players may have non-IL roster_slot in some data)
    s = str(status).strip().lower()
    return s in {"il10", "il15", "il60", "il", "na", "dl"}


def _is_sp(positions: str) -> bool:
    """Return True if player has SP eligibility."""
    pos_set = {p.strip().upper() for p in str(positions).split(",") if p.strip()}
    return "SP" in pos_set


def _is_hitter_row(row: pd.Series) -> bool:
    """Determine if a row represents a hitter."""
    if "is_hitter" in row.index:
        return bool(row["is_hitter"])
    pos_set = {p.strip().upper() for p in str(row.get("positions", "")).split(",") if p.strip()}
    pitcher_pos = {"SP", "RP", "P"}
    return not bool(pos_set & pitcher_pos) or bool(pos_set - pitcher_pos)


def _get_opponent_team(player_team: str, teams_playing: list[str]) -> str | None:
    """Attempt to infer the opponent team.

    This is a best-effort lookup. Without a full schedule we cannot know
    the actual opponent, so we return None when we cannot determine it.
    The caller uses get_team_strength on the opponent when available.
    """
    # We do not have pairings here; return None so the caller skips
    # opponent-specific logic unless it can resolve it another way.
    return None


def _build_schedule_context() -> tuple[dict[str, str], set[str]]:
    """Build schedule pairings and set of probable starters from statsapi.

    Returns
    -------
    pairings : dict[str, str]
        Maps team abbreviation to opponent abbreviation (both directions).
    probable_starters : set[str]
        Lowercased last names of today's probable pitchers.
    """
    pairings: dict[str, str] = {}
    probable_starters: set[str] = set()
    try:
        from datetime import datetime, timedelta, timezone

        import statsapi

        # MLB schedule dates are in US Eastern time, not UTC.
        # Using UTC after ~8pm ET would pull tomorrow's schedule.
        _ET = timezone(timedelta(hours=-4))  # EDT (summer)
        _FULL_TO_ABBR: dict[str, str] = {
            "Arizona Diamondbacks": "ARI",
            "Atlanta Braves": "ATL",
            "Baltimore Orioles": "BAL",
            "Boston Red Sox": "BOS",
            "Chicago Cubs": "CHC",
            "Chicago White Sox": "CWS",
            "Cincinnati Reds": "CIN",
            "Cleveland Guardians": "CLE",
            "Colorado Rockies": "COL",
            "Detroit Tigers": "DET",
            "Houston Astros": "HOU",
            "Kansas City Royals": "KC",
            "Los Angeles Angels": "LAA",
            "Los Angeles Dodgers": "LAD",
            "Miami Marlins": "MIA",
            "Milwaukee Brewers": "MIL",
            "Minnesota Twins": "MIN",
            "New York Mets": "NYM",
            "New York Yankees": "NYY",
            "Athletics": "ATH",
            "Oakland Athletics": "ATH",
            "Philadelphia Phillies": "PHI",
            "Pittsburgh Pirates": "PIT",
            "San Diego Padres": "SD",
            "San Francisco Giants": "SF",
            "Seattle Mariners": "SEA",
            "St. Louis Cardinals": "STL",
            "Tampa Bay Rays": "TB",
            "Texas Rangers": "TEX",
            "Toronto Blue Jays": "TOR",
            "Washington Nationals": "WSH",
        }
        try:
            from src.game_day import get_target_game_date

            _target = get_target_game_date()
        except Exception:
            _target = datetime.now(_ET).strftime("%Y-%m-%d")
        sched = statsapi.schedule(date=_target)
        for game in sched:
            home = _FULL_TO_ABBR.get(game.get("home_name", ""), "")
            away = _FULL_TO_ABBR.get(game.get("away_name", ""), "")
            if home and away:
                pairings[home] = away
                pairings[away] = home
            # Collect probable pitchers (last name, lowercased)
            for key in ("home_probable_pitcher", "away_probable_pitcher"):
                name = game.get(key, "") or ""
                if name and name != "TBD":
                    # Store last name lowercased for fuzzy matching
                    parts = name.strip().split()
                    if parts:
                        probable_starters.add(parts[-1].lower())
    except Exception:
        logger.warning("Failed to build schedule context from statsapi")
    return pairings, probable_starters


def compute_todays_actions(
    roster: pd.DataFrame,
    teams_playing: list[str] | None = None,
    matchup: dict | None = None,
    losing_cats: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute today's specific roster actions.

    Parameters
    ----------
    roster : pd.DataFrame
        Current roster with columns: name, team, positions, roster_slot,
        is_hitter, ip, mlb_id, player_id, sv, avg, obp, hr, rbi, sb, r,
        w, l, k, era, whip, status.
    teams_playing : list[str] | None
        Team abbreviations with games today. Fetched automatically if None.
    matchup : dict | None
        Current H2H matchup data (optional).
    losing_cats : list[str] | None
        Categories the user is currently losing (e.g. ["SB", "K"]).

    Returns
    -------
    list[dict]
        Up to 5 prioritised action dicts sorted by priority.
    """
    if roster is None or roster.empty:
        logger.warning("compute_todays_actions called with empty roster")
        return []

    # ------------------------------------------------------------------
    # Resolve today's schedule
    # ------------------------------------------------------------------
    if teams_playing is None:
        try:
            from src.weekly_report import get_todays_mlb_games

            teams_playing = get_todays_mlb_games()
        except Exception:
            logger.warning("Could not fetch today's MLB schedule — returning no actions")
            return []

    if not teams_playing:
        logger.info("No MLB games today — no actions to compute")
        return []

    teams_playing_upper = {t.upper() for t in teams_playing}
    losing_cats = [c.upper() for c in (losing_cats or [])]

    # Build schedule pairings and probable starters from MLB API
    _schedule_pairings, _probable_starters = _build_schedule_context()

    # Determine if we are winning ERA/WHIP (affects risky-start threshold)
    winning_era = matchup is not None and "ERA" not in losing_cats
    winning_whip = matchup is not None and "WHIP" not in losing_cats

    actions: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Helper: is a player's team playing today?
    # ------------------------------------------------------------------
    def _team_plays(team: Any) -> bool:
        return str(team).strip().upper() in teams_playing_upper

    # ------------------------------------------------------------------
    # Priority 1 — Off-day starters (swap opportunities)
    # ------------------------------------------------------------------
    for _, starter in roster.iterrows():
        if _is_bench(str(starter.get("roster_slot", "BN")), str(starter.get("status", ""))):
            continue
        if _team_plays(starter.get("team", "")):
            continue

        # This starter's team is NOT playing — look for a bench swap
        starter_slot = str(starter.get("roster_slot", "")).strip().upper()
        starter_name = starter.get("name", "Unknown")
        starter_team = str(starter.get("team", "")).strip().upper()

        best_replacement: str | None = None
        for _, bench_p in roster.iterrows():
            if not _is_bench(str(bench_p.get("roster_slot", ""))):
                continue
            if str(bench_p.get("roster_slot", "")).strip().upper() in ("IL", "IL+", "DL", "NA"):
                continue
            if not _team_plays(bench_p.get("team", "")):
                continue
            if _position_eligible(str(bench_p.get("positions", "")), starter_slot):
                best_replacement = bench_p.get("name", "Unknown")
                break

        swap_detail = (
            f"swap with {best_replacement} if possible"
            if best_replacement
            else "no eligible bench replacement available"
        )
        actions.append(
            {
                "priority": 1,
                "action_type": "bench",
                "player": starter_name,
                "team": starter_team,
                "detail": f"{starter_name} is OFF today ({starter_team} not playing) — {swap_detail}",
                "category_impact": [],
                "urgency": "high",
            }
        )

    # ------------------------------------------------------------------
    # Priority 2 — Bench players who should start (losing cat upside)
    # ------------------------------------------------------------------
    if losing_cats:
        for _, bench_p in roster.iterrows():
            if not _is_bench(str(bench_p.get("roster_slot", ""))):
                continue
            if str(bench_p.get("roster_slot", "")).strip().upper() in ("IL", "IL+", "DL", "NA"):
                continue
            if not _team_plays(bench_p.get("team", "")):
                continue

            player_name = bench_p.get("name", "Unknown")
            player_team = str(bench_p.get("team", "")).strip().upper()
            is_hitter = _is_hitter_row(bench_p)

            relevant_cats: list[str] = []
            for cat in losing_cats:
                stat_col = _CAT_STAT_MAP.get(cat)
                if stat_col is None:
                    continue
                # Filter: hitters help hitting cats, pitchers help pitching cats
                if is_hitter and cat not in _HITTING_CATS:
                    continue
                if not is_hitter and cat not in _PITCHING_CATS:
                    continue
                # Check if the player has relevant projected production
                val = bench_p.get(stat_col, 0)
                try:
                    val = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    val = 0.0

                # For rate stats, any pitcher on the roster is potentially helpful
                if cat in ("ERA", "WHIP"):
                    relevant_cats.append(cat)
                elif val > 0:
                    relevant_cats.append(cat)

            if relevant_cats:
                cat_str = "/".join(relevant_cats)
                actions.append(
                    {
                        "priority": 2,
                        "action_type": "start",
                        "player": player_name,
                        "team": player_team,
                        "detail": f"{player_name} on bench but {player_team} plays today — start for {cat_str} upside",
                        "category_impact": relevant_cats,
                        "urgency": "medium",
                    }
                )

    # ------------------------------------------------------------------
    # Priority 3 — SP matchup quality (only probable starters)
    # ------------------------------------------------------------------
    for _, player in roster.iterrows():
        if _is_bench(str(player.get("roster_slot", "BN")), str(player.get("status", ""))):
            continue
        if not _is_sp(str(player.get("positions", ""))):
            continue
        if not _team_plays(player.get("team", "")):
            continue

        player_name = player.get("name", "Unknown")
        player_team = str(player.get("team", "")).strip().upper()

        # Only show SP alerts for probable starters (from MLB schedule)
        if _probable_starters:
            name_parts = str(player_name).strip().split()
            last_name = name_parts[-1].lower() if name_parts else ""
            if last_name not in _probable_starters:
                continue  # Not a probable starter today — skip

        # Look up actual opponent from schedule pairings
        opp_wrc: float | None = None
        opp_label: str = _schedule_pairings.get(player_team, "")
        if opp_label:
            try:
                from src.game_day import get_team_strength

                strength = get_team_strength(opp_label)
                if strength and isinstance(strength, dict):
                    wrc = strength.get("wrc_plus")
                    if wrc is not None:
                        opp_wrc = float(wrc)
            except Exception:
                pass

        if opp_wrc is not None and opp_wrc > 110:
            # Tough matchup
            protect_note = ""
            if winning_era or winning_whip:
                cats_protecting = []
                if winning_era:
                    cats_protecting.append("ERA")
                if winning_whip:
                    cats_protecting.append("WHIP")
                protect_note = f" — protect {'/'.join(cats_protecting)} lead"
            actions.append(
                {
                    "priority": 3,
                    "action_type": "monitor",
                    "player": player_name,
                    "team": player_team,
                    "detail": f"{player_name} has a tough matchup vs {opp_label} ({opp_wrc:.0f} wRC+) — consider benching{protect_note}",
                    "category_impact": ["ERA", "WHIP"],
                    "urgency": "high",
                }
            )
        elif opp_wrc is not None and opp_wrc < 95:
            actions.append(
                {
                    "priority": 3,
                    "action_type": "start",
                    "player": player_name,
                    "team": player_team,
                    "detail": f"{player_name} vs {opp_label} ({opp_wrc:.0f} wRC+) — favorable matchup, start with confidence",
                    "category_impact": ["K", "W", "ERA"],
                    "urgency": "medium",
                }
            )
        else:
            # Unknown or neutral opponent — still note the start
            opp_note = f" vs {opp_label}" if opp_label else ""
            actions.append(
                {
                    "priority": 3,
                    "action_type": "start",
                    "player": player_name,
                    "team": player_team,
                    "detail": f"{player_name} starting today{opp_note} — standard matchup",
                    "category_impact": ["K", "W"],
                    "urgency": "medium",
                }
            )

    # ------------------------------------------------------------------
    # Priority 4 — Confirmed starters (low-urgency affirmations, max 2)
    # ------------------------------------------------------------------
    confirmed_count = 0
    for _, player in roster.iterrows():
        if confirmed_count >= 2:
            break
        if _is_bench(str(player.get("roster_slot", "BN")), str(player.get("status", ""))):
            continue
        if not _team_plays(player.get("team", "")):
            continue
        # Skip players already covered by higher-priority actions
        player_name = player.get("name", "Unknown")
        already_covered = any(a["player"] == player_name for a in actions)
        if already_covered:
            continue

        player_team = str(player.get("team", "")).strip().upper()
        actions.append(
            {
                "priority": 4,
                "action_type": "start",
                "player": player_name,
                "team": player_team,
                "detail": f"{player_name} playing today — locked in",
                "category_impact": [],
                "urgency": "low",
            }
        )
        confirmed_count += 1

    # ------------------------------------------------------------------
    # Sort by priority, trim to 5
    # ------------------------------------------------------------------
    actions.sort(key=lambda a: a["priority"])
    return actions[:5]
