"""Contextual draft factors — closer hierarchy, platoon risk, lineup protection,
schedule strength, and contract year boost.

Provides five standalone functions that quantify contextual adjustments for
draft valuation.  Each returns a multiplicative factor or additive SGP bonus
that the draft engine can layer on top of the base ``pick_score``.

Functions
---------
detect_closer_role
    Classify a relief pitcher's closer role and assign a draft bonus.
compute_platoon_risk
    Estimate plate-appearance discount from platoon disadvantage.
compute_lineup_protection
    Estimate SGP bonus from likely batting-order position.
compute_schedule_strength
    Division park-factor schedule-strength multiplier.
contract_year_boost
    Small premium for hitters in a contract (walk) year.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

# Closer-role SV thresholds and associated bonuses
_CLOSER_SV_THRESHOLD: int = 25
_SETUP_SV_THRESHOLD: int = 15
_COMMITTEE_SV_THRESHOLD: int = 5

_CLOSER_CONFIDENCE: float = 0.9
_SETUP_CONFIDENCE: float = 0.6
_COMMITTEE_CONFIDENCE: float = 0.4
_MIDDLE_CONFIDENCE: float = 0.0

# Platoon factors — research constants from *The Book* (Tango/Lichtman/Dolphin)
_PLATOON_FACTOR_LHB: float = 0.90  # LHB vs LHP: ~40 fewer PA, ~15 lower wRC+
_PLATOON_FACTOR_RHB: float = 0.96  # RHB vs RHP: slight disadvantage
_PLATOON_FACTOR_SWITCH: float = 1.0  # Switch-hitter: no platoon risk
_PLATOON_FACTOR_DEFAULT: float = 1.0  # Unknown handedness: no adjustment

# Lineup slot PA/game — historical MLB averages
# (matches src/engine/context/matchup.py LINEUP_SLOT_PA)
LINEUP_SLOT_PA: dict[int, float] = {
    1: 4.65,
    2: 4.55,
    3: 4.45,
    4: 4.35,
    5: 4.25,
    6: 4.15,
    7: 4.05,
    8: 3.95,
    9: 3.85,
}

# Baseline PA/game for an "average" starter (slot 5-6 territory)
_BASELINE_PA_PER_GAME: float = 4.30

# Approximate games in a season used for PA → SGP conversion
_GAMES_PER_SEASON: int = 162

# SGP per extra PA (rough composite across R, HR, RBI, SB, AVG, OBP).
# Derived empirically: each PA is worth ~0.0018 composite SGP for an
# average hitter.  We use this to translate the PA delta into SGP bonus.
_SGP_PER_EXTRA_PA: float = 0.0018

# Leadoff archetype thresholds
_LEADOFF_SB_MIN: int = 25
_LEADOFF_OBP_MIN: float = 0.340

# Cleanup / star archetype thresholds
_STAR_HR_MIN: int = 30
_STAR_RBI_MIN: int = 90

# MLB Division membership (stable for 2026 — 15 AL / 15 NL)
MLB_DIVISIONS: dict[str, list[str]] = {
    "AL_East": ["NYY", "BOS", "TOR", "BAL", "TB"],
    "AL_Central": ["CLE", "MIN", "DET", "CWS", "KC"],
    "AL_West": ["HOU", "TEX", "SEA", "LAA", "OAK"],
    "NL_East": ["ATL", "NYM", "PHI", "MIA", "WSH"],
    "NL_Central": ["MIL", "CHC", "STL", "PIT", "CIN"],
    "NL_West": ["LAD", "SD", "ARI", "SF", "COL"],
}

# Fraction of the season spent playing divisional opponents (~76 / 162)
_DIVISION_GAME_FRACTION: float = 0.46

# Contract-year boost — FanGraphs 2019 research: ~2 % for hitters
_CONTRACT_YEAR_HITTER_BOOST: float = 1.02

# ── Team alias normalization ──────────────────────────────────────────

_TEAM_ALIASES: dict[str, str] = {
    "WSN": "WSH",
    "AZ": "ARI",
    "CHW": "CWS",
    "TBR": "TB",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
}


def _normalize_team(team: str) -> str:
    """Normalize a team abbreviation to the canonical form used in MLB_DIVISIONS."""
    if not isinstance(team, str):
        return ""
    t = team.strip().upper()
    return _TEAM_ALIASES.get(t, t)


# ── Reverse lookup: team → division ──────────────────────────────────

_TEAM_TO_DIVISION: dict[str, str] = {}
for _div, _teams in MLB_DIVISIONS.items():
    for _tm in _teams:
        _TEAM_TO_DIVISION[_tm] = _div


# =====================================================================
# Function 1: detect_closer_role
# =====================================================================


def detect_closer_role(player: pd.Series) -> dict:
    """Classify a relief pitcher's closer role from projected saves.

    Uses the player's projected SV count (from FanGraphs / blended
    projections) to bucket into one of four roles.  Only pitchers with
    ``is_hitter == False`` and a position string containing ``"RP"`` are
    eligible; everyone else is classified as ``"Middle"`` with zero bonus.

    Thresholds
    ----------
    * SV >= 25 → **Closer** (confidence 0.9, bonus scales linearly
      from +1.5 at SV=25 to +2.0 at SV>=35, capped at 2.0)
    * SV 15-24 → **Setup** (confidence 0.6, bonus +0.5)
    * SV 5-14  → **Committee** (confidence 0.4, bonus +0.3)
    * SV < 5 or not RP → **Middle** (confidence 0.0, bonus 0.0)

    Parameters
    ----------
    player : pd.Series
        Must contain at least ``sv`` (projected saves).  Optionally
        ``is_hitter`` and ``positions`` for eligibility filtering.

    Returns
    -------
    dict
        ``{role: str, confidence: float, draft_bonus: float}``
    """
    # Default result for non-RP or non-pitcher
    middle = {"role": "Middle", "confidence": _MIDDLE_CONFIDENCE, "draft_bonus": 0.0}

    # Check pitcher eligibility
    is_hitter = player.get("is_hitter", None)
    if is_hitter is True or is_hitter == 1:
        return middle

    positions = str(player.get("positions", ""))
    if "RP" not in positions.upper() and is_hitter is not False and is_hitter != 0:
        # If we can't confirm this is a pitcher AND can't find RP, bail
        return middle

    sv = float(player.get("sv", 0) or 0)

    if sv >= _CLOSER_SV_THRESHOLD:
        # Linear bonus: 1.5 at SV=25, increasing by 0.05 per SV up to 2.0
        bonus = np.clip(1.5 + (sv - 25) * 0.05, 1.5, 2.0)
        return {
            "role": "Closer",
            "confidence": _CLOSER_CONFIDENCE,
            "draft_bonus": float(bonus),
        }
    elif sv >= _SETUP_SV_THRESHOLD:
        return {
            "role": "Setup",
            "confidence": _SETUP_CONFIDENCE,
            "draft_bonus": 0.5,
        }
    elif sv >= _COMMITTEE_SV_THRESHOLD:
        return {
            "role": "Committee",
            "confidence": _COMMITTEE_CONFIDENCE,
            "draft_bonus": 0.3,
        }
    else:
        return middle


# =====================================================================
# Function 2: compute_platoon_risk
# =====================================================================


def compute_platoon_risk(player_bats: str) -> float:
    """Estimate PA discount factor from platoon disadvantage.

    Research constants are drawn from *The Book* (Tango / Lichtman /
    Dolphin, 2007).

    Parameters
    ----------
    player_bats : str
        Batting side: ``"L"`` (left), ``"R"`` (right), ``"S"``/``"B"``
        (switch), or empty / ``None`` for unknown.

    Returns
    -------
    float
        Multiplicative factor in ``[0.85, 1.0]``.  Values less than 1.0
        indicate an expected PA loss from platoon usage.
    """
    if not player_bats or not isinstance(player_bats, str):
        return _PLATOON_FACTOR_DEFAULT

    bats = player_bats.strip().upper()

    if bats == "L":
        return _PLATOON_FACTOR_LHB
    elif bats == "R":
        return _PLATOON_FACTOR_RHB
    elif bats in ("S", "B"):
        return _PLATOON_FACTOR_SWITCH
    else:
        return _PLATOON_FACTOR_DEFAULT


# =====================================================================
# Function 3: compute_lineup_protection
# =====================================================================


def compute_lineup_protection(player: pd.Series) -> float:
    """Estimate SGP bonus from likely batting-order position.

    Hitters projected into premium lineup slots (leadoff, cleanup) get
    more PA per game than a replacement-level starter batting 7th-9th.
    The delta in PA translates into a small composite SGP bonus.

    Archetypes
    ----------
    * **Leadoff** — SB >= 25 AND OBP >= .340 → slot 1 (4.65 PA/G)
    * **Star / Cleanup** — HR >= 30 OR RBI >= 90 → slot 4 (4.35 PA/G)
    * **Regular** — everyone else → baseline (4.30 PA/G)
    * **Pitcher** — ``is_hitter == False`` → 0.0

    The SGP bonus is::

        extra_pa = (slot_pa - baseline_pa) * 162
        bonus    = extra_pa * SGP_per_extra_PA

    Parameters
    ----------
    player : pd.Series
        Must contain hitting stats (``hr``, ``rbi``, ``sb``, ``obp``)
        and ``is_hitter``.

    Returns
    -------
    float
        Additive SGP bonus >= 0.0.
    """
    is_hitter = player.get("is_hitter", None)
    if is_hitter is False or is_hitter == 0:
        return 0.0

    sb = float(player.get("sb", 0) or 0)
    obp = float(player.get("obp", 0) or 0)
    hr = float(player.get("hr", 0) or 0)
    rbi = float(player.get("rbi", 0) or 0)

    # Determine which lineup slot this player likely occupies
    if sb >= _LEADOFF_SB_MIN and obp >= _LEADOFF_OBP_MIN:
        slot_pa = LINEUP_SLOT_PA[1]  # leadoff
    elif hr >= _STAR_HR_MIN or rbi >= _STAR_RBI_MIN:
        slot_pa = LINEUP_SLOT_PA[4]  # cleanup
    else:
        slot_pa = _BASELINE_PA_PER_GAME  # regular

    extra_pa = (slot_pa - _BASELINE_PA_PER_GAME) * _GAMES_PER_SEASON
    if extra_pa <= 0:
        return 0.0

    bonus = extra_pa * _SGP_PER_EXTRA_PA
    return float(bonus)


# =====================================================================
# Function 4: compute_schedule_strength
# =====================================================================


def compute_schedule_strength(
    player_team: str,
    park_factors: dict[str, float],
) -> float:
    """Schedule-based multiplicative adjustment from division park factors.

    Teams play roughly 76 games against divisional opponents (46 % of the
    162-game season).  We compute the mean park factor of the other four
    division mates and dampen it to reflect the actual schedule share::

        factor = 1.0 + (avg_opponent_pf - 1.0) * 0.46

    Parameters
    ----------
    player_team : str
        Team abbreviation (e.g., ``"COL"``, ``"NYY"``).
    park_factors : dict[str, float]
        Mapping of team abbreviation → park factor (values > 1.0 are
        hitter-friendly).

    Returns
    -------
    float
        Multiplicative factor, typically in ``[0.95, 1.05]``.  Returns
        ``1.0`` when the team or its division cannot be resolved.
    """
    if not player_team or not isinstance(player_team, str):
        return 1.0
    if not park_factors:
        return 1.0

    team = _normalize_team(player_team)
    division = _TEAM_TO_DIVISION.get(team)
    if division is None:
        return 1.0

    division_mates = [t for t in MLB_DIVISIONS[division] if t != team]
    if not division_mates:
        return 1.0

    # Collect park factors for division opponents
    pf_values = []
    for mate in division_mates:
        pf = park_factors.get(mate)
        if pf is None:
            # Try alias resolution for the park_factors dict
            for alias, canonical in _TEAM_ALIASES.items():
                if canonical == mate and alias in park_factors:
                    pf = park_factors[alias]
                    break
            # Also try reverse: mate might be stored under a different key
            if pf is None:
                for alias, canonical in _TEAM_ALIASES.items():
                    if alias == mate and canonical in park_factors:
                        pf = park_factors[canonical]
                        break
        if pf is not None:
            pf_values.append(pf)

    if not pf_values:
        return 1.0

    avg_pf = sum(pf_values) / len(pf_values)
    factor = 1.0 + (avg_pf - 1.0) * _DIVISION_GAME_FRACTION
    return float(factor)


# =====================================================================
# Function 5: contract_year_boost
# =====================================================================


def contract_year_boost(is_contract_year: bool, is_hitter: bool) -> float:
    """Small premium for hitters in a contract (walk) year.

    FanGraphs research (2019) found an approximate 2 % performance
    boost for hitters in their final year before free agency.  No
    statistically significant effect has been found for pitchers.

    Parameters
    ----------
    is_contract_year : bool
        ``True`` if the player is in the final year of his contract.
    is_hitter : bool
        ``True`` for position players; ``False`` for pitchers.

    Returns
    -------
    float
        Multiplicative factor: ``1.02`` for a contract-year hitter,
        ``1.0`` otherwise.
    """
    if is_contract_year and is_hitter:
        return _CONTRACT_YEAR_HITTER_BOOST
    return 1.0
