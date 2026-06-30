"""Layer-0 availability survival — per-player games/IP-remaining distribution (gap G1).

The single largest unmodeled rest-of-season value swing, with no academic prior — built
in-house by COMPOSING HEATER's injury signals:
  * chronic durability: pool health_score x injury_model.age_risk_adjustment;
  * acute status: il_manager.classify_il_type / estimate_il_duration +
    in_season._il_weight_from_status / _return_date_weight.

Produces an AvailabilitySurvival whose sample_active_weeks(rng, n) draws integer active-week
realizations so the Layer-2 MC propagates availability uncertainty per replicate (rather than
applying a single deterministic playing-time scalar). Pure; no DB / network; never raises.

Hazard/duration seeds are in-house first guesses, calibratable against HEATER injury logs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.injury_model import age_risk_adjustment


def _f(value, default: float = 0.0) -> float:
    """NaN/inf/None-safe float coercion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


@dataclass(frozen=True)
class AvailabilitySurvival:
    """Per-player rest-of-season availability. `chronic_avail` is the per-week available
    probability absent a current injury; `status_weight` is the near-term availability from
    the current status/return-date; `il_weeks_out` is the expected weeks fully sidelined now;
    `expected_active_weeks` = E[active weeks over `weeks_remaining`]; `weekly_hazard` is the
    per-week probability of a NEW availability loss. `sample_active_weeks` draws realizations."""

    player_id: int
    is_hitter: bool
    weeks_remaining: float
    status: str
    status_weight: float
    chronic_avail: float
    il_weeks_out: float
    expected_active_weeks: float
    expected_active_fraction: float
    weekly_hazard: float


def chronic_availability(health_score, age, is_pitcher: bool, position) -> float:
    """Long-run per-week availability absent a current injury: pool health_score tempered
    by age/position risk. Clamped to [0, 1]. NaN-safe (missing health -> 0.85 league avg)."""
    hs = _f(health_score, default=0.85)
    hs = min(max(hs, 0.0), 1.0)
    pos = position if isinstance(position, str) else ""
    try:
        risk = age_risk_adjustment(int(_f(age, default=28)), bool(is_pitcher), pos or None)
    except Exception:
        risk = 1.0
    risk = min(max(_f(risk, default=1.0), 0.0), 1.0)
    return float(min(max(hs * risk, 0.0), 1.0))
