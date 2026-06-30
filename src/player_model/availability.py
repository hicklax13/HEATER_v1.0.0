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

import numpy as np

from src.il_manager import classify_il_type, estimate_il_duration
from src.in_season import _il_weight_from_status, _return_date_weight
from src.injury_model import age_risk_adjustment

# Per-week probability of a NEW availability loss, seeded from chronic durability:
# a fully-healthy player (chronic=1) carries a small residual hazard; a fragile player more.
_BASE_WEEKLY_HAZARD: float = 0.02  # calibratable against HEATER injury logs (follow-on)


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


def _normalize_status(status) -> tuple[str, str | None]:
    """Return (normalized_status, il_type). il_type is None when the player is active.
    Normalized status is one of ACTIVE/DTD/IL10/IL15/IL60."""
    if status is None or (isinstance(status, float) and math.isnan(status)):
        return "ACTIVE", None
    raw = str(status).strip()
    if not raw:
        return "ACTIVE", None
    il_type = classify_il_type(raw)  # -> IL10/IL15/IL60/DTD or None
    if il_type is None:
        return "ACTIVE", None
    if il_type == "DTD":
        return "DTD", None
    return il_type, il_type


def _is_hitter(row) -> bool:
    raw = row.get("is_hitter") if hasattr(row, "get") else row["is_hitter"]
    return _f(raw, default=1.0) >= 0.5


def availability_survival(
    row,
    status=None,
    expected_return_days=None,
    weeks_remaining: float | None = None,
) -> AvailabilitySurvival:
    """Build the rest-of-season availability distribution for one pool row + current status.
    `weeks_remaining` defaults to LeagueConfig.season_weeks. Never raises."""
    from src.valuation import LeagueConfig

    weeks = _f(weeks_remaining) if weeks_remaining else float(LeagueConfig().season_weeks)
    weeks = max(0.0, weeks)
    is_hit = _is_hitter(row)
    position = row.get("positions") if hasattr(row, "get") else None
    chronic = chronic_availability(
        row.get("health_score") if hasattr(row, "get") else None,
        row.get("age") if hasattr(row, "get") else None,
        is_pitcher=not is_hit,
        position=position,
    )

    norm_status, il_type = _normalize_status(status)

    # Near-term status weight: the return-date curve wins when known, else status default.
    rd = _return_date_weight(expected_return_days) if expected_return_days is not None else None
    status_weight = rd if rd is not None else _il_weight_from_status(norm_status, expected_return_days)
    status_weight = min(max(_f(status_weight, default=1.0), 0.0), 1.0)

    # Expected weeks fully sidelined now (IL only; DTD/active -> 0).
    pos_str = position if isinstance(position, str) else ""
    il_weeks_out = estimate_il_duration(il_type, pos_str) if il_type in {"IL10", "IL15", "IL60"} else 0.0
    il_weeks_out = min(max(_f(il_weeks_out), 0.0), weeks)

    active_window = max(0.0, weeks - il_weeks_out)
    expected_active_weeks = active_window * chronic
    expected_active_fraction = (expected_active_weeks / weeks) if weeks > 0 else 0.0
    weekly_hazard = min(max(_BASE_WEEKLY_HAZARD * (1.0 - chronic) / max(1.0 - 0.85, 1e-6), 0.0), 1.0)

    return AvailabilitySurvival(
        player_id=int(_f(row.get("player_id") if hasattr(row, "get") else 0)),
        is_hitter=is_hit,
        weeks_remaining=weeks,
        status=norm_status,
        status_weight=status_weight,
        chronic_avail=chronic,
        il_weeks_out=il_weeks_out,
        expected_active_weeks=expected_active_weeks,
        expected_active_fraction=expected_active_fraction,
        weekly_hazard=weekly_hazard,
    )


def sample_active_weeks(survival: AvailabilitySurvival, rng, n_samples: int = 1) -> np.ndarray:
    """Draw `n_samples` integer active-week realizations for the season MC. The current IL
    window is fully out; each remaining week is active ~ Bernoulli(chronic_avail). Returns a
    length-n_samples int array in [0, floor(active_window)]. Deterministic given `rng`."""
    weeks = max(0.0, survival.weeks_remaining)
    active_window = int(math.floor(max(0.0, weeks - survival.il_weeks_out)))
    n = max(1, int(n_samples))
    if active_window <= 0:
        return np.zeros(n, dtype=int)
    p = min(max(survival.chronic_avail, 0.0), 1.0)
    return rng.binomial(active_window, p, size=n).astype(int)
