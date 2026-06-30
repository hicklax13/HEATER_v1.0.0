"""Unified Layer-0 player model facade (Advanced Value Engine, Phase 1).

Assembles the single source-of-truth Layer-0 output per the spec §6 contract:
    (player_row, league_context) -> PlayerModel{posteriors, availability, display_g_score}

Composes the locked slice APIs (posterior.player_posteriors, availability.availability_survival,
gscore.player_gscore) and adds the one missing piece — build_league_context, the per-category
league baseline the G-score standardizes against, computed at the posterior's per-week/rate scale.
Pure; no DB / network; never raises. Every later layer and surface consumes a PlayerModel; none
re-derives a value (spec §5 single source of truth).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from src.player_model.availability import AvailabilitySurvival, availability_survival
from src.player_model.gscore import LeagueContext, player_gscore
from src.player_model.posterior import player_posteriors
from src.valuation import LeagueConfig

# Volume floors that define "fantasy-relevant" for the league baseline (excludes AAA scrubs).
# Calibratable (slice 5). A hitter needs projected PA, a pitcher projected IP, to count.
_MIN_CONTEXT_HITTER_PA: float = 200.0
_MIN_CONTEXT_PITCHER_IP: float = 30.0


def _f(value, default: float = 0.0) -> float:
    """NaN/inf/None-safe float coercion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _col_per_week(frame: pd.DataFrame, col: str, weeks: float) -> pd.Series:
    vals = pd.to_numeric(frame.get(col), errors="coerce") if col in frame else pd.Series(dtype=float)
    return vals.dropna() / weeks if weeks > 0 else vals.dropna()


def build_league_context(
    pool: pd.DataFrame, config: LeagueConfig | None = None, weeks: float | None = None
) -> LeagueContext:
    """Per-category league mean + spread from the fantasy-relevant slice of the pool, at the
    posterior's scale (per-week for counting cats, the rate for rate cats). Empty/degenerate ->
    zero means/sds (never raises).

    NOTE (display-only G-score scale): for RATE cats the resulting `sds` are season-talent
    spreads, while the slice-1 posterior `tau2` is the weekly beta-binomial outcome variance,
    which is the larger term — so the rate-cat G-score denominator is tau2-dominated and the
    rate-cat G is a per-week RANKING signal, not a calibrated season z-score. This is intentional
    and honest (a 20-AB week is genuinely noisier than the cross-player talent spread); calibrated
    win/tie/loss probability is the NB/Skellam tiers' job, not the display G (gap G5)."""
    cfg = config or LeagueConfig()
    weeks = float(weeks) if (weeks is not None and weeks > 0) else float(cfg.season_weeks)
    means: dict[str, float] = {}
    sds: dict[str, float] = {}

    if pool is None or len(pool) == 0:
        return LeagueContext(means={c: 0.0 for c in cfg.all_categories}, sds={c: 0.0 for c in cfg.all_categories})

    is_hit = (
        pd.to_numeric(pool["is_hitter"] if "is_hitter" in pool else pd.Series([1] * len(pool)), errors="coerce").fillna(
            1
        )
        >= 0.5
    )
    pa = pd.to_numeric(pool["pa"] if "pa" in pool else pd.Series([0] * len(pool)), errors="coerce").fillna(0)
    ip = pd.to_numeric(pool["ip"] if "ip" in pool else pd.Series([0] * len(pool)), errors="coerce").fillna(0)
    hitters = pool[is_hit & (pa >= _MIN_CONTEXT_HITTER_PA)]
    pitchers = pool[(~is_hit) & (ip >= _MIN_CONTEXT_PITCHER_IP)]

    hitting = set(cfg.hitting_categories)
    for cat in cfg.all_categories:
        frame = hitters if cat in hitting else pitchers
        col = cfg.STAT_MAP.get(cat, cat.lower())
        if cat in cfg.rate_stats:
            series = pd.to_numeric(frame.get(col), errors="coerce").dropna() if col in frame else pd.Series(dtype=float)
            series = series[series > 0]  # rate of 0 means "no data", not a true 0.000 talent
        else:
            series = _col_per_week(frame, col, weeks)
        means[cat] = float(series.mean()) if len(series) else 0.0
        sds[cat] = float(series.std(ddof=0)) if len(series) > 1 else 0.0

    return LeagueContext(means=means, sds=sds)


@dataclass(frozen=True)
class PlayerModel:
    """The single Layer-0 output every later layer / surface consumes (spec §6). `posteriors`
    is {CAT: CategoryPosterior} for the player's relevant cats; `availability` is the survival
    distribution; `g_score`/`g_by_category` are the display-only Rosenof value scalars."""

    player_id: int
    name: str
    is_hitter: bool
    posteriors: dict = field(default_factory=dict)
    availability: AvailabilitySurvival | None = None
    g_score: float = 0.0
    g_by_category: dict = field(default_factory=dict)


def build_player_model(
    row,
    league_context: LeagueContext,
    config: LeagueConfig | None = None,
    weeks: float | None = None,
    status=None,
    expected_return_days=None,
) -> PlayerModel:
    """Assemble the Layer-0 PlayerModel for one pool row: posteriors (slice 1) + availability
    (slice 2) + display G-score (slice 3). Never raises."""
    cfg = config or LeagueConfig()
    posteriors = player_posteriors(row, cfg, weeks)
    avail = availability_survival(row, status=status, expected_return_days=expected_return_days, weeks_remaining=weeks)
    g = player_gscore(posteriors, league_context, cfg, detail=True)
    is_hit = _f(row.get("is_hitter") if hasattr(row, "get") else 1, default=1.0) >= 0.5
    name = row.get("name") if hasattr(row, "get") else ""
    return PlayerModel(
        player_id=int(_f(row.get("player_id") if hasattr(row, "get") else 0)),
        name=str(name) if name is not None else "",
        is_hitter=bool(is_hit),
        posteriors=posteriors,
        availability=avail,
        g_score=float(g["total"]),
        g_by_category=g["per_category"],
    )


def build_player_models(
    pool: pd.DataFrame,
    config: LeagueConfig | None = None,
    weeks: float | None = None,
    status_map: dict | None = None,
    return_days_map: dict | None = None,
) -> dict[int, PlayerModel]:
    """Batch: build the league context once, then a PlayerModel per pool row (keyed by
    player_id). status_map/return_days_map are optional {player_id: value} overrides
    (default: active). Never raises; a row that fails is skipped."""
    cfg = config or LeagueConfig()
    status_map = status_map or {}
    return_days_map = return_days_map or {}
    ctx = build_league_context(pool, cfg, weeks)
    out: dict[int, PlayerModel] = {}
    if pool is None or len(pool) == 0:
        return out
    for i in range(len(pool)):
        row = pool.iloc[i]
        pid = int(_f(row.get("player_id")))
        try:
            out[pid] = build_player_model(
                row,
                ctx,
                cfg,
                weeks=weeks,
                status=status_map.get(pid),
                expected_return_days=return_days_map.get(pid),
            )
        except Exception:
            continue
    return out
