"""Layer-0 display G-score (Rosenof, arXiv 2307.02188v4) — a per-player, per-category
interpretable VALUE/RANKING scalar. DISPLAY ONLY: it gives the UI a readable "how good is
this player in this category" number; it is NEVER the win/tie/loss probability engine (that
is the heteroscedastic NB/Skellam in the proxy/deep tiers — research-validation gap G5).

    G = (mu - mu_league) / sqrt(sigma2_league + kappa * tau2),   kappa = 2N / (2N - 1)

mu/tau2 come from a slice-1 CategoryPosterior; mu_league/sigma2_league/N are league context.
Inverse cats (L/ERA/WHIP) flip the numerator so good play is always positive. Reduces to the
classic z-score when tau2 -> 0 (the paper's |W|=1 special case). Pure; never raises; a zero
denominator returns 0.0 (no defined signal, not a divide error).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.valuation import LeagueConfig


def _f(value, default: float = 0.0) -> float:
    """NaN/inf/None-safe float coercion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


@dataclass(frozen=True)
class LeagueContext:
    """Per-category league context for the G-score denominator/numerator.
    `means`/`sds` are per-category {CAT: value}; `n_slots` is the number of roster slots
    contributing to a category total (hitters vs pitchers), driving the kappa correction."""

    means: dict = field(default_factory=dict)
    sds: dict = field(default_factory=dict)
    hitter_slots: int = 10  # FourzynBurn active hitters: C/1B/2B/3B/SS/3OF/2Util
    pitcher_slots: int = 8  # FourzynBurn active pitchers: 2SP/2RP/4P


def kappa(n_slots: int) -> float:
    """Small-sample correction 2N/(2N-1) for a head-to-head difference of two N-player sums.
    >1, decreasing to 1 as N grows. N<=0 -> 1.0 (no correction)."""
    n = int(n_slots)
    if n <= 0:
        return 1.0
    return (2.0 * n) / (2.0 * n - 1.0)


def category_gscore(
    mean: float, tau2: float, league_mean: float, league_sd: float, n_slots: int, inverse: bool = False
) -> float:
    """Rosenof G for one category: (mu - mu_league)/sqrt(sigma2_league + kappa*tau2).
    Inverse cats flip the numerator (lower is better -> positive G). Zero denominator -> 0.0."""
    mu = _f(mean)
    t2 = max(0.0, _f(tau2))
    lm = _f(league_mean)
    lsd = _f(league_sd)
    denom2 = lsd * lsd + kappa(n_slots) * t2
    if denom2 <= 0.0:
        return 0.0
    num = (lm - mu) if inverse else (mu - lm)
    return float(num / math.sqrt(denom2))


def player_gscore(posteriors: dict, context: LeagueContext, config: LeagueConfig | None = None, detail: bool = False):
    """Aggregate display value: sum of signed per-category G across the player's posteriors
    (inverse cats flipped to positive-is-good). `posteriors` is the slice-1 player_posteriors
    output {CAT: CategoryPosterior}. Returns a float total, or {'total', 'per_category'} when
    detail=True. Never raises."""
    cfg = config or LeagueConfig()
    inverse = cfg.inverse_stats
    hitting = set(cfg.hitting_categories)
    per_cat: dict[str, float] = {}
    for cat, post in posteriors.items():
        n_slots = context.hitter_slots if cat in hitting else context.pitcher_slots
        per_cat[cat] = category_gscore(
            mean=post.mean,
            tau2=post.tau2,
            league_mean=context.means.get(cat, 0.0),
            league_sd=context.sds.get(cat, 0.0),
            n_slots=n_slots,
            inverse=cat in inverse,
        )
    total = float(sum(per_cat.values()))
    return {"total": total, "per_category": per_cat} if detail else total
