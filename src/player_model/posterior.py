"""Layer-0 posterior variance core — per-player, per-category {mean, sigma2, tau2}.

Generalizes the per-team Kalman stabilization-shrinkage in
src/engine/output/weekly_matrix.build_team_kalman_variances to PER-PLAYER, and adds:
  * an irreducible projection-error floor so true-talent bands never falsely vanish
    (research-validation gap G3 — "champ/playoff bands are dishonestly narrow" without it);
  * explicit week-to-week MARGIN descriptors (NB for counting cats, beta-binomial for
    AVG/OBP, ratio-normal for ERA/WHIP) so the Layer-1 proxy and the Layer-2 deep MC
    sample the SAME margins (research-validation Tier-1 correction #2).

sigma2 = between-player TRUE-TALENT (epistemic) variance: how unsure we are of the
         player's true rate. Shrinks with the player's own YTD sample, never to zero.
tau2   = week-to-week ALEATORY variance: how much a single week's outcome scatters
         around the true rate. Counting -> NB; AVG/OBP -> beta-binomial; ERA/WHIP -> ratio-normal.

All per-category SEED constants below are documented harness-calibratable (slice 5 tunes
them via SURE on the game-log backtest). Pure functions over a pool row (pd.Series);
no DB / network. Never raises on missing columns or NaN — degrades to max-uncertainty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from src.valuation import LeagueConfig

# ── Seeded constants (HARNESS-CALIBRATABLE — slice 5) ───────────────────────────
# Per-team analog: weekly_matrix._KALMAN_TALENT_CV / _KALMAN_STABILIZE_PA/IP. Per-PLAYER
# stabilization scales are smaller than per-team (one player's season ~600 PA vs a roster's
# ~6000). Seeds below are deliberate, sensible, and replaced by slice-5 SURE calibration.
_TALENT_CV: float = 0.40  # reducible true-talent spread coefficient (counting cats)
_PROJ_FLOOR_CV: float = 0.10  # irreducible projection-error CV — NEVER vanishes (gap G3)
_COUNTING_ABS_FLOOR: float = 1e-4  # minimum absolute floor for zero-mean / NaN-mean rows (gap G3)
_STABILIZE_PA: float = 1200.0  # ytd_pa at which true-talent shrink == 0.5 (hitter cats)
_STABILIZE_IP: float = 400.0  # ytd_ip at which true-talent shrink == 0.5 (pitcher cats)

# Week-to-week overdispersion for counting cats: tau2 = phi * mu_week (phi=1 => Poisson).
# SB/SV are role-/opportunity-driven and most overdispersed; K is steadiest. (FanGraphs
# NB run-distribution work: counting outcomes are ~1.3-2x overdispersed vs Poisson.)
_COUNTING_OVERDISPERSION: dict[str, float] = {
    "R": 1.5,
    "HR": 1.6,
    "RBI": 1.5,
    "SB": 1.8,
    "W": 1.4,
    "L": 1.4,
    "SV": 2.0,
    "K": 1.3,
}

# Rate cats: per-week ABSOLUTE std at a reference weekly volume (seeded from
# scenario_generator._RATE_STD). tau2 scales inversely with the player's weekly volume.
_RATE_TALENT_STD: dict[str, float] = {"AVG": 0.020, "OBP": 0.022, "ERA": 0.60, "WHIP": 0.09}
_RATE_FLOOR_STD: dict[str, float] = {"AVG": 0.012, "OBP": 0.013, "ERA": 0.35, "WHIP": 0.05}
_RATE_WEEK_STD: dict[str, float] = {"AVG": 0.022, "OBP": 0.025, "ERA": 0.70, "WHIP": 0.10}
_RATE_REF_AB: float = 20.0  # reference weekly AB the AVG/OBP week-std is quoted at
_RATE_REF_IP: float = 20.0  # reference weekly IP the ERA/WHIP week-std is quoted at
_MIN_WEEK_VOL: float = 1.0  # floor on weekly volume to avoid division blow-ups

_HITTER_VOL_COL = "ab"  # rate_prop weekly-volume source column (season AB)
_PITCHER_VOL_COL = "ip"  # rate_ratio weekly-volume source column (season IP)


@dataclass(frozen=True)
class CategoryPosterior:
    """Per-player, per-category posterior. `mean` is per-WEEK for counting cats and the
    rate value for rate cats. `sigma2` = between-player true-talent variance; `tau2` =
    week-to-week outcome variance. `margin` reconstructs the sampling distribution:
      counting  -> {"dist": "nb", "mean": mu_week, "r": dispersion}      (r=inf => Poisson)
      rate_prop -> {"dist": "beta_binomial", "theta": rate, "n": v_week, "rho": icc}
      rate_ratio-> {"dist": "ratio_normal", "mean": rate, "std_week": sqrt(tau2)}
    """

    category: str
    kind: str
    mean: float
    sigma2: float
    tau2: float
    margin: dict = field(default_factory=dict)


def classify_kind(category: str, config: LeagueConfig | None = None) -> str:
    """Map an UPPERCASE category to its variance kind: 'counting' | 'rate_prop' | 'rate_ratio'.
    AVG/OBP are bounded proportions (beta-binomial); ERA/WHIP are events-per-IP ratios."""
    cfg = config or LeagueConfig()
    if category not in cfg.rate_stats:
        return "counting"
    return "rate_prop" if category in {"AVG", "OBP"} else "rate_ratio"


def _f(value, default: float = 0.0) -> float:
    """NaN/inf/None-safe float coercion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return out


def _mean_for(row, category: str, kind: str, config: LeagueConfig, weeks: float) -> float:
    col = config.STAT_MAP[category]
    raw = _f(row.get(col)) if hasattr(row, "get") else _f(row[col])
    if kind == "counting":
        return raw / weeks if weeks > 0 else 0.0
    return raw  # rate cats: the rate itself


def _shrink(n: float, is_hitter: bool) -> float:
    """True-talent shrink factor S0/(S0+n): 1.0 at n=0 (no info), -> 0 with large samples.
    Generalizes weekly_matrix.build_team_kalman_variances' per-team shrink to per-player."""
    s0 = _STABILIZE_PA if is_hitter else _STABILIZE_IP
    n = max(0.0, _f(n))
    return s0 / (s0 + n) if (s0 + n) > 0 else 1.0


def between_player_sigma2(mean: float, kind: str, category: str, n: float, is_hitter: bool) -> float:
    """Between-player (epistemic) true-talent variance. Reducible part shrinks with the
    player's own sample `n` (ytd_pa for hitters / ytd_ip for pitchers); the floor never
    vanishes (gap G3). Counting cats use a CV on the per-week mean; rate cats use absolute
    std seeds. Returns a variance (>= floor^2 > 0)."""
    mean = _f(mean)
    shrink = _shrink(n, is_hitter)
    if kind == "counting":
        reducible = abs(mean) * _TALENT_CV * shrink
        floor = max(abs(mean) * _PROJ_FLOOR_CV, _COUNTING_ABS_FLOOR)
    else:  # rate_prop / rate_ratio
        reducible = _RATE_TALENT_STD.get(category, 0.02) * shrink
        floor = _RATE_FLOOR_STD.get(category, 0.01)
    return float(reducible * reducible + floor * floor)


def week_to_week_tau2(mean: float, kind: str, category: str, weekly_vol: float) -> tuple[float, dict]:
    """Week-to-week (aleatory) variance + sampling margin. Counting -> NB (tau2 = phi*mu,
    r = mu/(phi-1); phi<=1 -> Poisson, r=inf). rate_prop (AVG/OBP) -> beta-binomial over
    `weekly_vol` AB. rate_ratio (ERA/WHIP) -> ratio-normal with week-std scaled by volume.
    Returns (tau2, margin_dict). Never raises."""
    mean = _f(mean)
    if kind == "counting":
        phi = _COUNTING_OVERDISPERSION.get(category, 1.4)
        mu = max(0.0, mean)
        tau2 = phi * mu
        r = mu / (phi - 1.0) if (phi > 1.0 and mu > 0.0) else math.inf
        return float(tau2), {"dist": "nb", "mean": mu, "r": r}

    if kind == "rate_prop":
        ref = _RATE_REF_AB
        vol = max(_MIN_WEEK_VOL, _f(weekly_vol))
        base = _RATE_WEEK_STD.get(category, 0.022)
        tau2 = (base * base) * (ref / vol)
        # Beta-binomial intra-class correlation implied by the inflated week variance over a
        # binomial baseline theta(1-theta)/vol; clamped to [0, 0.5] for stability.
        theta = min(max(mean, 1e-3), 1 - 1e-3)
        binom_var = theta * (1 - theta) / vol
        rho = 0.0 if binom_var <= 0 else min(0.5, max(0.0, (tau2 - binom_var) / max(binom_var, 1e-9)))
        return float(tau2), {"dist": "beta_binomial", "theta": theta, "n": vol, "rho": rho}

    # rate_ratio (ERA/WHIP): events-per-IP, not a proportion.
    ref = _RATE_REF_IP
    vol = max(_MIN_WEEK_VOL, _f(weekly_vol))
    base = _RATE_WEEK_STD.get(category, 0.70)
    tau2 = (base * base) * (ref / vol)
    return float(tau2), {"dist": "ratio_normal", "mean": mean, "std_week": math.sqrt(max(tau2, 0.0))}


def _is_hitter(row) -> bool:
    raw = row.get("is_hitter") if hasattr(row, "get") else row["is_hitter"]
    return _f(raw, default=1.0) >= 0.5


def _weekly_volume(row, kind: str, weeks: float) -> float:
    """Per-week AB (rate_prop) or IP (rate_ratio) from the pool's season projection."""
    col = _HITTER_VOL_COL if kind == "rate_prop" else _PITCHER_VOL_COL
    season = _f(row.get(col)) if hasattr(row, "get") else _f(row[col])
    return season / weeks if weeks > 0 else 0.0


def category_posterior(
    row, category: str, config: LeagueConfig | None = None, weeks: float | None = None
) -> CategoryPosterior:
    """Build the per-category posterior for one pool row. `weeks` defaults to
    config.season_weeks. Never raises on missing columns / NaN (degrades gracefully)."""
    cfg = config or LeagueConfig()
    weeks = float(weeks) if weeks else float(cfg.season_weeks)
    kind = classify_kind(category, cfg)
    is_hit = _is_hitter(row)
    mean = _mean_for(row, category, kind, cfg, weeks)
    n = _f(row.get("ytd_pa")) if is_hit else _f(row.get("ytd_ip"))
    sigma2 = between_player_sigma2(mean, kind, category, n, is_hit)
    weekly_vol = _weekly_volume(row, kind, weeks) if kind != "counting" else 0.0
    tau2, margin = week_to_week_tau2(mean, kind, category, weekly_vol)
    return CategoryPosterior(category=category, kind=kind, mean=mean, sigma2=sigma2, tau2=tau2, margin=margin)


def player_posteriors(
    row, config: LeagueConfig | None = None, weeks: float | None = None
) -> dict[str, CategoryPosterior]:
    """All posteriors for the cats relevant to this player (hitting cats for hitters,
    pitching cats for pitchers, by the pool `is_hitter` flag)."""
    cfg = config or LeagueConfig()
    cats = cfg.hitting_categories if _is_hitter(row) else cfg.pitching_categories
    return {c: category_posterior(row, c, cfg, weeks) for c in cats}
