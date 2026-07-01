"""Per-category H2H win/tie/loss probabilities from two teams' category-total moments.

Margin D = A - B, mu_D = a_mu - b_mu, sigma_D^2 = a_var + b_var (variances add: disjoint rosters
are independent). ONE unified three-cell shape with tie half-width h:
    P(win)  = Phi((mu_D - h)/sigma_D)          # = norm.sf(h,  mu_D, sigma_D)
    P(tie)  = Phi((h - mu_D)/sigma_D) - Phi((-h - mu_D)/sigma_D)
    P(loss) = Phi((-h - mu_D)/sigma_D)         # = norm.cdf(-h, mu_D, sigma_D)
  * counting cats: h = 0.5 (integer continuity correction); optional moment-matched Skellam upgrade
    in the low-count regime (sigma_D < 3 and feasible) where the discrete skew matters.
  * rate cats: h = rounding_unit/2 (Yahoo display collision: 0.001 AVG/OBP, 0.01 ERA/WHIP).
  * inverse cats (L/ERA/WHIP, lower wins): swap P(win) <-> P(loss); P(tie) invariant.
Triple sums to exactly 1 (telescoping Phi terms — algebraic, not a CLT approximation). Pure; no DB/
network; never raises (NaN/negative inputs coerced to a safe degenerate result).
"""

from __future__ import annotations

import math

from src.valuation import LeagueConfig

# Below this sigma_D (and only when the Skellam is feasible) the discrete Skellam earns its Bessel
# cost — that is the low-count regime (thin cats: SV, SB, a low-W week) where Normal-CC over-states
# the tie. Above it, Normal-CC agrees to <1% (overdispersion enters only via sigma_D, so MORE
# overdispersion -> LARGER sigma_D -> Normal-CC gets BETTER). Ratified default: 3.0 (conservative).
_SIGMA_SKELLAM_THRESHOLD: float = 3.0

# Yahoo display rounding units per rate category (tie = both round to the same shown value).
_ROUNDING_UNITS: dict[str, float] = {"AVG": 0.001, "OBP": 0.001, "ERA": 0.01, "WHIP": 0.01}


def _f(value, default: float = 0.0) -> float:
    """NaN/inf/None-safe float coercion."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def kind_for(category: str, config: LeagueConfig | None = None) -> str:
    """'counting' or 'rate' for an UPPERCASE category, from LeagueConfig (no hardcoded lists)."""
    cfg = config or LeagueConfig()
    return "rate" if category in cfg.rate_stats else "counting"


def rounding_unit_for(category: str) -> float:
    """Yahoo display rounding unit for a rate category (0 for counting — h defaults to 0.5)."""
    return _ROUNDING_UNITS.get(category, 0.0)


def category_win_tie_loss(
    a_mu, a_var, b_mu, b_var, kind: str, inverse: bool, rounding_unit: float
) -> tuple[float, float, float]:
    """P(win)/P(tie)/P(loss) for the H2H category margin D = A - B (team A's perspective).
    Sums to 1.0. `kind` in {'counting','rate'}; `inverse` True for lower-wins cats; `rounding_unit`
    is the Yahoo display unit for rate cats (ignored for counting). Never raises."""
    from scipy.stats import norm

    a_mu, b_mu = _f(a_mu), _f(b_mu)
    a_var = max(0.0, _f(a_var))
    b_var = max(0.0, _f(b_var))
    mu_d = a_mu - b_mu
    var_d = a_var + b_var
    sigma_d = math.sqrt(var_d)
    h = 0.5 if kind == "counting" else _f(rounding_unit) / 2.0

    def finish(p_win: float, p_tie: float, p_loss: float) -> tuple[float, float, float]:
        if inverse:
            p_win, p_loss = p_loss, p_win
        return (p_win, p_tie, p_loss)

    # Deterministic: no variance -> hard comparison against the tie band (guard before any cdf).
    if sigma_d <= 0.0:
        if abs(mu_d) < h:
            return (0.0, 1.0, 0.0)
        return finish(1.0, 0.0, 0.0) if mu_d > 0 else finish(0.0, 0.0, 1.0)

    # Counting low-count regime: exact moment-matched Skellam (discrete tie mass + skew).
    if kind == "counting" and sigma_d < _SIGMA_SKELLAM_THRESHOLD:
        lam1 = 0.5 * (var_d + mu_d)
        lam2 = 0.5 * (var_d - mu_d)
        if lam1 >= 0.0 and lam2 >= 0.0:  # feasible <=> var_d >= |mu_d|
            from scipy.stats import skellam

            lam1 = max(lam1, 0.0)
            lam2 = max(lam2, 0.0)
            p_win = float(skellam.sf(0, lam1, lam2))  # P(D >= 1)
            p_tie = float(skellam.pmf(0, lam1, lam2))  # P(D  = 0)
            p_loss = float(skellam.cdf(-1, lam1, lam2))  # P(D <= -1)
            s = p_win + p_tie + p_loss
            if s > 0:  # defensive renorm (swallow 1-ulp float drift)
                p_win, p_tie, p_loss = p_win / s, p_tie / s, p_loss / s
            return finish(p_win, p_tie, p_loss)

    # Normal / Welch shape (rate cats, and counting Normal-CC fallback/large-sigma).
    p_win = float(norm.sf(h, loc=mu_d, scale=sigma_d))
    p_loss = float(norm.cdf(-h, loc=mu_d, scale=sigma_d))
    p_tie = 1.0 - p_win - p_loss
    return finish(p_win, p_tie, p_loss)
