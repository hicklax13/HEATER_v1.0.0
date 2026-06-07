"""26-week × 12-category H2H win-probability matrix.

Per report Section B.5: for each remaining matchup week w and category c,
compute

    p_{w,c} = Phi((mu_H - mu_O) / sqrt(sigma_H^2 + sigma_O^2))

where (mu_H, sigma_H) are the user's projected weekly mean and standard
deviation for category c, and (mu_O, sigma_O) are the same for whichever
opponent is scheduled in week w. For inverse cats (L, ERA, WHIP) the
result is flipped (1 - p).

Variance source (per 2026-05-23 design Q3 + design-note in the commit):
the user requested Kalman-filter posterior variance. The existing Kalman
pipeline in src/engine/signals/kalman.py captures TRUE-TALENT uncertainty
("Trout's true AVG is .300 ± .005"), which is the wrong distribution for
PER-WEEK win probability — per-week noise is dominated by sampling
variance ("Trout will hit somewhere between .150 and .450 this week"),
not by talent uncertainty.

This module uses a per-category coefficient-of-variation (CV) approach
calibrated by category-level empirical week-to-week stability:
  - Rate hitting (AVG, OBP): low CV (~4-5%) — stabilize quickly with PA
  - Counting hitting (R, RBI): moderate CV (~15%)
  - Power/speed (HR, SB): higher CV (~20-25%) — Poisson-like
  - Pitching wins/losses: high CV (~30%) — small per-week sample
  - Pitching counting K: moderate CV (~15%)
  - Pitching rates ERA/WHIP: moderate CV (~12-18%) — IP-bounded

Kalman augmentation is a layered enhancement for future work; the CV
approach gives a working weekly matrix today.

Wires into:
  - src/valuation.py: LeagueConfig for cats / inverse_stats / season_weeks
  - src/engine/output/trade_evaluator.py: invoked when enable_weekly_matrix=True

Returns DataFrames indexed by week, columns by category — naturally
renderable as a heatmap by Streamlit.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, skellam

from src.valuation import LeagueConfig

logger = logging.getLogger(__name__)

# Per-category coefficient of variation — fraction of mean used as per-week SD.
# Calibrated by category type per the docstring rationale above. These are
# defaults; callers can override per-trade if backtesting suggests different
# empirical values for the league.
_DEFAULT_CV: dict[str, float] = {
    # Hitting rates (stabilize fastest)
    "AVG": 0.05,
    "OBP": 0.04,
    # Counting hitting
    "R": 0.15,
    "RBI": 0.15,
    "HR": 0.20,
    "SB": 0.25,
    # Pitching counting
    "W": 0.30,
    "L": 0.30,
    "SV": 0.35,
    "K": 0.15,
    # Pitching rates
    "ERA": 0.18,
    "WHIP": 0.12,
}

# Floor for sigma to avoid divide-by-zero when projected mean is 0.
# A weekly mean of exactly 0 with CV>0 would give sigma=0, breaking Phi().
# The floor represents irreducible variance — even a 0-projection player
# can pop a HR or steal a base in a given week.
_MIN_SIGMA: float = 0.01

# TE-E4 / LO-E3: low-count COUNTING categories that route to the Skellam
# (Poisson-difference) model by default. Their weekly totals are small enough
# (single digits / low teens) that the Gaussian approximation understates the
# spread and mis-handles the right skew (report C.5 / H.9). High-count counting
# cats (R/HR/RBI/K) and all rate cats keep the Normal model. The set is
# intersected with the league's counting (non-rate) cats so it cannot drift
# from LeagueConfig's category set. Mirrors src/optimizer/h2h_engine.SKELLAM_CATS.
_LOW_COUNT_SKELLAM_NAMES: frozenset[str] = frozenset({"SB", "SV", "W", "L"})


def _skellam_cats(config: LeagueConfig) -> frozenset[str]:
    """Low-count counting cats to route through Skellam, derived from config.

    Intersects the low-count name set with the league's counting (non-rate)
    categories so the routing can't drift from LeagueConfig.
    """
    counting = {c for c in config.all_categories} - set(config.rate_stats)
    return frozenset(_LOW_COUNT_SKELLAM_NAMES & counting)


def compute_weekly_matrix(
    user_roster_ids: list[int],
    player_pool: pd.DataFrame,
    schedule: dict[int, str],
    all_team_rosters: dict[str, list[int]],
    config: LeagueConfig | None = None,
    variance_cv: dict[str, float] | None = None,
    variance_model: str = "auto",
    enable_kalman: bool = False,
) -> dict[str, Any]:
    """Build the 26 × 12 weekly win-probability matrix for one roster.

    Args:
        user_roster_ids: Hickey's player IDs (post-trade if evaluating a trade).
        player_pool: Full enriched player pool DataFrame.
        schedule: Mapping {week_number: opponent_team_name} for each remaining
            matchup week. Caller is responsible for sourcing this from Yahoo
            (typically via yds.get_schedule() or similar).
        all_team_rosters: Mapping {team_name: [player_ids]} for all 12 teams.
            Used to compute opponent per-week means.
        config: LeagueConfig instance (defaults to LeagueConfig()).
        variance_cv: Optional per-category CV override. Defaults to
            _DEFAULT_CV calibrated per the docstring rationale.
        variance_model: Per-category win-prob model selector (TE-E4):
          - ``"auto"`` (default): low-count counting cats (SB/SV/W/L per
            :func:`_skellam_cats`) use the skew-aware Skellam model; every
            other cat (rate cats + high-count counting cats) uses Normal.
          - ``"normal"``: force ALL cats through the Normal model (back-compat
            all-Normal override; keeps the legacy ``'cv-based'`` method tag).
          - ``"skellam"``: force every COUNTING cat through Skellam (rate cats
            always stay Normal since Skellam is integer-only).
            The empirical per-category CV table needs prior-season box-score
            data we don't have, so it is left as-is (needs-data).
        enable_kalman: Blend Kalman true-talent variance into the Normal-model
            cats (Kalman is a Normal-only augmentation; Skellam cats ignore it).

    Returns:
        Dict with keys:
          - matrix: pd.DataFrame indexed by week, columns are categories,
                    values are p_{w,c} in [0, 1].
          - schedule: Echo of the schedule used.
          - user_weekly_means: Dict[cat -> float] of user's per-week means.
          - opponent_weekly_means: Dict[team -> Dict[cat -> float]].
          - method: 'cv-based' (all-Normal), 'cv-based+skellam-lowcount'
                    (auto default), or 'skellam' (force-all-counting), each
                    optionally '+kalman'.
          - cv_used: The CV dict actually used (for diagnostic transparency).
    """
    if config is None:
        config = LeagueConfig()
    if variance_cv is None:
        variance_cv = dict(_DEFAULT_CV)

    cats = list(config.all_categories)
    inverse = set(config.inverse_stats)
    counting_cats = set(cats) - set(config.rate_stats)
    season_weeks = int(config.season_weeks)

    # TE-E4: resolve the per-category model. In "auto" (default), low-count
    # counting cats route to Skellam and everything else to Normal; "normal"
    # forces all-Normal; "skellam" forces every counting cat to Skellam.
    skellam_cats = _skellam_cats(config)

    def _model_for(cat: str) -> str:
        if variance_model == "auto":
            return "skellam" if cat in skellam_cats else "normal"
        return variance_model

    # User per-week means (one set, used for every week — projections are
    # season-flat; per-week schedule strength comes from opponent variance)
    user_means = _per_week_means(user_roster_ids, player_pool, cats, season_weeks)

    # Opponent per-week means — pre-computed per team
    opp_team_means: dict[str, dict[str, float]] = {}
    for team_name, ids in all_team_rosters.items():
        opp_team_means[team_name] = _per_week_means(ids, player_pool, cats, season_weeks)

    # Optional Kalman true-talent variance (per-week mean units) per team.
    user_kalman: dict[str, float] = {}
    opp_kalman: dict[str, dict[str, float]] = {}
    if enable_kalman:
        user_kalman = build_team_kalman_variances(user_roster_ids, player_pool, cats, season_weeks, config)
        for team_name, ids in all_team_rosters.items():
            opp_kalman[team_name] = build_team_kalman_variances(ids, player_pool, cats, season_weeks, config)

    weeks = sorted(schedule.keys())
    matrix = pd.DataFrame(index=weeks, columns=cats, dtype=float)

    for week in weeks:
        opp_name = schedule.get(week)
        if opp_name is None or opp_name not in opp_team_means:
            # Unknown opponent for this week — neutral 0.5 across all cats
            logger.debug(
                "weekly_matrix: no opponent data for week %s (opp=%r) — defaulting to 0.5",
                week,
                opp_name,
            )
            matrix.loc[week, :] = 0.5
            continue
        opp_means = opp_team_means[opp_name]
        opp_k = opp_kalman.get(opp_name, {})

        for cat in cats:
            p = _category_win_prob(
                mu_h=user_means.get(cat, 0.0),
                mu_o=opp_means.get(cat, 0.0),
                cv=variance_cv.get(cat, 0.15),
                inverse=cat in inverse,
                kalman_var_h=user_kalman.get(cat, 0.0),
                kalman_var_o=opp_k.get(cat, 0.0),
                variance_model=_model_for(cat),
                counting=cat in counting_cats,
            )
            matrix.loc[week, cat] = p

    if variance_model == "auto":
        method = "cv-based+skellam-lowcount"
    elif variance_model == "normal":
        method = "cv-based"
    else:
        method = variance_model
    if enable_kalman:
        method = f"{method}+kalman"

    return {
        "matrix": matrix,
        "schedule": dict(schedule),
        "user_weekly_means": user_means,
        "opponent_weekly_means": opp_team_means,
        "method": method,
        "cv_used": dict(variance_cv),
    }


def compute_trade_weekly_delta(
    before_roster_ids: list[int],
    after_roster_ids: list[int],
    player_pool: pd.DataFrame,
    schedule: dict[int, str],
    all_team_rosters: dict[str, list[int]],
    config: LeagueConfig | None = None,
    variance_cv: dict[str, float] | None = None,
    variance_model: str = "auto",
    enable_kalman: bool = False,
) -> dict[str, Any]:
    """Compute the before / after / delta weekly matrices for a trade.

    Used by evaluate_trade() when enable_weekly_matrix=True. Returns three
    matrices: pre-trade win probability per week-cat, post-trade ditto, and
    the element-wise delta (positive = trade improves your win probability
    in that week-cat).

    Args:
        before_roster_ids: Pre-trade roster IDs.
        after_roster_ids: Post-trade roster IDs.
        player_pool: Full enriched pool.
        schedule: {week: opponent_team_name} for remaining weeks.
        all_team_rosters: {team: [player_ids]} for opponent means.
        config: LeagueConfig.
        variance_cv: Optional CV override.

    Returns:
        Dict with keys:
          - before: pre-trade matrix (DataFrame)
          - after: post-trade matrix (DataFrame)
          - delta: after - before (DataFrame)
          - schedule: echo
          - summary: per-week summary including expected_cats_won_delta
          - method: 'cv-based'
    """
    before = compute_weekly_matrix(
        before_roster_ids, player_pool, schedule, all_team_rosters, config, variance_cv, variance_model, enable_kalman
    )
    after = compute_weekly_matrix(
        after_roster_ids, player_pool, schedule, all_team_rosters, config, variance_cv, variance_model, enable_kalman
    )
    delta = after["matrix"] - before["matrix"]

    # Per-week summary: expected cat-wins (sum of p_c) before/after/delta.
    # In a 12-cat H2H, expected cat-wins ranges 0-12. A delta of +1 means
    # the trade flips one full category in your favor in expectation.
    summary = pd.DataFrame(
        {
            "expected_cat_wins_before": before["matrix"].sum(axis=1),
            "expected_cat_wins_after": after["matrix"].sum(axis=1),
            "delta_cat_wins": delta.sum(axis=1),
            "opponent": [schedule.get(w, "?") for w in before["matrix"].index],
        },
        index=before["matrix"].index,
    )

    return {
        "before": before["matrix"],
        "after": after["matrix"],
        "delta": delta,
        "schedule": before["schedule"],
        "summary": summary,
        "method": before["method"],
        "cv_used": before["cv_used"],
    }


def _category_win_prob(
    mu_h: float,
    mu_o: float,
    cv: float,
    inverse: bool,
    kalman_var_h: float = 0.0,
    kalman_var_o: float = 0.0,
    variance_model: str = "normal",
    counting: bool = False,
) -> float:
    """Per-category single-week win probability.

    Two variance models (per report Section H.9 / C.5):

    * ``"normal"`` (default): Gaussian approximation
      ``p = Phi((mu_h - mu_o) / sqrt(sigma_h^2 + sigma_o^2))``. Symmetric;
      adequate for rate stats and high-count categories.
    * ``"skellam"``: for COUNTING categories, model each side's weekly total
      as Poisson and use the Skellam distribution (the exact difference of
      two Poissons) for ``P(X_h > X_o)``. This captures the right-skew the
      report flags for low-count categories like SB and SV, where the
      Gaussian approximation understates the spread and mis-handles
      asymmetry. Rate categories always use the normal model regardless,
      since Skellam is only defined for integer counts.

    For inverse cats, the "winning" direction is lower (fewer L, lower
    ERA/WHIP), handled per model.

    Kalman-augmented mode (normal model only, per design Q3 follow-up):
    when ``kalman_var_h`` / ``kalman_var_o`` are non-zero, the effective σ
    blends per-week sampling variance with Kalman true-talent uncertainty:
    ``sigma_effective² = (CV × mean)² + kalman_var``. Defaults to 0.0
    preserves the legacy CV-only behavior.

    Args:
        mu_h: User's per-week mean for category.
        mu_o: Opponent's per-week mean for category.
        cv: Per-week coefficient of variation (e.g. 0.20 for HR).
        inverse: True for ERA/WHIP/L (lower wins).
        kalman_var_h: Optional Kalman true-talent variance for user side.
        kalman_var_o: Optional Kalman true-talent variance for opponent side.
        variance_model: "normal" or "skellam".
        counting: True for counting categories. Only counting cats use the
            Skellam model; rate cats fall back to normal even under "skellam".

    Returns:
        Win probability in [0, 1].
    """
    if variance_model == "skellam" and counting:
        return _category_win_prob_skellam(mu_h, mu_o, inverse)

    sigma_h_cv = max(abs(mu_h) * cv, _MIN_SIGMA)
    sigma_o_cv = max(abs(mu_o) * cv, _MIN_SIGMA)
    # Blend: σ_total² = σ_CV² + σ_Kalman² (independent variance addition)
    sigma_h_total_sq = sigma_h_cv**2 + max(kalman_var_h, 0.0)
    sigma_o_total_sq = sigma_o_cv**2 + max(kalman_var_o, 0.0)
    combined_sigma = float(np.sqrt(sigma_h_total_sq + sigma_o_total_sq))

    if combined_sigma < 1e-12:
        # Degenerate: both means and sigmas are 0. Coin-flip.
        p = 0.5
    else:
        z = (mu_h - mu_o) / combined_sigma
        p = float(norm.cdf(z))

    if inverse:
        p = 1.0 - p
    # Clamp to (0, 1) for numerical safety
    return max(0.0, min(1.0, p))


def _category_win_prob_skellam(mu_h: float, mu_o: float, inverse: bool) -> float:
    """Skew-aware single-week win prob for a counting category (report C.5/H.9).

    Models each side's weekly count as Poisson(mu) and uses the Skellam
    distribution K = X_h - X_o for the exact head-to-head probability:

        P(X_h > X_o) = P(K >= 1) = 1 - skellam.cdf(0, mu_h, mu_o)
        P(tie)       = skellam.pmf(0, mu_h, mu_o)

    A category tie is scored as half a win (Yahoo splits tied categories).
    For inverse counting cats (Losses), the winning direction flips to
    ``X_h < X_o`` = ``P(K <= -1)``. This Poisson-difference model captures
    the right skew of low-count categories (SB, SV) that the Gaussian
    approximation in the "normal" model misses.
    """
    mh = max(float(mu_h), 1e-6)
    mo = max(float(mu_o), 1e-6)
    p_tie = float(skellam.pmf(0, mh, mo))
    if inverse:
        # Lower count wins: P(K <= -1) + half the tie mass.
        p = float(skellam.cdf(-1, mh, mo)) + 0.5 * p_tie
    else:
        # Higher count wins: P(K >= 1) = 1 - P(K <= 0); add half the tie mass.
        p = (1.0 - float(skellam.cdf(0, mh, mo))) + 0.5 * p_tie
    return max(0.0, min(1.0, p))


# IL statuses excluded from per-week means (TE-C1). Matches the engine's
# IL convention in src/alerts.get_il_stash_names (compared upper-cased).
_IL_STATUSES: frozenset[str] = frozenset({"IL10", "IL15", "IL60", "IL", "DTD"})


def _per_week_means(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    cats: list[str],
    season_weeks: int,
) -> dict[str, float]:
    """Aggregate per-cat season totals from roster, divide by season weeks.

    For counting stats: sum projection / season_weeks → per-week mean.
    For rate stats: weighted by volume (AB for AVG, PA for OBP, IP for
    ERA/WHIP) so a roster's rate is the volume-correct blend, not a
    naive mean of rates.
    """
    rows = player_pool[player_pool["player_id"].isin(roster_ids)]
    # TE-C1: exclude IL-status players (they don't contribute to weekly
    # totals) so per-week counting means aren't inflated and rate-stat
    # volume weights aren't skewed toward non-contributing IL stashes.
    # Opt-in on the column's presence — pools without a status column
    # (e.g. unit-test fixtures) behave exactly as before. IL convention
    # matches src/alerts.get_il_stash_names: {IL10, IL15, IL60, IL, DTD}.
    if "status" in rows.columns and not rows.empty:
        status_upper = rows["status"].astype(str).str.strip().str.upper()
        rows = rows[~status_upper.isin(_IL_STATUSES)]
    if rows.empty:
        return dict.fromkeys(cats, 0.0)

    def _sum(col: str) -> float:
        if col not in rows.columns:
            return 0.0
        return float(pd.to_numeric(rows[col], errors="coerce").fillna(0).sum())

    sum_h = _sum("h")
    sum_ab = _sum("ab")
    sum_bb = _sum("bb")
    sum_hbp = _sum("hbp")
    sum_sf = _sum("sf")
    sum_ip = _sum("ip")
    sum_er = _sum("er")
    sum_bb_allowed = _sum("bb_allowed")
    sum_h_allowed = _sum("h_allowed")

    means: dict[str, float] = {}
    for cat in cats:
        if cat == "AVG":
            means[cat] = (sum_h / sum_ab) if sum_ab > 0 else 0.0
        elif cat == "OBP":
            denom = sum_ab + sum_bb + sum_hbp + sum_sf
            means[cat] = ((sum_h + sum_bb + sum_hbp) / denom) if denom > 0 else 0.0
        elif cat == "ERA":
            means[cat] = (sum_er * 9.0 / sum_ip) if sum_ip > 0 else 0.0
        elif cat == "WHIP":
            means[cat] = ((sum_bb_allowed + sum_h_allowed) / sum_ip) if sum_ip > 0 else 0.0
        else:
            # Counting stat: per-week mean = season total / season weeks
            col = cat.lower()
            total = _sum(col)
            means[cat] = total / max(season_weeks, 1)

    return means


# Kalman true-talent variance wiring (report C.5/C.6, design Q3 follow-up).
# Stabilization sample sizes (Carleton/Russell-style): a roster with this
# much YTD volume has roughly half its true-talent uncertainty resolved.
_KALMAN_STABILIZE_PA: float = 200.0
_KALMAN_STABILIZE_IP: float = 70.0
# True-talent SD as a fraction of the per-week mean for a fully-UNSAMPLED
# roster. Shrinks toward 0 as YTD sample grows (shrink factor below).
_KALMAN_TALENT_CV: float = 0.40


def build_team_kalman_variances(
    roster_ids: list[int],
    player_pool: pd.DataFrame,
    cats: list[str],
    season_weeks: int,
    config: LeagueConfig | None = None,
) -> dict[str, float]:
    """Per-team per-category Kalman true-talent variance (per-week mean units).

    Turns the Kalman ``_category_win_prob`` plumbing into a real caller per
    the design Q3 follow-up. The idea: a roster whose projections rest on
    little YTD evidence carries extra TRUE-TALENT uncertainty (is this rookie
    really a .300 hitter?), separate from week-to-week sampling noise. That
    uncertainty widens the win-probability distribution toward 0.5.

    Model: team-level true-talent SD for category ``c`` is
    ``|mu_c| × _KALMAN_TALENT_CV × shrink`` where ``shrink = S0 / (S0 +
    sample)`` uses the roster's aggregate YTD volume (PA for hitting cats,
    IP for pitching cats). Fully-sampled rosters → shrink ≈ 0 → behaves like
    the legacy CV-only model; rookie-heavy rosters → larger variance.

    Returns ``{cat: variance}`` (variance, not SD) ready to pass as
    ``kalman_var_h`` / ``kalman_var_o`` to :func:`_category_win_prob`.
    """
    if config is None:
        config = LeagueConfig()
    rows = player_pool[player_pool["player_id"].isin(roster_ids)]
    if rows.empty:
        return dict.fromkeys(cats, 0.0)

    hitting = set(config.hitting_categories)
    means = _per_week_means(roster_ids, player_pool, cats, season_weeks)

    def _num(col: str) -> pd.Series:
        if col not in rows.columns:
            return pd.Series(np.zeros(len(rows)), index=rows.index)
        return pd.to_numeric(rows[col], errors="coerce").fillna(0)

    is_hit = _num("is_hitter")
    ytd_pa = _num("ytd_pa")
    ytd_ip = _num("ytd_ip")
    hitter_pa = float(ytd_pa[is_hit == 1].sum())
    pitcher_ip = float(ytd_ip[is_hit == 0].sum())

    shrink_hit = _KALMAN_STABILIZE_PA / (_KALMAN_STABILIZE_PA + hitter_pa)
    shrink_pit = _KALMAN_STABILIZE_IP / (_KALMAN_STABILIZE_IP + pitcher_ip)

    out: dict[str, float] = {}
    for cat in cats:
        m = means.get(cat, 0.0)
        shrink = shrink_hit if cat in hitting else shrink_pit
        sd = abs(m) * _KALMAN_TALENT_CV * shrink
        out[cat] = float(sd * sd)
    return out
