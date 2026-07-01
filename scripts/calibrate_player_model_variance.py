"""Empirical calibration analysis for the Layer-0 week-to-week variance (tau2).

Measures, per scoring category, the EMPIRICAL within-player week-to-week variance from the local
game_logs (pooled 2025 full season + 2026 YTD), using each player's OWN realized rate as the mean
(so this isolates the aleatory tau2 from projection error + availability), and compares it to what
the current model seeds produce -- yielding a per-category tau2 calibration multiplier.

For COUNTING cats: empirical overdispersion phi_emp = Var_week(X) / Mean_week(X) (the NB moment
estimator), aggregated as the week-weighted median across players; the calibration multiplier vs the
seed is phi_emp / phi_seed.  For RATE cats: the empirical within-player weekly-rate variance vs the
model's tau2 at that player's weekly volume.

Read-only analysis (prints a table); it does NOT modify any seed. Run:
    python -m scripts.calibrate_player_model_variance
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.database import get_connection
from src.player_model.posterior import (
    _COUNTING_OVERDISPERSION,
    _RATE_ICC,
    week_to_week_tau2,
)
from src.valuation import LeagueConfig

SEASONS = (2025, 2026)
MIN_WEEKS = 5  # a player needs this many played weeks to contribute a within-player variance


def _weekly(seasons=SEASONS) -> pd.DataFrame:
    conn = get_connection()
    try:
        q = "SELECT * FROM game_logs WHERE season IN ({})".format(",".join("?" * len(seasons)))
        gl = pd.read_sql(q, conn, params=list(seasons))
    finally:
        conn.close()
    gl["game_date"] = pd.to_datetime(gl["game_date"], errors="coerce")
    gl = gl.dropna(subset=["game_date"])
    gl["week"] = gl["game_date"].dt.isocalendar().week.astype(int)
    sums = [
        "pa",
        "ab",
        "h",
        "r",
        "hr",
        "rbi",
        "sb",
        "bb",
        "hbp",
        "sf",
        "ip",
        "w",
        "l",
        "sv",
        "k",
        "er",
        "bb_allowed",
        "h_allowed",
    ]
    for c in sums:
        gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0.0)
    agg = gl.groupby(["player_id", "season", "week"], as_index=False)[sums].sum()
    ab = agg["ab"].replace(0, np.nan)
    ip = agg["ip"].replace(0, np.nan)
    agg["avg"] = agg["h"] / ab
    agg["obp"] = (agg["h"] + agg["bb"] + agg["hbp"]) / (agg["ab"] + agg["bb"] + agg["hbp"] + agg["sf"]).replace(
        0, np.nan
    )
    agg["era"] = agg["er"] * 9.0 / ip
    agg["whip"] = (agg["bb_allowed"] + agg["h_allowed"]) / ip
    return agg


def main() -> None:
    cfg = LeagueConfig()
    agg = _weekly()
    hitting = set(cfg.hitting_categories)
    print(f"Empirical tau2 calibration analysis (seasons {SEASONS}, min {MIN_WEEKS} played weeks/player)")
    print(f"{'CAT':5} {'players':>7} {'seed':>8} {'emp':>8} {'mult':>6}   note")
    for cat in cfg.all_categories:
        col = cfg.STAT_MAP[cat]
        is_hit = cat in hitting
        vol_col = "ab" if is_hit else "ip"
        # played weeks only
        played = agg[agg[vol_col] >= 1].copy()
        mults, seed_vals, emp_vals, n_players = [], [], [], 0
        for (_pid, _season), grp in played.groupby(["player_id", "season"]):
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(vals) < MIN_WEEKS:
                continue
            mu = float(vals.mean())
            var_emp = float(vals.var(ddof=1))
            if mu <= 0 or not np.isfinite(var_emp):
                continue
            vol = float(grp[vol_col].mean())
            tau2_seed, _margin = week_to_week_tau2(mu, _kind(cat, cfg), cat, vol)
            if tau2_seed <= 0:
                continue
            n_players += 1
            emp_vals.append(var_emp)
            seed_vals.append(tau2_seed)
            mults.append(var_emp / tau2_seed)
        if n_players < 10:
            print(f"{cat:5} {n_players:>7}  (too few players)")
            continue
        mult = float(np.median(mults))
        if cat not in cfg.rate_stats:
            seed_disp = _COUNTING_OVERDISPERSION.get(cat, 1.4)
            emp_disp = seed_disp * mult
            note = f"phi {seed_disp:.2f} -> {max(1.0, emp_disp):.2f}"
        else:
            seed_disp = float(np.median(seed_vals))
            emp_disp = float(np.median(emp_vals))
            note = f"tau2 x{mult:.2f}  (ICC seed {_RATE_ICC.get(cat, 0.0)})"
        print(f"{cat:5} {n_players:>7} {np.median(seed_vals):>8.4g} {np.median(emp_vals):>8.4g} {mult:>6.2f}   {note}")

    _projection_error(cfg, agg)


def _projection_error(cfg: LeagueConfig, agg: pd.DataFrame) -> None:
    """Measure projection error (sigma2): pool per-week projection vs each player's realized
    2026 per-week mean. This is the epistemic term the projection-based coverage gap reflects."""
    from src.database import load_player_pool

    pool = load_player_pool()
    weeks = float(cfg.season_weeks)
    hitting = set(cfg.hitting_categories)
    a26 = agg[agg["season"] == 2026]
    print("\nProjection error (sigma) = pool per-week projection vs realized 2026 per-week mean:")
    print(f"{'CAT':5} {'players':>7} {'proj_err_sd':>11} {'as_CV':>7}   note")
    for cat in cfg.all_categories:
        col = cfg.STAT_MAP[cat]
        is_hit = cat in hitting
        vol_col = "ab" if is_hit else "ip"
        played = a26[a26[vol_col] >= 1]
        errs, cvs = [], []
        for (pid, _s), grp in played.groupby(["player_id", "season"]):
            vals = pd.to_numeric(grp[col], errors="coerce").dropna()
            if len(vals) < MIN_WEEKS:
                continue
            prow = pool[pool["player_id"] == pid]
            if prow.empty:
                continue
            proj_season = pd.to_numeric(prow.iloc[0].get(col), errors="coerce")
            if pd.isna(proj_season):
                continue
            proj_week = float(proj_season) / weeks if cat not in cfg.rate_stats else float(proj_season)
            realized_week = float(vals.mean())
            err = realized_week - proj_week
            errs.append(err)
            if abs(realized_week) > 1e-6:
                cvs.append(abs(err) / abs(realized_week))
        if len(errs) < 10:
            print(f"{cat:5} {len(errs):>7}  (too few)")
            continue
        sd = float(np.std(errs, ddof=1))
        cv = float(np.median(cvs)) if cvs else 0.0
        print(f"{cat:5} {len(errs):>7} {sd:>11.4g} {cv:>7.2f}")


def _kind(cat: str, cfg: LeagueConfig) -> str:
    if cat not in cfg.rate_stats:
        return "counting"
    return "rate_prop" if cat in {"AVG", "OBP"} else "rate_ratio"


if __name__ == "__main__":
    main()
