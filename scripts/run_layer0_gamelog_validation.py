"""Real game-log validation of the Layer-0 player model (Phase-1 gate, offline).

Aggregates the local `game_logs` table (2026 YTD) into weekly per-player category actuals and
checks, per category, whether the model's posterior interval (mean +/- from sigma2+tau2) covers
the realized weekly totals -- i.e. is the total variance roughly calibrated? This is the REAL
game-log measurement the synthetic slice-5 tests could not provide (no network needed; the
actuals are already in the DB).

Method + honest caveats:
  * Actuals = 2026 game_logs summed to (player, ISO-week) totals; rate cats recomputed from summed
    components (avg=H/AB, obp=(H+BB+HBP)/(AB+BB+HBP+SF), era=9*ER/IP, whip=(BB+H)/IP).
  * We only score player-weeks the player actually PLAYED (AB>=1 hitters / IP>=1 pitchers), so the
    coverage measures the outcome model CONDITIONAL on availability (the availability layer models
    played-or-not separately -- mixing them here would spuriously "cover" IL weeks with a 0).
  * The model per-week mean is season_projection/26 (a flat weekly rate); it does not know how many
    games fell in a given week, so some coverage miss is expected structural noise, not a model bug.
  * gaussian_interval_coverage treats sigma_total as a Normal proxy for the (skewed) NB/beta-binomial
    margin -- a first-order diagnostic. coverage << 0.80 => intervals too narrow (widen sigma via the
    sigma_scale hint); >> 0.80 => too wide. This is the input to the deferred sigma/rho calibration.

Run:  python -m scripts.run_layer0_gamelog_validation
"""

from __future__ import annotations

import math

import pandas as pd

from src.database import get_connection, load_player_pool
from src.player_model import build_player_models
from src.player_model.validation import validate_layer0
from src.valuation import LeagueConfig

SEASON = 2026


def weekly_actuals(season: int = SEASON) -> pd.DataFrame:
    """Sum game_logs into (player_id, week) category totals + recomputed rate stats."""
    conn = get_connection()
    try:
        gl = pd.read_sql("SELECT * FROM game_logs WHERE season = ?", conn, params=(season,))
    finally:
        conn.close()
    gl["game_date"] = pd.to_datetime(gl["game_date"], errors="coerce")
    gl = gl.dropna(subset=["game_date"])
    gl["week"] = gl["game_date"].dt.isocalendar().week.astype(int)
    sum_cols = [
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
    for c in sum_cols:
        gl[c] = pd.to_numeric(gl[c], errors="coerce").fillna(0.0)
    agg = gl.groupby(["player_id", "week"], as_index=False)[sum_cols].sum()
    ab = agg["ab"].replace(0, pd.NA)
    ip = agg["ip"].replace(0, pd.NA)
    agg["avg"] = agg["h"] / ab
    agg["obp"] = (agg["h"] + agg["bb"] + agg["hbp"]) / (agg["ab"] + agg["bb"] + agg["hbp"] + agg["sf"]).replace(
        0, pd.NA
    )
    agg["era"] = agg["er"] * 9.0 / ip
    agg["whip"] = (agg["bb_allowed"] + agg["h_allowed"]) / ip
    return agg


def main() -> None:
    cfg = LeagueConfig()
    agg = weekly_actuals()
    played_ids = {int(p) for p in agg["player_id"].unique()}

    pool = load_player_pool()
    pool = pool[pool["player_id"].isin(played_ids)].copy()
    models = build_player_models(pool, cfg)

    hitting = set(cfg.hitting_categories)
    print(f"Layer-0 game-log validation  (season {SEASON}, {len(models)} players, {len(agg)} player-weeks)")
    print(f"{'CAT':5} {'n':>6} {'MAE':>8} {'cov80':>7} {'pit':>6} {'sig_scale':>9} {'pass':>5}")
    for cat in cfg.all_categories:
        col = cfg.STAT_MAP[cat]
        is_hit = cat in hitting
        rows = []
        for a in agg.itertuples(index=False):
            pid = int(a.player_id)
            pm = models.get(pid)
            if pm is None:
                continue
            post = pm.posteriors.get(cat)
            if post is None:
                continue  # player not this cat's type
            # played filter (reduce availability confound)
            if is_hit and a.ab < 1:
                continue
            if (not is_hit) and a.ip < 1:
                continue
            realized = getattr(a, col)
            if realized is None or (isinstance(realized, float) and math.isnan(realized)):
                continue
            rows.append(
                {
                    "model_mean": post.mean,
                    "model_sigma": math.sqrt(max(post.sigma2 + post.tau2, 0.0)),
                    "baseline_pred": post.mean,
                    "realized": float(realized),
                }
            )
        if not rows:
            print(f"{cat:5} {'--':>6}  (no played player-weeks)")
            continue
        recs = pd.DataFrame(rows)
        out = validate_layer0(
            recs,
            mean_col="model_mean",
            sigma_col="model_sigma",
            baseline_col="baseline_pred",
            realized_col="realized",
            level=0.80,
        )
        print(
            f"{cat:5} {out['n']:>6} {out['mae_model']:>8.3f} {out['coverage']:>7.3f} "
            f"{out['pit_mean']:>6.3f} {out['sigma_scale_hint']:>9.2f} {str(out['passes_gate']):>5}"
        )


if __name__ == "__main__":
    main()
