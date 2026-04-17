"""YTD-aware projection blending for trade evaluation.

Fixes a gap in the trade evaluator: pre-season projections were used verbatim
even when current-season YTD actuals diverged materially. Two failure modes
this module addresses:

1. **Hot/cold starts** — A hitter projected for .270 but hitting .330 through
   60 PA should have his forward projection shrunk toward his YTD pace by
   sample size. Same for pitchers with low ERA YTD vs mid projection.

2. **Role changes** — A reliever projected as a 30-SV closer who has 7 GP
   and 0 SV has almost certainly lost the closer role. The sample-size blend
   alone is too gentle for this qualitative shift; a separate detector caps
   the SV projection.

Exports:
    blend_projection_with_ytd(pool) -> pool with blended stat columns
    detect_role_change(pool) -> pool with role-adjusted SV column
    apply_ytd_corrections(pool) -> convenience: runs both
"""

from __future__ import annotations

import pandas as pd

# Regression (shrinkage) sample sizes. A player with this much YTD gets ~40%
# weight on YTD; below it scales linearly. Tuned so early-April (10-20 PA)
# stays mostly-projection, while late-April (80-100 PA) meaningfully shifts.
REGRESSION_PA: float = 250.0
REGRESSION_IP: float = 80.0
MAX_YTD_WEIGHT: float = 0.40

# Role-change detection thresholds.
# A reliever projected for >=10 SV who has >=5 GP with a save rate under
# this threshold gets flagged as likely-not-the-closer.
MIN_PROJ_SV_FOR_ROLE_CHECK: int = 10
MIN_GP_FOR_ROLE_CHECK: int = 5
CLOSER_SV_RATE_FLOOR: float = 0.15  # saves per GP below this = not closing
DEMOTED_SV_RETENTION: float = 0.25  # keep 25% of projection (occasional fill-in)
DEMOTED_SV_MIN: int = 3  # don't drop below 3 SV (hold-based occasional saves)


def _ytd_blend_weight(sample: float, regression_point: float) -> float:
    """Compute shrinkage weight from YTD sample size.

    Returns 0.0 when sample is 0, scales linearly up to MAX_YTD_WEIGHT.
    """
    if sample <= 0 or regression_point <= 0:
        return 0.0
    return min(sample / regression_point, MAX_YTD_WEIGHT)


def blend_projection_with_ytd(
    pool: pd.DataFrame,
    regression_pa: float = REGRESSION_PA,
    regression_ip: float = REGRESSION_IP,
) -> pd.DataFrame:
    """Blend pre-season projections with YTD actuals by sample size.

    For rate stats (AVG, ERA, WHIP), blends the rate directly.
    For counting stats (HR, RBI, SB, K, SV), blends the per-PA/per-IP rate
    then rescales to the original full-season volume projection.

    Only modifies rows where ytd_gp > 0 so pre-season-only rows are untouched.
    """
    if pool.empty:
        return pool.copy()

    out = pool.copy()
    # Cast numeric stat columns to float so blended rescaling doesn't trigger
    # int64->float dtype incompatibility warnings on assignment.
    for col in ("avg", "hr", "rbi", "sb", "era", "whip", "k", "sv"):
        if col in out.columns:
            out[col] = out[col].astype(float)

    ytd_gp = out.get("ytd_gp", pd.Series([0] * len(out))).fillna(0)
    is_hitter = out.get("is_hitter", pd.Series([1] * len(out))).fillna(1).astype(int)
    has_ytd = ytd_gp > 0
    if not has_ytd.any():
        return out

    # --- Hitter stats ---
    hit_mask = has_ytd & (is_hitter == 1)
    if hit_mask.any():
        ytd_pa = out.loc[hit_mask, "ytd_pa"].fillna(0)
        weights = ytd_pa.apply(lambda pa: _ytd_blend_weight(pa, regression_pa))

        # Rate stat: AVG
        proj_avg = out.loc[hit_mask, "avg"].fillna(0.0)
        ytd_avg = out.loc[hit_mask, "ytd_avg"].fillna(proj_avg)
        out.loc[hit_mask, "avg"] = (1 - weights) * proj_avg + weights * ytd_avg

        # Counting stats: rescale rate
        proj_pa = out.loc[hit_mask, "pa"].fillna(0.0).replace(0, 1.0)
        ytd_pa_safe = ytd_pa.replace(0, 1.0)
        for stat in ["hr", "rbi", "sb"]:
            ytd_col = f"ytd_{stat}"
            if ytd_col not in out.columns:
                continue
            proj_val = out.loc[hit_mask, stat].fillna(0.0)
            proj_rate = proj_val / proj_pa
            ytd_val = out.loc[hit_mask, ytd_col].fillna(0.0)
            ytd_rate = ytd_val / ytd_pa_safe
            blended_rate = (1 - weights) * proj_rate + weights * ytd_rate
            out.loc[hit_mask, stat] = blended_rate * proj_pa

    # --- Pitcher stats ---
    pit_mask = has_ytd & (is_hitter == 0)
    if pit_mask.any():
        # Estimate YTD IP from GP (league avg ~1 IP/appearance for RP, 5-6 for SP)
        # We don't have ytd_ip, so use ytd_gp scaled by proj IP-per-GP as a proxy.
        proj_ip = out.loc[pit_mask, "ip"].fillna(0.0).replace(0, 1.0)
        ytd_gp_p = out.loc[pit_mask, "ytd_gp"].fillna(0)
        # Proxy YTD IP: assume same IP-per-appearance as projection
        # Note: this is an approximation; proper fix requires adding ytd_ip to DB schema.
        # For the blend weight we use ytd_gp as a sample-size proxy scaled to IP.
        ytd_ip_proxy = ytd_gp_p * 1.5  # conservative avg for mixed RP/SP starts
        weights = ytd_ip_proxy.apply(lambda ip: _ytd_blend_weight(ip, regression_ip))

        # Rate stats: ERA, WHIP
        for stat in ["era", "whip"]:
            ytd_col = f"ytd_{stat}"
            if ytd_col not in out.columns:
                continue
            proj_val = out.loc[pit_mask, stat].fillna(0.0)
            ytd_val = out.loc[pit_mask, ytd_col].fillna(proj_val)
            # Guard against extreme early-season YTD (e.g., 14.40 ERA in 5 IP)
            # by capping YTD at 3x the projection for blending purposes.
            ytd_capped = ytd_val.where(ytd_val <= proj_val * 3, proj_val * 3)
            out.loc[pit_mask, stat] = (1 - weights) * proj_val + weights * ytd_capped

        # Counting stats: K, SV (W, L not in YTD schema)
        for stat in ["k", "sv"]:
            ytd_col = f"ytd_{stat}"
            if ytd_col not in out.columns:
                continue
            proj_val = out.loc[pit_mask, stat].fillna(0.0)
            proj_rate = proj_val / proj_ip
            ytd_val = out.loc[pit_mask, ytd_col].fillna(0.0)
            ytd_rate = ytd_val / ytd_ip_proxy.replace(0, 1.0)
            blended_rate = (1 - weights) * proj_rate + weights * ytd_rate
            out.loc[pit_mask, stat] = blended_rate * proj_ip

    return out


def detect_role_change(pool: pd.DataFrame) -> pd.DataFrame:
    """Flag and adjust projections for relievers who've lost the closer role.

    Rule: a pitcher projected for >=10 SV who has pitched in >=5 games with
    a save rate under 0.15 (less than 1 save per ~7 GP) is no longer closing.
    Their SV projection is reduced to DEMOTED_SV_RETENTION of the original.

    This is intentionally only applied to relievers, not starters. Adds a
    'role_status_inferred' column with values {"closer", "demoted", "unknown"}.
    """
    if pool.empty:
        return pool.copy()

    out = pool.copy()
    if "sv" in out.columns:
        out["sv"] = out["sv"].astype(float)
    if "role_status_inferred" not in out.columns:
        out["role_status_inferred"] = "unknown"

    is_hitter = out.get("is_hitter", pd.Series([1] * len(out))).fillna(1).astype(int)
    proj_sv = out.get("sv", pd.Series([0] * len(out))).fillna(0)
    ytd_gp = out.get("ytd_gp", pd.Series([0] * len(out))).fillna(0)
    ytd_sv = out.get("ytd_sv", pd.Series([0] * len(out))).fillna(0)

    is_pitcher = is_hitter == 0
    has_closer_projection = proj_sv >= MIN_PROJ_SV_FOR_ROLE_CHECK
    enough_gp = ytd_gp >= MIN_GP_FOR_ROLE_CHECK

    candidate_mask = is_pitcher & has_closer_projection & enough_gp
    if not candidate_mask.any():
        return out

    sv_rate = pd.Series(0.0, index=out.index)
    denom = ytd_gp.where(ytd_gp > 0, 1.0)
    sv_rate.loc[candidate_mask] = ytd_sv.loc[candidate_mask] / denom.loc[candidate_mask]

    demoted_mask = candidate_mask & (sv_rate < CLOSER_SV_RATE_FLOOR)
    confirmed_mask = candidate_mask & ~demoted_mask

    out.loc[confirmed_mask, "role_status_inferred"] = "closer"
    out.loc[demoted_mask, "role_status_inferred"] = "demoted"
    out.loc[demoted_mask, "sv"] = (out.loc[demoted_mask, "sv"] * DEMOTED_SV_RETENTION).clip(lower=float(DEMOTED_SV_MIN))

    return out


def apply_ytd_corrections(pool: pd.DataFrame) -> pd.DataFrame:
    """Convenience: run role-change detection then YTD blending.

    Order matters: role-change runs first so the SV cap is based on the
    original projection, and the blend then nudges the capped value with
    YTD sample weight.
    """
    out = detect_role_change(pool)
    out = blend_projection_with_ytd(out)
    return out
