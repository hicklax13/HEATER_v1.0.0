"""Closer depth chart monitor with job security scoring."""

from __future__ import annotations

import pandas as pd


def _compute_gmli_component(gmli: float) -> float:
    """Convert gmLI to a 0-1 trust component via piecewise linear interpolation.

    Thresholds:
    - gmli >= 1.8: 1.0 (high-leverage usage = trusted closer)
    - gmli  = 1.0: 0.5 (average leverage = questionable)
    - gmli <= 0.5: 0.0 (low leverage = not trusted)
    """
    gmli = max(0.0, gmli)
    if gmli >= 1.8:
        return 1.0
    elif gmli >= 1.0:
        # Linear from 0.5 (at 1.0) to 1.0 (at 1.8)
        return 0.5 + (gmli - 1.0) / (1.8 - 1.0) * 0.5
    elif gmli >= 0.5:
        # Linear from 0.0 (at 0.5) to 0.5 (at 1.0)
        return (gmli - 0.5) / (1.0 - 0.5) * 0.5
    else:
        return 0.0


def compute_job_security(
    hierarchy_confidence: float,
    projected_sv: float,
    gmli: float | None = None,
    gmli_prev: float | None = None,
) -> float:
    """Compute closer job security score [0, 1].

    Original formula (gmli=None):
        security = 0.6 * hierarchy + 0.4 * saves_component

    Enhanced formula (gmli provided):
        security = 0.45 * hierarchy + 0.30 * saves_component + 0.25 * gmli_component

    gmli_trend penalty:
        If gmli_prev provided and gmli dropped > 0.5: additional -0.10 penalty.
    """
    sv_component = min(1.0, max(0.0, projected_sv) / 30.0)
    hierarchy_clamped = max(0.0, min(1.0, hierarchy_confidence))

    if gmli is None:
        # Original formula — backward compatible
        raw = 0.6 * hierarchy_clamped + 0.4 * sv_component
    else:
        gmli_component = _compute_gmli_component(gmli)
        raw = 0.45 * hierarchy_clamped + 0.30 * sv_component + 0.25 * gmli_component

        # Trend penalty: large drop signals demotion risk
        if gmli_prev is not None and (gmli_prev - gmli) > 0.5:
            raw -= 0.10

    return max(0.0, min(1.0, raw))


def get_security_color(security: float) -> str:
    """Return color hex based on job security level.

    >= 0.7 -> green (#2d6a4f)
    >= 0.4 -> yellow (#ff9f1c)
    <  0.4 -> red (#e63946)
    """
    if security >= 0.7:
        return "#2d6a4f"
    elif security >= 0.4:
        return "#ff9f1c"
    else:
        return "#e63946"


def compute_skill_decay(
    season_k_pct: float,
    recent_k_pct: float,
    season_kbb_pct: float,
    recent_kbb_pct: float,
) -> dict:
    """Detect closer skill decay from K% and K-BB% trends.

    Args:
        season_k_pct: Season-average strikeout rate.
        recent_k_pct: Rolling L14 strikeout rate.
        season_kbb_pct: Season-average K-BB%.
        recent_kbb_pct: Rolling L14 K-BB%.

    Returns:
        Dict with k_pct_drop, kbb_warning, severity, and message.
    """
    k_drop = season_k_pct - recent_k_pct
    kbb_low = recent_kbb_pct < 10.0

    if k_drop >= 8.0 or kbb_low:
        severity = "CRITICAL"
        if k_drop >= 8.0:
            msg = f"K% dropped {k_drop:.1f} pts"
        else:
            msg = f"K-BB% at {recent_kbb_pct:.1f}% (danger zone)"
    elif k_drop >= 5.0:
        severity = "WARNING"
        msg = f"K% declining ({k_drop:.1f} pts from season avg)"
    else:
        severity = "NONE"
        msg = ""

    return {
        "k_pct_drop": round(k_drop, 1),
        "kbb_warning": kbb_low,
        "severity": severity,
        "message": msg,
    }


def build_closer_grid(
    depth_data: dict,
    player_pool: pd.DataFrame | None = None,
) -> list[dict]:
    """Build 30-team closer grid from depth chart data.

    Args:
        depth_data: Dict keyed by team code, each value has keys:
            - closer: str (closer name or "Committee")
            - setup: list[str] (setup arm names)
            - closer_confidence: float [0, 1]
        player_pool: Optional DataFrame with player stats (name, team, sv, era, whip).

    Returns:
        List of dicts sorted by team code, each with:
        team, closer_name, setup_names, job_security, security_color,
        projected_sv, era, whip.
    """
    if not depth_data:
        return []

    grid = []
    for team, info in sorted(depth_data.items()):
        closer_name = info.get("closer", "Unknown")
        setup_names = info.get("setup", [])
        confidence = float(info.get("closer_confidence", 0.5))
        projected_sv = 0.0
        era = 0.0
        whip = 0.0

        mlb_id = None
        if player_pool is not None and not player_pool.empty:
            match = player_pool[(player_pool["name"] == closer_name) & (player_pool["team"] == team)]
            if not match.empty:
                row = match.iloc[0]
                _sv = row.get("sv", 0)
                projected_sv = 0.0 if pd.isna(_sv) else float(_sv or 0)
                _era = row.get("era", 0)
                era = 0.0 if pd.isna(_era) else float(_era or 0)
                _whip = row.get("whip", 0)
                whip = 0.0 if pd.isna(_whip) else float(_whip or 0)
                mlb_id = row.get("mlb_id")

        security = compute_job_security(confidence, projected_sv)
        grid.append(
            {
                "team": team,
                "closer_name": closer_name,
                "setup_names": setup_names,
                "job_security": round(security, 3),
                "security_color": get_security_color(security),
                "projected_sv": projected_sv,
                "era": era,
                "whip": whip,
                "mlb_id": mlb_id,
            }
        )

    return grid
