"""Pure, self-contained helper functions extracted from pages/1_My_Team.py.

All functions here are:
- Stateless and side-effect-free (no Streamlit calls, no DB access)
- Independent of page-local closure state
- Testable in isolation

Moved here to reduce the size of pages/1_My_Team.py while keeping
all structural-invariant guards passing (the page still imports every
symbol these functions depended on; guards that scan the *page* file
for patterns are unaffected because the page now has import lines for
these names instead of their definitions).
"""

from __future__ import annotations

import pandas as pd

from src.ui_shared import THEME, format_stat

T = THEME

# -- Source badge colors by provider --
_SOURCE_BADGE_COLORS = {
    "espn": {"bg": "#c41230", "label": "ESPN"},
    "rotowire": {"bg": "#1a73e8", "label": "RotoWire"},
    "mlb": {"bg": "#002d72", "label": "MLB"},
    "yahoo": {"bg": "#6001d2", "label": "Yahoo"},
}

# -- Sentiment indicator thresholds --
_SENTIMENT_THRESHOLDS = [
    (0.2, T["green"], "Positive"),
    (-0.2, T["warn"], "Neutral"),
    (float("-inf"), T["danger"], "Negative"),
]


def sentiment_indicator(score: float) -> str:
    """Return an HTML span with colored dot and label for a sentiment score."""
    for threshold, color, label in _SENTIMENT_THRESHOLDS:
        if score >= threshold:
            return (
                f'<span style="display:inline-flex;align-items:center;gap:4px;">'
                f'<span style="width:8px;height:8px;border-radius:50%;'
                f'background:{color};display:inline-block;"></span>'
                f'<span style="font-size:12px;color:{T["tx2"]};">{label}</span></span>'
            )
    # Fallback (should not reach here)
    return ""


def ownership_arrow(direction: str, delta: float) -> str:
    """Return an HTML snippet for ownership trend arrow and delta."""
    if direction == "up":
        color = T["green"]
        arrow = f'<span style="color:{color};font-weight:700;">&#9650;</span>'
    elif direction == "down":
        color = T["danger"]
        arrow = f'<span style="color:{color};font-weight:700;">&#9660;</span>'
    else:
        color = T["tx2"]
        arrow = f'<span style="color:{color};font-weight:700;">&#8212;</span>'
    delta_str = f"{delta:+.1f}%" if delta != 0.0 else "0.0%"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:3px;">'
        f'{arrow} <span style="font-size:12px;color:{color};">{delta_str}</span></span>'
    )


def source_badge(source: str) -> str:
    """Return an HTML badge for the news source."""
    info = _SOURCE_BADGE_COLORS.get(source, {"bg": T["tx2"], "label": source.upper()})
    return (
        f'<span style="display:inline-block;padding:1px 8px;border-radius:10px;'
        f"background:{info['bg']};color:#ffffff;font-size:11px;font-weight:700;"
        f'letter-spacing:0.5px;vertical-align:middle;">{info["label"]}</span>'
    )


def news_type_label(news_type: str) -> str:
    """Return a styled label for the news type."""
    type_map = {
        "injury": {"color": T["danger"], "label": "Injury"},
        "transaction": {"color": T["purple"], "label": "Transaction"},
        "callup": {"color": T["green"], "label": "Call-Up"},
        "lineup": {"color": T["sky"], "label": "Lineup"},
        "general": {"color": T["tx2"], "label": "General"},
    }
    info = type_map.get(news_type, type_map["general"])
    return (
        f'<span style="font-size:11px;font-weight:600;color:{info["color"]};'
        f'text-transform:uppercase;letter-spacing:0.5px;">{info["label"]}</span>'
    )


def compute_category_totals(df: pd.DataFrame) -> tuple[dict, dict]:
    """Compute hitting and pitching category totals from a DataFrame with is_hitter column.

    Returns (hit_stats, pitch_stats) dicts with display-ready values.
    """
    num_cols = [
        "r",
        "hr",
        "rbi",
        "sb",
        "ab",
        "h",
        "bb",
        "hbp",
        "sf",
        "w",
        "l",
        "sv",
        "k",
        "ip",
        "er",
        "bb_allowed",
        "h_allowed",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    hitters = df[df["is_hitter"] == 1]
    pitchers = df[df["is_hitter"] == 0]

    hit_stats: dict = {}
    if not hitters.empty:
        for cat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            hit_stats[cat] = int(hitters[col].sum()) if col in hitters.columns else 0
        ab = hitters["ab"].sum() if "ab" in hitters.columns else 0
        h = hitters["h"].sum() if "h" in hitters.columns else 0
        hit_stats["AVG"] = format_stat(h / ab, "AVG") if ab > 0 else ".000"
        hit_bb = hitters["bb"].sum() if "bb" in hitters.columns else 0
        hit_hbp = hitters["hbp"].sum() if "hbp" in hitters.columns else 0
        hit_sf = hitters["sf"].sum() if "sf" in hitters.columns else 0
        obp_denom = ab + hit_bb + hit_hbp + hit_sf
        hit_stats["OBP"] = format_stat((h + hit_bb + hit_hbp) / obp_denom, "OBP") if obp_denom > 0 else ".000"

    pitch_stats: dict = {}
    if not pitchers.empty:
        for cat, col in [("W", "w"), ("L", "l"), ("SV", "sv"), ("K", "k")]:
            pitch_stats[cat] = int(pitchers[col].sum()) if col in pitchers.columns else 0
        ip = pitchers["ip"].sum() if "ip" in pitchers.columns else 0
        er = pitchers["er"].sum() if "er" in pitchers.columns else 0
        bb = pitchers["bb_allowed"].sum() if "bb_allowed" in pitchers.columns else 0
        ha = pitchers["h_allowed"].sum() if "h_allowed" in pitchers.columns else 0
        pitch_stats["ERA"] = format_stat(er * 9 / ip, "ERA") if ip > 0 else "0.00"
        pitch_stats["WHIP"] = format_stat((bb + ha) / ip, "WHIP") if ip > 0 else "0.00"

    return hit_stats, pitch_stats


def rank_priority_losing_cats(
    gap_rows: list[dict],
    sigmas: dict[str, float],
    top_n: int = 2,
) -> list[dict]:
    """Rank the losing categories by NORMALIZED closeness-to-flip (BR-2b).

    The "Priority Targets" callout should surface the most ACTIONABLE losing
    categories — the ones closest to flipping — not the ones with the biggest
    raw gap. Sorting losing cats by raw ``diff`` mixes counting cats (R behind
    by 6) with rate cats (AVG behind by 0.069) on incompatible scales, so the
    big-count cats almost always sort first and a winnable rate cat never
    surfaces.

    Each losing cat's gap is normalized by that category's weekly standard
    deviation (the canonical ``h2h_engine.default_weekly_sigmas()``, keyed by
    uppercase cat) to a unit-free z-gap ``|diff| / sigma``. The SMALLEST
    |z-gap| losing cats are the closest to flipping = the real priority
    targets. Inverse cats use the gap MAGNITUDE (the ``diff`` is already
    oriented so positive = winning), so ``abs`` is correct for all cats. A
    category with no sigma entry is treated as an infinitely-wide gap (sorted
    last) rather than crashing.

    Args:
        gap_rows: Per-category gap rows (each carrying ``cat``, ``diff``,
            ``above``, ``tied``).
        sigmas: Uppercase-keyed per-team weekly category standard deviations.
        top_n: Number of priority targets to return.

    Returns:
        Up to ``top_n`` losing gap rows, ordered closest-to-flip first.
    """
    losing = [r for r in gap_rows if not r["above"] and not r["tied"]]

    def _z_gap(row: dict) -> float:
        sigma = sigmas.get(row["cat"])
        if not sigma or sigma <= 0:
            return float("inf")  # no scale -> treat as far from flipping
        return abs(float(row["diff"])) / float(sigma)

    return sorted(losing, key=_z_gap)[:top_n]
