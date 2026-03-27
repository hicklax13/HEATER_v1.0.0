"""HTML/PDF cheat sheet generation with print CSS."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

WEASYPRINT_AVAILABLE = False
try:
    import weasyprint  # noqa: F401

    WEASYPRINT_AVAILABLE = True
except ImportError:
    pass


@dataclass
class CheatSheetOptions:
    sort_by: str = "pick_score"
    positions: list[str] = field(default_factory=lambda: ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"])
    show_percentiles: bool = True
    show_health_badges: bool = True
    show_tiers: bool = True
    top_n_per_position: int = 20
    paper_size: str = "letter"


_PRINT_CSS = """
@media print {
    body { font-size: 10px; margin: 0.5in; }
    .page-break { page-break-before: always; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ccc; padding: 3px 6px; text-align: left; }
    th { background: #1a1a2e; color: #fff; }
    .tier-break { border-top: 3px solid #e63946; }
    .tag-badge { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 9px; color: #fff; }
}
"""

_SCREEN_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f4f5f0; }
h1 { color: #1a1a2e; border-bottom: 3px solid #e63946; padding-bottom: 8px; }
h2 { color: #e63946; margin-top: 24px; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; background: #fff; border-radius: 8px; overflow: hidden; }
th { background: linear-gradient(135deg, #1a1a2e, #16213e); color: #fff; padding: 8px 10px; text-align: left; font-size: 12px; }
td { padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 12px; }
tr:hover { background: rgba(230, 92, 0, 0.05); }
.tier-break td { border-top: 3px solid #e63946; }
.tag-badge { display: inline-block; padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: 600; color: #fff; margin-left: 4px; }
.health-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 4px; }
"""


def _player_row_html(player: dict, show_health: bool = True, show_percentiles: bool = True) -> str:
    """Generate a single player row."""
    name = player.get("name", "")
    team = player.get("team", "")
    pos = player.get("positions", "")
    score = player.get("pick_score", 0)
    adp = player.get("adp", 0)
    tags_html = ""
    for tag in player.get("tags", []):
        color = {
            "Sleeper": "#6c63ff",
            "Target": "#2d6a4f",
            "Avoid": "#e63946",
            "Breakout": "#ff6d00",
            "Bust": "#6b7280",
        }.get(tag, "#6b7280")
        tags_html += f'<span class="tag-badge" style="background:{color};">{tag}</span>'
    health_html = ""
    if show_health:
        hs = player.get("health_score", 0.85)
        color = "#2d6a4f" if hs >= 0.9 else "#ff9f1c" if hs >= 0.7 else "#e63946"
        health_html = f'<span class="health-dot" style="background:{color};"></span>'
    pct_html = ""
    if show_percentiles:
        p10 = player.get("p10", "")
        p90 = player.get("p90", "")
        if p10 or p90:
            pct_html = f'<small style="color:#6b7280;">{p10}-{p90}</small>'
    return (
        f"<td>{health_html}{name}{tags_html}</td><td>{team}</td><td>{pos}</td>"
        f"<td>{score:.2f}</td><td>{adp}</td><td>{pct_html}</td>"
    )


def _tier_break_html() -> str:
    return '<tr class="tier-break"><td colspan="6"></td></tr>'


def generate_cheat_sheet_html(
    player_pool: pd.DataFrame,
    options: CheatSheetOptions | None = None,
    tags_by_player: dict[int, list[str]] | None = None,
    health_scores: dict[int, float] | None = None,
) -> str:
    """Generate full HTML cheat sheet document."""
    opts = options or CheatSheetOptions()
    tags_lookup = tags_by_player or {}
    health_lookup = health_scores or {}
    sort_col = opts.sort_by if opts.sort_by in player_pool.columns else "pick_score"
    if sort_col not in player_pool.columns:
        sort_col = player_pool.columns[0] if len(player_pool.columns) > 0 else None
    if sort_col is not None:
        ascending = sort_col in ("adp",)  # Lower ADP = better
        sorted_pool = player_pool.sort_values(sort_col, ascending=ascending).reset_index(drop=True)
    else:
        sorted_pool = player_pool.reset_index(drop=True)
    sections_html = ""
    # Overall rankings
    sections_html += "<h2>Overall Rankings</h2>"
    sections_html += _rankings_table(sorted_pool.head(opts.top_n_per_position * 2), opts, tags_lookup, health_lookup)
    # Per-position rankings
    for pos in opts.positions:
        pos_col = "positions" if "positions" in sorted_pool.columns else None
        if pos_col:
            pos_pool = sorted_pool[sorted_pool[pos_col].str.contains(pos, case=False, na=False)]
        else:
            pos_pool = sorted_pool
        if not pos_pool.empty:
            sections_html += f'<h2 class="page-break">{pos} Rankings</h2>'
            sections_html += _rankings_table(pos_pool.head(opts.top_n_per_position), opts, tags_lookup, health_lookup)
    return (
        f"<!DOCTYPE html>\n"
        f'<html><head><meta charset="utf-8"><title>HEATER Cheat Sheet</title>\n'
        f"<style>{_SCREEN_CSS}{_PRINT_CSS}</style></head>\n"
        f"<body><h1>HEATER Draft Cheat Sheet</h1>{sections_html}</body></html>"
    )


def _rankings_table(
    df: pd.DataFrame,
    opts: CheatSheetOptions,
    tags: dict,
    health: dict,
) -> str:
    header = "<tr><th>Player</th><th>Team</th><th>Pos</th><th>Score</th><th>ADP</th><th>Range</th></tr>"
    rows = []
    prev_tier = None
    for _, row in df.iterrows():
        pid = int(row.get("player_id", 0)) if "player_id" in row.index else 0
        score_val = float(row.get(opts.sort_by, 0)) if opts.sort_by in row.index else 0.0
        player = {
            "name": row.get("name", row.get("player_name", "")),
            "team": row.get("team", ""),
            "positions": row.get("positions", ""),
            "pick_score": score_val,
            "adp": int(row.get("adp", 0)) if "adp" in row.index else "",
            "tags": tags.get(pid, []),
            "health_score": health.get(pid, 0.85),
        }
        # Tier breaks every 20 players
        tier = len(rows) // 20
        if prev_tier is not None and tier != prev_tier and opts.show_tiers:
            rows.append(_tier_break_html())
        prev_tier = tier
        rows.append(f"<tr>{_player_row_html(player, opts.show_health_badges, opts.show_percentiles)}</tr>")
    return f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


def generate_cheat_sheet_pdf(html_string: str) -> bytes | None:
    """Convert HTML to PDF bytes. Returns None if weasyprint unavailable."""
    if not WEASYPRINT_AVAILABLE:
        return None
    try:
        import weasyprint

        return weasyprint.HTML(string=html_string).write_pdf()
    except Exception:
        return None
