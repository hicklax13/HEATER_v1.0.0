"""Generate the HEATER Trade Value Chart as a standalone HTML file.

This script is HEATER's primary SEO marketing asset. It pulls live player
valuations from the local SQLite database, computes trade values via the same
engine used in the Streamlit app, and renders a branded, sortable HTML table
for the top 200 players.

Usage:
    python content/trade-value-chart/generate_chart.py

Output:
    content/trade-value-chart/output/trade-values-YYYY-MM-DD.html
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — make sure HEATER's src/ package is importable whether this
# script is run from the repo root or from its own directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # content/trade-value-chart -> content -> repo root
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# HEATER imports
# ---------------------------------------------------------------------------
from src.database import init_db, load_player_pool  # noqa: E402
from src.trade_value import compute_trade_values  # noqa: E402
from src.valuation import LeagueConfig, value_all_players  # noqa: E402

# ---------------------------------------------------------------------------
# Brand constants
# ---------------------------------------------------------------------------
COLOR_FLAME = "#e63946"
COLOR_AMBER = "#ff6d00"
COLOR_GOLD = "#ffd60a"
COLOR_BG = "#f4f5f0"
COLOR_TEXT = "#1d1d1f"

TOP_N = 200
OUTPUT_DIR = _SCRIPT_DIR / "output"

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>HEATER Trade Value Chart &mdash; {date}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: {bg};
      color: {text};
      min-height: 100vh;
    }}

    /* ── Header ── */
    .site-header {{
      background: linear-gradient(135deg, {flame} 0%, {amber} 60%, {gold} 100%);
      padding: 2.5rem 1.5rem 2rem;
      text-align: center;
      box-shadow: 0 4px 24px rgba(230,57,70,.25);
    }}

    .site-header .logo-mark {{
      display: inline-block;
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.03em;
      color: #fff;
      text-shadow: 0 2px 8px rgba(0,0,0,.18);
      margin-bottom: 0.25rem;
    }}

    .site-header h1 {{
      font-size: 1.5rem;
      font-weight: 700;
      color: #fff;
      text-shadow: 0 1px 4px rgba(0,0,0,.15);
      margin-bottom: 0.4rem;
    }}

    .site-header .subtitle {{
      font-size: 0.875rem;
      font-weight: 500;
      color: rgba(255,255,255,.88);
      letter-spacing: 0.01em;
    }}

    /* ── Main content ── */
    .container {{
      max-width: 960px;
      margin: 2rem auto;
      padding: 0 1rem;
    }}

    /* ── Table ── */
    .table-wrapper {{
      background: #fff;
      border-radius: 12px;
      box-shadow: 0 2px 16px rgba(29,29,31,.08);
      overflow: hidden;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}

    thead {{
      background: {text};
      color: #fff;
    }}

    thead th {{
      padding: 0.85rem 1rem;
      text-align: left;
      font-weight: 600;
      font-size: 0.8rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      white-space: nowrap;
      cursor: pointer;
      user-select: none;
    }}

    thead th:hover {{
      background: #333;
    }}

    thead th::after {{
      content: ' \\25B8';
      opacity: 0.35;
      font-size: 0.7em;
    }}

    thead th.sorted-asc::after  {{ content: ' \\25B4'; opacity: 1; }}
    thead th.sorted-desc::after {{ content: ' \\25BE'; opacity: 1; }}

    tbody tr {{
      border-bottom: 1px solid #f0f0f0;
      transition: background 0.12s ease;
    }}

    tbody tr:last-child {{ border-bottom: none; }}

    tbody tr:hover {{
      background: #fff7ed;  /* light amber */
    }}

    tbody td {{
      padding: 0.7rem 1rem;
      vertical-align: middle;
    }}

    .col-rank {{
      width: 4rem;
      font-weight: 600;
      color: #888;
      font-size: 0.8rem;
    }}

    .col-player {{
      font-weight: 600;
      color: {text};
    }}

    .col-team {{
      color: #555;
      font-size: 0.85rem;
    }}

    .col-pos {{
      font-size: 0.8rem;
      font-weight: 600;
      color: #666;
    }}

    .col-value {{
      font-weight: 700;
      color: {flame};
      font-size: 1rem;
      text-align: right;
    }}

    thead th:last-child {{
      text-align: right;
    }}

    /* ── CTA Banner ── */
    .cta-banner {{
      margin: 2rem 0 3rem;
      background: linear-gradient(135deg, {flame} 0%, {amber} 100%);
      border-radius: 12px;
      padding: 2rem 2rem;
      text-align: center;
      box-shadow: 0 4px 20px rgba(230,57,70,.2);
    }}

    .cta-banner h2 {{
      font-size: 1.25rem;
      font-weight: 700;
      color: #fff;
      margin-bottom: 0.5rem;
    }}

    .cta-banner p {{
      font-size: 0.95rem;
      color: rgba(255,255,255,.9);
      margin-bottom: 1.25rem;
      font-style: italic;
    }}

    .cta-btn {{
      display: inline-block;
      background: #fff;
      color: {flame};
      font-weight: 700;
      font-size: 0.95rem;
      padding: 0.75rem 2rem;
      border-radius: 8px;
      text-decoration: none;
      letter-spacing: 0.01em;
      box-shadow: 0 2px 8px rgba(0,0,0,.12);
      transition: box-shadow 0.15s ease, transform 0.1s ease;
    }}

    .cta-btn:hover {{
      box-shadow: 0 4px 16px rgba(0,0,0,.18);
      transform: translateY(-1px);
    }}

    /* ── Footer ── */
    footer {{
      text-align: center;
      font-size: 0.78rem;
      color: #aaa;
      padding: 1rem 0 2rem;
    }}

    @media (max-width: 600px) {{
      .col-team {{ display: none; }}
    }}
  </style>
</head>
<body>

  <header class="site-header">
    <div class="logo-mark">HEATER</div>
    <h1>Trade Value Chart &mdash; {date}</h1>
    <p class="subtitle">Updated {date}&nbsp;&nbsp;|&nbsp;&nbsp;Powered by 10,000 Monte Carlo simulations&nbsp;&nbsp;|&nbsp;&nbsp;Top 200 Players</p>
  </header>

  <main class="container">

    <div class="table-wrapper">
      <table id="tv-table">
        <thead>
          <tr>
            <th class="col-rank" data-col="rank">Rank</th>
            <th data-col="player">Player</th>
            <th data-col="team">Team</th>
            <th data-col="position">Position</th>
            <th class="col-value sorted-desc" data-col="value">Trade Value</th>
          </tr>
        </thead>
        <tbody>
{rows}
        </tbody>
      </table>
    </div>

    <div class="cta-banner">
      <h2>Want personalized trade analysis for YOUR league?</h2>
      <p>Try HEATER Free &mdash; 10,000 simulations. One decision.</p>
      <a class="cta-btn" href="#">Launch HEATER &rarr;</a>
    </div>

  </main>

  <footer>
    &copy; {year} HEATER Fantasy Baseball &mdash; Values are estimates based on projected ROS performance and 12-team H2H categories scoring.
  </footer>

  <script>
    // Lightweight client-side sort — no external deps
    (function () {{
      var table = document.getElementById('tv-table');
      var headers = table.querySelectorAll('thead th');
      var tbody = table.querySelector('tbody');
      var sortCol = 'value';
      var sortAsc = false;

      headers.forEach(function (th) {{
        th.addEventListener('click', function () {{
          var col = th.getAttribute('data-col');
          if (col === sortCol) {{
            sortAsc = !sortAsc;
          }} else {{
            sortCol = col;
            sortAsc = (col !== 'value');  // value defaults desc, others asc
          }}
          headers.forEach(function (h) {{
            h.classList.remove('sorted-asc', 'sorted-desc');
          }});
          th.classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');
          sortTable(col, sortAsc);
        }});
      }});

      function sortTable(col, asc) {{
        var rows = Array.from(tbody.querySelectorAll('tr'));
        rows.sort(function (a, b) {{
          var aVal = a.querySelector('[data-col="' + col + '"]').getAttribute('data-val');
          var bVal = b.querySelector('[data-col="' + col + '"]').getAttribute('data-val');
          var aNum = parseFloat(aVal);
          var bNum = parseFloat(bVal);
          if (!isNaN(aNum) && !isNaN(bNum)) {{
            return asc ? aNum - bNum : bNum - aNum;
          }}
          return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
        }});
        rows.forEach(function (r) {{ tbody.appendChild(r); }});
        // Rewrite rank column after sort
        rows.forEach(function (r, i) {{
          r.querySelector('.col-rank').textContent = i + 1;
        }});
      }}
    }})();
  </script>

</body>
</html>
"""

_ROW_TEMPLATE = (
    "          <tr>"
    '<td class="col-rank" data-col="rank" data-val="{rank}">{rank}</td>'
    '<td class="col-player" data-col="player" data-val="{player}">{player}</td>'
    '<td class="col-team" data-col="team" data-val="{team}">{team}</td>'
    '<td class="col-pos" data-col="position" data-val="{position}">{position}</td>'
    '<td class="col-value" data-col="value" data-val="{value_raw}">{value}</td>'
    "</tr>"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_player_name(row) -> str:
    """Return best available player name from a pool row."""
    for col in ("player_name", "name", "full_name"):
        val = row.get(col, "")
        if val and str(val).strip():
            return str(val).strip()
    return "Unknown"


def _resolve_position(row) -> str:
    """Return best available position string from a pool row."""
    for col in ("position", "positions", "eligible_positions", "pos"):
        val = row.get(col, "")
        if val and str(val).strip():
            return str(val).strip()
    return "—"


def _format_trade_value(value: float) -> str:
    """Format a 0-100 trade value for display."""
    return f"{value:.1f}"


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------


def generate_chart() -> Path:
    """Build the trade value chart HTML and write it to the output directory.

    Returns:
        Path to the written HTML file.

    Raises:
        SystemExit: If the player pool is empty or cannot be loaded.
    """
    now_utc = datetime.now(UTC)
    date_str = now_utc.strftime("%Y-%m-%d")
    year_str = now_utc.strftime("%Y")

    # 1. Initialise database (no-op if tables already exist)
    print("Initialising database...")
    init_db()

    # 2. Load player pool
    print("Loading player pool...")
    pool = load_player_pool()

    if pool is None or pool.empty:
        print(
            "ERROR: Player pool is empty. Run `python load_sample_data.py` or bootstrap the app first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Loaded {len(pool):,} players from database.")

    # 3. Compute full valuations (SGP, VORP, pick_score)
    config = LeagueConfig()
    print("Computing player valuations (value_all_players)...")
    pool = value_all_players(pool, config)

    # 4. Compute trade values (0-100 scale)
    print("Computing trade values (compute_trade_values)...")
    tv_df = compute_trade_values(pool, config)

    if tv_df is None or tv_df.empty:
        print(
            "ERROR: compute_trade_values returned an empty result.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"  Trade values computed for {len(tv_df):,} players.")

    # 5. Take top N
    tv_df = tv_df.head(TOP_N).reset_index(drop=True)

    # 6. Build HTML rows
    rows_html_parts: list[str] = []
    for rank, row in enumerate(tv_df.itertuples(index=False), start=1):
        row_dict = row._asdict()

        player = _resolve_player_name(row_dict)
        team = str(row_dict.get("team", "—") or "—").strip() or "—"
        position = _resolve_position(row_dict)

        raw_value = float(row_dict.get("trade_value", 0) or 0)
        value_str = _format_trade_value(raw_value)

        rows_html_parts.append(
            _ROW_TEMPLATE.format(
                rank=rank,
                player=player,
                team=team,
                position=position,
                value_raw=f"{raw_value:.4f}",
                value=value_str,
            )
        )

    rows_html = "\n".join(rows_html_parts)

    # 7. Render full HTML
    html = _HTML_TEMPLATE.format(
        date=date_str,
        year=year_str,
        rows=rows_html,
        flame=COLOR_FLAME,
        amber=COLOR_AMBER,
        gold=COLOR_GOLD,
        bg=COLOR_BG,
        text=COLOR_TEXT,
    )

    # 8. Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"trade-values-{date_str}.html"
    output_path.write_text(html, encoding="utf-8")

    print(f"\nChart written to: {output_path}")
    print(f"  Players included: {len(tv_df)}")
    print(f"  Top player: {_resolve_player_name(tv_df.iloc[0].to_dict())} ({tv_df.iloc[0].get('trade_value', 0):.1f})")

    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_chart()
