"""Task 4.4 — Matchup Planner week-cap + label fixes.

Guards:
1. The week navigator forward-cap uses `config.season_weeks` (not the literal 24).
2. The "Tie" label in _build_win_prob_context_html is relabeled "Draw (6-6)".
3. The "Average Games" metric has an offline guard that shows explanatory text
   when no schedule data is available (i.e. games_count is all zeros).
"""

from __future__ import annotations

import re
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "pages" / "5_Matchup_Planner.py"


def _src() -> str:
    return PAGE.read_text(encoding="utf-8")


# ── 1. Week-cap uses config.season_weeks, not literal 24 ─────────────────────


def test_week_cap_not_hardcoded_24():
    """The forward week-nav cap must NOT be the literal integer 24."""
    src = _src()
    # Look for the pattern `< 24` (the old cap) adjacent to the next-week button
    assert "< 24" not in src, (
        "pages/5_Matchup_Planner.py still contains '< 24' as the week-nav cap. "
        "Replace with `< config.season_weeks` (or equivalent) so weeks 25-26 "
        "are reachable in a 26-week season."
    )


def test_week_cap_uses_season_weeks():
    """The forward week-nav cap must reference season_weeks from LeagueConfig."""
    src = _src()
    # Accept either `config.season_weeks` or `_season_weeks` / `SEASON_WEEKS`
    # (any name that captures it from LeagueConfig)
    has_season_weeks_cap = bool(
        re.search(r"<\s*config\.season_weeks", src)
        or re.search(r"<\s*_?season_weeks", src, re.IGNORECASE)
        or re.search(r"<\s*SEASON_WEEKS", src)
    )
    assert has_season_weeks_cap, (
        "The week-nav forward cap must use LeagueConfig.season_weeks (e.g. "
        "'if ... < config.season_weeks') so the cap is always league-correct "
        "(FourzynBurn = 26, not 24)."
    )


# ── 2. Draw label relabeled "Draw (6-6)" ────────────────────────────────────


def test_draw_label_not_called_tie():
    """The 6-6 draw probability must not be labeled 'Tie' in the HTML output.

    The original label 'Tie {pct}%' is confusing — H2H categories don't tie
    in the traditional sense; 6-6 is a draw.  The label must say 'Draw (6-6)'.
    """
    src = _src()
    # Make sure the bare "Tie " label is gone
    # It appeared as `Tie {tie_pct:.0f}%` — look for that pattern
    has_bare_tie = bool(re.search(r"['\"]Tie\s+\{tie_pct", src))
    assert not has_bare_tie, (
        "The 'Tie {tie_pct:.0f}%' label must be replaced with 'Draw (6-6) {tie_pct:.0f}%' "
        "(or similar) so the 6-6 split label is meaningful to users."
    )


def test_draw_6_6_label_present():
    """The 6-6 draw probability must use the label 'Draw (6-6)'."""
    src = _src()
    has_draw_label = bool(re.search(r"Draw\s*\(6-6\)", src) or re.search(r"Draw.*6.*6", src))
    assert has_draw_label, (
        "The 6-6 matchup probability label must be 'Draw (6-6)' (e.g. "
        "'Draw (6-6) {tie_pct:.0f}%') — update _build_win_prob_context_html."
    )


# ── 3. Offline "Average Games" guard ─────────────────────────────────────────


def test_avg_games_has_offline_guard():
    """The 'Average Games: 0.00' metric must be guarded with an explanation.

    When no schedule data is available (offline / no Yahoo connection),
    avg_games will be 0.0 and the metric is misleading.  The page must either:
    - conditionally render the metric only when avg_games > 0 (`if avg_games`), OR
    - add a help= kwarg or st.caption near the metric explaining the zero, OR
    - show a note when games_count is all-zero.
    """
    src = _src()
    lines = src.splitlines()

    # Find a line that mentions "Average Games" (may span a multi-line call)
    metric_line_idx = None
    for i, line in enumerate(lines):
        if "Average Games" in line:
            metric_line_idx = i
            break

    assert metric_line_idx is not None, (
        "Could not find 'Average Games' in the page. Expected an m4.metric('Average Games', ...) call."
    )

    # Check in a window around the metric call for a guard or explanation
    window_start = max(0, metric_line_idx - 5)
    window_end = min(len(lines), metric_line_idx + 15)
    window = "\n".join(lines[window_start:window_end])

    has_guard = bool(
        re.search(r"if\s+avg_games", window)
        or re.search(r"avg_games\s*>\s*0", window)
        or re.search(r"help\s*=", window)
        or re.search(r"st\.caption", window)
        or re.search(r"No schedule|no schedule|offline", window, re.IGNORECASE)
    )
    assert has_guard, (
        "When avg_games == 0.0 (offline / no schedule data), the 'Average Games' "
        "metric shows '0.00' with no explanation.  Add an `if avg_games > 0:` guard "
        "or a help= kwarg like help='0.00 when no schedule data is available' so users "
        "understand why the metric is zero.  Check around the m4.metric('Average Games', ...) call."
    )
