"""No off-palette hex literals in pages/, app.py, ui_analytics_badge, or src/ (Combustion Finale, 2026-06-10).

The Combustion Index palette lives in src/ui_shared.py THEME + the --fp-* tokens.
Pages and src/ modules must not hardcode the Material/Tailwind-era colors that
leaked in before the finale sweep.

Intentional allowlist (NOT banned):
- News-source brand colors in My Team: ESPN #c41230, Yahoo #6001d2, RotoWire
  #1a73e8, MLB #002d72.
- TEAM_BRAND MLB team colors in src/ui_shared.py (e.g. #BA0021 Angels, etc.).
- src/ui_shared.py itself is the THEME definition file — exempt from this scan.
- src/cheat_sheet.py is a print artifact — out of scope.
- src/injury_model.py health-dot colors (#fb923c moderate, #ff9f1c elevated,
  #f43f5e high-risk) are functional status colors distinct from the banned
  Material set; only the lime #84cc16 "Low Risk" dot is banned.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# ── Files scanned for the pages/app ban ────────────────────────────────
_PAGES_SCANNED = sorted((_ROOT / "pages").glob("*.py")) + [
    _ROOT / "app.py",
    _ROOT / "src" / "ui_analytics_badge.py",
]

# ── src/ files scanned for the extended ban (Task 5.3) ─────────────────
# Excludes ui_shared.py (THEME definition) and cheat_sheet.py (print artifact).
_SRC_EXEMPTED = {"ui_shared.py", "cheat_sheet.py"}
_SRC_SCANNED = [p for p in sorted((_ROOT / "src").glob("**/*.py")) if p.name not in _SRC_EXEMPTED]

_HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")

_BANNED = {
    "#ff9800",
    "#f97316",  # material/tailwind orange -> --fp-primary
    "#9e9e9e",
    "#6b7280",
    "#666666",
    "#666",  # material grays -> --fp-tx-muted
    "#9ca3af",  # tailwind gray-400 -> --fp-tx-subtle
    "#4caf50",
    "#22c55e",  # material/tailwind green -> THEME green
    "#84cc16",  # tailwind lime -> THEME green_l
    "#ef4444",  # tailwind red -> THEME danger
    "#2c2f36",  # near-charcoal drift -> --fp-tx
    "#f5f5f5",  # material gray-100 -> --fp-divider
    "#e8f5e9",  # material green-50 -> rgba green tint
    "#457b9d",
    "#9c27b0",
    "#999",  # legacy badge colors -> THEME sky/purple/tx_subtle
}

# ── Per-file hex allowlists (intentional brand / functional data) ───────
# Key = file stem; value = set of hex literals (lowercase) that are
# deliberately kept in that file because they represent brand identity
# or functional status signals, NOT palette drift.
_FILE_ALLOWLIST: dict[str, set[str]] = {
    # injury_model.py: health-risk CSS dot colors are a distinct semantic
    # set (amber/orange for moderate/elevated, red-rose for high risk).
    # Only #84cc16 (lime "Low Risk") is banned; others are intentional.
    "injury_model.py": set(),
    # player_tags.py: tag colors are player-annotation signals.
    # Only #6b7280 (Bust gray) is banned; others map to Combustion tokens.
    "player_tags.py": set(),
    # schedule_grid.py: tier colors are matchup-schedule signals.
    # Only #6b7280 (neutral) is banned; #2d6a4f (smash/green) is not
    # in the banned set.
    "schedule_grid.py": set(),
    # trade_value.py: TIER_COLORS drift — #457b9d + #666666 are banned.
    "trade_value.py": set(),
}


def _scan_for_banned(paths: list[Path]) -> list[str]:
    hits: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        file_allow = _FILE_ALLOWLIST.get(path.name, set())
        for m in _HEX_RE.finditer(text):
            hex_lower = m.group().lower()
            if hex_lower in _BANNED and hex_lower not in file_allow:
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(_ROOT)}:{line} {m.group()}")
    return hits


def test_no_offpalette_hex():
    """pages/, app.py, ui_analytics_badge must have no banned hex."""
    hits = _scan_for_banned(_PAGES_SCANNED)
    assert not hits, "Off-palette hex literals found (use THEME/--fp-* tokens):\n" + "\n".join(hits)


def test_no_offpalette_hex_in_src():
    """src/ modules (excluding ui_shared.py and cheat_sheet.py) must have no banned hex.

    Extended by Task 5.3 (2026-06-13) to catch Material/Tailwind-era drift
    in engine and utility modules.
    """
    hits = _scan_for_banned(_SRC_SCANNED)
    assert not hits, "Off-palette hex literals found in src/ (use THEME/--fp-* tokens):\n" + "\n".join(hits)
