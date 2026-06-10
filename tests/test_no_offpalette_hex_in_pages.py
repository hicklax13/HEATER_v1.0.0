"""No off-palette hex literals in pages/, app.py, or ui_analytics_badge (Combustion Finale, 2026-06-10).

The Combustion Index palette lives in src/ui_shared.py THEME + the --fp-* tokens.
Pages must not hardcode the Material/Tailwind-era colors that leaked in before the
finale sweep. News-source brand colors in My Team (ESPN #c41230, Yahoo #6001d2,
RotoWire #1a73e8, MLB #002d72) and TEAM_BRAND team colors are intentional and NOT
in the banned set. src/cheat_sheet.py (print artifact) is out of scope.
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

_SCANNED = sorted((_ROOT / "pages").glob("*.py")) + [
    _ROOT / "app.py",
    _ROOT / "src" / "ui_analytics_badge.py",
]

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


def test_no_offpalette_hex():
    hits: list[str] = []
    for path in _SCANNED:
        text = path.read_text(encoding="utf-8")
        for m in _HEX_RE.finditer(text):
            if m.group().lower() in _BANNED:
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(_ROOT)}:{line} {m.group()}")
    assert not hits, "Off-palette hex literals found (use THEME/--fp-* tokens):\n" + "\n".join(hits)
