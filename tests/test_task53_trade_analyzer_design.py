"""Task 5.3 — Trade Analyzer design-system drift regression guards.

Tests written FIRST (TDD) before the fixes:
1. Verdict banner must not contain 'slideUp' (persistent card — no entrance animation).
2. Expander labels must not contain emoji characters.
3. Internal engineering labels must not appear in user-facing copy.
4. The extended off-palette hex guard must catch banned hex in src/ (trade_value TIER_COLORS).
"""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TA_PATH = _ROOT / "pages" / "11_Trade_Analyzer.py"
_TV_PATH = _ROOT / "src" / "trade_value.py"

# ── Same banned-hex set as test_no_offpalette_hex_in_pages.py ──────────────
_HEX_RE = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
_BANNED = {
    "#ff9800",
    "#f97316",
    "#9e9e9e",
    "#6b7280",
    "#666666",
    "#666",
    "#9ca3af",
    "#4caf50",
    "#22c55e",
    "#84cc16",
    "#ef4444",
    "#2c2f36",
    "#f5f5f5",
    "#e8f5e9",
    "#457b9d",
    "#9c27b0",
    "#999",
}

# ── emoji detector ─────────────────────────────────────────────────────
_EMOJI_RE = re.compile(
    "["
    "\U00002600-\U000027bf"  # Misc symbols
    "\U0001f300-\U0001f9ff"  # Emoticons / misc symbols
    "\U0001fa00-\U0001faff"  # Symbols and pictographs extended
    "⌀-⏿"  # Misc technical
    "✀-➿"  # Dingbats
    "]+",
    flags=re.UNICODE,
)

# ── internal engineering label patterns that must not reach users ───────
# These are references to internal report sections and task codes.
# Only checked in non-comment lines (lines that don't start with '#'
# after stripping, and are inside string literals / HTML / help= text).
_INTERNAL_LABELS = [
    r"\bFeature 2\b",
    r"\bFeature 3\b",
    r"report B\.\d",
    r"report Q\(",
    r"report Section",
    r"\bP3\.5\b",
    r"\bTask 3\.\d",
]

# Lines that are pure Python comments are excluded from the check.
# User-visible content lives in strings (st.markdown, help=, st.warning, etc.);
# code comments starting with '#' are fine to keep for maintainability.
_COMMENT_LINE_RE = re.compile(r"^\s*#")


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ── 1. No slideUp on the persistent verdict banner ─────────────────────


def test_verdict_banner_no_slideup():
    """The verdict glass-div must NOT have animation:slideUp.

    Persistent cards must not replay an entrance animation on every Streamlit
    rerun (Combustion rule — see CLAUDE.md 'NO entrance keyframes on persistent
    cards').
    """
    text = _read(_TA_PATH)
    # Find the verdict banner glass-div — it contains 'verdict' and 'glass'
    # Look for slideUp within the same markdown() call as the verdict banner.
    assert "slideUp" not in text, (
        "pages/11_Trade_Analyzer.py: verdict banner contains 'slideUp' entrance "
        "animation. Persistent cards must not replay on every rerun — remove the "
        "animation property from the verdict glass-div."
    )


# ── 2. No emoji in expander labels ─────────────────────────────────────


def test_no_emoji_in_expander_labels():
    """st.expander() calls in the Trade Analyzer must not use emoji in their label."""
    text = _read(_TA_PATH)
    # Extract the label string passed to st.expander(...)
    expander_re = re.compile(r'st\.expander\(\s*["\']([^"\']*)["\']', re.MULTILINE)
    hits = []
    for m in expander_re.finditer(text):
        label = m.group(1)
        if _EMOJI_RE.search(label):
            hits.append(repr(label))
    assert not hits, (
        "pages/11_Trade_Analyzer.py: emoji found in st.expander() labels "
        "(Combustion rule — no emoji, use PAGE_ICONS SVGs instead):\n" + "\n".join(hits)
    )


# ── 3. No internal engineering labels in user-facing strings ───────────


def test_no_internal_labels_in_user_copy():
    """Engineering references like 'Feature 2', 'report B.5', 'P3.5' must not
    appear in user-facing strings (st.caption, st.markdown, help=, st.metric
    labels). Pure Python comment lines (starting with '#') are excluded.
    """
    text = _read(_TA_PATH)
    lines = text.splitlines()
    hits = []
    for pat in _INTERNAL_LABELS:
        for m in re.finditer(pat, text):
            line_no = text.count("\n", 0, m.start())  # 0-based index
            raw_line = lines[line_no] if line_no < len(lines) else ""
            # Skip pure comment lines — those are maintainer notes, not user copy
            if _COMMENT_LINE_RE.match(raw_line):
                continue
            hits.append(f"line {line_no + 1}: {m.group()!r}")
    assert not hits, (
        "pages/11_Trade_Analyzer.py: internal engineering labels found in "
        "user-facing strings — replace with plain language:\n" + "\n".join(hits)
    )


# ── 4. src/trade_value.py must not contain banned hex literals ─────────


def test_trade_value_no_banned_hex():
    """src/trade_value.py TIER_COLORS must not use banned Material/Tailwind hex."""
    text = _read(_TV_PATH)
    hits = []
    for m in _HEX_RE.finditer(text):
        if m.group().lower() in _BANNED:
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"src/trade_value.py:{line} {m.group()}")
    assert not hits, "Off-palette hex literals in src/trade_value.py (use THEME/--fp-* tokens):\n" + "\n".join(hits)
