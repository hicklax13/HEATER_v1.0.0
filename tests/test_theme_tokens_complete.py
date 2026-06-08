"""Guard: the FP-revamp design tokens exist in THEME (revamp task 1).

The HEATER UI styling derives from the THEME dict + the --fp-* CSS custom
properties emitted by inject_custom_css(). A missing THEME key crashes any page
that subscripts it (the "BR-4" crash class), so the token set is structurally
guarded here. This locks BOTH the kept "HEATER soul" keys and the new FP-style
neutrals introduced by the fantasypros.com-inspired revamp.
"""

import re

from src.ui_shared import THEME

# Kept (HEATER soul) + new (FP neutrals) tokens every renderer/page may reference.
_REQUIRED_TOKENS = (
    # kept
    "bg",
    "card",
    "border",
    "tx",
    "primary",
    "warn",
    "ink",
    # new FP neutrals
    "surface",
    "sidebar_bg",
    "sidebar_ink",
    "divider",
    "tx_muted",
    "tx_subtle",
)

_NEW_FP_NEUTRALS = (
    "surface",
    "sidebar_bg",
    "sidebar_ink",
    "divider",
    "tx_muted",
    "tx_subtle",
)

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def test_new_fp_tokens_present():
    """Every kept + new FP token must be a key in THEME."""
    missing = [k for k in _REQUIRED_TOKENS if k not in THEME]
    assert not missing, f"THEME missing token(s): {missing}"


def test_fp_neutral_tokens_are_hex():
    """The new FP neutral tokens must be real 6-digit hex colors (not placeholders)."""
    for k in _NEW_FP_NEUTRALS:
        val = THEME.get(k)
        assert isinstance(val, str) and _HEX_RE.match(val), f"THEME[{k!r}]={val!r} is not a #rrggbb hex color"


def test_primary_stays_heater_red():
    """The revamp keeps HEATER's red as the primary/action color (not FP yellow)."""
    assert THEME["primary"] == "#e63946", f"primary must stay HEATER red #e63946, got {THEME['primary']!r}"
