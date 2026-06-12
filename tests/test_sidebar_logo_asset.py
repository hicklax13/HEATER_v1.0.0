"""The v2 sidebar logo asset must ship + be embeddable.

The sidebar logo is injected as a base64 data-URI built from assets/heater_logo_v2.png
(a transparent PNG keyed from the brand artwork). If the asset goes missing, the
injection silently falls back to a plain text wordmark — so guard its presence and
that it base64-encodes. .dockerignore must keep assets/ so it reaches Railway.
"""

import pathlib


def test_logo_asset_present():
    p = pathlib.Path("assets/heater_logo_v2.png")
    assert p.exists(), "assets/heater_logo_v2.png must exist (sidebar logo)"
    assert p.stat().st_size > 2000, "logo PNG looks truncated/empty"
    assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", "must be a real PNG (transparency)"


def test_logo_b64_helper_returns_encoding():
    from src.ui_shared import _heater_logo_b64

    b64 = _heater_logo_b64()
    assert b64 and len(b64) > 2000, "logo must base64-encode for the data-URI injection"
    # base64 alphabet only — safe to drop into the JS single-quoted img src
    assert all(c.isalnum() or c in "+/=" for c in b64)


def test_dockerignore_keeps_assets():
    text = pathlib.Path(".dockerignore").read_text(encoding="utf-8")
    # assets/ must not be excluded, or the logo won't reach the Railway image
    assert "\nassets" not in text and not text.startswith("assets"), "assets/ must ship in the image"
