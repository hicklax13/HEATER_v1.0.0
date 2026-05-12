"""BUG-025 fix: Streamlit XSRF + CORS protections enabled."""

import re
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"


def test_xsrf_protection_enabled():
    """enableXsrfProtection should be true; false leaves the app vulnerable
    to CSRF (especially Yahoo OAuth form submits) when hosted beyond localhost."""
    assert CONFIG.exists()
    text = CONFIG.read_text(encoding="utf-8")
    m = re.search(r"enableXsrfProtection\s*=\s*(true|false)", text, re.IGNORECASE)
    assert m, "enableXsrfProtection setting missing from .streamlit/config.toml"
    assert m.group(1).lower() == "true", (
        f"BUG-025 regression: enableXsrfProtection = {m.group(1)}. "
        "Must be true; false exposes the app to CSRF when hosted beyond localhost."
    )


def test_cors_enabled():
    """enableCORS should be true (or absent — Streamlit's default is true).
    Combined with disabled XSRF, false is the BUG-025 high-severity config."""
    assert CONFIG.exists()
    text = CONFIG.read_text(encoding="utf-8")
    m = re.search(r"enableCORS\s*=\s*(true|false)", text, re.IGNORECASE)
    if m is None:
        return  # Default is true; no setting needed
    assert m.group(1).lower() == "true", (
        f"BUG-025 regression: enableCORS = {m.group(1)}. Set to true or remove the line (Streamlit default is true)."
    )
