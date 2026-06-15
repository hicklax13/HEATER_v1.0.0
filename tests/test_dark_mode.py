"""TDD tests for dark-mode theme toggle (R-10).

Covers:
  1. inject_custom_css() with heater_theme="dark"  → dark-palette overrides emitted
  2. inject_custom_css() with heater_theme="light"  → default light output unchanged
  3. inject_custom_css() with no heater_theme key   → same as light (default-off)
  4. render_theme_toggle() exists, is callable, persists via save_view("ui","theme",...)
  5. app.py hydrates st.session_state["heater_theme"] from load_view("ui","theme")
  6. User-data failure in load_view → default "light", no crash
  7. Dark output uses only CSS variable names for theming (no banned off-palette hex)
"""

from __future__ import annotations

import ast
import importlib
import types
import unittest.mock as mock
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_APP_SRC = (_ROOT / "app.py").read_text(encoding="utf-8")
_UI_SRC = (_ROOT / "src" / "ui_shared.py").read_text(encoding="utf-8")


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_st_stub(session_state: dict | None = None):
    """Return a minimal Streamlit stub that captures markdown() calls."""
    captured: list[str] = []

    class _Sidebar:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def toggle(self, *a, **kw):
            return False

    class _St:
        def __init__(self):
            self.session_state = session_state or {}
            self.sidebar = _Sidebar()

        def markdown(self, body, *a, **kw):
            captured.append(body)

        def toggle(self, *a, **kw):
            return False

        def __getattr__(self, name):
            return lambda *a, **kw: None

    stub = _St()
    stub._captured = captured
    return stub


def _css_output(theme_mode: str | None = None) -> str:
    """Call inject_custom_css() with the given heater_theme and return the full CSS blob."""
    import src.ui_shared as ui_module

    session: dict = {}
    if theme_mode is not None:
        session["heater_theme"] = theme_mode

    stub = _make_st_stub(session)
    real_st = ui_module.st
    ui_module.st = stub  # type: ignore[assignment]
    try:
        ui_module.inject_custom_css()
    finally:
        ui_module.st = real_st

    return "".join(stub._captured)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Dark mode emits --fp-bg / canvas override to a dark value
# ══════════════════════════════════════════════════════════════════════════════


class TestDarkModeCSS:
    def test_dark_mode_emits_dark_canvas_override(self):
        """When heater_theme='dark', inject_custom_css() appends a dark-canvas override."""
        css = _css_output("dark")
        # Must contain a --fp-app-bg or --fp-surface override pointing to a dark value.
        # We check for both common token names and a pattern that looks dark.
        assert "--fp-app-bg" in css or "--fp-surface" in css, "dark mode must emit at least one canvas token override"
        # The dark-mode block must override the canvas to something dark.
        # Strategy: look for any hex that starts with #0|#1 (dark navy/near-black range)
        # OR explicit CSS var reassignment targeting bg.
        # Accept any of: "#0", "#1", "#2" prefix OR "dark" in any comment/selector near the override.
        import re

        dark_block_marker = re.search(
            r"heater_theme.*dark|dark.*mode|\.heater-dark|dark-palette|--fp-app-bg:\s*#[012]",
            css,
            re.IGNORECASE | re.DOTALL,
        )
        assert dark_block_marker is not None, (
            "dark-mode CSS must contain a recognisable dark-canvas override "
            "(expected '--fp-app-bg: #0xx' or a dark-mode selector/comment)"
        )

    def test_dark_mode_preserves_orange_accent(self):
        """Dark mode must keep orange (#ff6d00) as the primary accent."""
        css = _css_output("dark")
        # The primary token itself isn't reset (orange stays).
        # Either the light sheet still defines it, or dark sheet leaves it unchanged.
        assert "#ff6d00" in css, "dark mode must retain the Combustion orange accent #ff6d00"

    def test_dark_mode_emits_light_body_text_override(self):
        """Dark canvas needs lighter body text — dark mode block must override text color."""
        css = _css_output("dark")
        # Expect the dark block to set --fp-tx to a light value (e.g. #eef or similar).
        import re

        light_text_override = re.search(
            r"--fp-tx:\s*#[89a-fA-F][0-9a-fA-F]{5}|--fp-tx:\s*rgba\(2[0-9]{2}",
            css,
            re.IGNORECASE,
        )
        assert light_text_override is not None, "dark mode must override --fp-tx to a light colour (e.g. #eef1f6)"

    def test_dark_mode_extra_style_block_is_additive(self):
        """Dark mode adds CSS on top — the base light stylesheet must still be present."""
        css = _css_output("dark")
        # The base Combustion tokens (orange primary, Inter) must still be in the output.
        assert "--fp-primary: #ff6d00" in css, "base light stylesheet must be present in dark output"
        assert "Archivo" in css, "base font declarations must be present in dark output"

    def test_dark_mode_no_banned_hex(self):
        """Dark mode additions must not introduce banned off-palette hex literals.

        Only the dark-mode-specific <style> block is scanned (identified by the
        marker comment "HEATER Dark Mode overrides" that is emitted only in dark mode).
        The base light stylesheet may legitimately contain approved hex values.
        """
        css = _css_output("dark")

        import re

        # Extract only the dark-mode-specific style block (marked with R-10 comment).
        dark_block_match = re.search(
            r"HEATER Dark Mode overrides.*?</style>",
            css,
            re.DOTALL | re.IGNORECASE,
        )
        assert dark_block_match is not None, (
            "dark CSS output must contain the 'HEATER Dark Mode overrides' marker block"
        )
        dark_only = dark_block_match.group()

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
        hex_re = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
        hits = []
        for m in hex_re.finditer(dark_only):
            if m.group().lower() in _BANNED:
                hits.append(m.group())
        assert not hits, f"dark-mode style block contains banned hex: {hits}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. Default (light) output is byte-identical to today's baseline
# ══════════════════════════════════════════════════════════════════════════════


class TestLightModeUnchanged:
    def test_light_explicit_matches_no_key(self):
        """Explicit heater_theme='light' and absent key must produce identical CSS."""
        css_explicit = _css_output("light")
        css_absent = _css_output(None)
        assert css_explicit == css_absent, "explicit 'light' and absent heater_theme key must produce identical CSS"

    def test_light_mode_no_dark_override_block(self):
        """Light mode output must NOT contain a dark canvas override."""
        css = _css_output("light")
        # Dark canvas would set bg to #0xx or #1xx range.
        import re

        dark_canvas = re.search(r"--fp-app-bg:\s*#[012][0-9a-fA-F]{5}", css, re.IGNORECASE)
        assert dark_canvas is None, "light mode must not inject a dark canvas override"

    def test_light_mode_contains_combustion_tokens(self):
        """Light mode output contains all expected Combustion lock tokens."""
        css = _css_output("light")
        for token in ("--fp-app-bg", "--fp-primary", "--fp-navy", "--fp-sidebar-bg"):
            assert token in css, f"light mode must still emit {token}"


# ══════════════════════════════════════════════════════════════════════════════
# 3. render_theme_toggle() exists + persists via save_view
# ══════════════════════════════════════════════════════════════════════════════


class TestRenderThemeToggle:
    def test_render_theme_toggle_is_importable(self):
        """render_theme_toggle must be importable from src.ui_shared."""
        from src.ui_shared import render_theme_toggle  # noqa: F401

    def _make_toggle_stub(self, current_mode: str, toggle_returns: bool):
        """Build a Streamlit stub that simulates st.toggle returning toggle_returns.

        In real Streamlit, `with st.sidebar: st.toggle(...)` renders the widget
        in the sidebar but the call is still made on the `st` module object.
        The stub models this by making `st.toggle(...)` return the chosen value.
        """
        session: dict = {"heater_theme": current_mode}

        class _SidebarStub:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

        class _StStub:
            def __init__(self):
                self.session_state = session
                self.sidebar = _SidebarStub()

            def toggle(self, label="", value=False, **kw):
                return toggle_returns

            def __getattr__(self, name):
                return lambda *a, **kw: None

        return _StStub(), session

    def test_render_theme_toggle_calls_save_view_on_change(self):
        """When the toggle flips to dark, render_theme_toggle calls save_view('ui','theme',...)."""
        import src.ui_shared as ui_module

        stub, _session = self._make_toggle_stub("light", True)  # light -> dark

        saved: list[tuple] = []

        def _mock_save_view(kind, name, payload):
            saved.append((kind, name, payload))

        real_st = ui_module.st
        ui_module.st = stub  # type: ignore[assignment]
        try:
            with mock.patch("src.ui_shared.save_view", side_effect=_mock_save_view):
                ui_module.render_theme_toggle()
        finally:
            ui_module.st = real_st

        assert len(saved) >= 1, "render_theme_toggle must call save_view at least once on change"
        kind, name, payload = saved[0]
        assert kind == "ui", f"save_view kind must be 'ui', got {kind!r}"
        assert name == "theme", f"save_view name must be 'theme', got {name!r}"
        assert "mode" in payload, f"save_view payload must have 'mode' key, got {payload!r}"

    def test_render_theme_toggle_persists_dark_mode(self):
        """When dark is toggled, payload mode == 'dark'."""
        import src.ui_shared as ui_module

        stub, _session = self._make_toggle_stub("light", True)  # light -> dark

        saved: list[tuple] = []

        def _mock_save_view(kind, name, payload):
            saved.append((kind, name, payload))

        real_st = ui_module.st
        ui_module.st = stub  # type: ignore[assignment]
        try:
            with mock.patch("src.ui_shared.save_view", side_effect=_mock_save_view):
                ui_module.render_theme_toggle()
        finally:
            ui_module.st = real_st

        assert saved, "save_view must have been called"
        _, _, payload = saved[0]
        assert payload.get("mode") == "dark", f"persisted mode must be 'dark' when toggle is on, got {payload!r}"

    def test_render_theme_toggle_persists_light_mode(self):
        """When toggled off (light), payload mode == 'light'."""
        import src.ui_shared as ui_module

        stub, _session = self._make_toggle_stub("dark", False)  # dark -> light

        saved: list[tuple] = []

        def _mock_save_view(kind, name, payload):
            saved.append((kind, name, payload))

        real_st = ui_module.st
        ui_module.st = stub  # type: ignore[assignment]
        try:
            with mock.patch("src.ui_shared.save_view", side_effect=_mock_save_view):
                ui_module.render_theme_toggle()
        finally:
            ui_module.st = real_st

        assert saved, "save_view must have been called"
        _, _, payload = saved[0]
        assert payload.get("mode") == "light", f"persisted mode must be 'light' when toggle is off, got {payload!r}"

    def test_render_theme_toggle_no_crash_on_save_view_failure(self):
        """If save_view throws, render_theme_toggle must not crash."""
        import src.ui_shared as ui_module

        stub, _session = self._make_toggle_stub("light", True)  # triggers save path

        real_st = ui_module.st
        ui_module.st = stub  # type: ignore[assignment]
        try:
            with mock.patch("src.ui_shared.save_view", side_effect=RuntimeError("DB down")):
                # Must not raise
                ui_module.render_theme_toggle()
        finally:
            ui_module.st = real_st


# ══════════════════════════════════════════════════════════════════════════════
# 4. app.py hydrates heater_theme from load_view("ui","theme") on start
# ══════════════════════════════════════════════════════════════════════════════


class TestAppHydration:
    def test_app_imports_load_view(self):
        """app.py must import load_view from src.user_data."""
        assert "load_view" in _APP_SRC, "app.py must import load_view"

    def test_app_calls_load_view_for_theme(self):
        """app.py source must call load_view('ui', 'theme') or load_view(\"ui\", \"theme\")."""
        import re

        pattern = re.compile(
            r"""load_view\s*\(\s*['"]ui['"]\s*,\s*['"]theme['"]\s*\)""",
            re.DOTALL,
        )
        assert pattern.search(_APP_SRC), "app.py must call load_view('ui', 'theme') to hydrate the theme preference"

    def test_app_sets_heater_theme_in_session_state(self):
        """app.py must write heater_theme into st.session_state when a saved view is found."""
        assert "heater_theme" in _APP_SRC, "app.py must reference 'heater_theme' in session_state hydration"

    def test_app_hydration_defaults_to_light_on_failure(self):
        """If load_view raises / returns None, heater_theme must default to 'light' (no crash)."""
        # This is an AST / pattern check — look for a try/except or ternary
        # fallback in the vicinity of the load_view("ui","theme") call.
        import re

        # Accept either:
        # a) load_view(...) or {} / or None within a try block, or
        # b) x = load_view(...) followed by None check / fallback within 10 lines.
        # We just check that a safe default "light" appears near the load_view call.
        m = re.search(
            r"""load_view\s*\(\s*['"]ui['"]\s*,\s*['"]theme['"]\s*\)""",
            _APP_SRC,
        )
        assert m, "load_view call not found"
        context = _APP_SRC[max(0, m.start() - 50) : m.end() + 300]
        # Expect either "light" literal default OR try/except nearby
        has_default = '"light"' in context or "'light'" in context
        has_try = "try:" in context or "except" in context
        assert has_default or has_try, "app.py theme hydration must have a 'light' fallback or be wrapped in try/except"
