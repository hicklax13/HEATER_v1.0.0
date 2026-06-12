"""Regression guard for the 2026-06-12 blank-page outage.

The floating chat window's frame must be scoped to its OWN container via
streamlit-float's float_parent(css=...). A hand-rolled global rule like
`div[data-testid="stVerticalBlock"]:has(> div #...-anchor) { position: fixed; }`
over-matches EVERY ancestor that contains the anchor — including the page's
top-level content block — pinning the whole page into the 380px chat box and
blanking the main column. These tests fail if that pattern ever returns.
"""

import ast
import pathlib


def test_float_window_css_has_no_global_frame_rule():
    from src.ai.chat_shell import CONTAINER_ID, float_window_css

    css = float_window_css()
    # The launcher button is legitimately position:fixed; the WINDOW FRAME is not.
    assert f"{CONTAINER_ID}-anchor" not in css, (
        "frame must not match the anchor in a global rule (it climbs to ancestors)"
    )
    assert ":has(" not in css, "no :has() in the window stylesheet (a descendant :has over-matches ancestors)"
    assert "380px" not in css, "window sizing belongs in window_frame_css (scoped via float_parent)"
    # Structural Streamlit selectors are allowed ONLY when scoped to the window
    # class — an unscoped stVerticalBlock rule would bleed onto every page block.
    for line in css.splitlines():
        if "stVerticalBlock" in line or "stLayoutWrapper" in line:
            assert ".heater-ai-window" in line, f"structural rule must be scoped to .heater-ai-window: {line.strip()}"


def test_window_frame_css_is_scoped_declarations():
    from src.ai.chat_shell import window_frame_css

    frame = window_frame_css()
    assert "<style" not in frame, "frame must be raw declarations for float_parent(css=...)"
    assert ":has(" not in frame, "frame must not carry its own :has() selector"
    assert "position: fixed" in frame
    assert "width: 380px" in frame
    assert "resize: both" in frame


def test_chat_widget_calls_float_parent_with_scoped_css():
    src = pathlib.Path("src/ai/chat.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "float_parent"
    ]
    assert calls, "render_chat_widget must call float_parent"
    for c in calls:
        assert c.args or c.keywords, "float_parent must be called WITH scoped css, never bare"


def test_window_is_flex_column_with_capped_height():
    """The window is a flex column (input pins to the bottom), height capped to the
    viewport — NOT the old fixed 540px box that left dead space below the input."""
    from src.ai.chat_shell import window_frame_css

    frame = window_frame_css()
    assert "height: 540px" not in frame, "fixed 540px with a mid-window input left dead space"
    assert "flex-direction: column" in frame, "window must be a flex column to pin the input"
    assert "82vh" in frame or "max-height" in frame, "window height must be capped to the viewport"


def test_internal_layout_pins_input_to_bottom():
    """float_window_css scopes the transcript-fills / input-at-bottom rules to the
    JS-tagged .heater-ai-window, and the shell tags the window with that class."""
    from src.ai.chat_shell import _shell_script, float_window_css

    css = float_window_css()
    assert ".heater-ai-window" in css, "layout rules must be scoped to the window class"
    assert "stChatInput" in css, "the chat input must be positioned by the layout rules"
    assert "flex: 1 1 auto" in css, "the transcript must flex to fill the middle"
    assert "heater-ai-window" in _shell_script(), "the shell must tag the window with the class"


def test_shell_does_not_persist_window_size():
    """Size is NOT saved to localStorage: persisting the initial 540px froze the
    window and reintroduced the dead-space bug. Position + open state still persist."""
    from src.ai.chat_shell import _shell_script

    js = _shell_script()
    assert "ResizeObserver" not in js, "size-persisting ResizeObserver must stay removed"
    assert "s.height = r.height" not in js and "s.width = r.width" not in js, "must not save window size"
    assert "s.left" in js and "s.open" in js, "position + open state still persist"
