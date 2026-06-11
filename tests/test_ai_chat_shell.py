"""The shell injects the launcher + window behaviors (string-level wiring check)."""


def test_shell_js_has_window_behaviors():
    from src.ai.chat_shell import _shell_script

    js = _shell_script(container_id="heater-ai-window", launcher_id="heater-ai-launcher")
    # drag, resize, minimize, close, launcher, persistence
    for needle in ["mousedown", "localStorage", "heater-ai-window", "heater-ai-launcher", "minimize", "close"]:
        assert needle in js, f"shell JS missing {needle!r}"


def test_launcher_label():
    from src.ai.chat_shell import LAUNCHER_LABEL

    assert LAUNCHER_LABEL == "AI Chat"


def test_launcher_markup_is_emoji_free():
    # the module source must not contain emoji (Combustion design-system rule)
    import pathlib

    import src.ai.chat_shell as shell
    from src.ai.chat_shell import render_launcher_and_shell  # import must not raise

    src = pathlib.Path(shell.__file__).read_text(encoding="utf-8")
    # no characters above the BMP basic range typical of emoji
    assert all(ord(ch) < 0x2190 for ch in src), "chat_shell.py must be emoji-free"
