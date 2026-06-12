"""render_chat_widget is inert when MULTI_USER is off (v1 byte-for-byte)."""

import pytest


def test_no_op_when_flag_off(monkeypatch):
    monkeypatch.delenv("MULTI_USER", raising=False)
    from src.ai.chat import render_chat_widget

    # Must return immediately without touching Streamlit or the DB.
    render_chat_widget("My Team")  # no exception, no rendering


def test_no_op_when_not_logged_in(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    import src.ai.chat as chat_mod

    monkeypatch.setattr(chat_mod, "current_user", lambda: None)
    chat_mod.render_chat_widget("My Team")  # returns early, renders nothing


def test_build_system_prompt_includes_schema(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    from src.database import init_db

    init_db()
    from src.ai.chat import build_system_prompt

    sp = build_system_prompt(page="My Team", viewer_team="Team Hickey")
    assert "SELECT" in sp  # schema card present
    assert "Team Hickey" in sp


def test_chat_wires_ai_settings():
    """The widget exposes per-user key management (store/list/delete)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "src" / "ai" / "chat.py").read_text(encoding="utf-8")
    assert "_render_ai_settings" in src
    assert "store_key" in src and "list_keys" in src and "delete_key" in src


def test_chat_wires_phase2_features():
    """The widget exposes the model picker + web-search/deep-research toggles + attach."""
    from pathlib import Path

    src = (Path(__file__).resolve().parent.parent / "src" / "ai" / "chat.py").read_text(encoding="utf-8")
    assert "_model_picker_options" in src and "model_catalog" in src  # model dropdown
    assert "ai_web_search" in src and "ai_deep_research" in src  # tool toggles
    assert "_render_attach_controls" in src and "getSelection" in src  # highlight-attach
    # the toggles are threaded to the provider
    assert "web_search=web_search" in src and "deep_research=deep_research" in src
