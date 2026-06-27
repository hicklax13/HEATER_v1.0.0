"""build_system_prompt mentions the attached page data + the explainer tools."""

from src.database import init_db


def test_system_prompt_mentions_explainers_and_page_data(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()
    from src.ai.chat import build_system_prompt

    sp = build_system_prompt("optimizer", "Team Hickey")
    assert "explain_metric" in sp
    assert "explain_constant" in sp
    assert "displayed on the page" in sp
    assert "Team Hickey" in sp and "optimizer" in sp
