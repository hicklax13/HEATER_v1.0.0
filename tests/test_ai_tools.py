"""Tool definitions are OpenAI-schema; dispatch routes to the service layer."""

import pytest

from src.database import init_db


@pytest.fixture(autouse=True)
def _db(monkeypatch):
    monkeypatch.setenv("MULTI_USER", "1")
    init_db()


def test_tool_specs_are_openai_schema():
    from src.ai.tools import tool_specs

    specs = tool_specs()
    names = {t["function"]["name"] for t in specs}
    assert {"query_data", "request_refresh"}.issubset(names)
    for t in specs:
        assert t["type"] == "function"
        assert "parameters" in t["function"]


def test_dispatch_query_data():
    from src.ai.tools import dispatch_tool

    out = dispatch_tool("query_data", {"sql": "SELECT 1 AS one"}, user_id=99)
    assert "one" in out  # serialized result contains the column


def test_dispatch_request_refresh_enqueues():
    from src.ai.refresh_queue import status_of
    from src.ai.tools import dispatch_tool

    out = dispatch_tool("request_refresh", {"source": "players"}, user_id=99)
    # the tool returns a request id we can check
    import json

    rid = json.loads(out)["request_id"]
    assert status_of(rid) == "pending"


def test_dispatch_unknown_tool_returns_error():
    from src.ai.tools import dispatch_tool

    out = dispatch_tool("does_not_exist", {}, user_id=99)
    assert "error" in out.lower()


def test_convenience_tools_present():
    from src.ai.tools import tool_specs

    names = {t["function"]["name"] for t in tool_specs()}
    assert {"get_my_team", "get_free_agents", "compare_players"}.issubset(names)


def test_search_tools_gated_by_flags():
    from src.ai.tools import tool_specs

    base = {t["function"]["name"] for t in tool_specs()}
    assert "web_search" not in base and "deep_research" not in base
    with_search = {t["function"]["name"] for t in tool_specs(web_search_enabled=True)}
    assert "web_search" in with_search and "deep_research" not in with_search
    with_both = {t["function"]["name"] for t in tool_specs(web_search_enabled=True, deep_research_enabled=True)}
    assert {"web_search", "deep_research"}.issubset(with_both)


def test_dispatch_web_search_when_mocked(monkeypatch):
    import json

    from src.ai import search
    from src.ai.tools import dispatch_tool

    monkeypatch.setattr(search, "_ddgs_text", lambda q, n: [{"title": "T", "href": "http://x", "body": "b"}])
    out = dispatch_tool("web_search", {"query": "acuna injury"}, user_id=99)
    data = json.loads(out)
    assert data["results"][0]["url"] == "http://x"


def test_dispatch_web_search_missing_query():
    import json

    from src.ai.tools import dispatch_tool

    out = dispatch_tool("web_search", {}, user_id=99)
    assert "error" in json.loads(out)
