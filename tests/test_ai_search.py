"""web_search + deep_research: shapes, graceful failure, fetch fallback. Mocked (no network)."""


def test_web_search_shapes(monkeypatch):
    from src.ai import search

    monkeypatch.setattr(search, "_ddgs_text", lambda q, n: [{"title": "T", "href": "http://x", "body": "snip"}])
    out = search.web_search("acuna")
    assert out["error"] is None
    assert out["results"][0] == {"title": "T", "url": "http://x", "snippet": "snip"}


def test_web_search_never_raises(monkeypatch):
    from src.ai import search

    def boom(q, n):
        raise RuntimeError("net down")

    monkeypatch.setattr(search, "_ddgs_text", boom)
    out = search.web_search("x")
    assert out["results"] == [] and out["error"]


def test_deep_research_fetches_then_snippets(monkeypatch):
    from src.ai import search

    monkeypatch.setattr(
        search,
        "_ddgs_text",
        lambda q, n: [
            {"title": "A", "href": "http://a", "body": "sa"},
            {"title": "B", "href": "http://b", "body": "sb"},
        ],
    )
    monkeypatch.setattr(search, "_fetch_url", lambda u: "FETCHED:" + u)
    out = search.deep_research("acuna trade", max_fetch=1)
    assert out["error"] is None
    assert out["sources"][0]["content"].startswith("FETCHED:")
    # the second result (beyond max_fetch) is snippet-only
    assert out["sources"][1]["content"] == "sb"


def test_deep_research_fetch_failure_falls_back_to_snippet(monkeypatch):
    from src.ai import search

    monkeypatch.setattr(search, "_ddgs_text", lambda q, n: [{"title": "A", "href": "http://a", "body": "snippetA"}])

    def boom(u):
        raise RuntimeError("403")

    monkeypatch.setattr(search, "_fetch_url", boom)
    out = search.deep_research("q", max_fetch=1)
    assert out["sources"][0]["content"] == "snippetA"


def test_deep_research_propagates_search_error(monkeypatch):
    from src.ai import search

    def boom(q, n):
        raise RuntimeError("net down")

    monkeypatch.setattr(search, "_ddgs_text", boom)
    out = search.deep_research("q")
    assert out["sources"] == [] and out["error"]
