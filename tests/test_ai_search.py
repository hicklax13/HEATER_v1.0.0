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


import pytest  # noqa: E402


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/x",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://10.0.0.5/internal",
        "http://192.168.1.1/admin",
        "ftp://example.com/file",  # non-http scheme
        "file:///etc/passwd",
        "http:///nohost",
        "not a url",
    ],
)
def test_fetcher_blocks_non_public_urls(url):
    from src.ai.search import _is_public_http_url

    assert _is_public_http_url(url) is False


def test_fetcher_allows_public_url(monkeypatch):
    import socket

    # mock DNS so there's no real network: resolve the host to a public IP
    monkeypatch.setattr(socket, "gethostbyname", lambda h: "93.184.216.34")
    from src.ai.search import _is_public_http_url

    assert _is_public_http_url("https://example.com/page") is True


def test_fetch_url_refuses_private(monkeypatch):
    """_fetch_url raises (caught upstream) on a non-public URL before any request."""
    from src.ai import search

    with pytest.raises(ValueError):
        search._fetch_url("http://169.254.169.254/")
