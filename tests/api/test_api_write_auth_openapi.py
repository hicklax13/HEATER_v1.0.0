"""Contract guard: the gated write endpoints must ADVERTISE their 401 in the
OpenAPI schema. They are protected by `require_principal` (deny-by-default
bearer auth), but a `dependencies=[...]` gate does not auto-document the 401 —
so the frontend (which generates its client from api/openapi.json) would not
know these endpoints can reject an unauthenticated caller. This pins that the
401 stays documented on the two write routes."""

from api.main import create_app

_WRITE_PATHS = ("/api/lineup/set", "/api/transactions/add-drop")


def test_write_endpoints_document_401():
    schema = create_app().openapi()
    for path in _WRITE_PATHS:
        responses = schema["paths"][path]["post"]["responses"]
        assert "401" in responses, f"{path} POST must document a 401 (auth required)"
        assert responses["401"].get("description")


def test_read_endpoint_does_not_document_401():
    # Reads are not gated, so they should NOT advertise an auth 401.
    schema = create_app().openapi()
    responses = schema["paths"]["/api/standings"]["get"]["responses"]
    assert "401" not in responses
