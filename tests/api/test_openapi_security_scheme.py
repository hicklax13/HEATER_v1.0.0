from api.main import create_app


def test_openapi_documents_bearer_auth_scheme():
    schema = create_app().openapi()
    schemes = schema.get("components", {}).get("securitySchemes", {})
    assert "BearerAuth" in schemes
    assert schemes["BearerAuth"]["type"] == "http"
    assert schemes["BearerAuth"]["scheme"] == "bearer"
