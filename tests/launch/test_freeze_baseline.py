from scripts.launch.freeze_baseline import (
    openapi_operation_count,
    route_inventory,
)

_FAKE_OPENAPI = {
    "paths": {
        "/healthz": {"get": {"operationId": "healthz_healthz_get"}},
        "/api/standings": {"get": {"operationId": "get_standings"}},
        "/api/lineup/set": {
            "post": {"operationId": "set_lineup"},
            "parameters": [],  # not an HTTP method — must be ignored
        },
    }
}


def test_openapi_operation_count_counts_only_http_methods():
    assert openapi_operation_count(_FAKE_OPENAPI) == 3


def test_route_inventory_is_sorted_by_path_and_typed():
    # Sorted by (path, method): "/api/lineup/set" < "/api/standings" < "/healthz".
    rows = route_inventory(_FAKE_OPENAPI)
    assert rows == [
        {"method": "POST", "path": "/api/lineup/set", "operation_id": "set_lineup"},
        {"method": "GET", "path": "/api/standings", "operation_id": "get_standings"},
        {"method": "GET", "path": "/healthz", "operation_id": "healthz_healthz_get"},
    ]


def test_route_inventory_ignores_non_method_keys():
    rows = route_inventory(_FAKE_OPENAPI)
    assert all(r["method"] in {"GET", "POST", "PUT", "PATCH", "DELETE"} for r in rows)
