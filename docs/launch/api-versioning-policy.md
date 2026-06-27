# HEATER API versioning & compatibility policy

**Status:** adopted 2026-06-26 (Phase 0b). The `/api/v1` prefix migration is
DEFERRED to the strangler-fig cutover (a coordinated frontend+backend change);
this policy governs change management until then.

## Surface

- Current surface: `/api/*` (unversioned), proxied server-side by the Next.js
  frontend (`web/next.config.ts`), so there are no browser cross-origin calls.
- The committed contract is `api/openapi.json`, snapshot-guarded by
  `tests/api/test_openapi_contract.py`; the frontend types are generated from it
  and drift-guarded by the `openapi-ts-sync` CI job.

## Compatibility rules

- **Additive changes** (new endpoint, new OPTIONAL field, new enum value behind a
  flag) do not require a version bump. Regenerate the snapshot + frontend types.
- **Breaking changes** (removing/renaming a field, tightening a type, changing a
  status code the frontend branches on — 401/402/409, or changing success shape)
  require: a migration window, a deprecation note in the operation description,
  and a coordinated frontend update in the same release train.
- **Errors** use the standard envelope (`api/errors.py`); error *bodies* may evolve
  freely because the frontend reads only the status code. Status codes are part of
  the contract and follow the breaking-change rule.

## Authentication

- Documented in OpenAPI as the `BearerAuth` scheme (Clerk session JWT). Per-operation
  security requirements are annotated progressively as each phase hardens its routes.

## When `/api/v1` lands (cutover)

- Introduce `/api/v1` as an alias, keep `/api/*` serving for one migration window,
  add `Deprecation`/`Sunset` headers to the unversioned surface, switch the frontend
  proxy, then remove the unversioned surface after the window.
