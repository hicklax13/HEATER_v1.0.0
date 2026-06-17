"""FastAPI app factory for the HEATER backend API (Sub-project B, Slice 1).

Thin transport layer over the existing Python engines. Routers do auth/
validate/serialize only — all data work lives in api/services/.
"""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(title="HEATER API", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    # team_router import added in Task 3; temporarily commented for Task 1 green run
    # from api.routers.team import router as team_router
    # app.include_router(team_router)
    return app
