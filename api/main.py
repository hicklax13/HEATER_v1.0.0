"""FastAPI app factory for the HEATER backend API (Sub-project B, Slices 1+2).

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

    from api.routers.free_agents import router as fa_router
    from api.routers.lineup import router as lineup_router
    from api.routers.standings import router as standings_router
    from api.routers.team import router as team_router

    app.include_router(team_router)
    app.include_router(fa_router)
    app.include_router(lineup_router)
    app.include_router(standings_router)
    return app
