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

    from api.routers.closers import router as closers_router
    from api.routers.compare import router as compare_router
    from api.routers.databank import router as databank_router
    from api.routers.free_agents import router as fa_router
    from api.routers.leaders import router as leaders_router
    from api.routers.lineup import router as lineup_router
    from api.routers.matchup import router as matchup_router
    from api.routers.punt import router as punt_router
    from api.routers.standings import router as standings_router
    from api.routers.streaming import router as streaming_router
    from api.routers.team import router as team_router
    from api.routers.trade import router as trade_router
    from api.routers.trade_finder import router as trade_finder_router

    app.include_router(team_router)
    app.include_router(fa_router)
    app.include_router(lineup_router)
    app.include_router(standings_router)
    app.include_router(closers_router)
    app.include_router(leaders_router)
    app.include_router(matchup_router)
    app.include_router(streaming_router)
    app.include_router(punt_router)
    app.include_router(trade_router)
    app.include_router(trade_finder_router)
    app.include_router(compare_router)
    app.include_router(databank_router)
    return app
