"""FastAPI app factory for the HEATER backend API (Sub-project B).

Thin transport layer over the existing Python engines. Routers do auth/
validate/serialize only — all data work lives in api/services/.
"""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# The Next.js frontend calls this API from a different origin, so the browser
# requires CORS. Default to local Next.js dev origins; override per environment
# with HEATER_API_CORS_ORIGINS (comma-separated — e.g. the deployed Vercel domain).
_DEFAULT_CORS_ORIGINS = "http://localhost:3000,http://127.0.0.1:3000"


def _cors_origins() -> list[str]:
    raw = os.environ.get("HEATER_API_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def create_app() -> FastAPI:
    app = FastAPI(title="HEATER API", version="0.1.0")

    # Explicit origin allowlist (never "*") because credentials are allowed;
    # production origins are supplied via HEATER_API_CORS_ORIGINS at deploy time.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    from api.routers.billing import router as billing_router
    from api.routers.closers import router as closers_router
    from api.routers.compare import router as compare_router
    from api.routers.databank import router as databank_router
    from api.routers.draft import router as draft_router
    from api.routers.free_agents import router as fa_router
    from api.routers.leaders import router as leaders_router
    from api.routers.lineup import router as lineup_router
    from api.routers.matchup import router as matchup_router
    from api.routers.players import router as players_router
    from api.routers.playoff import router as playoff_router
    from api.routers.punt import router as punt_router
    from api.routers.roster_write import router as roster_write_router
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
    app.include_router(roster_write_router)
    app.include_router(draft_router)
    app.include_router(playoff_router)
    app.include_router(players_router)
    app.include_router(billing_router)
    return app
