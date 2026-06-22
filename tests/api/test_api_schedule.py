"""Endpoint tests for /api/schedule/probables and /api/schedule/hitter-matchups (DB-free)."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.contracts.schedule import (
    HitterMatchupCell,
    HitterMatchupGridResponse,
    HitterMatchupTeamRow,
    HitterTeamTotals,
    ProbableCell,
    ProbableGridResponse,
    ProbableTeamRow,
)
from api.deps import get_schedule_service
from api.routers import schedule as schedule_router


class _FakeScheduleService:
    def __init__(self):
        self.seen = {}
        self.hitter_calls: list[tuple[int, str | None]] = []

    def probables(self, days=7, team_name=None):
        self.seen = {"days": days, "team_name": team_name}
        return ProbableGridResponse(
            days=["2026-06-21"],
            teams=[
                ProbableTeamRow(
                    team="LAD",
                    cells=[
                        ProbableCell(
                            opponent="SF",
                            is_home=True,
                            difficulty=70.0,
                            band="easy",
                            two_start=True,
                            availability="yours",
                        )
                    ],
                )
            ],
        )

    def hitter_matchups(self, days=7, team_name=None):
        self.hitter_calls.append((days, team_name))
        return HitterMatchupGridResponse(
            days=["2026-06-21"],
            teams=[
                HitterMatchupTeamRow(
                    team="BOS",
                    cells=[
                        HitterMatchupCell(
                            opp_sp_throws="R",
                            difficulty=70.0,
                            band="easy",
                        )
                    ],
                    totals=HitterTeamTotals(games=1, vs_rhp=1),
                )
            ],
        )


def _client(fake=None):
    app = FastAPI()
    app.include_router(schedule_router.router)
    app.dependency_overrides[get_schedule_service] = lambda: fake or _FakeScheduleService()
    return TestClient(app)


def test_probables_ok():
    r = _client().get("/api/schedule/probables", params={"days": 7})
    assert r.status_code == 200
    body = r.json()
    assert body["days"] == ["2026-06-21"]
    assert body["teams"][0]["team"] == "LAD"
    assert body["teams"][0]["cells"][0]["band"] == "easy"
    assert body["teams"][0]["cells"][0]["availability"] == "yours"


def test_probables_days_out_of_range_is_422():
    r = _client().get("/api/schedule/probables", params={"days": 99})
    assert r.status_code == 422  # Query(ge=1, le=14)


def test_probables_forwards_team_name_for_yours_tag():
    fake = _FakeScheduleService()
    _client(fake).get("/api/schedule/probables", params={"days": 5, "team_name": "Team Hickey"})
    assert fake.seen == {"days": 5, "team_name": "Team Hickey"}


def test_hitter_matchups_ok():
    fake = _FakeScheduleService()
    r = _client(fake).get("/api/schedule/hitter-matchups", params={"days": 3, "team_name": "Team Hickey"})
    assert r.status_code == 200
    body = r.json()
    assert body["days"] == ["2026-06-21"]
    assert body["teams"][0]["team"] == "BOS"
    assert body["teams"][0]["totals"]["games"] == 1
    assert body["teams"][0]["cells"][0]["band"] == "easy"
    assert fake.hitter_calls == [(3, "Team Hickey")]


def test_hitter_matchups_days_out_of_range_is_422():
    r = _client().get("/api/schedule/hitter-matchups", params={"days": 99})
    assert r.status_code == 422  # Query(ge=1, le=14)


def test_probables_uses_viewer_context_team_over_client_param():
    # The 'yours' tag must come from the resolved viewer (Clerk identity), NOT a
    # client-supplied team_name a user could spoof — consistent with the 7 other
    # personalized routers (ctx.effective_team).
    from api.tenancy import ViewerContext, require_viewer_context

    fake = _FakeScheduleService()
    app = FastAPI()
    app.include_router(schedule_router.router)
    app.dependency_overrides[get_schedule_service] = lambda: fake
    app.dependency_overrides[require_viewer_context] = lambda: ViewerContext(
        user_id=1, league_id=1, team_name="Real Team"
    )
    TestClient(app).get("/api/schedule/probables", params={"days": 5, "team_name": "Spoofed"})
    assert fake.seen["team_name"] == "Real Team"


def test_hitter_matchups_uses_viewer_context_team_over_client_param():
    from api.tenancy import ViewerContext, require_viewer_context

    fake = _FakeScheduleService()
    app = FastAPI()
    app.include_router(schedule_router.router)
    app.dependency_overrides[get_schedule_service] = lambda: fake
    app.dependency_overrides[require_viewer_context] = lambda: ViewerContext(
        user_id=1, league_id=1, team_name="Real Team"
    )
    TestClient(app).get("/api/schedule/hitter-matchups", params={"days": 3, "team_name": "Spoofed"})
    assert fake.hitter_calls[-1] == (3, "Real Team")
