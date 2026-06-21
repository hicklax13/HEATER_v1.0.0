"""Endpoint tests for /api/schedule/probables via a fake service (DB-free)."""

from fastapi import FastAPI
from starlette.testclient import TestClient

from api.contracts.schedule import ProbableCell, ProbableGridResponse, ProbableTeamRow
from api.deps import get_schedule_service
from api.routers import schedule as schedule_router


class _FakeScheduleService:
    def __init__(self):
        self.seen = {}

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
