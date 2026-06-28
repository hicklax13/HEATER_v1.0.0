"""DB-free DI tests for the async optimize pair (/optimize/start + /optimize/result).

The start endpoint must return a job_id and the result endpoint must surface the
stored result. We fake the LineupService via dependency_overrides; the background
task runs synchronously inside TestClient's request, so by the time start returns,
the job is already done (the runner called optimize_jobs.finish)."""

from starlette.testclient import TestClient

from api.contracts.common import PlayerRef
from api.contracts.lineup import LineupOptimizeResponse, LineupSlot
from api.deps import get_lineup_service
from api.main import create_app
from api.services import optimize_jobs


class _FakeLineupService:
    def __init__(self):
        self.calls = []

    def optimize(self, team_name, date=None, scope="rest_of_season", mode="standard", depth="standard"):
        self.calls.append((team_name, date, scope, mode, depth))
        return LineupOptimizeResponse(
            team_name=team_name,
            date=date or "2027-04-05",
            slots=[
                LineupSlot(
                    slot="OF",
                    player=PlayerRef(id=1, name="Starter", positions="OF"),
                    action="START",
                    status="start",
                    value=87.0,
                )
            ],
            summary=f"1 starter ({depth})",
            mode="standard",
            depth=depth,
        )


def _client(svc):
    optimize_jobs.reset()
    app = create_app()
    app.dependency_overrides[get_lineup_service] = lambda: svc
    return TestClient(app)


def test_start_returns_job_id_and_enhanced_depth_passes_through():
    svc = _FakeLineupService()
    client = _client(svc)
    body = client.post(
        "/api/lineup/optimize/start",
        json={"team_name": "Team Hickey", "scope": "rest_of_season", "depth": "enhanced"},
    ).json()
    assert body["status"] == "running"
    assert isinstance(body["job_id"], str) and body["job_id"]
    # The background task ran synchronously under TestClient → service was called
    # with depth="enhanced".
    assert svc.calls and svc.calls[-1][4] == "enhanced"


def test_result_returns_stored_result_after_completion():
    svc = _FakeLineupService()
    client = _client(svc)
    start = client.post(
        "/api/lineup/optimize/start",
        json={"team_name": "Team Hickey", "scope": "rest_of_week", "depth": "standard"},
    ).json()
    job_id = start["job_id"]
    res = client.get(f"/api/lineup/optimize/result/{job_id}")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "done"
    assert body["error"] is None
    assert body["result"]["team_name"] == "Team Hickey"
    assert body["result"]["slots"][0]["value"] == 87.0


def test_result_simulated_completion_via_finish():
    # Independent of the background runner: create a job, finish it directly, assert
    # the result endpoint surfaces it.
    svc = _FakeLineupService()
    client = _client(svc)
    job_id = optimize_jobs.create()
    optimize_jobs.finish(
        job_id,
        result=LineupOptimizeResponse(team_name="Team X", date="2027-04-05", slots=[], summary="done"),
    )
    body = client.get(f"/api/lineup/optimize/result/{job_id}").json()
    assert body["status"] == "done"
    assert body["result"]["team_name"] == "Team X"


def test_result_unknown_job_is_404():
    client = _client(_FakeLineupService())
    assert client.get("/api/lineup/optimize/result/does-not-exist").status_code == 404


def test_run_optimize_job_records_error_on_failure():
    # The runner must NEVER raise — a service exception becomes the job's error.
    class _Boom:
        def optimize(self, *a, **k):
            raise RuntimeError("kaboom")

    optimize_jobs.reset()
    job_id = optimize_jobs.create()
    optimize_jobs.run_optimize_job(_Boom(), job_id, "Team", None, "rest_of_season", "enhanced")
    job = optimize_jobs.get(job_id)
    assert job["status"] == "error"
    assert "kaboom" in (job["error"] or "")
