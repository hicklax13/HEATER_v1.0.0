from api.contracts.jobs import JobResource, JobStatus


def test_job_status_values():
    assert {s.value for s in JobStatus} == {
        "queued",
        "running",
        "succeeded",
        "failed",
        "canceled",
        "expired",
    }


def test_job_resource_minimal():
    job = JobResource(job_id="job_1", status=JobStatus.queued, job_type="playoff_sim")
    assert job.progress == 0.0
    assert job.result_url is None
    assert job.status is JobStatus.queued


def test_job_resource_serializes_status_as_string():
    job = JobResource(job_id="job_1", status=JobStatus.running, job_type="trade_mc")
    assert job.model_dump()["status"] == "running"
