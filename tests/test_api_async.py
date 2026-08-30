from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_create_job_returns_202():
    from api.app import manager

    def fake_queue(job):
        job.worker = "researcher"
        job.research_mode = "normal"
        job.status = "queued"
        return job

    original_queue = manager.queue

    manager.queue = fake_queue

    try:
        response = client.post(
            "/v1/jobs",
            json={
                "objective": "Research AI agent platforms"
            },
        )

        assert response.status_code == 202

        payload = response.json()

        assert payload["status"] == "queued"
        assert payload["worker"] == "researcher"
        assert payload["research_mode"] == "normal"

    finally:
        manager.queue = original_queue


def test_create_job_rejects_empty_objective():
    response = client.post(
        "/v1/jobs",
        json={
            "objective": ""
        },
    )

    assert response.status_code == 422


def test_health_reports_queue_state(tmp_path):
    import api.app as api_module
    from agents.manager import Manager

    original_registry = api_module.registry
    original_manager = api_module.manager

    registry = api_module.RunRegistry(
        tmp_path / "test.db"
    )

    manager = Manager(
        registry=registry
    )

    api_module.registry = registry
    api_module.manager = manager

    try:
        response = client.get(
            "/v1/health"
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload["status"] == "ok"
        assert payload["service"] == "agent-os"
        assert payload["queued"] == 0
        assert payload["running"] == 0

    finally:
        api_module.registry = original_registry
        api_module.manager = original_manager


def test_api_does_not_execute_jobs_inline(monkeypatch):
    from api.app import manager

    def fail_execute(job):
        raise AssertionError(
            "API must not execute jobs inline."
        )

    monkeypatch.setattr(
        manager,
        "execute",
        fail_execute,
    )

    response = client.post(
        "/v1/jobs",
        json={
            "objective": "Research AI agent platforms"
        },
    )

    assert response.status_code == 202

    payload = response.json()

    assert payload["status"] == "queued"
