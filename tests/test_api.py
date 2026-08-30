from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_health():
    response = client.get("/v1/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["service"] == "agent-os"


def test_missing_run_returns_404():
    response = client.get(
        "/v1/runs/does-not-exist"
    )

    assert response.status_code == 404


def test_invalid_job_returns_422():
    response = client.post(
        "/v1/jobs",
        json={"objective": ""},
    )

    assert response.status_code == 422


def test_runs_endpoint():
    response = client.get(
        "/v1/runs",
        params={"limit": 5},
    )

    assert response.status_code == 200

    payload = response.json()

    assert "count" in payload
    assert "runs" in payload
    assert isinstance(payload["runs"], list)
