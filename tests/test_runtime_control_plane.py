from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import api.app as app_module
from agents.manager import Manager
from api.app import app
from tools.run_registry import RunRecord, RunRegistry


client = TestClient(app)


# ---------------------------------------------------------------------------
# /v1/runtime endpoint tests
# ---------------------------------------------------------------------------

class TestRuntimeEndpoint:
    def test_returns_200(self):
        response = client.get("/v1/runtime")
        assert response.status_code == 200

    def test_required_fields_present(self):
        response = client.get("/v1/runtime")
        payload = response.json()
        assert payload["service"] == "agent-os"
        assert payload["status"] == "ok"
        assert "runs" in payload
        assert "total" in payload["runs"]
        assert "queued" in payload["runs"]
        assert "running" in payload["runs"]
        assert "worker" in payload
        assert "configured_concurrency" in payload["worker"]
        assert "routing" in payload
        assert "adaptive" in payload["routing"]

    def test_queued_running_come_from_registry(self, tmp_path, monkeypatch):
        test_registry = RunRegistry(tmp_path / "test.db")

        test_registry.save(
            RunRecord(
                run_id="q1",
                objective="queued job",
                worker="researcher",
                research_mode="normal",
                target_url=None,
                tool="firecrawl.search+scrape",
                status="queued",
                started_at=None,
                completed_at=None,
                duration_ms=None,
                result=None,
                error=None,
            )
        )
        test_registry.save(
            RunRecord(
                run_id="r1",
                objective="running job",
                worker="researcher",
                research_mode="normal",
                target_url=None,
                tool="firecrawl.search+scrape",
                status="running",
                started_at=None,
                completed_at=None,
                duration_ms=None,
                result=None,
                error=None,
            )
        )

        with patch.object(app_module, "registry", test_registry):
            response = client.get("/v1/runtime")

        payload = response.json()
        assert payload["runs"]["queued"] == 1
        assert payload["runs"]["running"] == 1
        assert payload["runs"]["total"] == 2

    def test_adaptive_routing_true(self, monkeypatch):
        monkeypatch.setenv("AGENT_OS_ADAPTIVE_ROUTING", "true")
        response = client.get("/v1/runtime")
        assert response.json()["routing"]["adaptive"] is True

    def test_adaptive_routing_false_when_missing(self, monkeypatch):
        monkeypatch.delenv("AGENT_OS_ADAPTIVE_ROUTING", raising=False)
        response = client.get("/v1/runtime")
        assert response.json()["routing"]["adaptive"] is False

    def test_adaptive_routing_false_when_false(self, monkeypatch):
        monkeypatch.setenv("AGENT_OS_ADAPTIVE_ROUTING", "false")
        response = client.get("/v1/runtime")
        assert response.json()["routing"]["adaptive"] is False

    def test_worker_concurrency_reflected(self, monkeypatch):
        monkeypatch.setenv("AGENT_OS_WORKER_CONCURRENCY", "8")
        response = client.get("/v1/runtime")
        assert response.json()["worker"]["configured_concurrency"] == 8

    def test_worker_concurrency_default(self, monkeypatch):
        monkeypatch.delenv("AGENT_OS_WORKER_CONCURRENCY", raising=False)
        response = client.get("/v1/runtime")
        assert response.json()["worker"]["configured_concurrency"] == 4


# ---------------------------------------------------------------------------
# Lifecycle regression tests
# ---------------------------------------------------------------------------

def _make_registry(tmp_path):
    return RunRegistry(tmp_path / "lifecycle.db")


def _queued_record(run_id: str, registry: RunRegistry) -> RunRecord:
    saved = registry.get(run_id)
    assert saved is not None
    return saved


class TestLifecycle:
    def test_create_job_queued(self, tmp_path):
        registry = _make_registry(tmp_path)
        mgr = Manager(registry=registry)

        job = mgr.create_job("Summarise the latest AI news")
        mgr.queue(job)

        record = registry.get(job.id)
        assert record is not None
        assert record.status == "queued"

    def test_claim_queued_run_becomes_running(self, tmp_path):
        registry = _make_registry(tmp_path)
        mgr = Manager(registry=registry)

        job = mgr.create_job("Summarise the latest AI news")
        mgr.queue(job)

        claimed = registry.claim_next_queued()
        assert claimed is not None
        assert claimed.status == "running"
        assert claimed.run_id == job.id

    def test_execute_mocked_completed(self, tmp_path):
        registry = _make_registry(tmp_path)
        mgr = Manager(registry=registry)

        job = mgr.create_job("Summarise the latest AI news")
        mgr.queue(job)
        registry.claim_next_queued()

        def fake_execute(j):
            j.status = "completed"
            j.result = "mocked result"
            registry.save(
                RunRecord(
                    run_id=j.id,
                    parent_run_id=j.parent_run_id,
                    attempt=j.attempt,
                    objective=j.objective,
                    worker=j.worker or "researcher",
                    research_mode=j.research_mode,
                    target_url=j.target_url,
                    tool="firecrawl.search+scrape",
                    status="completed",
                    started_at=None,
                    completed_at=None,
                    duration_ms=100,
                    result={"mocked": True},
                    error=None,
                )
            )
            return j

        with patch.object(mgr, "execute", fake_execute):
            result = mgr.execute(job)

        assert result.status == "completed"
        persisted = registry.get(job.id)
        assert persisted is not None
        assert persisted.status == "completed"

    def test_completed_state_persists(self, tmp_path):
        registry = _make_registry(tmp_path)
        mgr = Manager(registry=registry)

        job = mgr.create_job("Summarise the latest AI news")
        mgr.queue(job)
        registry.claim_next_queued()

        def fake_execute(j):
            j.status = "completed"
            registry.save(
                RunRecord(
                    run_id=j.id,
                    parent_run_id=j.parent_run_id,
                    attempt=j.attempt,
                    objective=j.objective,
                    worker=j.worker or "researcher",
                    research_mode=j.research_mode,
                    target_url=j.target_url,
                    tool="firecrawl.search+scrape",
                    status="completed",
                    started_at=None,
                    completed_at=None,
                    duration_ms=50,
                    result={"done": True},
                    error=None,
                )
            )
            return j

        with patch.object(mgr, "execute", fake_execute):
            mgr.execute(job)

        persisted = registry.get(job.id)
        assert persisted.status == "completed"

    def test_failed_execute_persists_failure(self, tmp_path):
        registry = _make_registry(tmp_path)
        mgr = Manager(registry=registry)

        job = mgr.create_job("Summarise the latest AI news")
        mgr.queue(job)
        registry.claim_next_queued()

        def fake_execute_fail(j):
            j.status = "failed"
            j.error = "RuntimeError: mocked failure"
            registry.save(
                RunRecord(
                    run_id=j.id,
                    parent_run_id=j.parent_run_id,
                    attempt=j.attempt,
                    objective=j.objective,
                    worker=j.worker or "researcher",
                    research_mode=j.research_mode,
                    target_url=j.target_url,
                    tool="firecrawl.search+scrape",
                    status="failed",
                    started_at=None,
                    completed_at=None,
                    duration_ms=10,
                    result=None,
                    error="RuntimeError: mocked failure",
                )
            )
            return j

        with patch.object(mgr, "execute", fake_execute_fail):
            result = mgr.execute(job)

        assert result.status == "failed"
        assert result.error is not None
        persisted = registry.get(job.id)
        assert persisted.status == "failed"
        assert persisted.error is not None

    def test_run_id_unchanged_through_lifecycle(self, tmp_path):
        registry = _make_registry(tmp_path)
        mgr = Manager(registry=registry)

        job = mgr.create_job("Summarise the latest AI news")
        original_id = job.id

        mgr.queue(job)
        assert job.id == original_id

        queued_record = registry.get(original_id)
        assert queued_record is not None
        assert queued_record.run_id == original_id

        claimed = registry.claim_next_queued()
        assert claimed.run_id == original_id

        def fake_execute(j):
            j.status = "completed"
            registry.save(
                RunRecord(
                    run_id=j.id,
                    parent_run_id=j.parent_run_id,
                    attempt=j.attempt,
                    objective=j.objective,
                    worker=j.worker or "researcher",
                    research_mode=j.research_mode,
                    target_url=j.target_url,
                    tool="firecrawl.search+scrape",
                    status="completed",
                    started_at=None,
                    completed_at=None,
                    duration_ms=50,
                    result={"done": True},
                    error=None,
                )
            )
            return j

        with patch.object(mgr, "execute", fake_execute):
            result = mgr.execute(job)

        assert result.id == original_id
        persisted = registry.get(original_id)
        assert persisted.run_id == original_id
