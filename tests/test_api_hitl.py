"""API plumbing tests for BUG-008/010 (status + human-input resume).

These use FastAPI's TestClient against the real app, with the LLM factory
seam pointing at a ScriptedChatModel — no network, no stdin.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(install_model):
    install_model([
        "I need a decision from a human. [[NEXT:human]]",
        "Human approved; wrapping up. [[END]]",
    ])
    from orchestrator.api import app

    with TestClient(app) as c:
        c.headers["X-API-Key"] = "test-api-key"  # BUG-007: /tasks/* requires auth
        yield c


def _invoke(client) -> str:
    resp = client.post(
        "/tasks/invoke",
        json={"description": "Build the thing", "repo_owner": "octo", "repo_name": "demo"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "awaiting_human", body  # BUG-017 status taxonomy
    return body["task_id"]


def test_invoke_pauses_at_human_node(client):
    """BUG-008/010: invoke interrupts instead of touching stdin / auto-approving."""
    task_id = _invoke(client)
    assert task_id


def test_status_reads_real_graph_state(client):
    """BUG-010: /tasks/{id}/status reflects the checkpoint, not a stub."""
    task_id = _invoke(client)
    resp = client.get(f"/tasks/{task_id}/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "awaiting_human"
    assert body["task_id"] == task_id


def test_status_unknown_task_is_404(client):
    resp = client.get("/tasks/does-not-exist/status")
    assert resp.status_code == 404


def test_human_input_resumes_run(client):
    """BUG-008/010: POST human-input resumes via Command(resume=...)."""
    task_id = _invoke(client)
    resp = client.post("/tasks/human-input", json={"task_id": task_id, "answer": "go ahead"})
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "completed"

    status = client.get(f"/tasks/{task_id}/status").json()
    assert status["status"] == "completed"
    assert status["human_input"] == "go ahead"


def test_human_input_unknown_task_is_404(client):
    resp = client.post("/tasks/human-input", json={"task_id": "nope", "answer": "x"})
    assert resp.status_code == 404
