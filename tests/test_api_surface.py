"""API-surface hardening tests (D6 batch).

Covers BUG-007 (API-key auth), BUG-016 (timeout + concurrency cap),
BUG-017 (status derivation taxonomy), BUG-023 (LLM boundary -> structured 502),
BUG-037 (graceful shutdown / draining), BUG-039 (no internal detail leaks),
BUG-045 (AGENT_HOST/AGENT_PORT resolution), BUG-046 (single version source),
BUG-047 (full uuid4 task ids).

All offline: fake chat models via the ``set_llm_factory`` seam — no network,
no real LLM, no MCP.
"""

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import tomllib
from conftest import ScriptedChatModel
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_API_KEY = "test-api-key"
PAYLOAD = {"description": "Build the thing", "repo_owner": "octo", "repo_name": "demo"}
AUTH = {"X-API-Key": TEST_API_KEY}


# ──────────────────────────────────────────────────────────────
# Fake models local to this suite
# ──────────────────────────────────────────────────────────────

class SlowChatModel(ScriptedChatModel):
    """Sleeps ``delay`` seconds per call; records whether it was cancelled."""

    delay: float = 30.0
    started: bool = False
    cancelled: bool = False

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.started = True
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="[[END]]"))])


def _install_model(model) -> None:
    from orchestrator import state_machine as sm

    sm.set_llm_factory(lambda: model)


def _make_raising_model(exc: BaseException) -> ScriptedChatModel:
    """A fake model that raises ``exc`` on every invocation."""

    class _Raising(ScriptedChatModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            raise exc

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise exc

    return _Raising()


def _sync_client(app, **kwargs) -> TestClient:
    client = TestClient(app, **kwargs)
    client.headers["X-API-Key"] = TEST_API_KEY
    return client


# ──────────────────────────────────────────────────────────────
# BUG-007 — API-key auth on all /tasks/* endpoints
# ──────────────────────────────────────────────────────────────

def test_health_is_open_without_api_key(install_model):
    install_model(["[[END]]"])
    from orchestrator.api import app

    with TestClient(app) as c:
        resp = c.get("/health")
    assert resp.status_code == 200


def test_invoke_rejects_missing_and_wrong_key(install_model):
    install_model(["[[END]]"])
    from orchestrator.api import app

    with TestClient(app) as c:
        assert c.post("/tasks/invoke", json=PAYLOAD).status_code == 401
        assert (
            c.post("/tasks/invoke", json=PAYLOAD, headers={"X-API-Key": "wrong"}).status_code
            == 401
        )
        resp = c.post("/tasks/invoke", json=PAYLOAD, headers=AUTH)
    assert resp.status_code == 200, resp.text


def test_status_and_human_input_require_key(install_model):
    install_model(["[[END]]"])
    from orchestrator.api import app

    with TestClient(app) as c:
        assert c.get("/tasks/whatever/status").status_code == 401
        assert (
            c.post("/tasks/human-input", json={"task_id": "x", "answer": "y"}).status_code
            == 401
        )
        # With the right key they reach the real handler (404: unknown task).
        assert c.get("/tasks/whatever/status", headers=AUTH).status_code == 404


def test_startup_fails_fast_without_api_key(monkeypatch, install_model):
    install_model(["[[END]]"])
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    from orchestrator.api import app

    with pytest.raises(RuntimeError, match="AGENT_API_KEY"), TestClient(app):
        pass


@pytest.mark.asyncio
async def test_tasks_refused_when_key_unconfigured(monkeypatch, install_model):
    """If the app is somehow up without a key, /tasks/* must still refuse."""
    install_model(["[[END]]"])
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    from orchestrator.api import app

    transport = httpx.ASGITransport(app=app)  # no lifespan -> simulates misconfig
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        resp = await c.get("/tasks/whatever/status")
    assert resp.status_code == 503
    assert resp.json()["error_type"] == "auth_not_configured"


def test_compose_requires_agent_api_key():
    text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "${AGENT_API_KEY:?" in text, "agent service must require AGENT_API_KEY"


# ──────────────────────────────────────────────────────────────
# BUG-016 — per-task timeout + concurrency cap
# ──────────────────────────────────────────────────────────────

def test_timeout_cancels_run_and_returns_504(monkeypatch):
    monkeypatch.setenv("TASK_TIMEOUT_SECONDS", "1")
    model = SlowChatModel(delay=30)
    _install_model(model)
    from orchestrator import api as api_mod

    with _sync_client(api_mod.app) as c:
        resp = c.post("/tasks/invoke", json=PAYLOAD)
    assert resp.status_code == 504, resp.text
    body = resp.json()
    assert body["error_type"] == "task_timeout"
    assert body["timeout_seconds"] == 1
    assert model.cancelled is True, "cancellation must propagate into the graph run"
    assert api_mod.in_flight.count == 0


@pytest.mark.asyncio
async def test_concurrency_cap_rejects_excess_with_503(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_TASKS", "1")
    model = SlowChatModel(delay=30)
    _install_model(model)
    from orchestrator import api as api_mod

    transport = httpx.ASGITransport(app=api_mod.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        first = asyncio.create_task(c.post("/tasks/invoke", json=PAYLOAD, headers=AUTH))
        for _ in range(200):
            if model.started:
                break
            await asyncio.sleep(0.02)
        assert model.started, "first run never reached the model"
        second = await c.post("/tasks/invoke", json=PAYLOAD, headers=AUTH)
        assert second.status_code == 503, second.text
        assert second.json()["error_type"] == "capacity_exceeded"
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    for _ in range(50):
        if api_mod.in_flight.count == 0:
            break
        await asyncio.sleep(0.02)
    assert api_mod.in_flight.count == 0


# ──────────────────────────────────────────────────────────────
# BUG-017 — status derivation for every termination path
# ──────────────────────────────────────────────────────────────

def _snapshot(values: dict, *, interrupts: bool = False, next_nodes: tuple = ()):
    tasks = [SimpleNamespace(interrupts=[object()] if interrupts else [])]
    return SimpleNamespace(values=values, tasks=tasks, next=next_nodes)


def test_run_status_unit_taxonomy():
    from orchestrator.api import _run_status

    assert _run_status(_snapshot({}, interrupts=True)) == "awaiting_human"
    assert _run_status(_snapshot({}, next_nodes=("alice",))) == "running"
    assert _run_status(_snapshot({"active_agent": "end"})) == "completed"
    forced = {"active_agent": "bob", "iteration_count": 20, "max_iterations": 20}
    assert _run_status(_snapshot(forced)) == "terminated_max_iterations"
    anomalous = {"active_agent": "bob", "iteration_count": 1, "max_iterations": 20}
    assert _run_status(_snapshot(anomalous)) == "failed"


def test_invoke_completed_on_end_sentinel(install_model):
    install_model(["All done. [[END]]"])
    from orchestrator.api import app

    with _sync_client(app) as c:
        resp = c.post("/tasks/invoke", json=PAYLOAD)
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "completed"
        status = c.get(f"/tasks/{resp.json()['task_id']}/status").json()
    assert status["status"] == "completed"


def test_invoke_terminated_at_max_iterations(install_model):
    install_model(["[[NEXT:bob]]", "[[NEXT:bob]]", "[[NEXT:bob]]"])
    from orchestrator.api import app

    payload = {**PAYLOAD, "max_iterations": 3}
    with _sync_client(app) as c:
        resp = c.post("/tasks/invoke", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["status"] == "terminated_max_iterations", body
        assert body["iteration_count"] == 3
        status = c.get(f"/tasks/{body['task_id']}/status").json()
    assert status["status"] == "terminated_max_iterations"


def test_invoke_awaiting_human_then_completed(install_model):
    install_model([
        "Need a human decision. [[NEXT:human]]",
        "Approved; done. [[END]]",
    ])
    from orchestrator.api import app

    with _sync_client(app) as c:
        resp = c.post("/tasks/invoke", json=PAYLOAD)
        task_id = resp.json()["task_id"]
        assert resp.json()["status"] == "awaiting_human"
        assert c.get(f"/tasks/{task_id}/status").json()["status"] == "awaiting_human"
        resumed = c.post("/tasks/human-input", json={"task_id": task_id, "answer": "go"})
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["status"] == "completed"
        assert c.get(f"/tasks/{task_id}/status").json()["status"] == "completed"


# ──────────────────────────────────────────────────────────────
# BUG-023 — LLM boundary: timeout/retries on client + structured 502
# ──────────────────────────────────────────────────────────────

def test_get_llm_configures_timeout_and_retries(monkeypatch):
    from orchestrator import state_machine as sm

    sm.reset_llm_factory()
    sm.reset_llm_caches()
    monkeypatch.setenv("LITELLM_API_KEY", "sk-dummy")
    monkeypatch.delenv("LLM_REQUEST_TIMEOUT", raising=False)
    monkeypatch.delenv("LLM_MAX_RETRIES", raising=False)
    llm = sm.get_llm()
    assert llm.max_retries == 2, "bounded retries by default"
    assert float(llm.request_timeout) == 60.0, "explicit request timeout by default"

    monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "12")
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    sm.reset_llm_caches()
    llm = sm.get_llm()
    assert llm.max_retries == 5
    assert float(llm.request_timeout) == 12.0
    sm.reset_llm_caches()


def test_llm_connection_error_maps_to_structured_502():
    from openai import APIConnectionError

    exc = APIConnectionError(
        message="Connection error contacting http://litellm-proxy:4000/v1",
        request=httpx.Request("POST", "http://litellm-proxy:4000/v1/chat/completions"),
    )
    _install_model(_make_raising_model(exc))
    from orchestrator.api import app

    with _sync_client(app, raise_server_exceptions=False) as c:
        resp = c.post("/tasks/invoke", json=PAYLOAD)
    assert resp.status_code == 502, resp.text
    body = resp.json()
    assert body["error_type"] == "upstream_unavailable"
    assert "litellm-proxy" not in resp.text, "internal URL must not leak"
    assert "Connection error" not in resp.text


def test_llm_rate_limit_maps_to_502_taxonomy():
    from openai import RateLimitError

    exc = RateLimitError(
        message="rate limited",
        response=httpx.Response(429, request=httpx.Request("POST", "http://x")),
        body=None,
    )
    _install_model(_make_raising_model(exc))
    from orchestrator.api import app

    with _sync_client(app, raise_server_exceptions=False) as c:
        resp = c.post("/tasks/invoke", json=PAYLOAD)
    assert resp.status_code == 502, resp.text
    assert resp.json()["error_type"] == "upstream_rate_limited"


# ──────────────────────────────────────────────────────────────
# BUG-037 — graceful shutdown of in-flight runs
# ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_drain_cancels_registered_tasks():
    from orchestrator import api as api_mod

    async def sleeper():
        await asyncio.sleep(30)

    task = asyncio.create_task(sleeper())
    api_mod.in_flight.register(task)
    assert api_mod.in_flight.count == 1
    await api_mod.in_flight.drain(grace_seconds=1)
    assert task.cancelled()
    assert api_mod.in_flight.draining is True
    api_mod.in_flight.draining = False


def test_health_reports_draining_and_in_flight(install_model):
    install_model(["[[END]]"])
    from orchestrator import api as api_mod

    with _sync_client(api_mod.app) as c:
        body = c.get("/health").json()
        assert body["draining"] is False
        assert body["in_flight"] == 0
        api_mod.in_flight.draining = True
        try:
            assert c.get("/health").json()["draining"] is True
        finally:
            api_mod.in_flight.draining = False


@pytest.mark.asyncio
async def test_invoke_rejected_while_draining(install_model):
    install_model(["[[END]]"])
    from orchestrator import api as api_mod

    api_mod.in_flight.draining = True
    try:
        transport = httpx.ASGITransport(app=api_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            resp = await c.post("/tasks/invoke", json=PAYLOAD, headers=AUTH)
        assert resp.status_code == 503
        assert resp.json()["error_type"] == "server_draining"
    finally:
        api_mod.in_flight.draining = False


# ──────────────────────────────────────────────────────────────
# BUG-039 — internal exception details never reflected to callers
# ──────────────────────────────────────────────────────────────

def test_internal_error_returns_generic_body_with_error_id():
    secret = "sensitive-internal-detail-litellm-proxy:4000"
    _install_model(_make_raising_model(RuntimeError(secret)))
    from orchestrator.api import app

    with _sync_client(app, raise_server_exceptions=False) as c:
        resp = c.post("/tasks/invoke", json=PAYLOAD)
    assert resp.status_code == 500, resp.text
    body = resp.json()
    assert body["detail"] == "internal error"
    assert re.fullmatch(r"[0-9a-f]{32}", body["error_id"]), body
    assert secret not in resp.text


# ──────────────────────────────────────────────────────────────
# BUG-045 — AGENT_HOST / AGENT_PORT resolution
# ──────────────────────────────────────────────────────────────

def test_serve_settings_default_to_loopback(monkeypatch):
    monkeypatch.delenv("AGENT_HOST", raising=False)
    monkeypatch.delenv("AGENT_PORT", raising=False)
    from orchestrator import api as api_mod

    assert api_mod.resolve_serve_settings() == ("127.0.0.1", 8000)


def test_serve_settings_honor_env(monkeypatch):
    monkeypatch.setenv("AGENT_HOST", "0.0.0.0")
    monkeypatch.setenv("AGENT_PORT", "9000")
    from orchestrator import api as api_mod

    assert api_mod.resolve_serve_settings() == ("0.0.0.0", 9000)


def test_serve_settings_invalid_port_falls_back(monkeypatch):
    monkeypatch.setenv("AGENT_PORT", "not-a-port")
    from orchestrator import api as api_mod

    assert api_mod.resolve_serve_settings() == ("127.0.0.1", 8000)


def test_compose_binds_all_interfaces_explicitly():
    text = (REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "AGENT_HOST: 0.0.0.0" in text, "container use must opt into 0.0.0.0"


# ──────────────────────────────────────────────────────────────
# BUG-046 — single version source
# ──────────────────────────────────────────────────────────────

def _pyproject_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def test_health_and_app_version_match_pyproject(install_model):
    install_model(["[[END]]"])
    import orchestrator
    from orchestrator.api import app

    with TestClient(app) as c:
        body = c.get("/health").json()
    expected = _pyproject_version()
    assert body["version"] == expected
    assert app.version == expected
    assert orchestrator.__version__ == expected


# ──────────────────────────────────────────────────────────────
# BUG-047 — full uuid4 task ids
# ──────────────────────────────────────────────────────────────

def test_task_id_is_full_uuid4_hex_and_round_trips(install_model):
    install_model(["Done. [[END]]"])
    from orchestrator.api import app

    with _sync_client(app) as c:
        resp = c.post("/tasks/invoke", json=PAYLOAD)
        assert resp.status_code == 200, resp.text
        task_id = resp.json()["task_id"]
        assert re.fullmatch(r"[0-9a-f]{32}", task_id), task_id
        status = c.get(f"/tasks/{task_id}/status")
        assert status.status_code == 200
        assert status.json()["task_id"] == task_id
