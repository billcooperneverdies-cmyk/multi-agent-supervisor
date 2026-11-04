"""LangGraph FastAPI Service — 生产级四代理团队 API。"""

import asyncio
import hmac
import logging
import os
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from orchestrator import __version__
from orchestrator.state_machine import (
    make_run_config,
    new_task_state,
    normalize_content,
    team_graph,
)

logger = logging.getLogger(__name__)

API_KEY_HEADER = "X-API-Key"


class ApiError(Exception):
    """Structured API error rendered as JSON with an error-type taxonomy."""

    def __init__(self, status_code: int, error_type: str, detail: str, **extra: Any):
        super().__init__(detail)
        self.status_code = status_code
        self.error_type = error_type
        self.detail = detail
        self.extra = extra


def configured_api_key() -> str:
    """The API key callers must present; empty means 'not configured'."""
    return os.environ.get("AGENT_API_KEY", "").strip()


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """BUG-007: API-key auth dependency for every /tasks/* endpoint.

    The server must be configured with AGENT_API_KEY (fail-fast in lifespan;
    defense-in-depth 503 here); /health intentionally stays open for probes.
    Comparison is constant-time.
    """
    configured = configured_api_key()
    if not configured:
        raise ApiError(
            503,
            "auth_not_configured",
            "server has no AGENT_API_KEY configured; refusing to serve tasks",
        )
    if not x_api_key or not hmac.compare_digest(x_api_key, configured):
        raise HTTPException(status_code=401, detail="missing or invalid API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。Fail-fast without AGENT_API_KEY (BUG-007)."""
    if not configured_api_key():
        raise RuntimeError(
            "AGENT_API_KEY is not set. The API refuses to start without an API "
            "key — set AGENT_API_KEY (see .env.example); /tasks/* require the "
            f"{API_KEY_HEADER} header."
        )
    in_flight.draining = False
    logger.info("启动多代理团队服务...")
    yield
    # BUG-037: graceful shutdown — stop accepting work, cancel in-flight runs
    # with a bounded wait (checkpoints remain resumable per BUG-010).
    logger.info("关闭多代理团队服务：取消 %d 个在途任务...", in_flight.count)
    await in_flight.drain(shutdown_grace_seconds())


app = FastAPI(
    title="AI Software Development Team",
    description="Alice (PM) · Bob (Frontend) · Charlie (Backend) · Diana (QA)",
    version=__version__,  # BUG-046: single version source
    lifespan=lifespan,
)


@app.exception_handler(ApiError)
async def _api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    """Render ApiError as a structured body (no internals)."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_type": exc.error_type, **exc.extra},
    )


@app.exception_handler(Exception)
async def _unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """BUG-039: callers get a generic body + correlation id; the traceback is
    logged server-side under the same id — never reflected to the client."""
    error_id = uuid.uuid4().hex
    logger.exception("unhandled internal error (error_id=%s): %s", error_id, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "error_id": error_id},
    )


class TeamTaskRequest(BaseModel):
    """团队任务请求。"""

    description: str = Field(..., description="任务需求描述")
    repo_owner: str = Field(..., description="GitHub 仓库 owner")
    repo_name: str = Field(..., description="GitHub 仓库名")
    slack_channel: str = Field(default="#general", description="Slack 通知频道")
    thread_ts: str | None = Field(default=None, description="Slack 线程时间戳")
    max_iterations: int = Field(default=20, ge=1, le=50)


class TeamTaskResponse(BaseModel):
    """团队任务响应。"""

    task_id: str
    status: str
    active_agent: str
    iteration_count: int
    messages: list[dict]
    pr_urls: list[str]
    issue_urls: list[str]
    summary: str | None


class HumanInputRequest(BaseModel):
    """人类输入请求（用于中断恢复）。"""

    task_id: str = Field(..., description="任务 ID")
    answer: str = Field(..., description="人类回答")


def _upstream_error_type(exc: BaseException) -> str | None:
    """BUG-023: classify LLM/proxy failures into the structured 502 taxonomy.

    Returns None for non-LLM errors (those fall through to the generic 500
    handler). Subclass checks run before their base classes on purpose.
    """
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            RateLimitError,
        )
        from openai import APIError as OpenAIError
    except ImportError:  # pragma: no cover - openai ships with langchain-openai
        return None
    if isinstance(exc, APITimeoutError):
        return "upstream_timeout"
    if isinstance(exc, APIConnectionError):
        return "upstream_unavailable"
    if isinstance(exc, RateLimitError):
        return "upstream_rate_limited"
    if isinstance(exc, APIStatusError):
        return "upstream_provider_error"
    if isinstance(exc, OpenAIError):
        return "upstream_error"
    return None


async def _call_graph(run: Any, task_id: str) -> dict[str, Any]:
    """Run one graph invocation, mapping LLM/proxy failures to structured 502
    (BUG-023); anything else bubbles to the generic 500 handler (BUG-039)."""
    try:
        return await run()
    except ApiError:
        raise
    except Exception as exc:
        error_type = _upstream_error_type(exc)
        if error_type is not None:
            logger.warning(
                "task %s: upstream LLM failure (%s, %s)",
                task_id,
                error_type,
                type(exc).__name__,
            )
            raise ApiError(
                502, error_type, "upstream LLM provider error", task_id=task_id
            ) from exc
        raise


def _parse_int(raw: str, default: int, name: str) -> int:
    if not raw or not raw.strip():
        return default  # unset/empty is the normal case — not worth a warning
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; falling back to %d", name, raw, default)
        return default


def task_timeout_seconds() -> int:
    """BUG-016: overall deadline for a single task run (default 300s)."""
    return _parse_int(os.environ.get("TASK_TIMEOUT_SECONDS", ""), 300, "TASK_TIMEOUT_SECONDS")


def max_concurrent_tasks() -> int:
    """BUG-016: concurrency cap for in-flight task runs (default 4)."""
    return _parse_int(os.environ.get("MAX_CONCURRENT_TASKS", ""), 4, "MAX_CONCURRENT_TASKS")


def shutdown_grace_seconds() -> int:
    """BUG-037: bounded wait while cancelling in-flight runs on shutdown."""
    return _parse_int(
        os.environ.get("SHUTDOWN_GRACE_SECONDS", ""), 5, "SHUTDOWN_GRACE_SECONDS"
    )


class InFlightRegistry:
    """Tracks in-flight graph-run tasks (BUG-016 cap, BUG-037 drain)."""

    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()
        self.draining = False

    @property
    def count(self) -> int:
        return len(self._tasks)

    def register(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self, grace_seconds: float) -> None:
        """BUG-037: cancel every in-flight run with a bounded wait."""
        self.draining = True
        pending = [task for task in self._tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=grace_seconds)


in_flight = InFlightRegistry()


async def _run_graph(
    run: Callable[[], Awaitable[dict[str, Any]]], task_id: str
) -> dict[str, Any]:
    """Run a graph invocation as a registered task with a deadline (BUG-016).

    - Capacity: 503 when MAX_CONCURRENT_TASKS runs are already in flight.
    - Deadline: TASK_TIMEOUT_SECONDS via asyncio.wait_for; on expiry the run
      is cancelled (cancellation propagates into the graph) and the caller
      gets a structured 504.
    - Registry: the task is tracked so shutdown can drain it (BUG-037).
    """
    if in_flight.draining:
        raise ApiError(
            503, "server_draining", "server is shutting down; not accepting new tasks"
        )
    limit = max_concurrent_tasks()
    if in_flight.count >= limit:
        raise ApiError(
            503,
            "capacity_exceeded",
            "too many concurrent tasks; retry later",
            max_concurrent_tasks=limit,
        )
    task = asyncio.create_task(_call_graph(run, task_id))
    in_flight.register(task)
    timeout = task_timeout_seconds()
    try:
        return await asyncio.wait_for(task, timeout=timeout)
    except asyncio.TimeoutError as exc:
        logger.warning("task %s exceeded %ss deadline; run cancelled", task_id, timeout)
        raise ApiError(
            504,
            "task_timeout",
            "task execution timed out",
            timeout_seconds=timeout,
            task_id=task_id,
        ) from exc


def resolve_serve_settings() -> tuple[str, int]:
    """BUG-045: bind host/port actually consumed by the uvicorn entry point.

    Defaults to loopback (127.0.0.1:8000); containers opt into 0.0.0.0
    explicitly via AGENT_HOST (compose sets it for the agent service).
    """
    host = os.environ.get("AGENT_HOST", "").strip() or "127.0.0.1"
    port = _parse_int(os.environ.get("AGENT_PORT", ""), 8000, "AGENT_PORT")
    return host, port


def _thread_config(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}}


def _terminal_status(values: dict) -> str:
    """BUG-017: status of a run that has no pending work, from its values."""
    if values.get("active_agent") == "end":
        return "completed"
    iteration = values.get("iteration_count", 0)
    maximum = values.get("max_iterations", 0)
    if maximum and iteration >= maximum:
        # route_agent force-stops at max_iterations WITHOUT writing
        # active_agent='end' — that path is a termination, not 'pending'.
        return "terminated_max_iterations"
    return "failed"


def _final_status(values: dict) -> str:
    """BUG-017: status taxonomy for a state dict returned by a graph run."""
    if "__interrupt__" in values:
        return "awaiting_human"
    return _terminal_status(values)


def _run_status(snapshot: Any) -> str:
    """BUG-017: derive the run status from a real checkpoint snapshot."""
    values = snapshot.values
    if any(getattr(task, "interrupts", None) for task in snapshot.tasks):
        return "awaiting_human"
    if "__interrupt__" in values:
        return "awaiting_human"
    if snapshot.next:
        return "running"
    return _terminal_status(values)


async def _snapshot_or_404(task_id: str):
    """Fetch the checkpoint snapshot for a task; 404 if it never existed."""
    snapshot = await team_graph.aget_state(_thread_config(task_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"unknown task_id: {task_id}")
    return snapshot


@app.get("/health")
async def health_check() -> dict:
    """健康检查；reports draining state during shutdown (BUG-037)."""
    return {
        "status": "ok",
        "service": "ai-dev-team",
        "version": __version__,
        "draining": in_flight.draining,
        "in_flight": in_flight.count,
    }


@app.post(
    "/tasks/invoke",
    response_model=TeamTaskResponse,
    dependencies=[Depends(require_api_key)],
)
async def invoke_team(request: TeamTaskRequest) -> TeamTaskResponse:
    """启动四代理团队协作任务。"""
    # BUG-047: full uuid4 hex — 8-char ids collide at the birthday bound (~2^16).
    task_id = uuid.uuid4().hex

    initial_state = new_task_state(
        request.description,
        task_id,
        request.max_iterations,
        slack_thread_ts=request.thread_ts,
        slack_channel=request.slack_channel,
        repo_owner=request.repo_owner,
        repo_name=request.repo_name,
    )

    # BUG-039: no str(e) to clients — unexpected errors hit the generic
    # 500 handler (error_id + server-side traceback).
    final_state = await _run_graph(
        lambda: team_graph.ainvoke(
            initial_state, config=make_run_config(task_id, request.max_iterations)
        ),
        task_id,
    )

    waiting = "__interrupt__" in final_state
    summary = None
    if waiting:
        interrupts = final_state["__interrupt__"]
        payload = getattr(interrupts[0], "value", interrupts[0])
        if isinstance(payload, dict):
            summary = normalize_content(payload.get("request"))[:500]
        else:
            summary = normalize_content(payload)[:500]
    else:
        messages = final_state.get("messages", [])
        if messages and hasattr(messages[-1], "content"):
            summary = normalize_content(messages[-1].content)[:500]

    status = _final_status(final_state)

    return TeamTaskResponse(
        task_id=task_id,
        status=status,
        active_agent=final_state.get("active_agent", "unknown"),
        iteration_count=final_state.get("iteration_count", 0),
        messages=[
            m.model_dump() if hasattr(m, "model_dump") else {"type": "unknown", "content": str(m)}
            for m in final_state.get("messages", [])
        ],
        pr_urls=final_state.get("pr_urls", []),
        issue_urls=final_state.get("issue_urls", []),
        summary=summary,
    )


@app.post("/tasks/human-input", dependencies=[Depends(require_api_key)])
async def provide_human_input(request: HumanInputRequest) -> dict:
    """提供人类输入以恢复中断的工作流（Command(resume=...)）。"""
    await _snapshot_or_404(request.task_id)
    final_state = await _run_graph(
        lambda: team_graph.ainvoke(
            Command(resume=request.answer), config=_thread_config(request.task_id)
        ),
        request.task_id,
    )
    status = _final_status(final_state)
    return {
        "task_id": request.task_id,
        "status": status,
        "human_input": request.answer,
        "iteration_count": final_state.get("iteration_count", 0),
    }


@app.get("/tasks/{task_id}/status", dependencies=[Depends(require_api_key)])
async def get_task_status(task_id: str) -> dict:
    """获取任务状态（从真实检查点快照读取）。"""
    snapshot = await _snapshot_or_404(task_id)
    values = snapshot.values
    return {
        "task_id": task_id,
        "status": _run_status(snapshot),
        "active_agent": values.get("active_agent"),
        "iteration_count": values.get("iteration_count", 0),
        "pending_human": values.get("pending_human", False),
        "human_input": values.get("human_input"),
        "next": list(snapshot.next),
        "pr_urls": values.get("pr_urls", []),
        "issue_urls": values.get("issue_urls", []),
    }


if __name__ == "__main__":
    import uvicorn

    _host, _port = resolve_serve_settings()
    uvicorn.run(app, host=_host, port=_port)
