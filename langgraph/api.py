"""LangGraph FastAPI Service — Production-ready multi-agent team API.

Uses Redis-backed state store for multi-worker safety.
Scheduled task cleanup. Reusable HTTP client for health checks.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from langgraph.state_machine import AgentState, team_graph
from shared.dependencies import (
    init_shared,
    get_state_store,
    get_http_client,
    close_http_client,
    logger,
)


class TeamTaskRequest(BaseModel):
    description: str = Field(..., description="Task requirement description")
    repo_owner: str = Field(..., description="GitHub repository owner")
    repo_name: str = Field(..., description="GitHub repository name")
    slack_channel: str = Field(default="#general", description="Slack notification channel")
    thread_ts: str | None = Field(default=None, description="Slack thread timestamp")
    max_iterations: int = Field(default=20, ge=1, le=50)


class TeamTaskResponse(BaseModel):
    task_id: str
    status: str
    active_agent: str
    iteration_count: int
    messages: list[dict]
    pr_urls: list[str]
    issue_urls: list[str]
    summary: str | None


class HumanInputRequest(BaseModel):
    task_id: str = Field(..., description="Task ID")
    answer: str = Field(..., description="Human answer")


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    active_agent: str
    iteration_count: int
    max_iterations: int
    started_at: float | None
    completed_at: float | None
    error: str | None


_cleanup_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI Dev Team service...")
    await init_shared()
    global _cleanup_task
    _cleanup_task = asyncio.create_task(_periodic_cleanup())
    yield
    logger.info("Shutting down...")
    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
    await close_http_client()


async def _periodic_cleanup(interval: int = 300) -> None:
    while True:
        try:
            await asyncio.sleep(interval)
            store = get_state_store()
            cleaned = await store.cleanup_expired()
            if cleaned:
                logger.info("Periodic cleanup: removed %s expired tasks", cleaned)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Cleanup task error: %s", exc)


async def _check_dependencies() -> dict[str, str]:
    import os
    results: dict[str, str] = {}
    client = get_http_client()
    try:
        litellm_base = os.environ.get("LITELLM_API_BASE", "http://localhost:4000")
        r = await client.get(f"{litellm_base}/health", timeout=5.0)
        results["litellm"] = "ok" if r.status_code == 200 else f"unhealthy ({r.status_code})"
    except Exception as exc:
        results["litellm"] = f"unreachable ({type(exc).__name__})"
    try:
        from shared.dependencies import get_redis
        r = await get_redis()
        await r.ping()
        results["redis"] = "ok"
    except Exception as exc:
        results["redis"] = f"unreachable ({type(exc).__name__})"
    return results


app = FastAPI(
    title="AI Software Development Team",
    description="Alice (PM) · Bob (Frontend) · Charlie (Backend) · Diana (QA)",
    version="0.3.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict:
    deps = await _check_dependencies()
    healthy = all(v == "ok" for v in deps.values())
    return {
        "status": "healthy" if healthy else "degraded",
        "service": "ai-dev-team",
        "version": "0.3.0",
        "dependencies": deps,
    }


@app.post("/tasks/invoke", response_model=TeamTaskResponse)
async def invoke_team(
    request: TeamTaskRequest,
    background_tasks: BackgroundTasks,
) -> TeamTaskResponse:
    task_id = str(uuid.uuid4())[:8]
    started_at = time.time()

    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.description)],
        "active_agent": "alice",
        "task_id": task_id,
        "task_description": request.description,
        "project_board": None,
        "slack_thread_ts": request.thread_ts,
        "slack_channel": request.slack_channel,
        "repo_owner": request.repo_owner,
        "repo_name": request.repo_name,
        "current_branch": None,
        "frontend_branch": None,
        "backend_branch": None,
        "pr_urls": [],
        "issue_urls": [],
        "tool_calls": None,
        "tool_results": None,
        "iteration_count": 0,
        "max_iterations": request.max_iterations,
        "human_input": None,
        "pending_human": False,
    }

    store = get_state_store()
    await store.set(task_id, {
        "status": "running",
        "state": _serialize_state(initial_state),
        "started_at": started_at,
        "completed_at": None,
        "error": None,
    })

    background_tasks.add_task(_run_workflow, task_id, initial_state)

    return TeamTaskResponse(
        task_id=task_id,
        status="running",
        active_agent="alice",
        iteration_count=0,
        messages=[{"type": "human", "content": request.description}],
        pr_urls=[],
        issue_urls=[],
        summary="Task started, running in background...",
    )


async def _run_workflow(task_id: str, initial_state: AgentState) -> None:
    logger.info("[task:%s] Starting workflow", task_id)
    store = get_state_store()
    try:
        final_state = await team_graph.ainvoke(initial_state)
        status = "completed" if final_state.get("active_agent") == "end" else "awaiting_human"
        await store.set(task_id, {
            "status": status,
            "state": _serialize_state(final_state),
            "started_at": None,
            "completed_at": time.time(),
            "error": None,
        })
        logger.info("[task:%s] Workflow completed with status=%s", task_id, status)
    except Exception as exc:
        logger.error("[task:%s] Workflow failed: %s", task_id, exc, exc_info=True)
        await store.set(task_id, {
            "status": "failed",
            "state": _serialize_state(initial_state),
            "started_at": None,
            "completed_at": time.time(),
            "error": str(exc),
        })


@app.post("/tasks/human-input")
async def provide_human_input(request: HumanInputRequest) -> dict:
    store = get_state_store()
    task = await store.get(request.task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {request.task_id} not found")
    if task["status"] != "awaiting_human":
        raise HTTPException(status_code=400, detail=f"Task status is {task['status']}, not awaiting_human")

    state = task["state"]
    state["human_input"] = request.answer
    state["pending_human"] = False
    task["status"] = "running"
    await store.set(request.task_id, task)
    logger.info("[task:%s] Resuming with human input", request.task_id)

    return {
        "task_id": request.task_id,
        "status": "resumed",
        "human_input": request.answer,
        "message": f"Check /tasks/{request.task_id}/status for progress.",
    }


@app.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str) -> TaskStatusResponse:
    store = get_state_store()
    task = await store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    state = task.get("state", {})
    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        active_agent=state.get("active_agent", "unknown"),
        iteration_count=state.get("iteration_count", 0),
        max_iterations=state.get("max_iterations", 20),
        started_at=task.get("started_at"),
        completed_at=task.get("completed_at"),
        error=task.get("error"),
    )


def _serialize_state(state: AgentState) -> dict:
    serialized = dict(state)
    if "messages" in serialized:
        serialized["messages"] = [
            m.model_dump() if hasattr(m, "model_dump") else {"type": "unknown", "content": str(m)}
            for m in serialized["messages"]
        ]
    return serialized


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
