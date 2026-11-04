"""LangGraph FastAPI Service — 生产级四代理团队 API。"""

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

from orchestrator.state_machine import (
    make_run_config,
    new_task_state,
    normalize_content,
    team_graph,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理。"""
    print("🚀 启动多代理团队服务...")
    yield
    print("👋 关闭多代理团队服务...")


app = FastAPI(
    title="AI Software Development Team",
    description="Alice (PM) · Bob (Frontend) · Charlie (Backend) · Diana (QA)",
    version="0.2.0",
    lifespan=lifespan,
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


def _thread_config(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}}


def _run_status(snapshot: Any) -> str:
    """Derive the run status from a real checkpoint snapshot."""
    values = snapshot.values
    if any(getattr(task, "interrupts", None) for task in snapshot.tasks):
        return "waiting_human"
    if "__interrupt__" in values:
        return "waiting_human"
    if snapshot.next:
        return "running"
    return "completed" if values.get("active_agent") == "end" else "stopped"


async def _snapshot_or_404(task_id: str):
    """Fetch the checkpoint snapshot for a task; 404 if it never existed."""
    snapshot = await team_graph.aget_state(_thread_config(task_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"unknown task_id: {task_id}")
    return snapshot


@app.get("/health")
async def health_check() -> dict:
    """健康检查。"""
    return {"status": "ok", "service": "ai-dev-team", "version": "0.2.0"}


@app.post("/tasks/invoke", response_model=TeamTaskResponse)
async def invoke_team(request: TeamTaskRequest) -> TeamTaskResponse:
    """启动四代理团队协作任务。"""
    task_id = str(uuid.uuid4())[:8]

    initial_state = new_task_state(
        request.description,
        task_id,
        request.max_iterations,
        slack_thread_ts=request.thread_ts,
        slack_channel=request.slack_channel,
        repo_owner=request.repo_owner,
        repo_name=request.repo_name,
    )

    try:
        final_state: dict[str, Any] = await team_graph.ainvoke(
            initial_state, config=make_run_config(task_id, request.max_iterations)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

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

    if waiting:
        status = "waiting_human"
    else:
        status = "completed" if final_state.get("active_agent") == "end" else "pending"

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


@app.post("/tasks/human-input")
async def provide_human_input(request: HumanInputRequest) -> dict:
    """提供人类输入以恢复中断的工作流（Command(resume=...)）。"""
    await _snapshot_or_404(request.task_id)
    final_state: dict[str, Any] = await team_graph.ainvoke(
        Command(resume=request.answer), config=_thread_config(request.task_id)
    )
    waiting = "__interrupt__" in final_state
    if waiting:
        status = "waiting_human"
    else:
        status = "completed" if final_state.get("active_agent") == "end" else "pending"
    return {
        "task_id": request.task_id,
        "status": status,
        "human_input": request.answer,
        "iteration_count": final_state.get("iteration_count", 0),
    }


@app.get("/tasks/{task_id}/status")
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

    uvicorn.run(app, host="0.0.0.0", port=8000)
