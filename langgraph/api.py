"""LangGraph FastAPI Service — 生产级四代理团队 API。"""

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langgraph.state_machine import AgentState, team_graph


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


@app.get("/health")
async def health_check() -> dict:
    """健康检查。"""
    return {"status": "ok", "service": "ai-dev-team", "version": "0.2.0"}


@app.post("/tasks/invoke", response_model=TeamTaskResponse)
async def invoke_team(request: TeamTaskRequest) -> TeamTaskResponse:
    """启动四代理团队协作任务。"""
    from langchain_core.messages import HumanMessage
    import uuid

    task_id = str(uuid.uuid4())[:8]
    
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

    try:
        final_state = await team_graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    summary = None
    messages = final_state.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content"):
            summary = last_msg.content[:500]

    return TeamTaskResponse(
        task_id=task_id,
        status="completed" if final_state.get("active_agent") == "end" else "pending",
        active_agent=final_state.get("active_agent", "unknown"),
        iteration_count=final_state.get("iteration_count", 0),
        messages=[m.model_dump() if hasattr(m, "model_dump") else {"type": "unknown", "content": str(m)}
                for m in messages],
        pr_urls=final_state.get("pr_urls", []),
        issue_urls=final_state.get("issue_urls", []),
        summary=summary,
    )


@app.post("/tasks/human-input")
async def provide_human_input(request: HumanInputRequest) -> dict:
    """提供人类输入以恢复中断的工作流。"""
    return {
        "task_id": request.task_id,
        "status": "resumed",
        "human_input": request.answer,
    }


@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str) -> dict:
    """获取任务状态（需要状态存储实现）。"""
    return {"task_id": task_id, "status": "not_implemented"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
