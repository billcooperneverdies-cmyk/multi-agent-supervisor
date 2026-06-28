"""LangGraph Agent — FastAPI Service Entrypoint.

Production-ready HTTP server exposing the supervisor agent via FastAPI.
Includes health check and MCP server subprocess management.
"""

import asyncio
import os
import subprocess
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langgraph.state_machine import AgentState, graph

# ───────────────────────────────────────────────────────
# MCP Server Subprocess Management
# ───────────────────────────────────────────────────────

mcp_process: subprocess.Popen | None = None


def start_mcp_server() -> subprocess.Popen:
    """Start the MCP server as a subprocess for stdio communication."""
    cmd = os.environ.get("MCP_SERVER_CMD", "python -m mcp_server.main")
    return subprocess.Popen(
        cmd.split(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


# ───────────────────────────────────────────────────────
# FastAPI Lifespan
# ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage MCP server lifecycle alongside FastAPI."""
    global mcp_process
    mcp_process = start_mcp_server()
    yield
    if mcp_process:
        mcp_process.terminate()
        mcp_process.wait()


# ───────────────────────────────────────────────────────
# FastAPI Application
# ───────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Agent Supervisor",
    description="LangGraph + MCP + LiteLLM orchestration layer",
    version="0.1.0",
    lifespan=lifespan,
)


class AgentRequest(BaseModel):
    """Request body for agent invocation."""

    message: str
    thread_ts: str | None = None
    github_context: dict | None = None


class AgentResponse(BaseModel):
    """Response from agent invocation."""

    messages: list[dict]
    slack_thread_id: str | None
    github_context: dict | None


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "mcp_running": mcp_process is not None and mcp_process.poll() is None}


@app.post("/invoke", response_model=AgentResponse)
async def invoke_agent(request: AgentRequest) -> AgentResponse:
    """Invoke the supervisor agent with a user message."""
    from langchain_core.messages import HumanMessage

    initial_state: AgentState = {
        "messages": [HumanMessage(content=request.message)],
        "slack_thread_id": request.thread_ts,
        "github_context": request.github_context,
        "next_node": None,
        "tool_calls": None,
        "iteration_count": 0,
    }

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AgentResponse(
        messages=[msg.model_dump() for msg in final_state.get("messages", [])],
        slack_thread_id=final_state.get("slack_thread_id"),
        github_context=final_state.get("github_context"),
    )


# ───────────────────────────────────────────────────────
# CLI Runner (for local development)
# ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
