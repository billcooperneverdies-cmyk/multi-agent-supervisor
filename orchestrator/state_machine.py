"""Multi-Agent LangGraph State Machine — Alice, Bob, Charlie, Diana Collaboration.

四代理协作软件开发团队工作流：
- Alice (PM): 需求分析 → 任务分配 → 进度跟踪 → 验收
- Bob (Frontend): 接收任务 → 实现 UI → 提交代码 → 通知 QA
- Charlie (Backend): 接收任务 → 实现 API → 提交代码 → 通知 QA
- Diana (QA): 接收审查任务 → 代码审查 → Bug 报告 → 验收批准

Routing uses an explicit whole-token sentinel protocol (BUG-011/012). Nodes are
async and call ``llm.ainvoke`` (BUG-015); LLM clients are cached per config
(BUG-034); history is windowed per call (BUG-038).
"""

import logging
import operator
import os
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Annotated, Any, Literal

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AnyMessage,
    BaseMessage,
    HumanMessage,
)
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import interrupt
from pydantic import SecretStr
from typing_extensions import TypedDict

from agents import (
    create_alice_prompt,
    create_bob_prompt,
    create_charlie_prompt,
    create_diana_prompt,
)

logger = logging.getLogger(__name__)

HISTORY_WINDOW_ENV = "HISTORY_WINDOW"
DEFAULT_HISTORY_WINDOW = 20

# BUG-023: bounded resilience at the LiteLLM boundary (factory parameters).
LLM_REQUEST_TIMEOUT_ENV = "LLM_REQUEST_TIMEOUT"
DEFAULT_LLM_REQUEST_TIMEOUT = 60  # seconds per LLM HTTP request
LLM_MAX_RETRIES_ENV = "LLM_MAX_RETRIES"
DEFAULT_LLM_MAX_RETRIES = 2

VALID_AGENTS = ("alice", "bob", "charlie", "diana", "human", "end")


# ───────────────────────────────────────────────────────
# AgentState — 多代理共享状态
# ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    """共享状态，由所有代理节点和工具节点读写。"""

    messages: Annotated[list[AnyMessage], add_messages]
    active_agent: Literal["alice", "bob", "charlie", "diana", "human", "end"]
    task_id: str | None
    task_description: str | None
    project_board: dict | None
    slack_thread_ts: str | None
    slack_channel: str | None
    repo_owner: str | None
    repo_name: str | None
    current_branch: str | None
    frontend_branch: str | None
    backend_branch: str | None
    pr_urls: Annotated[list[str], operator.add]
    issue_urls: Annotated[list[str], operator.add]
    tool_calls: list[dict] | None
    tool_results: Annotated[list[dict], operator.add]
    iteration_count: int
    max_iterations: int
    human_input: str | None
    pending_human: bool


# ───────────────────────────────────────────────────────
# LLM 初始化 — cached factory + test seam (BUG-033/034)
# ───────────────────────────────────────────────────────
_LLM_FACTORY: Callable[[], BaseChatModel] | None = None


def set_llm_factory(factory: Callable[[], BaseChatModel] | None) -> None:
    """Install a process-wide LLM factory (test seam; no network in tests)."""
    global _LLM_FACTORY
    _LLM_FACTORY = factory


def reset_llm_factory() -> None:
    global _LLM_FACTORY
    _LLM_FACTORY = None


@lru_cache(maxsize=8)
def _cached_llm(
    model: str, base_url: str, api_key: str, timeout: float, max_retries: int
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=0.2,
        # BUG-023: explicit request timeout + bounded retries instead of the
        # open-ended client defaults that hung/never-retried node calls.
        timeout=timeout,
        max_retries=max_retries,
    )


def _llm_request_timeout() -> float:
    try:
        return float(os.environ.get(LLM_REQUEST_TIMEOUT_ENV, DEFAULT_LLM_REQUEST_TIMEOUT))
    except ValueError:
        return float(DEFAULT_LLM_REQUEST_TIMEOUT)


def _llm_max_retries() -> int:
    try:
        return max(0, int(os.environ.get(LLM_MAX_RETRIES_ENV, DEFAULT_LLM_MAX_RETRIES)))
    except ValueError:
        return DEFAULT_LLM_MAX_RETRIES


def reset_llm_caches() -> None:
    """Drop cached clients (used by tests)."""
    _cached_llm.cache_clear()


def get_llm(model: str = "gpt-4o") -> BaseChatModel:
    """Return the shared chat-model client (one per config, BUG-034).

    BUG-005: LITELLM_API_KEY has NO default. A missing key fails fast with an
    actionable error instead of silently falling back to a well-known
    credential that would also grant proxy-admin access.
    """
    if _LLM_FACTORY is not None:
        return _LLM_FACTORY()
    base_url = f"{os.environ.get('LITELLM_API_BASE', 'http://localhost:4000')}/v1"
    api_key = os.environ.get("LITELLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "LITELLM_API_KEY is not set. The agent authenticates to the LiteLLM "
            "proxy with this key — set it to the proxy master key "
            "(LITELLM_MASTER_KEY) or a proxy virtual key; see .env.example. "
            "No default credential is provided."
        )
    return _cached_llm(model, base_url, api_key, _llm_request_timeout(), _llm_max_retries())


# ───────────────────────────────────────────────────────
# 内容归一化 (BUG-018) 与哨兵路由协议 (BUG-011/012)
# ───────────────────────────────────────────────────────

def normalize_content(content: Any) -> str:
    """Flatten AIMessage.content (str | list of blocks) into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return " ".join(parts)
    return str(content)


_SENTINEL_RE = re.compile(
    r"\[\[\s*(NEXT\s*:\s*[A-Za-z]+|QA\s*:\s*(?:pass|fail)|END)\s*\]\]",
    re.IGNORECASE,
)


def parse_sentinels(text: str) -> list[tuple[str, str]]:
    """Extract sentinel tokens as (kind, value) tuples, in order of appearance.

    kinds: "next" (value=agent name), "qa" (value=pass|fail), "end" (value="").
    """
    tokens: list[tuple[str, str]] = []
    for match in _SENTINEL_RE.finditer(text):
        token = re.sub(r"\s+", "", match.group(1)).lower()
        if token == "end":
            tokens.append(("end", ""))
        elif token.startswith("qa:"):
            tokens.append(("qa", token.split(":", 1)[1]))
        elif token.startswith("next:"):
            tokens.append(("next", token.split(":", 1)[1]))
    return tokens


def alice_route(content: str, state: Mapping[str, Any]) -> str:
    """Alice routing decision from normalized content.

    Termination tokens are evaluated BEFORE agent-name tokens (BUG-012);
    free text never triggers routing (BUG-011).
    """
    tokens = parse_sentinels(content)
    for kind, _ in tokens:
        if kind == "end":
            return "end"
    for kind, value in tokens:
        if kind == "next" and value in ("bob", "charlie", "diana", "human"):
            return value
    # Fallback from branch state (BUG-032): pick the side not yet delivered.
    frontend_done = bool(state.get("frontend_branch"))
    backend_done = bool(state.get("backend_branch"))
    if frontend_done and not backend_done:
        return "charlie"
    if backend_done and not frontend_done:
        return "bob"
    return "bob"


def worker_route(content: str) -> str:
    """Bob/Charlie routing: explicit [[NEXT:...]] only; default reports to Alice."""
    tokens = parse_sentinels(content)
    for kind, _ in tokens:
        if kind == "end":
            return "end"
    for kind, value in tokens:
        if kind == "next" and value in ("alice", "bob", "charlie", "diana", "human"):
            return value
    return "alice"


def diana_route(content: str) -> str:
    """Diana (QA) routing: QA verdict first, agent names only pick the fixer."""
    tokens = parse_sentinels(content)
    for kind, _ in tokens:
        if kind == "end":
            return "end"
    qa_verdict = next((value for kind, value in tokens if kind == "qa"), None)
    if qa_verdict == "pass":
        return "qa_pass"
    if qa_verdict == "fail":
        for kind, value in tokens:
            if kind == "next" and value in ("bob", "charlie"):
                return value
        return "alice"
    for kind, value in tokens:
        if kind == "next" and value in ("alice", "bob", "charlie", "human"):
            return value
    return "alice"


# ───────────────────────────────────────────────────────
# MCP 工具加载 (BUG-014) — lazy, graceful degradation
# ───────────────────────────────────────────────────────
_MCP_TOOLS: "list[Any] | None" = None


async def get_mcp_tools() -> list[Any]:
    """Load MCP tools over a spawned stdio server (MCP_SERVER_CMD).

    Graceful degradation: MCP_SERVER_CMD unset, the adapters package missing,
    or the server failing to spawn -> bind nothing, log a warning, graph still
    runs.
    """
    global _MCP_TOOLS
    if _MCP_TOOLS is not None:
        return _MCP_TOOLS
    cmd = os.environ.get("MCP_SERVER_CMD", "").strip()
    if not cmd:
        logger.info("MCP_SERVER_CMD not set; agent nodes run without MCP tools.")
        _MCP_TOOLS = []
        return _MCP_TOOLS
    try:
        import shlex

        from langchain_mcp_adapters.client import MultiServerMCPClient

        parts = shlex.split(cmd)
        # mcp's stdio transport does NOT inherit arbitrary parent env vars;
        # pass the environment explicitly so server config (tokens etc.) flows.
        client = MultiServerMCPClient(
            {
                "team-tools": {
                    "command": parts[0],
                    "args": parts[1:],
                    "transport": "stdio",
                    "env": dict(os.environ),
                }
            }
        )
        _MCP_TOOLS = list(await client.get_tools())
        logger.info("Loaded %d MCP tools via MCP_SERVER_CMD.", len(_MCP_TOOLS))
    except Exception as exc:
        logger.warning("MCP tool load failed (%s); continuing without tools.", exc)
        _MCP_TOOLS = []
    return _MCP_TOOLS


def reset_mcp_tools() -> None:
    """Drop the cached MCP tool list (used by tests and on config change)."""
    global _MCP_TOOLS
    _MCP_TOOLS = None


_PR_URL_RE = re.compile(r"https?://[^\s\"'<>]+/pull/\d+")
_ISSUE_URL_RE = re.compile(r"https?://[^\s\"'<>]+/issues/\d+")


def extract_result_urls(text: str) -> tuple[list[str], list[str]]:
    """Pull PR / issue URLs out of a tool result string."""
    return _PR_URL_RE.findall(text), _ISSUE_URL_RE.findall(text)


async def _execute_tool_calls(
    response: BaseMessage, tools: list[Any]
) -> tuple[list[Any], list[dict[str, Any]], list[str], list[str]]:
    """Execute the LLM's tool calls; collect ToolMessages, results and URLs."""
    from langchain_core.messages import ToolMessage

    tools_by_name = {getattr(t, "name", ""): t for t in tools}
    tool_messages: list[Any] = []
    results: list[dict[str, Any]] = []
    pr_urls: list[str] = []
    issue_urls: list[str] = []
    for call in getattr(response, "tool_calls", None) or []:
        name = call.get("name")
        tool = tools_by_name.get(name)
        if tool is None:
            output_text = f"unknown tool: {name}"
        else:
            try:
                output = await tool.ainvoke(call.get("args") or {})
                output_text = output if isinstance(output, str) else normalize_content(output)
            except Exception as exc:  # tool failures must not kill the run
                output_text = f"tool error: {exc}"
                logger.warning("MCP tool %s failed: %s", name, exc)
        tool_messages.append(
            ToolMessage(content=output_text, tool_call_id=call.get("id"), name=name)
        )
        results.append({"tool": name, "args": call.get("args") or {}, "result": output_text})
        prs, issues = extract_result_urls(output_text)
        pr_urls.extend(prs)
        issue_urls.extend(issues)
    return tool_messages, results, pr_urls, issue_urls


# ───────────────────────────────────────────────────────
# 历史窗口 (BUG-038)
# ───────────────────────────────────────────────────────

def history_window_size() -> int:
    try:
        return max(1, int(os.environ.get(HISTORY_WINDOW_ENV, DEFAULT_HISTORY_WINDOW)))
    except ValueError:
        return DEFAULT_HISTORY_WINDOW


def window_history(messages: list[BaseMessage], window: int | None = None) -> list[BaseMessage]:
    """Bound the history sent to the LLM: first task message + last N (BUG-038)."""
    if window is None:
        window = history_window_size()
    if len(messages) <= window + 1:
        return list(messages)
    return [messages[0], *messages[-window:]]


# ───────────────────────────────────────────────────────
# 运行配置 (BUG-030)
# ───────────────────────────────────────────────────────

def make_run_config(task_id: str, max_iterations: int) -> RunnableConfig:
    """Explicit recursion_limit coherent with max_iterations (BUG-030) and a
    checkpointer thread_id (BUG-010)."""
    return {
        "recursion_limit": 2 * max_iterations + 10,
        "configurable": {"thread_id": task_id},
    }


def new_task_state(
    description: str,
    task_id: str | None,
    max_iterations: int,
    **overrides: Any,
) -> AgentState:
    """Build a complete initial AgentState for a team run."""
    state: AgentState = {
        "messages": [HumanMessage(content=description)],
        "active_agent": "alice",
        "task_id": task_id,
        "task_description": description,
        "project_board": None,
        "slack_thread_ts": None,
        "slack_channel": "#general",
        "repo_owner": None,
        "repo_name": None,
        "current_branch": None,
        "frontend_branch": None,
        "backend_branch": None,
        "pr_urls": [],
        "issue_urls": [],
        "tool_calls": None,
        "tool_results": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "human_input": None,
        "pending_human": False,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ───────────────────────────────────────────────────────
# 节点函数 (async, BUG-015)
# ───────────────────────────────────────────────────────

async def _invoke_agent(
    state: AgentState, prompt_factory: Callable[[], Any]
) -> tuple[BaseMessage, dict[str, Any]]:
    """Run one agent LLM call: persona prompt + windowed history + ainvoke.

    Binds real MCP tools when available (BUG-014); executes any tool calls for
    one round and returns the extra messages/results so nodes can merge them
    into state.
    """
    llm: BaseChatModel = get_llm()
    tools = await get_mcp_tools()
    model: Runnable[LanguageModelInput, BaseMessage] = (
        llm.bind_tools(tools) if tools else llm
    )
    chain = prompt_factory() | model
    history = window_history(list(state["messages"]))
    response = await chain.ainvoke({"messages": history})

    out_messages: list[BaseMessage] = [response]
    extra: dict[str, Any] = {
        "messages": out_messages,
        "tool_calls": None,
        "tool_results": [],
        "pr_urls": [],
        "issue_urls": [],
    }
    if tools and getattr(response, "tool_calls", None):
        extra["tool_calls"] = [dict(c) for c in response.tool_calls]
        tool_msgs, results, pr_urls, issue_urls = await _execute_tool_calls(response, tools)
        out_messages.extend(tool_msgs)
        extra["tool_results"] = results
        extra["pr_urls"] = pr_urls
        extra["issue_urls"] = issue_urls
        # One follow-up call so the model can react to the tool output.
        response = await chain.ainvoke({"messages": [*history, *out_messages]})
        out_messages.append(response)
    return response, extra


async def alice_node(state: AgentState) -> dict[str, Any]:
    """Alice (PM) 节点：需求分析、任务分解、路由决策。"""
    response, extra = await _invoke_agent(state, create_alice_prompt)
    next_agent = alice_route(normalize_content(response.content), state)
    update: dict[str, Any] = {
        **extra,
        "active_agent": next_agent,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
    if next_agent == "human":
        update["pending_human"] = True
    return update


def _branch_name(kind: str, state: AgentState) -> str:
    return f"{kind}/{state.get('task_id') or 'local'}"


async def bob_node(state: AgentState) -> dict[str, Any]:
    """Bob (Frontend) 节点：实现前端代码并记录 frontend_branch (BUG-032)。"""
    response, extra = await _invoke_agent(state, create_bob_prompt)
    next_agent = worker_route(normalize_content(response.content))
    update: dict[str, Any] = {
        **extra,
        "active_agent": next_agent,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "frontend_branch": state.get("frontend_branch") or _branch_name("frontend", state),
    }
    if next_agent == "human":
        update["pending_human"] = True
    return update


async def charlie_node(state: AgentState) -> dict[str, Any]:
    """Charlie (Backend) 节点：实现后端代码并记录 backend_branch (BUG-032)。"""
    response, extra = await _invoke_agent(state, create_charlie_prompt)
    next_agent = worker_route(normalize_content(response.content))
    update: dict[str, Any] = {
        **extra,
        "active_agent": next_agent,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "backend_branch": state.get("backend_branch") or _branch_name("backend", state),
    }
    if next_agent == "human":
        update["pending_human"] = True
    return update


async def diana_node(state: AgentState) -> dict[str, Any]:
    """Diana (QA) 节点：代码审查和验收。"""
    response, extra = await _invoke_agent(state, create_diana_prompt)
    route = diana_route(normalize_content(response.content))
    next_agent = "alice" if route == "qa_pass" else route
    update: dict[str, Any] = {
        **extra,
        "active_agent": next_agent,
        "iteration_count": state.get("iteration_count", 0) + 1,
    }
    if next_agent == "human":
        update["pending_human"] = True
    return update


async def human_node(state: AgentState) -> dict[str, Any]:
    """Human-in-the-Loop 节点：interrupt() 暂停，Command(resume=...) 恢复 (BUG-008/010)。

    No stdin/stdout blocking here: the pause is persisted by the checkpointer
    and resumed by the API (/tasks/human-input) or the CLI.
    """
    last_message = state["messages"][-1] if state["messages"] else None
    request_text = normalize_content(last_message.content) if last_message else ""
    answer = interrupt({"type": "human_review", "request": request_text})
    text = answer if isinstance(answer, str) else normalize_content(answer)

    return {
        "messages": [HumanMessage(content=f"[Human Input] {text}")],
        "active_agent": "alice",
        "pending_human": False,
        "human_input": text,
    }


# ───────────────────────────────────────────────────────
# 条件路由
# ───────────────────────────────────────────────────────

def route_agent(state: AgentState) -> Literal["alice", "bob", "charlie", "diana", "human", "end"]:
    """条件路由：根据 active_agent 状态字段决定下一个节点。"""
    active = state.get("active_agent", "alice")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 20)

    if iteration >= max_iter:
        logger.warning("达到最大迭代次数 %s，强制终止工作流。", max_iter)
        return "end"

    if state.get("pending_human") and active != "end":
        return "human"

    return active


# ───────────────────────────────────────────────────────
# 检查点 / 持久化 (BUG-008/010)
# ───────────────────────────────────────────────────────

def get_checkpointer() -> Any:
    """Default InMemorySaver; PostgresSaver when DATABASE_URL is set.

    PostgresSaver is a lazy import with graceful fallback + warning, so the
    graph always builds even without the optional postgres package/driver.
    """
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url:
        try:
            from langgraph.checkpoint.base import BaseCheckpointSaver
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

            saver = AsyncPostgresSaver.from_conn_string(db_url)
            if not isinstance(saver, BaseCheckpointSaver):
                raise TypeError(
                    "AsyncPostgresSaver.from_conn_string returned a context manager; "
                    "manage its lifecycle explicitly to use Postgres persistence"
                )
            logger.info("Using PostgresSaver checkpoint store (DATABASE_URL).")
            return saver
        except Exception as exc:
            logger.warning(
                "DATABASE_URL is set but PostgresSaver is unavailable (%s); "
                "falling back to InMemorySaver.",
                exc,
            )
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()


# ───────────────────────────────────────────────────────
# 图构建
# ───────────────────────────────────────────────────────

def build_team_workflow(checkpointer: Any = None) -> CompiledStateGraph[AgentState]:
    """
    构建四代理协作工作流图。

    ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
    │  alice  │────▶│   bob   │     │ charlie │     │  diana  │     │  human  │
    │  (PM)   │◄────│(前端)    │     │(后端)    │     │  (QA)   │     │(人类)   │
    │  调度中心│     └────┬────┘     └────┬────┘     └────┬────┘     └────┬────┘
    │         │◄──────────┘◄──────────────┘◄──────────────┘◄──────────────┘
    └────┬────┘
         │
         ▼
    ┌─────────┐
    │   END   │
    └─────────┘
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("alice", alice_node)
    workflow.add_node("bob", bob_node)
    workflow.add_node("charlie", charlie_node)
    workflow.add_node("diana", diana_node)
    workflow.add_node("human", human_node)

    workflow.set_entry_point("alice")

    for node_name in ["alice", "bob", "charlie", "diana", "human"]:
        workflow.add_conditional_edges(
            node_name,
            route_agent,
            {
                "alice": "alice",
                "bob": "bob",
                "charlie": "charlie",
                "diana": "diana",
                "human": "human",
                "end": END,
            },
        )

    return workflow.compile(checkpointer=checkpointer)


# 全局编译图（带检查点持久化，BUG-010）
team_graph = build_team_workflow(checkpointer=get_checkpointer())
