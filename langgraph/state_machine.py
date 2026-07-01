"""Multi-Agent LangGraph State Machine — Alice, Bob, Charlie, Diana Collaboration.

Four-agent collaborative software development workflow:
- Alice (PM): Requirements → Task decomposition → Routing
- Bob (Frontend): Receives tasks → Implements UI → Signals completion
- Charlie (Backend): Receives tasks → Implements API → Signals completion
- Diana (QA): Code review → Approve / Reject with structured routing
"""
from __future__ import annotations

import logging
from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agents import (
    create_alice_prompt,
    create_bob_prompt,
    create_charlie_prompt,
    create_diana_prompt,
)

logger = logging.getLogger("langgraph.state_machine")


# ── Structured Output Schemas ───────────────────────────────────────────────

class RoutingDecision(BaseModel):
    """Alice's structured routing decision."""
    next_agent: Literal["bob", "charlie", "diana", "human", "end"] = Field(
        ..., description="Which agent to route to next"
    )
    reason: str = Field(..., description="Brief reason for routing decision")
    task_for_next: str | None = Field(
        default=None, description="Specific task for the next agent"
    )


class QARoutingDecision(BaseModel):
    """Diana's structured QA decision."""
    action: Literal["approve", "reject_frontend", "reject_backend", "escalate"] = Field(
        ..., description="QA decision"
    )
    reason: str = Field(..., description="Reason for the decision")
    issue_details: str | None = Field(default=None, description="Issues found if rejecting")


class DevCompletion(BaseModel):
    """Bob/Charlie structured completion signal."""
    is_complete: bool = Field(..., description="Task is complete")
    summary: str = Field(..., description="What was accomplished")
    needs_human: bool = Field(default=False, description="Needs human input")
    human_question: str | None = Field(default=None, description="Question for human")


# ── AgentState ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    active_agent: Literal["alice", "bob", "charlie", "diana", "human", "end"]
    task_id: Optional[str]
    task_description: Optional[str]
    project_board: Optional[dict]
    slack_thread_ts: Optional[str]
    slack_channel: Optional[str]
    repo_owner: Optional[str]
    repo_name: Optional[str]
    current_branch: Optional[str]
    frontend_branch: Optional[str]
    backend_branch: Optional[str]
    pr_urls: Annotated[list[str], list]
    issue_urls: Annotated[list[str], list]
    tool_calls: Optional[list[dict]]
    tool_results: Optional[list[dict]]
    iteration_count: int
    max_iterations: int
    human_input: Optional[str]
    pending_human: bool


# ── LLM Singleton ───────────────────────────────────────────────────────────

_LLM_CACHE: dict[str, ChatOpenAI] = {}


def get_llm(model: str = "gpt-4o") -> ChatOpenAI:
    if model in _LLM_CACHE:
        return _LLM_CACHE[model]
    import os
    llm = ChatOpenAI(
        model=model,
        api_key=os.environ.get("LITELLM_API_KEY", "sk-litellm-master-key"),
        base_url=f"{os.environ.get('LITELLM_API_BASE', 'http://localhost:4000')}/v1",
        temperature=0.2,
        max_retries=3,
        timeout=60,
    )
    _LLM_CACHE[model] = llm
    return llm


def clear_llm_cache() -> None:
    _LLM_CACHE.clear()


def _get_structured_llm(model: str, schema: type):
    return get_llm(model).with_structured_output(schema)


# ── Node Functions ──────────────────────────────────────────────────────────

def _fallback_agent(state: AgentState) -> str:
    fb = state.get("frontend_branch")
    bb = state.get("backend_branch")
    if fb and not bb:
        return "charlie"
    if bb and not fb:
        return "bob"
    if fb and bb:
        return "diana"
    return "bob"


def alice_node(state: AgentState) -> dict:
    """Alice (PM): Requirements analysis, task decomposition, routing."""
    messages = state["messages"]
    task_id = state.get("task_id", "unknown")
    iteration = state.get("iteration_count", 0)

    logger.info("[task:%s] Alice executing iteration %d", task_id, iteration)

    router = _get_structured_llm("gpt-4o", RoutingDecision)
    prompt = create_alice_prompt()

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="You are project manager Alice.")] + messages

    try:
        decision: RoutingDecision = router.invoke(messages)
        next_agent = decision.next_agent
        content = decision.reason
        if decision.task_for_next:
            content += f"\n\nTask: {decision.task_for_next}"
        logger.info(
            "[task:%s] Alice routing to %s: %s",
            task_id, next_agent, decision.reason,
        )
    except Exception as exc:
        next_agent = _fallback_agent(state)
        content = f"Routing decision failed ({exc}), falling back to {next_agent}."
        logger.warning("[task:%s] Alice routing failed: %s", task_id, exc)

    ai_msg = AIMessage(content=content)

    if iteration >= state.get("max_iterations", 20) - 3:
        logger.warning(
            "[task:%s] Approaching max iterations (%d/%d)",
            task_id, iteration, state.get("max_iterations", 20),
        )

    return {
        "messages": [ai_msg],
        "active_agent": next_agent,
        "iteration_count": iteration + 1,
    }


def bob_node(state: AgentState) -> dict:
    """Bob (Frontend): Implement frontend code."""
    messages = state["messages"]
    task_id = state.get("task_id", "unknown")
    iteration = state.get("iteration_count", 0)

    logger.info("[task:%s] Bob executing iteration %d", task_id, iteration)

    checker = _get_structured_llm("gpt-4o", DevCompletion)
    prompt = create_bob_prompt()

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="You are frontend developer Bob.")] + messages

    try:
        result: DevCompletion = checker.invoke(messages)
    except Exception as exc:
        logger.error("[task:%s] Bob completion parse failed: %s", task_id, exc)
        ai_msg = AIMessage(content=f"Working on the frontend implementation... (parse error: {exc})")
        return {"messages": [ai_msg], "active_agent": "bob", "iteration_count": iteration + 1}

    ai_msg = AIMessage(content=result.summary)

    if result.is_complete:
        logger.info("[task:%s] Bob completed: %s", task_id, result.summary[:100])
        return {"messages": [ai_msg], "active_agent": "alice", "iteration_count": iteration + 1}

    if result.needs_human:
        logger.info("[task:%s] Bob requesting human input", task_id)
        return {
            "messages": [ai_msg],
            "active_agent": "human",
            "pending_human": True,
            "iteration_count": iteration + 1,
        }

    logger.debug("[task:%s] Bob continuing work", task_id)
    return {"messages": [ai_msg], "active_agent": "bob", "iteration_count": iteration + 1}


def charlie_node(state: AgentState) -> dict:
    """Charlie (Backend): Implement backend code."""
    messages = state["messages"]
    task_id = state.get("task_id", "unknown")
    iteration = state.get("iteration_count", 0)

    logger.info("[task:%s] Charlie executing iteration %d", task_id, iteration)

    checker = _get_structured_llm("gpt-4o", DevCompletion)
    prompt = create_charlie_prompt()

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="You are backend developer Charlie.")] + messages

    try:
        result: DevCompletion = checker.invoke(messages)
    except Exception as exc:
        logger.error("[task:%s] Charlie completion parse failed: %s", task_id, exc)
        ai_msg = AIMessage(content=f"Working on the backend implementation... (parse error: {exc})")
        return {"messages": [ai_msg], "active_agent": "charlie", "iteration_count": iteration + 1}

    ai_msg = AIMessage(content=result.summary)

    if result.is_complete:
        logger.info("[task:%s] Charlie completed: %s", task_id, result.summary[:100])
        return {"messages": [ai_msg], "active_agent": "alice", "iteration_count": iteration + 1}

    if result.needs_human:
        logger.info("[task:%s] Charlie requesting human input", task_id)
        return {
            "messages": [ai_msg],
            "active_agent": "human",
            "pending_human": True,
            "iteration_count": iteration + 1,
        }

    logger.debug("[task:%s] Charlie continuing work", task_id)
    return {"messages": [ai_msg], "active_agent": "charlie", "iteration_count": iteration + 1}


def diana_node(state: AgentState) -> dict:
    """Diana (QA): Code review and acceptance."""
    messages = state["messages"]
    task_id = state.get("task_id", "unknown")
    iteration = state.get("iteration_count", 0)

    logger.info("[task:%s] Diana executing iteration %d", task_id, iteration)

    qa = _get_structured_llm("gpt-4o", QARoutingDecision)
    prompt = create_diana_prompt()

    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content="You are QA engineer Diana.")] + messages

    try:
        decision: QARoutingDecision = qa.invoke(messages)
    except Exception as exc:
        logger.error("[task:%s] Diana routing parse failed, escalating: %s", task_id, exc)
        ai_msg = AIMessage(content=f"QA routing failed ({exc}), escalating to Alice.")
        return {"messages": [ai_msg], "active_agent": "alice", "iteration_count": iteration + 1}

    ai_msg = AIMessage(content=f"{decision.reason}\n\nDetails: {decision.issue_details or 'N/A'}")

    if decision.action == "approve":
        logger.info("[task:%s] Diana approved: %s", task_id, decision.reason[:100])
        return {"messages": [ai_msg], "active_agent": "end", "iteration_count": iteration + 1}

    if decision.action == "reject_frontend":
        logger.info("[task:%s] Diana rejected frontend: %s", task_id, decision.reason[:100])
        return {"messages": [ai_msg], "active_agent": "bob", "iteration_count": iteration + 1}

    if decision.action == "reject_backend":
        logger.info("[task:%s] Diana rejected backend: %s", task_id, decision.reason[:100])
        return {"messages": [ai_msg], "active_agent": "charlie", "iteration_count": iteration + 1}

    logger.info("[task:%s] Diana escalating: %s", task_id, decision.reason[:100])
    return {"messages": [ai_msg], "active_agent": "alice", "iteration_count": iteration + 1}


def human_node(state: AgentState) -> dict:
    """Human-in-the-Loop: Pause workflow for human input."""
    task_id = state.get("task_id", "unknown")
    logger.info("[task:%s] Workflow paused awaiting human input", task_id)

    last_message = state["messages"][-1] if state["messages"] else None
    if last_message and hasattr(last_message, "content"):
        print(f"\n{'='*60}")
        print(f"🛑 Workflow paused — awaiting human input")
        print(f"{'='*60}")
        print(f"\nAgent request: {last_message.content[:200]}")

    try:
        human_input = input("\nYour reply (or 'continue'): ")
    except (EOFError, KeyboardInterrupt):
        human_input = "continue"

    human_msg = HumanMessage(content=f"[Human] {human_input}")
    logger.info("[task:%s] Human input received: %s...", task_id, human_input[:50])

    return {
        "messages": [human_msg],
        "active_agent": "alice",
        "pending_human": False,
        "human_input": human_input,
    }


# ── Conditional Routing ─────────────────────────────────────────────────────

def route_agent(state: AgentState) -> Literal["alice", "bob", "charlie", "diana", "human", "end"]:
    active = state.get("active_agent", "alice")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 20)

    if iteration >= max_iter:
        logger.warning("Max iterations (%d) reached, terminating workflow", max_iter)
        return "end"

    if state.get("pending_human"):
        return "human"

    return active


# ── Graph Construction ──────────────────────────────────────────────────────

def build_team_workflow() -> StateGraph:
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

    return workflow.compile()


team_graph = build_team_workflow()
