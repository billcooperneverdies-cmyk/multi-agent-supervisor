"""LangGraph State Machine — Agent Orchestration Layer.

Defines AgentState, workflow graph construction, conditional routing,
and tool integration with the MCP server layer.
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict

from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ───────────────────────────────────────────────────────
# AgentState — TypedDict Schema
# ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    """Shared state across the LangGraph workflow.

    Fields:
        messages: Conversation history (managed by add_messages reducer)
        slack_thread_id: Active Slack thread timestamp for continuity
        github_context: Context from GitHub operations (PR numbers, URLs, etc.)
        next_node: Routing target set by conditional edges
        tool_calls: Pending tool call results awaiting LLM processing
        iteration_count: Safety counter to prevent infinite loops
    """

    messages: Annotated[list[AnyMessage], add_messages]
    slack_thread_id: Optional[str]
    github_context: Optional[dict]
    next_node: Optional[str]
    tool_calls: Optional[list[dict]]
    iteration_count: int


# ───────────────────────────────────────────────────────
# Node Functions
# ───────────────────────────────────────────────────────

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a Multi-Agent Supervisor. You coordinate tasks between GitHub and Slack.\n"
        "Available tools:\n"
        "  - github_create_pr: Create a GitHub pull request\n"
        "  - slack_send_message: Send a message to Slack\n"
        "\nRules:\n"
        "  1. If the user asks to create a PR, use github_create_pr.\n"
        "  2. If the user asks to notify a channel, use slack_send_message.\n"
        "  3. If a Slack thread_ts exists, always reply in that thread.\n"
        "  4. After creating a PR, post the PR URL to Slack if a channel is active.\n"
        "  5. Route to 'end' when the task is complete.\n"
    )
)


def supervisor_node(state: AgentState) -> dict:
    """Supervisor node: decides next action or routes to END.

    In a real implementation, this calls an LLM via LiteLLM proxy
    with the available tools defined in the MCP server.
    """
    # Placeholder: real implementation binds LLM + tools here
    # e.g., model.bind_tools(mcp_tools).invoke(state["messages"])
    return {"next_node": "tool_executor"}


def tool_executor_node(state: AgentState) -> dict:
    """Tool Executor: invokes MCP tools and returns ToolMessage results.

    In production, this node uses langchain-mcp-adapters or a custom
    MCP stdio client to call the mcp_server tools.
    """
    # Placeholder: real implementation calls MCP tools
    # e.g., mcp_client.call_tool("github_create_pr", {...})
    tool_results = []
    return {
        "messages": [ToolMessage(content=str(tool_results), tool_call_id="demo")],
        "next_node": "supervisor",
    }


# ───────────────────────────────────────────────────────
# Conditional Routing
# ───────────────────────────────────────────────────────

def route_after_supervisor(state: AgentState) -> str:
    """Conditional edge: decide where to route after supervisor."""
    next_node = state.get("next_node")
    iteration = state.get("iteration_count", 0)

    # Safety: max 10 iterations to prevent infinite loops
    if iteration >= 10:
        return END

    # Route to tool executor if tools are needed
    if next_node == "tool_executor":
        return "tool_executor"

    # Route to end if task is complete
    if next_node == "end":
        return END

    # Default: continue to supervisor
    return "supervisor"


def route_after_tools(state: AgentState) -> str:
    """Conditional edge: always return to supervisor after tool execution."""
    return "supervisor"


# ───────────────────────────────────────────────────────
# Graph Construction
# ───────────────────────────────────────────────────────

def build_workflow() -> StateGraph:
    """Build and return the LangGraph state machine."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tool_executor", tool_executor_node)

    # Set entry point
    workflow.set_entry_point("supervisor")

    # Add conditional edges
    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "tool_executor": "tool_executor",
            "supervisor": "supervisor",
            END: END,
        },
    )
    workflow.add_conditional_edges(
        "tool_executor",
        route_after_tools,
        {
            "supervisor": "supervisor",
        },
    )

    return workflow.compile()


# ───────────────────────────────────────────────────────
# Compiled Graph (Singleton)
# ───────────────────────────────────────────────────────
graph = build_workflow()
