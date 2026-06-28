"""LangGraph Agent — Supervisor Entrypoint.

Demonstrates how to initialize the state machine and run an agent
with MCP tool integration and LiteLLM model routing.
"""

import asyncio
import os

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.state_machine import AgentState, graph

# LiteLLM proxy configuration
LITELLM_API_BASE = os.environ.get("LITELLM_API_BASE", "http://localhost:4000")
LITELLM_API_KEY = os.environ.get("LITELLM_API_KEY", "sk-litellm-master-key")


def create_llm() -> ChatOpenAI:
    """Initialize LLM routed through LiteLLM proxy."""
    return ChatOpenAI(
        model="gpt-4o",  # Routed through LiteLLM proxy
        api_key=LITELLM_API_KEY,
        base_url=f"{LITELLM_API_BASE}/v1",
        temperature=0.2,
    )


async def run_agent(input_message: str, thread_ts: str | None = None) -> None:
    """Run the supervisor agent with user input."""
    initial_state: AgentState = {
        "messages": [HumanMessage(content=input_message)],
        "slack_thread_id": thread_ts,
        "github_context": None,
        "next_node": None,
        "tool_calls": None,
        "iteration_count": 0,
    }

    # Invoke the compiled graph
    final_state = await graph.ainvoke(initial_state)

    # Print final response
    for msg in final_state.get("messages", []):
        print(f"[{msg.type}]: {msg.content}")


async def main() -> None:
    """Interactive agent runner."""
    print("🤖 Multi-Agent Supervisor (LangGraph + MCP + LiteLLM)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() in ("exit", "quit"):
            break
        await run_agent(user_input)
        print()


if __name__ == "__main__":
    asyncio.run(main())
