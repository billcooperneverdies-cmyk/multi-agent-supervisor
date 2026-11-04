"""Shared test fixtures for the graph-core suite.

Injects a deterministic, offline fake chat model through the
``orchestrator.state_machine.set_llm_factory`` seam. NO real LLM and NO
network access is used anywhere in these tests.
"""

import os
import sys
from typing import Any

import pytest
from pydantic import Field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from langchain_core.language_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402


class ScriptedChatModel(BaseChatModel):
    """Deterministic fake chat model.

    - Serves ``script`` items (str or BaseMessage) one per invocation, in order.
    - Records every prompt it receives in ``received``.
    - Counts sync vs async invocations so tests can assert the async path.
    - ``bind_tools`` records the tools and returns ``self`` (the base class
      raises NotImplementedError, so we override).
    """

    script: list = Field(default_factory=list)
    received: list = Field(default_factory=list)
    sync_calls: int = 0
    async_calls: int = 0
    bound_tools: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-fake"

    def _next_message(self) -> BaseMessage:
        if not self.script:
            # Safety net: never loop forever if a test under-scripts.
            return AIMessage(content="[[END]]")
        item = self.script.pop(0)
        if isinstance(item, BaseMessage):
            return item
        return AIMessage(content=str(item))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.sync_calls += 1
        self.received.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.async_calls += 1
        self.received.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = list(tools)
        return self


@pytest.fixture
def install_model():
    """Install a ScriptedChatModel into the LLM factory seam.

    Returns an ``install(script)`` callable: pass a list of response strings /
    messages and get the model back for assertions.
    """
    from orchestrator import state_machine as sm

    def install(script: list) -> ScriptedChatModel:
        model = ScriptedChatModel(script=list(script))
        sm.set_llm_factory(lambda: model)
        return model

    yield install
    sm.reset_llm_factory()


@pytest.fixture(autouse=True)
def _no_mcp_no_factory_leak(monkeypatch):
    """Keep tests hermetic: no MCP spawn, no factory leak between tests.

    Also provides a default AGENT_API_KEY (BUG-007): the API refuses to serve
    without one, so tests that exercise auth explicitly override/delete it.
    """
    monkeypatch.delenv("MCP_SERVER_CMD", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("AGENT_API_KEY", "test-api-key")
    yield
    try:
        from orchestrator import state_machine as sm

        sm.reset_llm_factory()
        sm.reset_mcp_tools()
    except Exception:
        pass
    try:
        from orchestrator import api as api_mod

        api_mod.in_flight.draining = False
    except Exception:
        pass


def initial_state(description: str = "Build a demo app", **overrides) -> dict:
    """A complete AgentState-shaped dict for graph invocations in tests."""
    state: dict[str, Any] = {
        "messages": [HumanMessage(content=description)],
        "active_agent": "alice",
        "task_id": "test-task",
        "task_description": description,
        "project_board": None,
        "slack_thread_ts": None,
        "slack_channel": "#general",
        "repo_owner": "octo",
        "repo_name": "demo",
        "current_branch": None,
        "frontend_branch": None,
        "backend_branch": None,
        "pr_urls": [],
        "issue_urls": [],
        "tool_calls": None,
        "tool_results": [],
        "iteration_count": 0,
        "max_iterations": 20,
        "human_input": None,
        "pending_human": False,
    }
    state.update(overrides)
    return state
